# Kế hoạch: Kiểm chứng V-JEPA 2 ViT-L trên EK100 (32.7 R@5) và nghiên cứu cải tiến

## Bối cảnh
Thầy giao 2 việc:
1. **Kiểm chứng**: tái tạo con số **32.7 mean-class Recall@5 (action)** của V-JEPA 2 ViT-L/16 (256px) trên EPIC-KITCHENS-100 action anticipation (tập validation, anticipation 1s).
2. **Nghiên cứu**: dùng checkpoint pretrained, "mở" kiến trúc, đề xuất cải tiến, tinh chỉnh và ra kết quả cho một bài báo.

Ràng buộc thực tế:
- Repo `Testing-And-Research-VJEPA` hiện **trống**, ta xây từ đầu, nhánh `claude/vjepa2-verify-improve-dsfgtg`.
- Chạy trên **Kaggle**: 2×T4 16GB (hoặc 1×P100), ~30 giờ GPU/tuần, mỗi phiên tối đa 12 giờ, `/kaggle/working` ~20GB. T4 **không hỗ trợ bf16 và FlashAttention-2**, nên phải dùng fp16 + SDPA memory-efficient.
- Dataset: Kaggle dataset mà bạn gửi. Cần xác nhận nó chứa **video gốc hay frames**, đủ **tập val** chưa, và có file annotation `EPIC_100_train.csv` / `EPIC_100_validation.csv` không.
- **Pretrain V-JEPA 2 từ đầu là bất khả thi** trên Kaggle (Meta dùng hàng trăm GPU). Vì vậy "giai đoạn pretrained" được hiểu là: dùng encoder/predictor đã pretrain và cải tiến các phần phía trên nó.

---

## Giai đoạn 0: Chuẩn bị (tuần 1)
1. Clone `facebookresearch/vjepa2` làm tham chiếu (không vendor toàn bộ, chỉ import như package hoặc git submodule).
2. Đọc và ghi lại **đúng giao thức** của paper/repo:
   - `configs/eval/vitl/ek100.yaml` và module `evals/action_anticipation_frozen/` (frames, fps, anticipation time, độ phân giải, số block của attentive probe, lr, số epoch).
   - Kiểm tra probe có nhận **encoder + predictor output** hay chỉ encoder (theo trí nhớ, V-JEPA 2 ghép cả output của predictor; phải xác nhận trong code).
   - Kiểm tra README có phát hành **checkpoint attentive probe EK100 cho ViT-L** không (chắc chắn có cho ViT-g/384; còn ViT-L thì cần kiểm tra). **Đây là điểm rẽ nhánh quan trọng nhất.**
3. Kiểm tra dataset Kaggle: số segment train/val, định dạng, fps, đối chiếu với CSV chính thức.
4. Benchmark tốc độ: forward ViT-L với 1 clip (32 frames, 256px) fp16 trên T4, đo clip/giây và VRAM, để lên ngân sách giờ GPU.

Cấu trúc repo đề xuất:
```
configs/            # yaml cho eval, extract, train head
src/data/ek100.py   # dataset: cắt clip kết thúc trước action_start 1s
src/models/         # wrapper tải encoder/predictor V-JEPA 2 + các head mới
src/metrics.py      # mean-class recall@5 (verb/noun/action), đúng như code gốc
scripts/            # extract_features.py, train_probe.py, eval.py
notebooks/          # notebook Kaggle mỏng, chỉ gọi scripts
results/            # bảng kết quả + log (json)
```

## Giai đoạn 1: Kiểm chứng 32.7 (tuần 1–3)
**Nhánh A: có checkpoint probe ViT-L**. ✅ ĐÃ XÁC NHẬN: `ek100-vitl-256.pt` có trong README vjepa2
- Chỉ cần chạy **eval** trên ~9.7k segment val: encoder (+predictor) + probe đã train, rồi tính mean-class R@5.
- Ước tính ~1–3 giờ trên 2×T4 (xác nhận lại bằng benchmark ở Giai đoạn 0).
- Kỳ vọng ra 32.7 ± 0.3. Nếu lệch, soát lại: sampling frames/fps, anticipation offset, normalize ảnh, cách tính metric (mean over classes có mặt trong val), và fp16 so với bf16.

**Nhánh B: không có checkpoint probe ViT-L**, phải tự train probe
- Train đầy đủ theo paper (~67k clip × nhiều epoch × augmentation) là **không khả thi trên Kaggle**.
- Phương án: **cache feature của encoder đóng băng** một lần (không augmentation), rồi train probe trên feature.
  - Lưu toàn bộ token (4096×1024) thì quá lớn (~500GB), nên cần **giảm token**: pool không gian 2×2 hoặc 4×4 theo từng bước thời gian, lưu fp16, sharded, và đưa lên Kaggle Dataset để dùng lại.
  - Ước tính extract train mất ~10–20 giờ GPU, tức 1–2 tuần quota.
- Kết quả kỳ vọng thấp hơn 32.7 một chút (do không augment, token đã pool). Báo cáo trung thực: "tái tạo trong giao thức tính toán hạn chế", kèm phân tích khoảng lệch.
- **Khuyến nghị**: dù đi nhánh nào, vẫn chạy extract feature, vì Giai đoạn 2 cần dùng nó.

**Deliverable**: `results/verify_ek100_vitl.md` gồm con số, config, log, và so sánh với paper.

## Giai đoạn 2: Mở kiến trúc và phân tích (tuần 3–4)
- Vẽ sơ đồ luồng: video → tubelet 2×16×16 → ViT-L (24 block, 1024-d, RoPE 3D) → (predictor 12 block với mask token cho tương lai) → attentive probe (cross-attention với query verb/noun/action).
- Phân tích lỗi trên val từ baseline:
  - Recall theo nhóm **head / mid / tail** class (EK100 phân bố rất lệch; metric mean-class bị class hiếm kéo xuống).
  - Độ tương quan giữa đúng verb, đúng noun và đúng action.
  - Ảnh hưởng của độ dài context và vị trí temporal token (attention map của probe).
- Kết quả phân tích này là cơ sở để chọn và biện minh cải tiến trong bài báo.

## Giai đoạn 3: Cải tiến (tuần 4–9)
Chọn các hướng **chạy được trên feature đã cache** (rẻ, lặp nhanh trên T4), xếp theo độ ưu tiên:

1. **Head nhận biết long-tail + cấu trúc verb–noun** (rủi ro thấp, khả năng tăng cao nhất)
   - Logit adjustment / Balanced Softmax theo tần suất class, vì mean-class recall rất nhạy với class hiếm.
   - Tính action logit theo cấu trúc: `action = f(verb, noun)` kết hợp prior đồng xuất hiện verb–noun từ tập train, và chỉ xét các cặp (verb, noun) hợp lệ.
2. **"Tưởng tượng tương lai" bằng predictor pretrained** (điểm mới chính của bài)
   - Dùng predictor V-JEPA 2 để rollout latent của các frame trong khoảng 1s tương lai (mask token ở vị trí thời gian tương lai), rồi cho probe cross-attend vào cả latent quá khứ và latent "tưởng tượng".
   - Ablation: không predictor / predictor 1 bước / rollout nhiều bước / horizon khác nhau.
3. **Probe đa thang thời gian** (ablation phụ)
   - Query riêng cho context gần và context xa, hoặc pooling phân cấp theo thời gian.
4. *(Tùy chọn, nếu còn quota)* **LoRA trên 2–4 block cuối của encoder**, train end-to-end với số clip ít. Đây là cách "tinh chỉnh model" đúng nghĩa, nhưng tốn quota nhiều nhất.

Nguyên tắc thí nghiệm:
- Mọi so sánh đều **trong cùng một giao thức** (cùng feature, cùng seed).
- Chạy ≥3 seed, báo cáo mean ± std.
- Hyperparameter chọn trên một **split nhỏ tách từ train**, không chọn trên val.

## Giai đoạn 4: Kết quả cuối và viết bài (tuần 9–12)
- Bảng chính: baseline tái tạo, rồi +long-tail head, +predictor imagination, rồi full, với verb/noun/action R@5.
- Bảng ablation, kèm phân tích head/tail và hình attention.
- Nếu có thể xin thầy vài giờ trên GPU A100 (server trường / Colab Pro): chạy lại cấu hình tốt nhất theo **giao thức đầy đủ của paper** để so sánh trực tiếp với 32.7.
- Viết bài theo khung: Intro, Related (V-JEPA/V-JEPA 2, anticipation, long-tail), Method, Experiments, Analysis, Limitations (ghi rõ ràng buộc tính toán).

---

## Kiểm chứng (cách đánh giá kết quả)
- Unit test `src/metrics.py`: đối chiếu với hàm tính recall trong repo gốc trên dữ liệu giả, kết quả phải khớp tuyệt đối.
- Sanity check dataloader: in ra thời điểm clip kết thúc so với `start_timestamp − 1s` cho vài segment, và xem thử frames.
- Giai đoạn 1: con số action R@5 trên val so với 32.7 (và verb/noun so với số trong paper).
- Mọi run lưu config + seed + commit hash vào `results/*.json` để có thể tái lập.

## Việc cần làm ngay khi được duyệt
1. Khởi tạo cấu trúc repo, `requirements.txt`, README.
2. Đọc code `vjepa2` (config EK100, eval module, danh sách checkpoint) để chốt Nhánh A hay B.
3. Viết `src/data/ek100.py`, `src/metrics.py`, `scripts/eval.py` cùng notebook Kaggle đầu tiên (benchmark tốc độ và kiểm tra dataset).

---
## Cập nhật sau khi đọc code vjepa2 (commit 204698b)
- Có checkpoint probe ViT-L, nên đi **Nhánh A**: chỉ cần eval. Probe nhận **encoder + predictor** (predictor dự đoán 1 bước ở +1s).
- Khi eval, clip quan sát kết thúc tại `stop_frame − 1s` (xem docs/protocol_notes.md, mục 2). Ta sẽ báo cáo:
  (a) con số theo giao thức gốc, dùng để kiểm chứng 32.7, và (b) con số theo giao thức chuẩn EK100 (`--anchor action_start`).
  Mọi cải tiến ở Giai đoạn 3 đều so sánh theo **(b)**, vì đó là cách so sánh công bằng.
- Val gốc chạy trên 64 GPU nên đếm clip không đều. Ta báo cáo cả số "official" (mô phỏng lại cách đếm này) và số "clean".
- Khi chạy eval, bật `--save_feats_pool 4` để có sẵn feature cho Giai đoạn 2–3 mà không phải chạy encoder lại.
- Công trình liên quan cần đọc: JFAA (giải nhất EK100 Action Anticipation Challenge EgoVis 2026, dựa trên V-JEPA 2).
