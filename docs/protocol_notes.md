# Ghi chú giao thức EK100 trong code V-JEPA 2 (commit `204698b`)

Nguồn: `evals/action_anticipation_frozen/` và `configs/inference/vitl/ek100.yaml` trong `facebookresearch/vjepa2`.

## Cấu hình ViT-L (inference)
| Mục | Giá trị |
|---|---|
| Encoder | ViT-L/16, tubelet 2, RoPE, 256px, 32 frame @ 8 fps (≈4s ngữ cảnh) → 16×16×16 = 4096 token |
| Predictor | 12 block, dim 384, dự đoán **1 bước thời gian** (2 frame) ở vị trí +1s trong tương lai |
| Đầu vào probe | **concat(token encoder, token predictor)** = 4096 + 256 token (`vit_encoder_predictor_concat_ar`) |
| Probe | AttentivePooler 4 block, 16 head, 3 query (verb / noun / action) → 3 lớp Linear |
| Loss khi train | sigmoid focal loss, 20 epoch, batch 2/GPU, grid 20 tổ hợp lr × wd |
| Metric | mean-class recall@5, trung bình trên các class **có mặt** trong val |
| Checkpoint | `vitl.pt` (encoder `target_encoder` + `predictor`), probe `evals/ek100-vitl-256.pt` |
| Phần cứng gốc | 8 node × 8 GPU, bf16 |

## Các điểm dễ gây lệch kết quả (đã xử lý trong repo này)
1. **Chỉ số class được sinh từ `set()` của file train.** Probe đã phát hành chỉ dùng được khi có **đúng file
   `EPIC_100_train.csv` đầy đủ**. Nếu dùng bản train bị cắt bớt, thứ tự class bị xáo trộn và kết quả thành vô nghĩa.
   `model.build_probe` báo lỗi nếu kích thước không khớp. `tests/test_protocol.py` kiểm tra mapping giống hệt code gốc.
2. **Điểm neo của clip.** Code gốc tính `af = int(sf*ap + (1-ap)*ef - 1s*fps)` với `ap = 0` khi val, nên clip
   quan sát **kết thúc tại `stop_frame − 1s`**, không phải `start_frame − 1s` như định nghĩa chuẩn của EK100.
   Với action dài hơn 1s, clip có thể chứa một phần của chính action cần dự đoán. Muốn tái tạo 32.7 thì phải giữ
   nguyên cách này (`--anchor official`). Chạy thêm `--anchor action_start` để đo con số theo giao thức chuẩn;
   đây là một phân tích đáng đưa vào bài. Issue [facebookresearch/vjepa2#173](https://github.com/facebookresearch/vjepa2/issues/173)
   (tháng 7/2026, chưa có phản hồi) cũng nêu đúng vấn đề này.
3. **Cách đếm clip trong lúc validate.** Mỗi rank chạy đúng `ipe = N // (64×2)` batch và quay lại đầu khi hết dữ liệu.
   Vì mỗi rank giữ `videos[rank::64]` với số clip rất chênh lệch, có clip bị đếm nhiều lần và có clip không được
   đếm. `protocol.official_eval_multiplicity` mô phỏng lại việc này. Test so sánh với pipeline
   webdataset + DataLoader thật cho kết quả khớp hoàn toàn. `compute_metrics.py` báo cả hai con số **official** và **clean**.
4. **Frame id trong annotation luôn tính theo ~60fps, nhưng code gốc dùng thẳng trên video gốc.**
   Tập val có 133 video 59.94fps, 3 video 29.97fps, 1 video 47.95fps, 1 video 90fps. Dataset Kaggle khớp đúng với
   `EPIC_100_video_info.csv`, nên đây là video gốc, không bị encode lại. Ví dụ `P09_07` (29.97fps, dài 55s) có
   `stop_frame = 3071` = 51.19s × 60. Vì vậy trên 5 video không phải 60fps, clip do code gốc cắt ra rơi **sai thời điểm**
   (ở video 30fps là gấp đôi thời gian thật), hoặc vượt quá cuối video. Khi vượt quá, decord báo lỗi và code gốc
   **bỏ qua clip đó âm thầm**.
   - Để tái tạo 32.7, ta giữ nguyên hành vi này, vì lần chạy của Meta dùng cùng những video gốc đó.
     `prepare_ek100.py --probe_videos --video_info ...` dự đoán clip nào sẽ lỗi (cột `expected_fail_*`,
     mục `out_of_range_by_video`, `videos_where_frame_ids_are_not_native`).
   - Clip lỗi bị loại khỏi cả hai metric. Phần mô phỏng official bỏ chúng khỏi luồng dữ liệu, nhưng vẫn tính vào `ipe`, giống code gốc.
   - Với giao thức chuẩn dùng cho nghiên cứu: phải lấy frame theo `timestamp × fps thật` (việc cần làm tiếp).
5. **Độ chính xác số.** Paper dùng bf16, T4 thì dùng fp16. Nếu gặp tràn số (NaN hoặc inf), batch đó được tính lại bằng fp32 (`fp32_retries` trong log).
6. Clip decode lỗi bị code gốc bỏ qua âm thầm. Ở đây chúng được ghi vào `shard*/failed.json` (gộp qua các lần chạy).
7. Trên Kaggle, DataLoader worker từng bị chết khi thoát (`pure virtual method called`, do decord). Cách xử lý:
   giải phóng VideoReader bằng `atexit` trong mỗi worker, và nếu vẫn crash thì lưu phần đã làm, thoát mã 3 rồi tự chạy lại.

## Kết quả lần chạy đầu và các giả thuyết về khoảng lệch
Official 28.61 và clean 27.09, so với 32.7 của paper (xem `results/verify_ek100_vitl.md`). Các giả thuyết và cách kiểm tra
(chạy bằng `notebooks/kaggle_diagnose_ek100.ipynb`):

| Giả thuyết | Kiểm tra |
|---|---|
| Con số official phụ thuộc vào số GPU và số worker của lần chạy gốc | `diagnose_ek100.py`: quét world size từ 1 đến 128 |
| Nhiễu thống kê của mean-class recall (hàng nghìn class chỉ có 1–2 clip) | bootstrap, khoảng tin cậy 95% |
| Lỗi trong pipeline tính metric | xáo trộn nhãn, kết quả phải về mức ngẫu nhiên |
| Cách chấm điểm của code gốc (sigmoid bf16 + torch.topk) | mô phỏng lại trên logits đã lưu |
| fp16 trên T4 so với bf16 của Meta | chạy fp32 trên 256 clip rồi so logits |
| decord trả sai frame khi seek | `check_decoding.py`, so với PyAV |
| 5 video có frame id không đúng fps gốc | tách riêng trong `diagnose_ek100.py` |
| Probe/encoder đã phát hành không khớp với lần chạy trong paper (file `vitl.pt` có ngày 06/06/2025, probe có ngày 02/06/2025) | đối chứng bằng probe SSv2/Diving48 của cùng `vitl.pt` (cần thêm dataset) |
