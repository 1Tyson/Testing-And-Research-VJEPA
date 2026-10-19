# Hướng 2: train lại các lớp phân loại của probe với loss cân bằng lớp và head verb–noun có cấu trúc

## Câu hỏi nghiên cứu

Chỉnh logits sau khi có kết quả (logit adjustment + ghép verb–noun) đã đưa ViT-L từ 32.7 lên 39.3 (official), không cần train
(`results/predictor_oracle.md`, bước 0d–2). Phân tích cho thấy hai điều:

- prior của tập train chỉ nằm trong logits khoảng 0.2–0.27 lần, do probe được train bằng sigmoid focal loss, nên τ = 1 thất bại;
- mức tăng là một sự đánh đổi: lớp phổ biến mất điểm, lớp hiếm được điểm.

Hướng 2 hỏi: **nếu xử lý mất cân bằng lớp ngay lúc train (thay vì sửa sau), thì được thêm bao nhiêu, và vì sao?** Cụ thể:

1. Loss cân bằng lớp (Balanced Softmax, logit-adjusted CE) hay lấy mẫu cân bằng (cRT) có vượt cách chỉnh post-hoc không?
2. Ghép verb–noun **học được** lúc train (head action có cấu trúc) có tốt hơn ghép cố định với β chọn sau không?
3. Hai cách có cộng dồn không (train cân bằng + chỉnh post-hoc)?
4. Mức tăng còn lại bao nhiêu ở giao thức chuẩn `action_start`, nơi clip không nhìn thấy action?

## Thiết kế

Probe của Meta = **attentive pooler** (4 block, 3 query verb/noun/action) → **3 lớp Linear**. Ta giữ nguyên pooler đã train
(decoupling: biểu diễn giữ nguyên, chỉ học lại bộ phân loại; Kang et al. 2020) và chỉ train lại 3 lớp Linear. Như vậy:

- chỉ cần lưu **đầu ra pooler**: 3 × 1024 fp16 mỗi clip, khoảng 6 KB, tức khoảng 0.4 GB cho cả tập train, thay vì lưu token;
- head mới bắt đầu từ chất lượng của probe Meta (khởi tạo từ trọng số của Meta), không phải từ một probe nhỏ train trên ít dữ liệu;
- mỗi lần train chỉ mất vài phút, nên chạy được nhiều recipe × nhiều seed.

Hạn chế: pooler đã được train (với focal loss) trên chính các clip train, nên đặc trưng train "dễ" hơn đặc trưng val. Bước 4
(tuỳ chọn, tốn hơn) là train lại cả pooler trên token.

## Quy trình (3 notebook, theo thứ tự)

| bước | notebook | tài nguyên | ghi chú |
|---|---|---|---|
| 1 | `kaggle_reencode_train_hf` | CPU | bản sao 292p của tập train. Phần mở rộng EK100 (201 video, 55.7% segment) lấy từ Hugging Face, không phụ thuộc data.bris.ac.uk. Phần EK55 (294 video) chờ Bristol. yuv420p crf 18, khoảng 0.8 GB mỗi giờ video |
| 2 | `kaggle_extract_pooled` | 2×T4 | đầu ra pooler + logits của probe Meta, cho val (bản 292p sẵn có) và train, 2 anchor. Khoảng 1.3 giờ (val) + 2.6 giờ mỗi anchor cho phần mở rộng của train |
| 3 | `kaggle_train_head` | 1×T4 | `scripts/train_head.py`: các recipe × 3 seed, thô và sau post-hoc, clean và official, kèm phân tích theo nhóm lớp |

Script mới hoặc được sửa:

- `scripts/prepare_train_clips.py`, `protocol.build_train_clips`: danh sách clip train với thời gian anticipation / điểm neo riêng
  cho từng clip. View 0 giống val; view 1+ ngẫu nhiên theo đúng khoảng trong config train của Meta. Clip id cố định, nên thêm video
  hay thêm view sau này không làm lệch phần đã trích.
- `scripts/eval_ek100.py --clips_csv train_clips.csv --save_pooled`: chạy trên clip train, lưu thêm đầu ra pooler.
- `scripts/reencode_ek100.py --source hf --video_kind extension`, `stream_eval_ek100.py --source`: nguồn tải Hugging Face.
- `scripts/train_head.py`: train và chấm các recipe. Có `--synthetic` để tự kiểm tra trong 1 phút.

## Giao thức đánh giá

- **So sánh chính ở `action_start`** (giao thức chuẩn EK100, `docs/PLAN.md`); `official` báo cáo kèm để so với 32.7.
- Val trên bản 292p, baseline là probe Meta trên chính các clip đó: khoảng 32.2 official / 12.85 official, theo `results/predictor_oracle.md`.
  Con số cuối cùng có thể đo lại trên video gốc bằng `stream_eval_ek100.py --eval_args "--save_pooled"` rồi áp head đã train.
- Epoch chọn trên 10% người tham gia của train được giữ lại. τ, β của bước post-hoc cross-fitted theo người tham gia của val (giống
  `posthoc_ek100.py`). Không chọn gì khác trên val.
- 3 seed, mean ± std. Khoảng tin cậy 95% bootstrap của (recipe − meta) trên val (clean).
- Báo cáo mức tăng theo nhóm lớp phổ biến / trung bình / hiếm (`analyze_posthoc.py`), vì mean-class recall bị chi phối bởi lớp hiếm.

## Rủi ro đã biết

- Mới có 55.7% tập train (phần mở rộng) cho tới khi Bristol hoạt động lại. Phân phối lớp, người tham gia và camera khác với toàn bộ train,
  nên chỉ coi là kết quả sơ bộ. Prior dùng trong loss cân bằng là prior của chính các clip đã dùng để train.
- Bản train nén crf 18 yuv420p, bản val crf 12 yuv444p: cùng độ phân giải, khác mức nén. Mức khớp với video gốc gần như nhau
  (45% so với 43% tập top-5 trùng), nhưng không phải cùng một cách tạo.
- Balanced Softmax, logit adjustment, cRT là kỹ thuật đã biết. Đóng góp nằm ở việc đo cẩn thận cho anticipation (2 giao thức, so với
  post-hoc) và giải thích vì sao (prior trong logits do focal loss, đánh đổi giữa lớp phổ biến và lớp hiếm).
