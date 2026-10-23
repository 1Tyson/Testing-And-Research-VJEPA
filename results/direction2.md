# Hướng 2: train lại các lớp phân loại của probe trên đầu ra pooler (kết quả sơ bộ)

Thiết kế: `docs/DIRECTION2.md`. Notebook: `kaggle_extract_pooled` (bước 2) và `kaggle_train_head` (bước 3), Kaggle 2×T4, fp16.

## Dữ liệu

- **Val:** bản 292p (`thaiminhdat/reencode-0/1`, yuv444p crf 12). 9,264 clip (`action_start`) / 9,260 clip (`official`) sau khi bỏ các clip lỗi giải mã.
- **Train:** chỉ phần mở rộng EK100 (201 video, **55.7% segment**, 18 người tham gia). Bản 292p yuv420p crf 18, video gốc lấy từ
  Hugging Face. Phần EK55 (294 video) chờ data.bris.ac.uk.
  - `action_start`: đủ 37,455 clip.
  - `official`: 31,744 clip (khoảng 84%, 17 người), vì phiên trích bị cắt ở mốc 12 giờ.
- Mỗi segment 1 clip (view 0, anticipation 1 s). Epoch chọn trên 10% số người tham gia của train (5,618 clip) được giữ lại. 3 seed.
- **Kiểm tra pipeline:** lớp Linear của Meta áp lên đầu ra pooler đã lưu cho cùng top-1 với logits gốc ở 99.9–100% số clip. Baseline
  của Meta trùng với số đã đo trước đây trên bản 292p (14.10 / 12.85 và 30.86 / 32.20, so với 14.11 / 12.85 và 30.86 / 32.20 trong
  `predictor_oracle.md`).

## Giao thức chuẩn `action_start` (so sánh chính)

Mean-class R@5, mean ± std qua 3 seed, mỗi ô ghi clean / official. "+ post-hoc": τ, β cross-fitted theo người tham gia của val
(`posthoc_ek100.py`). CI95: bootstrap của (action thô − meta), clean, seed 0.

| recipe | action | action + post-hoc | verb | noun | CI95 |
|---|---|---|---|---|---|
| meta (probe Meta) | 14.10 / 12.85 | 16.80 / 14.87 | 27.43 / 27.88 | 36.45 / 34.42 | |
| focal | 13.49±0.16 / 12.68 | 17.74±0.19 / 16.82 | 27.70 / 28.68 | 32.87 / 31.75 | [−1.43, −0.03] |
| ce | 10.13±0.11 / 10.14 | 15.63±0.09 / 15.09 | 21.97 / 21.08 | 27.24 / 26.23 | [−4.82, −3.32] |
| bsm | 15.87±0.32 / 15.03 | 17.30±0.24 / 16.50 | **44.00** / 41.44 | 36.27 / 33.72 | [+0.25, +2.06] |
| bsm+cb | 10.05±0.11 / 8.92 | 12.03±0.31 / 10.81 | 37.86 / 36.32 | 30.58 / 29.03 | [−6.04, −4.25] |
| **bsm+compose** | **17.53±0.50 / 16.41** | **17.73±0.58 / 16.74** | 43.39 / 40.98 | 36.39 / 34.11 | **[+1.88, +3.53]** |
| focal+compose | 13.71±0.03 / 13.24 | 17.12±0.16 / 16.45 | 26.39 / 26.62 | 30.87 / 29.82 | [−1.02, +0.39] |

(Recipe `la` ở lần chạy này dùng `la_tau = 1`, trùng hoàn toàn với `bsm`; đã đổi mặc định thành 0.5.)

**Đọc kết quả:**
- **Balanced Softmax + head action ghép verb–noun học được (`bsm+compose`): +3.4 điểm action** so với probe Meta (17.53 so với 14.10,
  CI [+1.88, +3.53]). Kết quả này **không tune gì trên val**, nhưng đã cao hơn cách chỉnh post-hoc trên probe Meta (16.80, τ, β
  cross-fitted trên val). Áp thêm post-hoc lên head mới chỉ được +0.2: cân bằng lúc train đã làm gần hết việc của post-hoc.
- Verb tăng mạnh nhất (27.4 → 44.0 với `bsm`); noun gần như không đổi.
- Chỉ train tiếp (`focal`, `ce`) thì không giúp. Lấy mẫu cân bằng (`+cb`) làm giảm mạnh: holdout tốt nhất ở epoch 1 rồi giảm, tức là
  head học thuộc các mẫu hiếm (mỗi lớp hiếm chỉ có vài clip, bị lặp lại nhiều lần).
- `compose` chỉ có ích khi đi cùng loss cân bằng (`focal+compose` ≈ `focal`).

## Giao thức của code Meta `official` (tham khảo; train chỉ có 84% số clip)

| recipe | action | action + post-hoc | verb | noun | CI95 |
|---|---|---|---|---|---|
| meta | 30.86 / 32.20 | 37.61 / 38.67 | 51.66 / 58.48 | 50.57 / 51.49 | |
| focal | 25.89 / 27.79 | 35.45 / 37.18 | 50.82 / 56.38 | 48.26 / 49.17 | [−5.14, −3.49] |
| ce | 20.64 / 22.57 | 30.55 / 32.66 | 40.05 / 43.61 | 40.51 / 42.21 | [−10.95, −8.89] |
| bsm | 28.00 / 29.52 | 31.39 / 33.05 | 64.09 / 65.77 | 50.19 / 51.44 | [−3.73, −1.67] |
| bsm+compose | 29.87 / 32.14 | 32.88 / 35.15 | 58.85 / 61.28 | 49.75 / 51.26 | [−1.68, +0.37] |
| focal+compose | 27.60 / 29.68 | 33.48 / 35.61 | 48.07 / 52.09 | 46.25 / 47.24 | [−3.63, −1.63] |

**Đọc kết quả:** ở giao thức này mọi head train lại đều kém probe Meta (tốt nhất `bsm+compose` 29.87 so với 30.86; sau post-hoc 32.88
so với 37.61). Giải thích có khả năng nhất: pooler của Meta được train (focal loss) trên **chính các clip train ở giao thức `official`**,
nên trên các clip đó đặc trưng "dễ" và tự tin hơn trên val. Head mới học theo đặc trưng đó và không tổng quát được (holdout tốt nhất ở epoch 1
với `focal`, rồi giảm). Ở `action_start`, pooler chưa từng thấy kiểu clip này, đặc trưng train và val có cùng phân bố, nên train lại có ích.
**Hệ quả:** train lại bộ phân loại trên một pooler đóng băng chỉ có ích khi pooler không bị "khớp" vào chính các clip đó; muốn cải thiện ở
giao thức `official` phải train lại cả pooler (bước 4), hoặc tách clip train mà pooler chưa thấy.

## Giới hạn

- Train mới có phần mở rộng (55.7%, 18 người): prior, người tham gia và camera khác toàn bộ train. Cần chạy lại khi có phần EK55.
- Val nén crf 12 yuv444p, train nén crf 18 yuv420p.
- Epoch chọn trên 2 người tham gia được giữ lại, nên nhiễu. Nhiều recipe chọn epoch rất sớm (1–3).
- Chưa đo trên video gốc. Số cuối cùng nên đo lại bằng `stream_eval_ek100.py --eval_args "--save_pooled"` khi Bristol hoạt động lại.

## Việc tiếp theo

1. Tách đóng góp: `ce+compose` (chỉ ghép, không cân bằng), `bsm+compose+random` (không khởi tạo từ Meta), `la` với τ = 0.5.
2. Thêm view ngẫu nhiên (augmentation theo config train của Meta), xem `bsm+compose` có tăng tiếp không.
3. Trích nốt `official` và phần EK55 của train khi có video.
