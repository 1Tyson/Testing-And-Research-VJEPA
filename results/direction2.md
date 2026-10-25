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

## Tách đóng góp (`action_start`, lần chạy 2)

Cùng dữ liệu, cùng giao thức. Action R@5 clean, mean ± std qua 3 seed; cột "holdout" = action R@5 trên 2 người tham gia của **train**
được giữ lại (không bao giờ nhìn val).

| recipe | loss (τ) | ghép verb–noun | khởi tạo | action (val) | + post-hoc | verb | holdout (train) |
|---|---|---|---|---|---|---|---|
| meta | focal (Meta) | | | 14.10 | 16.80 | 27.43 | 34.38* |
| ce+compose | CE | có | Meta | 12.65±0.24 | 17.06 | 20.01 | 15.87 |
| la | LA, τ = 0.5 | | Meta | 13.39±0.64 | 17.14 | 31.77 | 16.67 |
| bsm | BSM, τ = 1 | | Meta | 15.87±0.32 | 17.30 | 44.00 | 20.05 |
| la+compose | LA, τ = 0.5 | có | Meta | 16.53±0.26 | 17.59 | 32.64 | 21.49 |
| **bsm+compose** | BSM, τ = 1 | có | Meta | **17.53±0.50** | **17.73** | 43.39 | **23.17** |
| bsm+random | BSM, τ = 1 | | ngẫu nhiên | 10.90±0.11 | 12.57 | 36.77 | 6.01 |
| bsm+compose+random | BSM, τ = 1 | có | ngẫu nhiên | 10.83±0.37 | 12.49 | 34.80 | 7.18 |

\* Không so được: probe Meta được train trên toàn bộ tập train, gồm cả 2 người được giữ lại. Từ commit sau, `meta` không được chọn trên
holdout.

**Đọc kết quả:**
- **Chọn recipe không nhìn val:** trong các head train lại, holdout của train xếp hạng **đúng y hệt** thứ tự trên val
  (bsm+compose > la+compose > bsm > la > ce+compose > hai bản khởi tạo ngẫu nhiên), và chọn `bsm+compose`. Con số 17.53 vì vậy không
  phải do chọn recipe trên val.
- **Cân bằng lớp lúc train là phần lớn nhất:** cùng có ghép, BSM so với CE là +4.9 (17.53 so với 12.65); không ghép, +5.7 (15.87 so với
  10.13 ở lần chạy 1).
- **Cân bằng đủ (τ = 1) tốt hơn một nửa (τ = 0.5):** +2.5 không ghép, +1.0 có ghép; verb 44.0 so với 31.8. Ngược với post-hoc trên
  probe Meta (τ tốt nhất khoảng 0.3): logits của probe Meta (focal loss) chỉ chứa khoảng 0.2–0.27 lần log prior, còn head train bằng
  softmax CE học đúng 1 lần log prior, nên trừ đi đúng 1 lần (τ = 1, đúng như lý thuyết logit adjustment) là phù hợp.
- **Ghép verb–noun luôn giúp:** +1.7 (với BSM), +3.1 (với LA), +2.5 (với CE, so với `ce` ở lần chạy 1). Lợi ích lớn hơn khi cân bằng
  chưa đủ, vì phần ghép tự mang thông tin cho lớp hiếm.
- **Khởi tạo từ head của Meta là cần thiết:** khởi tạo ngẫu nhiên mất khoảng 6.7 điểm (10.83 so với 17.53). Lưu ý khi diễn giải: head
  của Meta đã được train trên **toàn bộ** tập train (gồm phần EK55 ta chưa có), nên một phần kết quả là tinh chỉnh lại tri thức đó, không
  phải học từ 55.7% dữ liệu đang có.
- Post-hoc sau khi train gần như không thêm gì cho `bsm+compose` (+0.2), nhưng thêm +1.1 cho `la+compose`: phần cân bằng còn thiếu lúc
  train được bù sau.

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

## Lần chạy 3: toàn bộ tập train (67k clip, 32 người, cả 2 anchor)

Train: đủ 495 video (phần mở rộng từ Hugging Face + phần EK55 từ Bristol, mỗi video EK55 đã đối chiếu md5 chính thức), 66,958 clip
`action_start` / 66,951 clip `official` (bỏ các cửa sổ vượt quá cuối video). Epoch chọn trên 9,502 clip của các người tham gia train
được giữ lại. Val như cũ (bản 292p). Action R@5, mean ± std qua 3 seed, clean / official; CI95: bootstrap của (action thô − meta).

**`official` (giao thức của code Meta, để so với 32.7):**

| recipe | action | action + post-hoc | verb | noun | CI95 | holdout (train) |
|---|---|---|---|---|---|---|
| meta | 30.86 / 32.20 | 37.61 / 38.67 | 51.66 / 58.48 | 50.57 / 51.49 | | (57.05*) |
| focal | 31.10±0.23 / 32.97 | 38.02 / 39.47 | 54.50 / 60.18 | 53.18 / 54.34 | [−0.43, +1.00] | 48.07 |
| ce | 27.80±0.00 / 30.35 | 37.19 / 38.98 | 50.54 / 55.64 | 51.20 / 52.18 | [−3.47, −1.81] | 30.29 |
| bsm | 32.41±0.98 / 34.28 | 35.51 / 37.68 | 70.87 / 74.41 | 56.65 / 56.90 | [+1.37, +3.32] | 47.13 |
| **bsm+compose** | **38.11±0.17 / 39.40** | 38.20 / 39.58 | **71.55 / 74.45** | **57.83 / 57.65** | **[+5.08, +7.18]** | **53.73** |
| la+compose | 37.28±0.64 / 39.43 | 38.36 / 40.27 | 65.75 / 70.35 | 56.14 / 57.04 | [+4.66, +6.68] | 51.37 |
| ce+compose | 33.04±0.66 / 35.52 | 38.71 / 40.37 | 49.39 / 53.65 | 49.91 / 50.85 | [+2.24, +3.74] | 42.63 |
| bsm+compose+random | 35.47±0.17 / 37.66 | 36.73 / 39.12 | 69.26 / 72.34 | 57.16 / 58.61 | [+2.76, +5.11] | 33.45 |

**`action_start` (giao thức chuẩn EK100):**

| recipe | action | action + post-hoc | verb | noun | CI95 | holdout (train) |
|---|---|---|---|---|---|---|
| meta | 14.10 / 12.85 | 16.80 / 14.87 | 27.43 / 27.88 | 36.45 / 34.42 | | (31.79*) |
| focal | 15.96±0.38 / 15.95 | 19.81 / 18.94 | 29.17 / 30.14 | 37.02 / 35.63 | [+1.63, +3.08] | 28.35 |
| ce | 14.42±0.20 / 14.44 | 19.65 / 19.14 | 26.22 / 27.01 | 33.46 / 32.66 | [+0.00, +1.36] | 16.43 |
| bsm | 17.92±0.46 / 17.15 | 19.15 / 18.49 | 48.60 / 48.40 | 40.58 / 37.77 | [+2.96, +4.68] | 30.60 |
| **bsm+compose** | **19.88±0.30 / 19.19** | 20.18 / 19.43 | 49.05 / 48.56 | 40.74 / 38.07 | **[+4.19, +6.24]** | **36.02** |
| la+compose | 20.24±0.33 / 19.76 | 20.52 / 19.91 | 39.93 / 41.15 | 39.77 / 37.87 | [+5.46, +7.25] | 33.67 |
| ce+compose | 16.99±0.33 / 16.83 | 20.84 / 20.33 | 24.95 / 26.06 | 32.85 / 32.13 | [+2.67, +4.00] | 23.80 |
| bsm+compose+random | 20.22±0.08 / 20.51 | 21.16 / 21.17 | 48.77 / 51.39 | 38.13 / 36.89 | [+4.10, +6.16] | 18.51 |

\* Không so được (probe Meta đã train trên các clip đó).

**Đọc kết quả:**
- **Recipe được chọn trên holdout của train (không nhìn val) là `bsm+compose` ở cả 2 anchor.** `official`: **39.40** (cách đếm của
  Meta) so với 32.20 của probe Meta trên cùng bản 292p, **+7.2 điểm**, CI [+5.08, +7.18] (clean); cao hơn cả cách chỉnh post-hoc trên
  probe Meta (38.67, có dùng val để cross-fit), mà không cần chỉnh gì trên val. `action_start`: 19.88 so với 14.10, **+5.8**.
- **Verb tăng mạnh nhất:** 58.5 → 74.5 (official), 27.4 → 49.1 (action_start); noun tăng ít hơn (+6, +4). Đúng như phân tích trước: probe
  Meta thiên về các verb phổ biến (take, put, wash...), cân bằng lớp sửa phần lớn điều đó.
- **Đính chính kết luận của lần 1 ở `official`:** lần 1 mọi head train lại đều thua probe Meta, và mình đã giải thích là pooler "khớp" vào
  các clip train. Với toàn bộ tập train, điều đó không còn: `focal` (chỉ train tiếp) ≈ meta, `bsm+compose` +7.2. Nguyên nhân chính của
  lần 1 là **thiếu dữ liệu** (55.7% segment, và chỉ 84% số đó ở `official`): prior và phân phối lớp lệch khỏi toàn bộ train.
- **Khởi tạo:** ở `action_start`, khởi tạo ngẫu nhiên giờ ngang khởi tạo từ Meta (20.22 so với 19.88; lần 2: −6.7). Khoảng cách lần 2
  là do head Meta mang theo tri thức của phần dữ liệu ta chưa có, không phải do khởi tạo tốt hơn. Ở `official` vẫn còn khoảng cách
  (35.47 so với 38.11), và holdout của bản ngẫu nhiên thấp hẳn (33 so với 54).
- **Post-hoc sau khi train gần như không thêm gì** cho `bsm+compose` (+0.1 đến +0.3): cân bằng lúc train đã làm việc của logit adjustment.
  Với CE (`ce+compose`) post-hoc thêm +5: hai cách gần như thay thế nhau.
- `la+compose` (τ = 0.5) ngang `bsm+compose` trên val (trong khoảng nhiễu), nhưng thấp hơn trên holdout nên không được chọn.

## Giới hạn

- ~~Train mới có phần mở rộng (55.7%)~~: đã đủ ở lần chạy 3.
- Val nén crf 12 yuv444p, train nén crf 18 yuv420p. Mọi số là trên bản 292p của val: probe Meta đạt 32.20 ở đây so với 32.7 trên video
  gốc; mức tăng được đo trên cùng một bản.
- Epoch chọn trên 2 người tham gia được giữ lại, nên nhiễu. Nhiều recipe chọn epoch rất sớm (1–3).
- Chưa đo trên video gốc. Số cuối cùng nên đo lại bằng `stream_eval_ek100.py --eval_args "--save_pooled"` khi Bristol hoạt động lại.

## Việc tiếp theo

1. ~~Tách đóng góp~~ (lần 2). ~~Toàn bộ tập train, cả 2 anchor~~ (lần 3).
2. Đo lại head `bsm+compose` trên **video val gốc** (data.bris.ac.uk), để có con số so trực tiếp với 32.7.
3. Thêm view ngẫu nhiên (augmentation theo config train của Meta), xem `bsm+compose` có tăng tiếp không.
