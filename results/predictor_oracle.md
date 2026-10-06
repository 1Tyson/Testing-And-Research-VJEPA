# Bước 0: predictor còn có thể cải thiện bao nhiêu? (kết quả)

## Kiểm tra bản sao 292p (`notebooks/kaggle_verify_292.ipynb`)
Bản sao: resize giống eval (cạnh ngắn 292, cv2 INTER_LINEAR), x264 yuv444p crf 12, giữ mọi frame. So với video gốc,
cùng code, cùng checkpoint:

| Giao thức | Dữ liệu | official | clean | verb (clean) | noun (clean) |
|---|---|---|---|---|---|
| gốc (`official` anchor) | video gốc | 32.71 | 31.23 | 53.58 | 52.76 |
| gốc (`official` anchor) | bản 292p | 32.20 | 30.86 | 51.66 | 50.57 |
| chuẩn (`action_start`) | video gốc | 13.22 | 14.63 | 28.42 | 36.86 |
| chuẩn (`action_start`) | bản 292p | 12.85 | 14.10 | 27.43 | 36.45 |

Từng clip: top-1 trùng 84–85%, cả tập top-5 trùng y hệt 43%, chênh lệch logit trung bình 0.13.
**Kết luận:** bản 292p thấp hơn khoảng 0.4–0.5 điểm action (xấp xỉ 1 độ lệch chuẩn bootstrap) và khoảng 1–2 điểm
verb/noun. Đủ tốt để train và so sánh tương đối; còn con số cuối cùng để báo cáo thì nên đo trên video gốc.

## Oracle (`notebooks/kaggle_predictor_oracle.ipynb`): toàn bộ val, bản 292p, `--fix_frame_ids`, probe của Meta
Mean-class R@5, mỗi clip đếm 1 lần.

| Giao thức | biến thể | action | verb | noun |
|---|---|---|---|---|
| chuẩn (`action_start`, 9295 clip) | base (predictor V-JEPA 2) | 14.11 | 27.50 | 36.49 |
| | oracle (token tương lai thật) | **15.73** | **31.80** | 37.20 |
| | enconly (không có token tương lai) | 11.83 | 26.96 | 31.66 |
| | copylast (lặp bước cuối) | 11.82 | 27.13 | 31.42 |
| gốc (`official`, 9293 clip) | base | 30.89 | 51.89 | 50.41 |
| | oracle | 23.83 | 46.04 | 43.67 |
| | enconly | 26.00 | 50.73 | 45.24 |
| | copylast | 25.69 | 50.55 | 44.61 |

Cosine giữa token dự đoán và token tương lai thật: predictor 0.46, copylast 0.37. Predictor gần token thật hơn copylast ở
**100%** số clip.

**Đọc kết quả:**
- Token của predictor có ích thật: base cao hơn enconly 2.3 điểm action (giao thức chuẩn) và 4.9 điểm (giao thức gốc).
  Predictor cũng dự báo tốt hơn "không có gì thay đổi" ở mọi clip, nhưng còn xa token thật (cosine 0.46).
- Với probe đóng băng, "predictor hoàn hảo" chỉ thêm **+1.6 action / +4.3 verb** ở giao thức chuẩn, và còn làm **giảm
  7 điểm** ở giao thức gốc. Điều này cho thấy probe đã quen với phân bố đầu ra của predictor, nên không dùng được token
  thật (lệch phân bố), và cũng không dùng được khi bỏ hẳn token tương lai. Vì vậy các con số oracle/enconly với probe
  đóng băng là **cận dưới bị nhiễu**, không phải khoảng cải thiện thật.
- Phép thử công bằng là **train probe mới** cho từng loại đầu vào (enc / enc+pred / enc+token thật).

## Bước 0b: train lại probe, cross-validation trên val (`notebooks/kaggle_cv_probe.ipynb`)
Cách làm:
- `eval_ek100.py --oracle --save_feats_pool 4` lưu 3 loại token, mỗi clip khoảng 0.6 MB, toàn bộ 5.4 GB:
  - token encoder: 16 bước thời gian × lưới 4×4;
  - token do predictor dự đoán cho bước được dự đoán, pool 4×4;
  - token thật của bước đó, pool 4×4.
- `scripts/cv_probe.py` train một attentive probe nhỏ:
  - kiến trúc: embedding vị trí cho từng (bước thời gian, ô), 1 block self-attention, 3 query verb/noun/action;
  - chia fold: người tham gia của val được chia thành 2 fold (4631 / 4632 clip), không ai nằm ở cả hai;
  - chọn epoch: 10% của fold train được giữ lại để chọn epoch;
  - chạy 3 seed; mỗi clip được dự đoán out-of-fold, rồi tính mean-class R@5 trên toàn val.
- Giao thức `action_start`, 9263 clip (bỏ các clip lỗi hoặc vượt cuối video), mean ± std qua 3 seed:

| biến thể | action | verb | noun |
|---|---|---|---|
| probe của Meta (train trên 67k clip, chỉ để tham chiếu) | 14.11 | 27.51 | 36.66 |
| `enc` | 1.89 ± 0.08 | 12.93 ± 0.38 | 6.50 ± 0.11 |
| `enc_pred` (predictor V-JEPA 2) | 1.90 ± 0.08 | 12.42 ± 0.56 | 6.78 ± 0.19 |
| `enc_copylast` (đối chứng) | 1.97 ± 0.09 | 12.31 ± 0.31 | 6.71 ± 0.06 |
| `enc_real` (predictor hoàn hảo) | **3.64 ± 0.15** | **17.30 ± 0.56** | **9.71 ± 0.27** |
| `pred` (chỉ token dự đoán) | 1.56 ± 0.06 | 11.26 ± 0.23 | 5.68 ± 0.12 |
| `real` (chỉ token thật) | 3.34 ± 0.14 | 17.36 ± 0.69 | 9.68 ± 0.33 |

Hiệu số action (trung bình theo seed; trong ngoặc là khoảng tin cậy 95% bootstrap của ensemble 3 seed):
- `enc_real − enc_pred` = **+1.74** [+2.18, +2.84]
- `enc_pred − enc` = +0.01 [−0.22, +0.21]
- `enc_pred − enc_copylast` = −0.07 [−0.23, +0.15]
- `real − pred` = +1.78 [+2.25, +2.89]

(Bảng in ở cuối lần chạy 1 bị lệch cột: cột "action" thực ra là verb, "verb" là noun, "noun" là action. Bảng trên đã sửa. Từ commit
sau, script in đúng thứ tự.)

**Đọc kết quả:**
- **Số tuyệt đối rất thấp** so với probe của Meta (action 1.9 so với 14.1). Lý do: chỉ có khoảng 4.2k clip để train cho 3806 lớp action,
  rất nhiều lớp ở fold test không hề xuất hiện ở fold train, và token đã bị pool. Vì vậy chỉ so sánh các biến thể với nhau.
- **Token tương lai thật chứa nhiều thông tin hơn hẳn**: action tăng khoảng ×1.9, verb +4.4, noun +3.2, và khoảng tin cậy không chứa 0.
  Riêng `real` (chỉ một bước tương lai) đã tốt ngang `enc_real`. Ở giao thức `action_start`, bước được dự đoán chính là lúc action bắt
  đầu, nên con số này là **cận trên** (giống nhận dạng hơn là dự báo). Không predictor nào đạt tới được, nhưng nó cho thấy còn nhiều chỗ
  để cải thiện.
- **Với probe train từ đầu trên ít dữ liệu, token của predictor V-JEPA 2 không thêm gì so với encoder**: `enc_pred` ≈ `enc` ≈ `enc_copylast`.
  Token dự đoán dùng riêng còn kém hơn token encoder. Mức +2.3 điểm của predictor với probe của Meta (bước 0) đến từ việc probe được train
  trên 67k clip cùng với predictor, chứ ở đây không thấy thông tin mới nào.
- Kết luận cho hướng nghiên cứu: khoảng cách giữa token dự đoán và token thật là lớn (cosine 0.46), và token thật có ích rõ ràng, nên
  **cải tiến predictor có cơ sở**. Bước tiếp theo rẻ nhất là thêm biến thể `enc_fc` / `enc_predfc`: một forecaster nhỏ (hoặc phần tinh
  chỉnh predictor V-JEPA 2) được train để tiến gần token thật, chạy lại trên token đã lưu mà không cần trích lại.

## Bước 0c: forecaster học từ token thật (`enc_fc`, `enc_predfc`, cùng giao thức CV)
| biến thể | action | verb | noun | cosine (pool) với token thật |
|---|---|---|---|---|
| `enc` | 1.89 ± 0.08 | 12.90 ± 0.40 | 6.51 ± 0.13 | |
| `enc_pred` | 1.90 ± 0.08 | 12.42 ± 0.56 | 6.78 ± 0.19 | 0.664 (predictor V-JEPA 2) |
| `enc_fc` | 1.98 ± 0.04 | 13.67 ± 0.63 | 6.61 ± 0.49 | 0.789 |
| `enc_predfc` | 1.86 ± 0.14 | 12.81 ± 0.25 | 6.31 ± 0.21 | 0.818 |
| `enc_real` | 3.64 ± 0.15 | 17.32 ± 0.57 | 9.69 ± 0.25 | 1 |

Copy-last (lặp bước cuối) có cosine 0.737 trên token đã pool, tức là cao hơn predictor V-JEPA 2 (0.664). Trên token đầy đủ thì ngược lại:
0.46 so với 0.37.

Hiệu số action, trung bình theo seed (khoảng tin cậy 95% của ensemble):
- `enc_fc − enc` = +0.09 [−0.12, +0.34]
- `enc_predfc − enc_pred` = −0.04 [−0.23, +0.23]
- `enc_real − enc_predfc` = +1.79 [+2.21, +2.84]

**Đọc kết quả:**
- Forecaster tiến gần token thật hơn hẳn (cosine 0.79–0.82, cao hơn cả predictor lẫn copy-last), nhưng **recall không tăng**. Phần lớn
  cosine đến từ thành phần "tĩnh" (cảnh, đồ vật đã thấy). Phần phân biệt được action, tức là điều sắp xảy ra, chỉ là một phần dư nhỏ, và
  forecaster train trên 4.2k clip không học được phần đó. Vì vậy **cosine/L2 với token thật không phải thước đo tốt** cho chất lượng dự báo.
- Về nguyên tắc: forecaster chỉ nhìn token encoder và được train trên **cùng** các clip với probe, nên không thể mang thêm thông tin mới.
  Cái lợi chỉ có thể đến từ tri thức học được trên **nhiều dữ liệu hơn**, ví dụ video không nhãn của tập train EK100 (vài trăm giờ),
  và từ một mục tiêu nhắm vào phần thay đổi (ví dụ dự đoán `real − last` hoặc dùng loss contrastive) thay vì cosine thuần.
- Hệ quả: hướng predictor cần làm ở **quy mô tập train**. Việc đó tốn kém: phải encode lại train (10 shard CPU) và trích token khoảng
  15 giờ GPU, đồng thời có rủi ro ra kết quả âm tính.

## Bước 0d: chấm lại logits của probe Meta, không train (`notebooks/kaggle_posthoc_ek100.ipynb`)
`scripts/posthoc_ek100.py` thử hai cách chỉnh:
- logit adjustment: `z − τ·log prior`, với prior là tần suất của lớp trong `EPIC_100_train.csv`;
- với action, ghép thêm verb–noun: `log_softmax(z_a) + β·(log_softmax(z_v)[v] + log_softmax(z_n)[n])`.

τ và β được chọn bằng cross-fitting theo người tham gia (2 fold): chọn trên một nửa số người, đo trên nửa còn lại. Logits là của **video
gốc** (lần ra 32.71). Mỗi ô ghi clean / official; khoảng tin cậy 95% là bootstrap của mức tăng clean.

| giao thức | task | probe Meta | sau khi chỉnh (cross-fitted) | tăng (clean) [95% CI] | τ / β đã chọn theo fold |
|---|---|---|---|---|---|
| gốc (`official`, 9248 clip) | **action** | 31.23 / **32.69** | 37.53 / **39.27** | **+6.30** [+5.01, +7.00] | 0.4/1.0, 0.4/0.5 |
| | verb | 53.58 / 58.99 | 67.55 / 72.39 | +13.97 [+10.21, +16.67] | 0.3, 0.4 |
| | noun | 52.76 / 53.82 | 56.77 / 57.58 | +4.01 [+2.30, +5.09] | 0.3, 0.2 |
| chuẩn (`action_start`, 9249 clip) | **action** | 14.60 / 13.22 | 16.82 / 14.80 | **+2.22** [+1.34, +2.70] | 0.3/1.0, 0.3/0.25 |
| | verb | 28.42 / 29.01 | 42.19 / 39.85 | +13.77 [+8.18, +16.02] | 0.4, 0.4 |
| | noun | 36.86 / 34.53 | 40.57 / 38.23 | +3.71 [+2.43, +4.92] | 0.3, 0.3 |
| chuẩn, bản 292p (9263 clip) | action | 14.11 / 12.85 | 16.85 / 14.91 | +2.74 [+1.87, +3.21] | 0.3/1.0, 0.3/0.5 |

- Action chỉ dùng logit adjustment (không ghép verb–noun), cross-fitted: giao thức gốc 34.05 / 34.92, giao thức chuẩn 15.45 / 13.80.
  Như vậy khoảng một nửa mức tăng action đến từ prior, nửa còn lại đến từ việc ghép verb–noun.
- Ở giao thức gốc, probe Meta ra 32.69 thay vì 32.71 như `compute_metrics`. Chênh lệch nhỏ này do các logit fp16 bằng nhau (tie) được xếp
  thứ tự khác nhau khi lấy top-5.
- τ và β gần như giống nhau giữa các fold và giữa các giao thức (τ ≈ 0.3–0.4, β ≈ 0.5–1). Nghĩa là cách chỉnh này ổn định, không phải do
  may mắn khi tune.

**Đọc kết quả:**
- Đây là cải thiện lớn và rẻ: **32.7 → 39.3** (official) trên đúng giao thức của bài báo, không cần train, không cần GPU. Nguyên nhân:
  metric mean-class bị chi phối bởi các lớp hiếm, mà probe lại thiên về các lớp phổ biến của tập train.
- Giới hạn khi viết bài:
  1. Logit adjustment (Menon et al., 2021) và việc ghép verb–noun đều là kỹ thuật đã biết, nên điểm mới nằm ở phân tích và ở cách áp
     dụng cho anticipation, không nằm ở bản thân kỹ thuật.
  2. Tham số được chọn trên val (dù có cross-fitting). Bản sạch hơn là chọn trên một phần tách ra từ tập train.
  3. So sánh với các phương pháp khác chỉ công bằng khi áp cùng cách chỉnh, hoặc khi nói rõ là cách chỉnh này áp dụng được cho mọi model.

## Bước 1: độ bền của cách chấm lại (video gốc, action, clean / official)
| cách chọn τ, β | giao thức gốc (`official`) | giao thức chuẩn (`action_start`) |
|---|---|---|
| probe Meta (không chỉnh) | 31.23 / 32.69 | 14.60 / 13.22 |
| cross-fit 2 fold | 37.53 / 39.27 | 16.82 / 14.80 |
| cross-fit 5 fold | 37.24 / 38.56 | 17.09 / 14.93 |
| leave-one-participant-out (32 người) | 37.16 / 38.73 | 16.64 / 14.36 |
| cố định τ=0.5, β=0.5 (không tune) | 37.69 / 38.84 | 17.24 / 14.89 |
| cố định τ=0.3, β=0.5 (không tune) | 36.83 / 38.50 | 16.99 / 14.75 |
| cố định τ=1, β=0 | 16.21 / 16.01 | 6.89 / 5.83 |

- **Kết quả bền:** mọi cách chia fold và cả tham số cố định đều cho action official khoảng 38.5–39.3 (so với 32.7) ở giao thức gốc, và
  khoảng 14.4–15.2 (so với 13.2) ở giao thức chuẩn. Trên bảng lưới, mọi điểm có τ ∈ [0.2, 0.6] và β ∈ [0.25, 2] đều cho khoảng 35–39.4
  (official, giao thức gốc).
- **τ = 1** (giá trị tối ưu theo lý thuyết, nếu xác suất được hiệu chỉnh đúng) lại làm điểm **giảm mạnh**, còn τ tốt nhất khoảng 0.3–0.5.
  Nghĩa là logits của probe không phải log-xác suất được hiệu chỉnh đúng. Đây là một điểm đáng phân tích trong bài báo.
- Verb (τ 0.3–0.4): +13 đến +14 ở mọi cách chia. Noun (τ 0.3): +4.

Lưu ý: không tune trên tập train. Probe của Meta đã được train trên chính tập train, nên logits trên train quá tự tin, và τ chọn trên
đó sẽ bị lệch.

## Bước 2 (đang chạy): tính tổng quát (`kaggle_eval_vitg384`)
- chạy probe ViT-g/384 của Meta trên video gốc (bài báo báo cáo 39.7);
- áp cùng cách chỉnh.

`scripts/slim_checkpoint.py` chỉ giữ target encoder và predictor ở fp16 (khoảng 2 GB), để 2 process eval vừa RAM của Kaggle.
