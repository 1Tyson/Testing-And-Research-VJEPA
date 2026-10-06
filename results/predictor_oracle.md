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
