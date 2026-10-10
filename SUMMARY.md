# Tóm tắt phiên làm việc: kiểm chứng và cải tiến V-JEPA 2 trên EPIC-KITCHENS-100

Repo `1Tyson/Testing-And-Research-VJEPA`, nhánh `claude/vjepa2-verify-improve-dsfgtg`. Mọi thí nghiệm đều chạy trên Kaggle
(2×T4 16 GB, hoặc chỉ CPU; mỗi phiên tối đa 12 giờ; output tối đa 20 GB). Chi tiết số liệu nằm ở `results/verify_ek100_vitl.md` và
`results/predictor_oracle.md`.

## 1. Mục tiêu và giả thuyết

**Mục tiêu** (do thầy giao):
1. Kiểm chứng con số **32.7 mean-class Recall@5 (action)** của V-JEPA 2 ViT-L/16 256px trên EK100 action anticipation (tập val,
   anticipation 1 giây).
2. Từ checkpoint pretrained, tìm một cải tiến đủ để viết bài báo.

**Các giả thuyết đã kiểm tra:**
- H1: điểm thấp hơn của các lần chạy trước (28–31) là do cách chạy (dữ liệu, giao thức, cách đếm), không phải do model.
- H2: một predictor tốt hơn, tức là token tương lai gần token thật hơn, sẽ làm tăng recall. Đây là hướng "pretrain/cải tiến predictor"
  đã thống nhất.
- H3: metric mean-class bị chi phối bởi lớp hiếm, nên chỉnh logits theo prior của lớp và ghép verb–noun sẽ tăng điểm mà không cần train.

## 2. Những gì đã kiểm chứng được

Đơn vị: mean-class R@5, tập val (9296 clip, 138 video). **official** = cách đếm clip của lần chạy 64 GPU của Meta (có clip bị đếm nhiều
lần). **clean** = mỗi clip đếm đúng 1 lần.

**Kiểm chứng ViT-L/256 trên video gốc** (fp16, T4, có PyAV dự phòng khi decode lỗi):

| giao thức | clean | official | verb / noun (clean) |
|---|---|---|---|
| gốc (`official`: clip kết thúc 1 giây trước *stop_frame*) | 31.23 | **32.71** (bài báo 32.7) | 53.58 / 52.76 |
| chuẩn EK100 (`action_start`: kết thúc 1 giây trước *start_frame*) | 14.63 | 13.22 | 28.42 / 36.86 |

- Bỏ qua 48 clip: 36 clip có cửa sổ vượt quá cuối video (code gốc cũng bỏ), 12 clip còn lại decode lỗi ngay ở video gốc.
- Bootstrap: độ lệch chuẩn của clean khoảng 0.66.
- Nguyên nhân của con số 28.6 trước đây: dataset video đã bị encode lại về 256p crf23. Bản 292p yuv444p crf12 cho 32.20 / 30.86.

**Hướng predictor (H2): kết quả âm tính hoặc chưa kết luận được**, giao thức `action_start`, dùng bản 292p:
- Oracle với probe đóng băng của Meta: base 14.11; dùng token tương lai thật 15.73; chỉ encoder 11.83. Cosine giữa token dự đoán và token
  thật là 0.46, so với 0.37 của copy-last.
- Train lại probe, cross-validation 2 fold theo người tham gia, 3 seed, token pool 4×4. Action: `enc` 1.89, `enc_pred` 1.90,
  `enc_real` **3.64**. Token tương lai thật giúp rõ (khoảng tin cậy +2.18…+2.84), còn token của predictor thì không thêm gì
  (+0.01, [−0.22, +0.21]).
- Forecaster học từ token thật: cosine 0.79–0.82, nhưng action vẫn không tăng (`enc_fc − enc` +0.09; `enc_predfc − enc_pred` −0.04).

**Chỉnh logits sau khi có kết quả (H3): dương tính, bền, tổng quát.** Cách chỉnh gồm logit adjustment `z − τ·log prior_train`, và với
action thì ghép thêm `β·(log_softmax verb + log_softmax noun)`. τ và β được chọn bằng cross-fitting theo người tham gia.

| model / giao thức | probe Meta (clean / official) | sau khi chỉnh (cross-fit 2 fold) | mức tăng clean [95% CI] |
|---|---|---|---|
| ViT-L/256, giao thức gốc | 31.23 / 32.69 | 37.53 / **39.27** | +6.30 [+5.01, +7.00] |
| ViT-L/256, `action_start` | 14.60 / 13.22 | 16.82 / 14.80 | +2.22 [+1.34, +2.70] |
| ViT-g/384, giao thức gốc | 38.02 / **39.03** (bài báo 39.7) | 42.78 / **43.94** | +4.76 [+3.79, +5.63] |

- Độ bền (official, giao thức gốc):
  - ViT-L: 5 fold cho 38.56, leave-one-participant-out cho 38.73, tham số cố định τ=0.5, β=0.5 cho 38.84.
  - ViT-g: 5 fold cho 43.76, leave-one-participant-out cho 43.84, tham số cố định τ=0.5, β=1.0 cho 44.21.
  - τ tốt nằm trong khoảng 0.3–0.6, β trong khoảng 0.5–1, ở cả hai model. τ=1 làm điểm giảm mạnh (ViT-L 16.0, ViT-g 20.6).
- Chỉ dùng logit adjustment (không ghép verb–noun), cross-fitted: ViT-L 34.92, ViT-g 40.37. Verb tăng khoảng +14, noun tăng +4 đến +6.

## 3. Các quyết định thiết kế đã chốt

- **Tái tạo đúng code gốc:**
  - chỉ số lớp lấy từ Python set trên toàn bộ `EPIC_100_train.csv`;
  - thứ tự clip giống loader gốc;
  - mô phỏng cách đếm clip của 64 GPU (`official_eval_multiplicity`), và kiểm tra lại với pipeline webdataset thật trong test;
  - báo cáo cả official lẫn clean.
- **Eval trên video gốc theo kiểu stream** (tải, eval, xoá), vì video gốc khoảng 184 GB không vừa một Kaggle dataset, còn bản encode lại
  thì làm tụt điểm.
- **Bản 292p** (resize giống eval, x264 yuv444p crf12, giữ mọi frame) chỉ dùng cho thí nghiệm tương đối. Con số cuối cùng đo trên video gốc.
- **Báo cáo cả 2 giao thức**, vì giao thức gốc của Meta lộ một phần action.
- **Cross-fitting theo người tham gia** để chọn τ và β. **Không tune trên tập train**, vì probe đã được train trên chính tập đó, nên logits
  trên train quá tự tin.
- **Tạm gác hướng predictor:** token thật có ích, nhưng forecaster train trên cùng dữ liệu với probe không mang thêm thông tin mới; muốn có
  lợi thì phải làm ở quy mô tập train, tốn kém và rủi ro.
- **ViT-g:** dùng checkpoint rút gọn (chỉ target encoder và predictor, fp16, 2 GB) để 2 process vừa RAM; chia val thành 2 phần theo người
  tham gia để chạy song song trên 2 tài khoản.

## 4. File, script, notebook

**Thư viện** (`src/vjepa_ek100/`)
- `protocol.py`: ánh xạ lớp, danh sách clip, frame index (anchor `official` / `action_start`), mô phỏng cách đếm 64 GPU.
- `metrics.py`: top-k hits và mean-class recall, khớp `ClassMeanRecall` gốc.
- `data.py`: dataset decode clip bằng decord, có PyAV dự phòng và token tương lai cho oracle.
- `model.py`: dựng backbone (encoder + predictor) và attentive probe từ config và checkpoint.

**Scripts** (`scripts/`)
- `prepare_ek100.py`: tạo `val_clips.csv`, `classes.json`, `dataset_report.json`, kiểm tra video.
- `eval_ek100.py`: chạy model + probe, ghi logits theo chunk, chạy tiếp được; có các tuỳ chọn `--oracle`, `--save_feats_pool`, `--watch_dir`.
- `stream_eval_ek100.py`: tải video gốc song song (aria2c), eval từng video trên mỗi GPU rồi xoá, in tốc độ mỗi 10 phút.
- `compute_metrics.py`: tính điểm official và clean (`--paper_action` để đặt con số so sánh).
- `diagnose_ek100.py`: chẩn đoán cách đếm, bootstrap, so sánh logits giữa 2 lần chạy.
- `check_decoding.py`: so sánh frame decode bằng decord và PyAV.
- `reencode_ek100.py`: encode lại video về 292p trên CPU, chia shard.
- `summarize_oracle.py`: tính điểm các biến thể oracle với probe đóng băng.
- `cv_probe.py`: train lại probe với cross-validation theo người tham gia (enc / pred / real / fc).
- `posthoc_ek100.py`: chỉnh logits bằng logit adjustment và ghép verb–noun, kèm cross-fitting, kiểm tra độ bền và bảng lưới τ×β.
- `slim_checkpoint.py`: rút gọn checkpoint V-JEPA 2 (chỉ target encoder và predictor, fp16).

**Config**: `configs/ek100_vitl_inference.yaml` và `configs/ek100_vitg384_inference.yaml` là bản sao config inference gốc của Meta.

**Notebook** (`notebooks/`)
- `kaggle_eval_ek100_original`: eval ViT-L trên video gốc, kết quả 32.71.
- `kaggle_verify_ek100`, `kaggle_diagnose_ek100`, `kaggle_check_dataset`: bản đầu, chẩn đoán, kiểm tra dataset.
- `kaggle_reencode_ek100`: tạo bản sao 292p (CPU).
- `kaggle_verify_292`: so sánh bản 292p với video gốc.
- `kaggle_predictor_oracle`: oracle với probe đóng băng.
- `kaggle_cv_probe`: trích token pool và chạy cross-validation probe.
- `kaggle_posthoc_ek100`: chỉnh logits (CPU).
- `kaggle_eval_vitg384`: eval ViT-g/384 (2 phần) rồi chỉnh logits.

**Tài liệu và test**: `results/verify_ek100_vitl.md`, `results/predictor_oracle.md`, `docs/protocol_notes.md`, `docs/PLAN.md`, `README.md`.
`tests/` có 14 test, tất cả đều pass.

## 5. Kế hoạch chạy trên Kaggle (đã chạy, theo thứ tự)

| bước | notebook | tài nguyên | thời gian | input cần thêm |
|---|---|---|---|---|
| 1 | `kaggle_eval_ek100_original` | GPU T4×2, Internet | khoảng 2.5 giờ (chủ yếu là tải video) | output lần trước nếu chạy tiếp |
| 2 | `kaggle_reencode_ek100` (val: `SHARD_ID` 0 và 1) | CPU, Internet | mỗi shard khoảng vài giờ, cho ra khoảng 12 GB | (chạy trên 2 tài khoản) |
| 3 | `kaggle_verify_292` | GPU T4×2 | | dataset 292p val-0 và val-1 |
| 4 | `kaggle_predictor_oracle` | GPU T4×2 | khoảng 2.5–3 giờ | dataset 292p |
| 5 | `kaggle_cv_probe` | GPU T4×2 | trích token khoảng 2.3 giờ (cho ra 5.4 GB), CV khoảng 1–1.5 giờ | dataset 292p; output cũ nếu chạy lại |
| 6 | `kaggle_posthoc_ek100` | CPU | khoảng 3 phút cho mỗi thư mục logits | output của bước 1 (và bước 5) |
| 7 | `kaggle_eval_vitg384`, `PART=0` và `PART=1` | GPU T4×2 cho mỗi phần | khoảng 4 giờ mỗi phần (0.15–0.2 clip/s mỗi GPU) | ghép: output phần 0 và phần 1 |

Lưu ý Kaggle:
- Chọn **Accelerator None** cho các job chỉ cần CPU.
- Muốn chạy tiếp thì *Add Input* output của chính notebook đó.
- Gắn output notebook từ tài khoản khác có thể bị lỗi mount. Cách xử lý: để notebook ở chế độ Public, hoặc biến output thành Dataset.

## 6. Vấn đề còn mở, rủi ro, việc tiếp theo

**Rủi ro khi viết bài:**
- Logit adjustment (Menon và cộng sự, 2021) và việc ghép verb–noun là kỹ thuật đã biết. Đóng góp hiện tại là phân tích và đo lường cẩn
  thận, chưa phải phương pháp mới.
- Mức tăng lớn chủ yếu ở giao thức gốc, vốn lộ action. Ở giao thức chuẩn, mức tăng chỉ khoảng +2.2 điểm clean.
- τ và β được chọn trên val (dù có cross-fitting). Chưa đo trên test set chính thức (server challenge EK100).
- So sánh với các phương pháp khác chỉ công bằng khi áp cùng cách chỉnh cho chúng.
- ViT-g kém bài báo 0.7 điểm (39.03 so với 39.7). Nguyên nhân có thể là fp16 so với bf16, chưa kiểm tra.

**Vấn đề mở:**
- Vì sao τ=1 thất bại, tức là logits của probe không được hiệu chỉnh đúng như xác suất. Cần phân tích.
- Hướng predictor chỉ có thể có lợi ở quy mô tập train, với mục tiêu khác cosine. Hiện chưa được kiểm chứng.

**Việc tiếp theo (đang chờ thầy quyết):**
1. Phân tích cho bài báo, chỉ dùng logits đã có: mức tăng theo nhóm lớp phổ biến / trung bình / hiếm, phân tích τ, đóng góp của việc
   ghép verb–noun.
2. Chọn mức bài báo. Nếu nhắm cao hơn workshop: làm bước 3, tức là train probe với loss cân bằng lớp và head verb–noun có cấu trúc. Bước
   này tốn khoảng vài chục giờ GPU và cần token của tập train (phải encode lại tập train thành 10 shard trên CPU).
3. (Tuỳ chọn) Chạy ViT-g với `ANCHORS="official action_start"` để có số ở giao thức chuẩn.
