# Kiểm chứng V-JEPA 2 ViT-L/16 256px trên EK100 (probe `ek100-vitl-256.pt` của Meta)

> **Cập nhật:** các số bên dưới được đo trên dataset `download-ek100`. Sau đó mới phát hiện đây là bản **đã encode lại**
> (`ffmpeg scale=-2:256, crf 23`), không phải video gốc (1920x1080, khoảng 184 GB). Eval của V-JEPA 2 resize cạnh ngắn về 292
> trước khi crop 256, nên ảnh 256p bị phóng to (mờ) và có nhiễu nén. Đây là nguyên nhân chính của khoảng lệch: một notebook
> cũ chạy code gốc trên video gốc (tải từ Bristol, theo lô 15 video, 1 GPU, batch 4) ra **31.24**.
> Cần chạy lại trên video gốc bằng `notebooks/kaggle_eval_ek100_original.ipynb`.

Chạy trên Kaggle 2×T4, fp16, vjepa2 commit `204698b`, 9296 clip val (138 video gốc, khớp `EPIC_100_video_info.csv`).

## Video gốc: lần chạy 1 (CHƯA ĐỦ, 120/138 video, 7669/9296 clip)
`notebooks/kaggle_eval_ek100_original.ipynb`, 2×T4 fp16, 10.8 giờ (phần lớn thời gian là tải video).

| Giao thức | Cách đếm | Action | Verb | Noun |
|---|---|---|---|---|
| Code gốc (`official` anchor) | official (64 GPU) | **33.56** | 58.91 | 54.46 |
| Code gốc (`official` anchor) | clean | 33.25 | 56.12 | 54.43 |
| Code gốc (`official` anchor) | theo lô 15 video, 1 GPU, batch 4 | 33.24 | | |
| Chuẩn EK100 (`action_start`) | official | 13.49 | 29.27 | 34.97 |
| Chuẩn EK100 (`action_start`) | clean | 15.69 | 30.85 | 38.51 |

Official dao động 33.06–33.86 khi đổi world size; bootstrap std 0.65. **Đây mới là con số tạm thời:** mean-class recall
trên một tập con không so trực tiếp được với con số trên toàn bộ tập val (notebook cũ theo lô: 33.43 sau 8/10 lô, 31.24 khi đủ).
Còn 18 video (P29–P32), cần chạy thêm 1 phiên.

## Dataset `download-ek100` (bản encode lại 256p crf 23): kết quả chính (mean-class Recall@5)
| Giao thức | Cách đếm | Action | Verb | Noun |
|---|---|---|---|---|
| Code gốc (clip kết thúc tại `stop_frame − 1s`) | official (mô phỏng 64 GPU) | **28.61** | 55.23 | 46.79 |
| Code gốc (clip kết thúc tại `stop_frame − 1s`) | clean (mỗi clip 1 lần) | 27.09 | 48.95 | 45.88 |
| Chuẩn EK100 (clip kết thúc tại `start_frame − 1s`) | official | 11.75 | 26.08 | 31.39 |
| Chuẩn EK100 (clip kết thúc tại `start_frame − 1s`) | clean | **12.78** | 25.61 | 32.40 |
| Paper / README | | **32.7** | | |

36 clip (giao thức gốc) và 32 clip (giao thức chuẩn) bị bỏ qua vì cửa sổ frame vượt quá cuối video, giống code gốc.

## Chẩn đoán khoảng lệch 28.6 so với 32.7 (`notebooks/kaggle_diagnose_ek100.ipynb`)
| Giả thuyết | Kết quả | Kết luận |
|---|---|---|
| Số GPU và worker của lần chạy gốc | official dao động 27.1–29.1 khi world size từ 1 đến 128; 28.2–28.6 khi số worker từ 0 đến 8 | Không giải thích được |
| Nhiễu thống kê | Độ lệch chuẩn bootstrap 0.62 (khoảng tin cậy bootstrap bị lệch vì metric trung bình theo class, chỉ nên dùng std) | 32.7 cách xa khoảng 6–9 std, nên không phải do nhiễu |
| Lỗi pipeline metric | Xáo trộn nhãn cho 2.87 (mức ngẫu nhiên, vì model thiên về các class phổ biến) | Metric hoạt động đúng |
| Chấm điểm bf16 (sigmoid + topk) | 14 hit thay đổi; official 28.54 | Không đáng kể |
| fp16 so với fp32 (256 clip) | top-1 trùng 98.8%, tập top-5 trùng 96.5% | Không đáng kể |
| decord trả sai frame | 40/40 clip khớp tuyệt đối với PyAV | decode đúng |
| 5 video có frame id không đúng fps gốc (94 clip) | top-5 acc 11.7% (so với 57.6% ở các video khác); bỏ các video này thì clean chỉ lên 27.44 | Có ảnh hưởng nhưng nhỏ (+0.35) |
| Code vjepa2 thay đổi sau khi probe được train | Đã đọc diff 06/2025 đến 03/2026: không có thay đổi nào ảnh hưởng tính toán của ViT-L | Không phải nguyên nhân |
| Chọn nhầm 1 trong 20 probe | File probe chỉ chứa 1 classifier (~54M tham số) | Không phải nguyên nhân |

**Kết luận:** pipeline tái tạo trung thực code gốc, nhưng checkpoint đã phát hành chỉ cho **28.6** (cách đếm official),
**thấp hơn 32.7 khoảng 4.1 điểm**, và chưa giải thích được. Các khả năng còn lại đều nằm ngoài tầm kiểm soát của ta:
- `vitl.pt` hoặc probe đã phát hành khác với bản dùng trong paper (có thể kiểm tra bằng probe SSv2/Diving48)
- con số trong paper được chọn trên val (lấy max của 20 probe và qua các epoch)
- bản copy video của Meta khác với bản gốc

**Phát hiện quan trọng:** với giao thức anticipation chuẩn (clip dừng 1s trước khi action bắt đầu), cùng model chỉ đạt
**12.8**, tức giảm hơn một nửa. Điều này cho thấy con số của code gốc phụ thuộc nhiều vào việc clip nhìn thấy một phần action.
Lưu ý: probe được train với clip neo gần cuối action, nên muốn so sánh công bằng thì cần train lại probe theo giao thức chuẩn.
Việc này nằm trong giai đoạn nghiên cứu. Vấn đề này cũng được nêu ở [facebookresearch/vjepa2#173](https://github.com/facebookresearch/vjepa2/issues/173).
