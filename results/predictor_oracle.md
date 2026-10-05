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
