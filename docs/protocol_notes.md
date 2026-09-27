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
   đây là một phân tích đáng đưa vào bài.
3. **Cách đếm clip trong lúc validate.** Mỗi rank chạy đúng `ipe = N // (64×2)` batch và quay lại đầu khi hết dữ liệu.
   Vì mỗi rank giữ `videos[rank::64]` với số clip rất chênh lệch, có clip bị đếm nhiều lần và có clip không được
   đếm. `protocol.official_eval_multiplicity` mô phỏng lại việc này. Test so sánh với pipeline
   webdataset + DataLoader thật cho kết quả khớp hoàn toàn. `compute_metrics.py` báo cả hai con số **official** và **clean**.
4. **fps của video.** Bước lấy mẫu là `int(vfps/8)`: bằng 7 ở 60fps và 6 ở 50fps. Nếu video đã bị encode lại
   (ví dụ 30fps), cả khoảng thời gian quan sát lẫn frame id trong CSV đều lệch. `prepare_ek100.py --probe_videos` sẽ cảnh báo.
5. **Độ chính xác số.** Paper dùng bf16, T4 thì dùng fp16. Nếu gặp tràn số (NaN hoặc inf), batch đó được tính lại bằng fp32 (`fp32_retries` trong log).
6. Clip decode lỗi bị code gốc bỏ qua âm thầm. Ở đây chúng được ghi vào `failed` trong `run_*.json`.
