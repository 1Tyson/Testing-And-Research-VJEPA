# Giám sát hành động học sinh trong lớp học dựa trên V-JEPA 2

Ràng buộc: **không tự gán nhãn một dataset lớn**. Phần cứng như các thí nghiệm EK100: Kaggle 2×T4.

## 1. Có làm được không?

Được, nhưng có hai giới hạn:

- **V-JEPA 2 không tự phân loại được khi chưa có nhãn.** Encoder chỉ cho ra đặc trưng video. Để biết "giơ tay" hay "dùng điện thoại",
  phải train một probe nhỏ (attentive probe, như probe EK100) trên dữ liệu có nhãn. Nhãn có thể lấy từ dataset công khai hoặc nhãn giả,
  không nhất thiết do mình gán.
- **Vẫn cần một tập kiểm tra nhỏ từ lớp học thật của mình** (khoảng 200–500 clip, vài giờ gán nhãn). Nếu không có, không đo được model chạy
  đúng tới đâu trên camera thật. Lớp học ở dataset công khai (phần lớn ở Trung Quốc, Ấn Độ) khác lớp học Việt Nam về góc camera, bàn ghế,
  đồng phục.

## 2. Dataset công khai có thể dùng (không tự gán nhãn)

| dataset | dạng | quy mô | ghi chú |
|---|---|---|---|
| SCB-Dataset (Whiffe/SCB-dataset) | **ảnh**, bbox | ~13k ảnh, ~123k nhãn, 19–20 lớp | giơ tay, đọc, viết, ngáp, dùng điện thoại, cúi đầu… Có biến thể SCB-ST (spatio-temporal, video) |
| SAV (Ritatanz/SAV) | **video**, đa nhãn | 4,324 clip, 758 lớp học, 15 lớp hành động | gần nhất với bài toán; định dạng hợp với V-JEPA |
| EduNet | **video** | ~7,851 clip, 20 lớp (giáo viên + học sinh) | clip từ YouTube |
| AVA | **video**, bbox theo người | phim | có các hành động nguyên tử như ngồi, đứng, viết, đọc, nghe, nhắn tin/xem điện thoại, ngủ; dùng để pretrain đầu dò theo người |

Cần kiểm tra license từng dataset trước khi dùng (một số chỉ cho nghiên cứu, phải điền form).

## 3. Các hướng tiếp cận

**A. Transfer từ dataset công khai (hướng chính).**
Phát hiện người (YOLO) → theo dõi (ByteTrack) → cắt tube video quanh từng học sinh → V-JEPA 2 ViT-L đóng băng → attentive probe đa nhãn.
Train probe trên SAV (+ EduNet). Encoder không cần train, nên vừa sức T4.

**B. Nhãn giả từ model ngôn ngữ–thị giác.**
Thu video lớp học chưa gán nhãn (camera của mình hoặc video công khai), cho một video-LLM / CLIP-like model gán nhãn tự động theo danh sách
hành động, lọc theo độ tin cậy, rồi train probe V-JEPA trên nhãn giả. Thay công gán nhãn bằng công kiểm tra ngẫu nhiên một phần nhãn.

**C. Không nhãn: phát hiện bất thường bằng predictor.**
Đây là điểm riêng của V-JEPA: predictor dự đoán token tương lai, sai số dự đoán ("độ bất ngờ") cao nghĩa là có chuyện khác thường
(đứng dậy, rời chỗ, đùa nghịch). Không cho biết *hành động gì*, chỉ cho biết *có gì đó bất thường, ở đâu, lúc nào*. Không cần nhãn nào để chạy;
chỉ cần vài ví dụ để đánh giá. Có thể là đóng góp nghiên cứu.

Đề xuất: A làm baseline, C làm hướng nghiên cứu, B nếu A không đủ tốt trên dữ liệu thật.

## 4. Các bước

1. Tải SAV, kiểm tra license, định dạng, danh sách 15 lớp.
2. Trích đặc trưng V-JEPA 2 ViT-L cho SAV (tái dùng `src/vjepa_ek100/model.py`), lưu ra đĩa.
3. Train attentive probe đa nhãn, báo mAP; so với baseline 67.9 mAP của bài báo SAV.
4. Pipeline video lớp học: YOLO + tracker → tube từng người → probe.
5. Gán nhãn tập kiểm tra nhỏ từ camera thật; đo mức tụt điểm khi chuyển miền.
6. Thử hướng C (surprise của predictor) trên cùng video.

## 5. Quyền riêng tư

Đối tượng là học sinh, thường là trẻ vị thành niên. Cần sự đồng ý của nhà trường và phụ huynh, tuân theo Nghị định 13/2023/NĐ-CP về bảo vệ dữ liệu
cá nhân. Nên xử lý trên máy tại chỗ, không lưu khuôn mặt, chỉ lưu thống kê hành vi theo lớp thay vì theo tên từng em.

## Nguồn

- SCB-Dataset: https://arxiv.org/abs/2304.02488
- SAV: https://arxiv.org/abs/2409.00926
- EduNet: https://www.mdpi.com/1424-8220/21/17/5699
