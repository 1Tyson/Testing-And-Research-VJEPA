# Kiểm chứng V-JEPA 2 ViT-L/16 256px trên EK100 (probe `ek100-vitl-256.pt` của Meta)

Chạy trên Kaggle 2×T4, fp16, `--anchor official` (giống code gốc), vjepa2 commit `204698b`.
9296 clip val, trong đó 36 clip bị bỏ qua vì cửa sổ frame vượt quá cuối video (code gốc cũng bỏ các clip này).
Thời gian: khoảng 40 phút (2.0 và 1.8 clip/s trên mỗi GPU). `fp32_retries = 0`.

| Cách đếm | Action R@5 | Verb R@5 | Noun R@5 |
|---|---|---|---|
| **official** (mô phỏng cách đếm của lần chạy 64 GPU) | **28.61** | 55.23 | 46.79 |
| clean (mỗi clip đếm 1 lần) | 27.09 | 48.95 | 45.88 |
| Paper / README | **32.7** | – | – |

**Kết luận tạm thời: lệch −4.1 điểm, CHƯA tái tạo được con số 32.7.** Đang chẩn đoán nguyên nhân bằng
`notebooks/kaggle_diagnose_ek100.ipynb` (xem `docs/protocol_notes.md`).
