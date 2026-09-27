# Testing-And-Research-VJEPA

Kiểm chứng và nghiên cứu cải tiến **V-JEPA 2** cho bài toán action anticipation trên **EPIC-KITCHENS-100**.

- Mục tiêu 1: tái tạo **32.7 mean-class Recall@5 (action)** của ViT-L/16 256px bằng probe do Meta phát hành.
- Mục tiêu 2: cải tiến phần phía trên encoder đã pretrain. Xem [docs/PLAN.md](docs/PLAN.md).
- Chi tiết giao thức và những điểm dễ gây lệch: [docs/protocol_notes.md](docs/protocol_notes.md).

## Chạy nhanh trên Kaggle
Mở [notebooks/kaggle_verify_ek100.ipynb](notebooks/kaggle_verify_ek100.ipynb) (GPU T4 x2, Internet On).
Notebook `git clone` repo này, nên repo phải để **public**, hoặc upload repo lên Kaggle dưới dạng dataset.

## Chạy thủ công
```bash
git clone https://github.com/facebookresearch/vjepa2 && git -C vjepa2 checkout 204698b
export VJEPA2_ROOT=$PWD/vjepa2
pip install -r requirements.txt

# 1. index clip + class (CPU)
python scripts/prepare_ek100.py --video_root <EK100 videos> \
  --train_csv EPIC_100_train.csv --val_csv EPIC_100_validation.csv --out_dir work --probe_videos
# 2. logits (mỗi GPU một shard, chạy tiếp được nếu bị ngắt)
python scripts/eval_ek100.py --work_dir work --out_dir logits --vitl_ckpt vitl.pt \
  --probe_ckpt ek100-vitl-256.pt --device cuda:0 --shard_id 0 --num_shards 2
# 3. metric: official (cách đếm 64 GPU của Meta) và clean
python scripts/compute_metrics.py --work_dir work --logits_dir logits/official --out results/verify_ek100_vitl.json
```

## Cấu trúc
```
configs/ek100_vitl_inference.yaml   # bản sao config inference gốc
src/vjepa_ek100/protocol.py         # class mapping, danh sách clip, frame index, mô phỏng cách đếm của Meta
src/vjepa_ek100/metrics.py          # mean-class recall@k (khớp ClassMeanRecall gốc)
src/vjepa_ek100/data.py             # dataset map-style, decode giống code gốc
src/vjepa_ek100/model.py            # load encoder + predictor + probe từ code gốc
scripts/                            # prepare / eval / compute_metrics
tests/                              # so sánh với code gốc (cần VJEPA2_ROOT)
```

## Test
```bash
VJEPA2_ROOT=/path/to/vjepa2 python -m pytest -q tests
```
