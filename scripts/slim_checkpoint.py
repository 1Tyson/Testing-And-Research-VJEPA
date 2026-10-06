"""Keep only what inference needs from a V-JEPA 2 pretraining checkpoint (target encoder + predictor, fp16).

The ViT-g checkpoints are large; two eval processes (one per GPU) each loading the full file would not fit in
Kaggle's RAM. The file is memory-mapped, so slimming itself needs little RAM.

python scripts/slim_checkpoint.py vitg-384.pt vitg-384-slim.pt
"""

import sys

import torch


def main(src, dst, keys=("target_encoder", "predictor")):
    try:
        ckpt = torch.load(src, map_location="cpu", mmap=True, weights_only=False)
    except RuntimeError:  # old (non-zip) serialization cannot be memory-mapped
        ckpt = torch.load(src, map_location="cpu", weights_only=False)
    missing = [k for k in keys if k not in ckpt]
    if missing:
        raise SystemExit(f"{src}: no {missing} (has {list(ckpt)})")
    out = {k: {n: (t.half() if t.is_floating_point() else t).clone() for n, t in ckpt[k].items()} for k in keys}
    torch.save(out, dst)
    n = sum(t.numel() for d in out.values() for t in d.values())
    print(f"{dst}: {', '.join(keys)}; {n / 1e6:.0f}M parameters in fp16")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
