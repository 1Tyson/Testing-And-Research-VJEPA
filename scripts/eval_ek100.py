"""Run the released V-JEPA 2 ViT-L EK100 probe on the val clips and dump logits.

Resumable (chunked .npz) and shardable across GPUs / Kaggle sessions:
  python scripts/eval_ek100.py --work_dir work/ek100 --out_dir work/logits \
      --vitl_ckpt vitl.pt --probe_ckpt ek100-vitl-256.pt --shard_id 0 --num_shards 2 --device cuda:0
Metrics are computed afterwards by scripts/compute_metrics.py.
"""

import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from vjepa_ek100.data import EK100ClipDataset, collate, worker_init_fn  # noqa: E402
from vjepa_ek100.model import add_vjepa2_to_path, build_backbone, build_probe, eval_transform  # noqa: E402

DTYPES = dict(fp16=torch.float16, bf16=torch.bfloat16, fp32=torch.float32)


def done_clip_ids(shard_dir):
    done = set()
    for f in glob.glob(os.path.join(shard_dir, "chunk_*.npz")):
        done.update(np.load(f)["clip_id"].tolist())
    return done


def load_failed(shard_dir):
    path = os.path.join(shard_dir, "failed.json")
    if not os.path.exists(path):
        return set()
    with open(path) as f:
        return set(json.load(f))


def save_failed(shard_dir, failed):
    with open(os.path.join(shard_dir, "failed.json"), "w") as f:
        json.dump(sorted(int(c) for c in failed), f)


def pool_tokens(x, grid, pool):
    """[B, T*grid*grid, D] -> [B, T, pool*pool, D] by spatial average pooling."""
    B, N, D = x.shape
    T = N // (grid * grid)
    x = x.reshape(B * T, grid, grid, D).permute(0, 3, 1, 2)
    x = F.adaptive_avg_pool2d(x.float(), pool)
    return x.permute(0, 2, 3, 1).reshape(B, T, pool * pool, D)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=os.path.join(os.path.dirname(__file__), "..", "configs", "ek100_vitl_inference.yaml"))
    p.add_argument("--vjepa2_root", default=None)
    p.add_argument("--work_dir", required=True, help="output of prepare_ek100.py")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--vitl_ckpt", required=True)
    p.add_argument("--probe_ckpt", required=True)
    p.add_argument("--shard_id", type=int, default=0)
    p.add_argument("--num_shards", type=int, default=1)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--dtype", choices=list(DTYPES), default="fp16", help="T4 has no native bf16; paper used bf16")
    p.add_argument("--anchor", choices=["official", "action_start"], default="official")
    p.add_argument("--max_clips", type=int, default=0, help="stop after N clips (speed benchmark)")
    p.add_argument("--save_every", type=int, default=256)
    p.add_argument("--save_feats_pool", type=int, default=0, help="also save features pooled to PxP per time step")
    args = p.parse_args()

    add_vjepa2_to_path(args.vjepa2_root)
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    data_cfg = cfg["experiment"]["data"]
    at = float(data_cfg["anticipation_time_sec"][0])
    device = torch.device(args.device)
    dtype = DTYPES[args.dtype]

    clips = pd.read_csv(os.path.join(args.work_dir, "val_clips.csv"))
    with open(os.path.join(args.work_dir, "classes.json")) as f:
        classes = json.load(f)
    clips = clips[clips.video_order % args.num_shards == args.shard_id]
    shard_dir = os.path.join(args.out_dir, f"{args.anchor}", f"shard{args.shard_id}")
    os.makedirs(shard_dir, exist_ok=True)
    prev_failed = load_failed(shard_dir)  # clips that cannot be decoded count as done
    done = done_clip_ids(shard_dir) | prev_failed
    todo = clips[~clips.clip_id.isin(done)]
    if args.max_clips:
        todo = todo.iloc[: args.max_clips]
    print(f"shard {args.shard_id}/{args.num_shards}: {len(clips)} clips, {len(done)} done, {len(todo)} to go")
    if len(todo) == 0:
        return 0

    model = build_backbone(cfg, args.vitl_ckpt, device)
    probe = build_probe(
        cfg,
        num_verbs=len(classes["verbs"]),
        num_nouns=len(classes["nouns"]),
        num_actions=len(classes["actions"]),
        embed_dim=model.embed_dim,
        device=device,
        probe_checkpoint=args.probe_ckpt,
    )

    ds = EK100ClipDataset(
        todo,
        eval_transform(data_cfg["resolution"]),
        frames_per_clip=data_cfg["frames_per_clip"],
        fps=data_cfg["frames_per_second"],
        anticipation_time=at,
        anchor=args.anchor,
    )
    loader = torch.utils.data.DataLoader(
        ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate, pin_memory=True,
        worker_init_fn=worker_init_fn if args.num_workers > 0 else None,
    )

    buf = dict(clip_id=[], verb=[], noun=[], action=[], feats=[])
    failed, stats = [], dict(clips=0, fp32_retries=0)
    chunk = len(glob.glob(os.path.join(shard_dir, "chunk_*.npz")))

    def flush():
        nonlocal chunk
        if not buf["clip_id"]:
            return
        out = {k: np.concatenate(v) for k, v in buf.items() if v}
        np.savez(os.path.join(shard_dir, f"chunk_{chunk:05d}.npz"), **out)
        stats["clips"] += len(out["clip_id"])
        chunk += 1
        for v in buf.values():
            v.clear()

    t0, crash = time.time(), None
    grid = data_cfg["resolution"] // 16
    try:
        with torch.inference_mode():
            for batch in loader:
                failed += batch["failed"]
                if batch["video"] is None:
                    continue
                x = batch["video"].to(device, non_blocking=True)
                ats = torch.full((x.shape[0],), at, device=device)
                with torch.autocast(device.type, dtype=dtype, enabled=dtype != torch.float32):
                    feats = model(x, ats)
                    if not torch.isfinite(feats).all():  # fp16 overflow: redo this batch in fp32
                        stats["fp32_retries"] += 1
                        with torch.autocast(device.type, enabled=False):
                            feats = model(x.float(), ats)
                    out = probe(feats)
                buf["clip_id"].append(batch["clip_id"].numpy())
                for k in ("verb", "noun", "action"):
                    buf[k].append(out[k].float().cpu().numpy().astype(np.float16))
                if args.save_feats_pool:
                    buf["feats"].append(pool_tokens(feats, grid, args.save_feats_pool).cpu().numpy().astype(np.float16))
                if sum(len(c) for c in buf["clip_id"]) >= args.save_every:
                    flush()
                    rate = stats["clips"] / (time.time() - t0)
                    eta = (len(todo) - stats["clips"]) / rate / 60
                    print(f"{stats['clips']}/{len(todo)} clips  {rate:.2f} clip/s  eta {eta:.1f} min", flush=True)
    except RuntimeError as e:  # e.g. a DataLoader worker aborted: keep what we have, a rerun resumes
        crash = repr(e)
        print(f"[error] {crash}", flush=True)
    flush()
    save_failed(shard_dir, prev_failed | set(failed))
    remaining = len(clips) - len(done_clip_ids(shard_dir) | prev_failed | set(failed))
    el = time.time() - t0
    stats.update(seconds=el, clips_per_sec=stats["clips"] / max(el, 1e-6), failed=failed, remaining_in_shard=remaining,
                 crash=crash, dtype=args.dtype, anchor=args.anchor,
                 device=torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu")
    with open(os.path.join(shard_dir, f"run_{int(time.time())}.json"), "w") as f:
        json.dump(stats, f, indent=2)
    print(json.dumps(stats, indent=2))
    # 3 = crashed with work left in the shard: just run the same command again
    return 3 if crash and remaining > 0 and not args.max_clips else 0


if __name__ == "__main__":
    sys.exit(main())
