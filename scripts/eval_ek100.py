"""Run the released V-JEPA 2 ViT-L EK100 probe on the val clips and dump logits.

Resumable (chunked .npz) and shardable across GPUs / Kaggle sessions:
  python scripts/eval_ek100.py --work_dir work/ek100 --out_dir work/logits \
      --vitl_ckpt vitl.pt --probe_ckpt ek100-vitl-256.pt --shard_id 0 --num_shards 2 --device cuda:0

--watch_dir DIR: load the model once, then evaluate each video of this shard as soon as
`DIR/<video_id>.ready` appears (written by stream_eval_ek100.py after downloading it), optionally
delete the video afterwards, and exit once `DIR/ALL_DOWNLOADED` exists and nothing is left.

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


def pool_tokens(x, grid, pool):
    """[B, T*grid*grid, D] -> [B, T, pool*pool, D] by spatial average pooling."""
    B, N, D = x.shape
    T = N // (grid * grid)
    x = x.reshape(B * T, grid, grid, D).permute(0, 3, 1, 2)
    x = F.adaptive_avg_pool2d(x.float(), pool)
    return x.permute(0, 2, 3, 1).reshape(B, T, pool * pool, D)


class ShardWriter:
    """Chunked, append-only logits store for one (anchor, shard). Undecodable clips go to failed.json."""

    def __init__(self, shard_dir, save_every):
        os.makedirs(shard_dir, exist_ok=True)
        self.dir, self.save_every = shard_dir, save_every
        self.failed = load_failed(shard_dir)
        self.chunk = len(glob.glob(os.path.join(shard_dir, "chunk_*.npz")))
        self.buf = dict(clip_id=[], verb=[], noun=[], action=[], feats=[])
        self.written = 0

    def done(self):
        return done_clip_ids(self.dir) | self.failed

    def add(self, clip_id, out, feats=None):
        self.buf["clip_id"].append(clip_id)
        for k in ("verb", "noun", "action"):
            self.buf[k].append(out[k].float().cpu().numpy().astype(np.float16))
        if feats is not None:
            self.buf["feats"].append(feats)
        if sum(len(c) for c in self.buf["clip_id"]) >= self.save_every:
            self.flush()

    def add_failed(self, ids):
        if ids:
            self.failed |= set(int(c) for c in ids)
            with open(os.path.join(self.dir, "failed.json"), "w") as f:
                json.dump(sorted(self.failed), f)

    def flush(self):
        if not self.buf["clip_id"]:
            return
        out = {k: np.concatenate(v) for k, v in self.buf.items() if v}
        np.savez(os.path.join(self.dir, f"chunk_{self.chunk:05d}.npz"), **out)
        self.chunk += 1
        self.written += len(out["clip_id"])
        for v in self.buf.values():
            v.clear()


def evaluate(todo, anchor, writer, model, probe, data_cfg, args, device, dtype, stats):
    """Evaluate the clips in `todo`; returns an error string if the DataLoader died, else None."""
    at = float(data_cfg["anticipation_time_sec"][0])
    grid = data_cfg["resolution"] // 16
    ds = EK100ClipDataset(
        todo,
        eval_transform(data_cfg["resolution"]),
        frames_per_clip=data_cfg["frames_per_clip"],
        fps=data_cfg["frames_per_second"],
        anticipation_time=at,
        anchor=anchor,
    )
    loader = torch.utils.data.DataLoader(
        ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate, pin_memory=True,
        worker_init_fn=worker_init_fn if args.num_workers > 0 else None,
    )
    t0, seen = time.time(), 0
    try:
        with torch.inference_mode():
            for batch in loader:
                writer.add_failed(batch["failed"])
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
                pooled = None
                if args.save_feats_pool:
                    pooled = pool_tokens(feats, grid, args.save_feats_pool).cpu().numpy().astype(np.float16)
                writer.add(batch["clip_id"].numpy(), out, pooled)
                seen += x.shape[0]
                if seen % args.save_every < x.shape[0]:
                    rate = seen / (time.time() - t0)
                    print(f"[{anchor}] {seen}/{len(todo)} clips  {rate:.2f} clip/s  "
                          f"eta {(len(todo) - seen) / rate / 60:.1f} min", flush=True)
    except RuntimeError as e:  # e.g. a DataLoader worker aborted: keep what we have, a rerun resumes
        print(f"[error] {e!r}", flush=True)
        return repr(e)
    finally:
        writer.flush()
        stats["clips"] += seen
    return None


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
    p.add_argument("--anchor", "--anchors", dest="anchors", nargs="+", choices=["official", "action_start"],
                   default=["official"], help="clip end point(s); several are evaluated from the same video")
    p.add_argument("--max_clips", type=int, default=0, help="stop after N clips (speed benchmark)")
    p.add_argument("--save_every", type=int, default=256)
    p.add_argument("--only_videos", default=None, help="file with one video_id per line: evaluate only these")
    p.add_argument("--watch_dir", default=None, help="streaming mode, see module doc")
    p.add_argument("--delete_after", action="store_true", help="watch mode: delete each video once evaluated")
    p.add_argument("--save_feats_pool", type=int, default=0, help="also save features pooled to PxP per time step")
    args = p.parse_args()

    add_vjepa2_to_path(args.vjepa2_root)
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    data_cfg = cfg["experiment"]["data"]
    device = torch.device(args.device)
    dtype = DTYPES[args.dtype]

    clips = pd.read_csv(os.path.join(args.work_dir, "val_clips.csv"))
    with open(os.path.join(args.work_dir, "classes.json")) as f:
        classes = json.load(f)
    if args.only_videos:
        with open(args.only_videos) as f:
            clips = clips[clips.video_id.isin(f.read().split())]
    clips = clips[clips.video_order % args.num_shards == args.shard_id]
    writers = {a: ShardWriter(os.path.join(args.out_dir, a, f"shard{args.shard_id}"), args.save_every) for a in args.anchors}

    def todo_for(anchor, video_ids=None):
        c = clips if video_ids is None else clips[clips.video_id.isin(video_ids)]
        return c[~c.clip_id.isin(writers[anchor].done())]

    remaining = {a: len(todo_for(a)) for a in args.anchors}
    print(f"shard {args.shard_id}/{args.num_shards}: {len(clips)} clips; to go {remaining}", flush=True)
    if not any(remaining.values()):
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
    stats, crash, t0 = dict(clips=0, fp32_retries=0), None, time.time()

    if args.watch_dir is None:
        for a in args.anchors:
            todo = todo_for(a)
            if args.max_clips:
                todo = todo.iloc[: args.max_clips]
            crash = crash or evaluate(todo, a, writers[a], model, probe, data_cfg, args, device, dtype, stats)
    else:
        shard_videos = list(dict.fromkeys(clips.sort_values("video_order").video_id))
        pending = [v for v in shard_videos if any(len(todo_for(a, [v])) for a in args.anchors)]
        while pending and crash is None:
            ready = [v for v in pending if os.path.exists(os.path.join(args.watch_dir, f"{v}.ready"))]
            given_up = [v for v in pending if os.path.exists(os.path.join(args.watch_dir, f"{v}.failed"))]
            pending = [v for v in pending if v not in given_up]
            if not ready:
                if os.path.exists(os.path.join(args.watch_dir, "ALL_DOWNLOADED")):
                    break
                time.sleep(10)
                continue
            for a in args.anchors:
                crash = crash or evaluate(todo_for(a, ready), a, writers[a], model, probe, data_cfg, args, device,
                                          dtype, stats)
            if crash:
                break
            for v in ready:
                if args.delete_after:
                    path = clips[clips.video_id == v].video_path.iloc[0]
                    if os.path.exists(path):
                        os.remove(path)
                os.replace(os.path.join(args.watch_dir, f"{v}.ready"), os.path.join(args.watch_dir, f"{v}.done"))
            pending = [v for v in pending if v not in ready]
            print(f"[shard {args.shard_id}] evaluated {len(ready)} video(s), {len(pending)} left", flush=True)

    remaining = {a: len(todo_for(a)) for a in args.anchors}
    el = time.time() - t0
    stats.update(seconds=el, clips_per_sec=stats["clips"] / max(el, 1e-6), remaining_in_shard=remaining,
                 failed={a: sorted(w.failed) for a, w in writers.items()}, crash=crash, dtype=args.dtype,
                 device=torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu")
    for w in writers.values():
        with open(os.path.join(w.dir, f"run_{int(time.time())}.json"), "w") as f:
            json.dump(stats, f, indent=2)
    print(json.dumps({k: v for k, v in stats.items() if k != "failed"}, indent=2))
    # 3 = crashed with work left in the shard: just run the same command again
    return 3 if crash and any(remaining.values()) and not args.max_clips else 0


if __name__ == "__main__":
    sys.exit(main())
