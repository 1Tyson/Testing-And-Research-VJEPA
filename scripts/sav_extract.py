"""Encode SAV keyframes with the frozen V-JEPA 2 encoder and cache per-person features.

  python scripts/sav_extract.py --ann_dir sav/annotations --clips_dir sav/video_clips --out_dir feats \
      --ckpt vitl.pt --model vitl --device cuda:0 --shard_id 0 --num_shards 2

Writes <out_dir>/boxes_<split>.csv (one row per box, see classroom.sav.build_box_table) and
<out_dir>/<split>/shard<k>_chunk<n>.npz with box_id, box_tok [K, 9, D] fp16, key, ctx [M, 2, D] fp16, clip.
Resumable: clips already in a chunk are skipped. Clips missing on disk are listed in missing_<split>.txt.
"""

import argparse
import glob
import os
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from classroom.features import CHECKPOINTS, build_encoder, encode, pool_boxes  # noqa: E402
from classroom.sav import build_box_table  # noqa: E402
from classroom.video import decode_at_fps, keyframe_indices, target_size, to_model_input  # noqa: E402

DTYPES = dict(fp16=torch.float16, bf16=torch.bfloat16, fp32=torch.float32)


class ClipDataset(Dataset):
    """One item per clip: all its keyframes, resized to a common size."""

    def __init__(self, clips, keyframes, clips_dir, args):
        self.clips, self.keyframes, self.clips_dir, self.args = clips, keyframes, clips_dir, args

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, i):
        clip = self.clips[i]
        a = self.args
        try:
            frames = decode_at_fps(os.path.join(self.clips_dir, f"{clip}.mp4"), fps=a.fps)
        except Exception as e:  # corrupt clip: report and skip
            return dict(clip=clip, error=str(e))
        size = target_size(*frames.shape[1:3], short_side=a.short_side)
        xs, keys = [], []
        for key, ts in self.keyframes[clip]:
            idx = keyframe_indices(ts, a.num_frames, a.stride, a.fps, a.timestamp_start, a.timestamp_end)
            idx = np.clip(idx, 0, len(frames) - 1)
            xs.append(to_model_input(frames[idx], size))
            keys.append(key)
        return dict(clip=clip, x=torch.stack(xs), keys=keys)


def done_clips(split_dir, shard_id):
    done = set()
    for f in glob.glob(os.path.join(split_dir, f"shard{shard_id}_chunk*.npz")):
        done.update(np.load(f, allow_pickle=False)["clip"].tolist())
    return done


def write_table(df, path):
    if not os.path.exists(path):
        tmp = f"{path}.{os.getpid()}.tmp"
        df.to_csv(tmp, index=False)
        os.replace(tmp, path)


def run_split(split, model, args, deadline):
    df = build_box_table(args.ann_dir, split, with_proposals=split in args.proposal_splits, score_thr=args.det_thr)
    write_table(df, os.path.join(args.out_dir, f"boxes_{split}.csv"))
    split_dir = os.path.join(args.out_dir, split)
    os.makedirs(split_dir, exist_ok=True)

    clips = sorted(df.video_id.unique())
    clips = [c for i, c in enumerate(clips) if i % args.num_shards == args.shard_id]
    on_disk = {os.path.splitext(f)[0] for f in os.listdir(args.clips_dir)}
    missing = [c for c in clips if c not in on_disk]
    with open(os.path.join(args.out_dir, f"missing_{split}_shard{args.shard_id}.txt"), "w") as f:
        f.write("\n".join(missing))
    done = done_clips(split_dir, args.shard_id)
    todo = [c for c in clips if c in on_disk and c not in done]
    if args.max_clips:
        todo = todo[: args.max_clips]
    print(f"[{split}] shard {args.shard_id}: {len(clips)} clips, {len(missing)} missing on disk, "
          f"{len(done)} done, {len(todo)} to encode; {len(df)} boxes in split", flush=True)
    if not todo:
        return

    kf = df.drop_duplicates("key")[["video_id", "key", "ts"]]
    keyframes = {c: list(zip(g.key, g.ts)) for c, g in kf.groupby("video_id")}
    boxes_by_key = {k: g for k, g in df.groupby("key")}
    loader = DataLoader(ClipDataset(todo, keyframes, args.clips_dir, args), batch_size=None, shuffle=False,
                        num_workers=args.workers, prefetch_factor=4 if args.workers else None)

    chunk_id = len(glob.glob(os.path.join(split_dir, f"shard{args.shard_id}_chunk*.npz")))
    buf = dict(box_id=[], box_tok=[], key=[], ctx=[], clip=[])
    errors = []
    t0, n_kf = time.time(), 0

    def flush():
        nonlocal chunk_id
        if not buf["clip"]:
            return
        out = os.path.join(split_dir, f"shard{args.shard_id}_chunk{chunk_id:05d}.npz")
        np.savez(out + ".tmp.npz",
                 box_id=np.concatenate(buf["box_id"]), box_tok=np.concatenate(buf["box_tok"]),
                 key=np.array(buf["key"]), ctx=np.concatenate(buf["ctx"]), clip=np.array(buf["clip"]))
        os.replace(out + ".tmp.npz", out)
        chunk_id += 1
        for v in buf.values():
            v.clear()

    for item in loader:
        if "error" in item:
            errors.append(f"{item['clip']}\t{item['error']}")
            continue
        x = item["x"].to(args.device, non_blocking=True)
        for s in range(0, len(x), args.batch_size):
            xb = x[s: s + args.batch_size]
            keys = item["keys"][s: s + args.batch_size]
            with torch.autocast("cuda", dtype=DTYPES[args.dtype], enabled=args.dtype != "fp32"):
                fmap = encode(model, xb)
            rows = [boxes_by_key[k] for k in keys]
            boxes = torch.tensor(np.concatenate([r[["x1", "y1", "x2", "y2"]].to_numpy() for r in rows]),
                                 dtype=torch.float32, device=args.device)
            bidx = torch.tensor(np.concatenate([np.full(len(r), j) for j, r in enumerate(rows)]), device=args.device)
            box_tok, ctx = pool_boxes(fmap, boxes, bidx)
            buf["box_id"].append(np.concatenate([r.box_id.to_numpy() for r in rows]))
            buf["box_tok"].append(box_tok.half().cpu().numpy())
            buf["key"].extend(keys)
            buf["ctx"].append(ctx.half().cpu().numpy())
            n_kf += len(keys)
        buf["clip"].append(item["clip"])
        if len(buf["clip"]) >= args.clips_per_chunk:
            flush()
            rate = n_kf / (time.time() - t0)
            print(f"[{split}] shard {args.shard_id}: {n_kf} keyframes, {rate:.2f} kf/s", flush=True)
        if time.time() > deadline:
            print("time limit reached, stopping (rerun to resume)", flush=True)
            break
    flush()
    if errors:
        with open(os.path.join(args.out_dir, f"decode_errors_{split}_shard{args.shard_id}.txt"), "a") as f:
            f.write("\n".join(errors) + "\n")
        print(f"[{split}] {len(errors)} clips failed to decode", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ann_dir", required=True)
    ap.add_argument("--clips_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--model", default="vitl", choices=sorted(CHECKPOINTS))
    ap.add_argument("--splits", nargs="+", default=["train", "val"])
    ap.add_argument("--proposal_splits", nargs="*", default=["val"],
                    help="splits whose detected person boxes are encoded too (needed for detection mAP)")
    ap.add_argument("--det_thr", type=float, default=0.8)
    ap.add_argument("--short_side", type=int, default=256)
    ap.add_argument("--num_frames", type=int, default=16)
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--timestamp_start", type=int, default=0)
    ap.add_argument("--timestamp_end", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=4, help="keyframes per encoder forward")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--clips_per_chunk", type=int, default=200)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", default="fp16", choices=sorted(DTYPES))
    ap.add_argument("--shard_id", type=int, default=0)
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--max_hours", type=float, default=11.0)
    ap.add_argument("--max_clips", type=int, default=0, help="encode at most N clips per split (quick test)")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    deadline = time.time() + args.max_hours * 3600
    model = build_encoder(args.ckpt, CHECKPOINTS[args.model]["model_name"], num_frames=args.num_frames,
                          resolution=args.short_side).to(args.device)  # fp32 weights, autocast for speed
    for split in args.splits:
        run_split(split, model, args, deadline)
        if time.time() > deadline:
            break


if __name__ == "__main__":
    main()
