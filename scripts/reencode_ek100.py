"""Make a reusable, Kaggle-sized copy of EK100 videos at the eval resolution (CPU only).

The original videos are ~31 Mbit/s 1080p (val: 184 GB, train: ~1 TB). The V-JEPA 2 eval first
resizes every frame with cv2.INTER_LINEAR to a 292px short side (and skips that step when the
short side is already 292), then center-crops 256. So we decode each original video, apply
exactly that resize, pad the width to an even number (519 -> 520, which leaves the center crop
unchanged), and encode with x264 at high quality. Every frame and the exact frame rate are kept,
so annotation frame ids still index the same frames. The only difference left is compression
noise; check it with the val logits (see notebooks/kaggle_reencode_ek100.ipynb).

Shard the work over sessions / accounts with --shard_id/--num_shards; a session stops before the
20 GB output limit. Videos listed in any manifest.csv under --done_glob are skipped (resume).

python scripts/reencode_ek100.py --split val --annotations_dir ek100-ann --out_dir out/ek100_292 \
    --tmp_dir /kaggle/tmp/raw --shard_id 0 --num_shards 2
"""

import argparse
import glob
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stream_eval_ek100 import fetch  # noqa: E402


MANIFEST_COLS = ["video_id", "split", "ok", "mbps", "frames_written", "filled_frames", "filled_ranges", "src_fps",
                 "encode_sec", "src_w", "src_h", "out_w", "out_h", "pad_w", "pad_h", "src_frames", "out_frames",
                 "out_fps", "src_fps_decord", "error"]


def ffmpeg_exe():
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    import imageio_ffmpeg  # pip install imageio-ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


def target_size(h, w, short_side):
    """Same as get_resize_sizes in vjepa2 src/datasets/utils/video/functional.py."""
    if w < h:
        return int(short_side * h / w), short_side
    return short_side, int(short_side * w / h)


def ranges(idx):
    """[3,4,5,9] -> "3-5,9" (for the manifest)."""
    out, i = [], 0
    while i < len(idx):
        j = i
        while j + 1 < len(idx) and idx[j + 1] == idx[j] + 1:
            j += 1
        out.append(f"{idx[i]}-{idx[j]}" if j > i else f"{idx[i]}")
        i = j + 1
    return ",".join(out)


def decoded_frames(src, expected, filled):
    """Yield exactly `expected` RGB frames in presentation order, frame i at index i as decord counts them.

    Some original EK100 files contain corrupt packets (PyAV raises InvalidDataError, decord fails on
    clips near them). Those packets are skipped; a frame that cannot be decoded is replaced by the
    previous one so every later frame keeps its index. The replaced indices are appended to `filled`.
    """
    import av

    with av.open(src) as c:
        s = c.streams.video[0]
        s.thread_type = "AUTO"
        fps = float(s.average_rate)
        t0 = s.start_time or 0
        nxt, last, counter = 0, None, 0

        def frames_of(pkt):
            try:
                return pkt.decode()
            except av.error.InvalidDataError:
                return []

        for pkt in c.demux(s):
            for fr in frames_of(pkt):
                idx = round(float((fr.pts - t0) * s.time_base) * fps) if fr.pts is not None else counter
                counter += 1
                if idx < nxt or idx >= expected:
                    continue
                img = fr.to_ndarray(format="rgb24")
                while nxt < idx:  # frames lost to corrupt packets
                    filled.append(nxt)
                    yield last if last is not None else img
                    nxt += 1
                last = img
                yield img
                nxt += 1
    while nxt < expected:  # missing frames at the very end
        filled.append(nxt)
        yield last
        nxt += 1


def reencode(src, dst, short_side, crf, preset, threads, pix_fmt="yuv420p"):
    """Decode every frame, resize like the eval, encode. Returns a dict for the manifest."""
    import av
    import cv2
    from decord import VideoReader, cpu

    t0 = time.time()
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    vr = VideoReader(src, ctx=cpu(0))
    expected = len(vr)
    del vr
    with av.open(src) as c:
        rate = c.streams.video[0].average_rate
    proc, n, size, filled = None, 0, None, []
    for img in decoded_frames(src, expected, filled):
        if proc is None:
            h, w = img.shape[:2]
            oh, ow = target_size(h, w, short_side)
            pw, ph = ow + ow % 2, oh + oh % 2
            size = dict(src_w=w, src_h=h, out_w=ow, out_h=oh, pad_w=pw - ow, pad_h=ph - oh)
            cmd = [ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                   "-s", f"{pw}x{ph}", "-r", str(rate), "-i", "-", "-an", "-c:v", "libx264", "-preset", preset,
                   "-crf", str(crf), "-pix_fmt", pix_fmt, "-threads", str(threads), dst]
            proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        small = cv2.resize(img, (ow, oh), interpolation=cv2.INTER_LINEAR)
        if size["pad_w"] or size["pad_h"]:  # pad right / bottom: the center crop does not move
            small = np.pad(small, ((0, size["pad_h"]), (0, size["pad_w"]), (0, 0)), mode="edge")
        proc.stdin.write(small.tobytes())
        n += 1
    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError("ffmpeg failed")
    return dict(frames_written=n, src_fps=float(rate), encode_sec=round(time.time() - t0, 1),
                filled_frames=len(filled), filled_ranges=ranges(filled)[:500], **size)


def verify(src, dst, info):
    """The copy must have exactly the original frame count (as decord counts it) and frame rate."""
    from decord import VideoReader, cpu

    a, b = VideoReader(src, ctx=cpu(0)), VideoReader(dst, ctx=cpu(0))
    res = dict(src_frames=len(a), out_frames=len(b), out_fps=b.get_avg_fps(), src_fps_decord=a.get_avg_fps())
    del a, b
    ok = (res["src_frames"] == res["out_frames"] == info["frames_written"]
          and abs(res["out_fps"] - res["src_fps_decord"]) < 1e-3)
    return ok, res


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split", choices=["val", "train", "both"], default="val")
    p.add_argument("--annotations_dir", required=True, help="epic-kitchens-100-annotations clone")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--tmp_dir", required=True, help="scratch space for the original downloads")
    p.add_argument("--shard_id", type=int, default=0)
    p.add_argument("--num_shards", type=int, default=1)
    p.add_argument("--videos_file", default=None, help="restrict to these video_ids")
    p.add_argument("--done_glob", default="/kaggle/input/**/manifest.csv", help="manifests of earlier runs")
    p.add_argument("--short_side", type=int, default=292)
    p.add_argument("--crf", type=int, default=12)
    p.add_argument("--preset", default="veryfast")
    p.add_argument("--pix_fmt", default="yuv444p", help="yuv444p keeps full colour resolution (smallest error)")
    p.add_argument("--encode_workers", type=int, default=2)
    p.add_argument("--ffmpeg_threads", type=int, default=2)
    p.add_argument("--parallel_downloads", type=int, default=2)
    p.add_argument("--connections", type=int, default=8)
    p.add_argument("--tries", type=int, default=3)
    p.add_argument("--max_raw_on_disk", type=int, default=6)
    p.add_argument("--max_hours", type=float, default=11.0)
    p.add_argument("--max_output_gb", type=float, default=19.0, help="Kaggle keeps at most 20 GB of output")
    p.add_argument("--copy_from", default=None, help="take originals from a local folder (testing)")
    args = p.parse_args()

    t_start = time.time()
    lock = threading.Lock()
    os.makedirs(args.out_dir, exist_ok=True)
    manifest_path = os.path.join(args.out_dir, "manifest.csv")
    log_path = os.path.join(args.out_dir, "reencode.log")

    def log(msg):
        line = f"{time.strftime('%H:%M:%S')} {msg}"
        with lock:
            print(line, flush=True)
            with open(log_path, "a") as f:
                f.write(line + "\n")

    splits = ["validation", "train"] if args.split == "both" else ["validation" if args.split == "val" else "train"]
    vids = []
    for sp in splits:
        df = pd.read_csv(os.path.join(args.annotations_dir, f"EPIC_100_{sp}.csv"))
        vids += [(v, sp) for v in sorted(df.video_id.unique())]
    if args.videos_file:
        with open(args.videos_file) as f:
            keep = set(f.read().split())
        vids = [x for x in vids if x[0] in keep]
    vids = vids[args.shard_id :: args.num_shards]

    if os.path.exists(manifest_path):  # rewrite with the fixed columns, keeping only finished videos
        old = pd.read_csv(manifest_path, dtype=str)
        old = old[old.ok == "True"].reindex(columns=MANIFEST_COLS)
        old.to_csv(manifest_path, index=False)
    done = set()
    for m in glob.glob(args.done_glob, recursive=True) + [manifest_path]:
        if os.path.exists(m):
            d = pd.read_csv(m, dtype=str)
            done |= set(d.video_id[d.ok == "True"])
    todo = [x for x in vids if x[0] not in done]
    vi = pd.read_csv(os.path.join(args.annotations_dir, "EPIC_100_video_info.csv")).set_index("video_id")
    log(f"shard {args.shard_id}/{args.num_shards}: {len(vids)} videos, {len(vids) - len(todo)} already done, "
        f"{len(todo)} to go (~{vi.loc[[v for v, _ in todo], 'duration'].sum() / 3600:.1f} h of video)")

    class FetchArgs:  # what stream_eval_ek100.fetch needs
        copy_from, connections, tries = args.copy_from, args.connections, args.tries

    ready = queue.Queue()
    stop = threading.Event()

    def out_gb():
        return sum(os.path.getsize(f) for f in glob.glob(os.path.join(args.out_dir, "**", "*.MP4"), recursive=True)) / 1e9

    def budget_left():
        if (time.time() - t_start) / 3600 > args.max_hours:
            return "time budget"
        if out_gb() > args.max_output_gb:
            return "output size budget"
        return None

    def downloader():
        with ThreadPoolExecutor(args.parallel_downloads) as pool:
            for vid, sp in todo:
                while ready.qsize() >= args.max_raw_on_disk and not stop.is_set():
                    time.sleep(5)
                if stop.is_set() or budget_left():
                    break
                raw = os.path.join(args.tmp_dir, f"{vid}.MP4")
                ref = vi.loc[vid] if vid in vi.index else None

                def job(vid=vid, sp=sp, raw=raw, ref=ref):
                    t0 = time.time()
                    if fetch(vid, raw, ref, FetchArgs, log):
                        log(f"[download] {vid} {os.path.getsize(raw) / 1e9:.2f} GB in {time.time() - t0:.0f}s")
                        ready.put((vid, sp, raw))
                    else:
                        log(f"[FAIL download] {vid}")
                        append_manifest(dict(video_id=vid, split=sp, ok=False, error="download failed"))

                pool.submit(job)
        ready.put(None)

    def append_manifest(row):
        with lock:  # fixed column order: failed rows have fewer fields
            pd.DataFrame([row]).reindex(columns=MANIFEST_COLS).to_csv(
                manifest_path, mode="a", header=not os.path.exists(manifest_path), index=False)

    def encoder():
        while True:
            item = ready.get()
            if item is None:
                ready.put(None)
                return
            vid, sp, raw = item
            if stop.is_set():  # budget reached: drop what is still queued
                if os.path.exists(raw):
                    os.remove(raw)
                continue
            dst = os.path.join(args.out_dir, vid.split("_")[0], "videos", f"{vid}.MP4")
            row = dict(video_id=vid, split=sp)
            try:
                info = reencode(raw, dst, args.short_side, args.crf, args.preset, args.ffmpeg_threads, args.pix_fmt)
                ok, check = verify(raw, dst, info)
                dur = check["src_frames"] / check["src_fps_decord"]
                row.update(ok=ok, mbps=round(os.path.getsize(dst) * 8 / 1e6 / dur, 2), **info, **check)
                log(f"[{'ok' if ok else 'MISMATCH'}] {vid}: {check['src_frames']} frames, {info['encode_sec']:.0f}s "
                    f"({dur / max(info['encode_sec'], 1e-6):.1f}x realtime), {os.path.getsize(dst) / 1e9:.2f} GB, "
                    f"total out {out_gb():.1f} GB")
                if not ok and os.path.exists(dst):
                    os.remove(dst)
            except Exception as e:
                row.update(ok=False, error=repr(e)[:300])
                log(f"[FAIL encode] {vid}: {e!r}")
                if os.path.exists(dst):
                    os.remove(dst)
            finally:
                if os.path.exists(raw):
                    os.remove(raw)
            append_manifest(row)
            why = budget_left()
            if why:
                stop.set()
                log(f"[stop] {why} reached; already-downloaded videos are discarded, rerun to continue")

    os.makedirs(args.tmp_dir, exist_ok=True)
    th = threading.Thread(target=downloader)
    th.start()
    workers = [threading.Thread(target=encoder) for _ in range(args.encode_workers)]
    for w in workers:
        w.start()
    th.join()
    for w in workers:
        w.join()
    if os.path.exists(manifest_path):
        m = pd.read_csv(manifest_path)
        ok = m.ok.astype(str) == "True"
        log(f"done: {int(ok.sum())} ok, {int((~ok).sum())} failed in this output; {out_gb():.1f} GB, "
            f"{(time.time() - t_start) / 3600:.2f} h")


if __name__ == "__main__":
    main()
