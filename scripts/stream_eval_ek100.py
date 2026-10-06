"""Download the ORIGINAL EK100 val videos and evaluate them on the fly (no re-encoded copy).

The originals (1920x1080, ~184 GB for the 138 val videos) do not fit a Kaggle dataset, and a
downscaled re-encode (e.g. 256p crf 23) lowers the score by several points, because the eval
resizes to a 292px short side before the 256px crop. So videos are fetched a few at a time
into scratch space, evaluated by one watch-mode eval_ek100.py process per GPU (model loaded
once), and deleted. Everything is resumable: logits live in --out_dir, finished videos are
skipped on the next run.

python scripts/prepare_ek100.py --download_layout --video_root /kaggle/tmp/ek100 ... --out_dir work
python scripts/stream_eval_ek100.py --work_dir work --video_info EPIC_100_video_info.csv \
    --out_dir logits --vitl_ckpt vitl.pt --probe_ckpt ek100-vitl-256.pt --gpus 0 1
"""

import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

EK55_BASE = "https://data.bris.ac.uk/datasets/3h91syskeag572hl6tvuovwv4d/videos"
EK100_EXT_BASE = "https://data.bris.ac.uk/datasets/2g1n6qdydwa9u22shpxqzp0t8m"


def url_candidates(video_id):
    pid, num = video_id.split("_")
    if len(num) == 3:  # recorded for the EK100 extension
        return [f"{EK100_EXT_BASE}/{pid}/videos/{video_id}.MP4"]
    return [f"{EK55_BASE}/test/{pid}/{video_id}.MP4", f"{EK55_BASE}/train/{pid}/{video_id}.MP4"]


def done_clip_ids(anchor_dir):
    done = set()
    for f in glob.glob(os.path.join(anchor_dir, "shard*", "chunk_*.npz")):
        done.update(np.load(f)["clip_id"].tolist())
    for f in glob.glob(os.path.join(anchor_dir, "shard*", "failed.json")):
        with open(f) as fh:
            done.update(json.load(fh))
    return done


def reopen_failed(anchor_dir, keep, log):
    """Remove failed clips (except `keep`, e.g. windows past the video end) from failed.json so they are evaluated again."""
    reopened = []
    for f in glob.glob(os.path.join(anchor_dir, "shard*", "failed.json")):
        with open(f) as fh:
            failed = set(json.load(fh))
        retry = failed - keep
        if retry:
            with open(f, "w") as fh:
                json.dump(sorted(failed & keep), fh)
            reopened += sorted(retry)
    if reopened:
        log(f"[retry_failed] {os.path.basename(anchor_dir)}: re-evaluating clips {reopened}")
    return reopened


def check_video(path, ref):
    """The file must decode and match the official fps / duration / resolution."""
    try:
        from decord import VideoReader, cpu

        vr = VideoReader(path, ctx=cpu(0))
        fps, n = vr.get_avg_fps(), len(vr)
        h, w = vr[0].shape[:2]
        del vr
    except Exception as e:
        return f"cannot decode: {e!r}"
    if ref is None:
        return None
    if abs(fps - ref.fps) > 0.05 or abs(n / fps - ref.duration) > 2.0 or f"{w}x{h}" != str(ref.resolution):
        return f"got {w}x{h} {fps:.2f}fps {n / fps:.0f}s, expected {ref.resolution} {ref.fps:.2f}fps {ref.duration:.0f}s"
    return None


def fetch(video_id, dest, ref, args, log):
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    for attempt in range(1, args.tries + 1):
        if args.copy_from:
            src = glob.glob(os.path.join(args.copy_from, "**", f"{video_id}.MP4"), recursive=True)
            if src:
                shutil.copy(src[0], dest)
        else:
            for url in url_candidates(video_id):
                for p in (dest, dest + ".aria2"):
                    if os.path.exists(p):
                        os.remove(p)
                cmd = ["aria2c", "-x", str(args.connections), "-s", str(args.connections), "-k", "5M", "-c",
                       "--check-certificate=false", "--connect-timeout=30", "--timeout=120", "--max-tries=3",
                       "--retry-wait=10", "--lowest-speed-limit=200K", "--user-agent=Mozilla/5.0", "--allow-overwrite=true",
                       "--console-log-level=error", "--summary-interval=0",
                       "-d", os.path.dirname(dest), "-o", os.path.basename(dest), url]
                if subprocess.run(cmd, capture_output=True).returncode == 0:
                    break
        err = check_video(dest, ref) if os.path.exists(dest) else "download failed"
        if err is None:
            return True
        log(f"[retry {attempt}/{args.tries}] {video_id}: {err}")
        if os.path.exists(dest):
            os.remove(dest)
        time.sleep(min(30 * attempt, 90))
    return False


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--work_dir", required=True, help="prepare_ek100.py --download_layout output")
    p.add_argument("--video_info", required=True, help="EPIC_100_video_info.csv (to verify every download)")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--vitl_ckpt", required=True)
    p.add_argument("--probe_ckpt", required=True)
    p.add_argument("--gpus", nargs="+", default=["0"])
    p.add_argument("--anchors", nargs="+", default=["official", "action_start"])
    p.add_argument("--dtype", default="fp16")
    p.add_argument("--device_prefix", default="cuda:", help="'cpu' for a CPU smoke test")
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--parallel_downloads", type=int, default=3)
    p.add_argument("--connections", type=int, default=8, help="aria2c connections per file")
    p.add_argument("--max_videos_on_disk", type=int, default=10)
    p.add_argument("--tries", type=int, default=3)
    p.add_argument("--max_hours", type=float, default=10.5, help="stop starting new downloads after this")
    p.add_argument("--copy_from", default=None, help="take videos from a local folder instead of downloading")
    p.add_argument("--videos_file", default=None, help="only these video_ids (one per line)")
    p.add_argument("--retry_failed", action="store_true",
                   help="re-download and re-evaluate clips that failed to decode (not the windows past the video end)")
    p.add_argument("--decode_fallback", choices=["none", "pyav"], default="none", help="see eval_ek100.py")
    p.add_argument("--eval_args", default="", help="extra args for eval_ek100.py")
    args = p.parse_args()

    t_start = time.time()
    lock = threading.Lock()
    log_path = os.path.join(args.out_dir, "stream.log")
    os.makedirs(args.out_dir, exist_ok=True)

    def log(msg):
        line = f"{time.strftime('%H:%M:%S')} {msg}"
        with lock:
            print(line, flush=True)
            with open(log_path, "a") as f:
                f.write(line + "\n")

    clips = pd.read_csv(os.path.join(args.work_dir, "val_clips.csv"))
    if args.videos_file:
        with open(args.videos_file) as f:
            clips = clips[clips.video_id.isin(f.read().split())]
    vi = pd.read_csv(args.video_info).set_index("video_id")
    videos = clips.sort_values("video_order").drop_duplicates("video_id")
    if args.retry_failed:
        for a in args.anchors:
            col = f"expected_fail_{a}"
            keep = set(clips.clip_id[clips[col]].tolist()) if col in clips else set()
            reopen_failed(os.path.join(args.out_dir, a), keep, log)
    done = {a: done_clip_ids(os.path.join(args.out_dir, a)) for a in args.anchors}
    todo_videos = [
        r for r in videos.itertuples()
        if any(not set(clips.clip_id[clips.video_id == r.video_id]) <= done[a] for a in args.anchors)
    ]
    video_root = os.path.commonpath([os.path.dirname(p) for p in videos.video_path])
    watch = os.path.join(video_root, "_stream_state")
    shutil.rmtree(watch, ignore_errors=True)
    os.makedirs(watch)
    log(f"{len(videos)} val videos, {len(todo_videos)} still to evaluate ({', '.join(args.anchors)})")
    if not todo_videos:
        return 0

    # one watch-mode eval process per GPU, restarted if its DataLoader dies (exit code 3)
    here = os.path.dirname(os.path.abspath(__file__))
    procs = []
    for i, g in enumerate(args.gpus):
        device = "cpu" if args.device_prefix == "cpu" else f"{args.device_prefix}{g}"
        run = (f"{sys.executable} {here}/eval_ek100.py --work_dir {args.work_dir} --out_dir {args.out_dir} "
               f"--vitl_ckpt {args.vitl_ckpt} --probe_ckpt {args.probe_ckpt} --device {device} "
               f"--shard_id {i} --num_shards {len(args.gpus)} --dtype {args.dtype} --num_workers {args.num_workers} "
               f"--anchors {' '.join(args.anchors)} --watch_dir {watch} --delete_after --decode_fallback {args.decode_fallback} {args.eval_args}"
               + (f" --only_videos {os.path.abspath(args.videos_file)}" if args.videos_file else ""))
        cmd = f"for t in 1 2 3 4 5; do {run} && break; echo '[eval restart '$t']'; done"
        log_f = open(os.path.join(args.out_dir, f"eval_gpu{g}.log"), "a")
        procs.append(subprocess.Popen(["bash", "-c", cmd], stdout=log_f, stderr=subprocess.STDOUT))

    stats = dict(ok=0, failed=[], bytes=0)
    in_flight = [0]
    last_report = [time.time()]

    def report_speed(every=int(os.environ.get("SPEED_EVERY", 600))):
        """Every `every` s, log each GPU's latest progress line (visible in the notebook while it runs)."""
        if time.time() - last_report[0] < every:
            return
        last_report[0] = time.time()
        for g in args.gpus:
            try:
                with open(os.path.join(args.out_dir, f"eval_gpu{g}.log")) as f:
                    lines = [l.strip() for l in f if "clip/s" in l or "Traceback" in l or "Error" in l]
            except OSError:
                lines = []
            log(f"[speed gpu{g}] {lines[-1] if lines else 'model loading / first batch...'}")

    def on_disk():
        return len(glob.glob(os.path.join(watch, "*.ready"))) + in_flight[0]

    def job(r):
        ref = vi.loc[r.video_id] if r.video_id in vi.index else None
        t0 = time.time()
        ok = fetch(r.video_id, r.video_path, ref, args, log)
        with lock:
            in_flight[0] -= 1
        if ok:
            size = os.path.getsize(r.video_path)
            with lock:
                stats["ok"] += 1
                stats["bytes"] += size
            open(os.path.join(watch, f"{r.video_id}.ready"), "w").close()
            log(f"[ok] {r.video_id} {size / 1e9:.2f} GB in {time.time() - t0:.0f}s "
                f"({stats['ok']}/{len(todo_videos)} fetched, {stats['bytes'] / 1e9:.0f} GB)")
        else:
            stats["failed"].append(r.video_id)
            open(os.path.join(watch, f"{r.video_id}.failed"), "w").close()
            log(f"[FAIL] {r.video_id}: gave up (its clips stay missing; rerun later)")

    stopped_early = False
    with ThreadPoolExecutor(args.parallel_downloads) as pool:
        for r in todo_videos:
            while on_disk() >= args.max_videos_on_disk:  # back-pressure: wait for the GPUs
                if all(pr.poll() is not None for pr in procs):
                    break
                report_speed()
                time.sleep(5)
            report_speed()
            if (time.time() - t_start) / 3600 > args.max_hours:
                stopped_early = True
                log(f"[stop] {args.max_hours}h budget reached: no new downloads, finishing what is on disk")
                break
            with lock:
                in_flight[0] += 1
            pool.submit(job, r)
    open(os.path.join(watch, "ALL_DOWNLOADED"), "w").close()

    while any(pr.poll() is None for pr in procs):
        time.sleep(30)
        n_done = len(glob.glob(os.path.join(watch, "*.done")))
        log(f"[wait] evaluated {n_done}/{stats['ok']} downloaded videos")
        report_speed()
    codes = [pr.returncode for pr in procs]
    n_done = len(glob.glob(os.path.join(watch, "*.done")))
    log(f"finished: {n_done} videos evaluated, {len(stats['failed'])} download failures {stats['failed']}, "
        f"eval exit codes {codes}, {(time.time() - t_start) / 3600:.2f} h"
        + (" -- run again to continue" if stopped_early or stats["failed"] else ""))
    return 0 if all(c == 0 for c in codes) else 1


if __name__ == "__main__":
    sys.exit(main())
