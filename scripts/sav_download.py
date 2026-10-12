"""Download SAV (github.com/Ritatanz/SAV): annotations from Google Drive, source videos from their web links,
then cut the 3 s clips listed in clips_to_keep.txt exactly as the authors' cut_videos.sh does.

  python scripts/sav_download.py annotations --root /kaggle/working/sav
  python scripts/sav_download.py videos --root /kaggle/working/sav --clips_dir /kaggle/working/sav/video_clips

Full videos are deleted after cutting (unless --keep_videos), so disk use stays small.
`videos` resumes: clips already cut are skipped, failures are written to <root>/download_failures.csv.
"""

import argparse
import concurrent.futures as cf
import csv
import os
import re
import shutil
import subprocess
import sys
import time
import zipfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from classroom.sav import read_annotations  # noqa: E402

# Google Drive ids from the SAV README.
DRIVE = {
    "video_links": ("1S4EONu-faoUxy8y_dIYIqM5LOfyTlGNP", None),
    "clips_to_keep": ("1Z3x-farXpzLt7Q3nGSpq5sH_DzVCq-TD", "clips_to_keep.txt"),
    "frame_list_train": ("1UHlhz6p7-UMy82sy5DBdcrCCBW483fp8", "frame_list/train.csv"),
    "frame_list_val": ("1fx7adqC6MiKdQdB3tlGVZkLEpYokUqOS", "frame_list/val.csv"),
    "boxes_train": ("1QI169QjUKZN0PcMzSnTLTQqL7VOZwNOX", "annotations/class_detection_train_boxes_and_labels.csv"),
    "boxes_val": ("1tf2Fyu1Kl_k55kxsHeLWOlYiz3DLK0MQ", "annotations/class_detection_val_boxes_and_labels.csv"),
    "person_det_train": ("1EE0q6baN8QiMvKVbSdlHTf_KZcqGHhD7", "annotations/train_person_det.pkl"),
    "person_det_val": ("1T-y1z1bZz8421gdFR6x81WtvxQy9bHRg", "annotations/val_person_det.pkl"),
    "ann_train": ("1rHToVI9pmAqzFdrNJ-o2P_9pzLzYIcLh", "annotations/train.csv"),
    "ann_val": ("1lfWBtmJ98cd0xWNqyq0Ax888ZmdZ0p4L", "annotations/val.csv"),
    "excluded_train": ("1f4OGziejZfpckjNzrzrAeFxJbt0zMEa-", "annotations/train_excluded_timestamps.csv"),
    "excluded_val": ("1JRLGNTgEKT75ItPO4SBaoLkl7VT6I0Iq", "annotations/val_excluded_timestamps.csv"),
    "labelmap": ("1bRS5ia_9UUlBTRNuGvc8jBFv3lHLSseg", "annotations/education_first_label.pbtxt"),
}

# Removed from video file names before matching clips_to_keep.txt (cut_videos.sh).
NAME_SUFFIXES = ["课堂教学视频", "课堂优课实录", "课堂视频实录"]
URL_RE = re.compile(r"https?://\S+")
VIDEO_EXT = (".mp4", ".flv", ".mkv", ".webm", ".mov", ".avi", ".ts", ".m4v")


def clean_name(name):
    for s in NAME_SUFFIXES:
        name = name.replace(s, "")
    return name.strip()


def drive_download(file_id, out_path):
    import gdown

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    got = gdown.download(id=file_id, output=out_path, quiet=True)
    if not got or not os.path.exists(got):
        raise RuntimeError(f"Google Drive download failed for {file_id}")
    return got


def maybe_unzip(path, dest):
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as z:
            z.extractall(dest)
        print(f"  unzipped {os.path.basename(path)} -> {dest}")


def cmd_annotations(args):
    root = args.root
    for name, (fid, rel) in DRIVE.items():
        if rel is None:  # keep Drive's own file name
            out = os.path.join(root, "links") + os.sep
        else:
            out = os.path.join(root, rel)
            if os.path.exists(out) and os.path.getsize(out) > 0:
                print(f"have  {rel}")
                continue
        try:
            got = drive_download(fid, out)
            print(f"got   {os.path.relpath(got, root)} ({os.path.getsize(got) / 1e6:.2f} MB)")
            maybe_unzip(got, os.path.dirname(got))
        except Exception as e:  # keep going, report at the end
            print(f"FAIL  {name}: {e}")
    links = find_links_file(root)
    print(f"\nvideo link file: {links}")
    if links:
        with open(links, encoding="utf-8", errors="replace") as f:
            head = [next(f, "") for _ in range(8)]
        print("first lines:\n" + "".join(head))


def find_links_file(root):
    d = os.path.join(root, "links")
    if not os.path.isdir(d):
        return None
    files = [os.path.join(dp, f) for dp, _, fs in os.walk(d) for f in fs if f.lower().endswith((".txt", ".csv"))]
    return max(files, key=os.path.getsize) if files else None


def parse_links(path):
    """[(name or None, url)]. A line holds one URL; any other text on it is taken as the video name."""
    out = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            m = URL_RE.search(line)
            if not m:
                continue
            url = m.group(0).rstrip(",;\"'")
            rest = (line[: m.start()] + " " + line[m.end():]).strip(" \t\r\n,;:|\"'")
            name = os.path.splitext(rest)[0].strip() if rest else None
            out.append((name or None, url))
    return out


def read_segments(root):
    with open(os.path.join(root, "clips_to_keep.txt"), encoding="utf-8") as f:
        segs = [ln.strip() for ln in f if ln.strip()]
    by_video = {}
    for s in segs:
        parts = s.split("_")
        by_video.setdefault("_".join(parts[:2]), []).append((s, int(parts[2])))
    return by_video


def cut_clips(video_path, segments, clips_dir):
    """cut_videos.sh: start = (index - 1) * 2 + 1.5 s, 3 s long, x264."""
    made = 0
    for seg, index in segments:
        out = os.path.join(clips_dir, f"{seg}.mp4")
        if os.path.exists(out) and os.path.getsize(out) > 0:
            made += 1
            continue
        start = max(0.0, (index - 1) * 2 + 1.5)
        tmp = out + ".part.mp4"
        r = subprocess.run(
            ["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", video_path, "-ss", f"{start:.2f}", "-t", "3",
             "-c:v", "libx264", "-c:a", "aac", tmp],
            capture_output=True, text=True,
        )
        if r.returncode == 0 and os.path.exists(tmp) and os.path.getsize(tmp) > 0:
            os.replace(tmp, out)
            made += 1
        elif os.path.exists(tmp):
            os.remove(tmp)
    return made


def fetch(url, out_stem, timeout):
    """Download with yt-dlp (handles most video sites); direct files fall back to curl."""
    os.makedirs(os.path.dirname(out_stem), exist_ok=True)
    cmd = ["yt-dlp", "--no-playlist", "--quiet", "--no-warnings", "--retries", "3", "-f", "bv*+ba/b",
           "--merge-output-format", "mp4", "-o", out_stem + ".%(ext)s", url]
    try:
        subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        pass
    got = [os.path.join(os.path.dirname(out_stem), f) for f in os.listdir(os.path.dirname(out_stem))
           if f.startswith(os.path.basename(out_stem) + ".") and f.lower().endswith(VIDEO_EXT)]
    if not got and url.lower().split("?")[0].endswith(VIDEO_EXT):
        dst = out_stem + ".mp4"
        try:
            subprocess.run(["curl", "-sSL", "--retry", "3", "-o", dst, url], capture_output=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            pass
        if os.path.exists(dst) and os.path.getsize(dst) > 100_000:
            got = [dst]
    return got[0] if got else None


def cmd_videos(args):
    root = args.root
    clips_dir = args.clips_dir or os.path.join(root, "video_clips")
    tmp_dir = os.path.join(root, "videos")
    os.makedirs(clips_dir, exist_ok=True)
    by_video = read_segments(root)

    # Only clips that carry annotations are worth fetching.
    wanted = set()
    for split in ("train", "val"):
        p = os.path.join(root, "annotations", f"{split}.csv")
        if os.path.exists(p):
            wanted |= {k.rsplit(",", 1)[0] for k in read_annotations(p)}
    if wanted:
        by_video = {v: [s for s in segs if s[0] in wanted] for v, segs in by_video.items()}
        by_video = {v: segs for v, segs in by_video.items() if segs}
    n_clips = sum(len(s) for s in by_video.values())
    print(f"{len(by_video)} source videos, {n_clips} annotated clips")

    links = parse_links(args.links or find_links_file(root))
    print(f"{len(links)} links in the link file")
    jobs = []
    for i, (name, url) in enumerate(links):
        key = clean_name(name) if name else None
        if key is not None and key not in by_video:
            # The link name may carry extra text; match on the clip prefix.
            hits = [v for v in by_video if v in key]
            key = hits[0] if len(hits) == 1 else None
        if key is not None:
            todo = [s for s in by_video[key] if not os.path.exists(os.path.join(clips_dir, f"{s[0]}.mp4"))]
            if not todo:
                continue
        jobs.append((i, key, url))
    if args.limit:
        jobs = jobs[: args.limit]
    print(f"{len(jobs)} videos to fetch")

    deadline = time.time() + args.max_hours * 3600
    fail_path = os.path.join(root, "download_failures.csv")

    def run(job):
        i, key, url = job
        if time.time() > deadline:
            return job, "skipped (time limit)", 0
        stem = os.path.join(tmp_dir, f"v{i:05d}")
        path = fetch(url, stem, args.timeout)
        if path is None:
            return job, "download failed", 0
        k = key
        if k is None:  # name unknown from the link file: use the title yt-dlp saved
            title = subprocess.run(["yt-dlp", "--get-title", "--no-playlist", url], capture_output=True, text=True).stdout
            title = clean_name(title.strip())
            hits = [v for v in by_video if v == title or v in title]
            k = hits[0] if len(hits) == 1 else None
        if k is None:
            if not args.keep_videos:
                os.remove(path)
            return job, "no matching clips", 0
        made = cut_clips(path, by_video[k], clips_dir)
        if not args.keep_videos:
            os.remove(path)
        return job, "ok" if made == len(by_video[k]) else f"cut {made}/{len(by_video[k])}", made

    done = failed = 0
    with cf.ThreadPoolExecutor(args.workers) as ex, open(fail_path, "a", newline="", encoding="utf-8") as ff:
        w = csv.writer(ff)
        for n, (job, status, made) in enumerate(ex.map(run, jobs), 1):
            done += made
            if status != "ok":
                failed += 1
                w.writerow([job[0], job[1] or "", job[2], status])
                ff.flush()
            if n % 20 == 0 or n == len(jobs):
                have = len([f for f in os.listdir(clips_dir) if f.endswith(".mp4")])
                print(f"[{n}/{len(jobs)}] clips on disk {have}/{n_clips}, failed videos {failed}", flush=True)
    shutil.rmtree(tmp_dir, ignore_errors=True)
    have = len([f for f in os.listdir(clips_dir) if f.endswith(".mp4")])
    print(f"done: {have}/{n_clips} annotated clips on disk ({100 * have / max(1, n_clips):.1f}%). Failures: {fail_path}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("annotations")
    a.add_argument("--root", required=True)
    v = sub.add_parser("videos")
    v.add_argument("--root", required=True)
    v.add_argument("--clips_dir", default=None)
    v.add_argument("--links", default=None, help="link file (default: the one fetched by `annotations`)")
    v.add_argument("--workers", type=int, default=4)
    v.add_argument("--timeout", type=int, default=900, help="seconds per video download")
    v.add_argument("--max_hours", type=float, default=8.0)
    v.add_argument("--limit", type=int, default=0, help="only the first N videos (for a quick test)")
    v.add_argument("--keep_videos", action="store_true")
    args = ap.parse_args()
    {"annotations": cmd_annotations, "videos": cmd_videos}[args.cmd](args)


if __name__ == "__main__":
    main()
