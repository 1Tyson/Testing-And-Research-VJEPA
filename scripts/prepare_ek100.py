"""Build the val clip index + class maps once (CPU only) and report dataset problems.

python scripts/prepare_ek100.py --video_root /kaggle/input/<ek100> \
    --train_csv EPIC_100_train.csv --val_csv EPIC_100_validation.csv --out_dir work/ek100
"""

import argparse
import json
import os
import sys
from collections import Counter

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from vjepa_ek100.protocol import build_val_clips, filter_annotations, index_videos  # noqa: E402

# Numbers for the official annotation release, used as a sanity check.
EXPECTED = dict(train_rows=67217, val_rows=9668)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--video_root", required=True)
    p.add_argument("--train_csv", required=True)
    p.add_argument("--val_csv", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--probe_videos", action="store_true", help="read fps/resolution of every val video")
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    tdf = pd.read_csv(args.train_csv)
    vdf = pd.read_csv(args.val_csv)
    verbs, nouns, actions, vdf_f = filter_annotations(tdf, vdf)
    videos = index_videos(args.video_root)
    clips = build_val_clips(vdf_f, verbs, nouns, actions, videos)

    val_videos = list(dict.fromkeys(vdf_f["video_id"].values))
    missing = [v for v in val_videos if v not in videos]
    report = dict(
        train_rows=len(tdf),
        val_rows=len(vdf),
        val_rows_after_filter=len(vdf_f),
        num_verbs=len(verbs),
        num_nouns=len(nouns),
        num_actions=len(actions),
        val_videos=len(val_videos),
        val_videos_found=len(val_videos) - len(missing),
        missing_val_videos=missing,
        val_clips=len(clips),
        videos_found_total=len(videos),
    )
    warnings = []
    if len(tdf) != EXPECTED["train_rows"]:
        warnings.append(f"train csv has {len(tdf)} rows, official has {EXPECTED['train_rows']}: class indices may not match the probe")
    if len(vdf) != EXPECTED["val_rows"]:
        warnings.append(f"val csv has {len(vdf)} rows, official has {EXPECTED['val_rows']}")
    if missing:
        warnings.append(f"{len(missing)} val videos missing: metric is over a subset")

    if args.probe_videos:
        from decord import VideoReader, cpu

        meta = {}
        for vid in clips["video_id"].unique():
            vr = VideoReader(videos[vid], ctx=cpu(0))
            h, w = vr[0].shape[:2]
            meta[vid] = dict(fps=round(vr.get_avg_fps(), 2), frames=len(vr), h=h, w=w)
        report["video_meta"] = meta
        fps = Counter(m["fps"] for m in meta.values())
        report["fps_histogram"] = {str(k): v for k, v in fps.items()}
        # frame ids in the csv refer to the original 50/60fps videos, and the clip stride
        # int(vfps / 8) changes with fps, so re-encoded videos change the protocol.
        if any(f not in (50.0, 59.94, 60.0) for f in fps):
            warnings.append(f"non-original fps found {dict(fps)}: annotation frame ids may not line up")
        short = [v for v, m in meta.items() if m["frames"] < clips[clips.video_id == v].stop_frame.max()]
        if short:
            warnings.append(f"{len(short)} videos shorter than their annotations (re-encoded?): {short[:5]}")
    report["warnings"] = warnings

    clips.to_csv(os.path.join(args.out_dir, "val_clips.csv"), index=False)
    with open(os.path.join(args.out_dir, "classes.json"), "w") as f:
        json.dump(
            dict(
                verbs=[[int(k), v] for k, v in verbs.items()],
                nouns=[[int(k), v] for k, v in nouns.items()],
                actions=[[int(k[0]), int(k[1]), v] for k, v in actions.items()],
            ),
            f,
        )
    with open(os.path.join(args.out_dir, "dataset_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps({k: v for k, v in report.items() if k != "video_meta"}, indent=2))


if __name__ == "__main__":
    main()
