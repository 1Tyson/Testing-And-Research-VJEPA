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
from vjepa_ek100.protocol import build_val_clips, clip_frame_indices, filter_annotations, index_videos  # noqa: E402

# Numbers for the official annotation release, used as a sanity check.
EXPECTED = dict(train_rows=67217, val_rows=9668)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--video_root", required=True)
    p.add_argument("--train_csv", required=True)
    p.add_argument("--val_csv", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--probe_videos", action="store_true", help="read fps/resolution of every val video")
    p.add_argument("--video_info", default=None, help="EPIC_100_video_info.csv, to detect truncated/re-encoded files")
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
            meta[vid] = dict(fps=vr.get_avg_fps(), frames=len(vr), h=h, w=w)
        report["video_meta"] = meta

        if args.video_info:
            vi = pd.read_csv(args.video_info).set_index("video_id")
            mismatch = {}
            for vid, m in meta.items():
                ref_fps, ref_dur = vi.loc[vid, "fps"], vi.loc[vid, "duration"]
                dur = m["frames"] / m["fps"]
                if abs(m["fps"] - ref_fps) > 0.05 or abs(dur - ref_dur) > 1.0:
                    mismatch[vid] = dict(fps=round(m["fps"], 2), ref_fps=round(ref_fps, 2), duration=round(dur, 1),
                                         ref_duration=round(ref_dur, 1))
            report["mismatch_vs_official_video_info"] = mismatch
            if mismatch:
                warnings.append(f"{len(mismatch)} videos differ from EPIC_100_video_info.csv (truncated/re-encoded?)")
        # EK100 frame ids are at ~60fps for every video, but the official loader indexes the raw
        # video with them, so on 30/48/90fps videos the clip lands at the wrong time (or past the end).
        report["videos_where_frame_ids_are_not_native"] = {
            vid: round(m["fps"], 2) for vid, m in meta.items() if abs(m["fps"] - 59.94) > 0.1
        }
        fps = Counter(round(m["fps"], 2) for m in meta.values())
        report["fps_histogram"] = {str(k): v for k, v in fps.items()}

        # Clips whose frame window runs past the end of the video: decord raises on them, and
        # the official loader silently drops them. Checked per clip with the real fps.
        for anchor in ("official", "action_start"):
            clips[f"expected_fail_{anchor}"] = [
                int(clip_frame_indices(r.start_frame, r.stop_frame, meta[r.video_id]["fps"], anchor=anchor).max())
                >= meta[r.video_id]["frames"]
                for r in clips.itertuples()
            ]
        bad = clips[clips.expected_fail_official]
        report["out_of_range_clips_official"] = int(len(bad))
        report["out_of_range_by_video"] = {
            vid: dict(clips=int(len(g)), of=int((clips.video_id == vid).sum()), fps=round(meta[vid]["fps"], 2),
                      frames=meta[vid]["frames"], max_stop_frame=int(clips[clips.video_id == vid].stop_frame.max()))
            for vid, g in bad.groupby("video_id")
        }
        if len(bad):
            warnings.append(
                f"{len(bad)} clips in {bad.video_id.nunique()} videos run past the end of the video; the official "
                f"loader drops them too (see out_of_range_by_video)"
            )
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
