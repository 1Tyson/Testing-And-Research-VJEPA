"""Index of TRAIN clips (with per-clip anticipation time / point) for extracting frozen features (CPU only).

Used by the classifier-head study (docs/DIRECTION2.md): eval_ek100.py --clips_csv train_clips.csv --save_pooled
runs the frozen V-JEPA 2 encoder + predictor + the released probe's attentive pooler on these clips and stores the
pooler output (3 x 1024 per clip: the inputs of the verb / noun / action linear layers) and the released logits.

python scripts/prepare_train_clips.py --video_root /kaggle/input --train_csv EPIC_100_train.csv \
    --val_csv EPIC_100_validation.csv --video_info EPIC_100_video_info.csv --out_dir work --views 1

Only videos found under --video_root are listed; clip ids do not depend on which videos are present, so the
index can be rebuilt when more re-encoded videos become available and the already extracted clips are kept.
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from vjepa_ek100.protocol import (build_train_clips, clip_frame_indices, filter_annotations,  # noqa: E402
                                  index_videos, native_frame_id)


def classes_json(verbs, nouns, actions):
    return dict(verbs=[[int(k), v] for k, v in verbs.items()], nouns=[[int(k), v] for k, v in nouns.items()],
                actions=[[int(k[0]), int(k[1]), v] for k, v in actions.items()])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--video_root", required=True)
    p.add_argument("--train_csv", required=True)
    p.add_argument("--val_csv", required=True, help="only to build the same class maps as prepare_ek100.py")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--video_info", default=None, help="EPIC_100_video_info.csv: flag windows past the video end")
    p.add_argument("--views", type=int, default=1, help="view 0 = val setting (1 s, point 0); 1.. = random")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--fix_frame_ids", action="store_true", help="same option as eval_ek100.py (for the end check)")
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    tdf = pd.read_csv(args.train_csv)
    verbs, nouns, actions, _ = filter_annotations(tdf, pd.read_csv(args.val_csv))
    cj = classes_json(verbs, nouns, actions)
    path = os.path.join(args.out_dir, "classes.json")
    if os.path.exists(path):  # written by prepare_ek100.py: must be the very same mapping
        with open(path) as f:
            if json.load(f) != cj:
                raise SystemExit("classes.json differs from the one of prepare_ek100.py: use the same train csv")
    else:
        with open(path, "w") as f:
            json.dump(cj, f)

    train_ids = set(tdf.video_id)
    videos = {k: v for k, v in index_videos(args.video_root).items() if k in train_ids}
    clips = build_train_clips(tdf, verbs, nouns, actions, videos, views=args.views, seed=args.seed)
    report = dict(train_videos=len(train_ids), train_videos_found=len(videos),
                  segments_total=len(tdf), segments_found=int(clips.seg_id.nunique()), clips=len(clips),
                  views=args.views, participants=int(clips.participant.nunique()) if len(clips) else 0)

    if args.video_info and len(clips):
        vi = pd.read_csv(args.video_info).set_index("video_id")
        for anchor in ("official", "action_start"):
            bad = []
            for r in clips.itertuples():
                fps = float(vi.loc[r.video_id, "fps"])
                n_frames = int(vi.loc[r.video_id, "duration"] * fps)
                idx = clip_frame_indices(r.start_frame, r.stop_frame, fps, anticipation_time=r.anticipation_time,
                                         anticipation_point=r.anticipation_point, anchor=anchor,
                                         fix_frame_ids=args.fix_frame_ids)
                bad.append(int(idx.max()) >= n_frames)
            clips[f"expected_fail_{anchor}"] = bad
            report[f"out_of_range_clips_{anchor}"] = int(np.sum(bad))
        # frame ids are ~60 fps on every EK55 video; only a few train videos have another native rate
        report["videos_where_frame_ids_are_not_native"] = {
            v: float(vi.loc[v, "fps"]) for v in videos if native_frame_id(600, float(vi.loc[v, "fps"])) != 600}

    clips.to_csv(os.path.join(args.out_dir, "train_clips.csv"), index=False)
    with open(os.path.join(args.out_dir, "train_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))
    if not len(clips):
        raise SystemExit(f"no train video found under {args.video_root}: attach the re-encoded train datasets")


if __name__ == "__main__":
    main()
