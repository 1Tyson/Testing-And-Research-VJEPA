"""Check that decord returns the frames we ask for, against PyAV with timestamp-accurate seeking.

Decodes N random val clips both ways (32 frames each, same frame indices as the eval) and reports
the mean absolute pixel difference per clip. ~0 means decord is exact; large values mean decord
returned other frames than requested (seek inaccuracy), which would corrupt the eval.

pip install av
python scripts/check_decoding.py --work_dir work --num_clips 40 --out results/check_decoding.json
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from vjepa_ek100.data import pyav_frames  # noqa: E402
from vjepa_ek100.protocol import clip_frame_indices  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--work_dir", required=True)
    p.add_argument("--num_clips", type=int, default=40)
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    from decord import VideoReader, cpu

    clips = pd.read_csv(os.path.join(args.work_dir, "val_clips.csv"))
    if "expected_fail_official" in clips:
        clips = clips[~clips.expected_fail_official]
    sample = clips.sample(n=min(args.num_clips, len(clips)), random_state=args.seed)
    rows = []
    for r in sample.itertuples():
        vr = VideoReader(r.video_path, num_threads=-1, ctx=cpu(0))
        fps = vr.get_avg_fps()
        idx = clip_frame_indices(r.start_frame, r.stop_frame, fps)
        a = vr.get_batch(idx).asnumpy().astype(np.int16)
        try:
            b = pyav_frames(r.video_path, idx, fps).astype(np.int16)
            diff = np.abs(a - b).mean(axis=(1, 2, 3))
            rows.append(dict(clip_id=int(r.clip_id), video_id=r.video_id, fps=round(fps, 2), first=int(idx[0]),
                             mean_abs_diff=float(diff.mean()), max_frame_diff=float(diff.max()),
                             frames_off=int((diff > 2.0).sum())))
        except Exception as e:
            rows.append(dict(clip_id=int(r.clip_id), video_id=r.video_id, error=repr(e)))
        print(rows[-1], flush=True)
        del vr
    ok = [x for x in rows if "error" not in x]
    summary = dict(
        clips=len(rows),
        compared=len(ok),
        clips_with_any_frame_off=sum(x["frames_off"] > 0 for x in ok),
        mean_abs_diff=float(np.mean([x["mean_abs_diff"] for x in ok])) if ok else None,
        rows=rows,
    )
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)
    print({k: v for k, v in summary.items() if k != "rows"})


if __name__ == "__main__":
    main()
