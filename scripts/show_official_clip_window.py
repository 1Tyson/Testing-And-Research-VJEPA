"""Run Meta's OWN clip-sampling code (vjepa2 evals/action_anticipation_frozen/epickitchens.py,
decode_videos_to_clips.run) and print where every clip lies relative to its action. No GPU, no video needed:
decord.VideoReader is replaced by a stub that only records which frame indices the official code asks for.

    python scripts/show_official_clip_window.py --vjepa2 vjepa2 --annotations epic-kitchens-100-annotations \
        --video P01_11 --n 8

Two settings of the same official function are run:
  * the released eval config: anticipation_time = 1 s, anticipation_point = 0 (eval.py default
    val_anticipation_point = [0.0, 0.0])  ->  clip ends at stop_frame - 1 s
  * anticipation_point = 1, i.e. the EK100 definition                          ->  clip ends at start_frame - 1 s
"""

import argparse
import os
import sys
import types

import numpy as np
import pandas as pd


def stub_modules():
    """Only what epickitchens.py needs at import time, for machines without torch / webdataset / decord."""
    if "torch" not in sys.modules:
        try:
            import torch  # noqa: F401
        except ImportError:
            torch = types.ModuleType("torch")
            ud = types.ModuleType("torch.utils.data")
            ud.DataLoader = ud.IterableDataset = object
            udd = types.ModuleType("torch.utils.data.distributed")
            udd.DistributedSampler = object
            torch.utils = types.ModuleType("torch.utils")
            sys.modules.update({"torch": torch, "torch.utils": torch.utils, "torch.utils.data": ud,
                                "torch.utils.data.distributed": udd})
    try:
        import webdataset  # noqa: F401
    except ImportError:
        wds = types.ModuleType("webdataset")
        wds.PipelineStage = object
        sys.modules["webdataset"] = wds
    wif = types.ModuleType("src.datasets.utils.worker_init_fn")
    wif.pl_worker_init_function = None
    sys.modules.setdefault("src.datasets.utils.worker_init_fn", wif)


class RecordingVideoReader:
    """Stands in for decord.VideoReader: reports the real fps / length, records the requested indices."""

    fps = {}
    n_frames = {}
    calls = []

    def __init__(self, path, **kw):
        self.vid = os.path.basename(path).split(".")[0]

    def seek(self, i):
        pass

    def get_avg_fps(self):
        return self.fps[self.vid]

    def __len__(self):
        return self.n_frames[self.vid]

    def get_batch(self, indices):
        RecordingVideoReader.calls.append((self.vid, np.asarray(indices).copy()))
        n = len(indices)
        return types.SimpleNamespace(asnumpy=lambda: np.zeros((n, 1, 1, 3), np.uint8))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--vjepa2", required=True, help="clone of facebookresearch/vjepa2 (commit 204698b)")
    p.add_argument("--annotations", required=True, help="clone of epic-kitchens-100-annotations")
    p.add_argument("--video", default="P01_11")
    p.add_argument("--n", type=int, default=8, help="segments to print")
    args = p.parse_args()

    stub_modules()
    decord = types.ModuleType("decord")
    decord.VideoReader, decord.cpu = RecordingVideoReader, (lambda i=0: None)
    sys.modules["decord"] = decord
    sys.path.insert(0, args.vjepa2)
    from evals.action_anticipation_frozen.epickitchens import decode_videos_to_clips  # Meta's code

    vdf = pd.read_csv(os.path.join(args.annotations, "EPIC_100_validation.csv"))
    vi = pd.read_csv(os.path.join(args.annotations, "EPIC_100_video_info.csv")).set_index("video_id")
    ano = vdf[vdf.video_id == args.video].sort_values(by="start_frame")  # same as build_annotations
    fps = float(vi.loc[args.video, "fps"])
    RecordingVideoReader.fps[args.video] = fps
    RecordingVideoReader.n_frames[args.video] = int(vi.loc[args.video, "duration"] * fps)

    results = {}
    for name, point in (("Meta (config released)", [0.0, 0.0]), ("EK100 definition", [1.0, 1.0])):
        RecordingVideoReader.calls = []
        dec = decode_videos_to_clips(annotations={args.video: ano}, frames_per_clip=32, fps=8,
                                     anticipation_time_sec=[1.0, 1.0], anticipation_point=point)
        n_yield = sum(1 for _ in dec.run([f"/x/{args.video}.MP4"]))
        results[name] = (n_yield, list(RecordingVideoReader.calls))

    print(f"video {args.video}: {len(ano)} labelled actions, {fps:.2f} fps; official code: frames_per_clip=32, fps=8,"
          f" step int({fps:.2f}/8) = {int(fps / 8)} frames")
    for name, (n_yield, calls) in results.items():
        print(f"{name}: {n_yield} clips yielded = one per labelled action, each decoded independently")
    print()
    hdr = (f"{'action':32s} {'start':>6s} {'stop':>6s} | {'Meta clip (s)':>15s} {'action seen':>11s} |"
           f" {'EK100 clip (s)':>15s} {'action seen':>11s}")
    print(hdr)
    print("-" * len(hdr))
    meta_calls, ek_calls = results["Meta (config released)"][1], results["EK100 definition"][1]
    for k, r in enumerate(ano.itertuples()):
        if k >= args.n:
            break
        s, e = r.start_frame / fps, r.stop_frame / fps
        row = f"{r.narration[:32]:32s} {s:6.2f} {e:6.2f} |"
        for _, idx in (meta_calls[k], ek_calls[k]):
            a, b = idx.min() / fps, (idx.max() + 1) / fps  # first to last sampled frame
            seen = max(0.0, min(b, e) - max(a, s))
            row += f" {a:6.2f} - {b:6.2f} {seen:9.2f} s |"
        print(row)

    # whole val set, from the same official formula (no video needed)
    v = vdf.copy()
    v["fps"] = v.video_id.map(vi.fps)
    step = (v.fps / 8).astype(int)
    end_meta = (v.stop_frame - (1.0 * v.fps).astype(int)) / v.fps
    start_meta = end_meta - 32 * step / v.fps
    seen = (np.minimum(end_meta, v.stop_frame / v.fps) - np.maximum(start_meta, v.start_frame / v.fps)).clip(lower=0)
    print(f"\nAll {len(v)} val actions, Meta setting: clip spans {float((32 * step / v.fps).median()):.2f} s;"
          f" contains part of the action it must predict for {100 * (seen > 0).mean():.1f}% of them"
          f" (median {seen.median():.2f} s of it). EK100 setting: 0% by construction.")


if __name__ == "__main__":
    main()
