"""EK100 action-anticipation protocol, mirroring facebookresearch/vjepa2.

Reference: evals/action_anticipation_frozen/epickitchens.py (commit 204698b).
Everything here is pure python/numpy/pandas so it can be unit-tested without a GPU.
"""

import os
from collections import Counter

import numpy as np
import pandas as pd

VIDEO_EXTS = (".MP4", ".mp4")


def index_videos(video_root):
    """Map video_id (e.g. 'P01_11') -> absolute path, searching recursively.

    The official code expects either $root/P01/videos/P01_11.MP4 or $root/P01/P01_11.MP4;
    Kaggle mirrors use various layouts, so we look everywhere.
    """
    found = {}
    for dirpath, _, files in os.walk(os.path.abspath(video_root), followlinks=True):
        for f in files:
            stem, ext = os.path.splitext(f)
            if ext in VIDEO_EXTS and stem not in found:
                found[stem] = os.path.join(dirpath, f)
    return found


def filter_annotations(tdf, vdf):
    """Same class-index construction as the official `filter_annotations`.

    The released probe's output layers are indexed by these mappings, which come from
    enumerating python sets. The order is deterministic for integer keys, but only if
    the *full official* EPIC_100_train.csv is used.
    """
    tactions = set([(v, n) for v, n in zip(tdf["verb_class"].values, tdf["noun_class"].values)])
    tverbs = set([v for v, _ in tactions])
    tnouns = set([n for _, n in tactions])
    keep_inds = [(v, n) in tactions for v, n in zip(vdf["verb_class"].values, vdf["noun_class"].values)]
    vdf = vdf[keep_inds]

    verb_classes = {k: i for i, k in enumerate(tverbs)}
    noun_classes = {k: i for i, k in enumerate(tnouns)}
    action_classes = {k: i for i, k in enumerate(tactions)}
    return verb_classes, noun_classes, action_classes, vdf


def build_val_clips(vdf, verb_classes, noun_classes, action_classes, video_paths):
    """One row per val clip, ordered exactly like the official loader iterates them.

    Official order: videos in first-appearance order of the (filtered) csv, clips within a
    video sorted by `start_frame` (pandas default sort). Videos without a file are dropped.
    """
    rows = []
    unique_videos = list(dict.fromkeys(vdf["video_id"].values))
    video_order = 0
    for uv in unique_videos:
        if uv not in video_paths:
            continue
        ano = vdf[vdf["video_id"] == uv].sort_values(by="start_frame")
        for j, r in enumerate(ano.itertuples(index=False)):
            v, n = int(r.verb_class), int(r.noun_class)
            rows.append(
                dict(
                    narration_id=getattr(r, "narration_id", f"{uv}_{j}"),
                    video_id=uv,
                    video_path=video_paths[uv],
                    video_order=video_order,
                    clip_order=j,
                    start_frame=int(r.start_frame),
                    stop_frame=int(r.stop_frame),
                    verb_class=v,
                    noun_class=n,
                    verb_idx=verb_classes[v],
                    noun_idx=noun_classes[n],
                    action_idx=action_classes[(v, n)],
                )
            )
        video_order += 1
    df = pd.DataFrame(rows)
    df.insert(0, "clip_id", np.arange(len(df)))
    return df


def clip_frame_indices(
    start_frame,
    stop_frame,
    vfps,
    frames_per_clip=32,
    fps=8,
    anticipation_time=1.0,
    anticipation_point=0.0,
    anchor="official",
):
    """Frame indices of the observed clip.

    anchor="official": replicates the vjepa2 code,
        af = int(sf * ap + (1 - ap) * ef - at * vfps)
      With the val setting ap=0 this is `stop_frame - 1s`, i.e. the observed window can
      overlap the action itself for actions longer than 1s.
    anchor="action_start": the standard EK100 anticipation definition (observation ends
      tau_a = `anticipation_time` seconds before the action starts), af = sf - at * vfps.
    """
    fstp = int(vfps / fps)
    nframes = int(frames_per_clip * fstp)
    aframes = int(anticipation_time * vfps)
    if anchor == "official":
        af = int(start_frame * anticipation_point + (1 - anticipation_point) * stop_frame - aframes)
    elif anchor == "action_start":
        af = int(start_frame - aframes)
    else:
        raise ValueError(f"unknown anchor {anchor}")
    indices = np.arange(af - nframes, af, fstp).astype(np.int64)
    indices[indices < 0] = 0
    return indices


def official_eval_multiplicity(clips, world_size=64, batch_size=2, num_workers=2):
    """How many times each clip is counted by the official distributed val loop.

    The official `validate` runs exactly `ipe = num_clips // (world_size * batch_size)`
    batches on every rank, restarting the rank's iterator when it runs out. Ranks own
    `videos[rank::world_size]`, which hold very different clip counts, so some clips are
    counted several times and others never. This reproduces that behaviour so the reported
    32.7 can be compared like-for-like. See tests/test_protocol.py for the check against
    the real webdataset/DataLoader pipeline.

    :param clips: DataFrame from `build_val_clips` (needs clip_id, video_order, clip_order)
    :returns: Counter clip_id -> count
    """
    clips = clips.sort_values(["video_order", "clip_order"])
    per_video = [g["clip_id"].tolist() for _, g in clips.groupby("video_order", sort=True)]
    num_clips = sum(len(v) for v in per_video)
    ipe = num_clips // (world_size * batch_size)

    counts = Counter()
    for rank in range(world_size):
        rank_videos = per_video[rank::world_size]
        # wds.split_by_worker, then wds.batched(partial=True) inside each worker
        worker_batches = []
        for w in range(max(num_workers, 1)):
            stream = [c for vid in rank_videos[w :: max(num_workers, 1)] for c in vid]
            worker_batches.append([stream[i : i + batch_size] for i in range(0, len(stream), batch_size)])
        # DataLoader interleaves workers round-robin, skipping exhausted ones
        epoch = []
        i = 0
        while any(i < len(b) for b in worker_batches):
            for b in worker_batches:
                if i < len(b):
                    epoch.append(b[i])
            i += 1
        if not epoch:
            continue  # a rank with no data would hang in the official loop; ignore
        taken = 0
        while taken < ipe:
            for batch in epoch:
                if taken >= ipe:
                    break
                counts.update(batch)
                taken += 1
    return counts
