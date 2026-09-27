"""Diagnostics for the gap between our EK100 number and the paper's 32.7 (CPU only, uses saved logits).

  1. how much the "official" number moves with the eval world size / workers (the official loop
     counts clips unevenly, so the number depends on the cluster layout)
  2. bootstrap noise of mean-class recall (thousands of action classes with 1-2 val clips)
  3. chance level with shuffled labels (sanity check of the metric pipeline)
  4. official scoring numerics: sigmoid in bf16 + torch.topk tie-breaking
  5. effect of the 5 videos whose ~60fps frame ids do not match their native fps
  6. optional: agreement with a second logits run (e.g. an fp32 subset) on shared clips

python scripts/diagnose_ek100.py --work_dir work --logits_dir logits/official --out results/diagnose.json
"""

import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))
from compute_metrics import load_logits  # noqa: E402
from vjepa_ek100.metrics import mean_class_recall, topk_hits  # noqa: E402
from vjepa_ek100.protocol import official_eval_multiplicity  # noqa: E402

NON_NATIVE_FPS_VIDEOS = ["P09_07", "P09_08", "P17_02", "P18_02", "P18_09"]


def official_style_hits(logits, labels, k=5):
    """Top-k exactly as ClassMeanRecall does it under bf16 autocast: sigmoid in bf16, torch.topk."""
    probs = torch.sigmoid(torch.from_numpy(logits.astype(np.float32)).to(torch.bfloat16))
    top = probs.topk(k, dim=1).indices.numpy()
    return (top == labels[:, None]).any(axis=1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--work_dir", required=True)
    p.add_argument("--logits_dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--compare_logits_dir", default=None, help="second run on (a subset of) the same clips")
    p.add_argument("--bootstrap", type=int, default=200)
    args = p.parse_args()

    clips = pd.read_csv(os.path.join(args.work_dir, "val_clips.csv"))
    with open(os.path.join(args.work_dir, "classes.json")) as f:
        n_act = len(json.load(f)["actions"])
    lg = load_logits(args.logits_dir)
    anchor = os.path.basename(os.path.normpath(args.logits_dir))
    col = f"expected_fail_{anchor}"
    skipped = set(clips.clip_id[clips[col]].tolist()) if col in clips else set()
    for f in glob.glob(os.path.join(args.logits_dir, "shard*", "failed.json")):
        with open(f) as fh:
            skipped |= set(json.load(fh))
    keep = ~np.isin(lg["clip_id"], list(skipped))
    ids, logits = lg["clip_id"][keep], lg["action"][keep].astype(np.float32)
    meta = clips.set_index("clip_id").loc[ids]
    labels = meta.action_idx.values
    hits = topk_hits(logits, labels)
    res = {}

    def official(ws=64, bs=2, nw=2, h=hits):
        counts = official_eval_multiplicity(clips, ws, bs, nw, skipped=skipped)
        w = np.array([counts.get(int(c), 0) for c in ids], dtype=np.float64)
        return mean_class_recall(h, labels, n_act, weights=w)["recall"]

    res["clean"] = mean_class_recall(hits, labels, n_act)["recall"]
    res["official_64gpu"] = official()

    # 1. cluster layout sensitivity
    res["official_by_world_size"] = {ws: official(ws) for ws in (1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128)}
    res["official_64gpu_by_num_workers"] = {nw: official(64, 2, nw) for nw in (0, 1, 2, 4, 8)}
    vals = list(res["official_by_world_size"].values())
    res["official_world_size_range"] = [min(vals), max(vals)]

    # 2. bootstrap over clips
    rng = np.random.default_rng(0)
    boots = []
    for _ in range(args.bootstrap):
        i = rng.integers(0, len(labels), len(labels))
        boots.append(mean_class_recall(hits[i], labels[i], n_act)["recall"])
    res["clean_bootstrap_std"] = float(np.std(boots))
    res["clean_bootstrap_95ci"] = [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]

    # 3. chance level
    perm = rng.permutation(len(labels))
    res["clean_shuffled_labels"] = mean_class_recall(topk_hits(logits, labels[perm]), labels, n_act)["recall"]

    # 4. official numerics (bf16 sigmoid + torch.topk)
    h_bf16 = official_style_hits(logits, labels)
    res["clean_bf16_sigmoid_topk"] = mean_class_recall(h_bf16, labels, n_act)["recall"]
    res["official_64gpu_bf16_sigmoid_topk"] = official(h=h_bf16)
    res["hits_changed_by_bf16_scoring"] = int((h_bf16 != hits).sum())

    # 5. videos with mismatched frame ids
    bad = meta.video_id.isin(NON_NATIVE_FPS_VIDEOS).values
    res["clips_in_non_native_fps_videos"] = int(bad.sum())
    res["clean_without_non_native_fps_videos"] = mean_class_recall(hits[~bad], labels[~bad], n_act)["recall"]
    res["top5_acc_non_native_fps_videos"] = float(hits[bad].mean() * 100) if bad.any() else None
    res["top5_acc_other_videos"] = float(hits[~bad].mean() * 100)

    # 6. agreement with another run
    if args.compare_logits_dir:
        other = load_logits(args.compare_logits_dir)
        common, ia, ib = np.intersect1d(lg["clip_id"], other["clip_id"], return_indices=True)
        a, b = lg["action"][ia].astype(np.float32), other["action"][ib].astype(np.float32)
        ta = np.argsort(-a, axis=1)[:, :5]
        tb = np.argsort(-b, axis=1)[:, :5]
        res["compare"] = dict(
            clips=int(len(common)),
            top1_agree=float((ta[:, 0] == tb[:, 0]).mean()),
            top5_set_agree=float(np.mean([set(x) == set(y) for x, y in zip(ta, tb)])),
            max_abs_logit_diff=float(np.abs(a - b).max()),
            mean_abs_logit_diff=float(np.abs(a - b).mean()),
        )

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
