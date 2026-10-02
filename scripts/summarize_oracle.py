"""Step 0 of the predictor study: how much would a better future predictor help?

Reads the logits written by `eval_ek100.py --oracle` and scores, on the same clips:
  base      encoder + V-JEPA 2 predictor (as released)
  oracle    encoder + the REAL encoder tokens of the predicted time step (a perfect predictor)
  enconly   encoder tokens only
  copylast  encoder + the last observed time step repeated ("nothing changes")
plus the cosine similarity of the predicted tokens (and of copylast) to the real future tokens.

python scripts/summarize_oracle.py --work_dir work --logits_root logits_orig --anchor action_start --out oracle.json
"""

import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from vjepa_ek100.metrics import mean_class_recall, topk_hits  # noqa: E402

VARIANTS = ("base", "oracle", "enconly", "copylast")


def load(dir_, keys):
    parts = [np.load(f) for f in sorted(glob.glob(os.path.join(dir_, "shard*", "chunk_*.npz")))]
    out = {k: np.concatenate([p[k] for p in parts]) for k in keys if parts and k in parts[0]}
    if out:
        _, first = np.unique(out["clip_id"], return_index=True)
        out = {k: v[first] for k, v in out.items()}
    failed = set()
    for f in glob.glob(os.path.join(dir_, "shard*", "failed.json")):
        with open(f) as fh:
            failed |= set(json.load(fh))
    return out, failed


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--work_dir", required=True)
    p.add_argument("--logits_root", required=True)
    p.add_argument("--anchor", default="action_start")
    p.add_argument("--only_videos", default=None)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    clips = pd.read_csv(os.path.join(args.work_dir, "val_clips.csv")).set_index("clip_id")
    if args.only_videos:
        with open(args.only_videos) as f:
            clips = clips[clips.video_id.isin(f.read().split())]
    with open(os.path.join(args.work_dir, "classes.json")) as f:
        classes = json.load(f)
    n_cls = dict(verb=len(classes["verbs"]), noun=len(classes["nouns"]), action=len(classes["actions"]))

    data, failed = {}, set()
    for v in VARIANTS:
        d = os.path.join(args.logits_root, args.anchor if v == "base" else f"{args.anchor}__{v}")
        keys = ("clip_id", "verb", "noun", "action") + (("cos_pred", "cos_copylast") if v == "base" else ())
        data[v], f = load(d, keys)
        failed |= f
        if not data[v]:
            raise SystemExit(f"no logits in {d}: run eval_ek100.py --oracle first")
    common = set(clips.index) - failed
    for v in VARIANTS:
        common &= set(data[v]["clip_id"].tolist())
    common = np.array(sorted(common))

    res = dict(anchor=args.anchor, clips=int(len(common)), recall={})
    for v in VARIANTS:
        idx = np.searchsorted(data[v]["clip_id"], common)
        res["recall"][v] = {}
        for k in n_cls:
            labels = clips.loc[common, f"{k}_idx"].values
            hits = topk_hits(data[v][k][idx].astype(np.float32), labels)
            res["recall"][v][k] = round(mean_class_recall(hits, labels, n_cls[k])["recall"], 2)
    idx = np.searchsorted(data["base"]["clip_id"], common)
    cp, cc = data["base"]["cos_pred"][idx], data["base"]["cos_copylast"][idx]
    res["cosine_to_real_future"] = dict(
        predictor=float(cp.mean()), copylast=float(cc.mean()), predictor_better_share=float((cp > cc).mean())
    )
    r = res["recall"]
    res["headroom_action"] = dict(
        oracle_minus_base=round(r["oracle"]["action"] - r["base"]["action"], 2),
        base_minus_enconly=round(r["base"]["action"] - r["enconly"]["action"], 2),
        base_minus_copylast=round(r["base"]["action"] - r["copylast"]["action"], 2),
    )
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)

    print(f"{len(common)} clips, anchor={args.anchor}  (mean-class R@5, each clip once)")
    print(f"{'':10s} {'action':>7s} {'verb':>7s} {'noun':>7s}")
    for v in VARIANTS:
        print(f"{v:10s} {r[v]['action']:7.2f} {r[v]['verb']:7.2f} {r[v]['noun']:7.2f}")
    c = res["cosine_to_real_future"]
    print(f"cosine to real future tokens: predictor {c['predictor']:.3f}, copy-last {c['copylast']:.3f} "
          f"(predictor closer on {c['predictor_better_share']:.0%} of clips)")
    print("headroom (action):", res["headroom_action"])


if __name__ == "__main__":
    main()
