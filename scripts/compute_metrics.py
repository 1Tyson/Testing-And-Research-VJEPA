"""Mean-class recall@5 from dumped logits.

Reports two numbers:
  clean     - every val clip counted once (the well-defined metric)
  official  - clips weighted as the official 64-GPU inference run counts them
              (see protocol.official_eval_multiplicity); this is what the 32.7 was
              measured with.

python scripts/compute_metrics.py --work_dir work/ek100 --logits_dir work/logits/official --out results/verify_vitl.json
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
from vjepa_ek100.protocol import official_eval_multiplicity  # noqa: E402

PAPER = dict(action=32.7)


def load_logits(logits_dir):
    parts = [np.load(f) for f in sorted(glob.glob(os.path.join(logits_dir, "shard*", "chunk_*.npz")))]
    if not parts:
        raise SystemExit(f"no chunks under {logits_dir}")
    out = {k: np.concatenate([p[k] for p in parts]) for k in ("clip_id", "verb", "noun", "action")}
    _, first = np.unique(out["clip_id"], return_index=True)
    return {k: v[first] for k, v in out.items()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--work_dir", required=True)
    p.add_argument("--logits_dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--world_size", type=int, default=64, help="official inference: 8 nodes x 8 gpus")
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--allow_partial", action="store_true", help="report even if some clips were not evaluated yet")
    p.add_argument("--only_videos", default=None, help="file with video_ids: score only clips of these videos")
    p.add_argument("--paper_action", type=float, default=PAPER["action"], help="paper number to compare with (ViT-g/384: 39.7)")
    args = p.parse_args()
    PAPER["action"] = args.paper_action

    clips = pd.read_csv(os.path.join(args.work_dir, "val_clips.csv"))
    if args.only_videos:
        with open(args.only_videos) as f:
            clips = clips[clips.video_id.isin(f.read().split())]
    with open(os.path.join(args.work_dir, "classes.json")) as f:
        classes = json.load(f)
    n_cls = dict(verb=len(classes["verbs"]), noun=len(classes["nouns"]), action=len(classes["actions"]))

    lg = load_logits(args.logits_dir)
    in_scope = np.isin(lg["clip_id"], clips.clip_id.values)
    lg = {k: v[in_scope] for k, v in lg.items()}

    # Clips that cannot be decoded (errors seen at eval time, or windows past the video end).
    # The official loader drops these silently; they are excluded from both metrics.
    failed = set()
    for f in glob.glob(os.path.join(args.logits_dir, "shard*", "failed.json")):
        with open(f) as fh:
            failed.update(json.load(fh))
    anchor = os.path.basename(os.path.normpath(args.logits_dir)).split("__")[0]  # e.g. action_start__oracle
    col = f"expected_fail_{anchor}"
    expected_fail = set(clips.clip_id[clips[col]].tolist()) if col in clips else set()
    skipped = failed | expected_fail
    missing = set(clips.clip_id.tolist()) - set(lg["clip_id"].tolist()) - skipped
    if missing and not args.allow_partial:
        raise SystemExit(f"{len(missing)} clips not evaluated yet: rerun eval_ek100.py (or pass --allow_partial)")
    keep = ~np.isin(lg["clip_id"], list(skipped))
    lg = {k: v[keep] for k, v in lg.items()}
    lab = clips.set_index("clip_id").loc[lg["clip_id"]]
    labels = dict(verb=lab.verb_idx.values, noun=lab.noun_idx.values, action=lab.action_idx.values)
    hits = {k: topk_hits(lg[k].astype(np.float32), labels[k], k=5) for k in n_cls}

    res = dict(num_clips_total=len(clips), num_clips_evaluated=len(lg["clip_id"]), num_clips_skipped=len(skipped),
               num_clips_failed_at_eval=len(failed), num_clips_out_of_range=len(expected_fail),
               num_clips_missing=len(missing), clean={}, official={})
    for k in n_cls:
        res["clean"][k] = mean_class_recall(hits[k], labels[k], n_cls[k])

    counts = official_eval_multiplicity(clips, args.world_size, args.batch_size, args.num_workers, skipped=skipped)
    w = np.array([counts.get(int(c), 0) for c in lg["clip_id"]], dtype=np.float64)
    evaluated = set(lg["clip_id"].tolist())
    missing_weight = sum(v for c, v in counts.items() if c not in evaluated)
    for k in n_cls:
        res["official"][k] = mean_class_recall(hits[k], labels[k], n_cls[k], weights=w)
    res["official_protocol"] = dict(
        world_size=args.world_size,
        clips_never_counted=int((w == 0).sum()),
        clips_counted_more_than_once=int((w > 1).sum()),
        weight_of_missing_clips=int(missing_weight),
    )
    res["paper"] = PAPER
    res["gap_action_recall_vs_paper"] = dict(
        clean=res["clean"]["action"]["recall"] - PAPER["action"],
        official=res["official"]["action"]["recall"] - PAPER["action"],
    )

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    print(f"clips evaluated: {res['num_clips_evaluated']}/{res['num_clips_total']}  "
          f"(skipped {len(skipped)} undecodable: {len(failed)} failed at eval, {len(expected_fail)} predicted past video end; "
          f"missing {len(missing)})")
    for proto in ("clean", "official"):
        r = res[proto]
        print(
            f"[{proto:8s}] mean-class R@5  action {r['action']['recall']:.2f}  "
            f"verb {r['verb']['recall']:.2f}  noun {r['noun']['recall']:.2f}   (paper action {PAPER['action']})"
        )


if __name__ == "__main__":
    main()
