"""Post-hoc re-scoring of the released probe's logits for mean-class recall (no training, CPU only).

Mean-class R@5 averages over ~3.8k action classes, most of them rare, while the probe is trained on the
skewed train distribution. Two standard, training-free corrections are tried on dumped logits:

  logit adjustment   z_c - tau * log prior_c          (prior = class frequency in EPIC_100_train.csv)
  verb-noun fusion   action(v, n) = log_softmax(z_a) + beta * (log_softmax(z_v)[v] + log_softmax(z_n)[n]),
                     then logit adjustment with the action prior

tau / beta are chosen by cross-fitting over participant folds of val (tuned on the other participants, applied to
the held-out ones), so the reported numbers are not tuned on the clips they are measured on. The value tuned on
the whole val set is reported as well, marked optimistic.

python scripts/posthoc_ek100.py --work_dir work --train_csv EPIC_100_train.csv --logits_dir logits/official --out p.json
"""

import argparse
import glob
import itertools
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))
from compute_metrics import load_logits  # noqa: E402
from cv_probe import participant_folds  # noqa: E402
from vjepa_ek100.metrics import mean_class_recall, topk_hits  # noqa: E402
from vjepa_ek100.protocol import official_eval_multiplicity  # noqa: E402

TASKS = ("verb", "noun", "action")
TAUS = [round(0.1 * i, 1) for i in range(0, 16)]
BETAS = [0.0, 0.1, 0.25, 0.5, 1.0, 2.0]


def log_softmax(z):
    z = z - z.max(1, keepdims=True)
    return z - np.log(np.exp(z).sum(1, keepdims=True))


def class_priors(train_csv, classes):
    """log frequency of each class index in the train csv (add-one smoothed)."""
    tdf = pd.read_csv(train_csv)
    maps = dict(verb={int(k): i for k, i in classes["verbs"]}, noun={int(k): i for k, i in classes["nouns"]},
                action={(int(v), int(n)): i for v, n, i in classes["actions"]})
    idx = dict(verb=tdf.verb_class.map(maps["verb"]).values, noun=tdf.noun_class.map(maps["noun"]).values,
               action=np.array([maps["action"][(v, n)] for v, n in zip(tdf.verb_class, tdf.noun_class)]))
    out = {}
    for k in TASKS:
        cnt = np.bincount(idx[k].astype(np.int64), minlength=len(maps[k])) + 1.0
        out[k] = np.log(cnt / cnt.sum()).astype(np.float32)
    return out


def action_verb_noun(classes):
    """verb index and noun index of every action index."""
    vmap = {int(k): i for k, i in classes["verbs"]}
    nmap = {int(k): i for k, i in classes["nouns"]}
    a_v = np.zeros(len(classes["actions"]), np.int64)
    a_n = np.zeros(len(classes["actions"]), np.int64)
    for v, n, i in classes["actions"]:
        a_v[i], a_n[i] = vmap[int(v)], nmap[int(n)]
    return a_v, a_n


def score(task, lg, prior, a_v, a_n, tau, beta):
    if task != "action":
        return lg[task] - tau * prior[task]
    s = log_softmax(lg["action"])
    if beta:
        s = s + beta * (log_softmax(lg["verb"])[:, a_v] + log_softmax(lg["noun"])[:, a_n])
    return s - tau * prior["action"]


def grid(task):
    return [(t, b) for t, b in itertools.product(TAUS, BETAS if task == "action" else [0.0])]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--work_dir", required=True)
    p.add_argument("--train_csv", required=True)
    p.add_argument("--logits_dir", required=True, help="e.g. logits/official or logits_feats/action_start")
    p.add_argument("--out", required=True)
    p.add_argument("--num_folds", type=int, default=2)
    p.add_argument("--bootstrap", type=int, default=1000)
    args = p.parse_args()

    clips = pd.read_csv(os.path.join(args.work_dir, "val_clips.csv"))
    with open(os.path.join(args.work_dir, "classes.json")) as f:
        classes = json.load(f)
    n_cls = dict(verb=len(classes["verbs"]), noun=len(classes["nouns"]), action=len(classes["actions"]))
    prior = class_priors(args.train_csv, classes)
    a_v, a_n = action_verb_noun(classes)

    lg = load_logits(args.logits_dir)
    failed = set()
    for f in glob.glob(os.path.join(args.logits_dir, "shard*", "failed.json")):
        with open(f) as fh:
            failed |= set(json.load(fh))
    anchor = os.path.basename(os.path.normpath(args.logits_dir)).split("__")[0]
    col = f"expected_fail_{anchor}"
    skipped = failed | (set(clips.clip_id[clips[col]]) if col in clips else set())
    keep = np.isin(lg["clip_id"], clips.clip_id.values) & ~np.isin(lg["clip_id"], list(skipped))
    lg = {k: (v[keep] if k == "clip_id" else v[keep].astype(np.float32)) for k, v in lg.items()}
    c = clips.set_index("clip_id").loc[lg["clip_id"]]
    c = c.assign(participant=c.video_id.str.split("_").str[0])
    labels = {k: c[f"{k}_idx"].values for k in TASKS}
    folds = participant_folds(c, args.num_folds)
    counts = official_eval_multiplicity(clips, 64, 2, 2, skipped=skipped)
    w = np.array([counts.get(int(i), 0) for i in lg["clip_id"]], dtype=np.float64)

    def recall(hits, sel=slice(None), task="action", weights=None):
        return mean_class_recall(hits[sel], labels[task][sel], n_cls[task],
                                 None if weights is None else weights[sel])["recall"]

    res = dict(logits_dir=args.logits_dir, clips=int(len(lg["clip_id"])), tasks={})
    final_hits = {}
    for task in TASKS:
        hits_by = {}
        for tb in grid(task):
            hits_by[tb] = topk_hits(score(task, lg, prior, a_v, a_n, *tb), labels[task])
        base = hits_by[(0.0, 0.0)]
        # cross-fitting: choose (tau, beta) on the other folds, apply to this fold
        xfit, chosen = np.zeros_like(base), []
        for f in range(args.num_folds):
            tr, te = folds != f, folds == f
            best = max(hits_by, key=lambda tb: recall(hits_by[tb], tr, task))
            xfit[te] = hits_by[best][te]
            chosen.append(dict(tau=best[0], beta=best[1]))
        best_all = max(hits_by, key=lambda tb: recall(hits_by[tb], task=task))
        final_hits[task] = (base, xfit)
        r = dict(
            released=dict(clean=recall(base, task=task), official=recall(base, task=task, weights=w)),
            cross_fitted=dict(clean=recall(xfit, task=task), official=recall(xfit, task=task, weights=w),
                              chosen_per_fold=chosen),
            tuned_on_all_val_optimistic=dict(tau=best_all[0], beta=best_all[1], clean=recall(hits_by[best_all], task=task),
                                             official=recall(hits_by[best_all], task=task, weights=w)),
            logit_adjustment_only=None,
        )
        if task == "action":  # how much of the gain comes from the prior alone
            la = {tb: h for tb, h in hits_by.items() if tb[1] == 0.0}
            x2 = np.zeros_like(base)
            for f in range(args.num_folds):
                tr, te = folds != f, folds == f
                x2[te] = la[max(la, key=lambda tb: recall(la[tb], tr, task))][te]
            r["logit_adjustment_only"] = dict(clean=recall(x2, task=task), official=recall(x2, task=task, weights=w))
        rng = np.random.default_rng(0)
        d = []
        for _ in range(args.bootstrap):
            i = rng.integers(0, len(base), len(base))
            d.append(recall(xfit, i, task) - recall(base, i, task))
        r["cross_fitted_minus_released_clean_ci95"] = [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]
        res["tasks"][task] = r
    with open(args.out, "w") as fh:
        json.dump(res, fh, indent=2)

    print(f"{res['clips']} clips from {args.logits_dir}  (mean-class R@5; official = Meta's 64-GPU counting)")
    print(f"{'':8s} {'released':>17s} {'cross-fitted':>17s} {'gain (clean) 95% CI':>24s}  tuned-on-all (optimistic)")
    for task in TASKS:
        r = res["tasks"][task]
        a, b, o = r["released"], r["cross_fitted"], r["tuned_on_all_val_optimistic"]
        lo, hi = r["cross_fitted_minus_released_clean_ci95"]
        print(f"{task:8s} {a['clean']:7.2f} / {a['official']:7.2f}  {b['clean']:7.2f} / {b['official']:7.2f}  "
              f"{b['clean'] - a['clean']:+6.2f} [{lo:+.2f}, {hi:+.2f}]   "
              f"{o['clean']:.2f} / {o['official']:.2f} (tau {o['tau']}, beta {o['beta']})  chosen {b['chosen_per_fold']}")
    la = res["tasks"]["action"]["logit_adjustment_only"]
    print(f"action, logit adjustment only (no verb-noun fusion), cross-fitted: {la['clean']:.2f} / {la['official']:.2f}")
    print("columns: clean / official")


if __name__ == "__main__":
    main()
