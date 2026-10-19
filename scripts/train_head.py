"""Re-train the verb / noun / action layers of the released EK100 probe on frozen pooler features (direction 2).

The released probe is  attentive pooler (4 blocks, 3 queries)  ->  3 linear layers. Its pooler output for every clip
(eval_ek100.py --save_pooled) is a [3, 1024] array: the inputs of the verb / noun / action layers. This script keeps
the pooler frozen and re-trains only those layers on the TRAIN clips with long-tail aware objectives, then scores
the val clips like compute_metrics.py (clean and official counting).

Recipes (space separated, tokens joined by "+"):
  meta                    the released layers, no training (baseline; also checks the stored logits are reproduced)
  focal | ce | bsm | la   loss: sigmoid focal (as Meta), softmax CE, Balanced Softmax (CE on z + log prior),
                          logit-adjusted CE (CE on z + la_tau * log prior)
  +cb | +sqrt             class-balanced / square-root sampling by action label (decoupling, cRT style)
  +compose                action logit z_a + b_v * log_softmax(z_v)[verb(a)] + b_n * log_softmax(z_n)[noun(a)],
                          b_v, b_n learned (training-time version of the post-hoc verb-noun fusion)
  +random                 start from random layers instead of the released ones

Every recipe is also reported after the post-hoc re-scoring of posthoc_ek100.py (tau, beta cross-fitted over val
participants), so "balancing while training" and "re-scoring afterwards" are compared on the same footing.
The epoch is chosen on held-out TRAIN participants (--holdout); nothing is tuned on val except that cross-fitting.

python scripts/train_head.py --work_dir work --train_csv EPIC_100_train.csv --train_feats feats_train/action_start \
    --val_feats feats_val/action_start --probe_ckpt ek100-vitl-256.pt --out results/heads_action_start.json \
    --recipes meta focal ce bsm bsm+compose bsm+cb --seeds 0 1 2
python scripts/train_head.py --synthetic /tmp/syn --out /tmp/syn/heads.json      # self-test, no data needed
"""

import argparse
import copy
import glob
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))
from cv_probe import participant_folds  # noqa: E402
from posthoc_ek100 import grid, load_inputs, log_softmax  # noqa: E402
from vjepa_ek100.metrics import mean_class_recall, topk_hits  # noqa: E402

TASKS = ("verb", "noun", "action")  # = order of the pooler queries in AttentiveClassifier
LOSSES = ("focal", "ce", "bsm", "la")


# ----------------------------------------------------------------------------------------------- data
def load_feature_chunks(feats_dir, keys=("clip_id", "pooled")):
    files = sorted(glob.glob(os.path.join(feats_dir, "shard*", "chunk_*.npz")))
    if not files:
        raise SystemExit(f"no chunk_*.npz under {feats_dir}/shard*: run eval_ek100.py --save_pooled first")
    parts = []
    for f in files:
        with np.load(f) as z:
            if "pooled" in keys and "pooled" not in z.files:
                raise SystemExit(f"{f} has no 'pooled': extract with eval_ek100.py --save_pooled")
            parts.append({k: z[k] for k in keys})
    out = {k: np.concatenate([p[k] for p in parts]) for k in keys}
    _, first = np.unique(out["clip_id"], return_index=True)
    return {k: v[first] for k, v in out.items()}


def load_train(args):
    clips = pd.read_csv(os.path.join(args.work_dir, "train_clips.csv"))
    feats = load_feature_chunks(args.train_feats)
    c = clips.set_index("clip_id").reindex(feats["clip_id"])
    known = c.action_idx.notna().values
    if not known.all():
        print(f"[warn] {int((~known).sum())} extracted clips are not in train_clips.csv: ignored")
    c = c[known]
    if args.views is not None:
        keep = c["view"].values < args.views
        c, known_idx = c[keep], np.flatnonzero(known)[keep]
    else:
        known_idx = np.flatnonzero(known)
    x = feats["pooled"][known_idx]
    y = {k: c[f"{k}_idx"].astype(np.int64).values for k in TASKS}
    return x, y, c.reset_index()


def train_log_prior(y, n_cls):
    """log class frequency of the clips actually used for training (add-one smoothed), per task."""
    out = {}
    for k in TASKS:
        cnt = np.bincount(y[k], minlength=n_cls[k]).astype(np.float64) + 1.0
        out[k] = np.log(cnt / cnt.sum()).astype(np.float32)
    return out


# ----------------------------------------------------------------------------------------------- model
class Heads(nn.Module):
    def __init__(self, dim, n_cls, a_v, a_n, compose=False, init_sd=None):
        super().__init__()
        self.lin = nn.ModuleDict({k: nn.Linear(dim, n_cls[k]) for k in TASKS})
        if init_sd is not None:
            for k in TASKS:
                self.lin[k].weight.data.copy_(init_sd[f"{k}_classifier.weight"].float())
                self.lin[k].bias.data.copy_(init_sd[f"{k}_classifier.bias"].float())
        self.compose = compose
        if compose:
            self.beta = nn.Parameter(torch.tensor([0.5, 0.5]))
            self.register_buffer("a_v", torch.as_tensor(a_v, dtype=torch.long))
            self.register_buffer("a_n", torch.as_tensor(a_n, dtype=torch.long))

    def forward(self, x):
        z = {k: self.lin[k](x[:, i]) for i, k in enumerate(TASKS)}
        if self.compose:
            lv, ln = F.log_softmax(z["verb"], 1), F.log_softmax(z["noun"], 1)
            z["action"] = z["action"] + self.beta[0] * lv[:, self.a_v] + self.beta[1] * ln[:, self.a_n]
        return z


def sigmoid_focal(z, y, alpha=0.25, gamma=2.0):
    """vjepa2 evals/action_anticipation_frozen/losses.py, summed over classes, averaged over the batch."""
    t = F.one_hot(y, z.shape[1]).float()
    p = torch.sigmoid(z)
    ce = F.binary_cross_entropy_with_logits(z, t, reduction="none")
    p_t = p * t + (1 - p) * (1 - t)
    loss = ce * (1 - p_t) ** gamma
    loss = (alpha * t + (1 - alpha) * (1 - t)) * loss
    return loss.sum(1).mean()


def task_loss(kind, z, y, log_prior, la_tau):
    if kind == "focal":
        return sigmoid_focal(z, y)
    if kind == "ce":
        return F.cross_entropy(z, y)
    if kind == "bsm":
        return F.cross_entropy(z + log_prior, y)
    if kind == "la":
        return F.cross_entropy(z + la_tau * log_prior, y)
    raise ValueError(kind)


def parse_recipe(name):
    tok = name.split("+")
    loss = [t for t in tok if t in LOSSES]
    if name == "meta":
        return None
    if len(loss) != 1 or any(t not in LOSSES + ("cb", "sqrt", "compose", "random") for t in tok):
        raise SystemExit(f"bad recipe {name!r}: one of {LOSSES} plus optional +cb/+sqrt/+compose/+random")
    return dict(loss=loss[0], sampler="cb" if "cb" in tok else "sqrt" if "sqrt" in tok else "instance",
                compose="compose" in tok, init="random" if "random" in tok else "meta")


@torch.no_grad()
def predict(model, x, device, bs=2048):
    model.eval()
    outs = {k: [] for k in TASKS}
    for i in range(0, len(x), bs):
        z = model(torch.as_tensor(x[i:i + bs], device=device).float())
        for k in TASKS:
            outs[k].append(z[k].float().cpu().numpy())
    return {k: np.concatenate(v) for k, v in outs.items()}


def holdout_recall(model, x, y, n_cls, device):
    z = predict(model, x, device)
    return mean_class_recall(topk_hits(z["action"], y["action"]), y["action"], n_cls["action"])["recall"]


def train_recipe(rec, x, y, train_idx, hold_idx, n_cls, a_v, a_n, log_prior, init_sd, args, seed, device):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if rec["init"] == "meta" and init_sd is None:
        raise SystemExit("recipes start from the released layers: pass --probe_ckpt (or add +random)")
    model = Heads(x.shape[2], n_cls, a_v, a_n, rec["compose"], init_sd if rec["init"] == "meta" else None).to(device)
    lr = args.lr if rec["init"] == "meta" else args.lr_random
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=args.weight_decay)
    X = torch.as_tensor(x[train_idx], device=device if args.feats_on_gpu else "cpu")
    Y = {k: torch.as_tensor(y[k][train_idx], device=device) for k in TASKS}
    lp = {k: torch.as_tensor(log_prior[k], device=device) for k in TASKS}
    n = len(train_idx)
    if rec["sampler"] != "instance":
        cnt = np.bincount(y["action"][train_idx], minlength=n_cls["action"]).astype(np.float64)
        w = 1.0 / cnt[y["action"][train_idx]]
        w = np.sqrt(w) if rec["sampler"] == "sqrt" else w
        w = torch.as_tensor(w / w.sum(), dtype=torch.float64)
    steps = args.epochs * ((n + args.batch_size - 1) // args.batch_size)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 0.5 * (1 + np.cos(np.pi * min(s, steps) / steps)))
    gen = torch.Generator().manual_seed(seed)
    hist, best, best_state = [], -1.0, None
    for ep in range(args.epochs):
        model.train()
        order = (torch.multinomial(w, n, replacement=True, generator=gen) if rec["sampler"] != "instance"
                 else torch.randperm(n, generator=gen))
        tot = 0.0
        for i in range(0, n, args.batch_size):
            b = order[i:i + args.batch_size]
            xb = X[b.to(X.device)].to(device).float()
            z = model(xb)
            bd = b.to(device)
            loss = sum(task_loss(rec["loss"], z[k], Y[k][bd], lp[k], args.la_tau) for k in TASKS)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tot += float(loss) * len(b)
        h = dict(epoch=ep + 1, loss=tot / n)
        if len(hold_idx):
            h["holdout_action_recall"] = holdout_recall(model, x[hold_idx], {k: y[k][hold_idx] for k in TASKS},
                                                        n_cls, device)
            if h["holdout_action_recall"] > best:
                best, best_state = h["holdout_action_recall"], copy.deepcopy(model.state_dict())
        hist.append(h)
        print(f"    seed {seed} epoch {ep + 1:2d}  loss {h['loss']:.4f}"
              + (f"  holdout action R@5 {h['holdout_action_recall']:.2f}" if len(hold_idx) else ""), flush=True)
    if best_state is not None:
        model.load_state_dict(best_state)
    picked = max(hist, key=lambda h: h.get("holdout_action_recall", -1))["epoch"] if len(hold_idx) else args.epochs
    extra = dict(history=hist, epoch_used=picked)
    if rec["compose"]:
        extra["beta_verb_noun"] = [float(v) for v in model.beta.detach().cpu()]
    return model, extra


# ----------------------------------------------------------------------------------------------- scoring
def evaluate(lg, d, boot_reps=0, ref=None, rng=None):
    """lg: {task: [N, C]} aligned with d['labels']. Raw and cross-fitted post-hoc scores (clean / official)."""
    labels, n_cls, w, folds = d["labels"], d["n_cls"], d["w"], d["folds"]
    res, hits_out = {}, {}
    # same scores as posthoc_ek100.score, with the log-softmaxes computed once instead of once per grid point
    s_act = log_softmax(lg["action"])
    s_vn = log_softmax(lg["verb"])[:, d["a_v"]] + log_softmax(lg["noun"])[:, d["a_n"]]
    for task in TASKS:
        def scored(tau, beta):
            if task != "action":
                return lg[task] - tau * d["prior"][task]
            return (s_act + beta * s_vn if beta else s_act) - tau * d["prior"]["action"]

        hits_by = {tb: topk_hits(scored(*tb), labels[task]) for tb in grid(task)}

        def rec(h, sel=slice(None), weights=None):
            return mean_class_recall(h[sel], labels[task][sel], n_cls[task],
                                     None if weights is None else weights[sel])["recall"]

        raw = hits_by[(0.0, 0.0)]
        xfit, chosen = np.zeros_like(raw), []
        for f in np.unique(folds):
            tr, te = folds != f, folds == f
            best = max(hits_by, key=lambda tb: rec(hits_by[tb], tr))
            xfit[te] = hits_by[best][te]
            chosen.append(dict(tau=best[0], beta=best[1]))
        res[task] = dict(raw=dict(clean=rec(raw), official=rec(raw, weights=w)),
                         posthoc=dict(clean=rec(xfit), official=rec(xfit, weights=w), chosen_per_fold=chosen))
        hits_out[task] = dict(raw=raw, posthoc=xfit)
    if boot_reps and ref is not None:  # CI of (this - ref) on clean action recall, same clips resampled
        lab = labels["action"]
        for kind in ("raw", "posthoc"):
            diffs = []
            for _ in range(boot_reps):
                i = rng.integers(0, len(lab), len(lab))
                diffs.append(mean_class_recall(hits_out["action"][kind][i], lab[i], n_cls["action"])["recall"]
                             - mean_class_recall(ref["action"][kind][i], lab[i], n_cls["action"])["recall"])
            res["action"][kind]["minus_meta_clean_ci95"] = [float(np.percentile(diffs, 2.5)),
                                                           float(np.percentile(diffs, 97.5))]
    return res, hits_out


def save_logits(logits_out, recipe, val_feats, clip_ids, lg):
    """<logits_out>/<recipe>/<anchor>/shard0/chunk_00000.npz (+ the val run's failed.json), i.e. the layout of
    eval_ek100.py, so posthoc_ek100.py / analyze_posthoc.py / compute_metrics.py read it unchanged (they take the
    anchor from the folder name to know which clips are out of range)."""
    anchor = os.path.basename(os.path.normpath(val_feats))
    d = os.path.join(logits_out, recipe, anchor, "shard0")
    os.makedirs(d, exist_ok=True)
    np.savez(os.path.join(d, "chunk_00000.npz"), clip_id=clip_ids, **{k: lg[k].astype(np.float16) for k in TASKS})
    failed = set()
    for f in glob.glob(os.path.join(val_feats, "shard*", "failed.json")):
        with open(f) as fh:
            failed |= set(json.load(fh))
    with open(os.path.join(d, "failed.json"), "w") as fh:
        json.dump(sorted(failed), fh)


# ----------------------------------------------------------------------------------------------- synthetic
def make_synthetic(root, dim=32, V=6, N=8, n_train=6000, n_val=3000, seed=0):
    """Tiny fake work_dir / features / probe: skewed train labels, uniform val labels (like EK100's tail)."""
    rng = np.random.default_rng(seed)
    os.makedirs(root, exist_ok=True)
    actions = [(v, m) for v in range(V) for m in range(N)]
    A = len(actions)
    freq = np.exp(rng.normal(0, 1.5, A))
    mu = {"verb": rng.normal(0, 1, (V, dim)), "noun": rng.normal(0, 1, (N, dim)), "action": rng.normal(0, 1, (A, dim))}

    def feats(lab):
        v, m = np.array([actions[a][0] for a in lab]), np.array([actions[a][1] for a in lab])
        x = np.stack([mu["verb"][v], mu["noun"][m], mu["action"][lab]], 1) + rng.normal(0, 1.6, (len(lab), 3, dim))
        return x.astype(np.float16), v, m

    tr_lab = rng.choice(A, n_train, p=freq / freq.sum())
    va_lab = rng.integers(0, A, n_val)
    xt, tv, tn = feats(tr_lab)
    xv, vv, vn = feats(va_lab)
    pd.DataFrame(dict(verb_class=tv, noun_class=tn)).to_csv(os.path.join(root, "train.csv"), index=False)
    work = os.path.join(root, "work")
    os.makedirs(work, exist_ok=True)
    with open(os.path.join(work, "classes.json"), "w") as f:
        json.dump(dict(verbs=[[v, v] for v in range(V)], nouns=[[m, m] for m in range(N)],
                       actions=[[v, m, i] for i, (v, m) in enumerate(actions)]), f)
    part_t = [f"P{i % 12 + 1:02d}" for i in range(n_train)]
    pd.DataFrame(dict(clip_id=np.arange(n_train), seg_id=np.arange(n_train), view=0, participant=part_t,
                      video_id=[p + "_101" for p in part_t], verb_idx=tv, noun_idx=tn, action_idx=tr_lab)).to_csv(
        os.path.join(work, "train_clips.csv"), index=False)
    vid = [f"P{i % 8 + 13:02d}_01" for i in range(n_val)]
    order = {v: i for i, v in enumerate(sorted(set(vid)))}
    pd.DataFrame(dict(clip_id=np.arange(n_val), video_id=vid, video_order=[order[v] for v in vid],
                      clip_order=np.arange(n_val), verb_idx=vv, noun_idx=vn, action_idx=va_lab)).to_csv(
        os.path.join(work, "val_clips.csv"), index=False)
    # a "released" probe: least-squares layers fitted on the skewed train set (so they carry its prior)
    sd = {}
    for i, (k, lab, C) in enumerate((("verb", tv, V), ("noun", tn, N), ("action", tr_lab, A))):
        X = np.c_[xt[:, i].astype(np.float64), np.ones(n_train)]
        W = np.linalg.lstsq(X, np.eye(C)[lab] * 4 - 2, rcond=None)[0]
        sd[f"{k}_classifier.weight"], sd[f"{k}_classifier.bias"] = torch.tensor(W[:-1].T), torch.tensor(W[-1])
    torch.save(dict(classifiers=[{f"module.{k}": v for k, v in sd.items()}]), os.path.join(root, "probe.pt"))
    for name, x, ids, labs in (("feats_train", xt, np.arange(n_train), (tv, tn, tr_lab)),
                               ("feats_val", xv, np.arange(n_val), (vv, vn, va_lab))):
        d = os.path.join(root, name, "action_start", "shard0")
        os.makedirs(d, exist_ok=True)
        lg = {k: (x[:, i].astype(np.float32) @ sd[f"{k}_classifier.weight"].numpy().T.astype(np.float32)
                  + sd[f"{k}_classifier.bias"].numpy().astype(np.float32)).astype(np.float16) for i, k in enumerate(TASKS)}
        np.savez(os.path.join(d, "chunk_00000.npz"), clip_id=ids, pooled=x, **lg)
    return dict(work_dir=work, train_csv=os.path.join(root, "train.csv"), probe_ckpt=os.path.join(root, "probe.pt"),
                train_feats=os.path.join(root, "feats_train", "action_start"),
                val_feats=os.path.join(root, "feats_val", "action_start"))


# ----------------------------------------------------------------------------------------------- main
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--work_dir")
    p.add_argument("--train_csv", help="full EPIC_100_train.csv (prior for the post-hoc re-scoring)")
    p.add_argument("--train_feats", help="eval_ek100.py --clips_csv train_clips.csv --save_pooled output, one anchor")
    p.add_argument("--val_feats", help="eval_ek100.py --save_pooled output on val, same anchor")
    p.add_argument("--probe_ckpt", default=None, help="released probe (ek100-vitl-256.pt): initial layers")
    p.add_argument("--out", required=True)
    p.add_argument("--logits_out", default=None, help="save val logits of every recipe (seed 0) here, in the "
                   "format of eval_ek100.py, for posthoc_ek100.py / analyze_posthoc.py")
    p.add_argument("--recipes", nargs="+", default=["meta", "focal", "ce", "bsm", "la", "bsm+cb", "bsm+compose"])
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--views", type=int, default=None, help="use only train views < this (default: all extracted)")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-4, help="when starting from the released layers")
    p.add_argument("--lr_random", type=float, default=1e-3, help="when starting from random layers")
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--la_tau", type=float, default=1.0)
    p.add_argument("--holdout", type=float, default=0.1, help="share of train participants held out to pick the epoch")
    p.add_argument("--num_folds", type=int, default=2, help="val participant folds for the post-hoc cross-fitting")
    p.add_argument("--bootstrap", type=int, default=300)
    p.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    p.add_argument("--feats_on_gpu", action="store_true", help="keep the train features on the GPU (~0.4 GB/67k clips)")
    p.add_argument("--synthetic", default=None, help="build a tiny fake dataset in this folder and run on it")
    args = p.parse_args()

    if args.synthetic:
        for k, v in make_synthetic(args.synthetic).items():
            setattr(args, k, v)
    for k in ("work_dir", "train_csv", "train_feats", "val_feats"):
        if getattr(args, k) is None:
            raise SystemExit(f"--{k} is required (or --synthetic)")
    device = torch.device(args.device)
    t_start = time.time()

    # val: logits of the released probe + pooled features, with labels, official weights, participant folds
    d = load_inputs(args.work_dir, args.train_csv, args.val_feats, args.num_folds)
    val_ids = d["lg"]["clip_id"]
    vf = load_feature_chunks(args.val_feats)
    pos = pd.Series(np.arange(len(vf["clip_id"])), index=vf["clip_id"])
    xv = vf["pooled"][pos.loc[val_ids].values]
    n_cls, a_v, a_n = d["n_cls"], d["a_v"], d["a_n"]

    x, y, tclips = load_train(args)
    log_prior = train_log_prior({k: y[k][tclips["view"].values == 0] if (tclips["view"] == 0).any() else y[k] for k in TASKS},
                                n_cls)
    parts = tclips.participant.values if "participant" in tclips else tclips.video_id.str.split("_").str[0].values
    hold = np.zeros(len(tclips), bool)
    if args.holdout > 0:
        k = max(2, int(round(1 / args.holdout)))
        hold = participant_folds(pd.DataFrame(dict(participant=parts)), k) == 0
    train_idx, hold_idx = np.flatnonzero(~hold), np.flatnonzero(hold)
    print(f"train: {len(tclips)} clips ({tclips.seg_id.nunique() if 'seg_id' in tclips else len(tclips)} segments, "
          f"{len(set(parts))} participants; {len(hold_idx)} held out to pick the epoch) | val: {len(val_ids)} clips",
          flush=True)

    init_sd = None
    if args.probe_ckpt:
        ck = torch.load(args.probe_ckpt, map_location="cpu", weights_only=False)
        sd = ck["classifiers"][0] if "classifiers" in ck else ck
        init_sd = {k.removeprefix("module."): v for k, v in sd.items() if "_classifier." in k}

    res = dict(args={k: v for k, v in vars(args).items()}, train_clips=int(len(tclips)), val_clips=int(len(val_ids)),
               recipes={})
    rng = np.random.default_rng(0)
    meta_lg = {k: d["lg"][k] for k in TASKS}
    meta_res, meta_hits = evaluate(meta_lg, d)
    if init_sd is not None:  # the stored logits must be what the released layers give on the stored pooler output
        rebuilt = predict(Heads(xv.shape[2], n_cls, a_v, a_n, init_sd=init_sd).to(device), xv, device)
        res["meta_reconstruction_top1_agreement"] = {
            k: float(100 * (rebuilt[k].argmax(1) == meta_lg[k].argmax(1)).mean()) for k in TASKS}
        print("released layers on stored pooler output, top-1 agreement with stored logits (%):",
              {k: round(v, 2) for k, v in res["meta_reconstruction_top1_agreement"].items()}, flush=True)

    for name in args.recipes:
        rec = parse_recipe(name)
        print(f"== {name}", flush=True)
        if rec is None:
            res["recipes"][name] = dict(seeds=[dict(seed=None, **meta_res)])
            if args.logits_out:
                save_logits(args.logits_out, name, args.val_feats, val_ids, meta_lg)
            continue
        runs = []
        for seed in args.seeds:
            t0 = time.time()
            model, extra = train_recipe(rec, x, y, train_idx, hold_idx, n_cls, a_v, a_n, log_prior, init_sd, args,
                                        seed, device)
            lg = predict(model, xv, device)
            r, _ = evaluate(lg, d, args.bootstrap if seed == args.seeds[0] else 0, meta_hits, rng)
            runs.append(dict(seed=seed, seconds=round(time.time() - t0, 1), **extra, **r))
            if args.logits_out and seed == args.seeds[0]:
                save_logits(args.logits_out, name, args.val_feats, val_ids, lg)
            a = r["action"]
            print(f"  seed {seed}: action raw {a['raw']['clean']:.2f} / {a['raw']['official']:.2f}   "
                  f"+posthoc {a['posthoc']['clean']:.2f} / {a['posthoc']['official']:.2f}   "
                  f"(epoch {extra['epoch_used']}, {time.time() - t0:.0f}s)", flush=True)
        res["recipes"][name] = dict(seeds=runs)
        with open(args.out, "w") as f:  # keep partial results if the session stops
            json.dump(res, f, indent=1, default=float)

    # summary: mean +- std over seeds
    def agg(name, task, kind, proto):
        v = [s[task][kind][proto] for s in res["recipes"][name]["seeds"]]
        return float(np.mean(v)), float(np.std(v))

    summary = {}
    for name in res["recipes"]:
        summary[name] = {f"{task}_{kind}_{proto}": agg(name, task, kind, proto)
                         for task in TASKS for kind in ("raw", "posthoc") for proto in ("clean", "official")}
    res["summary"] = summary
    res["seconds"] = round(time.time() - t_start, 1)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=1, default=float)

    print(f"\nmean-class R@5 on val ({len(val_ids)} clips), mean +- std over seeds; each cell clean / official;"
          f" '+posthoc' = tau, beta cross-fitted over val participants")
    print(f"{'recipe':16s} {'action':>17s} {'action +posthoc':>19s} {'verb':>15s} {'noun':>15s}  CI95 action raw - meta")
    for name, s in summary.items():
        def cell(key):
            (c, sc), (o, so) = s[f"{key}_clean"], s[f"{key}_official"]
            return f"{c:5.2f}±{sc:.2f}/{o:5.2f}"
        ci = res["recipes"][name]["seeds"][0]["action"]["raw"].get("minus_meta_clean_ci95")
        print(f"{name:16s} {cell('action_raw'):>17s} {cell('action_posthoc'):>19s} {cell('verb_raw'):>15s} "
              f"{cell('noun_raw'):>15s}  {'' if ci is None else f'[{ci[0]:+.2f}, {ci[1]:+.2f}]'}")
    print(f"done in {res['seconds'] / 60:.1f} min -> {args.out}")


if __name__ == "__main__":
    main()
