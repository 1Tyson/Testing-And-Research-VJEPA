"""Step 0b of the predictor study: retrain probes for each kind of input (cross-validation on val).

The frozen Meta probe was trained on the predictor's outputs, so it cannot tell how much REAL future
tokens would help (summarize_oracle.py). Here a fresh attentive probe is trained per input variant on the
pooled tokens saved by `eval_ek100.py --oracle --save_feats_pool P`:

  enc           encoder tokens of the observed clip only
  enc_pred      + the V-JEPA 2 predictor's tokens for the anticipated time step   (= what V-JEPA 2 does)
  enc_real      + the REAL encoder tokens of that time step                        (= a perfect predictor)
  enc_copylast  + the last observed time step repeated  (control: an extra slot without new information)
  pred, real    the future tokens alone

Val participants are split into folds (no participant in both train and test of a fold). Every clip is
predicted by the probe that did not see its participant; mean-class R@5 is computed on these out-of-fold
predictions over the whole val set, for several seeds. The released probe is reported on the same clips as a
reference (it was trained on the 67k train clips, so it is the stronger one).

    enc_real - enc_pred  : headroom of a better predictor
    enc_pred - enc       : what the current predictor adds

python scripts/cv_probe.py --work_dir work --logits_root logits_feats --anchor action_start --out cv.json
"""

import argparse
import glob
import json
import math
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from vjepa_ek100.metrics import mean_class_recall, topk_hits  # noqa: E402

TASKS = ("verb", "noun", "action")
ALL_VARIANTS = ("enc", "enc_pred", "enc_real", "enc_copylast", "pred", "real")


def load_features(dir_):
    """Pooled tokens + released-probe logits from the chunks, without holding two copies in memory."""
    files = sorted(glob.glob(os.path.join(dir_, "shard*", "chunk_*.npz")))
    if not files:
        raise SystemExit(f"no chunks under {dir_}")
    ids = [np.load(f)["clip_id"] for f in files]
    with np.load(files[0]) as z:
        if "feats_real" not in z:
            raise SystemExit(f"{files[0]} has no pooled features: run eval_ek100.py --oracle --save_feats_pool 4")
        shapes = {k: z[k].shape[1:] for k in ("feats", "feats_pred", "feats_real") + TASKS}
    all_ids = np.concatenate(ids)
    uniq, first = np.unique(all_ids, return_index=True)
    keep = np.zeros(len(all_ids), bool)
    keep[first] = True
    out = {k: np.empty((len(uniq),) + s, np.float16) for k, s in shapes.items()}
    out["clip_id"] = uniq
    pos = np.searchsorted(uniq, all_ids)
    off = 0
    for f, i in zip(files, ids):
        sel, dst = keep[off:off + len(i)], pos[off:off + len(i)]
        with np.load(f) as z:
            for k in shapes:
                out[k][dst[sel]] = z[k][sel]
        off += len(i)
    failed = set()
    for f in glob.glob(os.path.join(dir_, "shard*", "failed.json")):
        with open(f) as fh:
            failed |= set(json.load(fh))
    return out, failed


def participant_folds(clips, num_folds):
    """Greedy: biggest participants first, each to the fold with the fewest clips so far."""
    sizes = clips.groupby("participant").size().sort_values(ascending=False)
    load, fold_of = [0] * num_folds, {}
    for p, n in sizes.items():
        f = int(np.argmin(load))
        fold_of[p], load[f] = f, load[f] + n
    return clips.participant.map(fold_of).values


class Probe(nn.Module):
    """Learned position per (time slot, pooled cell) -> optional self-attention -> 3 queries cross-attend -> heads.

    Pooling removes the encoder's RoPE positions, so time slot / cell embeddings are added; the anticipated time
    step has its own slot (index T), shared by enc_pred / enc_real / enc_copylast.
    """

    def __init__(self, in_dim, num_slots, cells, num_classes, dim=512, heads=8, depth=1, drop=0.1):
        super().__init__()
        self.inp = nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, dim), nn.Dropout(drop))
        self.pos = nn.Parameter(torch.zeros(num_slots * cells, dim))
        nn.init.trunc_normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(dim, heads, 4 * dim, drop, batch_first=True, norm_first=True, activation="gelu")
        self.blocks = nn.TransformerEncoder(layer, depth, enable_nested_tensor=False) if depth else nn.Identity()
        self.kv_norm = nn.LayerNorm(dim)
        self.query = nn.Parameter(torch.randn(len(num_classes), dim) * 0.02)
        self.xattn = nn.MultiheadAttention(dim, heads, dropout=drop, batch_first=True)
        self.out_norm = nn.LayerNorm(dim)
        self.drop = nn.Dropout(drop)
        self.heads = nn.ModuleList(nn.Linear(dim, c) for c in num_classes)

    def forward(self, tok, slots):
        h = self.inp(tok) + self.pos[slots]
        h = self.kv_norm(self.blocks(h))
        q = self.query.unsqueeze(0).expand(tok.shape[0], -1, -1)
        o = self.out_norm(q + self.xattn(q, h, h, need_weights=False)[0])
        return [head(self.drop(o[:, i])) for i, head in enumerate(self.heads)]


class Tokens:
    """Builds the token sequence of a variant for a batch of clip indices (features stay on `device`)."""

    def __init__(self, data, device):
        self.enc = torch.from_numpy(data["feats"]).to(device)  # [N, T, P, D]
        self.pred = torch.from_numpy(data["feats_pred"]).to(device)  # [N, P, D]
        self.real = torch.from_numpy(data["feats_real"]).to(device)
        _, self.T, self.P, self.D = self.enc.shape

    def slots(self, variant, device):
        enc = torch.arange(self.T * self.P)
        fut = torch.arange(self.T * self.P, (self.T + 1) * self.P)
        s = {"enc": enc, "pred": fut, "real": fut}.get(variant, torch.cat([enc, fut]))
        return s.to(device)

    def __call__(self, variant, idx):
        enc = lambda: self.enc[idx].flatten(1, 2)  # noqa: E731
        fut = dict(pred=lambda: self.pred[idx], real=lambda: self.real[idx], copylast=lambda: self.enc[idx, -1])
        if variant == "enc":
            return enc()
        if variant in fut:
            return fut[variant]()
        return torch.cat([enc(), fut[variant.split("_", 1)[1]]()], 1)


def recall_sum(logits, labels, n_cls):
    return sum(mean_class_recall(topk_hits(logits[k], labels[k]), labels[k], n_cls[k])["recall"] for k in TASKS)


def train_one(tokens, variant, train_idx, test_idx, labels, n_cls, args, seed, device):
    """Train on train_idx (10% held out to pick the epoch), return test logits {task: [n_test, C]} at the best epoch."""
    g = np.random.default_rng(seed)
    torch.manual_seed(seed)
    perm = g.permutation(train_idx)
    n_hold = max(1, int(len(perm) * args.holdout))
    hold, fit = perm[:n_hold], perm[n_hold:]
    model = Probe(tokens.D, tokens.T + 1, tokens.P, [n_cls[k] for k in TASKS], args.dim, args.heads, args.depth,
                  args.dropout).to(device)
    slots = tokens.slots(variant, device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps_per_epoch = math.ceil(len(fit) / args.batch_size)
    total, warm = args.epochs * steps_per_epoch, steps_per_epoch
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / total))))
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")
    lab_t = {k: torch.from_numpy(labels[k]).to(device) for k in TASKS}

    def predict(idx):
        model.eval()
        outs = {k: [] for k in TASKS}
        with torch.inference_mode(), torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            for i in range(0, len(idx), 512):
                b = torch.from_numpy(idx[i:i + 512]).to(device)
                for k, o in zip(TASKS, model(tokens(variant, b).float(), slots)):
                    outs[k].append(o.float().cpu().numpy())
        return {k: np.concatenate(v) for k, v in outs.items()}

    best, best_state, best_epoch = -1.0, None, -1
    for epoch in range(args.epochs):
        model.train()
        order = torch.from_numpy(g.permutation(fit)).to(device)
        for i in range(0, len(order), args.batch_size):
            b = order[i:i + args.batch_size]
            with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                outs = model(tokens(variant, b).float(), slots)
                loss = sum(F.cross_entropy(o.float(), lab_t[k][b], label_smoothing=args.label_smoothing)
                           for k, o in zip(TASKS, outs))
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
        score = recall_sum(predict(hold), {k: labels[k][hold] for k in TASKS}, n_cls)
        if score > best:
            best, best_epoch = score, epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return predict(test_idx), best_epoch


def scores(logits, labels, n_cls):
    return {k: mean_class_recall(topk_hits(logits[k], labels[k]), labels[k], n_cls[k])["recall"] for k in TASKS}


def bootstrap_diff(logits_a, logits_b, labels, n_cls, reps, rng):
    """95% interval of action recall(a) - recall(b), resampling clips (paired)."""
    lab = labels["action"]
    ha, hb = topk_hits(logits_a["action"], lab), topk_hits(logits_b["action"], lab)
    d = []
    for _ in range(reps):
        i = rng.integers(0, len(lab), len(lab))
        d.append(mean_class_recall(ha[i], lab[i], n_cls["action"])["recall"]
                 - mean_class_recall(hb[i], lab[i], n_cls["action"])["recall"])
    return [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--work_dir", required=True)
    p.add_argument("--logits_root", required=True, help="--out_dir of eval_ek100.py --oracle --save_feats_pool")
    p.add_argument("--anchor", default="action_start")
    p.add_argument("--out", required=True)
    p.add_argument("--variants", nargs="+", default=list(ALL_VARIANTS), choices=ALL_VARIANTS)
    p.add_argument("--num_folds", type=int, default=2)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight_decay", type=float, default=0.05)
    p.add_argument("--label_smoothing", type=float, default=0.1)
    p.add_argument("--dim", type=int, default=512)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--depth", type=int, default=1, help="self-attention blocks before the query cross-attention")
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--holdout", type=float, default=0.1, help="share of each training fold used to pick the epoch")
    p.add_argument("--bootstrap", type=int, default=1000)
    p.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    p.add_argument("--features_on_cpu", action="store_true", help="if the pooled tokens do not fit in GPU memory")
    args = p.parse_args()
    device = torch.device(args.device)

    clips = pd.read_csv(os.path.join(args.work_dir, "val_clips.csv")).set_index("clip_id")
    with open(os.path.join(args.work_dir, "classes.json")) as f:
        classes = json.load(f)
    n_cls = dict(verb=len(classes["verbs"]), noun=len(classes["nouns"]), action=len(classes["actions"]))

    t0 = time.time()
    data, failed = load_features(os.path.join(args.logits_root, args.anchor))
    col = f"expected_fail_{args.anchor}"
    skip = failed | (set(clips.index[clips[col]]) if col in clips else set())
    finite = np.isfinite(data["feats"].reshape(len(data["clip_id"]), -1)).all(1)
    finite &= np.isfinite(data["feats_pred"].reshape(len(finite), -1)).all(1)
    finite &= np.isfinite(data["feats_real"].reshape(len(finite), -1)).all(1)
    ok = np.isin(data["clip_id"], clips.index) & ~np.isin(data["clip_id"], list(skip)) & finite
    data = {k: v[ok] for k, v in data.items()}
    ids = data["clip_id"]
    c = clips.loc[ids]
    c = c.assign(participant=c.video_id.str.split("_").str[0])
    labels = {k: c[f"{k}_idx"].values.astype(np.int64) for k in TASKS}
    if c.participant.nunique() < args.num_folds:
        raise SystemExit(f"only {c.participant.nunique()} participants: cannot make {args.num_folds} folds")
    folds = participant_folds(c, args.num_folds)
    print(f"{len(ids)} clips ({(~finite).sum()} with non-finite features dropped), "
          f"tokens {data['feats'].shape[1:]}, folds {np.bincount(folds).tolist()} clips, "
          f"loaded in {time.time() - t0:.0f}s", flush=True)

    released = {k: data[k].astype(np.float32) for k in TASKS}
    res = dict(anchor=args.anchor, clips=int(len(ids)), folds=np.bincount(folds).tolist(),
               participants_per_fold=[sorted(c.participant[folds == f].unique().tolist()) for f in range(args.num_folds)],
               args=vars(args), released_probe=scores(released, labels, n_cls), runs={}, mean={}, std={})
    tokens = Tokens(data, torch.device("cpu") if args.features_on_cpu else device)
    for k in ("feats", "feats_pred", "feats_real"):
        del data[k]

    ens = {}
    for v in args.variants:
        per_seed = []
        prob_sum = {k: np.zeros((len(ids), n_cls[k]), np.float32) for k in TASKS}
        for seed in args.seeds:
            oof = {k: np.zeros((len(ids), n_cls[k]), np.float32) for k in TASKS}
            epochs = []
            for f in range(args.num_folds):
                tr, te = np.where(folds != f)[0], np.where(folds == f)[0]
                out, ep = train_one(tokens, v, tr, te, labels, n_cls, args, seed * 100 + f, device)
                for k in TASKS:
                    oof[k][te] = out[k]
                epochs.append(ep)
            s = scores(oof, labels, n_cls)
            per_seed.append(dict(seed=seed, best_epochs=epochs, **s))
            for k in TASKS:
                prob_sum[k] += torch.softmax(torch.from_numpy(oof[k]), 1).numpy()
            print(f"[{v:12s} seed {seed}] action {s['action']:6.2f}  verb {s['verb']:6.2f}  noun {s['noun']:6.2f}  "
                  f"(best epochs {epochs}, {(time.time() - t0) / 60:.0f} min)", flush=True)
        ens[v] = prob_sum
        res["runs"][v] = per_seed
        res["mean"][v] = {k: float(np.mean([r[k] for r in per_seed])) for k in TASKS}
        res["std"][v] = {k: float(np.std([r[k] for r in per_seed])) for k in TASKS}
        res.setdefault("seed_ensemble", {})[v] = scores(prob_sum, labels, n_cls)
        with open(args.out, "w") as fh:
            json.dump(res, fh, indent=2)

    rng = np.random.default_rng(0)
    pairs = [("enc_real", "enc_pred"), ("enc_pred", "enc"), ("enc_pred", "enc_copylast"), ("real", "pred")]
    res["action_diff"] = {}
    for a, b in pairs:
        if a in ens and b in ens:
            per_seed = [ra["action"] - rb["action"] for ra, rb in zip(res["runs"][a], res["runs"][b])]
            res["action_diff"][f"{a} - {b}"] = dict(
                mean=float(np.mean(per_seed)), per_seed=per_seed,
                seed_ensemble_ci95=bootstrap_diff(ens[a], ens[b], labels, n_cls, args.bootstrap, rng))
    with open(args.out, "w") as fh:
        json.dump(res, fh, indent=2)

    print(f"\n{len(ids)} clips, anchor={args.anchor}, {args.num_folds}-fold by participant, seeds {args.seeds} "
          f"(mean-class R@5, out-of-fold, mean +- std over seeds)")
    print(f"{'':14s} {'action':>14s} {'verb':>14s} {'noun':>14s}")
    r = res["released_probe"]
    print(f"{'released*':14s} {r['action']:14.2f} {r['verb']:14.2f} {r['noun']:14.2f}")
    for v in args.variants:
        m, s = res["mean"][v], res["std"][v]
        print(f"{v:14s} " + " ".join(f"{m[k]:8.2f} +-{s[k]:4.2f}" for k in TASKS))
    print("* Meta's probe trained on the 67k train clips (reference, not comparable in training data)")
    for name, d in res["action_diff"].items():
        lo, hi = d["seed_ensemble_ci95"]
        print(f"action {name:26s} {d['mean']:+6.2f}  (seed ensemble 95% CI [{lo:+.2f}, {hi:+.2f}])")


if __name__ == "__main__":
    main()
