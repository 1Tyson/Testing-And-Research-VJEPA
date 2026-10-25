"""Re-train the EK100 probe (heads only, or attentive pooler + heads) on stored full V-JEPA 2 encoder tokens.

Input: datasets written by the token-extraction notebooks (e.g. ek100-tok-train-p0 ... p5, ek100-tok-val-sub). Each
one is a folder <root> with
    <root>/work/val_clips.csv, <root>/work/classes.json     clip index (clip_id -> narration_id, labels) of THAT dataset
    <root>/<anchor>/shard*/chunk_*.npz                      clip_id, verb/noun/action logits of the released probe,
                                                            ctx_q [n, 4096, 1024] int8 + ctx_s [n, 4096] (encoder
                                                            tokens, one scale per token), real_q / real_s (true future)
clip_id restarts at 0 in every dataset, so everything is keyed by narration_id.

Subcommands
  index   one table of every stored clip: dataset, anchor, file, row, narration_id, split (train / val from the
          official csvs), labels in the class order of the released probe, participant, holdout flag. Checks that
          every dataset uses the released probe's class order, drops repeated narration ids, and that no val
          segment is in train.
  verify  on a few clips: dequantises the tokens (q * s or q * s / 127), runs the frozen predictor + released probe
          and compares with the stored logits and cosines; measures how fast chunks are read.
  train   for one anchor: (1) one pass with the frozen released pooler caches its output for train / holdout / val;
          (2) "heads:<recipe>" re-trains the verb / noun / action layers on that cache (fast);
          (3) "full:<recipe>" re-trains pooler + layers from the released probe, streaming tokens from disk.
          Recipes: focal (Meta's loss) | ce | bsm (Balanced Softmax) | la, plus "+compose" (learned verb-noun
          composition of the action logit). The epoch is picked on held-out TRAIN participants; val (here a subset)
          is only scored: mean-class R@5 (clean) and a bootstrap interval of (recipe - released) for action.

python scripts/tok_probe.py index  --root /kaggle/input --train_csv EPIC_100_train.csv --val_csv EPIC_100_validation.csv --out_dir tok
python scripts/tok_probe.py verify --index tok/index.csv --vitl_ckpt vitl.pt --probe_ckpt ek100-vitl-256.pt
python scripts/tok_probe.py train  --index tok/index.csv --anchor action_start --recipes meta heads:bsm+compose full:bsm+compose \
    --vitl_ckpt vitl.pt --probe_ckpt ek100-vitl-256.pt --out_dir tok
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
from vjepa_ek100.metrics import mean_class_recall, topk_hits  # noqa: E402
from vjepa_ek100.protocol import filter_annotations  # noqa: E402

TASKS = ("verb", "noun", "action")  # order of the pooler queries in AttentiveClassifier
ANCHORS = ("action_start", "official")
LOSSES = ("focal", "ce", "bsm", "la")


# =============================================================================================== index
def reference_classes(train_csv, val_csv):
    tdf, vdf = pd.read_csv(train_csv), pd.read_csv(val_csv)
    v, n, a, _ = filter_annotations(tdf, vdf)
    return {int(k): i for k, i in v.items()}, {int(k): i for k, i in n.items()}, {(int(x), int(y)): i for (x, y), i in a.items()}


def dataset_roots(root):
    return sorted({os.path.dirname(os.path.dirname(p)) for p in glob.glob(os.path.join(root, "**", "work", "val_clips.csv"),
                                                                            recursive=True)})


def holdout_participants(c, share):
    """Smallest set of whole participants (largest first, then alternating) holding ~share of the train clips."""
    sizes = c.groupby("participant").size().sort_values(ascending=False)
    target, picked, tot = share * len(c), [], 0
    for p, n in sizes.iloc[::-1].items():  # small participants first: keeps the holdout spread out
        if tot >= target:
            break
        picked.append(p)
        tot += n
    return set(picked)


def cmd_index(args):
    rv, rn, ra = reference_classes(args.train_csv, args.val_csv)
    train_ids = set(pd.read_csv(args.train_csv).narration_id)
    val_ids = set(pd.read_csv(args.val_csv).narration_id)
    rows, report = [], dict(datasets={})
    for root in dataset_roots(args.root):
        name = os.path.relpath(root, args.root)
        with open(os.path.join(root, "work", "classes.json")) as f:
            cj = json.load(f)
        same = ({int(k): i for k, i in cj["verbs"]} == rv and {int(k): i for k, i in cj["nouns"]} == rn
                and {(int(v), int(n)): i for v, n, i in cj["actions"]} == ra)
        if not same:
            raise SystemExit(f"{name}: classes.json differs from the released probe's class order "
                             "(built from another train csv): its logits / labels cannot be used as they are")
        clips = pd.read_csv(os.path.join(root, "work", "val_clips.csv")).set_index("clip_id")
        info = dict(clips_in_index=len(clips))
        for anchor in ANCHORS:
            files = sorted(glob.glob(os.path.join(root, anchor, "shard*", "chunk_*.npz")))
            n = 0
            for f in files:
                with np.load(f) as z:
                    ids = z["clip_id"]
                for r, cid in enumerate(ids.tolist()):
                    c = clips.loc[cid]
                    rows.append(dict(dataset=name, anchor=anchor, file=f, row=r, clip_id=cid, narration_id=c.narration_id,
                                     video_id=c.video_id, verb_idx=int(c.verb_idx), noun_idx=int(c.noun_idx),
                                     action_idx=int(c.action_idx)))
                n += len(ids)
            if files:
                info[anchor] = n
        report["datasets"][name] = info
    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit(f"no token datasets (work/val_clips.csv + <anchor>/shard*/chunk_*.npz) under {args.root}")
    df["split"] = np.where(df.narration_id.isin(val_ids), "val", np.where(df.narration_id.isin(train_ids), "train", "?"))
    df["participant"] = df.video_id.str.split("_").str[0]
    report["unknown_split"] = int((df.split == "?").sum())
    before = len(df)
    df = df[df.split != "?"].drop_duplicates(["split", "anchor", "narration_id"], keep="first")
    report["repeated_narration_ids_dropped"] = before - report["unknown_split"] - len(df)
    leak = set(df[df.split == "train"].narration_id) & set(df[df.split == "val"].narration_id)
    report["train_val_overlap"] = len(leak)
    if leak:
        df = df[~((df.split == "train") & df.narration_id.isin(leak))]
    tr = df[df.split == "train"]
    hold = holdout_participants(tr.drop_duplicates("narration_id"), args.holdout)
    df["holdout"] = (df.split == "train") & df.participant.isin(hold)
    report["holdout_participants"] = sorted(hold)
    report["counts"] = {f"{s}/{a}": int(((df.split == s) & (df.anchor == a)).sum()) for s in ("train", "val") for a in ANCHORS}
    report["counts"].update({f"holdout/{a}": int((df.holdout & (df.anchor == a)).sum()) for a in ANCHORS})
    report["train_participants"] = sorted(tr.participant.unique())
    report["val_participants"] = sorted(df[df.split == "val"].participant.unique())
    os.makedirs(args.out_dir, exist_ok=True)
    df.to_csv(os.path.join(args.out_dir, "index.csv"), index=False)
    with open(os.path.join(args.out_dir, "index_report.json"), "w") as f:
        json.dump(report, f, indent=1)
    print(json.dumps({k: v for k, v in report.items() if k != "datasets"}, indent=1))
    for k, v in report["datasets"].items():
        print(f"  {k}: {v}")


# =============================================================================================== model
def dequant(q, s, mode):
    x = q.float() * s.float().unsqueeze(-1)
    return x / 127.0 if mode == "qs127" else x


def build_models(args, device, n_cls):
    """(predict_fn, released AttentiveClassifier). --fake: tiny random models for the self-test."""
    import yaml
    from vjepa_ek100.model import add_vjepa2_to_path, build_backbone, build_probe

    add_vjepa2_to_path(args.vjepa2_root)
    if args.fake:
        from evals.action_anticipation_frozen.models import AttentiveClassifier

        probe = AttentiveClassifier(range(n_cls["verb"]), range(n_cls["noun"]), range(n_cls["action"]), embed_dim=args.fake_dim,
                                    num_heads=4, depth=2, use_activation_checkpointing=False)
        if args.probe_ckpt and os.path.exists(args.probe_ckpt):
            probe.load_state_dict(torch.load(args.probe_ckpt, map_location="cpu"))
        return (lambda ctx: 0.5 * ctx[:, -4:]), probe.to(device).eval()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    model = build_backbone(cfg, args.vitl_ckpt, device)
    model.encoder.to("cpu")  # only the predictor is needed: tokens are stored
    from eval_ek100 import predict_future

    ats_cache = {}

    def predict(ctx):
        b = ctx.shape[0]
        if b not in ats_cache:
            ats_cache[b] = torch.full((b,), 1.0, device=ctx.device)
        return predict_future(model, ctx, ats_cache[b])

    probe = build_probe(cfg, n_cls["verb"], n_cls["noun"], n_cls["action"], model.embed_dim, device, args.probe_ckpt)
    return predict, probe


class Composer(nn.Module):
    """Optional learned verb-noun composition on top of the probe's three logit vectors."""

    def __init__(self, a_v, a_n, enabled):
        super().__init__()
        self.enabled = enabled
        self.beta = nn.Parameter(torch.tensor([0.5, 0.5]), requires_grad=enabled)
        self.register_buffer("a_v", torch.as_tensor(a_v, dtype=torch.long))
        self.register_buffer("a_n", torch.as_tensor(a_n, dtype=torch.long))

    def forward(self, z):
        if not self.enabled:
            return z
        lv, ln = F.log_softmax(z["verb"].float(), 1), F.log_softmax(z["noun"].float(), 1)
        return dict(z, action=z["action"].float() + self.beta[0] * lv[:, self.a_v] + self.beta[1] * ln[:, self.a_n])


def sigmoid_focal(z, y, alpha=0.25, gamma=2.0):
    t = F.one_hot(y, z.shape[1]).float()
    p = torch.sigmoid(z)
    ce = F.binary_cross_entropy_with_logits(z, t, reduction="none")
    loss = ce * (1 - (p * t + (1 - p) * (1 - t))) ** gamma
    return ((alpha * t + (1 - alpha) * (1 - t)) * loss).sum(1).mean()


def task_loss(kind, z, y, log_prior, la_tau):
    z = z.float()
    if kind == "focal":
        return sigmoid_focal(z, y)
    if kind == "ce":
        return F.cross_entropy(z, y)
    if kind == "bsm":
        return F.cross_entropy(z + log_prior, y)
    return F.cross_entropy(z + la_tau * log_prior, y)


def parse_recipe(r):
    if r == "meta":
        return None
    scope, _, rest = r.partition(":")
    tok = rest.split("+")
    if scope not in ("heads", "full") or len(tok) < 1 or tok[0] not in LOSSES or any(t != "compose" for t in tok[1:]):
        raise SystemExit(f"bad recipe {r!r}: meta | heads:<loss>[+compose] | full:<loss>[+compose], loss in {LOSSES}")
    return dict(scope=scope, loss=tok[0], compose="compose" in tok)


# =============================================================================================== data
class TokenStream(torch.utils.data.IterableDataset):
    """Yields (ctx_q int8 [T, D], ctx_s [T], position in the table) reading each chunk file once per pass.

    Files are split over DataLoader workers; with shuffle, file order and rows inside a buffer are shuffled.
    """

    def __init__(self, table, shuffle=False, seed=0, buffer=64):
        self.groups = [(f, g["row"].values, g["pos"].values) for f, g in table.groupby("file", sort=True)]
        self.shuffle, self.seed, self.buffer, self.epoch = shuffle, seed, buffer, 0

    def __iter__(self):
        wi = torch.utils.data.get_worker_info()
        groups = self.groups if wi is None else self.groups[wi.id::wi.num_workers]
        rng = np.random.default_rng(self.seed + 1000 * self.epoch + (wi.id if wi else 0))
        if self.shuffle:
            groups = [groups[i] for i in rng.permutation(len(groups))]
        buf = []
        for f, rows, pos in groups:
            with np.load(f) as z:
                q, s = z["ctx_q"], z["ctx_s"]
            for r, p in zip(rows, pos):
                buf.append((torch.from_numpy(np.ascontiguousarray(q[r])), torch.from_numpy(s[r].astype(np.float32)), int(p)))
            if not self.shuffle:
                yield from buf
                buf = []
            elif len(buf) >= self.buffer:
                for i in rng.permutation(len(buf)):
                    yield buf[i]
                buf = []
        order = rng.permutation(len(buf)) if self.shuffle else range(len(buf))
        for i in order:
            yield buf[i]


def loader(table, args, shuffle=False, seed=0):
    ds = TokenStream(table, shuffle=shuffle, seed=seed, buffer=args.shuffle_buffer)
    return ds, torch.utils.data.DataLoader(ds, batch_size=args.batch_size, num_workers=args.num_workers,
                                           persistent_workers=False, pin_memory=torch.cuda.is_available())


def probe_input(predict, q, s, mode, device):
    ctx = dequant(q.to(device, non_blocking=True), s.to(device, non_blocking=True), mode)
    ctx = ctx.half() if device.type == "cuda" else ctx
    with torch.no_grad(), torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
        pred = predict(ctx)
    return torch.cat([ctx, pred.to(ctx.dtype)], 1)


# =============================================================================================== verify
def cmd_verify(args):
    df = pd.read_csv(args.index)
    device = torch.device(args.device)
    n_cls = dict(verb=args.n_verb, noun=args.n_noun, action=args.n_action)
    predict, probe = build_models(args, device, n_cls)
    res = dict(read_speed={}, checks=[])
    for (split, anchor), g in df.groupby(["split", "anchor"]):
        f = g.file.iloc[0]
        t0 = time.time()
        with np.load(f) as z:
            arr = {k: z[k] for k in z.files}
        dt = time.time() - t0
        mb = sum(v.nbytes for v in arr.values()) / 1e6
        res["read_speed"][f"{split}/{anchor}"] = dict(mb=round(mb), seconds=round(dt, 2), mb_per_s=round(mb / max(dt, 1e-6)))
        rows = g[g.file == f].row.values[: args.n]
        q, s = torch.from_numpy(arr["ctx_q"][rows]), torch.from_numpy(arr["ctx_s"][rows].astype(np.float32))
        rq, rs = torch.from_numpy(arr["real_q"][rows]), torch.from_numpy(arr["real_s"][rows].astype(np.float32))
        for mode in ("qs", "qs127"):
            x = probe_input(predict, q, s, mode, device)
            with torch.no_grad(), torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                out = probe(x)
            pred = x[:, -rq.shape[1]:].float()
            real = dequant(rq.to(device), rs.to(device), mode)
            cos = F.cosine_similarity(pred, real, dim=-1).mean(1).cpu().numpy()
            chk = dict(split=split, anchor=anchor, mode=mode, clips=len(rows),
                       cos_pred_recomputed=float(cos.mean()), cos_pred_stored=float(arr["cos_pred"][rows].mean()))
            for k in TASKS:
                a, b = out[k].float().cpu().numpy(), arr[k][rows].astype(np.float32)
                chk[f"{k}_top1_agree"] = float((a.argmax(1) == b.argmax(1)).mean())
                chk[f"{k}_max_abs_diff"] = float(np.abs(a - b).max())
            res["checks"].append(chk)
            print(chk, flush=True)
    score = {m: np.mean([c["action_top1_agree"] - c["action_max_abs_diff"] / 100 for c in res["checks"] if c["mode"] == m])
             for m in ("qs", "qs127")}
    res["dequant"] = max(score, key=score.get)
    ok = all(c["action_top1_agree"] >= 0.75 for c in res["checks"] if c["mode"] == res["dequant"])
    res["ok"] = bool(ok)
    with open(os.path.join(os.path.dirname(args.index), "verify.json"), "w") as f:
        json.dump(res, f, indent=1)
    print("read speed:", res["read_speed"])
    print(f"dequantisation: {res['dequant']}  ->  {'OK: stored logits reproduced' if ok else 'MISMATCH: do not train'}")
    if not ok:
        sys.exit(2)


# =============================================================================================== train
def log_prior(y, n):
    cnt = np.bincount(y, minlength=n).astype(np.float64) + 1.0
    return np.log(cnt / cnt.sum()).astype(np.float32)


def score(lg, y, n_cls, ref_hits=None, boot=0, seed=0):
    out = {}
    hits = {k: topk_hits(lg[k], y[k]) for k in TASKS}
    for k in TASKS:
        out[k] = mean_class_recall(hits[k], y[k], n_cls[k])["recall"]
    if ref_hits is not None and boot:
        rng, d, lab = np.random.default_rng(seed), [], y["action"]
        for _ in range(boot):
            i = rng.integers(0, len(lab), len(lab))
            d.append(mean_class_recall(hits["action"][i], lab[i], n_cls["action"])["recall"]
                     - mean_class_recall(ref_hits[i], lab[i], n_cls["action"])["recall"])
        out["action_minus_meta_ci95"] = [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]
    return out, hits


def stored_logits(table):
    """verb / noun / action logits written at extraction, aligned with the rows of `table`."""
    out = {k: [None] * len(table) for k in TASKS}
    tab = table.reset_index(drop=True)
    for f, g in tab.groupby("file"):
        with np.load(f) as z:
            arr = {k: z[k] for k in TASKS}
        for i, r in zip(g.index, g.row):
            for k in TASKS:
                out[k][i] = arr[k][r].astype(np.float32)
    return {k: np.stack(v) for k, v in out.items()}


@torch.no_grad()
def run_frozen(table, predict, probe, mode, args, device, want_pooled=True):
    """Released pooler output and logits for every row of `table` (one streaming pass)."""
    pooled = np.zeros((len(table), 3, probe.pooler.query_tokens.shape[-1]), np.float16)
    logits = {k: np.zeros((len(table), getattr(probe, f"{k}_classifier").out_features), np.float32) for k in TASKS}
    tab = table.reset_index(drop=True).assign(pos=np.arange(len(table)))
    _, dl = loader(tab, args)
    t0, n = time.time(), 0
    for q, s, pos in dl:
        x = probe_input(predict, q, s, mode, device)
        with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            p = probe.pooler(x)
        p = p.float()
        pooled[pos.numpy()] = p.cpu().numpy().astype(np.float16)
        for i, k in enumerate(TASKS):
            logits[k][pos.numpy()] = getattr(probe, f"{k}_classifier")(p[:, i]).cpu().numpy()
        n += len(pos)
        if n % (args.batch_size * 50) < args.batch_size:
            print(f"    frozen pass {n}/{len(tab)}  {n / (time.time() - t0):.1f} clip/s", flush=True)
    return pooled, logits


class Heads(nn.Module):
    def __init__(self, probe, a_v, a_n, compose):
        super().__init__()
        self.lin = nn.ModuleDict({k: copy.deepcopy(getattr(probe, f"{k}_classifier")).float() for k in TASKS})
        self.comp = Composer(a_v, a_n, compose)

    def forward(self, p):
        return self.comp({k: self.lin[k](p[:, i]) for i, k in enumerate(TASKS)})


def train_heads(rec, pooled, y, tr, ho, probe, a_v, a_n, priors, n_cls, args, seed, device):
    torch.manual_seed(seed)
    m = Heads(probe, a_v, a_n, rec["compose"]).to(device)
    opt = torch.optim.AdamW(m.parameters(), lr=args.lr_heads, weight_decay=args.weight_decay)
    X = torch.as_tensor(pooled[tr], device=device).float()
    Y = {k: torch.as_tensor(y[k][tr], device=device) for k in TASKS}
    lp = {k: torch.as_tensor(priors[k], device=device) for k in TASKS}
    Xh = torch.as_tensor(pooled[ho], device=device).float()
    steps = args.head_epochs * ((len(tr) + 255) // 256)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 0.5 * (1 + np.cos(np.pi * min(s, steps) / steps)))
    g = torch.Generator().manual_seed(seed)
    best, state, hist = -1, None, []
    for ep in range(args.head_epochs):
        m.train()
        for b in torch.randperm(len(tr), generator=g).split(256):
            b = b.to(device)
            z = m(X[b])
            loss = sum(task_loss(rec["loss"], z[k], Y[k][b], lp[k], args.la_tau) for k in TASKS)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
        m.eval()
        with torch.no_grad():
            zh = m(Xh)
        h = mean_class_recall(topk_hits(zh["action"].float().cpu().numpy(), y["action"][ho]), y["action"][ho], n_cls["action"])["recall"]
        hist.append(round(h, 3))
        if h > best:
            best, state = h, copy.deepcopy(m.state_dict())
    m.load_state_dict(state)
    return m, dict(holdout_curve=hist, epoch_used=int(np.argmax(hist)) + 1)


@torch.no_grad()
def heads_logits(m, pooled, device):
    m.eval()
    z = m(torch.as_tensor(pooled, device=device).float())
    return {k: v.float().cpu().numpy() for k, v in z.items()}


def train_full(rec, table, y_all, tr_tab, ho_tab, predict, probe0, a_v, a_n, priors, n_cls, mode, args, seed, device, t_end):
    """Pooler + layers from the released probe; streams tokens; returns the best (holdout) model."""
    from evals.action_anticipation_frozen.models import AttentiveClassifier

    torch.manual_seed(seed)
    probe = AttentiveClassifier(range(n_cls["verb"]), range(n_cls["noun"]), range(n_cls["action"]),
                                embed_dim=probe0.pooler.query_tokens.shape[-1],
                                num_heads=probe0.pooler.cross_attention_block.xattn.num_heads,
                                depth=1 + (len(probe0.pooler.blocks) if probe0.pooler.blocks is not None else 0),
                                use_activation_checkpointing=True).to(device)
    probe.load_state_dict(probe0.state_dict())
    comp = Composer(a_v, a_n, rec["compose"]).to(device)
    head_params = [p for n_, p in probe.named_parameters() if "classifier" in n_] + list(comp.parameters())
    pool_params = [p for n_, p in probe.named_parameters() if "classifier" not in n_]
    opt = torch.optim.AdamW([dict(params=pool_params, lr=args.lr_pooler), dict(params=head_params, lr=args.lr_heads)],
                            weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")
    lp = {k: torch.as_tensor(priors[k], device=device) for k in TASKS}
    Y = {k: torch.as_tensor(y_all[k], device=device) for k in TASKS}
    steps = args.full_epochs * ((len(tr_tab) + args.batch_size - 1) // args.batch_size) // args.accum
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 0.5 * (1 + np.cos(np.pi * min(s, steps) / max(steps, 1))))
    ds, dl = loader(tr_tab, args, shuffle=True, seed=seed)
    best, state, hist = -1, None, []
    for ep in range(args.full_epochs):
        if time.time() > t_end:
            print(f"    [time budget] stopping full training before epoch {ep + 1}", flush=True)
            break
        ds.epoch = ep
        probe.train()
        t0, n, tot = time.time(), 0, 0.0
        opt.zero_grad(set_to_none=True)
        for it, (q, s, pos) in enumerate(dl):
            x = probe_input(predict, q, s, mode, device)
            pos = pos.to(device)
            with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                z = comp(probe(x))
            loss = sum(task_loss(rec["loss"], z[k], Y[k][pos], lp[k], args.la_tau) for k in TASKS) / args.accum
            scaler.scale(loss).backward()
            if (it + 1) % args.accum == 0:
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(probe.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                sched.step()
            n += len(pos)
            tot += loss.item() * args.accum * len(pos)
            if n % (args.batch_size * 100) < args.batch_size:
                print(f"    epoch {ep + 1} {n}/{len(tr_tab)}  loss {tot / n:.3f}  {n / (time.time() - t0):.1f} clip/s", flush=True)
        probe.eval()
        _, lg = run_frozen(ho_tab, predict, probe, mode, args, device)
        lg = comp_logits(comp, lg, device)
        yh = {k: y_all[k][ho_tab.pos.values] for k in TASKS}
        h = mean_class_recall(topk_hits(lg["action"], yh["action"]), yh["action"], n_cls["action"])["recall"]
        hist.append(dict(epoch=ep + 1, loss=tot / max(n, 1), holdout_action=h, minutes=(time.time() - t0) / 60))
        print(f"    epoch {ep + 1}: loss {tot / max(n, 1):.3f}  holdout action R@5 {h:.2f}  ({hist[-1]['minutes']:.1f} min)", flush=True)
        if h > best:
            best, state = h, (copy.deepcopy(probe.state_dict()), copy.deepcopy(comp.state_dict()))
    if state is None:
        return None, None, dict(history=hist)
    probe.load_state_dict(state[0])
    comp.load_state_dict(state[1])
    return probe.eval(), comp.eval(), dict(history=hist, epoch_used=max(hist, key=lambda h: h["holdout_action"])["epoch"])


@torch.no_grad()
def comp_logits(comp, lg, device):
    z = comp({k: torch.as_tensor(v, device=device) for k, v in lg.items()})
    return {k: v.float().cpu().numpy() for k, v in z.items()}


def cmd_train(args):
    t_end = time.time() + args.max_hours * 3600
    df = pd.read_csv(args.index)
    df = df[df.anchor == args.anchor].reset_index(drop=True)
    vpath = os.path.join(os.path.dirname(args.index), "verify.json")
    mode = args.dequant or (json.load(open(vpath))["dequant"] if os.path.exists(vpath) else "qs")
    device = torch.device(args.device)
    n_cls = dict(verb=args.n_verb, noun=args.n_noun, action=args.n_action)
    predict, probe = build_models(args, device, n_cls)
    a_v = np.zeros(n_cls["action"], np.int64)
    a_n = np.zeros(n_cls["action"], np.int64)
    if not args.fake:
        rv, rn, ra = reference_classes(args.train_csv, args.val_csv)
        for (v, n), i in ra.items():
            a_v[i], a_n[i] = rv[v], rn[n]
    else:
        a_v, a_n = np.asarray(args.fake_av), np.asarray(args.fake_an)
    df["pos"] = np.arange(len(df))
    y = {k: df[f"{k}_idx"].values.astype(np.int64) for k in TASKS}
    tr = np.flatnonzero((df.split == "train") & ~df.holdout)
    ho = np.flatnonzero(df.holdout.values)
    va = np.flatnonzero(df.split == "val")
    print(f"[{args.anchor}] train {len(tr)}  holdout {len(ho)}  val {len(va)}  (dequant {mode})", flush=True)
    priors = {k: log_prior(y[k][tr], n_cls[k]) for k in TASKS}

    out_dir = os.path.join(args.out_dir, args.anchor)
    os.makedirs(out_dir, exist_ok=True)
    cache = os.path.join(out_dir, "frozen_pass.npz")
    if os.path.exists(cache):
        z = np.load(cache)
        pooled, meta_lg = z["pooled"], {k: z[k] for k in TASKS}
        print("    frozen pass loaded from", cache)
    else:
        pooled, meta_lg = run_frozen(df, predict, probe, mode, args, device)
        np.savez(cache, pooled=pooled, **meta_lg)
    yv = {k: y[k][va] for k in TASKS}
    res_path = os.path.join(out_dir, "results.json")
    res = json.load(open(res_path)) if os.path.exists(res_path) else {}
    meta_score, meta_hits = score({k: meta_lg[k][va] for k in TASKS}, yv, n_cls)
    stored = stored_logits(df.iloc[va])  # consistency: recomputed released logits vs those stored at extraction
    agree = {k: float(np.mean(meta_lg[k][va].argmax(1) == stored[k].argmax(1))) for k in TASKS}
    stored_score, _ = score(stored, yv, n_cls)
    res["meta"] = dict(val=meta_score, n_val=len(va), stored_logits_val=stored_score, top1_agree_with_stored=agree)
    print(f"    recomputed vs stored released logits, top-1 agreement on val: {agree}", flush=True)
    print(f"    meta (released probe) on val: {meta_score}", flush=True)
    for r in args.recipes:
        rec = parse_recipe(r)
        if rec is None or (r in res and not args.redo):
            continue
        runs = []
        for seed in (args.seeds if rec["scope"] == "heads" else args.seeds[:1]):
            if time.time() > t_end:
                print(f"    [time budget] skipping {r} seed {seed}")
                break
            t0 = time.time()
            if rec["scope"] == "heads":
                m, extra = train_heads(rec, pooled, y, tr, ho, probe, a_v, a_n, priors, n_cls, args, seed, device)
                lg = heads_logits(m, pooled[va], device)
            else:
                tab = df.copy()
                p_, c_, extra = train_full(rec, tab, y, tab.iloc[tr], tab.iloc[ho], predict, probe, a_v, a_n, priors,
                                           n_cls, mode, args, seed, device, t_end)
                if p_ is None:
                    break
                _, lg0 = run_frozen(tab.iloc[va], predict, p_, mode, args, device)
                lg = comp_logits(c_, lg0, device)
                torch.save(dict(probe=p_.state_dict(), composer=c_.state_dict()), os.path.join(out_dir, f"{r.replace(':', '_')}_s{seed}.pt"))
            sc, _ = score(lg, yv, n_cls, meta_hits["action"], args.bootstrap, seed)
            runs.append(dict(seed=seed, val=sc, minutes=(time.time() - t0) / 60, **extra))
            print(f"    {r} seed {seed}: " + ", ".join(f"{k} {sc[k]:.2f}" for k in ("action", "verb", "noun")) + f"  epoch {extra.get('epoch_used')}  ({runs[-1]['minutes']:.1f} min)", flush=True)
        if runs:
            res[r] = dict(runs=runs, mean={k: float(np.mean([x["val"][k] for x in runs])) for k in TASKS},
                          std={k: float(np.std([x["val"][k] for x in runs])) for k in TASKS})
        with open(res_path, "w") as f:
            json.dump(res, f, indent=1)
    print(f"\n[{args.anchor}] val subset ({len(va)} clips), mean-class R@5 clean, mean +- std over seeds")
    print(f"{'recipe':24s} {'action':>14s} {'verb':>14s} {'noun':>14s}  CI95 (action - meta)")
    show = ("action", "verb", "noun")
    print(f"{'meta':24s} " + " ".join(f"{meta_score[k]:14.2f}" for k in show))
    for r in args.recipes:
        if r in res and r != "meta":
            m_, s_ = res[r]["mean"], res[r]["std"]
            ci = res[r]["runs"][0]["val"].get("action_minus_meta_ci95")
            ci = f"[{ci[0]:+.2f}, {ci[1]:+.2f}]" if ci else ""
            print(f"{r:24s} " + " ".join(f"{m_[k]:8.2f} +-{s_[k]:4.2f}" for k in show) + f"  {ci}")


# =============================================================================================== self-test
def make_fake(root, n_train=600, n_val=300, T=16, D=32, V=4, N=5, seed=0):
    """Tiny fake token datasets + csvs (two train parts sharing local clip ids, one val part)."""
    rng = np.random.default_rng(seed)
    acts = [(v, n) for v in range(V) for n in range(N)]
    freq = np.exp(rng.normal(0, 1.2, len(acts)))
    proto = rng.normal(0, 1, (len(acts), D))

    def rows(n, prefix, split_lab):
        lab = rng.choice(len(acts), n, p=freq / freq.sum()) if split_lab == "train" else rng.integers(0, len(acts), n)
        return pd.DataFrame(dict(narration_id=[f"{prefix}_{i}" for i in range(n)],
                                 video_id=[f"P{(i % 8) + 1:02d}_{prefix[-2:]}" for i in range(n)],
                                 verb_class=[acts[a][0] for a in lab], noun_class=[acts[a][1] for a in lab]))

    tr_df = pd.concat([rows(n_train // 2, "P00_tr1", "train"), rows(n_train // 2, "P00_tr2", "train")], ignore_index=True)
    va_df = rows(n_val, "P00_va", "val")
    tr_df.to_csv(os.path.join(root, "train.csv"), index=False)
    va_df.to_csv(os.path.join(root, "val.csv"), index=False)
    rv, rn, ra = reference_classes(os.path.join(root, "train.csv"), os.path.join(root, "val.csv"))
    for name, part in (("tr1", tr_df.iloc[: n_train // 2]), ("tr2", tr_df.iloc[n_train // 2:]), ("va", va_df)):
        d = os.path.join(root, "input", f"ek100-tok-{name}", "full_tokens")
        os.makedirs(os.path.join(d, "work"), exist_ok=True)
        with open(os.path.join(d, "work", "classes.json"), "w") as f:
            json.dump(dict(verbs=[[k, i] for k, i in rv.items()], nouns=[[k, i] for k, i in rn.items()],
                           actions=[[k[0], k[1], i] for k, i in ra.items()]), f)
        c = part.reset_index(drop=True).assign(clip_id=lambda x: np.arange(len(x)))
        c["verb_idx"] = c.verb_class.map(rv)
        c["noun_idx"] = c.noun_class.map(rn)
        c["action_idx"] = [ra[(v, n)] for v, n in zip(c.verb_class, c.noun_class)]
        c.to_csv(os.path.join(d, "work", "val_clips.csv"), index=False)
        for anchor in ANCHORS:
            sd = os.path.join(d, anchor, "shard0")
            os.makedirs(sd, exist_ok=True)
            for i0 in range(0, len(c), 32):
                cc = c.iloc[i0:i0 + 32]
                x = rng.normal(0, 1, (len(cc), T, D)) + proto[cc.action_idx.values][:, None] * 1.5
                s = np.abs(x).max(-1) / 127.0
                np.savez(os.path.join(sd, f"chunk_{i0 // 32:05d}.npz"), clip_id=cc.clip_id.values,
                         ctx_q=np.round(x / s[..., None]).astype(np.int8), ctx_s=s.astype(np.float16),
                         real_q=np.zeros((len(cc), 4, D), np.int8), real_s=np.ones((len(cc), 4), np.float16),
                         cos_pred=np.zeros(len(cc), np.float32),
                         verb=np.zeros((len(cc), len(rv)), np.float16), noun=np.zeros((len(cc), len(rn)), np.float16),
                         action=np.zeros((len(cc), len(ra)), np.float16))
    a_v = [rv[v] for (v, n), i in sorted(ra.items(), key=lambda t: t[1])]
    a_n = [rn[n] for (v, n), i in sorted(ra.items(), key=lambda t: t[1])]
    return len(rv), len(rn), len(ra), a_v, a_n


# =============================================================================================== main
def main():
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["index", "verify", "train", "selftest"])
    p.add_argument("--root", default="/kaggle/input")
    p.add_argument("--train_csv")
    p.add_argument("--val_csv")
    p.add_argument("--out_dir", default="tok")
    p.add_argument("--index")
    p.add_argument("--holdout", type=float, default=0.1, help="share of train clips (whole participants) held out")
    p.add_argument("--config", default=os.path.join(os.path.dirname(__file__), "..", "configs", "ek100_vitl_inference.yaml"))
    p.add_argument("--vjepa2_root", default=None)
    p.add_argument("--vitl_ckpt")
    p.add_argument("--probe_ckpt")
    p.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    p.add_argument("--n", type=int, default=4, help="verify: clips per split/anchor")
    p.add_argument("--n_verb", type=int, default=97)
    p.add_argument("--n_noun", type=int, default=289)
    p.add_argument("--n_action", type=int, default=3568)
    p.add_argument("--anchor", default="action_start", choices=ANCHORS)
    p.add_argument("--recipes", nargs="+", default=["meta", "heads:bsm+compose", "full:bsm+compose"])
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2], help="heads recipes use all, full recipes the first")
    p.add_argument("--dequant", choices=["qs", "qs127"], default=None, help="default: verify.json, else qs")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--accum", type=int, default=4, help="gradient accumulation for full recipes (effective batch x4)")
    p.add_argument("--num_workers", type=int, default=3)
    p.add_argument("--shuffle_buffer", type=int, default=96)
    p.add_argument("--head_epochs", type=int, default=15)
    p.add_argument("--full_epochs", type=int, default=3)
    p.add_argument("--lr_heads", type=float, default=1e-4)
    p.add_argument("--lr_pooler", type=float, default=1e-5)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--la_tau", type=float, default=0.5)
    p.add_argument("--bootstrap", type=int, default=500)
    p.add_argument("--max_hours", type=float, default=11.0)
    p.add_argument("--redo", action="store_true", help="re-run recipes already in results.json")
    p.add_argument("--fake", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--fake_dim", type=int, default=32, help=argparse.SUPPRESS)
    args = p.parse_args()
    if args.cmd == "index":
        cmd_index(args)
    elif args.cmd == "verify":
        cmd_verify(args)
    elif args.cmd == "train":
        cmd_train(args)
    else:  # selftest: fake datasets, fake predictor, small probe; checks the whole pipeline runs and learns
        root = args.out_dir
        os.makedirs(root, exist_ok=True)
        nv, nn_, na, a_v, a_n = make_fake(root)
        args.root, args.train_csv, args.val_csv = os.path.join(root, "input"), os.path.join(root, "train.csv"), os.path.join(root, "val.csv")
        args.out_dir = os.path.join(root, "tok")
        cmd_index(args)
        args.index = os.path.join(args.out_dir, "index.csv")
        args.fake, args.n_verb, args.n_noun, args.n_action = True, nv, nn_, na
        args.fake_av, args.fake_an = a_v, a_n
        args.dequant, args.num_workers, args.batch_size, args.accum = "qs", 0, 16, 1
        args.lr_heads, args.lr_pooler, args.head_epochs, args.full_epochs, args.bootstrap = 3e-3, 1e-3, 8, 3, 50
        args.seeds = [0]
        args.recipes = ["meta", "heads:bsm+compose", "full:bsm+compose"]
        for anchor in ANCHORS:
            args.anchor = anchor
            cmd_train(args)


if __name__ == "__main__":
    main()
