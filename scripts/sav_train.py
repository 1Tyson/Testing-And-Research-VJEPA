"""Train a head on cached V-JEPA 2 person features (scripts/sav_extract.py) and report SAV frame mAP@0.5.

  python scripts/sav_train.py --feat_dir feats --ann_dir sav/annotations --out_dir results/sav --head tokens linear

Two numbers per head:
  gt_box_mAP   person boxes = ground truth (classification quality only)
  det_mAP      person boxes = the released detections (the protocol of the SAV paper, which reports 67.9 for
               VideoMAE-B fine-tuned end to end), only when val proposals were encoded.
Keyframes whose clip could not be downloaded/encoded are left out of the ground truth, and coverage is reported.
Fixed recipe, no early stopping on val; several seeds, mean and std.
"""

import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from classroom.ava_metric import frame_map, iou_matrix  # noqa: E402
from classroom.head import build_head  # noqa: E402
from classroom.sav import ann_paths, multi_hot, read_annotations, read_exclusions, read_labelmap  # noqa: E402


def load_split(feat_dir, split):
    table = pd.read_csv(os.path.join(feat_dir, f"boxes_{split}.csv"), keep_default_na=False)
    files = sorted(glob.glob(os.path.join(feat_dir, split, "shard*_chunk*.npz")))
    if not files:
        raise SystemExit(f"no features for {split} in {feat_dir}")
    box_ids, box_tok, keys, ctx, clips = [], [], [], [], []
    for f in files:
        z = np.load(f)
        box_ids.append(z["box_id"]), box_tok.append(z["box_tok"]), keys.extend(z["key"].tolist())
        ctx.append(z["ctx"]), clips.extend(z["clip"].tolist())
    box_ids = np.concatenate(box_ids)
    order = np.argsort(box_ids)
    box_ids, box_tok = box_ids[order], np.concatenate(box_tok)[order]
    _, first = np.unique(keys, return_index=True)  # a key may appear twice if a run was resumed
    keys = [keys[i] for i in sorted(first)]
    ctx = np.concatenate(ctx)[sorted(first)]
    _, first = np.unique(box_ids, return_index=True)
    box_ids, box_tok = box_ids[first], box_tok[first]
    t = table.set_index("box_id").loc[box_ids].reset_index()
    key_index = {k: i for i, k in enumerate(keys)}
    t["ctx_idx"] = t.key.map(key_index)
    return t, torch.from_numpy(box_tok), torch.from_numpy(ctx), set(clips)


def add_matched_proposals(t, iou_thr=0.9):
    """Training proposals take the labels of the GT box they overlap with IoU >= thr (mmaction's assigner)."""
    gt = t[t.source == "gt"]
    props = t[t.source == "prop"].copy()
    if props.empty:
        return t[t.source == "gt"]
    labels = []
    gt_by_key = {k: g for k, g in gt.groupby("key")}
    for key, g in props.groupby("key", sort=False):
        ref = gt_by_key.get(key)
        if ref is None:
            labels.extend([None] * len(g))
            continue
        iou = iou_matrix(g[["x1", "y1", "x2", "y2"]].to_numpy(), ref[["x1", "y1", "x2", "y2"]].to_numpy())
        best = iou.argmax(1)
        labels.extend([ref.labels.iloc[j] if iou[i, j] >= iou_thr else None for i, j in enumerate(best)])
    props = props.loc[[i for g in props.groupby("key", sort=False) for i in g[1].index]]
    props["labels"] = labels
    props = props[props.labels.notna()]
    return pd.concat([gt, props])


def predict(head, box_tok, ctx, rows, device, bs=4096):
    head.eval()
    out = []
    with torch.no_grad():
        for s in range(0, len(rows), bs):
            r = rows[s: s + bs]
            b = box_tok[r.index.to_numpy()].to(device)
            c = ctx[r.ctx_idx.to_numpy()].to(device)
            out.append(torch.sigmoid(head(b, c)).float().cpu())
    return torch.cat(out).numpy() if out else np.zeros((0, 0))


def detections(rows, scores):
    det = {}
    for key, g in rows.groupby("key", sort=False):
        idx = [rows.index.get_loc(i) for i in g.index]
        det[key] = (g[["x1", "y1", "x2", "y2"]].to_numpy(np.float32), scores[idx])
    return det


def train_one(name, args, tr, tr_tok, tr_ctx, y, num_classes, seed, device):
    torch.manual_seed(seed)
    np.random.seed(seed)
    dim = tr_tok.shape[-1]
    n_tokens = tr_tok.shape[1] + tr_ctx.shape[1]
    head = build_head(name, dim, num_classes, n_tokens, hidden=args.hidden, depth=args.depth,
                      dropout=args.dropout if name == "tokens" else 0.5).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr if name == "tokens" else args.lr_linear,
                            weight_decay=args.weight_decay)
    steps_per_epoch = int(np.ceil(len(tr) / args.batch_size))
    total = steps_per_epoch * args.epochs
    warm = steps_per_epoch * args.warmup_epochs
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / max(1, warm) if s < warm else 0.5 * (1 + np.cos(np.pi * (s - warm) / max(1, total - warm)))
    )
    tok_idx = torch.from_numpy(tr.index.to_numpy())
    ctx_idx = torch.from_numpy(tr.ctx_idx.to_numpy())
    y = torch.from_numpy(y)
    for ep in range(args.epochs):
        head.train()
        perm = torch.randperm(len(tr))
        tot = 0.0
        for s in range(0, len(perm), args.batch_size):
            p = perm[s: s + args.batch_size]
            b = tr_tok[tok_idx[p]].to(device, non_blocking=True)
            c = tr_ctx[ctx_idx[p]].to(device, non_blocking=True)
            if args.token_noise > 0:
                b = b.float() + args.token_noise * torch.randn_like(b, dtype=torch.float32)
            loss = F.binary_cross_entropy_with_logits(head(b, c), y[p].to(device))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item() * len(p)
        if (ep + 1) % max(1, args.epochs // 5) == 0 or ep == 0:
            print(f"  [{name} seed {seed}] epoch {ep + 1}/{args.epochs} loss {tot / len(tr):.4f}", flush=True)
    return head


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat_dir", required=True)
    ap.add_argument("--ann_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--head", nargs="+", default=["linear", "tokens"], choices=["linear", "tokens"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--warmup_epochs", type=int, default=2)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--lr_linear", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=0.05)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--token_noise", type=float, default=0.0)
    ap.add_argument("--train_props", action="store_true", help="also train on train proposals with IoU>=0.9 to GT")
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    labelmap = read_labelmap(ann_paths(args.ann_dir, "val")["labelmap"])
    label_ids = list(labelmap)
    print(f"{len(label_ids)} classes: " + ", ".join(f"{k}:{v}" for k, v in labelmap.items()))

    tr, tr_tok, tr_ctx, _ = load_split(args.feat_dir, "train")
    tr = add_matched_proposals(tr) if args.train_props else tr[tr.source == "gt"]
    y_tr = multi_hot(tr.labels.tolist(), label_ids)
    va, va_tok, va_ctx, va_clips = load_split(args.feat_dir, "val")
    va_gt_rows, va_prop_rows = va[va.source == "gt"], va[va.source == "prop"]

    p = ann_paths(args.ann_dir, "val")
    gt_all = read_annotations(p["ann"])
    excl = read_exclusions(p["excl"])
    gt = {k: v for k, v in gt_all.items() if k.rsplit(",", 1)[0] in va_clips}
    n_box_all = sum(len(v) for v in gt_all.values())
    n_box = sum(len(v) for v in gt.values())
    coverage = dict(
        train_boxes=len(tr), val_keyframes=len(gt), val_keyframes_total=len(gt_all),
        val_gt_boxes=n_box, val_gt_boxes_total=n_box_all, val_proposals=len(va_prop_rows),
    )
    print("coverage:", coverage, flush=True)

    results = dict(args=vars(args), coverage=coverage, labelmap=labelmap, runs={})
    for name in args.head:
        runs = []
        for seed in args.seeds:
            t0 = time.time()
            head = train_one(name, args, tr, tr_tok, tr_ctx, y_tr, len(label_ids), seed, args.device)
            r = dict(seed=seed, train_sec=round(time.time() - t0, 1))
            s = predict(head, va_tok, va_ctx, va_gt_rows, args.device)
            r["gt_box_mAP"], ap_gt = frame_map(gt, detections(va_gt_rows, s), label_ids, excl)
            r["gt_box_AP"] = {labelmap[k]: v for k, v in ap_gt.items()}
            if len(va_prop_rows):
                s = predict(head, va_tok, va_ctx, va_prop_rows, args.device)
                r["det_mAP"], ap_det = frame_map(gt, detections(va_prop_rows, s), label_ids, excl)
                r["det_AP"] = {labelmap[k]: v for k, v in ap_det.items()}
                if seed == args.seeds[0]:
                    write_ava_csv(os.path.join(args.out_dir, f"val_detections_{name}.csv"), va_prop_rows, s, label_ids)
            print(f"[{name} seed {seed}] gt_box_mAP {100 * r['gt_box_mAP']:.2f}"
                  + (f"  det_mAP {100 * r['det_mAP']:.2f}" if "det_mAP" in r else ""), flush=True)
            if seed == args.seeds[0]:
                torch.save(dict(state_dict=head.state_dict(), head=name, label_ids=label_ids, labelmap=labelmap,
                                args=vars(args)), os.path.join(args.out_dir, f"head_{name}.pt"))
            runs.append(r)
        results["runs"][name] = runs

    with open(os.path.join(args.out_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=1, ensure_ascii=False)
    md = summary_md(results)
    with open(os.path.join(args.out_dir, "results.md"), "w") as f:
        f.write(md)
    print(md)


def write_ava_csv(path, rows, scores, label_ids, thr=0.002):
    with open(path, "w") as f:
        for (_, r), s in zip(rows.iterrows(), scores):
            vid, ts = r.key.rsplit(",", 1)
            for lid, v in zip(label_ids, s):
                if v >= thr:
                    f.write(f"{vid},{ts},{r.x1:.3f},{r.y1:.3f},{r.x2:.3f},{r.y2:.3f},{lid},{v:.4f}\n")


def summary_md(res):
    c = res["coverage"]
    lines = [
        "# SAV, frozen V-JEPA 2 + head: frame mAP@0.5",
        "",
        f"Val coverage: {c['val_keyframes']}/{c['val_keyframes_total']} keyframes, "
        f"{c['val_gt_boxes']}/{c['val_gt_boxes_total']} GT boxes; train boxes {c['train_boxes']}; "
        f"val proposals {c['val_proposals']}.",
        "",
        "| head | GT boxes mAP | detected boxes mAP |",
        "|---|---|---|",
    ]
    for name, runs in res["runs"].items():
        g = np.array([r["gt_box_mAP"] for r in runs]) * 100
        d = np.array([r["det_mAP"] for r in runs if "det_mAP" in r]) * 100
        ds = f"{d.mean():.2f} ± {d.std():.2f}" if len(d) else "n/a"
        lines.append(f"| {name} | {g.mean():.2f} ± {g.std():.2f} | {ds} |")
    lines += ["", f"Mean ± std over seeds {res['args']['seeds']}. SAV paper (VideoMAE-B, fine-tuned, detected boxes): 67.9.",
              "", "Per-class AP (first seed, GT boxes / detected boxes):", "",
              "| class | " + " | ".join(res["runs"]) + " |", "|---|" + "---|" * len(res["runs"])]
    names = list(res["labelmap"].values())
    for cname in names:
        cells = []
        for runs in res["runs"].values():
            g = runs[0]["gt_box_AP"].get(cname, float("nan")) * 100
            d = runs[0].get("det_AP", {}).get(cname, float("nan")) * 100
            cells.append(f"{g:.1f} / {d:.1f}")
        lines.append(f"| {cname} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
