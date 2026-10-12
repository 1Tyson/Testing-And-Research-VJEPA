"""Where does the post-hoc re-scoring gain come from? (analysis for the paper, CPU only)

For each logits directory (e.g. ViT-L and ViT-g, official protocol):
  * recall by class-frequency group (train counts: many > 100, medium 20-100, few < 20), for the released probe,
    logit adjustment only, verb-noun fusion only, and both (each cross-fitted over participant folds);
  * share of top-5 slots taken by each group, vs the groups' share of val labels (the head bias);
  * how much of the train prior is already inside the logits: slope of each class's mean logit vs log prior;
  * calibration of the released probe (trained with a sigmoid focal loss, alpha 0.25, gamma 2): ECE of the
    top-1 sigmoid score;
  * the recall-vs-tau curve (beta = 0 and the best beta), and figures for both.

python scripts/analyze_posthoc.py --work_dir work --train_csv EPIC_100_train.csv \
    --logits ViT-L=logits_orig/official ViT-g=merged_vitg/official --out_dir analysis
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))
from posthoc_ek100 import BETAS, TAUS, TASKS, load_inputs, score  # noqa: E402
from vjepa_ek100.metrics import mean_class_recall, per_class_recall, topk_hits  # noqa: E402

GROUPS = (("many", 100, np.inf), ("medium", 20, 100), ("few", 0, 20))  # train count: > 100, 20-100, < 20
BLUE, ORANGE = "#2a78d6", "#eb6834"  # categorical slots 1-2 (validated pair)


def train_counts(train_csv, classes):
    tdf = pd.read_csv(train_csv)
    vmap = {int(k): i for k, i in classes["verbs"]}
    nmap = {int(k): i for k, i in classes["nouns"]}
    amap = {(int(v), int(n)): i for v, n, i in classes["actions"]}
    idx = dict(verb=tdf.verb_class.map(vmap).values, noun=tdf.noun_class.map(nmap).values,
               action=np.array([amap[(v, n)] for v, n in zip(tdf.verb_class, tdf.noun_class)]))
    size = dict(verb=len(vmap), noun=len(nmap), action=len(amap))
    return {k: np.bincount(idx[k].astype(np.int64), minlength=size[k]) for k in TASKS}


def group_of(counts):
    g = np.empty(len(counts), dtype=object)
    g[counts > 100] = "many"
    g[(counts >= 20) & (counts <= 100)] = "medium"
    g[counts < 20] = "few"
    return g


def topk_idx(z, k=5):
    return np.argpartition(-z, kth=k - 1, axis=1)[:, :k]


def ece(conf, correct, bins=15):
    edges = np.linspace(0, 1, bins + 1)
    out = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            out += m.mean() * abs(conf[m].mean() - correct[m].mean())
    return float(out)


def analyze(name, d, counts):
    lg, labels, n_cls, prior, a_v, a_n, folds = (d[k] for k in ("lg", "labels", "n_cls", "prior", "a_v", "a_n", "folds"))

    def recall(h, task, sel=slice(None)):
        return mean_class_recall(h[sel], labels[task][sel], n_cls[task])["recall"]

    def crossfit(task, configs):
        hits = {tb: topk_hits(score(task, lg, prior, a_v, a_n, *tb), labels[task]) for tb in configs}
        out, chosen = np.zeros(len(labels[task]), bool), {}
        for f in np.unique(folds):
            tr, te = folds != f, folds == f
            best = max(hits, key=lambda tb: recall(hits[tb], task, tr))
            out[te], chosen[int(f)] = hits[best][te], best
        return out, chosen, hits

    res = dict(name=name, clips=int(len(lg["clip_id"])), tasks={})
    for task in TASKS:
        g = group_of(counts[task])
        variants = {"released": [(0.0, 0.0)], "logit_adjustment": [(t, 0.0) for t in TAUS]}
        if task == "action":
            variants["fusion_only"] = [(0.0, b) for b in BETAS]
            variants["both"] = [(t, b) for t in TAUS for b in BETAS]
        r = dict(groups={}, overall={}, chosen={}, top5_share={}, label_share={})
        lab_g = g[labels[task]]
        for gname, _, _ in GROUPS:
            r["label_share"][gname] = float((lab_g == gname).mean())
        all_hits = {}
        for v, cfg in variants.items():
            h, chosen, hits = crossfit(task, cfg)
            all_hits.update(hits)
            r["overall"][v] = recall(h, task)
            r["chosen"][v] = {f: list(tb) for f, tb in chosen.items()}
            pcr, tot = per_class_recall(h, labels[task], n_cls[task])
            r["groups"][v] = {gname: dict(recall=float(np.nanmean(pcr[(g == gname) & (tot > 0)]) * 100),
                                          classes=int(((g == gname) & (tot > 0)).sum()))
                              for gname, _, _ in GROUPS}
            # which classes fill the top-5 slots (with each fold's chosen tau/beta)
            share = {gname: 0 for gname, _, _ in GROUPS}
            for f, tb in chosen.items():
                te = folds == f
                sub = {k: (v_[te] if k != "clip_id" else v_) for k, v_ in lg.items()}
                top = topk_idx(score(task, sub, prior, a_v, a_n, *tb))
                for gname, _, _ in GROUPS:
                    share[gname] += int((g[top] == gname).sum())
            n_slots = sum(share.values())
            r["top5_share"][v] = {k: s / n_slots for k, s in share.items()}
        # prior already inside the logits
        z = lg[task]
        mean_logit = z.mean(0)
        lp = np.log((counts[task] + 1.0) / (counts[task] + 1.0).sum())
        slope, _ = np.polyfit(lp, mean_logit, 1)
        r["prior_in_logits"] = dict(slope=float(slope), pearson=float(np.corrcoef(lp, mean_logit)[0, 1]))
        # calibration of the released probe (sigmoid scores, as trained)
        top1 = z.argmax(1)
        conf = 1.0 / (1.0 + np.exp(-z[np.arange(len(z)), top1]))
        correct = (top1 == labels[task]).astype(float)
        r["calibration"] = dict(ece_top1_sigmoid=ece(conf, correct), mean_confidence=float(conf.mean()),
                                top1_accuracy=float(correct.mean()))
        if task == "action":
            best_b = max(BETAS, key=lambda b: max(recall(all_hits[(t, b)], task) for t in TAUS))
            r["tau_curve"] = dict(beta_best=best_b,
                                  beta0=[recall(all_hits[(t, 0.0)], task) for t in TAUS],
                                  beta_best_curve=[recall(all_hits[(t, best_b)], task) for t in TAUS], taus=TAUS)
        res["tasks"][task] = r
    return res


def report(results):
    lines = ["# Phân tích cách chấm lại (clean mean-class R@5, cross-fit 2 fold theo người tham gia)", ""]
    for res in results:
        lines += [f"## {res['name']} ({res['clips']} clip)", ""]
        a = res["tasks"]["action"]
        lines += ["Action theo nhóm lớp (số lần xuất hiện trong train: many > 100, medium 20–100, few < 20):", "",
                  "| nhóm (số lớp có trong val) | probe Meta | chỉ logit adj. | chỉ ghép verb–noun | cả hai | tỉ lệ nhãn val | "
                  "tỉ lệ ô top-5: Meta → cả hai |", "|---|---|---|---|---|---|---|"]
        for gname, _, _ in GROUPS:
            gr = {v: a["groups"][v][gname]["recall"] for v in a["groups"]}
            lines.append(f"| {gname} ({a['groups']['released'][gname]['classes']}) | {gr['released']:.2f} | "
                         f"{gr['logit_adjustment']:.2f} | {gr['fusion_only']:.2f} | {gr['both']:.2f} | "
                         f"{a['label_share'][gname]:.1%} | {a['top5_share']['released'][gname]:.1%} → "
                         f"{a['top5_share']['both'][gname]:.1%} |")
        o = a["overall"]
        lines += [f"| **tất cả** | {o['released']:.2f} | {o['logit_adjustment']:.2f} | {o['fusion_only']:.2f} | "
                  f"{o['both']:.2f} | | |", ""]
        lines += ["Verb / noun theo nhóm lớp (probe Meta → logit adjustment):", ""]
        for task in ("verb", "noun"):
            t = res["tasks"][task]
            cells = ", ".join(f"{g} {t['groups']['released'][g]['recall']:.1f} → {t['groups']['logit_adjustment'][g]['recall']:.1f}"
                              f" ({t['groups']['released'][g]['classes']} lớp)" for g, _, _ in GROUPS)
            lines.append(f"- {task}: {cells}")
        lines += ["", "Prior của tập train đã nằm sẵn trong logits (độ dốc của logit trung bình mỗi lớp theo log prior) và "
                  "hiệu chỉnh của probe (sigmoid focal loss):", ""]
        for task in TASKS:
            t = res["tasks"][task]
            lines.append(f"- {task}: độ dốc {t['prior_in_logits']['slope']:.3f} (r = {t['prior_in_logits']['pearson']:.2f}); "
                         f"ECE top-1 {t['calibration']['ece_top1_sigmoid']:.3f}, độ tự tin trung bình "
                         f"{t['calibration']['mean_confidence']:.3f} so với accuracy top-1 {t['calibration']['top1_accuracy']:.3f}")
        tc = a["tau_curve"]
        i1 = tc["taus"].index(1.0)
        lines += ["", f"Đường recall theo τ (action): β=0 cho τ=0 → {tc['beta0'][0]:.2f}, τ=1 → {tc['beta0'][i1]:.2f}; "
                  f"β={tc['beta_best']}: tốt nhất {max(tc['beta_best_curve']):.2f} tại τ="
                  f"{tc['taus'][int(np.argmax(tc['beta_best_curve']))]}.", ""]
    return "\n".join(lines)


def figures(results, out_dir):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 10, "axes.edgecolor": "#52514e", "axes.labelcolor": "#0b0b0b",
                         "xtick.color": "#52514e", "ytick.color": "#52514e", "axes.spines.top": False,
                         "axes.spines.right": False})
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(4.6 * n, 3.4), squeeze=False)
    for ax, res in zip(axes[0], results):
        tc = res["tasks"]["action"]["tau_curve"]
        ax.plot(tc["taus"], tc["beta0"], color=BLUE, lw=2, marker="o", ms=4, label="β = 0 (chỉ logit adjustment)")
        ax.plot(tc["taus"], tc["beta_best_curve"], color=ORANGE, lw=2, marker="s", ms=4,
                label=f"β = {tc['beta_best']} (thêm ghép verb–noun)")
        ax.axvline(1.0, color="#52514e", lw=1, ls="--")
        ax.text(1.02, ax.get_ylim()[1] * 0.98, "τ = 1", color="#52514e", va="top", fontsize=9)
        ax.set_title(res["name"], color="#0b0b0b")
        ax.set_xlabel("τ")
        ax.set_ylabel("action mean-class R@5 (clean)")
        ax.grid(axis="y", color="#e5e4e0", lw=0.8)
        ax.set_axisbelow(True)
    axes[0][0].legend(frameon=False, fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig_tau_curve.png"), dpi=200)
    plt.close(fig)

    fig, axes = plt.subplots(1, n, figsize=(4.6 * n, 3.4), squeeze=False)
    names = [gname for gname, _, _ in GROUPS]
    x = np.arange(len(names))
    for ax, res in zip(axes[0], results):
        a = res["tasks"]["action"]["groups"]
        for off, v, col, lab in ((-0.2, "released", BLUE, "probe Meta"), (0.2, "both", ORANGE, "sau khi chấm lại")):
            vals = [a[v][g]["recall"] for g in names]
            bars = ax.bar(x + off, vals, width=0.38, color=col, label=lab, edgecolor="white", linewidth=2)
            for b, val in zip(bars, vals):
                ax.text(b.get_x() + b.get_width() / 2, val + 0.6, f"{val:.1f}", ha="center", fontsize=8, color="#0b0b0b")
        ax.set_xticks(x, [f"{g}\n({a['released'][g]['classes']} lớp)" for g in names])
        ax.set_title(res["name"], color="#0b0b0b")
        ax.set_ylabel("action recall@5 trung bình mỗi lớp")
        ax.grid(axis="y", color="#e5e4e0", lw=0.8)
        ax.set_axisbelow(True)
    axes[0][0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig_groups.png"), dpi=200)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--work_dir", required=True)
    p.add_argument("--train_csv", required=True)
    p.add_argument("--logits", nargs="+", required=True, help="NAME=logits_dir, e.g. ViT-L=logits_orig/official")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--no_figures", action="store_true")
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    results = []
    for item in args.logits:
        name, path = item.split("=", 1)
        d = load_inputs(args.work_dir, args.train_csv, path)
        counts = train_counts(args.train_csv, d["classes"])
        results.append(analyze(name, d, counts))
        print(f"[done] {name}", flush=True)
    with open(os.path.join(args.out_dir, "analysis.json"), "w") as f:
        json.dump(results, f, indent=2)
    md = report(results)
    with open(os.path.join(args.out_dir, "analysis.md"), "w") as f:
        f.write(md)
    print(md)
    if not args.no_figures:
        figures(results, args.out_dir)
        print(f"figures: {args.out_dir}/fig_tau_curve.png, {args.out_dir}/fig_groups.png")


if __name__ == "__main__":
    main()
