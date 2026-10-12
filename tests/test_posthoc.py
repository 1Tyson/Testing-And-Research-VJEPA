import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(__file__)


def make_biased_logits(tmp_path):
    """Synthetic val logits whose action scores carry +1.5 * log(train prior)."""
    rng = np.random.default_rng(0)
    V, N, n = 6, 5, 3000
    actions = [(v, m) for v in range(V) for m in range(N)]
    freq = np.exp(rng.normal(0, 1.5, len(actions)))
    train = rng.choice(len(actions), 20000, p=freq / freq.sum())
    pd.DataFrame(dict(verb_class=[actions[a][0] for a in train], noun_class=[actions[a][1] for a in train])).to_csv(
        tmp_path / "train.csv", index=False)
    lab = rng.integers(0, len(actions), n)  # val labels: uniform over classes
    work = tmp_path / "work"
    work.mkdir()
    pd.DataFrame(dict(clip_id=np.arange(n), video_id=[f"P{i % 6 + 1:02d}_01" for i in range(n)], video_order=0,
                      clip_order=np.arange(n), verb_idx=[actions[a][0] for a in lab],
                      noun_idx=[actions[a][1] for a in lab], action_idx=lab)).to_csv(work / "val_clips.csv", index=False)
    (work / "classes.json").write_text(json.dumps(dict(
        verbs=[[v, v] for v in range(V)], nouns=[[m, m] for m in range(N)],
        actions=[[v, m, i] for i, (v, m) in enumerate(actions)])))
    z = rng.normal(size=(n, len(actions))) * 1.5
    z[np.arange(n), lab] += 2.0
    z += 1.5 * np.log(freq / freq.sum())
    d = tmp_path / "logits" / "official" / "shard0"
    d.mkdir(parents=True)
    zv, zn = rng.normal(size=(n, V)), rng.normal(size=(n, N))
    np.savez(d / "chunk_00000.npz", clip_id=np.arange(n), verb=zv.astype(np.float16), noun=zn.astype(np.float16),
             action=z.astype(np.float16))
    return work, d


def test_logit_adjustment_undoes_a_prior_bias(tmp_path):
    """Cross-fitted tuning must pick tau > 0 and raise mean-class recall."""
    work, d = make_biased_logits(tmp_path)
    out = tmp_path / "p.json"
    subprocess.run([sys.executable, os.path.join(HERE, "..", "scripts", "posthoc_ek100.py"), "--work_dir", str(work),
                    "--train_csv", str(tmp_path / "train.csv"), "--logits_dir", str(d.parent), "--out", str(out),
                    "--bootstrap", "20"], check=True)
    a = json.loads(out.read_text())["tasks"]["action"]
    assert all(c["tau"] > 0 for c in a["cross_fitted"]["chosen_per_fold"])
    assert a["logit_adjustment_only"]["clean"] > a["released"]["clean"] + 5


def test_analysis_finds_the_prior_and_the_tail_gain(tmp_path):
    work, d = make_biased_logits(tmp_path)
    subprocess.run([sys.executable, os.path.join(HERE, "..", "scripts", "analyze_posthoc.py"), "--work_dir", str(work),
                    "--train_csv", str(tmp_path / "train.csv"), "--logits", f"X={d.parent}", "--out_dir",
                    str(tmp_path / "an"), "--no_figures"], check=True)
    a = json.loads((tmp_path / "an" / "analysis.json").read_text())[0]["tasks"]["action"]
    assert 1.2 < a["prior_in_logits"]["slope"] < 1.8  # 1.5 * log prior was added
    assert a["groups"]["logit_adjustment"]["few"]["recall"] > a["groups"]["released"]["few"]["recall"]
    assert a["top5_share"]["logit_adjustment"]["many"] < a["top5_share"]["released"]["many"]
