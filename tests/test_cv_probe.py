import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(__file__)


def test_cv_probe_finds_information_in_the_real_future(tmp_path):
    """Labels are encoded only in the 'real' future tokens: enc_real must beat enc, folds must split participants."""
    rng = np.random.default_rng(0)
    n, T, P, D, C = 240, 2, 4, 16, 16  # C > 5, else top-5 is always right
    labels = rng.integers(0, C, n)
    proto = rng.normal(size=(C, D)).astype(np.float32) * 3
    feats = rng.normal(size=(n, T, P, D)).astype(np.float16)
    real = (rng.normal(size=(n, P, D)) + proto[labels][:, None]).astype(np.float16)
    pred = rng.normal(size=(n, P, D)).astype(np.float16)
    logits = rng.normal(size=(n, C)).astype(np.float16)
    ids = np.arange(n)
    d = tmp_path / "logits" / "action_start" / "shard0"
    d.mkdir(parents=True)
    for part in (slice(0, 150), slice(150, n)):  # two chunks, one clip duplicated across them
        sl = np.r_[ids[part], [0]]
        np.savez(d / f"chunk_{part.start:05d}.npz", clip_id=sl, verb=logits[sl], noun=logits[sl], action=logits[sl],
                 feats=feats[sl], feats_pred=pred[sl], feats_real=real[sl])
    (d / "failed.json").write_text(json.dumps([5]))
    work = tmp_path / "work"
    work.mkdir()
    vids = [f"P{p:02d}_01" for p in range(1, 7)]
    pd.DataFrame(dict(clip_id=ids, video_id=[vids[i % 6] for i in ids], verb_idx=labels, noun_idx=labels,
                      action_idx=labels, expected_fail_action_start=ids == 7)).to_csv(work / "val_clips.csv", index=False)
    cls = [str(i) for i in range(C)]
    (work / "classes.json").write_text(json.dumps(dict(verbs=cls, nouns=cls, actions=cls)))
    out = tmp_path / "cv.json"
    subprocess.run([sys.executable, os.path.join(HERE, "..", "scripts", "cv_probe.py"), "--work_dir", str(work),
                    "--logits_root", str(tmp_path / "logits"), "--out", str(out), "--device", "cpu",
                    "--variants", "enc", "enc_real", "real", "--seeds", "0", "--epochs", "6",
                    "--dim", "32", "--heads", "2", "--batch_size", "32", "--lr", "3e-3", "--bootstrap", "50"],
                   check=True)
    res = json.loads(out.read_text())
    assert res["clips"] == n - 2  # failed + expected_fail removed, duplicate counted once
    a, b = res["participants_per_fold"]
    assert not set(a) & set(b) and len(a) + len(b) == 6
    assert res["mean"]["enc_real"]["action"] > 70
    assert res["mean"]["real"]["action"] > 70
    assert res["mean"]["enc_real"]["action"] - res["mean"]["enc"]["action"] > 30
    assert "enc_real - enc_pred" not in res["action_diff"]
