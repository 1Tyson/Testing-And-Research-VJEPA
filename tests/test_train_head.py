import json
import os
import subprocess
import sys

import pytest

HERE = os.path.dirname(__file__)


def test_train_head_on_synthetic_long_tail(tmp_path):
    """Skewed train / uniform val: a balanced loss must beat plain CE on mean-class recall, and the released
    layers must reproduce the stored logits."""
    pytest.importorskip("torch")
    out = tmp_path / "heads.json"
    subprocess.run([sys.executable, os.path.join(HERE, "..", "scripts", "train_head.py"), "--synthetic",
                    str(tmp_path / "syn"), "--out", str(out), "--recipes", "meta", "ce+random", "bsm+random",
                    "bsm+compose", "--seeds", "0", "--epochs", "8", "--bootstrap", "20", "--device", "cpu",
                    "--logits_out", str(tmp_path / "lg")], check=True)
    r = json.loads(out.read_text())
    assert min(r["meta_reconstruction_top1_agreement"].values()) > 99
    s = r["summary"]
    assert s["bsm+random"]["action_raw_clean"][0] > s["ce+random"]["action_raw_clean"][0] + 2
    assert "beta_verb_noun" in r["recipes"]["bsm+compose"]["seeds"][0]
    assert (tmp_path / "lg" / "bsm+random" / "action_start" / "shard0" / "chunk_00000.npz").exists()
