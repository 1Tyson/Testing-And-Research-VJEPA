import json
import os
import subprocess
import sys

HERE = os.path.dirname(__file__)


def test_tok_probe_selftest_learns(vjepa2_root, tmp_path):
    """Fake token datasets (local clip ids clash across datasets): index by narration id, train heads / full probe."""
    env = dict(os.environ, VJEPA2_ROOT=vjepa2_root)
    subprocess.run([sys.executable, os.path.join(HERE, "..", "scripts", "tok_probe.py"), "selftest", "--out_dir",
                    str(tmp_path), "--device", "cpu"], check=True, env=env)
    rep = json.loads((tmp_path / "tok" / "index_report.json").read_text())
    assert rep["counts"]["val/action_start"] == 300 and rep["train_val_overlap"] == 0
    assert rep["counts"]["train/official"] == 600  # two datasets with the same local clip ids, both kept
    for anchor in ("action_start", "official"):
        res = json.loads((tmp_path / "tok" / anchor / "results.json").read_text())
        meta = res["meta"]["val"]["action"]
        assert res["heads:bsm+compose"]["mean"]["action"] > meta
        assert res["full:bsm+compose"]["mean"]["action"] > meta
