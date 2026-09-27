import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "tests"))


@pytest.fixture(scope="session")
def vjepa2_root():
    root = os.environ.get("VJEPA2_ROOT")
    if not root or not os.path.isdir(os.path.join(root, "evals")):
        pytest.skip("set VJEPA2_ROOT to a facebookresearch/vjepa2 clone to compare against official code")
    if root not in sys.path:
        sys.path.insert(0, root)
    return root
