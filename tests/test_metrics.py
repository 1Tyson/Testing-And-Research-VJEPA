import numpy as np
import torch

from vjepa_ek100.metrics import mean_class_recall, topk_hits


def _official_recall(monkeypatch, logits, labels, num_classes, batch=7):
    import torch.distributed as dist
    from evals.action_anticipation_frozen.metrics import ClassMeanRecall

    monkeypatch.setattr(dist, "all_reduce", lambda t: None)
    m = ClassMeanRecall(num_classes=num_classes, device="cpu", k=5)
    for i in range(0, len(labels), batch):
        out = m(torch.tensor(logits[i : i + batch]), torch.tensor(labels[i : i + batch]))
    return float(out["recall"]), float(out["accuracy"])


def test_matches_official_class_mean_recall(vjepa2_root, monkeypatch):
    rng = np.random.default_rng(0)
    n, c = 500, 60
    labels = rng.zipf(1.5, n) % c  # long-tailed, some classes absent
    logits = rng.normal(size=(n, c)).astype(np.float32)
    logits[np.arange(n), labels] += rng.normal(1.0, 1.0, n)  # partially informative

    ref_recall, ref_acc = _official_recall(monkeypatch, logits, labels, c)
    ours = mean_class_recall(topk_hits(logits, labels), labels, c)
    assert abs(ours["recall"] - ref_recall) < 1e-3
    assert abs(ours["accuracy"] - ref_acc) < 1e-3


def test_weights_equal_duplication():
    rng = np.random.default_rng(1)
    labels = rng.integers(0, 10, 50)
    hits = rng.random(50) > 0.5
    w = rng.integers(0, 3, 50)
    dup = np.repeat(np.arange(50), w)
    a = mean_class_recall(hits, labels, 10, weights=w)
    b = mean_class_recall(hits[dup], labels[dup], 10)
    assert abs(a["recall"] - b["recall"]) < 1e-9


def test_absent_classes_ignored():
    labels = np.array([0, 0, 2])
    hits = np.array([True, False, True])
    assert mean_class_recall(hits, labels, 5)["recall"] == 75.0
