"""Mean-class recall@k exactly as `ClassMeanRecall` in vjepa2 (without torch.distributed)."""

import numpy as np


def topk_hits(logits, labels, k=5):
    """Boolean [N]: label is among the top-k logits. Sigmoid is monotonic, so it is skipped."""
    logits = np.asarray(logits, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int64)
    k = min(k, logits.shape[1])
    topk = np.argpartition(-logits, kth=k - 1, axis=1)[:, :k]
    return (topk == labels[:, None]).any(axis=1)


def mean_class_recall(hits, labels, num_classes, weights=None):
    """Returns dict(recall, accuracy) in percent.

    recall = mean over classes that appear (TP + FN > 0) of TP / (TP + FN).
    `weights` lets a sample count several times (see protocol.official_eval_multiplicity).
    """
    labels = np.asarray(labels, dtype=np.int64)
    hits = np.asarray(hits, dtype=bool)
    w = np.ones(len(labels)) if weights is None else np.asarray(weights, dtype=np.float64)
    tp = np.bincount(labels, weights=w * hits, minlength=num_classes)
    tot = np.bincount(labels, weights=w, minlength=num_classes)
    present = tot > 0
    return dict(
        recall=float(100.0 * np.mean(tp[present] / tot[present])),
        accuracy=float(100.0 * tp.sum() / tot.sum()),
        num_classes_present=int(present.sum()),
        num_samples=float(tot.sum()),
    )


def per_class_recall(hits, labels, num_classes):
    labels = np.asarray(labels, dtype=np.int64)
    tp = np.bincount(labels, weights=np.asarray(hits, dtype=np.float64), minlength=num_classes)
    tot = np.bincount(labels, minlength=num_classes)
    with np.errstate(invalid="ignore", divide="ignore"):
        return tp / tot, tot
