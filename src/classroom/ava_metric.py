"""Frame-level mAP @ IoU 0.5, as in the SAV repo's ava_utils.ava_eval (AVA protocol).

Per class: detections on all non-excluded keyframes that have ground truth are matched greedily (by score) to
ground-truth boxes of that class with IoU >= 0.5; AP is the area under the interpolated PR curve (Pascal VOC style,
`object_detection.utils.metrics`). The ground-truth count includes excluded keys, like the original. Classes
without any detection are left out of the mean (nanmean over what the original reports).
"""

from collections import defaultdict

import numpy as np


def iou_matrix(a, b):
    a = np.asarray(a, dtype=np.float32).reshape(-1, 4)
    b = np.asarray(b, dtype=np.float32).reshape(-1, 4)
    ix1 = np.maximum(a[:, None, 0], b[None, :, 0])
    iy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix2 = np.minimum(a[:, None, 2], b[None, :, 2])
    iy2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-12), 0.0)


def average_precision(scores, tp, num_gt):
    if num_gt == 0:
        return float("nan")
    order = np.argsort(-scores, kind="stable")
    tp = tp[order].astype(np.float64)
    fp = 1.0 - tp
    ctp, cfp = np.cumsum(tp), np.cumsum(fp)
    precision = ctp / np.maximum(ctp + cfp, 1e-12)
    recall = ctp / num_gt
    recall = np.concatenate([[0.0], recall, [1.0]])
    precision = np.concatenate([[0.0], precision, [0.0]])
    for i in range(len(precision) - 2, -1, -1):
        precision[i] = max(precision[i], precision[i + 1])
    idx = np.where(recall[1:] != recall[:-1])[0] + 1
    return float(np.sum((recall[idx] - recall[idx - 1]) * precision[idx]))


def frame_map(gt, det, label_ids, excluded=(), iou_thr=0.5, score_thr=0.002):
    """gt:  {key: [(box (4,), [label ids])]}   (all keys, excluded ones included)
    det: {key: (boxes (N, 4), scores (N, C))}   columns of scores follow `label_ids`
    Scores below `score_thr` are dropped, like mmaction's AVAMetric(action_thr=0.002).
    Returns (mAP, {label_id: AP}).
    """
    excluded = set(excluded)
    gt_count = defaultdict(int)
    for items in gt.values():
        for _, labels in items:
            for lid in labels:
                gt_count[lid] += 1

    scores_per, tp_per = defaultdict(list), defaultdict(list)
    for key, items in gt.items():
        if key in excluded or key not in det:
            continue
        boxes, scores = det[key]
        boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
        for c, lid in enumerate(label_ids):
            s = np.asarray(scores[:, c], dtype=np.float32)
            gtb = np.array([b for b, labels in items if lid in labels], dtype=np.float32).reshape(-1, 4)
            keep = s >= score_thr
            s, b = s[keep], boxes[keep]
            order = np.argsort(-s, kind="stable")
            s, b = s[order], b[order]
            tp = np.zeros(len(s), dtype=bool)
            if len(gtb) and len(b):
                iou = iou_matrix(b, gtb)
                best = iou.argmax(1)
                taken = np.zeros(len(gtb), dtype=bool)
                for i in range(len(b)):
                    j = best[i]
                    if iou[i, j] >= iou_thr and not taken[j]:
                        tp[i] = taken[j] = True
            scores_per[lid].append(s)
            tp_per[lid].append(tp)

    ap = {}
    for lid in label_ids:
        if not scores_per[lid]:
            continue
        ap[lid] = average_precision(np.concatenate(scores_per[lid]), np.concatenate(tp_per[lid]), gt_count[lid])
    vals = [v for v in ap.values() if not np.isnan(v)]
    return (float(np.mean(vals)) if vals else float("nan")), ap
