"""SAV (Student Action Video, Tan et al.) annotations. The format is AVA's.

annotations/{train,val}.csv           video_id, timestamp, x1, y1, x2, y2, action_id, person_id  (boxes normalised to [0, 1])
annotations/{train,val}_excluded_timestamps.csv   video_id, timestamp
annotations/{train,val}_person_det.pkl            {"video_id,0001": array (N, 5) [x1, y1, x2, y2, score]}
annotations/education_first_label.pbtxt           label map

`video_id` is the clip name from clips_to_keep.txt, e.g. "<name>_<part>_<index>", one 3 s mp4 per clip.
"""

import csv
import os
import pickle
import re
from collections import defaultdict

import numpy as np
import pandas as pd


def make_key(video_id, timestamp):
    return f"{video_id},{int(timestamp):04d}"


def split_key(key):
    video_id, ts = key.rsplit(",", 1)
    return video_id, int(ts)


def read_labelmap(path):
    """Return {label_id: name}. Accepts `label {name: .. label_id: ..}` and `item {name: .. id: ..}`."""
    text = open(path, encoding="utf-8").read()
    out = {}
    for block in re.findall(r"\{([^}]*)\}", text):
        name = re.search(r'name:\s*"([^"]*)"', block)
        lid = re.search(r"\b(?:label_)?id:\s*(\d+)", block)
        if name and lid:
            out[int(lid.group(1))] = name.group(1)
    if not out:
        raise ValueError(f"no labels found in {path}")
    return dict(sorted(out.items()))


def read_annotations(path):
    """Return {key: [(box (4,), sorted label ids)]}; rows of the same box are merged."""
    per_key = defaultdict(dict)
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.reader(f):
            if not row or len(row) < 7:
                continue
            key = make_key(row[0], row[1])
            box = tuple(round(float(v), 6) for v in row[2:6])
            per_key[key].setdefault(box, set()).add(int(row[6]))
    return {k: [(np.array(b, dtype=np.float32), sorted(lab)) for b, lab in boxes.items()] for k, boxes in per_key.items()}


def read_exclusions(path):
    if not path or not os.path.exists(path):
        return set()
    out = set()
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.reader(f):
            if len(row) >= 2:
                out.add(make_key(row[0], row[1]))
    return out


def read_proposals(path, score_thr=0.8):
    """Person boxes per key, filtered like mmaction's AVADataset: keep score >= min(thr, best score)."""
    with open(path, "rb") as f:
        raw = pickle.load(f, encoding="latin1")
    out = {}
    for key, arr in raw.items():
        arr = np.asarray(arr, dtype=np.float32).reshape(-1, arr.shape[-1] if np.ndim(arr) == 2 else 5)
        if len(arr) == 0:
            continue
        key = make_key(*split_key(key))
        if arr.shape[1] >= 5:
            thr = min(score_thr, float(arr[:, 4].max()))
            arr = arr[arr[:, 4] >= thr]
            out[key] = (arr[:, :4], arr[:, 4])
        else:
            out[key] = (arr[:, :4], np.ones(len(arr), dtype=np.float32))
    return out


def ann_paths(ann_dir, split):
    return dict(
        ann=os.path.join(ann_dir, f"{split}.csv"),
        excl=os.path.join(ann_dir, f"{split}_excluded_timestamps.csv"),
        props=os.path.join(ann_dir, f"{split}_person_det.pkl"),
        labelmap=os.path.join(ann_dir, "education_first_label.pbtxt"),
    )


def build_box_table(ann_dir, split, with_proposals=False, score_thr=0.8):
    """One row per box to encode: ground-truth boxes, and optionally detected person boxes.

    Columns: split, video_id, ts, key, x1, y1, x2, y2, source ('gt' | 'prop'), score, labels ('3;7', empty for proposals).
    Excluded keys are dropped. box_id is the row index.
    """
    p = ann_paths(ann_dir, split)
    gt = read_annotations(p["ann"])
    excl = read_exclusions(p["excl"])
    rows = []
    for key in sorted(gt):
        if key in excl:
            continue
        vid, ts = split_key(key)
        for box, labels in gt[key]:
            rows.append((split, vid, ts, key, *box.tolist(), "gt", 1.0, ";".join(map(str, labels))))
    if with_proposals and os.path.exists(p["props"]):
        props = read_proposals(p["props"], score_thr)
        for key in sorted(gt):
            if key in excl or key not in props:
                continue
            vid, ts = split_key(key)
            for box, s in zip(*props[key]):
                rows.append((split, vid, ts, key, *box.tolist(), "prop", float(s), ""))
    df = pd.DataFrame(rows, columns=["split", "video_id", "ts", "key", "x1", "y1", "x2", "y2", "source", "score", "labels"])
    df[["x1", "y1", "x2", "y2"]] = df[["x1", "y1", "x2", "y2"]].clip(0.0, 1.0)
    df.insert(0, "box_id", np.arange(len(df)))
    return df


def multi_hot(label_strings, label_ids):
    index = {lid: i for i, lid in enumerate(label_ids)}
    y = np.zeros((len(label_strings), len(label_ids)), dtype=np.float32)
    for r, s in enumerate(label_strings):
        if isinstance(s, str) and s:
            for lid in s.split(";"):
                if int(lid) in index:
                    y[r, index[int(lid)]] = 1.0
    return y
