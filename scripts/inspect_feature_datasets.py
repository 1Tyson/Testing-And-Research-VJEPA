"""Report what a set of attached feature datasets contains (CPU only, read-only).

For every dataset under --root (Kaggle: /kaggle/input/datasets/<owner>/<slug>, /kaggle/input/<slug> or a notebook
output under /kaggle/input/notebooks/...): file types and sizes, then for a sample of array files the keys, shapes,
dtypes and NaN/Inf, and every clip / narration id found. Then across datasets: ids per dataset and protocol, ids
present in two datasets, coverage of the EK100 train / val segments (with --train_csv / --val_csv), and whether the
ids look like the clip ids of this repo (prepare_ek100.py / prepare_train_clips.py) or narration ids.

python scripts/inspect_feature_datasets.py --root /kaggle/input --train_csv EPIC_100_train.csv \
    --val_csv EPIC_100_validation.csv --out report.json
"""

import argparse
import glob
import json
import os
import re
import warnings
from collections import Counter, defaultdict

import numpy as np

ARRAY_EXT = (".npz", ".npy", ".pt", ".pth", ".safetensors", ".h5", ".hdf5")
ID_KEYS = ("clip_id", "clip_ids", "narration_id", "narration_ids", "uid", "uids", "ids", "id")
PROTOCOLS = ("action_start", "official")


def dataset_roots(root):
    """Directories that are one attached dataset / notebook output."""
    out = []
    for pattern in ("datasets/*/*", "notebooks/*/*", "competitions/*", "*"):
        for d in sorted(glob.glob(os.path.join(root, pattern))):
            if os.path.isdir(d) and os.path.basename(d) not in ("datasets", "notebooks", "competitions") \
                    and not any(d.startswith(o + os.sep) or o.startswith(d + os.sep) for o in out):
                out.append(d)
    return out


def protocol_of(path):
    p = path.lower()
    hits = [k for k in PROTOCOLS if k in p]
    return hits[0] if len(hits) == 1 else ("?" if not hits else "+".join(hits))


def describe_value(v):
    a = np.asarray(v)
    d = dict(shape=list(a.shape), dtype=str(a.dtype))
    if a.dtype.kind == "f" and a.size:
        s = a.astype(np.float32).ravel()[:2_000_000]
        d.update(nan=int(np.isnan(s).sum()), inf=int(np.isinf(s).sum()),
                 mean=float(np.nanmean(s)), std=float(np.nanstd(s)))
    return d


def load_arrays(path):
    """{key: array} for one file, or {"_error": msg}. Never executes pickled code."""
    try:
        if path.endswith(".npz"):
            with np.load(path, allow_pickle=False) as z:
                return {k: z[k] for k in z.files}
        if path.endswith(".npy"):
            return {"array": np.load(path, allow_pickle=False, mmap_mode="r")}
        if path.endswith((".pt", ".pth")):
            import torch

            obj = torch.load(path, map_location="cpu", weights_only=True)
            flat = {}

            def walk(o, pre=""):
                if isinstance(o, torch.Tensor):
                    flat[pre or "tensor"] = o.numpy() if o.dtype != torch.bfloat16 else o.float().numpy()
                elif isinstance(o, dict):
                    for k, v in o.items():
                        walk(v, f"{pre}.{k}" if pre else str(k))
                elif isinstance(o, (list, tuple)) and o and all(isinstance(x, (int, str)) for x in o[:10]):
                    flat[pre or "list"] = np.asarray(o)
                elif isinstance(o, (list, tuple)):
                    for i, v in enumerate(o[:50]):
                        walk(v, f"{pre}[{i}]")
            walk(obj)
            return flat
        if path.endswith(".safetensors"):
            from safetensors.numpy import load_file

            return load_file(path)
        if path.endswith((".h5", ".hdf5")):
            import h5py

            out = {}
            with h5py.File(path, "r") as f:
                f.visititems(lambda n, o: out.__setitem__(n, o[()]) if hasattr(o, "shape") else None)
            return out
    except Exception as e:  # report, keep going
        return {"_error": f"{type(e).__name__}: {e}"[:300]}
    return {}


def ids_from(arrays):
    for k, v in arrays.items():
        if k.split(".")[-1].lower() in ID_KEYS:
            a = np.asarray(v).ravel()
            return k, [x.decode() if isinstance(x, bytes) else (x.item() if hasattr(x, "item") else x) for x in a]
    return None, []


def main():
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="/kaggle/input")
    p.add_argument("--train_csv", default=None)
    p.add_argument("--val_csv", default=None)
    p.add_argument("--clip_index", nargs="*", default=[], help="val_clips.csv / train_clips.csv of this repo, if any")
    p.add_argument("--sample_files", type=int, default=3, help="files per dataset whose arrays are described")
    p.add_argument("--out", default="feature_datasets_report.json")
    args = p.parse_args()

    report = dict(datasets={})
    ids_by = {}  # (dataset, protocol) -> set of ids
    for d in dataset_roots(args.root):
        name = os.path.relpath(d, args.root)
        files = [os.path.join(r, f) for r, _, fs in os.walk(d) for f in fs]
        ext = Counter(os.path.splitext(f)[1].lower() or "(none)" for f in files)
        size = sum(os.path.getsize(f) for f in files)
        arr_files = sorted(f for f in files if f.endswith(ARRAY_EXT))
        info = dict(files=len(files), size_gb=round(size / 1e9, 2), extensions=dict(ext.most_common()),
                    top_dirs=sorted({os.path.relpath(f, d).split(os.sep)[0] for f in files})[:30],
                    protocols=dict(Counter(protocol_of(os.path.relpath(f, d)) for f in arr_files)), samples={})
        for f in arr_files[: args.sample_files]:
            a = load_arrays(f)
            info["samples"][os.path.relpath(f, d)] = {k: (describe_value(v) if k != "_error" else v) for k, v in a.items()}
        # ids from every array file (cheap for npz: only the id key is read)
        n_rows, id_key, errors = 0, None, 0
        for f in arr_files:
            if f.endswith(".npz"):
                try:
                    with np.load(f, allow_pickle=False) as z:
                        key = next((k for k in z.files if k.lower() in ID_KEYS), None)
                        vals = z[key].ravel().tolist() if key else []
                except Exception:
                    errors += 1
                    continue
            else:
                key, vals = ids_from(load_arrays(f))
            id_key = id_key or key
            n_rows += len(vals)
            ids_by.setdefault((name, protocol_of(os.path.relpath(f, d))), set()).update(map(str, vals))
        info.update(id_key=id_key, id_rows=n_rows, unreadable_files=errors,
                    unique_ids={proto: len(s) for (n, proto), s in ids_by.items() if n == name})
        # non-array metadata worth seeing
        for f in files:
            if f.endswith((".json", ".csv", ".txt", ".md")) and os.path.getsize(f) < 2_000_000 and len(info.setdefault("meta", {})) < 8:
                with open(f, errors="replace") as fh:
                    info["meta"][os.path.relpath(f, d)] = fh.read(600)
        report["datasets"][name] = info

    # cross-dataset checks
    keys = sorted(ids_by)
    overlaps = []
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            if a[1] == b[1]:
                n = len(ids_by[a] & ids_by[b])
                if n:
                    overlaps.append(dict(a=list(a), b=list(b), shared_ids=n))
    report["overlaps_same_protocol"] = overlaps
    per_proto = defaultdict(set)
    for (n, proto), s in ids_by.items():
        per_proto[proto] |= s
    report["unique_ids_per_protocol"] = {k: len(v) for k, v in per_proto.items()}

    # what the ids are
    all_ids = set().union(*ids_by.values()) if ids_by else set()
    narr = re.compile(r"^P\d{2}_\d{2,3}_\d+$")
    report["id_kind"] = dict(narration_like=sum(bool(narr.match(x)) for x in all_ids),
                             integer_like=sum(x.lstrip("-").isdigit() for x in all_ids), total=len(all_ids))
    import pandas as pd

    for split, csv in (("train", args.train_csv), ("val", args.val_csv)):
        if csv:
            nid = set(pd.read_csv(csv).narration_id.astype(str))
            report[f"{split}_segments"] = len(nid)
            report[f"{split}_narration_ids_covered"] = {k: len(v & nid) for k, v in per_proto.items()}
    for ci in args.clip_index:
        c = pd.read_csv(ci)
        cid = set(c.clip_id.astype(str))
        report[f"covered_by_{os.path.basename(ci)}"] = {k: len(v & cid) for k, v in per_proto.items()}

    with open(args.out, "w") as f:
        json.dump(report, f, indent=1, default=str)

    # readable summary
    print(f"{len(report['datasets'])} datasets under {args.root}\n")
    for name, info in report["datasets"].items():
        print(f"== {name}: {info['files']} files, {info['size_gb']} GB, ext {info['extensions']}")
        print(f"   array files by protocol (from path): {info['protocols']} | id key: {info['id_key']} | "
              f"rows {info['id_rows']} | unique ids {info['unique_ids']} | unreadable {info['unreadable_files']}")
        for f, arrs in info["samples"].items():
            print(f"   sample {f}:")
            for k, v in list(arrs.items())[:12]:
                print(f"      {k}: {v}")
    print("\nunique ids per protocol (all datasets):", report["unique_ids_per_protocol"])
    print("id kind:", report["id_kind"])
    for split in ("train", "val"):
        if f"{split}_segments" in report:
            print(f"{split}: {report[f'{split}_segments']} segments; narration ids covered: {report[f'{split}_narration_ids_covered']}")
    for k in report:
        if k.startswith("covered_by_"):
            print(k, report[k])
    if overlaps:
        print("ids present in two datasets of the same protocol:")
        for o in overlaps:
            print("  ", o)
    else:
        print("no id appears in two datasets of the same protocol")
    print(f"\nfull report: {args.out}")


if __name__ == "__main__":
    main()
