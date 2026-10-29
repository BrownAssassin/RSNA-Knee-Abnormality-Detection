from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from .contracts import UID, sha256, write_json
from .dicom import read_series, select_series


def competition_root():
    roots = list(Path("/kaggle/input").glob("*/train.csv"))
    roots += list(Path("/kaggle/input").glob("competitions/*/train.csv"))
    roots = [p.parent for p in roots if (p.parent / "train_series.csv").is_file()]
    if len(roots) != 1:
        raise RuntimeError(f"Expected one official competition mount, got {roots}")
    return roots[0]


def cache_study(root, descriptors, study_uid, destination):
    target = destination / f"{study_uid}.npz"
    audit_path = destination / f"{study_uid}.json"
    if target.is_file() and audit_path.is_file():
        audit = json.loads(audit_path.read_text())
        if sha256(target) == audit["cache_sha256"]:
            return audit
        raise RuntimeError(f"Corrupt resumable cache {target}")
    selected = select_series(descriptors[descriptors[UID].eq(study_uid)])
    arrays, metadata = {}, []
    for slot, row in enumerate(selected):
        if row is None:
            metadata.append({"slot": slot, "missing": True})
            continue
        series_uid = str(row["SeriesInstanceUID"])
        stack, audit = read_series(root / "train_series" / study_uid / series_uid,
                                   expected_plane=row["Anatomical_Plane"])
        arrays[f"slot_{slot}"] = stack
        metadata.append({"slot": slot, "missing": False, "series_uid": series_uid,
                         "fluid": int(row["Fluid_Sensitive"]),
                         "fat_suppression": int(row["Fat_Suppression"]), **audit})
    if not arrays:
        raise ValueError(f"Study has no selected MRI acquisitions: {study_uid}")
    # Lossless compression of the explicitly quantized uint8 representation.
    # Quantization itself is lossy and is recorded, never called lossless DICOM conversion.
    with open(target.with_suffix(".tmp"), "wb") as handle:
        np.savez_compressed(handle, **arrays)
    target.with_suffix(".tmp").replace(target)
    audit = {UID: study_uid, "slots": metadata, "cache_sha256": sha256(target),
             "representation": "native ordered stacks; 384px,192mm; per-series p0.5/p99.5 uint8",
             "compression": "lossless NPZ after explicit uint8 quantization", "bytes": target.stat().st_size}
    write_json(audit_path, audit)
    return audit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    root = competition_root()
    train = pd.read_csv(root / "train.csv", dtype={UID: str})
    descriptors = pd.read_csv(root / "train_series.csv", dtype={UID: str, "SeriesInstanceUID": str})
    studies = sorted(train[UID].tolist())[args.shard::args.shards]
    if args.limit:
        studies = studies[:args.limit]
    out = Path("/kaggle/working/cache")
    out.mkdir(parents=True, exist_ok=True)
    started, receipts = time.time(), []
    def work(uid):
        return cache_study(root, descriptors, uid, out)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for receipt in pool.map(work, studies):
            receipts.append({UID: receipt[UID], "sha256": receipt["cache_sha256"], "bytes": receipt["bytes"]})
            if len(receipts) % 10 == 0:
                print(json.dumps({"completed": len(receipts), "total": len(studies),
                                  "elapsed_s": time.time() - started,
                                  "bytes": sum(r["bytes"] for r in receipts)}), flush=True)
            write_json("/kaggle/working/preparation_receipt.json", {
                "format": "independent-physical-mri-cache-v1", "complete": len(receipts) == len(studies),
                "shard": args.shard, "shards": args.shards, "expected_uids": studies,
                "source_train_sha256": sha256(root / "train.csv"),
                "series_descriptors_sha256": sha256(root / "train_series.csv"),
                "elapsed_seconds": time.time() - started, "receipts": receipts,
                "reports_exported": False, "errors_substituted": 0})


if __name__ == "__main__":
    main()
