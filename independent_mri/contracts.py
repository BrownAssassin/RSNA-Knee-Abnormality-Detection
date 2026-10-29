from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np

TARGETS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
           "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
UID = "StudyInstanceUID"
CONFIGS = {"P1": (224, [8, 6, 6, 4, 8]), "P2": (336, [18, 14, 12, 8, 12]),
           "P3": (384, [26, 22, 18, 12, 18])}
PLANE_AXES = {"Sagittal": ([0, 1, 0], [0, 0, -1], [1, 0, 0]),
              "Coronal": ([1, 0, 0], [0, 0, -1], [0, 1, 0]),
              "Axial": ([1, 0, 0], [0, 1, 0], [0, 0, 1])}


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temp.replace(path)


def require_kaggle_gpu():
    import torch
    if not Path("/kaggle/working").is_dir() or not os.environ.get("KAGGLE_KERNEL_RUN_TYPE"):
        raise RuntimeError("Training/inference is authorized on Kaggle only; no local fallback")
    if not torch.cuda.is_available():
        raise RuntimeError("A Kaggle GPU is required for model training and inference")


def validate_predictions(frame, expected):
    if frame.columns.tolist() != [UID] + TARGETS:
        raise ValueError("Submission target order/schema mismatch")
    if frame[UID].duplicated().any() or frame[UID].tolist() != list(expected):
        raise ValueError("Submission UID coverage/order mismatch")
    values = frame[TARGETS].to_numpy(float)
    if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError("Nonfinite or out-of-range predictions")


def percentile_ranks(values):
    from scipy.stats import rankdata
    values = np.asarray(values)
    if not np.isfinite(values).all():
        raise ValueError("Cannot rank invalid predictions")
    return np.stack([rankdata(values[:, j], method="average") / len(values)
                     for j in range(values.shape[1])], axis=1)
