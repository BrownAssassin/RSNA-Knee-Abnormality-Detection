"""Small, independently testable contracts shared by preview and runtime jobs."""
from __future__ import annotations

import hashlib
from pathlib import Path

UID = "StudyInstanceUID"
TARGETS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
           "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
ASSET_SLUG = "rsna-final-r6-assets-20261020"
ASSET_HASHES = {
    "model_weights.pt": "23a932a48872fb980e4593f2a35988770a7c560f59061b37fdfaf3f0a69b265c",
    "train_convnext.py": "86a5981ffdc4c2599a63056a0d8eb8161f8c1256cf7d729e90f30fdf2ab96b5d",
    "infer_convnext_submission.py": "49a6ec5d3d457f25f5addc18e839b5862fd75627626f03737024a14e6cfe08e0",
    "uncertainty_corrector.py": "861600d303744cbae7844bf823601df4c01d9f14559abbc31b509f387cdd5218",
    "relationship_rules.json": "a742a7b677b3e8f46a3f5bdaefa8c178e74549c886d4ae18b0fc3004804d207f",
}


def sha256(path: Path) -> str:
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def resolve_asset_root(input_root: Path) -> Path:
    root = Path(input_root)
    candidates = [root / ASSET_SLUG, root / "arvbali" / ASSET_SLUG,
                  root / "datasets" / "arvbali" / ASSET_SLUG]
    for pattern in (f"datasets/*/{ASSET_SLUG}", f"datasets/*/{ASSET_SLUG}/*",
                    f"*/{ASSET_SLUG}", f"{ASSET_SLUG}/*"):
        candidates.extend(root.glob(pattern))
    found = {p.resolve() for p in candidates
             if all((p / name).is_file() for name in ASSET_HASHES)}
    if len(found) != 1:
        raise FileNotFoundError(f"Expected exactly one complete pinned asset mount; found {sorted(map(str, found))}")
    return found.pop()


def verify_files(root: Path, expected: dict[str, str] = ASSET_HASHES) -> dict[str, str]:
    observed = {name: sha256(Path(root) / name) for name in expected}
    for name, digest in expected.items():
        if observed[name] != digest:
            raise ValueError(f"Pinned asset hash mismatch: {name}")
    return observed


def validate_predictions(frame, expected_uids: list[str]) -> None:
    import numpy as np

    if list(frame.columns) != [UID, *TARGETS]:
        raise ValueError("Expected exactly thirteen competition columns in order")
    ids = frame[UID].astype(str).tolist()
    if len(ids) != len(set(ids)) or ids != list(map(str, expected_uids)):
        raise ValueError("Missing, duplicate, extra, or reordered study IDs")
    values = frame[TARGETS].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError("Predictions must be finite probabilities in [0, 1]")


def check_correction(base, final, cap: float = 0.05) -> float:
    import numpy as np

    expected = base[UID].astype(str).tolist()
    validate_predictions(base, expected)
    validate_predictions(final, expected)
    delta = np.abs(final[TARGETS].to_numpy(float) - base[TARGETS].to_numpy(float))
    maximum = float(delta.max(initial=0))
    if maximum > cap + 1e-7:
        raise ValueError("Medical adjustment exceeded its fixed per-output cap")
    unchanged = [t for t in TARGETS if t not in {"Contusion", "Effusion", "Baker's"}]
    if not np.array_equal(base[unchanged].to_numpy(), final[unchanged].to_numpy()):
        raise ValueError("Correction changed an unrelated target")
    return maximum


def runtime_projection(elapsed_seconds: float, sample_rows: int = 200,
                       full_rows: int = 1300, margin: float = 1.2) -> float:
    import math

    if (not math.isfinite(elapsed_seconds) or elapsed_seconds <= 0
            or sample_rows <= 0 or full_rows <= 0 or margin < 1):
        raise ValueError("Invalid projection inputs")
    # Deliberately conservative: also scales initialization and all audits.
    return elapsed_seconds * full_rows / sample_rows * margin / 3600
