"""Build a label-blind runtime diagnostic for the fixed S6 + R9 blend."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import nbformat


DEFAULT_SOURCE = Path(
    r"D:\rsna-r9-s6-current-pull\rsna-knee-helios-s6-r9-80-20-20261013.ipynb"
)
DEFAULT_DEST = Path(r"D:\rsna-r9-s6-current-pull\runtime-benchmark")
KERNEL_ID = "mrinanksivakumar/rsna-knee-s6-r9-label-blind-runtime-20261013"
KERNEL_TITLE = "RSNA Knee S6 R9 Label-Blind Runtime 20261013"
CODE_FILE = "rsna-knee-s6-r9-runtime-20261013.ipynb"
UID = "StudyInstanceUID"
TARGETS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
    "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
    "Contusion", "Fracture",
]


def project_runtime(total_elapsed: float, raptor_seconds: float,
                   compact_seconds: float, r9_seconds: float, *,
                   sample_rows: int = 200, full_rows: int = 1300,
                   margin: float = 1.5) -> dict[str, float]:
    """Scale measured inference stages while counting notebook setup only once."""
    values = (total_elapsed, raptor_seconds, compact_seconds, r9_seconds, margin)
    if (sample_rows < 1 or full_rows < 1 or margin < 1
            or any(not math.isfinite(float(value)) or float(value) < 0
                   for value in values)):
        raise ValueError("runtime inputs must be finite/nonnegative and dimensions positive")
    scalable = float(raptor_seconds) + float(compact_seconds) + float(r9_seconds)
    fixed = float(total_elapsed) - scalable
    if fixed < -1e-6:
        raise ValueError("measured stage times exceed total elapsed time")
    fixed = max(fixed, 0.0)
    scale = float(full_rows) / float(sample_rows)
    projected = (fixed + scalable * scale) * float(margin) / 3600.0
    return {
        "scalable_stage_seconds": scalable,
        "fixed_setup_seconds": fixed,
        "study_count_scale": scale,
        "margin_multiplier": float(margin),
        "projected_full_test_hours": projected,
    }


def setup_cell(limit: int = 200, seed: int = 20261013) -> str:
    if limit < 1:
        raise ValueError("limit must be positive")
    return f'''# Label-blind diagnostic setup: select training UIDs only, never labels or reports.
import json as _diag_json
import os as _diag_os
from pathlib import Path as _DiagPath
import numpy as _diag_np
import pandas as _diag_pd

_diag_candidates = [
    _DiagPath("/kaggle/input/competitions/rsna-knee-abnormality-detection"),
    _DiagPath("/kaggle/input/rsna-knee-abnormality-detection"),
]
_diag_comp = next((p for p in _diag_candidates
                   if (p / "train.csv").is_file() and (p / "train_series.csv").is_file()), None)
if _diag_comp is None:
    raise FileNotFoundError("Competition training tables were not found")
_diag_uid_table = _diag_pd.read_csv(
    _diag_comp / "train.csv", usecols=["{UID}"], dtype={{"{UID}": str}}
)
_diag_all_ids = _diag_uid_table["{UID}"].astype(str).tolist()
if len(_diag_all_ids) < {limit} or len(set(_diag_all_ids)) != len(_diag_all_ids):
    raise RuntimeError("Training UID table is too small or contains duplicate IDs")
_diag_rng = _diag_np.random.default_rng({seed})
_diag_ix = _diag_np.sort(_diag_rng.choice(len(_diag_all_ids), size={limit}, replace=False))
_diag_ids = [_diag_all_ids[int(i)] for i in _diag_ix]
_diag_series = _diag_pd.read_csv(
    _diag_comp / "train_series.csv",
    dtype={{"StudyInstanceUID": str, "SeriesInstanceUID": str}},
)
_diag_series = _diag_series[_diag_series.StudyInstanceUID.isin(_diag_ids)].copy()
if set(_diag_series.StudyInstanceUID.astype(str)) != set(_diag_ids):
    raise RuntimeError("Some sampled studies have no series metadata")
_diag_image_root = next(
    (p for p in (_diag_comp / "train_series", _diag_comp / "train_images") if p.is_dir()), None
)
if _diag_image_root is None:
    raise FileNotFoundError("Competition training DICOM directory is missing")
_diag_root = _DiagPath("/kaggle/working/rsna_s6_r9_runtime_input")
_diag_root.mkdir(parents=True, exist_ok=False)
_diag_pd.DataFrame({{"{UID}": _diag_ids}}).to_csv(_diag_root / "test.csv", index=False)
_diag_series.to_csv(_diag_root / "test_series.csv", index=False)
_diag_sample = _diag_pd.DataFrame({{"{UID}": _diag_ids}})
for _diag_target in {TARGETS!r}:
    _diag_sample[_diag_target] = 0.5
_diag_sample.to_csv(_diag_root / "sample_submission.csv", index=False)
_diag_os.symlink(_diag_image_root, _diag_root / "test_series", target_is_directory=True)
_diag_os.environ["RSNA_DIAGNOSTIC_DATA_ROOT"] = str(_diag_root)
_diag_receipt = {{
    "format": "rsna-s6-r9-label-blind-runtime-setup-v1",
    "rows": len(_diag_ids), "series_rows": len(_diag_series),
    "sample_seed": {seed}, "source_split": "training images; UID-only selection",
    "image_root_is_read_only_symlink": True,
    "reports_read": False, "target_labels_read": False,
}}
(_DiagPath("/kaggle/working") / "runtime_setup_receipt.json").write_text(
    _diag_json.dumps(_diag_receipt, indent=2), encoding="utf-8"
)
print("RUNTIME_SETUP", _diag_json.dumps(_diag_receipt), flush=True)
'''


def audit_cell(limit: int = 200, full_rows: int = 1300,
               margin: float = 1.5, max_hours: float = 8.0) -> str:
    if limit < 1 or full_rows < 1 or margin < 1 or max_hours <= 0:
        raise ValueError("invalid runtime audit parameters")
    return f'''# Validate label-blind complete predictions and conservatively project full-test runtime.
import json as _audit_json
import os as _audit_os
from pathlib import Path as _AuditPath
import numpy as _audit_np
import pandas as _audit_pd

_audit_work = _AuditPath("/kaggle/working")
_audit_setup = _audit_json.loads((_audit_work / "runtime_setup_receipt.json").read_text())
_audit_compact = _audit_json.loads((_audit_work / "submission_receipt.json").read_text())
_audit_base = _audit_json.loads((_audit_work / "blend_receipt.json").read_text())
_audit_r9 = _audit_json.loads((_audit_work / "r9_blend_receipt.json").read_text())
_audit_r9_inference = _audit_json.loads((_audit_work / "r9_inference_report.json").read_text())
_audit_pred = _audit_pd.read_csv(_audit_work / "runtime_benchmark_predictions.csv", dtype={{"{UID}": str}})
_audit_expected = _audit_pd.read_csv(
    _AuditPath(_audit_os.environ["RSNA_DIAGNOSTIC_DATA_ROOT"]) / "test.csv",
    dtype={{"{UID}": str}},
)
_audit_cols = ["{UID}", *{TARGETS!r}]
if _audit_setup.get("rows") != {limit} or _audit_setup.get("target_labels_read") is not False:
    raise RuntimeError("Setup receipt does not prove the expected label-blind sample")
if (_audit_compact.get("status") != "PASS" or _audit_compact.get("rows") != {limit}
        or _audit_compact.get("scored") != {limit} or _audit_compact.get("fallback_studies") != 0
        or any(code != 0 for code in _audit_compact.get("worker_exit_codes", {{}}).values())):
    raise RuntimeError("Compact inference has incomplete coverage, fallbacks, or failed workers")
if _audit_base.get("rows") != {limit} or _audit_base.get("raptor_failed") != 0 or _audit_base.get("ours_failed") != 0:
    raise RuntimeError("S6 stage has incomplete coverage or fallback rows")
if _audit_r9.get("status") != "PASS" or _audit_r9.get("rows") != {limit}:
    raise RuntimeError("R9 stage did not pass complete-sample inference")
if (_audit_r9_inference.get("ok") is not True or _audit_r9_inference.get("n_test") != {limit}
        or _audit_r9_inference.get("n_scored") != {limit} or _audit_r9_inference.get("n_missing") != 0
        or _audit_r9_inference.get("partial") is not False
        or _audit_r9_inference.get("deadline_hit") is not False):
    raise RuntimeError("R9 inference report shows incomplete or timed-out inference")
if _audit_r9.get("s6_fallback_rows") != 0 or _audit_r9.get("r9_missing_rows") != 0:
    raise RuntimeError("R9 blend contains fallback or missing rows")
if _audit_r9.get("reports_read") is not False or _audit_r9.get("labels_read") is not False or _audit_r9.get("test_labels_read") is not False:
    raise RuntimeError("R9 stage read a forbidden report or label source")
if _audit_pred.columns.tolist() != _audit_cols:
    raise RuntimeError("Runtime prediction schema is not the exact 13-column contract")
if _audit_pred["{UID}"].astype(str).tolist() != _audit_expected["{UID}"].astype(str).tolist():
    raise RuntimeError("Runtime prediction UID order or coverage is incomplete")
_audit_values = _audit_pred[{TARGETS!r}].to_numpy(dtype=_audit_np.float64)
if not _audit_np.isfinite(_audit_values).all() or ((_audit_values < 0) | (_audit_values > 1)).any():
    raise RuntimeError("Runtime predictions contain invalid probabilities")
_audit_elapsed = float(_audit_r9["elapsed_since_notebook_start_seconds"])
_audit_scalable = (
    float(_audit_base["raptor_seconds"])
    + float(_audit_compact["seconds"])
    + float(_audit_r9_inference["runtime_s"])
)
_audit_fixed_setup = _audit_elapsed - _audit_scalable
if _audit_fixed_setup < 0:
    raise RuntimeError("Stage timings exceed total notebook elapsed time")
_audit_projected_hours = (
    (_audit_fixed_setup + _audit_scalable * ({full_rows} / {limit})) * {margin} / 3600.0
)
_audit_result = {{
    "format": "rsna-s6-r9-runtime-projection-v1",
    "sample_rows": {limit}, "projected_test_rows": {full_rows},
    "measured_elapsed_seconds": _audit_elapsed, "margin_multiplier": {margin},
    "scalable_stage_seconds": _audit_scalable,
    "fixed_setup_seconds": _audit_fixed_setup,
    "projected_full_test_hours": _audit_projected_hours,
    "max_hours": {max_hours}, "projection_pass": _audit_projected_hours < {max_hours},
    "uid_coverage": "exact", "schema": "exact 13 columns", "probabilities_valid": True,
    "reports_read": False, "target_labels_read": False, "test_labels_read": False,
    "raptor_fallback_rows": 0, "compact_fallback_rows": 0, "r9_missing_rows": 0,
}}
(_audit_work / "runtime_projection_receipt.json").write_text(
    _audit_json.dumps(_audit_result, indent=2), encoding="utf-8"
)
print("RUNTIME_PROJECTION", _audit_json.dumps(_audit_result), flush=True)
if not _audit_result["projection_pass"]:
    raise RuntimeError("Conservative full-test runtime projection exceeds the eight-hour gate")
'''


def build(source: Path, destination: Path, *, limit: int = 200,
          seed: int = 20261013) -> Path:
    notebook = nbformat.read(source, as_version=4)
    if len(notebook.cells) != 9:
        raise ValueError("Expected the audited nine-cell R9 + S6 scoring notebook")
    full_source = "\n".join(cell.source for cell in notebook.cells)
    if "gold_labels.csv" in full_source:
        raise ValueError("Runtime-only source must not read Gold labels")
    if "Version 13, bundle 209a32c8b50a6b1c" not in notebook.cells[2].source:
        raise ValueError("Source does not verify the pinned Version 13 compact bundle")
    if "W_RAPTOR, W_OURS = 0.5, 0.5" not in notebook.cells[7].source:
        raise ValueError("Expected the audited fixed 50/50 S6 component")
    if ("BASE_WEIGHT = 0.80" not in notebook.cells[8].source
            or "R9_WEIGHT = 0.20" not in notebook.cells[8].source
            or "R9_PACKAGE_SHA256" not in notebook.cells[8].source):
        raise ValueError("Expected the hash-checked fixed 80/20 R9 blend")

    # Cell 2 retains all source/hash checks but resolves the competition by its
    # training table; the diagnostic alias is created from UID/series columns only.
    c2 = notebook.cells[2]
    token = "_checked_mount('rsna-knee-abnormality-detection', 'test.csv')"
    if c2.source.count(token) != 1:
        raise ValueError("Could not identify the single competition input check")
    c2.source = c2.source.replace(
        token,
        "_checked_mount('rsna-knee-abnormality-detection', 'train.csv')\n\n" + setup_cell(limit, seed),
        1,
    )

    # Redirect both inference engines to the same diagnostic alias.
    c3 = notebook.cells[3]
    root_marker = "def find_test_root():\n"
    if c3.source.count(root_marker) != 1:
        raise ValueError("Could not identify the Raptor test-root resolver")
    c3.source = c3.source.replace(
        root_marker,
        root_marker
        + '    diagnostic_root = os.environ.get("RSNA_DIAGNOSTIC_DATA_ROOT")\n'
        + '    if diagnostic_root:\n'
        + '        return diagnostic_root\n',
        1,
    )

    c5 = notebook.cells[5]
    data_marker = "DATA = find_mount(COMPETITION, 'test.csv')"
    if c5.source.count(data_marker) != 1:
        raise ValueError("Could not identify the S6 data-root assignment")
    c5.source = c5.source.replace(
        data_marker,
        'DATA = Path(os.environ["RSNA_DIAGNOSTIC_DATA_ROOT"]) if os.environ.get("RSNA_DIAGNOSTIC_DATA_ROOT") else find_mount(COMPETITION, \'test.csv\')',
        1,
    )

    c7 = notebook.cells[7]
    output_marker = "WORK / 'submission.csv'"
    if c7.source.count(output_marker) != 1:
        raise ValueError("Expected one S6 output named submission.csv")
    c7.source = c7.source.replace(output_marker, "WORK / 'runtime_benchmark_predictions.csv'", 1)
    c7.source = c7.source.replace(
        "the blend cell writes submission.csv", "the diagnostic writes runtime_benchmark_predictions.csv"
    )

    c8 = notebook.cells[8]
    if c8.source.count('WORK / "submission.csv"') != 1:
        raise ValueError("Expected one R9 base-output path")
    if c8.source.count('WORK / "submission.csv.tmp"') != 1:
        raise ValueError("Expected one R9 temporary-output path")
    c8.source = c8.source.replace(
        'WORK / "submission.csv.tmp"', 'WORK / "runtime_benchmark_predictions.csv.tmp"', 1
    ).replace('WORK / "submission.csv"', 'WORK / "runtime_benchmark_predictions.csv"', 1)

    notebook.cells[0].source = (
        "# Private label-blind 200-study S6 + R9 runtime benchmark\n\n"
        "This diagnostic runs the fixed 80% S6 / 20% R9 blend on a deterministic "
        "sample of training images selected from UIDs only. It does not read reports "
        "or labels and never writes `submission.csv`. It is not a competition submission."
    )
    notebook.cells.append(nbformat.v4.new_code_cell(audit_cell(limit)))

    for cell in notebook.cells:
        if cell.cell_type == "code":
            cell.outputs = []
            cell.execution_count = None
    notebook.metadata["runtime_benchmark"] = {
        "purpose": "label-blind exact-pipeline runtime/integrity diagnostic",
        "sample_rows": limit,
        "sample_seed": seed,
        "blend": {"s6_rank": 0.8, "r9_rank": 0.2},
        "dataset_version": 13,
    }

    destination.mkdir(parents=True, exist_ok=True)
    notebook_path = destination / CODE_FILE
    nbformat.write(notebook, notebook_path)
    metadata = {
        "id": KERNEL_ID,
        "title": KERNEL_TITLE,
        "code_file": CODE_FILE,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": False,
        # The CLI request is versioned, but Kaggle normalizes the pulled
        # metadata to an unversioned slug. The notebook's bundle/hash check is
        # fail-closed; a durable UI pin is still required for scoring.
        "dataset_sources": [
            "heliosli/rsna-knee-compact-v1/13",
            "nartaa/rsna-knee-publication-swa-weights-20261007",
            "heliosli/rsna-orthofoundation-fold0-epoch7-20261004",
            "bliverrigpdoge/rsna-knee-leg-r9all",
        ],
        "kernel_sources": [],
        "competition_sources": ["rsna-knee-abnormality-detection"],
        "model_sources": [],
        "machine_shape": "NvidiaTeslaT4",
    }
    (destination / "kernel-metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return notebook_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--destination", type=Path, default=DEFAULT_DEST)
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20261013)
    args = parser.parse_args()
    print(build(args.source, args.destination, limit=args.limit, seed=args.seed))


if __name__ == "__main__":
    main()
