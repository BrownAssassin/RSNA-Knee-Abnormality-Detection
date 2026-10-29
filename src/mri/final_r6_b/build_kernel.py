"""Build private scoring and UID-only runtime copies of the exact team source."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import core
from ..helios_s6_runtime_benchmark.build_r9_runtime import setup_cell

UPSTREAM_SHA256 = "74a24749eb18b7e35ad7d56d0d3c86e7f1fc3d6c685ffd21cfc31788450d797d"
SOURCE_URL = "https://www.kaggle.com/code/arvbali/rsna-r6-b-portable-final-bundle"
DATASETS = [
    "goodpjw2008/rsna-knee-2-5d-convnext-reader/1",
    "rostigor/rsna-knee-r6-fe055406/1",
    "nartaa/rsna-knee-publication-swa-weights-20261007/1",
    "heliosli/rsna-orthofoundation-fold0-epoch7-20261004/1",
    "arvbali/rsna-final-r6-assets-20261020/2",
]


def source_of(cell: dict) -> str:
    value = cell["source"]
    return "".join(value) if isinstance(value, list) else value


def replace_once(source: str, before: str, after: str) -> str:
    if source.count(before) != 1:
        raise ValueError(f"Expected exactly one source marker: {before[:100]!r}")
    return source.replace(before, after, 1)


def code_cell(source: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": source}


def guard_cell(mode: str) -> str:
    support_source = Path(core.__file__).read_text(encoding="utf-8")
    return f'''# Verify the exact team asset before any image work.
import os as _final_os, time as _final_time, json as _final_json
from pathlib import Path as _FinalPath
import torch as _final_torch
_final_started = _final_time.monotonic()
if not _final_torch.cuda.is_available():
    raise RuntimeError("This final inference run requires a Kaggle CUDA GPU")
_final_support = {{"__name__": "final_r6_b_support"}}
exec(compile({support_source!r}, "final_r6_b_support.py", "exec"), _final_support)
_final_assets = _final_support["resolve_asset_root"](_FinalPath("/kaggle/input"))
_final_asset_hashes = _final_support["verify_files"](_final_assets)
_final_os.environ["RSNA_INFERENCE_CODE_DIR"] = str(_final_assets)
_final_os.environ["RSNA_MODEL_CHECKPOINT"] = str(_final_assets / "model_weights.pt")
_final_os.environ["HF_HUB_OFFLINE"] = "1"
_final_os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
_final_os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
_final_receipt = {{"source_url": {SOURCE_URL!r}, "source_sha256": {UPSTREAM_SHA256!r},
    "mode": {mode!r}, "asset_version": 2, "r6_version": 1,
    "asset_sha256": _final_asset_hashes,
    "gpus": [_final_torch.cuda.get_device_name(i) for i in range(_final_torch.cuda.device_count())],
    "reports_read": False, "labels_read": False}}
(_FinalPath("/kaggle/working") / "pinned_input_receipt.json").write_text(_final_json.dumps(_final_receipt, indent=2))
print("FINAL_INPUTS_READY", _final_json.dumps(_final_receipt), flush=True)
'''


def audit_cell(mode: str, limit: int) -> str:
    return f'''# Audit the complete recipe without changing any prediction.
import pandas as _final_pd, numpy as _final_np
_final_work = _FinalPath("/kaggle/working")
_final_frame = _final_pd.read_csv(_final_work / "submission.csv", dtype={{"StudyInstanceUID": str}})
_final_ids = _rsna_sample.StudyInstanceUID.astype(str).tolist()
_final_support["validate_predictions"](_final_frame, _final_ids)
_final_cap = _final_support["check_correction"](_rsna_base, _rsna_final)
_final_repeat, _ = _rsna_adjust_predictions(_rsna_base, mode="threeway",
    gate=_RsnaGate(support_positive=0.90, uncertain_low=0.35, uncertain_high=0.65,
                   influence=0.25, max_adjustment=0.05))
_final_pd.testing.assert_frame_equal(_final_repeat, _rsna_final)
_final_compact = _final_json.loads((_final_work / "submission_receipt.json").read_text())
_final_blend = _final_json.loads((_final_work / "blend_receipt.json").read_text())
_final_aux = _final_json.loads((_final_work / "inference_summary.json").read_text())
_final_elapsed = _final_time.monotonic() - _final_started
_final_audit = dict(_final_receipt, status="PASS", integrity_status="PASS", rows=len(_final_ids),
    schema="exact thirteen columns", uid_coverage="exact", correction_deterministic=True,
    maximum_correction=_final_cap, elapsed_seconds=_final_elapsed,
    raptor_failed=_final_blend.get("raptor_failed"), compact_failed=_final_compact.get("fallback_studies"),
    r224_completed=bool(R224_OK), goodpjw_completed=bool(_final_own_ok),
    trained_convnext_failed=_final_aux.get("failed_studies"),
    final_sha256=_final_support["sha256"](_final_work / "submission.csv"))
if {mode!r} == "runtime" or len(_final_ids) <= 3:
    if (not R224_OK or not _final_own_ok or _final_audit["raptor_failed"] != 0
        or _final_audit["compact_failed"] != 0 or _final_audit["trained_convnext_failed"] != 0):
        raise RuntimeError("Qualification requires every intended reader and zero fallback rows")
if {mode!r} == "runtime":
    if len(_final_ids) != {limit}:
        raise RuntimeError("Incomplete label-blind runtime sample")
    _final_projected = _final_support["runtime_projection"](_final_elapsed, {limit})
    _final_audit.update(projected_test_rows=1300, margin_multiplier=1.2,
        initialization_also_scaled=True, projected_full_test_hours=_final_projected,
        projection_below_former_eight_hour_threshold=_final_projected < 8.0,
        runtime_projection_is_advisory=True)
    (_final_work / "submission.csv").rename(_final_work / "runtime_predictions.csv")
(_final_work / "final_execution_receipt.json").write_text(_final_json.dumps(_final_audit, indent=2))
print("FINAL_EXECUTION_RECEIPT", _final_json.dumps(_final_audit), flush=True)
'''


def build(source: Path, destination: Path, mode: str = "scoring", limit: int = 200) -> Path:
    if mode not in {"scoring", "runtime"} or limit <= 0:
        raise ValueError("Unsupported build mode or sample size")
    if core.sha256(source) != UPSTREAM_SHA256:
        raise ValueError("Team source changed; audit and pin it before rebuilding")
    notebook = json.loads(source.read_text(encoding="utf-8"))
    if len(notebook["cells"]) != 21:
        raise ValueError("Expected the reviewed twenty-one-cell portable B source")
    cells = notebook["cells"]
    for cell in cells:
        cell["source"] = source_of(cell)
        if cell["cell_type"] == "code":
            cell["outputs"], cell["execution_count"] = [], None
    if "BUNDLE_ID = 'fe055406fad6a12f'" not in cells[9]["source"]:
        raise ValueError("Expected the exact frozen R6 bundle")
    if "KNOWLEDGE_ENABLED = True" not in cells[20]["source"]:
        raise ValueError("Expected B with the fixed correction enabled")
    cells[18]["source"] = "_final_own_ok = False\n" + replace_once(
        cells[18]["source"], "\n    except Exception as _o_e:\n",
        "\n        _final_own_ok = True\n    except Exception as _o_e:\n")
    if mode == "runtime":
        for index in (3, 6):
            cells[index]["source"] = replace_once(cells[index]["source"], "def find_test_root():\n",
                'def find_test_root():\n    diagnostic_root = os.environ.get("RSNA_DIAGNOSTIC_DATA_ROOT")\n'
                '    if diagnostic_root:\n        return diagnostic_root\n')
        cells[9]["source"] = replace_once(cells[9]["source"], "DATA = find_mount(COMPETITION, 'test.csv')",
            'DATA = Path(os.environ["RSNA_DIAGNOSTIC_DATA_ROOT"])')
        cells[18]["source"] = replace_once(cells[18]["source"],
            '_o_assets, _o_data = _o_find("knee.py"), _o_find("test_series.csv")',
            '_o_assets, _o_data = _o_find("knee.py"), _o_os.environ["RSNA_DIAGNOSTIC_DATA_ROOT"]')
        cells[20]["source"] = replace_once(cells[20]["source"], "_rsna_sample_candidates = [\n",
            '_rsna_sample_candidates = [\n    _RsnaPath(__import__("os").environ["RSNA_DIAGNOSTIC_DATA_ROOT"]) / "sample_submission.csv",\n')
        marker = "_rsna_convnext_path = _rsna_work / 'convnext_predictions.csv'"
        cells[20]["source"] = replace_once(cells[20]["source"], marker,
            'import infer_convnext_submission as _final_inference\n'
            '_final_inference.COMPETITION_CANDIDATES = [_RsnaPath(__import__("os").environ["RSNA_DIAGNOSTIC_DATA_ROOT"])]\n' + marker)
    introduction = {"cell_type": "markdown", "metadata": {}, "source":
        f"# Private final R6 B {'scoring' if mode == 'scoring' else 'label-blind runtime check'}\n\n"
        f"Source: {SOURCE_URL}. Team source SHA-256 `{UPSTREAM_SHA256}`. "
        "Shared assets Version 2; R6 bundle Version 1. The existing MRI-only entry scored 0.958; "
        "this B variant has no independently returned score yet. Original attribution follows. "
        "The correction and model/blend settings are unchanged. Keep all outputs private."}
    prefix = [introduction, code_cell(guard_cell(mode))]
    if mode == "runtime":
        prefix.append(code_cell(setup_cell(limit, seed=20261021)))
    notebook["cells"] = prefix + cells + [code_cell(audit_cell(mode, limit))]
    notebook["metadata"]["final_r6_b"] = {"upstream_sha256": UPSTREAM_SHA256,
        "mode": mode, "sample_rows": limit if mode == "runtime" else None,
        "dataset_sources": DATASETS}
    destination.mkdir(parents=True, exist_ok=True)
    slug = "rsna-final-r6-b-20261021" if mode == "scoring" else "rsna-final-r6-b-runtime-20261021"
    path = destination / f"{slug}.ipynb"
    path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    metadata = {"id": f"mrinanksivakumar/{slug}",
        "title": "RSNA Final R6 B 20261021" if mode == "scoring" else "RSNA Final R6 B Runtime 20261021",
        "code_file": path.name, "language": "python", "kernel_type": "notebook",
        "is_private": True, "enable_gpu": True, "enable_tpu": False, "enable_internet": False,
        "dataset_sources": DATASETS, "kernel_sources": [],
        "competition_sources": ["rsna-knee-abnormality-detection"], "model_sources": [],
        "machine_shape": "NvidiaTeslaT4"}
    (destination / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    (destination / "build_receipt.json").write_text(json.dumps({
        "source_url": SOURCE_URL, "source_sha256": UPSTREAM_SHA256,
        "notebook_sha256": core.sha256(path), "mode": mode, "dataset_sources": DATASETS,
        "code_file": path.name}, indent=2) + "\n", encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--mode", choices=["scoring", "runtime"], default="scoring")
    parser.add_argument("--limit", type=int, default=200)
    args = parser.parse_args()
    print(build(args.source, args.destination, args.mode, args.limit))


if __name__ == "__main__":
    main()
