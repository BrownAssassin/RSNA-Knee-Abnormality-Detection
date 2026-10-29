import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.mri.final_r6_b import core
from src.mri.final_r6_b.build_kernel import DATASETS, guard_cell, audit_cell, replace_once


@pytest.mark.parametrize("layout", ["{slug}", "datasets/arvbali/{slug}", "datasets/arvbali/{slug}/2"])
def test_asset_mount_layouts(tmp_path, layout):
    root = tmp_path / layout.format(slug=core.ASSET_SLUG)
    root.mkdir(parents=True)
    for name in core.ASSET_HASHES:
        (root / name).write_bytes(b"fixture")
    assert core.resolve_asset_root(tmp_path) == root.resolve()


def test_assets_missing_ambiguous_and_hash_mismatch(tmp_path):
    with pytest.raises(FileNotFoundError):
        core.resolve_asset_root(tmp_path)
    (tmp_path / "model_weights.pt").write_bytes(b"expected")
    expected = {"model_weights.pt": hashlib.sha256(b"expected").hexdigest()}
    core.verify_files(tmp_path, expected)
    (tmp_path / "model_weights.pt").write_bytes(b"wrong checkpoint")
    with pytest.raises(ValueError, match="hash mismatch"):
        core.verify_files(tmp_path, expected)
    for layout in (core.ASSET_SLUG, f"datasets/arvbali/{core.ASSET_SLUG}"):
        root = tmp_path / layout
        root.mkdir(parents=True)
        for name in core.ASSET_HASHES:
            (root / name).write_bytes(b"fixture")
    with pytest.raises(FileNotFoundError, match="exactly one"):
        core.resolve_asset_root(tmp_path)


def frame():
    return pd.DataFrame([["a", *([0.5] * 12)], ["b", *([0.6] * 12)]], columns=[core.UID, *core.TARGETS])


def test_prediction_contracts():
    value = frame()
    core.validate_predictions(value, ["a", "b"])
    for invalid in (value.iloc[::-1], value.iloc[:1], value.drop(columns=["ACL"]),
                    value.assign(StudyInstanceUID=["a", "a"]), value.assign(ACL=np.nan),
                    value.assign(ACL=1.1)):
        with pytest.raises(ValueError):
            core.validate_predictions(invalid, ["a", "b"])


def test_correction_boundaries_and_unchanged_targets():
    base = frame()
    final = base.copy()
    final.loc[0, "Contusion"] += 0.05
    assert core.check_correction(base, final) == pytest.approx(0.05)
    assert core.check_correction(base, final) == core.check_correction(base, final.copy())
    final.loc[0, "Contusion"] += 0.001
    with pytest.raises(ValueError, match="cap"):
        core.check_correction(base, final)
    with pytest.raises(ValueError, match="unrelated"):
        core.check_correction(base, base.assign(ACL=base.ACL + 0.01))


def test_projection_and_packaging_guards():
    assert core.runtime_projection(3600) == pytest.approx(7.8)
    with pytest.raises(ValueError):
        core.runtime_projection(float("nan"))
    with pytest.raises(ValueError):
        replace_once("changed upstream", "marker", "replacement")
    assert all("r7" not in value for value in DATASETS)
    assert "arvbali/rsna-final-r6-assets-20261020/2" in DATASETS
    for mode in ("runtime", "scoring"):
        compile(guard_cell(mode), "guard", "exec")
        compile(audit_cell(mode, 200), "audit", "exec")


@pytest.mark.parametrize("elapsed,below_former_threshold", [(3600.0, True), (4000.0, False)])
def test_runtime_projection_is_advisory_after_user_override(tmp_path, elapsed, below_former_threshold):
    value = pd.DataFrame({core.UID: [f"fixture-{i}" for i in range(200)],
                          **{target: np.linspace(0.1, 0.9, 200) for target in core.TARGETS}})
    value.to_csv(tmp_path / "submission.csv", index=False)
    for name, data in [("submission_receipt.json", {"fallback_studies": 0}),
                       ("blend_receipt.json", {"raptor_failed": 0}),
                       ("inference_summary.json", {"failed_studies": 0})]:
        (tmp_path / name).write_text(json.dumps(data), encoding="utf-8")
    scope = {
        "_FinalPath": lambda _: tmp_path, "_final_json": json,
        "_final_time": SimpleNamespace(monotonic=lambda: elapsed), "_final_started": 0,
        "_final_receipt": {"mode": "runtime"},
        "_final_support": {name: getattr(core, name) for name in
                           ("validate_predictions", "check_correction", "runtime_projection", "sha256")},
        "_rsna_sample": value, "_rsna_base": value, "_rsna_final": value,
        "_rsna_adjust_predictions": lambda *args, **kwargs: (value.copy(), []),
        "_RsnaGate": lambda **kwargs: None, "R224_OK": True, "_final_own_ok": True,
    }
    exec(compile(audit_cell("runtime", 200), "runtime_audit", "exec"), scope)
    receipt = json.loads((tmp_path / "final_execution_receipt.json").read_text())
    assert receipt["integrity_status"] == "PASS"
    assert receipt["status"] == "PASS"
    assert receipt["runtime_projection_is_advisory"] is True
    assert receipt["projection_below_former_eight_hour_threshold"] is below_former_threshold
    assert (tmp_path / "runtime_predictions.csv").is_file()
    assert not (tmp_path / "submission.csv").exists()
