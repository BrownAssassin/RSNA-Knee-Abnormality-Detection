import json
from types import SimpleNamespace

import pytest

from src.mri.final_r6_b.watch_submission import record, snapshot


@pytest.mark.parametrize("status,score,error,expected", [
    ("SubmissionStatus.PENDING", None, None, "PENDING"),
    ("SubmissionStatus.COMPLETE", "0.958", None, "SCORED"),
    ("COMPLETE", "0", None, "SCORED"),
    ("COMPLETE", "", None, "AWAITING_SCORE"),
    ("COMPLETE", None, "Notebook Threw Exception", "FAILED"),
    ("ERROR", None, "Execution failed", "FAILED"),
])
def test_score_observations(status, score, error, expected):
    entry = SimpleNamespace(ref=57485947, status=status, public_score=score,
                            error_description=error, url="/code/example/model")
    value = snapshot(entry, 57485947)
    assert value["state"] == expected
    assert value["automatic_resubmission"] is False
    if expected == "SCORED":
        assert value["public_score"] == float(score)


def test_observer_rejects_wrong_reference_and_bad_score(tmp_path):
    entry = SimpleNamespace(ref=1, status="COMPLETE", public_score="NaN", error_description=None)
    with pytest.raises(ValueError, match="reference mismatch"):
        snapshot(entry, 2)
    with pytest.raises(ValueError, match="AUC"):
        snapshot(entry, 1)
    entry.public_score = "0.959"
    value = snapshot(entry, 1)
    record(tmp_path, value)
    assert json.loads((tmp_path / "status.json").read_text()) == value
    assert len((tmp_path / "observations.jsonl").read_text().splitlines()) == 1
    assert not list(tmp_path.glob("*.tmp"))
