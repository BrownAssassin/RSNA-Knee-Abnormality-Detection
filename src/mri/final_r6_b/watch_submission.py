"""Read-only monitoring of one already-submitted Kaggle entry; never submits/cancels."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import time

COMPETITION = "rsna-knee-abnormality-detection"


def snapshot(submission, reference: int) -> dict:
    if int(submission.ref) != reference:
        raise ValueError("Submission reference mismatch")
    status = str(submission.status).rsplit(".", 1)[-1].upper()
    raw_score = getattr(submission, "public_score", None)
    score = None
    if raw_score is not None and str(raw_score).strip():
        score = float(raw_score)
        if not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError("Unexpected public AUC value")
    error = getattr(submission, "error_description", None)
    if status == "COMPLETE" and score is not None:
        state = "SCORED"
    elif status in {"ERROR", "FAILED", "CANCELLED", "CANCELED"} or (status == "COMPLETE" and error):
        state = "FAILED"
    elif status == "COMPLETE":
        state = "AWAITING_SCORE"
    else:
        state = "PENDING"
    return {
        "format": "rsna-submission-observation-v1",
        "checked_utc": datetime.now(timezone.utc).isoformat(),
        "competition": COMPETITION,
        "reference": reference,
        "observer_pid": os.getpid(),
        "submission_status": status,
        "state": state,
        "public_score": score,
        "error_description": error,
        "notebook_url": getattr(submission, "url", None),
        "automatic_resubmission": False,
    }


def record(directory: Path, value: dict) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, indent=2) + "\n"
    temporary = directory / f"status.{os.getpid()}.tmp"
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(directory / "status.json")
    with (directory / "observations.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value) + "\n")
    print(json.dumps(value), flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=int, required=True)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--interval-seconds", type=float, default=300)
    parser.add_argument("--max-hours", type=float, default=36,
                        help="Local observer lifetime only; does not limit or cancel Kaggle execution")
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    if args.reference <= 0 or args.interval_seconds < 60 or args.max_hours <= 0:
        parser.error("Expected positive reference/lifetime and at least 60 seconds between API checks")
    repo = Path(__file__).resolve().parents[3]
    directory = args.directory.resolve()
    if not directory.is_relative_to(repo / "kaggle_work"):
        parser.error("Private monitoring artifacts must stay under ignored kaggle_work/")

    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    started = time.monotonic()
    failures = missing_scores = 0
    while time.monotonic() - started < args.max_hours * 3600:
        try:
            entries = api.competition_submissions(COMPETITION, page_size=100) or []
            item = next((x for x in entries if int(x.ref) == args.reference), None)
            if item is None:
                raise LookupError("Submission not present in recent results")
            value = snapshot(item, args.reference)
            failures = 0
            missing_scores = missing_scores + 1 if value["state"] == "AWAITING_SCORE" else 0
            if missing_scores >= 3:
                value["state"] = "COMPLETE_WITHOUT_SCORE"
            record(directory, value)
            if value["state"] in {"SCORED", "FAILED", "COMPLETE_WITHOUT_SCORE"} or args.once:
                return 0
        except Exception as exc:
            failures += 1
            record(directory, {
                "checked_utc": datetime.now(timezone.utc).isoformat(),
                "reference": args.reference, "state": "MONITOR_ERROR",
                "error_type": type(exc).__name__, "consecutive_errors": failures,
                "automatic_resubmission": False,
            })
            if failures >= 3 or args.once:
                return 2
        # This runs in a separate Windows process, not a blocking assistant wait.
        next_check = time.monotonic() + args.interval_seconds
        while time.monotonic() < next_check:
            time.sleep(min(5, max(0, next_check - time.monotonic())))
    record(directory, {
        "checked_utc": datetime.now(timezone.utc).isoformat(),
        "reference": args.reference, "state": "OBSERVER_WINDOW_ENDED",
        "remote_job_cancelled": False, "automatic_resubmission": False,
    })
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
