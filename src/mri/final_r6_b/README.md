# Final R6 B packaging

Private copies of Arv's portable final notebook, with mount discovery and
execution receipts. The original inference and fixed medical correction are
preserved. The trained checkpoint is shared asset Version 2; the compact reader
is frozen R6 Version 1. The unused R7 attachment is omitted.

Download the exact upstream notebook with Kaggle CLI into ignored `kaggle_work/`.
Its source hash must be `74a24749eb18b7e35ad7d56d0d3c86e7f1fc3d6c685ffd21cfc31788450d797d`.
Then, with `PYTHONPATH=src`, run:

```powershell
.\.venv\Scripts\python.exe -m mri.final_r6_b.build_kernel --source <upstream.ipynb> --destination <private-scoring-directory>
.\.venv\Scripts\python.exe -m mri.final_r6_b.build_kernel --source <upstream.ipynb> --destination <private-runtime-directory> --mode runtime
```

The runtime copy selects 200 training UIDs without reports or target labels,
redirects every reader to the same virtual test directory, and renames the final
file to `runtime_predictions.csv`. Its conservative 1,300-study projection scales
the entire elapsed time (including initialization) and adds 20%. Following the
user's override, timing is advisory: it never blocks submission or makes an
otherwise valid diagnostic fail. Integrity failures still stop execution.

The scoring copy retains the official test input. Verify the saved input versions
on Kaggle as well as the embedded hashes, then submit the exact completed notebook
version through the code-competition CLI path. Preserve all upstream attribution;
never commit private artifacts or weights. The existing MRI-only entry scored
0.958; no new score is claimed by packaging this B candidate.

Execution, 21 October: the preview and complete 200-study integrity check passed.
The original runtime notebook stopped at the then-active timing gate:
5,178.348 seconds measured, 11.220 hours projected (9.350 without margin).
Those measurements and its ERROR are retained as historical facts. The user
subsequently removed that internal veto. The exact scoring **Version 1** was
submitted as **57485947** (scriptVersionId **360688364**) and is **PENDING**.
The scoring copy never had a forecast-based stop. Do not duplicate the entry.
Kaggle's actual execution limit and the parent's existing stage safeguards still
apply. All model, preprocessing, blending, and correction settings are unchanged.

A read-only native Windows observer polls every 300 seconds. Its current private
snapshot is `kaggle_work/final_r6_b_20261021/scoring_status/status.json`. To check
once without starting a second persistent observer:

```powershell
$env:PYTHONPATH = 'src'
.\.venv\Scripts\python.exe -m mri.final_r6_b.watch_submission --reference 57485947 --directory kaggle_work/final_r6_b_20261021/scoring_status --once
```

The observer only records submission status/score. Its lifetime limit affects the
local observer, never remote execution, and it cannot submit or cancel anything.

See [the aggregate result](../../../research/own_baseline/final_r6_b_20261021/RESULTS.md).
