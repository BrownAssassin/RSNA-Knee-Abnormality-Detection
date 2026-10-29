# Final R6 B execution - 21 October 2026

**Update, 21 October at 14:26 UTC: submitted as ref 57485947; PENDING.**
The user explicitly removed the internal timing veto after the initial hold.
The exact tested scoring Version 1 (scriptVersionId 360688364) was accepted
through Kaggle's code-competition CLI path with `submission.csv`. No recipe,
weights, or preprocessing changed. There is no returned score yet, and the
0.958 incumbent is untouched. Four daily submission slots remain.

The timing hold described below is historical, not a current instruction to
withhold this entry. Projections are now advisory; Kaggle's actual platform
limit still applies. The read-only five-minute observer writes latest status to
ignored `kaggle_work/final_r6_b_20261021/scoring_status/status.json`. Do not
duplicate the pending submission or confuse a preview's COMPLETE with scoring.

## What completed

The repository was fast-forwarded through 20 incoming commits after a named WIP
stash. Useful local experiments and tests were preserved and pushed in
`b838d25`; the final-B packaging is in `befa544`. The stash remains as a backup.

The private [scoring copy](https://www.kaggle.com/code/mrinanksivakumar/rsna-final-r6-b-20261021),
Version 1, completed its three-study preview in 632.187 seconds on two Tesla T4s.
The output SHA-256 exactly matches Arv's available A preview:
`a25ad964d0f69ec58b1285b4e638653729efaf0bf586a9de44f1669697604f30`.
The fixed B rule does not fire on those three studies. The latest available A
source, Version 3, differs from the upstream B source only in its correction
switch and added asset-discovery paths; this is not a claim to have downloaded
the hidden scoring outputs or measured A's hidden runtime.

The private [runtime copy](https://www.kaggle.com/code/mrinanksivakumar/rsna-final-r6-b-runtime-20261021),
Version 1, processed 200 deterministic training-image studies and 1,118 series,
selected using only UIDs (seed 20261021). It read no reports or target labels.
Every intended reader completed. Independent downloaded-artifact checks confirm
unique/exact UIDs, thirteen ordered columns, finite probabilities, matching
hashes, zero reader failures, and the original deterministic correction.
Eight studies received eight adjustments, with maximum movement 0.05; the other
nine targets are unchanged.

## Timing and decision

| Stage | Measured seconds |
| --- | ---: |
| Raptor 384 | 707.9 |
| Raptor 224 | 153, rounded log value |
| R6 controller | 2,990.0 |
| GoodPJW three-fold reader | 748, rounded log value |
| Shared trained ConvNeXt | 486.41 |
| Complete notebook, including setup and audits | 5,178.347638 |

The predeclared projection is
`elapsed_seconds * (1300 / 200) * 1.2 / 3600`, counting and scaling initialization
as well. It gives **11.219753 hours**, failing the **eight-hour** qualification
gate. Even omitting the margin gives **9.349794 hours**, above Kaggle's nine-hour
limit. These are sample-based projections, not measured full-test runtimes;
they neither prove that hidden execution would time out nor establish a score.
Arv's completed A result is retained, but its hidden runtime is unavailable.

Under the original policy, the runtime notebook deliberately raised
`Conservative complete-pipeline projection exceeds eight hours` after saving all
200 predictions. Its ERROR is therefore a known qualification stop, not an
unexplained inference exception. The legacy receipt's `status=PASS` describes
integrity only; `runtime_projection_pass=false` blocked submission at that time.
After the user's override, future diagnostics retain the projection as advisory
and no longer raise or withhold submission solely on that forecast.

No weight, crop, window, target, correction, or blend search was performed. The
initially suspected whole-tree scan was ruled out: the active runner already
installs `OneStudySource`. No redundant discovery patch was applied. Do not
alter model inputs to make the timing result look better. The later submission
was explicitly authorized despite the forecast, without modifying this recipe.

## Reproducibility and remaining resources

[The aggregate receipt](execution_receipt.json) records notebook versions,
input versions, source/checkpoint and prediction hashes, tests, and the hold.
Detailed logs, predictions, and UID metadata remain in ignored private
`kaggle_work/final_r6_b_20261021/`; none are committed. Images and reports stayed
on Kaggle; there was no new raw-data download, local image cache, fresh training,
R7 attachment, paid compute, or external report API.

Initial verification: **234 tests plus three subtests passed**, with three existing
pydicom deprecation warnings; `git diff --check` passed. The original preview and
runtime diagnostic are terminal; ref 57485947 is a separate hidden-scoring rerun,
currently PENDING. The trainer-only notebook does not write a submission CSV;
its already-trained checkpoint is used here instead of retraining. The deadline
is October 22, 23:59 UTC / 19:59 Toronto.

After the override, **241 tests plus three subtests pass**, covering advisory
timing and read-only submission monitoring. No public B score is claimed while
the entry remains pending.
