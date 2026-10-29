# Execution checkpoint — 2026-10-22, 04:40 UTC

User constraint: **all model training and submission inference on Kaggle GPUs**.
The local environment has CPU-only PyTorch for dummy-tensor contract tests;
training/inference entrypoints also reject non-Kaggle environments. No laptop
MRI fitting or inference has been performed. No WSL, paid compute or raw local
MRI download.

Standalone workspace: `D:\Repositories\RSNA-Independent-MRI`. It does not import
the team's source or private artifacts. A local Git repository was initialized
for safety checks; there is no remote, commit or push for this independent work.
The older shared repository is untouched by this research implementation.

## Completed evidence

- 38 CPU-only contract tests pass; compilation and `git diff --check` pass.
  No CPU test fits an MRI model or runs a real MRI encoder.
- Kaggle real-MRI numerical preflight v4: complete, FP16, two T4s, encoder update
  and strict initialization reload. All numerical smoke updates were discarded.
- Independent source-aware supervision v3: complete. 3,191 fitting, 570 monitor,
  583 comparison, 58 Gold plus five duplicate-report exclusions. Silence is masked.
- Independent 200-study cache benchmark: complete, 608.13 seconds, 2.19 GB;
  projected total around 45 GiB, kept on Kaggle.
- 200-study GPU multi-plane integration: complete, all cache hashes verified,
  6,076 windows finite, no substitutions; real eight-study backward/update check
  discarded. It is NOT a trained candidate or evidence of predictive accuracy.
- Two separate public anatomy screens: OAI DESS produced empty/sparse masks;
  SKM-TEA qDESS produced more foreground but one entirely empty case and missing
  structures in others. Neither is quality-approved or used for learning.
- Public S6 encoder audit: complete. Nartaa explicitly permits fine-tuning and
  competition use under CC BY-NC 4.0. Parent hash
  `0c7297d58f061ab4e5d7c55046f51fa3e8934928046f0647a0c7854e98371c12`;
  our encoder-only export hash
  `4381069ece2ce6cb5a02f193862e0b8994ca31c1d3d500f579e8741fe82b56a2`.
  All 340 encoder tensors strictly matched. No public classifier head or R6
  recipe is reused. Upstream MRI exposure means its adaptation diagnostics are
  not clean OOF; it is prohibited for our own OOF teachers.

## Running / waiting

Private cache kernels `rsna-independent-physical-cache-{a,b,c,d}-20261022` under
`mrinanksivakumar`: a/b/c COMPLETE at the last check; d RUNNING. Each reads the
official competition mount. Do not download their image outputs locally.

Two Windows orchestration watchers poll every five minutes. Their receipts/logs
are in ignored `private/orchestrator*.json` and `private/orchestrator*.log`.
After all four shards and current supervision complete, they launch:

- `rsna-independent-main-training-20261022`: ImageNet-only P1 control and P1
  pathology model, 12 epochs each, one model per T4.
- `rsna-independent-warm-training-20261022`: public-MRI-initialized P3 control
  and P3 pathology model, 12 epochs each, one model per T4.

Full candidate training had NOT started at this checkpoint. Initialization audits
and smoke steps must not be described as finished trained models.

`rsna-independent-pretraining-exposure-20261022` is COMPLETE: the released pool
has 4,263 nongold training studies, its Gold exclusions match official membership,
and 86 eligible cases are absent after duplicate-group checks. Our fixed split
places 68 of those in fitting, eight in monitor and ten in comparison. Warm
training receipts will count new fitting cases, and the ten unexposed comparison
cases will be reported separately with the small/source-selected-cohort caveat.
The SDK live-log RPC returned 404, but the official CLI uses a different HTTP
stream endpoint. That endpoint works. A bounded snapshot at 04:15 UTC verified
480/1,101 D-shard studies completed at the later 04:38 snapshot, without a
traceback or substitution. D is
roughly five times slower than the other shards but is making steady progress.
`monitor_logs.py` reads this stream with a bounded timeout. An idle read timeout
is not a remote-job error. `fetch_artifact.py` downloads one known output without
enumerating thousands of image-cache files; running outputs may return 404.

The small inference ensemble module is implemented with hash-pinned members,
fixed scalar weights and global cohort-level ranking. It is packaging machinery,
not evidence that a candidate or blend is good. No C/D recipe has been selected.
Per-epoch monitor predictions will be retained for matched-duration comparisons.
Full-cache auditing now rejects any renderer-source or source-order mismatch.

Private `rsna-independent-public-swa-benchmark-20261022` v1 is RUNNING on Kaggle
GPUs: 150/1,211 development studies at 04:36 UTC, about 3.3 seconds/study. It
strictly loaded the independently accessible, explicitly competition-permitted
Nartaa CoAtNet SWA pair. The 384-trained model uses the published native320
representation and plain/mirror probability mean; the 224 model uses its
published 0.8 crop/224 single view. No R6/R7 weights or teammate artifacts are
attached. These public models have upstream report-pool exposure and reused
Gold selection; the benchmark is NOT clean validation or a new trained model.

## Verified native background launchers

All poll every 300 seconds and perform no local model fitting/inference.

- Main: wrapper PID 2416, child 34692. Logs `private/orchestrator_v2.*.log`.
- Warm: wrapper PID 16420, child 19908. Logs `private/orchestrator_warm_v2.*.log`.
- Ablation A: wrapper PID 39368. Logs `private/orchestrator_ablation_a.*.log`.
- Ablation B: wrapper PID 28004. Logs `private/orchestrator_ablation_b.*.log`.
- CPU cache audit: wrapper PID 17268. Logs `private/orchestrator_cache_audit.*.log`.

The main/warm watchers were restarted safely after confirming their exact process
identities. They now use file-specific manifest reads and tolerate read outages.
Warm also explicitly waits for the upstream-exposure audit. Ambiguous interrupted
pushes require reconciliation; do not blindly relaunch.

The two ablation jobs wait for both priority candidate launch receipts, then for
remote GPU capacity. They cover the twelve required attention/branch/label/field
configurations with the SAME six-epoch schedule (not the earlier three-epoch
draft). Each kernel runs six models, one per GPU at a time. No ablation model is
trained yet. The full-cache CPU audit launches after all four shards complete.

Sampler randomness is now separate from model/dropout randomness, so matched
controls see identical study order. Checkpoints save sampler RNG state. The
own-model sagittal TTA helper was fixed BEFORE use: depth reflection occurs in
intensity space and then destination-channel ImageNet normalization is applied.
Reversing already-normalized RGB channels would be incorrect. A regression test
checks this; the public benchmark already mirrors uint8 before normalization.

Four tiny private SKM-TEA overlay panels were technically reviewed (no local
inference). Fragmentary masks and missing structures remain unapproved. This is
not expert segmentation validation. No anatomical/synthetic auxiliary loss is on.

`compare.py` computes CPU statistics on compact prediction banks: strict UID/
target alignment, per-target raw/rank/residual correlations, fixed 50/50 mean/rank
diagnostics and duplicate-report-cluster bootstrap. It never selects or submits
a model automatically. Trained-model results and Candidate C/D are still pending.

## Follow-up at 04:45 UTC

Public benchmark live progress: 275/1,211 at 04:43 UTC; both it and D were
re-polled as RUNNING with no failure message. All five launcher processes were
verified live. No full training or new official submission has started yet.

Added compact `best_manifest.json` sidecars to trained checkpoints so inspection
does not require local weight downloads. Inference now rejects changes to the
model, windowing, DICOM or contract source relative to the training manifest.
The builder can embed a fixed ensemble manifest in an offline private notebook;
its CPU packaging test performs no model execution or push.

Executed `supervision_audit.py` on the hash-matched 1.89-MB supervision bank:
all 21 source pairs, policy coverage/agreement, mention masks, Gold protection,
and duplicate-fold separation pass. An initially too-strict audit rejected the
intentional Gold/excluded-duplicate role distinction; it was corrected to treat
both as protected/nonfitting and a regression test was added. No real leakage
was found and no labels/splits were changed. The audit is in ignored
`private/current_supervision/source_conflict_audit.json`. See RESEARCH_NOTES.md
for the observed selective-coverage trade-off; no Gold-driven source tuning.

## Next work, still required

Verify launch receipts and real encoder-update receipts. Retrieve first epoch
metrics and actual throughput before deciding follow-up compute. Run the matched
ablation suite, clean ImageNet-only OOF teachers and matched students where
feasible. Run the full-cache metadata/re-render parity audit. Preserve negative
anatomy results; no invented masks or synthetic-anatomy claims. Compare actual
predictions with paired uncertainty, then package and submit two defensible,
technically distinct NEW candidates. No new official C/D submission exists yet.

The existing R6 B result is 0.958, not a C/D result. The goal remains active;
neither code completion nor one submission fulfills the full objective.

Loss correction before any full training: duplicate discounts are applied outside
per-study target normalization so they do not cancel for batch size one. A unit
regression test verifies the discount. Partial gradient-accumulation groups are
flushed and correctly normalized at the end of an epoch.

## Final-day update — 2026-10-22 15:34 UTC

All model training and MRI inference remains on Kaggle GPUs. The four matched
independent P1/P3 control/pathology models completed 12 epochs each on Kaggle;
their receipts confirm encoder updates and strict reload. Only compact receipts
and report-held prediction banks were downloaded for CPU-only statistical
analysis.

On the same 583-study report-held comparison cohort, the separately benchmarked
public Nartaa-384 mirror reference scored 0.960252 macro-AUC. Independent P1
models scored 0.832273/0.834945. Public-initialized P3 control/pathology models
scored 0.917316/0.911743. Fixed 50/50 global-rank blends with the reference
scored 0.950758/0.949713; their paired duplicate-group bootstrap 95% intervals
for delta versus reference were [-0.01697, -0.00405] and [-0.01849, -0.00455]
(0/1000 positive resamples for each). This is exploratory report-label evidence,
not clinical validation or a leaderboard estimate; the public initialization
has upstream training exposure. These independent candidates are excluded from
submission. A float32/float64 serialization check initially blocked comparison;
the comparator now permits at most 1e-7 numeric roundoff and has a regression
test that still rejects real label differences.

The two requested shared R6-Raptor variants are separate user-directed final-day
entries, not evidence for the independent research objective. The anatomy-
correction Kaggle GPU preview completed with 3/3 studies, no failed/fallback
rows, and the exact 13-column schema; its code submission is pending as ref
57530038. The three-study preview took 557.5 seconds, so full-test runtime still
needs confirmation against Kaggle's 9-hour execution limit. The no-anatomy
version was pushed as Kaggle version 1 but remains QUEUED behind remote GPU
capacity; its attempted code submission was rejected before acceptance, so the
last daily slot remains available. The three AtlasKnee submissions already
consumed the first three of five slots today. One independent ablation kernel
was RUNNING; it was canceled at 15:49 UTC in favor of the final-day Raptor
submissions. No local MRI training or inference was run.

## Final-day priority update — 2026-10-22 15:50 UTC

At the user's direction, the independent ablation-A Kaggle session was stopped
to prioritize the two requested R6-Raptor variants before the competition
deadline. Kaggle now reports `CANCEL_ACKNOWLEDGED`; the private notebook and
previous outputs were not deleted. Available GPU quota increased to 19.47 hours.
The no-anatomy Raptor notebook remains QUEUED, and anatomy-correction submission
57530038 remains PENDING. This interruption means the broad ablation suite is
unfinished and must not be represented as completed evidence. The independent
P1/P3 candidate comparisons remain negative as reported above; no new
independent candidate is currently qualified for submission.

## Independent continuation — 2026-10-22 16:17 UTC

Rechecked live state rather than relying on the earlier snapshot. All four
independent physical-cache kernels, the label job, full-cache audit, public-SWA
benchmark, main training, and warm training are COMPLETE. Ablation A is
CANCEL_ACKNOWLEDGED. No local experiment orchestrator is running. The ablation-B
receipt says `launched: false`; its watcher had exited during artifact retrieval
with a Windows sharing violation while publishing the shared supervision
manifest. It was not restarted while the two user-prioritized Raptor submissions
were pending.

Kaggle reports 19.33 GPU-hours remaining. Both Raptor submissions remain PENDING
(57530038 and 57531062). The three AtlasKnee and these two entries use all five
team submissions allowed today, so no independent C/D entry can be submitted
within today's remaining competition window. Neither Raptor result is evidence
for the independent research objective.

Added `analyze_existing_predictions.py` and
`independent_mri.evaluation.metrics_by_missing_views`. This is CPU-only analysis
of prior compact prediction banks. Its full report is
`private/missing_view_diagnostics_20261022.json`; interpretation and subgroup
support are recorded in RESEARCH_NOTES.md. Exact-mask paired bootstrap found the
P1 pathology branch uncertain (+0.00267 macro-AUC, 95% duplicate-group interval
[-0.00359, +0.00868]) and the P3 pathology branch worse (-0.00557, interval
[-0.00866, -0.00277]). Sparse missing-view subgroup AUCs are descriptive only.

Hardened `fetch_artifact.py` after the ablation-B orchestration exposed a shared
`.part`-file race on Windows: each download now uses a unique private sibling,
flushes before atomic publication, retries transient sharing violations, and
cleans its temporary file. Added retry/lock regression tests. Verification:
48 CPU-only tests passed; compileall and `git diff --check` passed. These checks
do not train or infer on MRI. The independent goal remains active and unmet; no
new independent model or submission has been qualified.
