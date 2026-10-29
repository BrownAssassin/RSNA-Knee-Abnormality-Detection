# Independent experiment ledger

Started 2026-10-22 UTC. Deadline: October 22, 23:59 UTC. Existing R6 B finished
at 0.958; it is neither new Candidate C nor D. Kaggle-only GPU execution is a
hard user constraint. Initial quota: 28.37 GPU-hours. Actual notebook runtime
limits apply; internal forecasts are advisory, not automatic submission vetoes.

| Requirement | Status | Required evidence |
|---|---|---|
| Independent P1/P2/P3 physical DICOM pipeline | Implemented; three full shards complete, D running | Full 4,407-study cache and integrity audit still pending |
| Public source-aware labels L1-L4 | Version 3 COMPLETE | Source table, 3,191 train / 570 monitor / 583 comparison; Gold+5 duplicates excluded |
| Actual updated MRI encoder | Kaggle numerical smoke PASSED; no trained candidate yet | Tesla T4, stage-2 parameter gradient 0.2318, max update 0.00010347, strict initialization reload |
| Matched M1-M4 attention controls | Implemented; six-epoch launcher prepared | Same split/init/schedule, predictions and metrics still pending |
| C1-C5 focal/diffuse/global architecture | Implemented; not trained | Trained controls, target-wise metrics still pending |
| Kenyon primary works | Repository abstract and selected thesis methods/limitations read | No anatomy claims from attention; full thesis/paper not read |
| Eligible anatomy masks and quality review | Six-study GPU screen failed usability | DOSMA strict-load screen: 2/6 entirely empty; missing/sparse structures in others; all non-DESS |
| K1-K5 anatomy/plausibility controls | Not enabled: reliable supervision unavailable in tested screen | No synthetic examples or anatomical training claims |
| T1-T4 leakage-safe teacher/student | Conditional, not run | Own out-of-training teachers, matched student |
| Correct mirroring, resolution/complementarity | Not run | Matched predictions and bootstrap |
| Public benchmark exposure/license audit | S6 audit COMPLETE; public CoAtNet GPU benchmark RUNNING | Sources/hashes audited; comparative prediction evidence pending |
| Candidate C original system | Not trained/submitted | Saved offline notebook, receipt and actual score |
| Candidate D distinct system | Not trained/submitted | Distinct learned system, receipt and actual score |

Do not mark this research goal complete based on code, a preview, an old 0.958
entry, or one successful submission. Preserve negative results and explicitly
identify any experiments that time/resources prevent completing.

## Executed GPU preflight

Private Kaggle kernel `mrinanksivakumar/rsna-independent-real-mri-preflight-20261022`,
Version 4: COMPLETE. Two T4s detected. Real nongold training MRI only; no test
images used for backward. All numerical-smoke updates were discarded. Initial
checkpoint SHA-256: `c4ad2da51831278066da37165374325ce1beb6c9e3ba2950d0edfbab7d4f8563`.
FP16 steady GPU update time for 12 windows: 224px 0.1081s; 336px 0.1809s;
384px 0.2195s. These omit data loading and are not full-training runtime claims.
Peak allocated VRAM in this smoke: approximately 1.46 / 2.38 / 2.86 GB.

Earlier smoke failures are retained privately: Version 1 found an FP16 indexed
assignment dtype mismatch; Version 2 detected that the initial GradScaler step
did not update the encoder. Casting branch outputs to their destination dtype
and using an initial scale of 128 resolved the failures. No failed smoke became
a trained checkpoint or submission.

## Preprocessing benchmark

Independent 200-study CPU run: COMPLETE, 608.13 seconds, 2,194,948,431 compressed
bytes. Linear storage estimate: 45.04 GiB total / 11.26 GiB per full shard. This
estimate is not a storage guarantee for long-tail acquisitions. No image cache
was downloaded to the laptop. All 200 studies completed without substitutions.

## Anatomy feasibility result

Private Kaggle kernel `mrinanksivakumar/rsna-independent-anatomy-screen-20261022`,
Version 1: COMPLETE. Public DOSMA normalized IWOAI checkpoint SHA-256
`a175bf63f22c9c2308a0fc7d0fdb8b5ecf6cd2f875dde1de8f385dc4e0e91fd1` strictly loaded
and ran through the Keras torch backend on Kaggle CUDA. All six sampled sagittal
acquisitions were non-DESS. At the published 0.5 sigmoid threshold, two returned
no foreground in any structure; several others missed whole structures. These
masks are NOT approved for supervision. Domain/preprocessing differences remain
possible causes; this is not proof that anatomy supervision cannot help in general.
No synthetic anatomical examples or anatomy-guided training have been performed.

## 04:20 UTC execution update

Three full cache shards completed: A/B/C contain 3,306 unique studies and about
33.24 GiB combined. D is still running: live logs verified 410/1,101 completed at
04:15 UTC. It is slow but has not demonstrated failure. Full training has not
started; both native Windows orchestration watchers remain active.

Supervision v3 is complete, including explicit mention-gating for report silence.
The eligible public S6 encoder and its upstream exposure were independently
audited. Its released fitting pool excludes Gold but contains most of our report
monitor/comparison studies; warm-model diagnostics are not clean OOF. There are
86 duplicate-safe eligible upstream-unseen studies (68 fit, 8 monitor, 10 final
comparison), too few to establish full clinical generalization.

Both anatomy screens remain unapproved. The actual Kenyon thesis was retrieved;
selected method/limitation sections were read. No anatomy-guided or synthetic
plausibility training has occurred. See RESEARCH_NOTES.md for source caveats.

Before full fitting, duplicate discounts were moved outside per-study loss
normalization, and partial accumulation groups now flush at epoch end. The
current CPU tests verify the discount; no laptop GPU is used. Every full run
retains its own actual encoder-update assertion, strict reload and manifests.

## 04:40 UTC execution update

- 32 CPU-only tests pass, compilation and whitespace checks pass.
- D live progress is 480/1,101. Four-model candidate training still has not begun.
- The separately licensed public 384/224 CoAtNet benchmark is running: 150/1,211
  development studies at the last live snapshot, around 3.3 seconds/study. Strict
  checkpoint loading passed. This is frozen inference on Kaggle, not new learning.
- Prepared two SIX-epoch ablation kernels, twelve configurations total, with
  candidate-launch priority and GPU-capacity waiting. No ablation results yet.
- Fixed own-model sagittal TTA to reflect intensities before destination-channel
  ImageNet normalization. This was caught before training/inference evaluation.
- Four generated SKM-TEA overlay panels were technically inspected privately;
  they remain sparse/incomplete. No anatomy supervision is approved or trained.
- Added strict paired statistics and duplicate-report-cluster bootstrap on compact
  outputs. C/D selection and official submissions remain outstanding.
