# Literature and source audit

## Kenyon-inspired plausibility (not ordinary spatial attention)

The primary repository abstract for Kenyon et al. (MIUA 2024), *Synthetic Cerebral
Blood Vessel Generator for Training Anatomically Plausible Deep Learning Models*,
describes synthetic structures that obey or violate explicit geometric/topological
rules, and a learned assessor of segmentation plausibility. The repository lists
the contribution as not peer reviewed. This is a method inspiration, not evidence
that synthetic knee supervision improves classification.

Source: https://nottingham-repository.worktribe.com/output/40558593/synthetic-cerebral-blood-vessel-generator-for-training-anatomically-plausible-deep-learning-models

The White Rose PDF at https://eprints.whiterose.ac.uk/id/eprint/216395/8/9782832512449.pdf
only contains front matter and a different paper, despite its searchable contents
mentioning Kenyon's contribution. It is not a full copy of the requested paper.

The Adelaide web tool could not retrieve the thesis, but its public repository API
provided the actual 272-page PDF. Retrieved independently and read the motivation,
synthetic-generator overview, AQA methods and limitations (especially printed
pp. 114-122, 127-130 and 209). Primary PDF:
https://digital.library.adelaide.edu.au/server/api/core/bitstreams/c1c57ac4-2108-492c-8430-62f08cff6384/content .
The method trains frozen critics on controlled structure-space defects, separates
generated volumes across splits, and uses critic feedback during segmentation.
It does not establish that a normal-anatomy penalty is safe for real knee injuries.
For knees we would need trustworthy structures, corruption-matched controls,
pathology-preserving objectives and matched MRI prediction evidence. The attempted
DOSMA mask screen did not supply trustworthy structures, so no plausibility loss
is enabled. The whole thesis and the inaccessible 2024 contribution were not fully
read; the sections consulted are recorded rather than overstated.

## Anatomical model candidates

DOSMA publishes OAI DESS and SKM-TEA qDESS knee cartilage/meniscus segmenters:
https://dosma.readthedocs.io/en/latest/models.html and https://github.com/ad12/DOSMA .
The official documentation states GPLv3 for the models. These are sequence/domain
specific; no masks have yet been generated or quality-checked on this competition.
The actual six-study GPU screen produced empty/sparse masks and was not accepted.
No anatomy loss or synthetic plausibility signal is currently part of training.

Technical visual screen, 2026-10-22: reviewed the central overlay from four
SKM-TEA cases (numbers 1, 3, 4 and 5), generated entirely on Kaggle. The four PNG
panels total about 0.52 MiB and were inspected privately; no local model inference
was used. They show only small fragments, omit substantial structures, and one
femoral-cartilage-colored fragment is near the anterior patellar region. This is
a coarse engineering review, not expert segmentation validation or diagnosis.
It supports withholding anatomy supervision for this pipeline, not a claim that
the pretrained models are intrinsically poor. Full-volume expert review, native
floating-point preprocessing and matched domain evaluation remain unperformed.

KneePreM (https://arxiv.org/abs/2609.31461) describes OAI-pretrained 3D knee
representations. Public checkpoint availability and distribution terms have not
yet been established. A paper alone is not permission or a usable model.

## Initial encoder

https://huggingface.co/timm/convnext_small.fb_in22k_ft_in1k is published with an
Apache-2.0 model card, ImageNet-22k pretraining and ImageNet-1k fine-tuning.
It is not a knee-task checkpoint and has no declared RSNA Gold training exposure.
The actual initialization hash is retained in the private Kaggle preflight receipt.

## Label caveats

Omer's 98.04% / 99.14% describe accepted label cells on the reused 58-study
development panel, selected from 35 rankings. They do not certify full-pool
accuracy or MRI prediction quality. Both omit Synovitis, disclose source MRI
teacher exposure, and retain CC-BY-NC derived-data terms. JEV's hard swap is
Gold-selected and reuses Steven/Pilkwang. All are kept in the audit, not treated
as additional independent votes in the primary report-only fusion.

Source metadata was fetched independently from the public Kaggle dataset API;
private audit copies and hashes are excluded from Git.

## Executed source-conflict/coverage audit

The 1.89-MB compact supervision bank was downloaded privately for CPU statistics;
no MRI images, reports or models are used by this audit. It exactly matches the
v3 manifest hash. All 21 source-pair comparisons were aggregated on the fitting
partition only, without using Gold labels to choose sources or thresholds.
No unmentioned cell carries training weight. Report groups do not cross fitting,
monitor or comparison boundaries. One protected group has a Gold row plus five
excluded duplicate rows: distinct audit roles, not a fitting/evaluation leak.

Known fitting cells / positives / negatives:

| Finding | L1/L3 | L2 selective |
|---|---|---|
| ACL | 2,736 / 551 / 2,185 | 2,436 / 404 / 2,032 |
| Medial OA | 2,403 / 1,079 / 1,324 | 2,067 / 781 / 1,286 |
| Synovitis | 556 / 415 / 141 | 393 / 268 / 125 |
| Fracture | 2,011 / 189 / 1,822 | 1,915 / 102 / 1,813 |

Selective supervision increases vote agreement but removes nearly half of the
fracture-positive examples. This is not proof of better labels or MRI learning.
Among jointly addressed Synovitis cells, Steven/Pilkwang agree on polarity in
464/464 fitting cases; Lixin/Dread conflict in 681/3,185. Source dependence and
mention-selection bias prevent interpreting agreement as clinical accuracy.
Actual matched prediction results remain necessary.

## Missing-view performance and matched branch comparison — 2026-10-22

`analyze_existing_predictions.py` adds reproducible subgroup diagnostics over the
compact prediction banks already exported from Kaggle. It loads no images or
reports and performs no model fitting or inference. The report is stored at
`private/missing_view_diagnostics_20261022.json`; subgroup coverage includes
per-target known-cell counts, ROC-AUC, PR-AUC, Brier score versus hard report
labels, and precision/recall at 0.5. Duplicate-report-group bootstrap is used
for the matched overall branch deltas (1,000 replicates). Sparse missing-view
subgroups are descriptive, without unstable subgroup confidence intervals.

The P1/P3 pipeline marks 402/583 studies complete, 153 with one absent view and
28 with two. Public benchmark representations mark 415 complete and 168 with one
absent view. These masks are representation-specific; public-vs-independent
missing-view strata are not compared as though they were the same patients.

| Representation / model | Complete ROC-AUC / PR-AUC (n=402 or 415) | Any missing ROC-AUC / PR-AUC (n=181 or 168) |
|---|---:|---:|
| Independent P1 control | 0.8194 / 0.6508 | 0.8534 / 0.7113 |
| Independent P1 pathology branch | 0.8245 / 0.6527 | 0.8525 / 0.7164 |
| Independent P3 control | 0.9081 / 0.8120 | 0.9392 / 0.8729 |
| Independent P3 pathology branch | 0.9014 / 0.8036 | 0.9355 / 0.8682 |
| Public 224 plain | 0.9664 / 0.9321 | 0.9729 / 0.9504 |
| Public 384 plain | 0.9624 / 0.9224 | 0.9665 / 0.9392 |
| Public 384 published mirror TTA | 0.9594 / 0.9173 | 0.9639 / 0.9383 |

These report-held scores are not clinical validation. The public models have
upstream MRI/report-pool exposure and reused Gold selection; they are reference
benchmarks, not independent candidates. Their higher AUC on studies with missing
views is not evidence that missing anatomy helps; cohort composition and the
representation-specific mask differ.

The P1 pathology/control macro-AUC delta is +0.00267, with a duplicate-group
paired 95% interval of [-0.00359, +0.00868] and 0.739 positive bootstrap
fraction. It is uncertain and does not rescue P1's low absolute score. The P3
pathology/control delta is -0.00557 (95% interval [-0.00866, -0.00277], 0/1,000
positive resamples); 10 of 12 target AUC differences are non-positive, led by
Medial Meniscus (-0.0198), Synovitis (-0.0126) and PF OA (-0.0101). The current
evidence therefore rejects these P3 pathology-branch weights for a candidate;
the pattern should guide a later matched branch redesign rather than target-wise
Gold tuning.
