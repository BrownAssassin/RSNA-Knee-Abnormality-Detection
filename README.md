# RSNA Knee Abnormality Detection

[![Python 3.12](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.11-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Kaggle Competition](https://img.shields.io/badge/Kaggle-RSNA%20Knee%20Abnormality%20Detection-20BEFF?logo=kaggle&logoColor=white)](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection)
[![Task](https://img.shields.io/badge/Task-12--label%20MRI%20classification-5B8C5A)](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection)
[![Best public AUC](https://img.shields.io/badge/Best%20public%20AUC-0.958-blue)](results/FINAL_RESULTS.md)
[![Final private AUC](https://img.shields.io/badge/Final%20private%20AUC-0.942-6f42c1)](results/FINAL_RESULTS.md)
[![Status](https://img.shields.io/badge/Competition-finished-lightgrey)](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection)

An MRI-only ensemble archive for the [RSNA Knee Abnormality Detection competition](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection). The challenge asked teams to estimate study-level probabilities for twelve knee MRI findings and ranked submissions by macro-averaged ROC AUC. Our two selected final submissions scored **0.958 public AUC** and **0.942 private AUC**.

This repository is both a competition archive and a portfolio record: it preserves the four highest-private-score and selected model notebooks, a reproducibility guide, the independent MRI research code developed during the sprint, and the final score receipts. Competition images, reports, trained weights, prediction files, and credentials are not included.

## Challenge and task

The [RSNA Knee Abnormality Detection AI Challenge](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection) asked participants to predict twelve study-level findings from knee MRI: ACL injury, MCL injury, medial and lateral meniscus injury, medial, lateral, and patellofemoral osteoarthritis, joint effusion, synovitis, Baker's cyst, bone contusion, and fracture. A study contains multiple DICOM series from one scanning session. Scanner, protocol, orientation, resolution, and intensity vary across international sites; series also use different transfer syntaxes, including JPEG Lossless and JPEG 2000. DICOM files retain an allowlisted set of 86 metadata tags. Only a small subset of studies has structured per-finding labels; radiology reports provide additional text paired with the wider imaging cohort. The challenge therefore supports image-only and multimodal training approaches, while each submission must return one confidence score per finding and study.

## Evaluation and efficiency track

The main leaderboard metric is the macro-average of ROC AUC over the twelve findings. `AUC_i` denotes the ROC AUC for finding `i`:

$$
\text{Main Score} = \frac{1}{12}\sum_{i=1}^{12} AUC_i
$$

Each finding contributes equally, regardless of prevalence. The submission contains `StudyInstanceUID` plus the twelve target columns. See the [competition evaluation and submission specification](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/overview#evaluation).

The competition also evaluated an efficiency track that combines private-test AUC and notebook runtime:

$$
\text{Efficiency} = \frac{\text{AUC}}{\text{Benchmark} - \max(\text{AUC})} + \frac{\text{RuntimeSeconds}}{32{,}400}
$$

Here, `AUC` is the submission's main metric on the private test set, `Benchmark` is the `sample_submission.csv` score, `max(AUC)` is the highest private-leaderboard AUC, and `RuntimeSeconds` is the notebook evaluation time. **Lower efficiency scores are better.** Eligibility also requires a selected (or automatically selected) final submission that beats the sample-submission benchmark on the private leaderboard. Kaggle code submissions were required to run offline and finish within nine hours. This repository reports our main leaderboard results; it does not claim an efficiency award or an efficiency-track rank. See the [official efficiency-track definition](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/overview#efficiency-prize-evaluation).

## Competition outcome

The four archived entries each returned **0.958 public / 0.942 private**. Kaggle marked the R6 B medical-correction entry and the Raptor no-anatomy-correction entry as the two final selections. Full references and caveats are in [the final results record](results/FINAL_RESULTS.md).

| Submission | Public AUC | Private AUC | Selected |
| --- | ---: | ---: | :---: |
| R6 Mirror R40 (compact-forward fixed blend) | 0.958 | 0.942 | No |
| R6 A (portable final bundle) | 0.958 | 0.942 | No |
| Final R6 B (fixed medical correction) | 0.958 | 0.942 | Yes |
| R6 Raptor (no anatomy correction) | 0.958 | 0.942 | Yes |

Scores are Kaggle leaderboard results, not internal validation metrics or clinical-performance claims. The final public leaderboard snapshot available to us reached 0.970; our private rank is not asserted here.

## Team

- **Arv Bali** — Kaggle [@arvbali](https://www.kaggle.com/arvbali), GitHub [@ArvBali2101](https://github.com/ArvBali2101). Shared and developed the R6/Raptor notebook family; original attribution is retained in the notebooks.
- **Mrinank Sivakumar** — Kaggle [@mrinanksivakumar](https://www.kaggle.com/mrinanksivakumar), GitHub [@BrownAssassin](https://github.com/BrownAssassin). Built independent research and verification tooling and submitted the final R6 B and Raptor no-anatomy entries.

## Model overview

The final systems are inference ensembles built from MRI readers rather than one newly trained end-to-end model. Depending on the entry, the notebooks combine Nartaa's Raptor/CoAtNet 384- and 224-pixel readers, Heliosli's compact R6 model with flip test-time augmentation, and the GoodPJW three-fold ConvNeXt reader. Fixed global rank blending is used; the selected R6 B variant applies a deterministic, bounded medical-rule correction, while the selected Raptor variant leaves MRI predictions unchanged.

The notebooks contain detailed method notes, upstream citations, and required acknowledgements. See [model sources and hashes](models/README.md) and [asset provenance](ASSET_PROVENANCE.md). These third-party readers are not presented as original work by this repository's team.

## Repository contents

```text
models/submission_notebooks/   Four Kaggle notebook source snapshots and metadata
independent_mri/               Independent pathology-evidence research implementation
src/mri/final_r6_b/             R6 B packaging, integrity contracts, and status observer
tests/                          CPU-side contracts for research and packaging code
docs/research/                  Independent experiment ledger and research notes
docs/archive/                   Dated execution snapshots from the final sprint
results/                        Final Kaggle score and selection record
requirements.txt                Local Windows CPU QA and Kaggle CLI dependencies
requirements-kaggle.txt         Reference packages for Kaggle-side research kernels
```

The independent research package was a separate final-day experiment. Its implementation and audit notes are retained for transparency, but that branch did not produce either of the two selected leaderboard entries. The sprint log describes what was and was not completed.

## Reproducing the archived submissions

The competition is closed, so the notebooks can be inspected and their inference rerun only where the required Kaggle inputs remain accessible; no new competition score can be obtained. Exact notebook links, input slugs, and source-snapshot SHA-256 hashes are listed in [`models/README.md`](models/README.md).

For local setup and contract tests, use native Windows Python 3.12 (no WSL):

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
pytest -q
```

Local commands are for editing, packaging, and CPU-side tests only. All MRI model inference or training must run on Kaggle GPU notebooks. To rerun an archived model, import its `.ipynb` into Kaggle, attach the competition input and the exact model datasets referenced in its metadata/source, pin the historical dataset versions in the Notebook Data panel, and use the documented GPU/runtime settings. Kaggle's source metadata snapshots may not encode historical dataset-version pins; do not assume that `latest` is equivalent. The `kernel-metadata.json` files are preserved source metadata, not ready-to-push templates for another Kaggle account.

The final inference notebooks reuse trained checkpoint assets; they do not retrain those checkpoints. Arv's [final ConvNeXt training notebook](https://www.kaggle.com/code/arvbali/rsna-final-convnext-single-run-20261019) and the [R6 A/B private asset dataset](https://www.kaggle.com/datasets/arvbali/rsna-final-r6-assets-20261020) are linked as training provenance. Assets must be retrieved from their original Kaggle sources under their own terms. For the exact team-side R6 B packaging, see [`src/mri/final_r6_b/README.md`](src/mri/final_r6_b/README.md) and the generated [scored notebook](models/submission_notebooks/r6_b_scored_version/rsna-final-r6-b-20261021.ipynb).

## Data and safe handling

The official [competition data page](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/data) hosts the MRI studies and labels. This GitHub repo intentionally contains no raw DICOMs, reports, study-level prediction dumps, model weights, caches, or Kaggle credentials. `.gitignore` excludes the common forms of these artifacts. External datasets and checkpoint files stay on Kaggle and retain their original attribution and terms.

## Research integrity

- Final inference is MRI-only; reports were used only by separate training-supervision experiments.
- Reused Gold-58 scores and report-held diagnostics are development evidence, not independent clinical validation.
- The competition score is not a diagnostic claim and this code is not intended for clinical use.
- Public source code and model assets retain their own provenance; no top-level license is asserted for all third-party components.

## Acknowledgements

The [Radiological Society of North America (RSNA)](https://www.rsna.org/artificial-intelligence/ai-image-challenge/knee-mri-ai-challenge) hosted and organized the challenge. We thank challenge co-leads Po-Hao “Howard” Chen and Naveen Subhas; the international imaging-site contributors; and the expert annotators and report/image reviewers, including the Society of Skeletal Radiology and International Skeletal Society communities. We also acknowledge MD.ai for annotation tooling and Kaggle for the competition platform. The [official acknowledgements](https://www.rsna.org/-/media/files/rsna/education/ai-resources-and-training/ai-image-challenge/2026-rsna-ai-challenge_knee-challenge-acknowledgment-page.pdf?hash=15BC97C9FCC91E33D0CC2D2C5D853F9F&rev=0e928b87a06b4fc4a769d9dcdb298ff0) list the full organizing, data contribution, annotation, and review teams.

We thank teammate Arv Bali for the shared R6/Raptor notebooks and model work, and acknowledge the upstream Kaggle authors whose readers, checkpoints, inference code, and blend ideas are credited inline, including Nartaa, Heliosli, GoodPJW2008, RostiGor, Varshitha Kolupuri, and SujanmajhiSuzan. Required OAI and model acknowledgements are preserved in the notebook source.
