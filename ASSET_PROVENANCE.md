# Asset provenance and attribution

This project combines team-authored integration with public Kaggle notebooks, pretrained models, and checkpoint datasets. The original notebooks are retained with their inline citations and acknowledgements. This file summarizes the main attributions; the notebook and dataset pages remain authoritative for the exact asset version and its terms.

| Asset or contribution | Role in archived methods | Source / attribution |
| --- | --- | --- |
| RSNA Knee Abnormality Detection data | Competition MRI and target schema | [Kaggle competition](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection) |
| Raptor / CoAtNet 384 and 224 readers | MRI study predictions and mirror TTA | [Nartaa 0.949 notebook](https://www.kaggle.com/code/nartaa/rsna-knee-0949-anatomical-mirror), [Nartaa weight dataset](https://www.kaggle.com/datasets/nartaa/rsna-knee-publication-swa-weights-20261007), with upstream attribution in the notebooks |
| Helios R6 compact reader and flip TTA | Compact model component | [Helios R6 notebook](https://www.kaggle.com/code/heliosli/this-is-new-version-0-9576-in-one-click), [R6 dataset copy](https://www.kaggle.com/datasets/rostigor/rsna-knee-r6-fe055406) |
| Per-finding blend recipe and reader changes | Fixed rank-fusion recipe in the R6 family | [Helios 0.958 notebook](https://www.kaggle.com/code/heliosli/blend-mine-in-private-lb-luck-to-all-upvoters), [SujanmajhiSuzan](https://www.kaggle.com/sujanmajhisuzan) |
| ConvNeXt-Small reader | Auxiliary three-fold MRI reader in A/B notebooks | [GoodPJW dataset](https://www.kaggle.com/datasets/goodpjw2008/rsna-knee-2-5d-convnext-reader), [Apache-2.0 initialization](https://huggingface.co/timm/convnext_small.fb_in22k_ft_in1k) |
| R6 B trained model assets | Shared final checkpoint and fixed-rule implementation | [Arv's R6 assets, Version 2](https://www.kaggle.com/datasets/arvbali/rsna-final-r6-assets-20261020), [Arv's portable R6 B notebook](https://www.kaggle.com/code/arvbali/rsna-r6-b-portable-final-bundle) |
| 224-crop adaptation | Additional 224-pixel Raptor/mirror inference lines | [Varshitha Kolupuri's notebook](https://www.kaggle.com/code/varshithakolupuri/rsna-knee-one-unused-checkpoint-lifts-the-0-956) |
| OrthoFoundation fold-0 component | Auxiliary checkpoint used by some source bundles | [Kaggle model dataset](https://www.kaggle.com/datasets/heliosli/rsna-orthofoundation-fold0-epoch7-20261004) |

The Helios notebook's source contains an Apache-2.0 notice, as reviewed by the repository maintainer. That code notice is preserved with the source; it does not replace the separate terms shown on a checkpoint dataset page. This repository links to external weights but does **not** redistribute those weights or the competition data. Model and dataset licenses must be checked independently at the source and version used.

The GoodPJW reader package and ConvNeXt initialization retain their upstream notices. The R6 notebooks also carry OAI acknowledgements and citations required by their upstream methods; those acknowledgement blocks remain in the archived notebook source.

No repository-wide license is added here because this archive contains multiple upstream contributions with distinct terms. A repository-level license decision should not be interpreted as relicensing external notebooks, model weights, or competition data.
