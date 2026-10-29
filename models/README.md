# Final submission notebook snapshots

The four notebooks below correspond to the entries shown in the final Kaggle submission screenshot and confirmed by the authenticated Kaggle submissions endpoint on 27 October 2026. Each downloaded notebook is a source snapshot with **no execution outputs**; model weights and competition data are not embedded.

| Entry | Kaggle notebook/source | Submission ref | Captured source file | SHA-256 | Selected |
| --- | --- | ---: | --- | --- | :---: |
| R6 Mirror R40 | [Arv's notebook](https://www.kaggle.com/code/arvbali/rsna-knee-r6-mirror-r40-20261017) | `57292701` | [`r6_mirror_r40/rsna-knee-r6-mirror-r40-20261017.ipynb`](submission_notebooks/r6_mirror_r40/rsna-knee-r6-mirror-r40-20261017.ipynb) | `6ca9246a17054bd5a01658d00a93797a58a7259b7afb34cd8e7a59904c4e4c10` | No |
| R6 A (portable final bundle) | [Arv's notebook](https://www.kaggle.com/code/arvbali/rsna-r6-a-portable-final-bundle) | `57433412` | [`r6_a/rsna-r6-a-portable-final-bundle.ipynb`](submission_notebooks/r6_a/rsna-r6-a-portable-final-bundle.ipynb) | `80cbd68fb389c3250334e67c639e7a81e39c52e2976d9bb93279bf8e78511b40` | No |
| Final R6 B | [Our Kaggle notebook](https://www.kaggle.com/code/mrinanksivakumar/rsna-final-r6-b-20261021) · based on [Arv's source](https://www.kaggle.com/code/arvbali/rsna-r6-b-portable-final-bundle) | `57485947` | [`r6_b/rsna-r6-b-portable-final-bundle.ipynb`](submission_notebooks/r6_b/rsna-r6-b-portable-final-bundle.ipynb) | `74a24749eb18b7e35ad7d56d0d3c86e7f1fc3d6c685ffd21cfc31788450d797d` | **Yes** |
| R6 Raptor — no anatomy correction | [Arv's source notebook](https://www.kaggle.com/code/arvbali/rsna-r6-raptor-no-anatomy-correction) · [our submission notebook](https://www.kaggle.com/code/mrinanksivakumar/rsna-r6-raptor-no-anatomy-correction-20261022) | `57531062` | [`raptor_no_anatomy/rsna-r6-raptor-no-anatomy-correction.ipynb`](submission_notebooks/raptor_no_anatomy/rsna-r6-raptor-no-anatomy-correction.ipynb) | `0a1428f101c7902807f28ab759d1d41d962b5e420084831eba1fc998ca425f80` | **Yes** |

All four entries scored **0.958 public / 0.942 private**. The two selected entries are R6 B and Raptor without the medical/anatomy correction. See [`results/FINAL_RESULTS.md`](../results/FINAL_RESULTS.md) for submission dates, descriptions, and the additional tied anatomy-corrected entry.

The R6 B row includes both Arv's upstream source snapshot and our reproducible team-side scoring build at [`r6_b_scored_version/rsna-final-r6-b-20261021.ipynb`](submission_notebooks/r6_b_scored_version/rsna-final-r6-b-20261021.ipynb). The generated file adds input integrity receipts and mount discovery while keeping the fixed R6 B inference/blend/correction recipe. Rebuild it from the hash-matched upstream snapshot using the command in [`src/mri/final_r6_b/README.md`](../src/mri/final_r6_b/README.md).

## Source snapshot caveat

The snapshots were pulled from the shared Kaggle notebook slugs on 27 October 2026. Some upstream notebooks continued to receive later documentation or code revisions after the competition submission. The score belongs to the exact submission reference above; a current source snapshot is not automatically proof of byte-for-byte identity with a historical run. R6 B's source hash matches the audited upstream source hash used by the local packaging builder. For strict historical reproduction, use the submission-version page and the pinned dataset versions recorded in that Kaggle run's Data panel.

The adjacent `kernel-metadata.json` files are preserved as retrieved provenance. Their `id` and privacy fields are Arv's original Kaggle metadata; edit a copy before pushing under another account. Input slugs are not always version-pinned in Kaggle's metadata export, so verify the original version pins rather than choosing `latest`.

## Main input sources

The source metadata and notebook text refer to these Kaggle sources, depending on the entry:

- [RSNA Knee Abnormality Detection competition data](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/data)
- [GoodPJW three-fold ConvNeXt reader](https://www.kaggle.com/datasets/goodpjw2008/rsna-knee-2-5d-convnext-reader)
- [Heliosli compact R7 bundle](https://www.kaggle.com/datasets/heliosli/rsna-knee-compact-r7cnxfsss) — attached in some source snapshots; not used by the cleaned R6 B packaging
- [R6 compact bundle, versioned copy](https://www.kaggle.com/datasets/rostigor/rsna-knee-r6-fe055406)
- [Nartaa SWA/Raptor weights](https://www.kaggle.com/datasets/nartaa/rsna-knee-publication-swa-weights-20261007)
- [Heliosli OrthoFoundation fold-0 model](https://www.kaggle.com/datasets/heliosli/rsna-orthofoundation-fold0-epoch7-20261004)
- [Arv's final R6 assets](https://www.kaggle.com/datasets/arvbali/rsna-final-r6-assets-20261020) — Version 2 for the R6 B bundle

Do not redistribute checkpoint files from this repository. Each linked dataset retains its own provenance and license; see [`ASSET_PROVENANCE.md`](../ASSET_PROVENANCE.md).
