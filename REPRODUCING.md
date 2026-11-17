# Reproducing the results

## 1. Saved predictions (GitHub checkout)

```bash
python scripts/reproduce.py --quick
```

Recomputes the headline metrics from the saved prediction tables and frozen thresholds.

## 2. Scores, checks and diagnostics from cached features (EOTDL snapshot)

```bash
python scripts/fetch_data.py --from <EOTDL snapshot folder>   # not needed inside the snapshot itself
python scripts/reproduce.py
```

This runs each step below and compares its output with the saved results:

| Step | Command | Compared with |
|---|---|---|
| Original experiment | `python offline.py verify` | `published/dataset_c/` predictions, thresholds, report |
| Comparator models | `python verify_revision.py --bundle .` | `reviewer_revision/results/` predictions, thresholds, evaluations |
| Patch agreement, patch-count sensitivity, conversion vs existing agriculture | `python reviewer_revision/source/additional_checks.py --bundle .` | `provenance/verification/additional_checks.json` |
| 2020 confusion on controls | `python reviewer_revision/source/control_date_audit.py --bundle .` | `provenance/verification/control_date_audit.json` |
| Biome boundary | `python reviewer_revision/source/check_biome.py --bundle .` | `provenance/verification/biome_check.json` |
| Geographic and threshold-stability diagnostics | `python scripts/run_diagnostics.py` | `reviewer_revision/results/diagnostics.json` |
| Average precision, 5/10/20% FPR-cap sensitivity, paired correctness | `python additional_diagnostics.py --bundle . --output rerun/diagnostics` | `diagnostics/*.csv` |

The verifiers check every data hash in `manifest.json`. They then recompute validation, test and control scores from the cached features and saved models, and check that the validation-selected thresholds are unchanged. A score changed by more than 1e-6 (original) or 2e-6 (comparators) fails the check. Confusion counts, control flags and the paired 10-km-block bootstrap must also match. Nothing under `published/` or `reviewer_revision/results/` is modified; new output goes to `rerun/`.

On the reference machine (Python 3.11.15, CPU, packages in `requirements-lock.txt`), every compared output matched exactly, apart from floating-point residues of order 1e-16. Another platform may show small differences within the stated tolerances.

The additional analyses are post hoc. They did not affect sampling, model fitting or the original threshold selection. Thresholds for the alternative FPR caps use validation scores only.

## Repeat the two original fits

```bash
python offline.py train --output rerun/fits
```

This uses the original seed, token order, optimiser, training/validation split, class weights, loss and stopping rule. New artifacts are written separately; the published heads are never overwritten. On the reference machine the repeated fits matched both published heads tensor for tensor (`provenance/training_reproduction.json`). Exact equality across platforms is not promised. The code never fits on test labels. Do not use test outcomes for further model selection.

## Features for new imagery

The cached features were computed with the BigEarthNet-adapted TerraFM backbone, whose original checkpoint has SHA-256 `5a7c585e66752364360d54f21c6c4197b87cc85eec670c11650012d86c8d7bdc`. The backbone, the land-use classifier and an Earth Engine downloader reproducing the study's compositing are released as [forest-to-agriculture-inference](https://github.com/alalbiol/forest-to-agriculture-inference). It reproduces the benchmark scores from raw imagery for its sample chips.

The scripts used to sample candidates and build references from the raw TerraClass and MapBiomas archives are not part of this release. The released references, splits and features are the inputs to every reported result. The raw source archives are identified by URL and hash in `provenance/source_manifest.json`. The `change` channel in the features comes from an earlier change-detection head, which is not released; it is used only by the `joint_original` and `change_original` comparators.
