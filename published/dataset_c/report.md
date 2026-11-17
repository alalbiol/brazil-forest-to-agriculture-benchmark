# Final Brazil land-use head effort

New geographic train/validation/test splits; all reference labels are derived from TerraClass and MapBiomas.

Usable imagery/features: 732/732 accepted reference chips. Counts below are actually used chips.

| Split | Agriculture | Secondary forest | Mining | Stable forest |
|---|---:|---:|---:|---:|
| train | 241 | 41 | 62 | 104 |
| validation | 72 | 14 | 18 | 32 |
| test | 84 | 10 | 19 | 35 |

## loss_endpoints

113 test chips: 84 agricultural, 29 controls.

| Method | AUROC (block 95% CI) | Recall | FPR | Accuracy | TP / FN / FP / TN |
|---|---:|---:|---:|---:|---:|
| joint_top9 | 0.8994 [0.8382, 0.9531] | 85.7% | 10.3% | 86.7% | 72 / 12 / 3 / 26 |
| landuse_top9 | 0.9639 [0.9216, 0.9903] | 86.9% | 6.9% | 88.5% | 73 / 11 / 2 / 27 |
| spectral_joint_top9 | 0.8678 [0.7946, 0.9309] | 76.2% | 10.3% | 79.6% | 64 / 20 / 3 / 26 |
| change_top9 | 0.7005 [0.5804, 0.8191] | 34.5% | 6.9% | 49.6% | 29 / 55 / 2 / 27 |
| optical_fraction | 0.7584 [0.6458, 0.8582] | 54.8% | 13.8% | 62.8% | 46 / 38 / 4 / 25 |

Paired joint-minus-change AUROC: 0.1989; block-bootstrap 95% interval [0.1, 0.3157].

## all_controls

148 test chips: 84 agricultural, 64 controls.

| Method | AUROC (block 95% CI) | Recall | FPR | Accuracy | TP / FN / FP / TN |
|---|---:|---:|---:|---:|---:|
| joint_top9 | 0.9513 [0.917, 0.9792] | 85.7% | 4.7% | 89.9% | 72 / 12 / 3 / 61 |
| landuse_top9 | 0.9717 [0.9437, 0.9927] | 86.9% | 4.7% | 90.5% | 73 / 11 / 3 / 61 |
| spectral_joint_top9 | 0.9371 [0.8981, 0.9693] | 76.2% | 4.7% | 84.5% | 64 / 20 / 3 / 61 |
| change_top9 | 0.8615 [0.7938, 0.9252] | 34.5% | 3.1% | 61.5% | 29 / 55 / 2 / 62 |
| optical_fraction | 0.8850 [0.8225, 0.941] | 54.8% | 6.2% | 71.6% | 46 / 38 / 4 / 60 |

Paired joint-minus-change AUROC: 0.0898; block-bootstrap 95% interval [0.0399, 0.1472].

## stable_forest

35 test chips: 0 agricultural, 35 controls.

| Method | AUROC (block 95% CI) | Recall | FPR | Accuracy | TP / FN / FP / TN |
|---|---:|---:|---:|---:|---:|
| joint_top9 | — | — | 0.0% | 100.0% | 0 / 0 / 0 / 35 |
| landuse_top9 | — | — | 2.9% | 97.1% | 0 / 0 / 1 / 34 |
| spectral_joint_top9 | — | — | 0.0% | 100.0% | 0 / 0 / 0 / 35 |
| change_top9 | — | — | 0.0% | 100.0% | 0 / 0 / 0 / 35 |
| optical_fraction | — | — | 0.0% | 100.0% | 0 / 0 / 0 / 35 |

## agriculture_vs_mining

103 test chips: 84 agricultural, 19 controls.

| Method | AUROC (block 95% CI) | Recall | FPR | Accuracy | TP / FN / FP / TN |
|---|---:|---:|---:|---:|---:|
| joint_top9 | 0.8941 [0.8236, 0.9512] | 85.7% | 10.5% | 86.4% | 72 / 12 / 2 / 17 |
| landuse_top9 | 0.9843 [0.9574, 0.9988] | 86.9% | 0.0% | 89.3% | 73 / 11 / 0 / 19 |
| spectral_joint_top9 | 0.8722 [0.7986, 0.9375] | 76.2% | 5.3% | 79.6% | 64 / 20 / 1 / 18 |
| change_top9 | 0.6068 [0.438, 0.7579] | 34.5% | 10.5% | 44.7% | 29 / 55 / 2 / 17 |
| optical_fraction | 0.6986 [0.5384, 0.8235] | 54.8% | 15.8% | 60.2% | 46 / 38 / 3 / 16 |

Paired joint-minus-change AUROC: 0.2873; block-bootstrap 95% interval [0.1612, 0.4432].

## agriculture_vs_secondary_forest

94 test chips: 84 agricultural, 10 controls.

| Method | AUROC (block 95% CI) | Recall | FPR | Accuracy | TP / FN / FP / TN |
|---|---:|---:|---:|---:|---:|
| joint_top9 | 0.9095 [0.8329, 0.9714] | 85.7% | 10.0% | 86.2% | 72 / 12 / 1 / 9 |
| landuse_top9 | 0.9250 [0.8391, 0.9864] | 86.9% | 20.0% | 86.2% | 73 / 11 / 2 / 8 |
| spectral_joint_top9 | 0.8595 [0.7381, 0.9565] | 76.2% | 20.0% | 76.6% | 64 / 20 / 2 / 8 |
| change_top9 | 0.8786 [0.7814, 0.9591] | 34.5% | 0.0% | 41.5% | 29 / 55 / 0 / 10 |
| optical_fraction | 0.8720 [0.7313, 0.9688] | 54.8% | 10.0% | 58.5% | 46 / 38 / 1 / 9 |

Paired joint-minus-change AUROC: 0.0310; block-bootstrap 95% interval [-0.0123, 0.0953].

## brazil_matogrosso

39 test chips: 38 agricultural, 1 controls.

| Method | AUROC (block 95% CI) | Recall | FPR | Accuracy | TP / FN / FP / TN |
|---|---:|---:|---:|---:|---:|
| joint_top9 | 1.0000  | 97.4% | 0.0% | 97.4% | 37 / 1 / 0 / 1 |
| landuse_top9 | 1.0000  | 94.7% | 0.0% | 94.9% | 36 / 2 / 0 / 1 |
| spectral_joint_top9 | 1.0000  | 89.5% | 0.0% | 89.7% | 34 / 4 / 0 / 1 |
| change_top9 | 1.0000  | 55.3% | 0.0% | 56.4% | 21 / 17 / 0 / 1 |
| optical_fraction | 0.9474  | 73.7% | 0.0% | 74.4% | 28 / 10 / 0 / 1 |

Fewer than five examples in one class: descriptive metrics only; bootstrap intervals suppressed.

## brazil_para

74 test chips: 46 agricultural, 28 controls.

| Method | AUROC (block 95% CI) | Recall | FPR | Accuracy | TP / FN / FP / TN |
|---|---:|---:|---:|---:|---:|
| joint_top9 | 0.8346 [0.7324, 0.9216] | 76.1% | 10.7% | 81.1% | 35 / 11 / 3 / 25 |
| landuse_top9 | 0.9433 [0.8779, 0.9856] | 80.4% | 7.1% | 85.1% | 37 / 9 / 2 / 26 |
| spectral_joint_top9 | 0.7911 [0.6793, 0.8897] | 65.2% | 10.7% | 74.3% | 30 / 16 / 3 / 25 |
| change_top9 | 0.5901 [0.4451, 0.7289] | 17.4% | 7.1% | 45.9% | 8 / 38 / 2 / 26 |
| optical_fraction | 0.6704 [0.524, 0.7955] | 39.1% | 14.3% | 56.8% | 18 / 28 / 4 / 24 |

Paired joint-minus-change AUROC: 0.2446; block-bootstrap 95% interval [0.129, 0.3645].

## Limits

- EO-product reference, not human or field truth and not legal EUDR compliance.
- Thresholds were selected on validation before test predictions; current 205-chip benchmark was development context.
- Purposive class-balanced geographic sampling; accuracy/precision are not population estimates.
- Blocks and 2km separation mitigate spatial dependence; map errors and pretraining exposure remain.
- Only 29 nonagricultural-loss test controls (19 mining, 10 secondary forest); regional transfer remains uncertain.
- Mato Grosso has only one nonagricultural-loss test control, so its ranking/FPR are descriptive and cannot establish regional robustness.
- Wilson recall/FPR intervals assume independent examples; AUROC intervals instead resample geographic blocks.
- Linear token classifier on frozen TerraFM; no backbone fine-tuning or architecture search.
