# Brazil 440 m land-use challenge

Resolved chips: 205. Classes: {'agri_loss': 159, 'hard_nonagri_loss': 46}. Nonagricultural endpoints: {'mining': 18, 'secondary_forest': 28}. Agricultural endpoints: {'crop': 4, 'pasture': 155}.

| Region | Score | AUROC | Bootstrap 95% interval |
|---|---|---:|---:|
| overall | head_token_top9 | 0.7643 | [0.6768, 0.8474] |
| overall | head_token_max | 0.7538 | [0.6603, 0.8445] |
| overall | terrafm_forest_agri_score | 0.5634 | [0.4565, 0.6649] |
| overall | terrafm_delta_score | 0.6697 | [0.5787, 0.7498] |
| overall | optical_fraction | 0.7826 | [0.7037, 0.8564] |
| overall | optical_radar_combined | 0.7656 | [0.682, 0.8409] |
| brazil_matogrosso | head_token_top9 | 0.8949 | [0.8011, 0.9653] |
| brazil_matogrosso | head_token_max | 0.8977 | [0.7974, 0.9716] |
| brazil_matogrosso | terrafm_forest_agri_score | 0.725 | [0.6017, 0.8364] |
| brazil_matogrosso | terrafm_delta_score | 0.7318 | [0.6341, 0.8273] |
| brazil_matogrosso | optical_fraction | 0.8821 | [0.7875, 0.9523] |
| brazil_matogrosso | optical_radar_combined | 0.8804 | [0.7801, 0.9568] |
| brazil_para | head_token_top9 | 0.5912 | [0.4461, 0.7342] |
| brazil_para | head_token_max | 0.5723 | [0.423, 0.7194] |
| brazil_para | terrafm_forest_agri_score | 0.3892 | [0.2479, 0.5237] |
| brazil_para | terrafm_delta_score | 0.6028 | [0.4694, 0.7279] |
| brazil_para | optical_fraction | 0.6814 | [0.5538, 0.8014] |
| brazil_para | optical_radar_combined | 0.6543 | [0.5226, 0.7751] |

## Outcome-specific discrimination

Each comparison retains agricultural chips and one nonagricultural endpoint. These strata are exploratory; small cells have wide intervals.

| Endpoint | Region | Agri / nonagri | Head AUROC (95% interval) | Optical AUROC (95% interval) |
|---|---|---:|---:|---:|
| mining | overall | 159 / 18 | 0.5049 [0.376, 0.6276] | 0.5575 [0.4327, 0.6728] |
| mining | brazil_matogrosso | 80 / 4 | 0.6281 [0.278, 0.8688] | 0.5875 [0.3688, 0.8063] |
| mining | brazil_para | 79 / 14 | 0.3942 [0.2405, 0.5543] | 0.5375 [0.3888, 0.6808] |
| secondary_forest | overall | 159 / 28 | 0.931 [0.8715, 0.9748] | 0.9273 [0.8786, 0.9678] |
| secondary_forest | brazil_matogrosso | 80 / 18 | 0.9542 [0.909, 0.9896] | 0.9476 [0.899, 0.9847] |
| secondary_forest | brazil_para | 79 / 10 | 0.8671 [0.6885, 0.9798] | 0.8829 [0.7614, 0.9715] |

## Fixed-rule flag rates

The TerraFM chip rule is fixed. The head threshold below was selected for a nine-chip window and is shown only as an uncalibrated transfer diagnostic.

| Region | Rule | Agriculture flagged | Nonagricultural loss flagged |
|---|---|---:|---:|
| overall | terrafm_chip_rule | 31/159 | 5/46 |
| overall | head_transferred_9tile_threshold | 81/159 | 10/46 |
| brazil_matogrosso | terrafm_chip_rule | 19/80 | 1/22 |
| brazil_matogrosso | head_transferred_9tile_threshold | 49/80 | 1/22 |
| brazil_para | terrafm_chip_rule | 12/79 | 4/24 |
| brazil_para | head_transferred_9tile_threshold | 32/79 | 9/24 |

## Limits

- The primary discrimination measure is threshold-free AUROC; the 9-tile head threshold was not calibrated for one chip.
- Fixed-rule rates are conditional on purposively sampled transition classes, not population accuracy or precision.
- Labels come from TerraClass and MapBiomas EO products and do not establish field truth or EUDR compliance.
- Positive and hard-negative chips were selected from different map-class frames; regional and endpoint stratification are reported to expose this selection.
- Bootstrap intervals resample chips within each class and do not account for product-label error or residual spatial clustering.
