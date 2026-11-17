# Brazil external-reference benchmark

Reference: TerraClass 2020 primary forest to 2024 mapped crop/pasture (≥0.5 ha), spatially supported by a MapBiomas 2022–2023 alert (≥0.5 ha). This tests agreement with public EO products, not field-truth accuracy or EUDR compliance.

Sampled: 300; resolved: 148; paired imagery and predictions: 148.
Reference counts: {'conversion': 75, 'out_of_scope': 69, 'uncertain': 83, 'no_conversion': 73}.

Conversion subclasses: {'primary_to_pasture': 71, 'primary_to_crop': 4}. Uncertain reasons: {'unobserved_or_recent_2024_clearing': 39, 'map_transition_without_sufficient_dated_alert_or_near_boundary': 17, 'geodata_query_did_not_complete': 1, 'geodata_query_timeout': 26}.

Negative composition: {'alert_2022_2023': 22, 'background': 51}; only 1 negative has ≥0.5 ha mapped primary forest to secondary forest or silviculture. Head false positives by stratum: {'alert_2022_2023': 3}.

| Region | Method | N covered | Abstain | TP | FN | FP | TN | Recall | FPR | Precision | Accuracy |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| overall | whole_rule | 148 | 0 | 7 | 68 | 0 | 73 | 0.0933 | 0.0 | 1.0 | 0.5405 |
| overall | tiles3_rule | 148 | 0 | 25 | 50 | 14 | 59 | 0.3333 | 0.1918 | 0.641 | 0.5676 |
| overall | head_rule_10fpr | 148 | 0 | 46 | 29 | 3 | 70 | 0.6133 | 0.0411 | 0.9388 | 0.7838 |
| brazil_matogrosso | whole_rule | 71 | 0 | 6 | 34 | 0 | 31 | 0.15 | 0.0 | 1.0 | 0.5211 |
| brazil_matogrosso | tiles3_rule | 71 | 0 | 15 | 25 | 8 | 23 | 0.375 | 0.2581 | 0.6522 | 0.5352 |
| brazil_matogrosso | head_rule_10fpr | 71 | 0 | 29 | 11 | 0 | 31 | 0.725 | 0.0 | 1.0 | 0.8451 |
| brazil_para | whole_rule | 77 | 0 | 1 | 34 | 0 | 42 | 0.0286 | 0.0 | 1.0 | 0.5584 |
| brazil_para | tiles3_rule | 77 | 0 | 10 | 25 | 6 | 36 | 0.2857 | 0.1429 | 0.625 | 0.5974 |
| brazil_para | head_rule_10fpr | 77 | 0 | 17 | 18 | 3 | 39 | 0.4857 | 0.0714 | 0.85 | 0.7273 |

## Ranking diagnostics

Scores are compared without selecting a threshold on this benchmark. Average precision reflects the sample prevalence.

| Region | Score | AUROC | Average precision |
|---|---|---:|---:|
| overall | whole_forest_agri_score | 0.7562 | 0.7315 |
| overall | whole_delta_score | 0.643 | 0.6732 |
| overall | tiles3_forest_agri_max | 0.7478 | 0.7393 |
| overall | tiles3_delta_max | 0.7313 | 0.7499 |
| overall | head_token_max | 0.9224 | 0.9231 |
| overall | head_token_top9 | 0.9198 | 0.9184 |
| overall | optical_fraction | 0.8629 | 0.8664 |
| overall | vh_drop_q90 | 0.7856 | 0.7443 |
| overall | optical_radar_combined | 0.8563 | 0.8545 |
| brazil_matogrosso | whole_forest_agri_score | 0.7363 | 0.7715 |
| brazil_matogrosso | whole_delta_score | 0.7327 | 0.8099 |
| brazil_matogrosso | tiles3_forest_agri_max | 0.7702 | 0.8122 |
| brazil_matogrosso | tiles3_delta_max | 0.8 | 0.8553 |
| brazil_matogrosso | head_token_max | 0.9734 | 0.981 |
| brazil_matogrosso | head_token_top9 | 0.9774 | 0.984 |
| brazil_matogrosso | optical_fraction | 0.8685 | 0.9002 |
| brazil_matogrosso | vh_drop_q90 | 0.8153 | 0.8272 |
| brazil_matogrosso | optical_radar_combined | 0.8694 | 0.8984 |
| brazil_para | whole_forest_agri_score | 0.7619 | 0.692 |
| brazil_para | whole_delta_score | 0.5503 | 0.4995 |
| brazil_para | tiles3_forest_agri_max | 0.7156 | 0.6117 |
| brazil_para | tiles3_delta_max | 0.6517 | 0.5638 |
| brazil_para | head_token_max | 0.8639 | 0.8114 |
| brazil_para | head_token_top9 | 0.8605 | 0.7902 |
| brazil_para | optical_fraction | 0.8622 | 0.8301 |
| brazil_para | vh_drop_q90 | 0.7612 | 0.6676 |
| brazil_para | optical_radar_combined | 0.8531 | 0.8056 |

## Interpretation limits

- Rules and head threshold were fixed before Brazil reference evaluation.
- Abstentions are excluded from conditional rule metrics and counted as reduced coverage.
- Average precision depends on the positive prevalence in this stratified sample.
- Raw accuracy and precision are also sample-dependent because alert windows were deliberately oversampled.
- Wilson intervals treat sampled windows as independent and do not adjust for spatial clustering or map error.
- Reference positives require a dated alert and a later mapped crop or pasture; map/alert omissions may cause apparent false positives.
- Timed-out TerraClass spatial queries are reference nonresponse and may reduce geographic representativeness, especially in Pará.
- Only one resolved negative has >=0.5ha mapped primary forest to secondary forest or silviculture. Other nonagricultural transitions were not separately enumerated, so these results do not establish discrimination among post-loss land uses.
