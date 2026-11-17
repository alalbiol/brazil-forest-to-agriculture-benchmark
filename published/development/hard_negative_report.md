# Brazil nonagricultural-loss stress test (1.32 km windows)

The same frozen models and thresholds were used on 18 newly selected, dated primary-forest-loss windows whose 2024 TerraClass endpoint is nonagricultural. These are deliberately difficult negatives; the groups below have different sampling frames.

| Group | N | Whole rule flagged | 3×3 rule flagged | Spatial head flagged |
|---|---:|---:|---:|---:|
| agricultural_conversion | 75 | 7/75 (9.3%) | 25/75 (33.3%) | 46/75 (61.3%) |
| original_no_conversion | 73 | 0/73 (0.0%) | 14/73 (19.2%) | 3/73 (4.1%) |
| nonagricultural_loss | 18 | 0/18 (0.0%) | 3/18 (16.7%) | 8/18 (44.4%) |

For the spatial head, the hard-negative false-alarm rate is 44.4% (Wilson 95% interval [0.2456, 0.6628]). This is a conditional stress test, not an estimate for all forest-loss events.

Hard-negative endpoints: {'secondary_forest': 11, 'mining': 7}. By region: {'brazil_matogrosso': 9, 'brazil_para': 9}.

## Limits

- Hard negatives are selected from public TerraClass and MapBiomas products, not field truth.
- The three groups have different sampling frames; do not pool them into a population accuracy or precision estimate.
- The head threshold was fixed on non-Brazil validation data for a 9-tile window.
- Hard-negative confidence intervals are wide because only 18 full windows qualified.
