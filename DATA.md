# Data layout and field definitions

All measurements are agreement with TerraClass/MapBiomas EO references. They are
not independently observed land-use truth, human-performance measurements or legal
compliance labels. Dataset C (the main dataset) was held out for the original
experiment; the later revision comparators reuse its inspected test outcomes.
Datasets A/B informed the design and remain development evidence.

## Files

| Path | Contents |
|---|---|
| `published/development/` | Dataset A screening and Dataset B endpoint inventories, references, saved predictions, designs and reports, and a separate 18-window stress test. Do not pool these denominators. |
| `published/dataset_c/` | All 1,077 selected candidates, 732 accepted references, splits, predictions, thresholds, audits, training histories, and the two fitted linear heads (`tokens_landuse_head.pt`, `spectral_landuse_head.pt`). `chips.geojson` gives nominal polygons for accepted and uncertain candidates (longitude/latitude). |
| `reviewer_revision/source/` | Comparator protocol (`PROTOCOL.md`), comparator models, evaluation, diagnostics and post hoc check code. |
| `reviewer_revision/results/` | Validation/test/control predictions, operating points, evaluation summaries, the 100 established-agriculture controls (sampling, reference, dispositions), fitted comparators (`bitemporal_terrafm.pkl`, `spectral_landuse_converged.pt`; `bitemporal_spectral_forest.pkl` on EOTDL only). |
| `cache/features/` | (EOTDL) 732 main-dataset 2024 feature files. |
| `cache/features_2020/` | (EOTDL) Paired 2020 features for the 732 chips. |
| `cache/labels/` | (EOTDL) Paired 44 × 44 TerraClass 2020/2024 class-code grids. |
| `cache/stable_agriculture/` | (EOTDL) Features, paired composites and 2020 TerraClass label grids for the 100 established-agriculture controls. |
| `cache/imagery/`, `cache/example_imagery/` | (EOTDL) Sentinel composites (2020 and 2024) for seven example chips. |
| `diagnostics/` | Average precision, 5/10/20% validation-FPR cap sensitivity and paired correctness. |
| `provenance/` | Source URLs/hashes (`source_manifest.json`), backbone checkpoint metadata, earlier change-head training distance audit, repeated-fit check, environment, verification records. |
| `sources/` | (EOTDL) `Biomas_250mil.zip`: IBGE 2019 biome map for the biome-boundary check. |

## Field dictionary

`reference.csv`: `window_id` uniquely identifies a chip; `region` is the state
frame; `lon/lat` give its centre; `alert_id/alert_year` identify the dated event
(empty for stable controls); `endpoint` is the candidate outcome
(`agriculture`, `mining`, `secondary_forest`, `stable_forest`);
`point_class_2024` preserves the source-map subclass at its centre; `block_id`
is the 10-km geographic split unit; `split` is train/validation/test;
`reference_label` is accepted/uncertain; `target_event_ha` is the primary-to-target
transition intersecting the specified alert; `opposite_transition_ha` is the
opposite-use area anywhere in the chip; primary fractions describe the two maps.
An uncertain candidate is never silently converted to a negative.

Feature NPZ keys (`cache/features/`): `tokens` [768, 14, 14] TerraFM tokens,
`spectral` [14, 14, 14] patch-mean bands, `token_labels` [14, 14]
(0 primary forest, 1 secondary forest, 2 agriculture, 3 mining, −1 ignored),
`change` [14, 14] earlier change-head output, and scalar `optical_fraction`.
`cache/features_2020/` holds the same keys for the 2020 composite.
Control features (`cache/stable_agriculture/features/`) hold both dates as
`tokens_2020` and `tokens_2024`.

Label NPZ keys `before`/`after` preserve the TerraClass 0–13 class codes listed in
the supplementary methods; these differ from the four classifier classes. Control
label files contain `before` (2020) only. Imagery NPZ key `imagery` has shape
[14, 44, 44]: Sentinel-1 VV, VH followed by twelve Sentinel-2 bands.

Prediction columns in `reviewer_revision/results/*_predictions.csv`:
`postclassification` (temporal TerraFM), `endpoint_terrafm` (2024 only),
`joint_original` (2024 × change), `bitemporal_terrafm`, `spectral_linear_endpoint`,
`spectral_linear_joint`, `spectral_forest_bitemporal`, `change_original`,
`optical_index`; `critical_audit_*_predictions.csv` add `spectral_postclassification`.
In `published/dataset_c/` the original experiment's columns are `joint_top9`,
`landuse_top9`, `spectral_joint_top9`, `change_top9` and `optical_fraction`.
All scores are uncalibrated rankings. Geographic-block bootstrap intervals condition
on these source labels and fixed models.
