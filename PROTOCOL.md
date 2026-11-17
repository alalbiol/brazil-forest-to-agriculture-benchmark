# Prespecified final model effort — 1 October 2026

User authorized one focused attempt to improve agricultural-use discrimination.
This protocol is written before new training or new test model scores.

## Model and hypothesis

Train a small land-use head on **frozen TerraFM post-event spatial tokens**, using
TerraClass 2024 weak labels. Classes are agriculture (crop/pasture), secondary
forest, mining, and primary forest. This directly supervises the missing endpoint
decision. Use a single 1x1 linear token classifier with weighted cross entropy;
select its epoch by validation loss only. No architecture or hyperparameter sweep.
The backbone and existing spatial change head remain fixed.

At inference combine agriculture token probabilities with the frozen change
probabilities for paired 2020/2024 inputs. The primary score is the mean of the nine
largest token-wise products. Also report endpoint-only top-nine agriculture score
and the existing change score, and a simple supervised spectral land-use baseline.
Masks from TerraClass or alert polygons are **not** supplied at inference.

## Data and separation

Discover new 2022/2023 MapBiomas alert interiors over the full Amazon portions of
Pará and Mato Grosso, using baseline primary forest and 2024 agriculture,
secondary forest or mining at the point. Add mapped stable-primary-forest controls.
Use 440 m chips with paired 2020/2024 imagery and TerraClass dense labels.
Exclude earlier Brazil benchmark/challenge locations by at least 2 km and earlier
model development locations by 12 km. Exclude previously evaluated alert IDs.
Assign 10 km geographic blocks to train/validation/test before acquiring imagery
or scoring; enforce a 2 km centre separation between different splits. Keep alert
groups together. Record the seed and every exclusion.

Labels must pass area checks: >=0.5 ha of baseline-primary transition to the target
2024 class overlapping the same dated alert, <0.2 ha of opposite agriculture/non-
agriculture transition, and <0.5 ha missing/recent-clearing follow-up. Stable
controls require >=90% primary forest at both dates and no alert in the chip.
Dense token labels use >=80% dominance in the token footprint; mixed and missing
tokens are ignored in loss. Training can use any mapped four-class token in an
accepted chip; endpoint decisions remain window references from the full check.

Existing 205 challenge chips have already informed model design. Report them as
development diagnostics only, never as the fresh confirmatory test.
New labels remain EO-product agreement, not field truth or legal EUDR outcomes.

## Evaluation and decision

Primary: new held-out agricultural-loss vs secondary/mining-loss AUROC, including
each endpoint and each state separately. Paired bootstrap difference against the
existing change score is reported. Secondary: fixed validation-selected threshold
targeting <=10% validation nonagri-loss FPR; test recall, FPR, confusion counts,
accuracy and balanced accuracy, plus stable-forest flag rate. Do not change
threshold after test scores. Bootstrap geographic blocks, not individual tokens.

A useful result requires improvement on agriculture/mining discrimination without
losing agriculture/secondary-forest discrimination, with uncertainty visible. No
human-level claim is possible without a measured expert comparator. An inconclusive
or negative outcome completes the effort and must be retained in the report.
