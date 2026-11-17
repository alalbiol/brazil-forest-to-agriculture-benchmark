# Reviewer-driven revision protocol — 15 October 2026

This protocol was written and committed before fitting new models, acquiring new
controls, or examining any new test predictions. The existing Dataset C results
remain fixed and are recoverable from tag
`ejrs-pre-reviewer-revision-2026-10-15`. Analyses of those saved predictions are
diagnostic; any new test set and model results will be reported regardless of
outcome.

## Questions

1. Does the change component suppress false positives on agriculture that was
   already present at the 31 December 2020 cutoff?
2. Does a frozen TerraFM representation add useful discrimination beyond strong
   conventional spectral/radar models?
3. How much of the reported discrimination is explained by state, endpoint
   composition, broad landscape context, or weak-reference construction?
4. Can a directly trained bitemporal score improve on the inherited change head
   without using the test set for model selection?

## Frozen existing analyses

The original Dataset C split, heads, thresholds, and predictions will not be
altered. We will add:

- class-by-state counts for every split and a state-only geographic baseline;
- Pará-only results, with Mato Grosso results described only when both classes
  have an adequate denominator;
- spectral endpoint-only performance, paired correctness comparisons, and exact
  McNemar tests at the frozen operating thresholds;
- operating rates with Wilson intervals in the main paper and explicit warnings
  where denominators are too small for inference;
- event-area, prior-agriculture, supervised-token, and within/outside-event
  diagnostics where the saved reference grids permit them;
- nearest-training-distance and coarser spatial grouping sensitivity analyses;
- an accounting of all unresolved candidates and technical nonresponses.

The global enriched-sample AUROC and average precision will be described as
conditional benchmark metrics, not regional prevalence or compliance estimates.

## Stable-agriculture control extension

Construct a new held-out control set from locations mapped as agriculture in both
TerraClass 2020 and 2024, with no MapBiomas alert intersecting the chip during
2021–2024. Use the same 440 m chip geometry, source products, imagery pipeline,
state domain, and data-quality checks as Dataset C. Exclude all prior benchmark,
training, validation, and test locations by at least 12 km and keep 10 km blocks
together. Target at least 100 accepted controls, balanced between Pará and Mato
Grosso when the mapped frame permits; retain the full candidate disposition and
report any shortfall. Selection must not use a model score.

The primary extension outcome is the false-positive count and rate on these
controls for the already frozen Dataset C thresholds. Compare endpoint-only,
the original joint score, post-classification, the direct bitemporal model, and
conventional baselines. Exact binomial intervals accompany every rate. These
controls are an EO-product-agreement test of the cutoff premise, not independent
field truth.

## Added models and fixed selection rules

All added models use Dataset C training labels, choose hyperparameters or stopping
epochs with Dataset C validation only, and are evaluated once on the existing test
and stable-agriculture extension. Random seeds and complete validation candidates
will be saved.

1. **Spectral endpoint-only:** refit the same linear spectral land-use head with
   the original optimizer and training data, extending the maximum from 100 to
   500 epochs and stopping after 30 epochs without validation-loss improvement;
   report it both alone and multiplied by the inherited change score. This fixes
   the original run's truncation at its best (final) epoch without changing its
   architecture or choosing an epoch from test performance.
2. **Post-classification:** apply the frozen four-class TerraFM land-use head to
   both dates and compute the top-nine mean of
   `P2020(primary forest) * P2024(agriculture)`.
3. **Direct bitemporal TerraFM head:** a regularized logistic classifier over
   concatenated 2020 tokens, 2024 tokens, their signed difference, and absolute
   difference. Regularization is selected from `C = {0.01, 0.1, 1, 10}` by
   validation log loss; ties favor stronger regularization. Token targets are the
   mapped 2020-to-2024 transition, and mixed/unknown tokens remain ignored.
4. **Conventional nonlinear baseline:** a class-weighted random forest over
   per-token 2020 and 2024 spectral/radar values, signed differences, absolute
   differences, NDVI, NDMI, NBR, and their temporal differences. Select
   `max_depth` from `{6, 12, None}` and `min_samples_leaf` from `{2, 10}` by
   validation log loss, with 500 trees and a fixed seed. Ties favor the shallower
   tree and larger leaf.

For bitemporal token scores, the chip score is the mean of the nine largest valid
token probabilities. Each new operating threshold is chosen on validation by
maximum agricultural-conversion recall under an empirical 10% nonagricultural
loss false-positive cap. No threshold is tuned on either test set. AUROC is the
threshold-free primary comparison; operating rates are secondary.

## Preprocessing and reference audits

- Quantify valid Sentinel-2 observations by year/month and audit the documented
  QA60 availability gap. Compare a fixed sample against a stronger cloud-quality
  mask if the source collections are accessible; otherwise state the limitation
  and remove any unsupported claim of cloud robustness.
- Quantify Sentinel-1 ascending/descending availability and score sensitivity on
  chips for which orbit-separated composites can be generated.
- State the native and resized spatial scales and test whether broader input
  context is feasible with the frozen backbone. Do not claim that interpolation
  adds spatial information.
- Review every Dataset C error plus a fixed random sample of 20 correct cases
  against the available temporal imagery. Because no independent interpreter is
  available, call this an author audit, record a reason code and confidence, and
  do not relabel the primary reference or claim independent validation.
- Verify bibliographic metadata against DOI/official records and replace vague or
  unversioned source citations.

## Interpretation rules

- If endpoint-only performs better on confirmed-loss endpoints but produces more
  stable-agriculture false positives, present this as the expected discrimination
  versus cutoff-verification trade-off and retain a temporal primary score.
- If the joint score does not materially reduce stable-agriculture false positives,
  it will no longer be advocated as the preferred method; the paper will be
  reframed around the benchmark and failure analysis.
- If conventional models match TerraFM within the paired uncertainty, state that
  the experiment does not establish a foundation-model advantage.
- Claims will be limited to mapped, predominantly pasture endpoints in this
  enriched proof-of-concept sample. No regional calibration, legal compliance,
  crop-generalization, independent-reference, or geographic-transfer claim will
  be made without direct evidence.

## Manuscript and artifact work

Revise the title, abstract, methods, tables, figures, discussion, conclusion,
supplement, cover letter, highlights, and reproducibility documentation to match
the final evidence. Use “spatially blocked” for the original split, distinguish
the project-specific BigEarthNet-adapted TerraFM checkpoint from public TerraFM,
describe the inherited change pipeline as prior project work, streamline repeated
limitations, and make the reference rules and class/state composition visible.
Regenerate all numerical tables and figures from saved artifacts, build both PDFs,
run reference and package checks, and retain the original tagged version.
