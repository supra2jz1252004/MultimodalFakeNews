# Conditional CLIP Reliance: measured study outcome

**Publication decision: NO-GO for a claimed superior new detection method.** VERITE's full official images are inaccessible without approved institutional access; the user confirmed no approved copy. The preregistered independent endpoint is **not evaluated** (0 external samples). Existing Fakeddit results below are exploratory and cannot replace it. A research/robustness technical report is justified; a journal claim requires independent evidence and a substantially clearer contribution beyond existing utility gating.

**Measured outcome: the proposed selector failed to improve on B.** All three seeds selected the all-B ridge endpoint on validation. CCR, ridge, intercept, tree and cosine gates therefore reproduce B on both exploratory test sets; the matched random gate is also degenerate. See [decision and ablations](decision_and_ablations.md) for this limitation, calibration tradeoffs and the exact publication decision.

## What was run

30 matched OOF A/B head fits, five folds, three classifier seeds 42/123/2026, 50,000 training examples per seed. Six historical final A/B heads reused. Gate families fixed before training: intercept, ridge alpha100, depth-3 tree/minimum 500 per leaf. Normalizers use only inner-fit rows; stopping uses only internal validation. The optional historical unimodal scores were excluded because their training-set scores would be in-sample. No encoder fine-tuning or gate fitting on either test set.

The OOF stage recorded 349.5 seconds wall time and process-lifetime peak working set 1.90 GiB. Times exclude literature work and prior encoder extraction. Historical inventory covers 58,759 files (4,552,909,479 bytes), including BRAS. See `preservation_check.json` for final rehash.

Protocol SHA-256: `c9468f8abbb9ffb22c69a70fdd66f2242f1f907b609b265ca1af9794c70f3084` (canonical JSON digest; raw-file SHA is recorded in preregistered_protocol.md). Local pre-training timestamp only, not a publicly registered trial. All policies/checkpoints were frozen before any CCR test inference.

## Validation guard and calibration

| classifier_seed | guard_passed | validation_ridge_B_fraction | validation_logloss_delta | CI_low | CI_high | A_temperature | B_temperature |
|---|---|---|---|---|---|---|---|
| 42 | False | 1.000000 | 0.000000 | 0.000000 | 0.000000 | 0.874055 | 1.262971 |
| 123 | False | 1.000000 | 0.000000 | 0.000000 | 0.000000 | 1.540914 | 1.431130 |
| 2026 | False | 1.000000 | 0.000000 | 0.000000 | 0.000000 | 1.347559 | 1.310925 |

Negative loss delta favors the candidate. CCR uses the candidate only when its validation paired CI upper bound is negative; otherwise it is exactly B. The unguarded ridge candidate remains visible as a secondary ablation. These validation intervals follow threshold selection on the same data and are optimistic diagnostics, not valid independent superiority tests. A/B checkpoint and architecture selection also used this validation set. Distribution summaries are in `validation/utility_distribution_shift.csv`; no test-dependent adjustment was made.

## Historical Fakeddit (n=2,500)

Exploratory diagnostics: these outcomes cannot confirm a method designed after earlier test inspection.

| variant | accuracy_mean | macro_f1_mean | log_loss_mean | brier_mean | ece_10_mean | b_fraction_mean |
|---|---|---|---|---|---|---|
| A | 0.855467 | 0.847358 | 0.333608 | 0.103676 | 0.033006 | 0.000000 |
| B | 0.878667 | 0.872115 | 0.297812 | 0.089077 | 0.020162 | 1.000000 |
| A_temperature | 0.855467 | 0.847358 | 0.325065 | 0.102513 | 0.014415 | 0.000000 |
| B_temperature | 0.878667 | 0.872115 | 0.292807 | 0.089311 | 0.018490 | 1.000000 |
| intercept | 0.878667 | 0.872115 | 0.297812 | 0.089077 | 0.020162 | 1.000000 |
| ridge_candidate | 0.878667 | 0.872115 | 0.297812 | 0.089077 | 0.020162 | 1.000000 |
| tree | 0.878667 | 0.872115 | 0.297812 | 0.089077 | 0.020162 | 1.000000 |
| cosine_gate | 0.878667 | 0.872115 | 0.297812 | 0.089077 | 0.020162 | 1.000000 |
| confidence_gate | 0.876933 | 0.870112 | 0.300207 | 0.089896 | 0.020696 | 0.798933 |
| entropy_gate | 0.876000 | 0.869118 | 0.300052 | 0.090095 | 0.022903 | 0.850267 |
| random_gate | 0.878667 | 0.872115 | 0.297812 | 0.089077 | 0.020162 | 1.000000 |
| CCR | 0.878667 | 0.872115 | 0.297812 | 0.089077 | 0.020162 | 1.000000 |

## Previously viewed Fakeddit (n=1,000)

Exploratory diagnostics: these outcomes cannot confirm a method designed after earlier test inspection.

| variant | accuracy_mean | macro_f1_mean | log_loss_mean | brier_mean | ece_10_mean | b_fraction_mean |
|---|---|---|---|---|---|---|
| A | 0.833000 | 0.832537 | 0.391883 | 0.120598 | 0.034252 | 0.000000 |
| B | 0.859667 | 0.859407 | 0.336571 | 0.102363 | 0.034833 | 1.000000 |
| A_temperature | 0.833000 | 0.832537 | 0.379662 | 0.119936 | 0.027625 | 0.000000 |
| B_temperature | 0.859667 | 0.859407 | 0.327896 | 0.101856 | 0.024084 | 1.000000 |
| intercept | 0.859667 | 0.859407 | 0.336571 | 0.102363 | 0.034833 | 1.000000 |
| ridge_candidate | 0.859667 | 0.859407 | 0.336571 | 0.102363 | 0.034833 | 1.000000 |
| tree | 0.859667 | 0.859407 | 0.336571 | 0.102363 | 0.034833 | 1.000000 |
| cosine_gate | 0.859667 | 0.859407 | 0.336571 | 0.102363 | 0.034833 | 1.000000 |
| confidence_gate | 0.853000 | 0.852672 | 0.347343 | 0.105680 | 0.035746 | 0.807000 |
| entropy_gate | 0.851667 | 0.851308 | 0.346531 | 0.105846 | 0.038741 | 0.854000 |
| random_gate | 0.859667 | 0.859407 | 0.336571 | 0.102363 | 0.034833 | 1.000000 |
| CCR | 0.859667 | 0.859407 | 0.336571 | 0.102363 | 0.034833 | 1.000000 |

## Seed-42 paired results against B

| dataset | candidate | logloss_delta | loss_CI_low | loss_CI_high | macro_f1_delta | corrected | new_errors | McNemar_p |
|---|---|---|---|---|---|---|---|---|
| original_test | ridge_candidate | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0 | 0 | 1.000000 |
| original_test | CCR | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0 | 0 | 1.000000 |
| previous_fresh_test | ridge_candidate | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0 | 0 | 1.000000 |
| previous_fresh_test | CCR | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0 | 0 | 1.000000 |

Every estimate is recomputed from saved prediction CSVs. Loss/Brier deltas are candidate minus B; negative favors the candidate. Accuracy/Macro-F1 deltas use the same direction, so positive favors it. 5,000 common-row bootstrap replicates; bootstrap_seed 20261106 is separate from classifier_seed. Known Fakeddit families are singletons, so family bootstrap equals row bootstrap; unknown event dependence remains. Exact McNemar uses discordant correctness counts. Per-seed full tables include Brier, accuracy and Macro-F1 intervals. Secondary intervals and slices are descriptive; no test winner selection or pooled-seed sample inflation.

## Calibration, slices and pair sensitivity

Figures include loss comparison, reliability diagrams, fixed-threshold risk/coverage, six-way category accuracy with counts and Wilson intervals, conditional-utility bins and cosine-label prevalence. ECE uses confidence calibration with 10 equal-width bins; 5/15/20-bin results are saved. Brier is the single real-class squared error, not twice that value. Coverage is achieved test coverage at validation-fixed confidence cutoffs; discarded examples are explicitly counted.

Every subreddit slice includes counts and row-level uncertainty; these are distant-supervision categories, not evidence of source-generalization. Source families are linked-submission components and all retained families have one row. Broader news-event independence is unverified. Train-quartile cosine prevalence is an observational shortcut audit. Differences in class balance affect proper losses/calibration and cannot be causally attributed to cosine.

Synthetic random/hard swaps reuse the exact 500-query training-donor selections from BRAS. `stress/summary.csv` and saved pair predictions report cosine/utility changes, routing changes and label-free prediction flips. No truth labels, factual accuracy, or log-loss are assigned to synthetic swaps.

## Previous BRAS result and prior-art correction

BRAS D mean Macro-F1 was 0.872889 on historical 2,500 versus B 0.872115, and 0.855797 on the previously viewed 1,000 versus B 0.859407. Seed 42 D−B Macro-F1 CIs included zero on both. This negative result is preserved, with the corrected classifier-seed/bootstrap-seed tables as authoritative. There was no reliable BRAS advantage.

For identical neighborhoods, `(s-r_image)+(s-r_text)=2s-r_image-r_text`: summed BRAS margins have the CSLS algebraic form. Separate margins and a restricted training-only bank describe an implementation choice, not an invented similarity principle. SSPC already explicitly learns a cross-fitted utility gate in multimodal fake-news detection. CCR studies an incremental scalar-CLIP decision in frozen heads; it does not establish invention of utility gates or factual verification. See the verified-overlap report for access limitations and exact sources.

## Independent evaluation and claims

External status: **blocked_access**. Public metadata/URLs do not supply the required full authorized image set. No gated data, external scores, or images were fabricated or redistributed. The local authorized-data adapter is supplied but was not run against real VERITE images; it is not evidence of completed external validation. VERITE labels describe truthful/miscaptioned/out-of-context pairs, whereas Fakeddit labels are distant supervision; any future evaluation must be reported as cross-task/domain transfer.

Supported claims: an auditable implementation and controlled exploratory comparison, with preserved negative results and measured routing/calibration tradeoffs. Unsupported claims: generalization to an independent benchmark, improved factual verification, SOTA, first invention, or guaranteed journal/Scopus acceptance. A negative-result paper would need a sharper hypothesis, justified external evaluation or a stronger multi-dataset robustness study, and complete prior-art verification. No journal submission is recommended from the present evidence alone.

## Reproduce

From the project directory, using the existing virtualenv Python:

```powershell
python -m unittest src.research_extension.ccr.tests -v
python -m src.research_extension.ccr smoke
python -m src.research_extension.ccr train
python -m src.research_extension.ccr freeze
python -m src.research_extension.ccr evaluate
python -m src.research_extension.ccr stress
python -m src.research_extension.ccr report
python -m src.research_extension.ccr verify-preservation
# Only after obtaining approved official data; never retune to its labels:
python -m src.research_extension.ccr verite-local --authorized-dir C:\path\to\authorized_VERITE
```

CLI defaults to this project root and writes only CCR namespaces. Completed fold checkpoints and epoch resume states are hash-bound; scientific results never overwrite historical original/BRAS artifacts. The report package excludes all image datasets and contains scripts, policies, predictions, statistics, figures and provenance needed to inspect the results. Large training caches/checkpoints stay in the project.


## Final verification

17 tests passed; 30 checkpoint replays reproduced 6,000 OOF probabilities, and 72 metric rows matched scikit-learn. All 58,759 original/BRAS files passed the end-of-run hash audit. A donor-reuse loading bug was fixed only for synthetic swaps, with archived source contracts and an explicit amendment; classification predictions were unchanged. Seven figures are supplied as PNG and PDF, including a validation-utility diagnostic. B/CCR curves overlap exactly.
