# CCR decision and ablation interpretation

**CCR did not improve on B. The validation-selected ridge threshold was the all-B endpoint for all three seeds.** Intercept, tree and cosine gates also chose all B. Operational CCR and the unguarded ridge candidate therefore reproduce B's probabilities exactly on both exploratory test sets. This is a negative result for the prespecified selector, not a successful learned conditional policy.

| Previously viewed evaluation set | B Macro-F1 | CCR Macro-F1 | B log-loss | CCR log-loss |
|---|---:|---:|---:|---:|
| Historical 2,500 | 0.872115 | 0.872115 | 0.297812 | 0.297812 |
| Previously viewed 1,000 | 0.859407 | 0.859407 | 0.336571 | 0.336571 |

Values are means across classifier seeds42,123,2026. CCR−B loss, Brier, accuracy and Macro-F1 differences are exactly zero for every seed, with zero corrected/new errors. The bootstrap[0,0] intervals follow from identical predictions; they do not prove general equivalence of all possible gates or independent generalization.

## Ablations and tradeoffs

The prespecified random gate matched the ridge validation B-routing fraction, which is1.0. It consequently degenerates to B and is not an informative randomized negative control. No replacement random fraction was chosen after seeing results. This limits the negative-control evidence and is reported rather than hidden.

Confidence/entropy gates routed a subset to A for seed42; their other seeds chose all B. Their mean exploratory loss and Macro-F1 were worse than B on both sets. A validation-fixed temperature baseline lowered mean log-loss from0.297812 to0.292807 on the historical set and from0.336571 to0.327896 on the previously viewed1,000. However, historical mean Brier increased from0.089077 to0.089311, while the second set's Brier decreased. Classification decisions/Macro-F1 did not change. These are descriptive tradeoffs, not independent confirmation or a new algorithmic contribution.

| dataset | variant | Macro-F1 | log_loss | Brier |
|---|---|---|---|---|
| original_test | A | 0.847358 | 0.333608 | 0.103676 |
| original_test | B | 0.872115 | 0.297812 | 0.089077 |
| original_test | CCR | 0.872115 | 0.297812 | 0.089077 |
| original_test | B_temperature | 0.872115 | 0.292807 | 0.089311 |
| original_test | confidence_gate | 0.870112 | 0.300207 | 0.089896 |
| original_test | entropy_gate | 0.869118 | 0.300052 | 0.090095 |
| previous_fresh_test | A | 0.832537 | 0.391883 | 0.120598 |
| previous_fresh_test | B | 0.859407 | 0.336571 | 0.102363 |
| previous_fresh_test | CCR | 0.859407 | 0.336571 | 0.102363 |
| previous_fresh_test | B_temperature | 0.859407 | 0.327896 | 0.101856 |
| previous_fresh_test | confidence_gate | 0.852672 | 0.347343 | 0.105680 |
| previous_fresh_test | entropy_gate | 0.851308 | 0.346531 | 0.105846 |

The ridge utility estimate does not transfer cleanly from OOF heads to final heads: for seed42 its mean predicted utility changes from0.08750 in OOF to−0.000344 on validation, and its validation utility MAE0.3190 exceeds the intercept's0.2316. Validation and exploratory utility-bin plots show why predicted utility should not be read as a factual-reliability score. No corrective refit was made after examining these diagnostics.

## Observational and synthetic checks

Historical cosine-quartile prevalence reproduced the blueprint exactly: real-label rates are83/753(11.0%) in Q1 and380/539(70.5%) in Q4, using training-defined boundaries. This strong association is observational. The six-way/subreddit panels include counts, family counts and uncertainty; they do not establish source-generalized benefits.

For seed42, random swaps flip31.0% of B/CCR predictions, and hard swaps flip20.8%; the corresponding A rates are19.6% and11.0%. Mean swapped-minus-original cosine is−0.09546 for random swaps and+0.02296 for hard swaps. These compare pair sensitivity only; no synthetic pair received a fabricated truth label. CCR offers no additional robustness because it is B.

## Independent endpoint and publication decision

**VERITE: not evaluated.** The user confirmed no approved image copy; institutional authorization is required by the [official release](https://huggingface.co/datasets/stefpapad/VERITE). No gated image data or test outcomes were fabricated. The full-image adapter is supplied but unvalidated on real VERITE images. Public metadata and a broken-URL subset cannot satisfy the full preregistered independent endpoint.

**NO-GO for journal submission as a superior new detection method.** Supported: reproducible implementation, controlled exploratory failure of this selector, calibration tradeoffs, and preserved BRAS negative findings. Unsupported: independent improvement, factual verification, SOTA, first invention, or acceptance in any indexed journal. [SSPC](https://aclanthology.org/2026.emnlp-main.1279/) already uses cross-fitted utility gating; summed BRAS margins have the CSLS form. A negative-result/robustness manuscript needs a stronger empirical contribution and independent evaluation before a submission recommendation is warranted.

## Verification and implementation amendment

17 unit/integration tests passed. All30 OOF checkpoints reproduced6,000 sampled held-out probabilities. Four metrics across72 saved prediction/model rows matched scikit-learn. All58,759 pre-existing project files were rehashed and remained byte-identical. The full OOF run took349.5 seconds and peaked at1.90 GiB process working set.

A repeated-donor bug initially stopped synthetic-swap feature loading. It was fixed by loading unique donors then restoring order, with a regression test and archived source/frozen-policy contracts. No trained model, selection threshold, classification prediction, metric or original/BRAS artifact changed. `results/amendments/duplicate_donors/amendment.json` records the exact source-hash change. The earlier smoke source receipt similarly records changes only to plotting and the unexecuted external adapter. The scientific protocol JSON remains unchanged.
