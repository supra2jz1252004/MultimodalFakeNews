# Robustness and verification

The primary uncertainty uses 336 connected components that join official families and repeated article/image/caption content. Replacing them with the 338 original families leaves both primary classifier conclusions unresolved; family-only paired Macro-F1 intervals are:

| comparison | seed | Macro_F1_CI_low | Macro_F1_CI_high |
|---|---|---|---|
| B − A | 42 | -0.00702 | 0.05137 |
| D − B | 42 | -0.02956 | 0.01150 |

No exact retained Fakeddit caption, byte-hash or pixel-hash overlap was found. The predeclared overlap-exclusion analysis therefore retains all 1,001 records and equals the full analysis. This is not evidence against near duplicates, shared events or encoder-pretraining overlap.

All alignment tie sensitivities:

| score | category | exact_ties | strict | half | tolerance_1e_6 |
|---|---|---|---|---|---|
| cosine | miscaptioned | 0 | 0.56509 | 0.56509 | 0.56509 |
| cosine | out-of-context | 0 | 0.82154 | 0.82154 | 0.82154 |
| m_text | miscaptioned | 0 | 0.65680 | 0.65680 | 0.65680 |
| m_text | out-of-context | 0 | 0.82154 | 0.82154 | 0.82154 |
| m_image | miscaptioned | 0 | 0.56509 | 0.56509 | 0.56509 |
| m_image | out-of-context | 0 | 0.80923 | 0.80923 | 0.80923 |
| bras_mean | miscaptioned | 0 | 0.61538 | 0.61538 | 0.61538 |
| bras_mean | out-of-context | 0 | 0.82769 | 0.82769 | 0.82769 |

Source-stratified paired rankings are descriptive; small cells and post-stratification do not support a source-specific superiority claim:

| score | category | source | pairs | ranking_half | mean_margin |
|---|---|---|---|---|---|
| cosine | miscaptioned | reuters | 78 | 0.64103 | 0.01428 |
| cosine | miscaptioned | snopes | 260 | 0.54231 | 0.00669 |
| cosine | out-of-context | reuters | 74 | 0.79730 | 0.03221 |
| cosine | out-of-context | snopes | 251 | 0.82869 | 0.04540 |
| bras_mean | miscaptioned | reuters | 78 | 0.67949 | 0.01775 |
| bras_mean | miscaptioned | snopes | 260 | 0.59615 | 0.00952 |
| bras_mean | out-of-context | reuters | 74 | 0.81081 | 0.03129 |
| bras_mean | out-of-context | snopes | 251 | 0.83267 | 0.04441 |

Numerical shared-modality cancellation checks give zero ranking disagreements in the theoretically invariant directions. The maximum margin discrepancy is below 1.8e-7, consistent with float32 rounding. No alternative score direction, threshold, neighborhood size or bank was selected on VERITE.

The final executed verifier independently recomputed 90 classifier metrics and 8 alignment AUC/AP values using scikit-learn. It checked 8 paired-ranking values, replayed 768 actual VERITE probabilities for 12 frozen heads (64 deterministic hash-selected records per head) with identical decisions and maximum probability discrepancy below 3e-7, and checked 120 expanded cluster-bootstrap Macro-F1 estimates against scikit-learn. Recomputed float64 CLIP vector products differed from stored cosines by at most 6.7e-08. All 24 unit tests passed, including image corruption, unsafe ZIP members, label mapping, paired content, cluster resampling, statistical ties, and frozen-head calculations. Dataset/source/model checksums are revalidated on cached resume. Every tested resume stage passed without changing protocol, predictions, model metrics or paired statistical outputs.

All six aggregate figures were visually inspected for legibility and clipping. PNG, vector PDF and SVG copies are available. No restricted images were used in these figures.

`verification.json`, `tests.txt`, `figure_manifest.json` and `preservation_check.json` provide machine-readable or captured receipts. Timings and exact feature contracts are separate from inference results; no performance number was inferred from a synthetic test fixture.
