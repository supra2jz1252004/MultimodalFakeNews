# Correspondence ranking, separate from classifier transfer

All scores use newly extracted pinned CLIP ViT-B/32 features. No VERITE examples enter the Fakeddit reference bank. Exact score ties count 0.5; strict/tolerance checks are retained. Uncertainty resamples shared-content components and uses the same draws for compared scores.

| score | comparison | pairs | ranking | CI_low | CI_high | mean_margin |
|---|---|---|---|---|---|---|
| cosine | miscaptioned | 338 | 0.56509 | 0.51479 | 0.61607 | 0.00844 |
| cosine | out-of-context | 325 | 0.82154 | 0.77846 | 0.86154 | 0.04240 |
| m_text | miscaptioned | 338 | 0.65680 | 0.60651 | 0.70588 | 0.01439 |
| m_text | out-of-context | 325 | 0.82154 | 0.77846 | 0.86154 | 0.04240 |
| m_image | miscaptioned | 338 | 0.56509 | 0.51479 | 0.61607 | 0.00844 |
| m_image | out-of-context | 325 | 0.80923 | 0.76543 | 0.84878 | 0.04045 |
| bras_mean | miscaptioned | 338 | 0.61538 | 0.56468 | 0.66568 | 0.01142 |
| bras_mean | out-of-context | 325 | 0.82769 | 0.78395 | 0.86769 | 0.04142 |

Paired primary effects and corrected tests:

| category | BRAS_mean_minus_cosine | CI_low | CI_high | Holm_p | gained | lost |
|---|---|---|---|---|---|---|
| miscaptioned | 0.05030 | 0.02065 | 0.08284 | 0.00820 | 24 | 7 |
| out-of-context | 0.00615 | -0.00923 | 0.02154 | 0.68003 | 4 | 2 |

Global score discrimination on all 1,001 rows, positive class truthful; these endpoints differ from within-family paired ranking:

| score | ROC_AUC | AUC_CI_low | AUC_CI_high | AP | AP_CI_low | AP_CI_high |
|---|---|---|---|---|---|---|
| bras_mean | 0.66238 | 0.63884 | 0.68525 | 0.48130 | 0.45883 | 0.51175 |
| cosine | 0.66388 | 0.63966 | 0.68679 | 0.47688 | 0.45433 | 0.50854 |
| m_image | 0.65490 | 0.63140 | 0.67813 | 0.47033 | 0.44854 | 0.49924 |
| m_text | 0.66438 | 0.64165 | 0.68675 | 0.49061 | 0.46700 | 0.52201 |

The miscaptioned ranking gain does not imply improved global AUC: BRAS mean AUC is slightly lower than raw cosine. No global AUC superiority test was prespecified or claimed.

All directional comparisons, discrimination AUC/AP and intervals, source denominators, and half/strict/tolerance tie counts appear in `alignment_statistics.json`. With equal reference candidate sets, the image-side penalty cancels for shared-image true/miscaptioned pairs, while the text-side penalty cancels for shared-caption true/out-of-context pairs. Numerical equivalence checks are recorded; identical rankings in those directions are algebraic, not independent supporting experiments. Remaining mean-margin changes and ranking gains can diverge.

Neither raw cosine nor BRAS margins independently verify factual truth. Their ranking asks whether the supplied true pairing scores above the paired misleading instance; classifier diagnostic accuracy measures a different fixed decision rule.
