# BRAS extension â€” results and audit summary

**Run:** `bras_b4096_k16_s20260916`
**Protocol:** 4,096 training references; top-*k*=16; seeds 42, 123 and 2026; fake label 0; primary metric Macro-F1; no test-guided tuning.

## Main result

Adding both reference-relative margins did not show a reliable benefit over the existing raw-CLIP-cosine head. On the previously inspected 2,500-row public test subset, D's mean Macro-F1 across three classifier seeds was 0.8729 versus 0.8721 for B, but the prespecified seed-42 comparison was negative: **D âˆ’ B = âˆ’0.00566 Macro-F1** (95% paired row-bootstrap CI âˆ’0.01473 to 0.00366; exact McNemar *p*=0.2467). D fixed 56 B errors and introduced 70 new errors.

On the fresh, balanced 1,000-row same-dataset slice, D's mean Macro-F1 was 0.8558 versus 0.8594 for B. The seed-42 difference was **âˆ’0.01409** (95% CI âˆ’0.03004 to 0.00095; exact McNemar *p*=0.0925); D corrected 23 B errors and introduced 37 new errors. Both primary intervals include zero. The point estimates do not support a consistent gain from the two margins.

The fresh slice adds a useful check on new records, but it is not an external benchmark or independent data collection: it comes from the same Fakeddit public-test pool and distant-supervision labels. The original 2,500-row test subset was already inspected in the earlier project and remains exploratory.

## Results on the original 2,500-row subset

Three-seed mean scores; Macro-F1 shows sample SD across fitted seeds. A and B reproduce the historical baseline and raw-cosine checkpoints.

| Variant | Accuracy | Macro-F1 mean Â± seed SD | ROC-AUC |
|---|---:|---:|---:|
| A â€” BERT + ResNet | 0.8555 | 0.8474 Â± 0.0056 | 0.9331 |
| B â€” A + raw CLIP cosine | 0.8787 | 0.8721 Â± 0.0094 | 0.9466 |
| C â€” A + two margins | 0.8769 | 0.8700 Â± 0.0046 | 0.9490 |
| D â€” A + cosine + two margins | 0.8791 | 0.8729 Â± 0.0032 | 0.9481 |
| E â€” A + cosine + two random controls | 0.8752 | 0.8677 Â± 0.0048 | 0.9459 |
| F â€” cosine-only head | 0.7204 | 0.7097 Â± 0.0013 | 0.7778 |
| G â€” three-similarity-only head | 0.7339 | 0.7204 Â± 0.0003 | 0.8014 |
| I â€” A + cosine + random CLIP projections | 0.8763 | 0.8700 Â± 0.0057 | 0.9463 |

The A/B values reproduce the previously reported project results to rounding: baseline 85.55% accuracy / 84.74% Macro-F1 / .9331 ROC-AUC; raw-cosine model 87.87% / 87.21% / .9466. The original text-only, image-only and baseline results are recorded in the project audit; they were not changed or folded into this extension's model table.

## Fresh 1,000-row same-dataset slice

The rows were selected before evaluation with fixed ID hashing and balanced label quotas. All 24 checkpoints were evaluated without tuning on these rows.

| Variant | Accuracy | Macro-F1 mean Â± seed SD | ROC-AUC |
|---|---:|---:|---:|
| A â€” BERT + ResNet | 0.8330 | 0.8325 Â± 0.0098 | 0.9189 |
| B â€” A + raw CLIP cosine | 0.8597 | 0.8594 Â± 0.0143 | 0.9389 |
| C â€” A + two margins | 0.8557 | 0.8554 Â± 0.0042 | 0.9398 |
| D â€” A + cosine + two margins | 0.8560 | 0.8558 Â± 0.0044 | 0.9392 |
| E â€” A + cosine + two random controls | 0.8513 | 0.8509 Â± 0.0065 | 0.9331 |
| F â€” cosine-only head | 0.7017 | 0.7012 Â± 0.0026 | 0.7611 |
| G â€” three-similarity-only head | 0.7120 | 0.7106 Â± 0.0026 | 0.7868 |
| I â€” A + cosine + random CLIP projections | 0.8567 | 0.8566 Â± 0.0152 | 0.9375 |

Across-seed means put D 0.0052 above E on the original subset and 0.0049 above E on the fresh slice. Against I, D is 0.0029 higher on the original subset and 0.0008 lower on the fresh slice. These are descriptive mean differences across three fits, not independent test replications. The paired seed-42 secondary comparisons do not survive the prespecified Holm family on either set.

## Primary paired comparisons

| Evaluation rows | N | D âˆ’ B Macro-F1 | 95% paired bootstrap CI | McNemar *p* | B errors corrected / new errors |
|---|---:|---:|---:|---:|---:|
| Previously inspected Fakeddit test subset | 2,500 | âˆ’0.00566 | [âˆ’0.01473, 0.00366] | 0.2467 | 56 / 70 |
| Fresh disjoint Fakeddit test slice | 1,000 | âˆ’0.01409 | [âˆ’0.03004, 0.00095] | 0.0925 | 23 / 37 |

Rows are paired and the bootstrap resamples rows, not source events. The 1,000-row sample is balanced by construction. Its image retrieval rate and composition can differ from the underlying official metadata pool.

## Six-way category slice

Seed 42 accuracy by official six-way label on the original test subset. Counts are shown because some groups are small.

| Label | N | B accuracy | D accuracy |
|---|---:|---:|---:|
| True | 991 | 0.8406 | 0.8385 |
| Satire/Parody | 144 | 0.8333 | 0.8264 |
| False Connection | 511 | 0.8767 | 0.8708 |
| Imposter Content | 56 | 0.8929 | 0.9107 |
| Manipulated Content | 695 | 0.9683 | 0.9655 |
| Misleading Content | 103 | 0.9320 | 0.8641 |

These are descriptive slices of distant-supervision categories, not separate confirmatory tests. The Misleading Content group has only 103 rows; results should not be generalized from this breakdown.

## Synthetic correspondence and sensitivity checks

On a fixed 500-row sample from the original test subset, random training-bank image replacements had lower original-pair cosine in 90.2% of rows (mean original-minus-swapped cosine 0.0955); the sum of the two margins was higher for original pairs in 92.0%. For â€œhardâ€ replacements chosen as the most text-similar eligible bank image, the original cosine was lower in 68.2% of rows. These are correspondence rankings only: the swapped pairs were not assigned fake/real labels.

Swaps changed the fixed-head prediction on 31.0% (B) and 34.8% (D) of random replacements, and 20.8% (B) and 20.4% (D) of the hard replacements. This is prediction sensitivity, not evidence that either prediction is factually correct. On this same 500-row stress slice, the D head's Macro-F1 ranged from 0.8772 at *k*=1 to 0.9052 at *k*=64, and changed by 4.0% of predictions when the bank was reduced from 4,096 to 256. These post-hoc fixed-head checks do not replace a retrained ablation.

## Fresh-slice selection and integrity

The initial audit found 56,819 official public-test metadata rows beyond the retained manifest and no cached images for them. The extension attempted 1,152 official image URLs, retained 1,000 pairs (500 fake / 500 real; 85,552,431 image bytes), and recorded failed retrievals and exclusions. The selected rows have zero overlap with the retained data on sample ID, audited linked-source family, normalized text, URL, file SHA-256 or decoded-pixel SHA-256. Their official IDs, binary labels, six-way labels and test membership are verified against the SHA-checked official TSVs.

This is the same Fakeddit source and public-test pool, so it cannot establish transfer to other platforms or benchmark definitions. Source-family exclusion does not prove event-level independence. Fakeddit labels are weak/distant supervision, and image availability can induce selection bias. CLIP web-scale pretraining overlap is unmeasured.

## Paired-statistics correction and source artifacts

The first output writer allowed the bootstrap RNG seed to overwrite the classifier-seed field, which also left the planned secondary Holm filter empty. Predictions, metric deltas, confidence intervals and primary McNemar p-values did not change. The source code now separates `seed` (classifier) and `bootstrap_seed`; corrected tables are recomputed from the saved predictions and have separate receipts with prediction and source-table hashes. Use the corrected files below. The original paired tables remain intact for the audit trail.

- Original subset: `results/research_extension/bras_b4096_k16_s20260916/paired_comparisons_corrected.json`
- Fresh slice: `data/features/research_extension/fresh_holdout/paired_comparisons_corrected.json`
- Original-subset figures: `results/research_extension/bras_b4096_k16_s20260916/plots/`
- Fresh manifest and selection audit: `data/features/research_extension/fresh_holdout/manifest.csv` and `selection_audit.csv`

## Commands executed

From the project root, with the project's Python 3.13.17 virtual environment:

```powershell
$researchPython = 'python'
& $researchPython -B run_research_extension.py audit --verify-images
& $researchPython -B run_research_extension.py extract-clip --threads 6 --encoder-batch-size 16
& $researchPython -B run_research_extension.py prepare-fresh-holdout --per-class 500 --download-workers 12
& $researchPython -B run_research_extension.py smoke
& $researchPython -B run_research_extension.py extract-fresh-holdout --encoder-batch-size 16
& $researchPython -B run_research_extension.py build-reference-bank
& $researchPython -B run_research_extension.py generate-features
& $researchPython -B run_research_extension.py train
& $researchPython -B run_research_extension.py evaluate
& $researchPython -B run_research_extension.py evaluate-fresh-holdout
& $researchPython -B run_research_extension.py recompute-paired --dataset-scope retained-test
& $researchPython -B run_research_extension.py recompute-paired --dataset-scope fresh-holdout
& $researchPython -B run_research_extension.py stress-test --limit 500
& $researchPython -B run_research_extension.py plots
& $researchPython -B run_research_extension.py audit --verify-images
& $researchPython -B run_research_extension.py audit --verify-preservation
& $researchPython -B -m unittest discover -s tests -p test_research_extension.py -v
```

The command log and per-stage receipts are under `results/research_extension/`; test output is also captured in the task work log. Final preservation audit verified all 1,454 inventoried historical files byte-for-byte.

## Claim boundary

The results do not support a reliable improvement over raw cosine on either primary seed-42 comparison. They do not establish causal image-text reasoning, factual verification, cross-dataset generalization, novelty, or publication readiness. The targeted prior-art evidence and access limitations are recorded in `literature_comparison.md`; a broader review and evaluation on a separate compatible benchmark remain necessary.
