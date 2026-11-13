# BRAS working-method methodology

**BRAS** means *Bidirectional Reference-Adjusted Similarity* in this codebase. This is a working label for an exploratory experiment, not an established method name or novelty claim.

## Inputs and feature contract

The historical project remains the source of the official split manifest and frozen BERT/ResNet features. Its `data/features/extraction.json` points to sample-addressed float32 cache chunks: BERT CLS features (768), ResNet50 pooled features (2,048), and paired CLIP cosine (1). The historical cache does not retain projected CLIP vectors, so `extract-clip` reuses only the pinned local CLIP ViT-B/32 model and saves normalized 512-dimensional text and image projections. It verifies their paired dot against the historical scalar before committing each chunk. No BERT or ResNet re-extraction occurs.

Each retained manifest row must preserve `sample_id`, official split, binary label, `label6`, source-family `group_id`, image byte hash, decoded-pixel hash, text key, and the existing input hash `SHA256(text + NUL + image_sha256)`. Feature chunks carry IDs, input hashes and splits. Receipts contain per-file SHA-256 digests. Incomplete or mismatched chunks are rejected; extraction resumes from the next verified chunk.

## Reference-bank algorithm

Defaults are bank size 4,096, top-*k* 16 and seed 20,260,916. Bank choice is independent of labels: sort training rows by `SHA256(seed + NUL + sample_id)` and greedily retain rows unique on available sample ID, linked-source-family ID, pair input hash, normalized text key, image byte hash and pixel hash, until the requested size is reached. Validation/test rows are ineligible. Bank IDs, input hashes, full bank hash and protocol hash are persisted.

Let `t_i` and `v_i` be the L2-normalized float32 CLIP projections for query `i`; the paired cosine is `s_i = t_i · v_i`. For each query-bank candidate, exclude a candidate if it shares the query sample ID, known linked-source family, exact pair/input hash, normalized text, image byte hash, or pixel hash. The same exclusion mask is applied in both directions. Candidate labels are not consulted.

For query batch size `b` and bank size `B`, compute only `b × B` dot products at a time. For text-conditioned matching, rank `t_i · v_j`; for image-conditioned matching, rank `v_i · t_j`. Keep up to `k` greatest valid nonmatching scores. If `n_i` valid scores exist, average `min(k,n_i)` of them:

`m_text_i = s_i − mean(top_min(k,n_i){t_i · v_j : j valid})`

`m_image_i = s_i − mean(top_min(k,n_i){v_i · t_j : j valid})`

If fewer than `k` valid candidates remain, use all available candidates. If none remain, store NaN and fail full feature generation rather than invent a margin. In an explicit missing-bank sensitivity test, the fitted training mean is used only as a neutral post-hoc input fallback and the missing-row count is reported.

The saved BRAS vector is `[s_i, m_text_i, m_image_i]`; column 0 is replaced with the original cached scalar after verifying the re-extracted cosine within tolerance. Feature scalers are fitted on training rows only. The historical BERT/ResNet/cosine normalization is recomputed and checked against the immutable historical normalizer. Margin normalization is separately fitted on training rows.

## Model variants and fair-comparison boundary

| ID | Inputs | Training treatment |
|---|---|---|
| A | BERT + ResNet | Reuse historical baseline checkpoints and predictions; no retraining |
| B | BERT + ResNet + raw cosine | Reuse historical raw-similarity checkpoints; no retraining |
| C | BERT + ResNet + two margins | New head |
| D | BERT + ResNet + cosine + two margins | Primary proposed head |
| E | BERT + ResNet + cosine + two deterministic random controls | Matched three-extra-feature width to D |
| F | Raw cosine only | Similarity-only head |
| G | Three BRAS features only | Similarity-only head |
| I | BERT + ResNet + cosine + two fixed CLIP unimodal random projections | Same extra-feature width as D; tests extra CLIP input beyond pair margins |

All heads use the locked historical AdamW settings, optimizer order, split, validation Macro-F1 selection and tie-break policy, and seeds 42/123/2026. Historical A/B checkpoint reuse preserves the old baseline and raw-scalar model exactly. New heads are trained with the same base architecture and seeds; they are not all parameter-matched to one another. D, E, and I have the same feature width and parameter count. F/G differ in input width. Reported overhead excludes the fixed encoder extraction costs unless explicitly labelled otherwise.

## Prespecified analysis

Primary metric: Macro-F1. Primary comparison: D versus B, classifier seed 42, two-sided exact McNemar test for paired correctness and 5,000 common-index paired bootstrap resamples (bootstrap seed 20,260,916) for Macro-F1 and accuracy differences. Other classifier seeds summarize fit variation on the same evaluation examples and are not pooled as independent datasets. Secondary seed-42 McNemar tests receive Holm correction. The original paired-statistics writer initially mislabeled the bootstrap seed as the classifier seed; corrected, hash-linked recomputations are saved as `paired_comparisons_corrected.json` with a separate receipt. Use those corrected tables. The underlying predictions, effect estimates, intervals, and primary p-values are unchanged.

The evaluator saves accuracy, Macro-F1, weighted-F1, fake-class precision/recall/F1, ROC-AUC, trapezoidal PR-AUC and average precision; individual-seed and mean/sample-SD tables; confusion matrices; per-example predictions; paired corrected/new errors; per-label six-way breakdowns; train-defined cosine quartiles; high-similarity fake and low-similarity real slices; and similarity feature distributions by binary/six-way labels. Group sizes and single-class/under-100 flags travel with each subgroup result.

Synthetic random and hard-negative image swaps are analyzed as correspondence ranking and fixed-head prediction sensitivity only. They are not assigned factual labels. Bank-size, top-*k*, missing-reference and directional-ablation probes reuse a fixed trained head and are reported as sensitivity checks, not retrained comparisons.

## Fresh disjoint evaluation slice

The extension also retrieves exactly 1,000 examples (500 per binary label) from the previously unused official Fakeddit public-test metadata. Selection is stable-hash-ranked within label after excluding every retained ID, known linked-submission family, normalized title and image URL. Retrieved images are then checked against all retained file and decoded-pixel hashes and against other selected rows. Exactly 1,000 image pairs were kept; 1,152 candidate URLs were attempted. The selected image bytes total 85,552,431. The selection and feature receipts record official metadata, original-manifest, source-family audit and per-image hashes.

This set is disjoint from the retained 55,000 rows by ID, known family, normalized text, image URL, byte hash and decoded-pixel hash. It is a fresh same-dataset evaluation slice, not an external benchmark or independent collection: it comes from the same Fakeddit public-test pool, shares the original label process, and is still susceptible to event dependence and image-availability selection. No model, feature, threshold or bank choices were tuned on this slice. Pinned BERT/ResNet/CLIP encoders and the training-fitted normalizers are reused without refitting. Results are reported beside, not pooled with, the previously inspected 2,500-row subset.

## Source map

* `src/research_extension/common.py`: immutable run protocol, manifest/hash checks, historical cache adapter, train-only scaler, environment and memory capture.
* `src/research_extension/audit.py`: official ID/label/split joins, source-family and normalization checks, original-artifact hash inventory.
* `src/research_extension/extraction.py`: resumable pinned-CLIP vector extraction only.
* `src/research_extension/reference.py`: deterministic bank selection, exclusion masks, directional top-*k* margins.
* `src/research_extension/features.py`: bank receipts, generated BRAS and control features.
* `src/research_extension/training.py`: model heads, train-only variant normalization, resume checkpoints, historical A/B reuse.
* `src/research_extension/statistics.py` and `evaluation.py`: metrics, exact paired tests, bootstrap, error and subgroup output.
* `src/research_extension/stress.py`: synthetic correspondence and reference-bank sensitivity.
* `src/research_extension/reporting.py`: publication-style figures made from saved outputs.
* `src/research_extension/fresh_holdout.py`: stable disjoint holdout selection, bounded image retrieval, pinned local feature extraction and evaluation.
* `run_research_extension.py`: argparse stages and command log.
