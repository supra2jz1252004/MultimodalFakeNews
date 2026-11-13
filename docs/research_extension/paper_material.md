# Paper-planning notes (exploratory study)

## Working title

**Reference-Relative CLIP Similarity for Multimodal Fake-News Classification: A Controlled Fakeddit Study**

Alternative short title: **Bidirectional Reference-Adjusted Similarity (BRAS)**. Keep BRAS as a working name until the prior-art review is extended and the exact operation is compared against related reference-bank and retrieval methods.

## Research questions

1. Do two bidirectional training-reference margins add predictive information beyond raw CLIP cosine?
2. Are any differences specific to the supplied pair, beyond the effect of generic extra CLIP inputs or a wider classifier head?
3. How do the features and errors behave among high-similarity fake-labeled pairs and low-similarity real-labeled pairs?
4. Do effects differ across the six Fakeddit supervision categories?
5. What computation and reference-bank sensitivity accompany any observed difference?

## Candidate contribution wording

This study implements and evaluates a fixed-bank, bidirectional nearest-reference margin feature for a frozen-feature Fakeddit pipeline. It contrasts raw cosine, two margins, the combined feature, similarity-only heads, randomized extra inputs, and fixed CLIP unimodal projections while preserving the original training protocol and checkpoints. The contribution, if supported by the executed results, is an empirical characterization of this particular feature construction and its controls. It is not a claim that multimodal similarity, reference banks, bidirectional matching, similarity gating, or controlled ablation are new.

## Methods section draft

The dataset is a filtered, availability-dependent subset of Fakeddit with 50,000 train, 2,500 validation, and 2,500 public test pairs under their official memberships. Binary labels are 0=fake and 1=real; the official six-way labels are retained for descriptive analysis. All splits are checked against SHA-verified official TSVs by sample ID. The fixed manifest contains source-family and content hashes. The binary labels follow Fakeddit distant supervision and should not be interpreted as independent fact-check judgments.

Text and image representations are the existing frozen BERT Base Uncased (768-D) and ResNet50 ImageNet V2 (2,048-D) caches. The historical paired CLIP ViT-B/32 cosine remains the raw-similarity feature. Since separate CLIP projections were not stored, pinned local CLIP weights are used to recover and cache normalized 512-D projected text and image embeddings only. The extracted paired dot is checked against the saved scalar, and all cache chunks are bound to IDs, splits, input hashes and SHA-256 receipts.

The reference bank is selected deterministically from training rows only using the fixed seed and hash-ranked sample IDs. Greedy selection excludes repeated known linked-source families, pair hashes, normalized text, and image hashes. For every query, same-ID, same-family, exact-pair, same-text, and same-image references are excluded. We calculate the paired cosine and subtract the mean of the top 16 remaining text-to-bank-image similarities and image-to-bank-text similarities. The initial bank size is 4,096. A query batch is compared with the bank in bounded matrix operations; no full-dataset all-pairs matrix is constructed. Missing candidates, fallback and valid counts are recorded as described in `methodology.md`.

The baseline head and raw-cosine head use their previously validated checkpoints; they are not overwritten or retrained. New feature heads use the historical BERT/ResNet classifier and the locked training policy with seeds 42, 123 and 2026, train-only normalization, and validation Macro-F1 selection. The principal endpoint is Macro-F1 and the principal comparison is combined cosine-plus-margins versus raw cosine, classifier seed 42. In addition to the previously inspected public test subset, we acquired a balanced 1,000-row slice from unused official Fakeddit public-test records. It is source-family-, text-, URL-, and content-hash-disjoint from the retained manifest, but remains same-dataset evaluation, not cross-dataset or independent-collection evidence. No choices were tuned on it.

## Preliminary results

Across the original 2,500-row test subset, D's three-seed mean Macro-F1 is 0.8729 (SD 0.0032) versus 0.8721 (SD 0.0094) for B. On the prespecified seed-42 paired comparison, D is lower by 0.0057 Macro-F1 (95% paired row-bootstrap CI −0.0147 to 0.0037; exact McNemar p=0.247). On the fresh 1,000-row slice, D's mean is 0.8558 (SD 0.0044) versus 0.8594 (SD 0.0143) for B. The seed-42 difference is −0.0141 (95% CI −0.0300 to 0.0010; p=0.092). Both intervals include zero; the primary comparisons do not show a reliable benefit from adding both margins. Use the exact tables and caveats in `results_summary.md` for all reported digits.

Report the null-to-negative primary findings with the same prominence as small positive control differences. Do not insert benchmark rankings or compare this filtered subset's raw accuracy directly against papers with different splits, labels, sample selection, or protocols.

The synthetic swap experiment measures pairwise correspondence ranking and fixed-head prediction sensitivity only. It does not establish that the model recognizes a causal image-text relation or verifies factual truth. Similarity-score associations and six-way category slices are descriptive and must include counts.

## Threats to validity

* **Label validity:** Fakeddit uses community/subreddit-derived distant supervision and six broad categories; it is not equivalent to human claim fact-checking.
* **Test reuse:** the historical public test subset was already inspected in the prior study. New paired inference on it is exploratory. The fresh same-dataset slice is row/source/content-disjoint from the retained set but still inherits Fakeddit's source and label process.
* **Source/event dependence:** known linked-source families and exact content are excluded across the retained manifest, but event identity, perceptual near-duplicates, and publisher/image provenance may still be dependent or uncertain. Row-level bootstrap intervals may understate event-cluster uncertainty.
* **Pretraining overlap:** CLIP's web-scale pretraining data may overlap with public examples; this is unmeasured.
* **Subset shift:** availability-based image retrieval and conservative source-family exclusion alter the original population composition.
* **Method search:** the initial defaults are specified in advance, but any follow-up tuning must use training/validation only and be reported as exploratory if influenced by test inspection.
* **External validity:** no independent benchmark or cross-dataset set was evaluated. The fresh Fakeddit slice cannot establish cross-platform transfer; no benchmark label crosswalk was available.
* **Mechanism:** predictive gains and attention-free scalar features do not prove causal reasoning or establish that the classifier uses margins for the intended reason.
* **Runtime:** head inference excludes image/text encoder inference and persistent-bank construction unless the specific table says otherwise. Feature construction cost is reported separately.

## Before submission

Re-review the full texts of MCSF, CMCM/PE-CMA, COOLANT++, and ViLID; search for equivalent top-*k* reference-relative margin constructions; obtain a fresh untouched or independent evaluation set with a defensible label crosswalk; consider source/event-clustered uncertainty; and rerun the prespecified protocol on that independent evaluation. Do not make a novelty, publication, or indexing guarantee.
