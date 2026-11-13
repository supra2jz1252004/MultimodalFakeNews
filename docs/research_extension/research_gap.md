# Research gap and claim boundary

## What is already established

Prior work has already established CLIP-derived image-text cosine features and similarity-guided fusion (FND-CLIP), bidirectional text/image alignment objectives (SARD and the IJCAI-2025 MFND model), explicit consistency branches on Fakeddit (CMCM), consistency suppression for high-similarity fake posts (MCSF), learned prototype/reference banks and confidence gates (COOLANT++), balanced and hard-misalignment evaluation (VERITE), controlled image swaps and semantic-edit directions (MEEL), and large controlled ablation / benchmark-shift studies. A targeted search also located positive-minus-negative similarity-margin filtering for retrieved text triplets, inter-sample contrastive learning, and evidence-grounded artifact-controlled verification. See [the evidence-qualified comparison](literature_comparison.md), including access limits.

The key scientific premise is therefore not new: semantic correspondence can support false claims, and semantic mismatch is not identical to falsehood. A classifier gain from a CLIP score can also reflect dataset structure, pretrained knowledge, or modality interactions rather than robust pairwise evidence.

## Narrow question this implementation tests

For this already locked BERT + ResNet50 Fakeddit experiment, does adding two explicit reference-relative scalar margins improve Macro-F1 over the existing raw CLIP cosine when both features are constructed from the same pinned CLIP encoder and a deterministic, source-family-aware, training-only bank? The two margins compare a query text with valid bank images and a query image with valid bank texts, then subtract the mean of the top-*k* negative similarities from the original-pair cosine.

The implementation adds a specific *measurement and ablation protocol*: fixed-example nearest-reference margins, direction-specific matching, exclusions for the same sample/family/text/pair/image hashes, split-pure references, and comparisons against no-CLIP, raw-only, margin-only, randomized-input, and CLIP-unimodal-projection controls. Bidirectional similarity matrices, momentum queues of cross-modal embeddings, and positive-negative retrieval margins are already present in related work. This is a candidate incremental contribution, not a verified novelty claim. We have not established that no previous paper uses a mathematically equivalent reference-relative residual.

## What the results can establish

On the existing 2,500-pair test subset, which the prior project already inspected, the analysis can estimate exploratory predictive differences for that split and these checkpoints. A positive D-versus-B difference would show incremental predictive value under this protocol; it would not prove a causal mechanism, factual verification, source independence, robustness to real-world shift, or novelty. A negative or inconclusive difference is a valid result.

The random-feature and CLIP-projection controls help separate model-head capacity and generic extra inputs from this feature construction, but they cannot rule out all training-distribution shortcuts. Synthetic image swaps test pairwise alignment ranking and prediction sensitivity only. The swapped pairs do not receive fabricated fake/real truth labels.

## Outstanding validity and novelty work

* The existing 2,500-row test subset is exploratory because it has already informed previous project analysis. Its bootstrap/McNemar results are conditional on a single public sample, and event/source dependence may make a row bootstrap optimistic.
* A second, fresh 1,000-row same-dataset evaluation slice (500 fake, 500 real) was obtained from unused Fakeddit public-test records. It is disjoint from the retained rows by ID, known linked-source family, normalized text, image URL, image-byte hash and decoded-pixel hash. This checks repeatability on new records, but it is not cross-dataset validation or an independent collection; labels and data source remain Fakeddit, and image availability can bias retrieval.
* Independent evaluation on VERITE or MMFakeBench was not run: neither benchmark is present locally, their label structures differ from Fakeddit's, and no validated label crosswalk/data provenance was available for a safe drop-in evaluation.
* MCSF, CMCM, COOLANT++, and ViLID need deeper full-text review before a paper submission. ViLID was only bibliographically identified in this pass.
* The proposed feature bank is a fixed collection of training examples, not an external knowledge base. The study cannot verify claims against world facts.
* A confirmatory claim requires a new, untouched held-out set or a separately defined compatible benchmark, preregistered analysis, independent source/event controls, and a broader literature search.

## Claim wording to retain

Safe: “We evaluated bidirectional top-*k* training-reference margins as additional frozen CLIP features in a controlled Fakeddit extension.”

Unsafe without further evidence: “first,” “novel,” “state of the art,” “detects factual truth,” or “generalizes to real-world misinformation.”
