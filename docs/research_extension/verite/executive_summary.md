# Executed VERITE evaluation: conclusions and evidence

**All feasible stages completed using the authorized local images.** The current official release contains 1,001 complete pairs: 338 true, 338 miscaptioned and 325 out-of-context. Every referenced image decoded; no image was substituted. The archive SHA matches the official Hugging Face LFS release, and annotations match the pinned official GitHub revision. The older 1,000-row description was not imposed on the current data. There is no remaining local VERITE data-access blocker.

**Does the original CLIP improvement transfer? Partially.** Across the three existing seeds, mean Macro-F1 increases from 0.51485 (A) to 0.54123 (B), and truthful-class ROC-AUC increases from 0.52098 to 0.59839. All three B−A Macro-F1 point differences are positive, but the prespecified primary seed-42 result is unresolved: ΔMacro-F1=+0.02152, 95% CI [-0.00687, +0.05098], Holm p=0.20098; corrected 154, introduced 188 errors. Seed 2026 has a positive secondary Macro-F1 result after its family's correction. Mean accuracy falls from 0.56044 to 0.54945; B's mean log loss and Brier score also worsen. Thus the original broad accuracy improvement does not consistently transfer, despite improved discrimination and descriptive Macro-F1.

**Does BRAS offer an independent benefit? Yes, narrowly in correspondence ranking.** For true versus miscaptioned pairs, raw cosine ranks the true instance higher in 191/338 cases (56.51%), versus 208/338 (61.54%) for the frozen BRAS mean. The paired gain is 5.03 percentage points, 95% component-bootstrap CI [2.06, 8.28], Holm p=0.00820; 24 rankings improve and 7 deteriorate. This is independently executed, external alignment evidence, not evidence of factual verification. The corresponding out-of-context gain is only 0.62 points, CI [-0.92, 2.15], Holm p=0.68003. The frozen BRAS D classifier has no resolved benefit over B: ΔMacro-F1=-0.00919, 95% CI [-0.02958, +0.01107], Holm p=0.38226; corrected 55, introduced 67 errors. C is seed-sensitive, with a positive secondary result for seed 123 and a negative point difference for seed 42. CCR equals B exactly for every seed.

Descriptive means across the same three trained seeds; these are not three independent evaluation datasets:

| model | accuracy_mean | macro_f1_mean | roc_auc_truthful_mean | average_precision_truthful_mean |
|---|---|---|---|---|
| A | 0.56044 | 0.51485 | 0.52098 | 0.35992 |
| B | 0.54945 | 0.54123 | 0.59839 | 0.42994 |
| C | 0.55744 | 0.54444 | 0.60932 | 0.43690 |
| CCR | 0.54945 | 0.54123 | 0.59839 | 0.42994 |
| D | 0.54612 | 0.53832 | 0.61658 | 0.44103 |

All classifier accuracies are below the always-misleading prevalence baseline (0.66234); their Macro-F1 values exceed its 0.39844. B recognizes more true examples but also calls more misleading examples truthful. This tradeoff explains why discrimination, Macro-F1 and accuracy need different interpretations.

The mapping `true → truthful; miscaptioned/out-of-context → misleading pair` defines a **cross-dataset, cross-task transfer diagnostic**. It does not turn VERITE into an equivalent Fakeddit fake-news benchmark. All results use actual pinned ViT-B/32 embeddings, original BERT/ResNet preprocessing, frozen heads/scalers, the original training-only BRAS bank and the existing CCR policy. No retraining, external calibration or evaluation-label model selection occurred. Dependence-aware analysis uses 336 shared-content components, 5,000 paired bootstrap draws and 10,000 component-level randomizations, with prespecified multiplicity correction.

**Publication decision: CONDITIONAL GO for manuscript development; NO-GO for a distinctive superior-method submission as-is.** The new evidence supports a focused empirical claim about reference normalization on miscaptioned-pair ranking and documents limits of fixed-head transfer. It does not establish a novel scoring rule, a robust BRAS classifier improvement or general factual reasoning. The contribution still needs a precise contrast to existing VERITE/MUSE findings and comparable published baselines or broader external evidence. See [publication assessment](publication_assessment.md) and [cross-dataset discussion](cross_dataset_discussion.md).

Verification passed: 24 unit tests, 90 independently recomputed classifier metrics, 768 actual frozen-head probabilities replayed through independent NumPy calculations, and 120 expanded-resample Macro-F1 checks. All 59,154 inventoried pre-existing files are unchanged. Images, full caption manifests and feature embeddings remain restricted locally; the deliverable package contains no such data.

Read [integrity/provenance](dataset_integrity_and_provenance.md), [all model seeds](model_transfer_results.md), [ranking statistics](correspondence_ranking.md), [failure analysis](failure_analysis.md), [robustness and verification](robustness_and_verification.md), and [reproduction](execution_and_reproduction.md).
