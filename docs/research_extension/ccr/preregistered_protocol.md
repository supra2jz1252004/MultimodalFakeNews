# CCR preregistered implementation protocol

This is a locally timestamped, pre-training specification, not a public registry record. The exact machine-readable protocol is `models/research_extension/ccr/protocol.json`; its SHA-256 is **565f01665a1e7a4014d6a0ceac79ff32d4479ddf3a9c0e237390189588d57722**.

## Question and interpretation

Can train-only, source-family-aware OOF incremental log-loss utility predict when to use the frozen CLIP-score head B instead of baseline A? Primary independent endpoint: VERITE seed-42 log-loss difference CCR minus B (negative favors CCR). Secondary outcomes include Brier, calibration, Macro-F1, accuracy, and coverage/risk. Both existing Fakeddit test sets are exploratory. No test-guided tuning.

## Locked choices

Five deterministic label-blind group-hash folds over the 50,000 training rows, three matched classifier seeds, and internal 10% group-hash stopping sets. Every fold's normalization and gradient updates exclude its held-out rows; stopping uses only its internal split. Existing frozen encoder caches are read-only. Fold-specific normalizers, IDs, checkpoints, histories, and predictions are saved with hashes.

The utility is clipped CE(A,y)-CE(B,y). Inputs are only A/B probabilities, log-odds/difference, margins, entropies and raw cosine. A priori estimators: intercept mean, standardized ridge alpha100 (primary), and depth3 tree with minimum500 leaves. No architecture search. Optional unimodal scores are omitted to avoid in-sample auxiliary leakage.

For each seed and estimator choose a hard threshold by minimum validation log-loss over 21 OOF-score quantiles, zero, and all-A/all-B endpoints; ties favor B. If the selected ridge candidate does not have a negative upper paired validation log-loss CI, operational CCR falls back to B. The unguarded candidate remains a separately reported secondary model. The validation guard is selection-biased, not inferential proof.

Baselines: A/B, separately temperature-calibrated A/B, intercept, tree, raw cosine, confidence difference, entropy difference, and deterministic random gate matched to ridge validation routing. Temperature bounds0.05–20. All thresholds and temperatures frozen from validation before test inference. Existing A/B checkpoints/configuration were already validation-selected; reuse must be disclosed.

Primary seed42;123/2026 sensitivity.5,000 paired bootstrap replicates, seed20261106, row and known component clusters. Separate bootstrap/classifier seed keys. Exact McNemar for correctness. Binary Brier means (p_real-y)^2; confidence ECE10 equal-width bins with5/15/20 sensitivity. Secondary intervals descriptive, no multiple-comparison winner claim. Risk curves use validation-fixed coverage cutoffs; actual test coverage reported.

## External access and shift

Official VERITE truthfulness maps true→1, miscaptioned/out-of-context→0. This is cross-task/domain transfer, not a same-task Fakeddit comparison. Full official images required for the primary endpoint; incomplete public-URL retrieval can only be descriptive. Do not bypass institutional access, redistribute images, or use provided incompatible CLIP-L/14 features in place of BERT/ResNet/CLIP-B/32. No training, thresholds, or selection from external labels. If access is unavailable, record no independent result.

## Diagnostics and limitations

Train-quartile cosine prevalence, six-way and subreddit groups with counts/intervals; labels and group metadata never enter gate inputs. Reuse label-free original-versus-random/hard swap comparisons. Family groups are linked submissions, not verified event clusters; broader source independence is unproven. Preserve BRAS's negative result and correct its prior-art characterization: its two directional margins sum algebraically to a CSLS-form score for the same neighborhoods.

Full JSON below is authoritative.

```json
{
  "schema": 1,
  "name": "CCR frozen-head utility selection",
  "locked_utc": "2026-11-06T18:00:34.748535+00:00",
  "status": "Locked before CCR training and before external row-label/score inspection; local timestamp, not public registration.",
  "manifest_sha256": "60aeb7e8a04a0296d92e6d5670de1396ae35037a84331c428ab25bf753450edb",
  "historical_inventory_sha256": "9883fe3a8a11ea5dc7996aec9ed1a686a2b8b75b399cc29bd302f68973f703cd",
  "seeds": [
    42,
    123,
    2026
  ],
  "primary_seed": 42,
  "folds": 5,
  "fold_seed": 20261106,
  "fold_rule": "SHA256(seed, group_id) modulo 5; labels not used in assignment. Only official train rows.",
  "inner_validation": "Other-fold groups with independent SHA256(inner-fold seed, group_id) uniform <0.10. Held-out labels not used. Fit normalizers on inner-fit rows only; early stopping on inner-validation only; do not refit after stopping.",
  "head_config": {
    "dimension": 128,
    "dropout": 0.3,
    "lr": 0.001,
    "weight_decay": 0.0001
  },
  "head_budget": {
    "batch_size": 128,
    "epochs": 40,
    "patience": 7,
    "optimizer": "AdamW",
    "criterion": "validation macro_f1; validation loss tie-break",
    "threads": 6
  },
  "cpu_threads": 6,
  "gate_inputs": [
    "p_A_real",
    "p_B_real",
    "logodds_A",
    "logodds_B",
    "logodds_B_minus_A",
    "margin_A",
    "margin_B",
    "entropy_A",
    "entropy_B",
    "cosine"
  ],
  "forbidden_inputs": [
    "label",
    "label6",
    "subreddit",
    "sample_id",
    "group_id",
    "text",
    "source identifiers",
    "test outcomes"
  ],
  "unimodal_scores": "Omitted optional features: historical text/image heads are in-sample for training rows; using them would require extra OOF fits.",
  "utility": "CE(A,y)-CE(B,y); binary probabilities clipped to [1e-6,1-1e-6]",
  "utility_models": {
    "intercept": "OOF utility mean",
    "ridge": "StandardScaler fit to OOF train inputs; Ridge alpha=100 fixed a priori",
    "tree": "DecisionTreeRegressor(max_depth=3,min_samples_leaf=500,random_state=seed), squared-error; fixed a priori"
  },
  "primary_model": "CCR = ridge candidate with validation guard; if guard fails select always B. Intercept/tree are fixed secondary ablations; no model-family selection. No soft mixture.",
  "threshold": "For each seed/model, min validation log-loss among -1e30 (all B), +1e30 (all A), zero, and 21 train-OOF score quantiles. B when score > threshold. Ties prefer largest B coverage then earliest grid value.",
  "guard": "For ridge candidate only: paired validation bootstrap CI95 for log-loss(candidate)-log-loss(B) must have upper <0. Otherwise CCR falls back to B. Selection-biased validation CI is a conservative operational heuristic, NOT independent statistical proof.",
  "baselines": [
    "A",
    "B",
    "A_temperature",
    "B_temperature",
    "intercept",
    "ridge_candidate",
    "tree",
    "cosine_gate",
    "confidence_gate",
    "entropy_gate",
    "random_gate",
    "CCR"
  ],
  "heuristics": "cosine score; max-probability(B)-max-probability(A); entropy(A)-entropy(B). Threshold rule above, OOF quantiles. Random gate uses deterministic SHA256(random seed, sample_id) < validation ridge B-coverage; ID only random-control RNG, never gate predictors.",
  "temperature": "Separate per seed A/B positive temperature, bounded [0.05,20], minimize log-loss on existing validation. Only calibrated baseline outputs; not used to redefine utility.",
  "validation_dependence": "Historical A/B architecture and checkpoints already selected on same 2500 validation. Gate thresholds and temperatures are also fit here. Validation results are dependent and optimistic.",
  "primary_endpoint": "VERITE seed42 paired mean log-loss(CCR)-log-loss(B), lower better; two-sided percentile 95% CI, article/image component cluster bootstrap primary and row bootstrap sensitivity.",
  "secondary_endpoints": [
    "Brier delta (binary real-class squared error)",
    "Macro-F1 delta",
    "accuracy delta",
    "ECE confidence bins 10, sensitivity 5/15/20",
    "risk at validation-fixed coverage cutoffs",
    "paired corrected/new errors and exact McNemar"
  ],
  "bootstrap": {
    "repetitions": 5000,
    "seed": 20261106,
    "alpha": 0.05,
    "batch": 100,
    "mcnemar": "two-sided exact binomial conditional on discordant pairs; classifier_seed and bootstrap_seed separate fields",
    "multiplicity": "No secondary superiority claims; all intervals descriptive, no selecting winner from test. Seeds reported separately, never pooled as independent rows."
  },
  "exploratory_sets": [
    "original_test_2500",
    "previously_viewed_fresh_test_1000"
  ],
  "subgroups": "Train-only cosine quartiles; label6; each subreddit with count and Wilson accuracy interval; paired gain/new-error counts. Subreddit is distant supervision, not an independent source validation.",
  "selective": "Confidence max(p,1-p); retain confidence >= validation quantile(1-c) for c in [.5,.6,.7,.8,.9,.95,1.0], c=1 cutoff0. Report achieved coverage and 0-1 risk; ties can increase coverage.",
  "stress": "Reuse prior saved 500-query random/hard train-donor swaps. Recompute A/B/CCR probabilities and utility; report label-free flips and cosine/utility shifts. No factual labels for swaps.",
  "external": {
    "sources": [
      "https://github.com/stevejpapad/image-text-verification",
      "https://zenodo.org/records/10474450"
    ],
    "label_map": {
      "true": 1,
      "miscaptioned": 0,
      "out-of-context": 0
    },
    "task_caveat": "Cross-task/domain transfer from Fakeddit distant supervision to image-caption truthfulness; accuracy not directly comparable.",
    "access": "Use official research access only. No bypass or redistribution of gated images. If institutional verification required and authorized files unavailable, record blocker and leave independent endpoint unevaluated. Public metadata may be acquired after protocol lock; public original URL route only if access conditions permit.",
    "population": "Official 1000 rows; preserve official IDs and labels; no label-dependent selection. Full official images required for primary endpoint. Any incomplete URL subset descriptive only; no substitute data.",
    "clusters": "Connected components sharing official article id or image path/hash; inspect before scores; report cluster counts."
  },
  "smoke": {
    "max_train": 400,
    "folds": 2,
    "epochs": 2,
    "patience": 2,
    "seed": 42,
    "tree_min_leaf": 20,
    "outputs": "Separate smoke directory; no scientific test claims"
  },
  "frozen_policy": "Freeze all thresholds, temperatures, gate models and source digests before any test inference. Bug fixes allowed only with dated amendment, no outcome-driven tuning; preserve previous artifacts."
}
```
