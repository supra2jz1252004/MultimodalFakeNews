# Cross-dataset discussion

| dataset | model | n | accuracy_mean | macro_f1_mean | macro_f1_seed_sd |
|---|---|---|---|---|---|
| Fakeddit historical 2500 | A | 2500 | 0.85547 | 0.84736 | 0.00556 |
| Fakeddit historical 2500 | B | 2500 | 0.87867 | 0.87212 | 0.00943 |
| Fakeddit historical 2500 | C | 2500 | 0.87693 | 0.87004 | 0.00459 |
| Fakeddit historical 2500 | D | 2500 | 0.87907 | 0.87289 | 0.00321 |
| Fakeddit historical 2500 | CCR | 2500 | 0.87867 | 0.87212 | 0.00943 |
| Fakeddit previous fresh 1000 | A | 1000 | 0.83300 | 0.83254 | 0.00981 |
| Fakeddit previous fresh 1000 | B | 1000 | 0.85967 | 0.85941 | 0.01431 |
| Fakeddit previous fresh 1000 | C | 1000 | 0.85567 | 0.85537 | 0.00423 |
| Fakeddit previous fresh 1000 | D | 1000 | 0.85600 | 0.85580 | 0.00438 |
| Fakeddit previous fresh 1000 | CCR | 1000 | 0.85967 | 0.85941 | 0.01431 |
| VERITE current 1001 | A | 1001 | 0.56044 | 0.51485 | 0.00416 |
| VERITE current 1001 | B | 1001 | 0.54945 | 0.54123 | 0.01211 |
| VERITE current 1001 | C | 1001 | 0.55744 | 0.54444 | 0.03130 |
| VERITE current 1001 | CCR | 1001 | 0.54945 | 0.54123 | 0.01211 |
| VERITE current 1001 | D | 1001 | 0.54612 | 0.53832 | 0.01835 |

These rows provide context; label definitions, prevalence and sample selection differ, so a cross-panel drop is not a calibrated estimate of a common-task domain effect. The original 2,500 Fakeddit and previously viewed 1,000 slice remain exploratory for BRAS/CCR. Historical BRAS D did not reliably improve B; CCR selected B for all seeds. The new VERITE test is external to Fakeddit and was not used for training or selection, with exact retained-data overlap checks negative.

NewsCLIPpings remains **ALIGNMENT ONLY**: author-provided CLIP embeddings on 1,000 paired groups, 250 per construction strategy, ranking 0.5900 (95% CI 0.5599–0.6201), ROC-AUC 0.716961, AP 0.710531. Its pinned encoder/BRAS comparison and frozen-head transfer were not run because selected image bytes were unavailable. Its CLIP-assisted construction and unverified equality to the pinned feature contract prevent pooling its scores with VERITE. This new authorized VERITE run does not retroactively upgrade the NewsCLIPpings evidence.

For the original CLIP feature claim, the correct external comparison is frozen B−A within VERITE, not a comparison to the authors' different CLIP model: ΔMacro-F1=+0.02152, 95% CI [-0.00687, +0.05098], Holm p=0.20098; corrected 154, introduced 188 errors. For the trained BRAS extension, use D−B: ΔMacro-F1=-0.00919, 95% CI [-0.02958, +0.01107], Holm p=0.38226; corrected 55, introduced 67 errors. Report each seed alongside these primary tests. Raw-score ranking benefits, if any, do not automatically establish improved binary transfer.
