# Frozen model transfer results

**B versus A, primary seed 42:** no statistically resolved transfer difference; ΔMacro-F1=+0.02152, 95% CI [-0.00687, +0.05098], Holm p=0.20098; corrected 154, introduced 188 errors.

**BRAS D versus B, primary seed 42:** no statistically resolved transfer difference; ΔMacro-F1=-0.00919, 95% CI [-0.02958, +0.01107], Holm p=0.38226; corrected 55, introduced 67 errors.

| seed | model | n | accuracy | macro_f1 | roc_auc_truthful | average_precision_truthful |
|---|---|---|---|---|---|---|
| 42 | A | 1001 | 0.57243 | 0.51326 | 0.52264 | 0.35366 |
| 42 | B | 1001 | 0.53846 | 0.53478 | 0.59839 | 0.43108 |
| 42 | C | 1001 | 0.50849 | 0.50849 | 0.61689 | 0.44424 |
| 42 | D | 1001 | 0.52647 | 0.52560 | 0.61572 | 0.44039 |
| 42 | CCR | 1001 | 0.53846 | 0.53478 | 0.59839 | 0.43108 |
| 123 | A | 1001 | 0.54645 | 0.51171 | 0.52298 | 0.37166 |
| 123 | B | 1001 | 0.53746 | 0.53369 | 0.60384 | 0.43492 |
| 123 | C | 1001 | 0.59041 | 0.56565 | 0.60282 | 0.43907 |
| 123 | D | 1001 | 0.53047 | 0.53002 | 0.62676 | 0.45391 |
| 123 | CCR | 1001 | 0.53746 | 0.53369 | 0.60384 | 0.43492 |
| 2026 | A | 1001 | 0.56244 | 0.51957 | 0.51733 | 0.35444 |
| 2026 | B | 1001 | 0.57243 | 0.55520 | 0.59294 | 0.42382 |
| 2026 | C | 1001 | 0.57343 | 0.55917 | 0.60824 | 0.42737 |
| 2026 | D | 1001 | 0.58142 | 0.55935 | 0.60725 | 0.42878 |
| 2026 | CCR | 1001 | 0.57243 | 0.55520 | 0.59294 | 0.42382 |

Every model/seed has component-bootstrap intervals in `model_metrics.csv`. Primary and secondary effect intervals, corrected p-values, probability/decision changes and corrected/introduced error counts are in `paired_classification.json` and the per-row comparison CSVs. Negative findings are retained. CCR equals B exactly for all three seeds; this follows from the frozen guard and does not validate a new selector.

Descriptive seed means and sample SD:

| model | accuracy_mean | accuracy_std | macro_f1_mean | macro_f1_std |
|---|---|---|---|---|
| A | 0.56044 | 0.01310 | 0.51485 | 0.00416 |
| B | 0.54945 | 0.01990 | 0.54123 | 0.01211 |
| C | 0.55744 | 0.04323 | 0.54444 | 0.03130 |
| CCR | 0.54945 | 0.01990 | 0.54123 | 0.01211 |
| D | 0.54612 | 0.03063 | 0.53832 | 0.01835 |

The always-misleading baseline and family-only/overlap sensitivities are recorded in `robustness.json`. Class prevalence is 663 misleading versus 338 truthful, so accuracy alone can be misleading. No claim of matching the authors' trained models is made: training data, encoder backbones, task mapping and current annotation version differ.
