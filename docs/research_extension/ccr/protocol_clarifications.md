# Protocol clarifications (no experimental changes)

The machine-readable protocol was locked before any CCR training. It is authoritative and has not been modified.

The prose sentence “depth3 tree with minimum500 leaves” in the initial protocol is a wording error: `min_samples_leaf=500` means at least500 training rows per leaf, with maximum tree depth3. The JSON, code, methodology and all scientific fits consistently use that setting. The separate smoke run uses20 rows per leaf on its400-row training subset, exactly as the locked JSON specifies.

The smoke run used100 validation rows to exercise policy freezing. Its purpose is execution/provenance checking, not an estimator comparison. Its validation outcomes do not choose any full-run settings. All scientific thresholds use the full2,500-row validation set.

Figure code was made robust to a six-way category with zero rows before the full-run policy freeze. Missing categories display n=0 rather than causing plot failure. This does not change predictions, metrics, selection or confidence intervals.

The local external adapter is supplied for authorized images only and was not executed. Access checking is complete; independent evaluation remains blocked because the user confirmed no approved copy. Nothing in the exploratory Fakeddit results changes that status.
