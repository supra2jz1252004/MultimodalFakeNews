# Results and discussion

The original CLIP feature helped on the retained Fakeddit subset. The external evaluation gives a more limited result: correspondence ranking, global discrimination and fixed classifier decisions do not improve together.

## Original Fakeddit finding

Across seeds 42, 123 and 2026, A had mean Macro-F1 84.74% and B 87.21%; mean accuracy was 85.55% and 87.87%. The primary-seed paired Macro-F1 interval was positive. This supports the added scalar under the controlled, availability-dependent Fakeddit setup. It does not show that semantic similarity establishes factual truth. Full precision, recall, seed variation and statistics remain in [the original results](../results_summary.md).

## BRAS and CCR classifiers

BRAS D did not establish a reliable improvement over B. Its mean Macro-F1 was almost unchanged on the historical test and lower on the previously viewed fresh slice. CCR's saved validation guard rejected active switching, so its final predictions were exactly B's. These findings answer the tested questions; they should not be hidden or generalized to every possible normalization or routing method. See the unchanged [BRAS results](../research_extension/results_summary.md) and [CCR results](../research_extension/ccr/results_summary.md).

## VERITE correspondence scores

The authorized evaluation contains 1,001 complete records. In the original primary comparison, symmetric BRAS improved true-versus-miscaptioned ranking from 56.51% to 61.54%: +5.03 percentage points, 95% CI [2.06, 8.28], Holm p=0.00820. The out-of-context difference was unresolved. The symmetric score is half-CSLS under the same references, so this benefit is evidence about an existing normalization form.

The larger full text-conditioned result was secondary. Later exploratory comparisons reused the observations and used a different correction family. Their p-values cannot replace the primary result or be presented as replication. No adjusted score established a global AUC/AP improvement after the exploratory correction. Within-pair reference offsets can cancel while offsets between families change, which explains why local ranking and global discrimination can disagree.

## VERITE frozen transfer

A-to-B mean Macro-F1 rose from 0.51485 to 0.54123 and mean truthful AUC from 0.52098 to 0.59839. Mean accuracy fell from 0.56044 to 0.54945. The primary seed-42 Macro-F1 interval included zero (Holm p=0.20098). B recognized more true pairs but also classified more misleading pairs as truthful. This is a descriptive improvement in some metrics, not confirmation of the primary transfer hypothesis.

BRAS D had no resolved Macro-F1 benefit over B; C varied across seeds. CCR again equaled B. All classifier accuracies were below the always-misleading baseline, although their Macro-F1 values exceeded it. Report both facts: accuracy alone can favor a majority prediction, while Macro-F1 gives each class equal weight. See [the complete VERITE summary](../research_extension/verite/executive_summary.md) and the manuscript's tables for every seed and negative result.

## NewsCLIPpings and the scope of evidence

NewsCLIPpings used official precomputed embeddings for 1,000 selected groups. Selected real image bytes were unavailable. Its ranking rate of 0.5900 is an alignment baseline, not pinned-encoder BRAS evidence or frozen Fakeddit transfer. Automatically mismatched pairs do not independently establish that a news claim is false. Preserve that distinction when presenting [the results](../external_eval/newsclippings/results.md).

The defensible contribution is a controlled empirical comparison with transparent limits. The evidence does not support a new normalization algorithm, general factual verification or superiority over complete published detectors.
