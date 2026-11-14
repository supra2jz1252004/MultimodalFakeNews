# Reproducibility and research artifact availability

This folder accompanies the SN Computer Science manuscript on CLIP similarity normalization, paired correspondence ranking, global discrimination and frozen classifier transfer on VERITE.

## Publicly available

- [aggregate_results.json](aggregate_results.json): 16 aggregate tables and 101 aggregate rows transcribed, without numerical modification, from existing project documentation. Values are strings to preserve precision and units; source paths and sections are recorded.
- Implementation, evaluation code, requirements, model settings and tests are in the root and source directories.
- [Original Fakeddit documentation](../docs/results_summary.md) and [VERITE research documentation](../docs/research_extension/verite/) give experimental context, hypotheses and limitations.

The JSON is TABLE-DERIVED REPORTING MATERIAL, not original machine-produced result artifacts, rerun evidence, complete reproducibility records, or per-record predictions. Do not pool Fakeddit and VERITE metrics as one dataset.

## Currently not publicly deposited

- Original trained project-specific checkpoints (.pt) and train-only normalization assets. The historical Fakeddit paths are models/final/<kind>_<seed>.pt, where kind is text, image, baseline, proposed and seed is 42, 123 or 2026.
- Original saved JSON/CSV experiment outputs and verification receipts, for example model_metrics.csv, paired_classification.json, alignment_statistics.json, and checkpoint manifests.
- Restricted VERITE images, full captions, dataset manifests, row-level outputs, embeddings and other source data that should not be redistributed.

## To release missing artifacts correctly

Inventory the ORIGINAL files on the research machine and map their exact paths to the Fakeddit and VERITE frozen-head protocols. Audit each file for credentials, private text, identifiers, images, predictions, and third-party distribution permissions. Check SHA-256 and record provenance. Publish approved shareable trained weights and aggregate saved experiment outputs using a versioned GitHub Release or archival repository if size requires it, then link from this folder. Preserve original model hashes and output values. Never silently retrain or make placeholder weights or invented measurements. If originals cannot be located, disclose this rather than claiming availability.

Newly fetched Fakeddit images can differ from those in the original availability-dependent sample. The public code does not by itself reproduce the original fixed input manifest.

## Interpretation

The primary miscaptioned ranking comparison showed a gain for symmetric BRAS, a half-CSLS-equivalent score under matched reference definitions. Primary fixed-head classifier-transfer results were not conclusive; there was no consistent BRAS/CCR transfer benefit or established global discrimination improvement. Exploratory findings are not independently replicated confirmatory results.