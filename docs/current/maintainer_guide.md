# Maintaining the project

Start by reading the current methodology and results guides, then follow the relevant entry point. Many research files are deliberately immutable: their hashes connect results to the exact source, features, checkpoints and protocol used.

## Code map

| Entry point or module | Responsibility |
|---|---|
| `run_experiments.py` | Original Fakeddit stage commands and one-pair inference |
| `src/data.py`, `src/feature_extractors.py` | Acquisition, manifests, pinned encoders and feature contracts |
| `src/models.py`, `src/train.py` | Original matched heads and validation-only training |
| `src/evaluate.py`, `src/similarity.py` | Original metrics and training-only similarity analysis |
| `src/predict.py`, `demo.py` | Saved A/B inference and a local Tk interface |
| `run_research_extension.py`, `src/research_extension/` | BRAS references, feature variants and historical evaluation |
| `src/research_extension/ccr/` | Cross-fitted utility policy, guard and fixed routing |
| `run_external_eval.py`, `src/external_eval/newsclippings/` | Separate NewsCLIPpings correspondence workflow |
| `run_verite_eval.py`, `src/research_extension/verite/` | Authorized image audit, frozen transfer and paired statistics |
| `results/`, `models/`, `data/` | Saved evidence, weights and inputs; preserve these directories |

## Names and data contracts

Use the A/B/C/D mapping in [methodology](methodology.md). Do not rename `baseline`, `proposed`, `m_text`, `m_image`, state-dict keys or CSV columns for style: other modules and saved files depend on them. A descriptive local variable is useful only if it does not change an interface or invalidate a locked source contract. Explain a cryptic frozen name in documentation instead.

`sample_id` identifies a Fakeddit record; `group_id` records a known source family. VERITE uses `row_id`, `family_id` and `cluster_id`; a linked component can connect multiple families. `classifier_seed` controls a fitted head; a bootstrap seed controls uncertainty resampling. Neither changes the dataset's number of independent observations.

## Working safely

Before editing, back up each affected file and record its hash, or create a branch in a version-controlled copy. This project currently has no Git repository. The editorial backup is in `editorial_review_20261112/backup/`. It preserves source/docs and the source manuscript; it does not duplicate restricted datasets.

Several source files are part of CCR/VERITE frozen contracts. Even formatting changes can invalidate their hash checks. Do not update a lock merely to make verification pass. Future algorithmic or protocol work needs a separate study and explicit amendment; this editorial pass leaves the frozen modules unchanged.

The current runtime observed during editorial checks is Python 3.13.17 with CPU PyTorch. The README preserves the original run's environment separately. Do not silently upgrade pinned dependencies to resolve a presentation issue.

For command discovery, run `python -B run_experiments.py --help` or the corresponding external entry point with `--help`. The demo is `python -B demo.py`; use an image you are authorized to show. It may need cached pretrained weights. Softmax scores are model outputs, not truth probabilities, and the application cannot prove that a supplied pair was unseen during training.

Do not use `all`, `train`, `search`, `extract-features`, `evaluate` or report-generation commands just to inspect completed work. Some loaders also refresh an index. Read saved reports first. Existing unit tests use synthetic fixtures and some read-only checkpoint replays; they are software checks, not new benchmark results. The editorial runner redirects scratch directories outside historical results.

## Before sharing

Keep VERITE images, complete captions, private manifests, per-record predictions and embeddings local under their access conditions. Review a code-only release separately from restricted data. No credentials belong in documentation or version control. The private repository upload remains pending; do not treat the proposed availability declaration as fact until verified.

Retain AI-use attribution and author-review flags. The [journal instructions](https://link.springer.com/journal/42979/submission-guidelines) distinguish copyediting from substantive assistance; the [publisher policy](https://www.springernature.com/gp/policies/editorial-policies/ai-manuscript-preparation) also calls for transparency and author accountability. These pages were checked on 12 November 2026. The relevant VIT-AP course/university rules have not been independently established here; the team should obtain its supervisor's requirements rather than assume permission or exemption. Natural wording does not remove disclosure obligations.
