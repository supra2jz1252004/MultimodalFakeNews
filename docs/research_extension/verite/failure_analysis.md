# Error breakdown and failure examples

`category_breakdown.csv` reports all three categories for every model and seed; `source_breakdown.csv` retains Snopes/Reuters counts and point metrics. The category figure shows descriptive means over the same three heads, not independent repetitions.

Failure selection was fixed before test metrics: ten highest-cosine misleading records and ten smallest cosine paired margins per comparison category, with deterministic row-ID tie breaks. `failure_cases_scores.csv` contains 28 unique selected records with IDs, categories, seed 42 predictions and scores. Full paired captions, article URLs and local image paths are retained only in `data/external/verite/restricted_cache/failure_cases_private.csv` for authorized review. No image grids or raw restricted records are distributed.

These cases expose high semantic similarity on benchmark-labeled misleading pairs and disagreements between frozen decisions and correspondence ranking. Such scores cannot resolve a date, location, identity or event claim. Case selection is exploratory; image content is not described without a documented visual review, and official annotations are not relabeled. Repeated caption/image control holds one modality fixed, allowing a cleaner sensitivity comparison but still not a causal proof of factual reasoning.

Three examples from the prespecified high-similarity list are summarized below. These are **annotation/caption contrasts**, not a manual visual assessment or independent fact check. No restricted image or complete caption is reproduced. Probabilities are the frozen seed-42 heads' Fakeddit-real output, mapped diagnostically to truthful.

| Record | Official paired-caption contrast, paraphrased | Misleading-pair cosine | A p(truthful) | B p(truthful) | D p(truthful) |
|---|---|---|---|---|---|
| verite_0283 | Sanitation-agency attribution versus street-artist attribution for New York posters | 0.43049 | 0.89097 | 0.92479 | 0.91158 |
| verite_0358 | A real polar bear on transport in Russia versus a puppet at a Greenpeace event in Brno | 0.42282 | 0.17479 | 0.87621 | 0.89186 |
| verite_0427 | A Hurricane Florence staging allegation versus a Hurricane Ike water-depth account | 0.41358 | 0.20258 | 0.86013 | 0.90814 |

All three misleading records score above their paired true counterparts under both raw cosine and BRAS mean. The first is wrong for all three heads. For the second and third, A correctly predicts misleading while B and D introduce errors. The score changes therefore fail to resolve the specific attribution, object identity or event/time distinction represented by the official annotations.
