# Results — ALIGNMENT ONLY

**Executed evidence: official precomputed CLIP correspondence on 1,000 groups / 2,000 rows, 921 observed content-linked components. No real selected images, pinned encoder results, BRAS comparison or frozen Fakeddit transfer results are claimed.** All selected groups had both required embeddings; zero groups were replaced or dropped for score availability. Strategy allocation: 250 groups each.

| Endpoint | Estimate [95% component-bootstrap CI] |
|---|---:|
| Paired ranking, half-credit ties | 0.5900 [0.5599, 0.6201] |
| ROC-AUC, pristine positive | 0.7170 [0.6952, 0.7382] |
| Average precision, pristine positive | 0.7105 [0.6864, 0.7348] |
| Mean pristine-minus-mismatch cosine | 0.0323 [0.0288, 0.0358] |

| Construction strategy | Groups | Linked components | Ranking success (95% CI) | ROC-AUC | AP, pristine positive |
|---|---:|---:|---|---:|---:|
| person_sbert_text_text | 250 | 217 | 0.4680 [0.4104, 0.5271] | 0.6528 | 0.6321 |
| scene_resnet_place | 250 | 241 | 0.5680 [0.5061, 0.6286] | 0.7877 | 0.7707 |
| semantics_clip_text_image | 250 | 247 | 0.4600 [0.3968, 0.5240] | 0.4453 | 0.5204 |
| semantics_clip_text_text | 250 | 233 | 0.8640 [0.8171, 0.9044] | 0.9098 | 0.9153 |

Exact ties: 0. Strict ranking: 0.5900; 1e-6 tolerance ranking: 0.5900. Component bootstrap and the prespecified robustness summaries appear in `official/statistics.json`. Holm-adjusted strategy tests are reported there without selective omission. Saved predictions, pair-level outcomes and plots are linked by hashes. These are pairing outcomes, not fake-news accuracy.

Verification: 24 unit tests passed before scoring; the final suite has 29 tests after adding download-resume, real-image integrity, feature-provenance and transfer-metric-label regressions. Coverage includes image refusal, source/pair semantics, cluster/tie behavior, independent metrics, immutable cache, malicious-pickle rejection and checkpoint-preservation detection. A bootstrap zero-weight edge case was found and corrected during synthetic tests **before** any external scores; test failures caused by the sandbox temporary directory were resolved by a bounded project-local test directory. Final integration/recomputation and preservation receipts are separate machine-readable artifacts (`verification_final.json`, `preservation_check.json`). A post-score code review found a metric-name ordering bug in the unexecuted transfer branch. It was fixed with an archived-source amendment and a synthetic fixture using the frozen heads; official predictions/statistics/protocol remained byte-identical. See `amendments/transfer_report/amendment.json`. Synthetic fixture results are software tests, not external evaluation evidence. Pre-change inventory: 59,067 files, including CCR. Preservation passed with no changed/missing files.

Conditional conclusion: this is a reproducible external correspondence baseline and a documented partial-data result. The primary cosine-versus-BRAS hypothesis remains untested. No new-method superiority or external factual-veracity performance is supported.

NewsCLIPpings pairing labels describe intact captions and intact images that were originally paired or automatically mismatched. They do not establish whether a news claim is true. CLIP influenced construction and adversarial filtering, so this is a distinct dataset source from Fakeddit but is not independent of CLIP method selection. Sources: [paper](https://aclanthology.org/2021.emnlp-main.545/), [official instructions](https://github.com/g-luo/news_clippings), [author release](https://huggingface.co/g-luo/news-clippings/tree/main).
