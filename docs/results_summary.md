# Completed Fakeddit experiment

Finalized 2026-09-17 07:30 UTC from saved, executed results.

## Research answer

The proposed model improved mean Macro-F1, and the primary-seed paired bootstrap interval for Macro-F1 excludes zero in the positive direction. The mean proposed-minus-baseline changes across seeds are **+2.476 percentage points Macro-F1**, **+1.854 points fake-class F1**, and **+2.320 points accuracy**. These conclusions apply to this filtered Fakeddit subset and these frozen encoders.

## Data and acquisition

The project remains in the existing `MultimodalFakeNews` Desktop directory. Official resources: [Fakeddit repository](https://github.com/entitize/Fakeddit), its linked multimodal-only metadata, and [dataset paper](https://aclanthology.org/2020.lrec-1.755/). The files contain 682,661 metadata rows; 680,798 have valid multimodal references. The text is `clean_title`; `2_way_label` maps **0=fake, 1=real**. Binary real was checked against six-way label 0 on every row.

| Official split | Retained pairs | Fake (0) | Real (1) |
| --- | --- | --- | --- |
| train | 50,000 | 30,312 | 19,688 |
| validation | 2,500 | 1,518 | 982 |
| test | 2,500 | 1,509 | 991 |

Original partitions and candidate seed 20260916 were preserved. Validation/test IDs, text, labels and image hashes were checked unchanged between the initial and expanded experiments. Metadata reservation excluded 173,105 lower-priority rows sharing text/URLs/source families with higher official partitions; 1,863 rows lacked image references. These are separate from the 5,214 exclusions encountered while filling the final retained quotas:

| Candidate exclusion reason | Count |
| --- | --- |
| http_404 | 3,173 |
| duplicate_content_group | 1,191 |
| removed_image_placeholder | 419 |
| duplicate_text | 276 |
| duplicate_image_url | 72 |
| duplicate_image_content | 32 |
| uniform_blank_source_image | 17 |
| ConnectionError | 15 |
| transfer_over_120_seconds | 7 |
| http_403 | 4 |
| http_530 | 2 |
| ConnectTimeout | 2 |
| visually_verified_loading_placeholder | 1 |
| http_429 | 1 |
| image_decode_UnidentifiedImageError | 1 |
| http_400 | 1 |

Every exclusion is traceable in metadata_audit.csv.gz or selection_audit.csv. The retained subset has zero missing/corrupt files and zero shared normalized titles, original/stored-byte hashes, decoded-pixel hashes or linked-source groups under the implemented checks. All-retrieval overlaps are separately recorded in content_overlap.json. Acquisition uses official image URLs, with a small number of URL/ID/hash-verified local source-image copies. No old features or classifier results were reused.

## Encoders and classifiers

- BERT: `google-bert/bert-base-uncased`, revision `86b5e0934494bd15c9632b12f734a8a67f723594`; CLS output **768**.
- Image: torchvision **ResNet50 IMAGENET1K_V2**; final ImageNet head removed; pooled output **2,048**. Its transform resizes to 232 and center-crops to 224 using ImageNet normalization.
- Similarity: `openai/clip-vit-base-patch32`, revision `3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268`; paired **512-dimensional** projected text/image embeddings normalized and dotted to yield **one scalar**. Explicit cosine/diagonal assertions passed; CLIP's own image preprocessing and 77-token text limit were used.
- BERT token lengths on training data: median 11.0, p90 19.0, p95 23.0, p99 35.00, maximum 71. Chosen limit **64**; 3 training titles truncated. CLIP truncates 0 training titles.
- Encoders were frozen, in eval/inference mode, with float32 cached features. All features use train-only coordinate-wise z-scores (standard deviation floor 1e-6).
- C: 768→128 ReLU and 2,048→128 ReLU projections; concatenate → 128 fusion ReLU → dropout → two logits.
- D: identical, with the normalized scalar appended before fusion. Shared initial weights and initial predictions match. D adds 128 trainable scalar weights, initialized at zero.

Final common settings: AdamW, learning rate **0.001**, weight decay **0.0001**, dropout **0.3**, batch **128**, maximum **40 epochs**, patience **7**, validation Macro-F1 selection with validation-loss tie-break. Four rational configurations were tested equally for C/D; their mean validation Macro-F1 selected the common configuration. Seeds **42, 123, 2026** were used for all four final models. Test data did not select configurations or checkpoints.

## Actual test results

All values below are means across the three fixed seeds. Fake (0) is the positive class. Individual runs, PR-AUC, average precision, false positives and false negatives are in seed_metrics.csv.

| Model | Accuracy | Fake precision | Fake recall | Fake F1 | Macro-F1 | Weighted-F1 | ROC-AUC |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Text Only | 82.69% | 83.87% | 88.31% | 86.03% | 81.64% | 82.55% | 0.9003 |
| Image Only | 76.63% | 80.00% | 81.71% | 80.84% | 75.44% | 76.56% | 0.8460 |
| Text + Image | 85.55% | 86.65% | 89.93% | 88.25% | 84.74% | 85.46% | 0.9331 |
| Text + Image + Similarity | 87.87% | 88.76% | 91.50% | 90.10% | 87.21% | 87.81% | 0.9466 |

Sample standard deviation across seeds (percentage-point units):

| Model | Accuracy mean ± SD | Macro-F1 mean ± SD |
| --- | --- | --- |
| Text Only | 82.69 ± 0.10% | 81.64 ± 0.13% |
| Image Only | 76.63 ± 0.44% | 75.44 ± 0.47% |
| Text + Image | 85.55 ± 0.49% | 84.74 ± 0.56% |
| Text + Image + Similarity | 87.87 ± 0.86% | 87.21 ± 0.94% |

The following checkpoints were chosen by validation, before looking at test performance. All paths are `models/final/<kind>_<seed>.pt`; kinds are text, image, baseline and proposed.

| Model | Seed | Parameters | Best epoch | Validation Macro-F1 | Test Macro-F1 |
| --- | --- | --- | --- | --- | --- |
| Text Only | 42 | 115,202 | 6 | 81.32% | 81.68% |
| Text Only | 123 | 115,202 | 6 | 81.82% | 81.49% |
| Text Only | 2026 | 115,202 | 4 | 81.48% | 81.75% |
| Image Only | 42 | 279,042 | 4 | 73.89% | 75.01% |
| Image Only | 123 | 279,042 | 1 | 72.87% | 75.36% |
| Image Only | 2026 | 279,042 | 1 | 73.20% | 75.94% |
| Text + Image | 42 | 393,858 | 1 | 84.16% | 85.13% |
| Text + Image | 123 | 393,858 | 3 | 84.01% | 84.10% |
| Text + Image | 2026 | 393,858 | 3 | 84.87% | 84.98% |
| Text + Image + Similarity | 42 | 393,986 | 3 | 86.44% | 88.22% |
| Text + Image + Similarity | 123 | 393,986 | 3 | 85.96% | 86.35% |
| Text + Image + Similarity | 2026 | 393,986 | 3 | 86.05% | 87.06% |

## Similarity and paired statistics

| Class | Count | Mean | Median | SD | Q1 | Q3 |
| --- | --- | --- | --- | --- | --- | --- |
| Fake | 30,312 | 0.25838 | 0.25579 | 0.05157 | 0.22033 | 0.29473 |
| Real | 19,688 | 0.29678 | 0.29887 | 0.04069 | 0.27186 | 0.32398 |

Training paired CLIP similarity: fake mean 0.25838, median 0.25579; real mean 0.29678, median 0.29887. Mann–Whitney U(real,fake)=431775808.0; p < 1e-300 (floating-point underflow); rank-biserial effect (real minus fake)=0.44701. Underflowed numeric p=0 is not an exactly zero probability. Distribution association does not by itself establish incremental classifier value.

Prespecified seed 42: D corrected **158** C errors and introduced **87** new errors. Exact two-sided McNemar **p=6.73401e-06**. Paired bootstrap (5,000 common-index resamples) gives Macro-F1 change **+3.093 points**, 95% interval **[+1.778, +4.362] points**. Fake-F1 relative change for this seed is **+2.467%**. Separate accuracy/fake-F1 intervals and other seeds' paired checks are in paired_comparison.json. Other seeds are sensitivity checks; repeated predictions were not pooled as independent examples.

## Error evidence and interpretation

Saved categories: {"Both correct": 2062, "Both wrong": 193, "Baseline wrong / Proposed correct": 158, "Baseline correct / Proposed wrong": 87}. Every row includes the ID, text, image path, true label, C/D predictions, C/D confidence and CLIP similarity. error_examples.csv samples each category across its similarity range; corresponding figures show the actual images. error_summary.csv uses thresholds from training similarity quartiles; error_sources.csv exposes source-community patterns. Differences in correctness do not establish that a model performed factual reasoning. Softmax confidence is uncalibrated.

| Training-defined similarity band | Test pairs | Fake pairs | C accuracy | D accuracy | D−C points |
| --- | --- | --- | --- | --- | --- |
| High (above train Q3) | 539 | 159 | 81.26% | 86.64% | +5.38 |
| Low (below train Q1) | 753 | 670 | 91.10% | 93.23% | +2.12 |
| Middle | 1208 | 680 | 84.85% | 87.00% | +2.15 |

These are descriptive slices of the prespecified seed-42 test predictions, not additional model-selection criteria or independently confirmed subgroup effects. Subtype slice counts/accuracies and unimodal predictions are also saved.

### Observations from saved predictions and actual images

These observations use the prespecified seed-42 checkpoints. They describe this held-out subset; no labels, images, thresholds or model settings were changed after inspecting the results.

- **Most of the net correction was on real examples.** False positives (real predicted fake) fell from 212 to 158, while false negatives (fake predicted real) fell from 139 to 122. The net improvement was 71 correct decisions: 158 corrections minus 87 newly introduced errors. Both models were correct on 2,062 pairs and wrong on 193.
- **The gain is not confined to low similarity.** Accuracy increased by 5.38 percentage points in the high-similarity band (539 pairs), 2.12 in the low band (753), and 2.15 in the middle band (1,208). Bands use training quartiles. Their class mixtures differ: 670 of the 753 low-band pairs are fake, versus 159 of 539 high-band pairs. These descriptive differences should not be interpreted as separately established subgroup effects.
- **The result varies across source categories.** Real-category accuracy rose from 78.61% to 84.06%. False-connection accuracy decreased slightly, from 88.26% to 87.67% (451 to 448 correct among 511). Manipulated-content accuracy was already high, 96.69% versus 96.83% (672 versus 673 correct among 695). Thus an overall improvement does not mean improvement on every kind of fake example. The small imposter category has only 56 test pairs, so its larger change is especially uncertain.
- **Validation curves show overfitting.** Training loss continues downward while validation loss rises. The saved seed-42 C checkpoint is epoch 1 and D is epoch 3, chosen by validation Macro-F1. Final predictions use those saved best checkpoints, not the last training epoch. No additional regularization search was conducted after test inspection.

#### Concrete reviewed examples

1. **Low alignment, corrected by D — `cv87ku`.** The title describes a wheat field at sunset, but the retrieved image shows a water-control structure. Its official subtype is false connection and CLIP cosine is 0.0574. C predicts real (confidence 0.554); D predicts fake (0.981). The image-only model was also correct. The mismatch is visually evident, but this one correction does not establish the model's internal reasoning.
2. **Low alignment, newly wrong in D — `3ji8ci`.** The real-labelled title concerns a cat-level street-view map in Japan. The current retrieved image is a “RebelMouse vs. WordPress” promotional graphic. Cosine is 0.0786; C predicts real (0.778), while D predicts fake (0.900). This demonstrates that low alignment is not equivalent to a fake label. It also exposes a remaining reconstruction limitation: an official image URL and a verified downloaded hash cannot prove that today's preview is the original article image. The pair remains in the fixed test set.
3. **High alignment, both correct — `4yz0zn`.** The text matches wording on an album-style cover. It is labelled fake in the official satire/parody subtype despite a cosine of 0.4207. Both C and D predict fake. A simple “high similarity means real” rule would be inappropriate.
4. **High alignment, newly wrong in D — `85f39a`.** Another album-style image carries wording matching its title. Cosine is 0.4156 and the official satire/parody label is fake. C predicts fake (0.758), but D predicts real (0.608). Text-only and image-only predictions were both correct. This is a concrete counterexample to universal benefit from adding the scalar.
5. **High alignment, both wrong — `drzvidl`.** The image depicts the title's hedgehog in a small kayak; cosine is 0.3962. The official subtype is manipulated content. Both multimodal heads predict real, with C confidence 0.790 and D 0.989. The image-only head predicts fake, while text-only predicts real. Semantic agreement can coexist with a manipulated-content label, and greater confidence can accompany an error. We did not independently authenticate the image or relabel it.
6. **Different unimodal decisions — `4h8yj8`.** A coyote photograph accompanies a short matching title. Text-only predicts fake, image-only predicts real, C predicts fake and D correctly predicts real (official label 1), with cosine 0.2937. This illustrates the value of retaining unimodal controls, without claiming a causal explanation for an individual neuron or decision.

All four error panels were visually reviewed, along with the dataset, source-composition, similarity, training, comparison, confusion-matrix and ROC/PR figures. Captions and images are readable. Individual predictions/confidences, source categories and similarity scores are preserved in `results/predictions/error_analysis.csv`; the 32 displayed cases are in `error_examples.csv`.

The retained manipulated-content share is about 10% of training pairs and 27.8% of test pairs. Conservative source-family exclusions and acquisition availability change the source mix despite preserving binary proportions and official partitions. Together with distant supervision, publisher graphics, source style and possible URL-content drift, this limits generalization. The observed gain measures the added value of this CLIP-derived scalar within the documented reconstructed subset; it does not establish a general-purpose fact-checker.


## Runtime, limits and reproduction

Recorded major completed stages total **3.22 hours**; see runtime_summary.csv for the scope of each timer. This excludes manual development, idle time and some interrupted retrieval/extraction work, so it is not a complete wall-clock duration. The feature cache contains **408.2 MiB** of compressed numeric chunks. The machine is a Ryzen 7 5800HS with about 16 GB RAM, integrated AMD graphics and no CUDA. The initial 20,000-training-pair phase is preserved in results/initial_20000.zip and has no test metrics.

Limitations: distant-supervision labels and subreddit style; broader-than-news fake categories; URL availability and image recompression; conservative duplicate/source-family exclusions and resulting source-distribution differences; possible remaining near-duplicates/events and unknown pretrained-web overlap; frozen encoders; one retained test subset; and only three training seeds. A positive result here is not a universal claim about misinformation detection. No optional encoder fine-tuning or Streamlit demo was added.

From the project root with the documented environment installed and `$projectPython` set as in README:

```powershell
& $projectPython -B run_experiments.py all --train 50000 --validation 2500 --test 2500
```

Use the saved input manifest/images/cache for exact reproduction; fresh web acquisition may differ. Source, cached features, final checkpoints, machine-readable metrics, predictions, report figures, README and the complete historical project_worklog.md are retained. Final verification and cleanup are recorded in the worklog. The documented all command successfully reused all final features and checkpoints after cleanup. Predictions, metrics, paired statistics and checkpoints remained byte-identical; all loaded feature matrices remained bit-identical. Required scientific stages are complete; optional fine-tuning/demo and larger-scale experiments are outside this completed primary study.
