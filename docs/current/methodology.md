# Methodology explained

The project asks two related questions: does a CLIP similarity feature help the original classifier, and does reference adjustment help when the evaluation task changes? These questions require different comparisons.

## From a pair to a prediction

A headline passes through frozen BERT to produce a 768-value CLS feature. Its image passes through frozen ResNet50 to produce 2,048 values. CLIP ViT-B/32 separately encodes both inputs into normalized 512-value vectors. Their dot product is a single cosine score. CLIP vectors do not enter the original classifier directly.

Training-fitted means and standard deviations normalize the classifier inputs. Each BERT/ResNet feature is projected to 128 values. The head combines the projections, applies a 128-unit fusion layer with ReLU and dropout 0.3, and produces two logits. The encoders remain frozen. The trained heads and preprocessing are reused unchanged for VERITE.

| Research label | Code/checkpoint name | Extra input beyond BERT and ResNet50 |
|---|---|---|
| A | `baseline` | None |
| B | `proposed` | Raw paired CLIP cosine, standardized before the head |
| C | BRAS `C` | Text-conditioned and image-conditioned margins |
| D | BRAS `D` | Cosine and both margins |
| CCR | Saved conditional-reliance policy | Routes between A and B; the executed guard falls back to B |

`proposed` is a historical checkpoint key for B. It is not a claim that B is a new state-of-the-art detector. C/D refer to the BRAS extension, not the order of models in an older four-model table. We retain those keys for compatibility.

## What BRAS computes

BRAS is the project's name for Bidirectional Reference-Adjusted Similarity. Its frozen bank contains 4,096 Fakeddit training examples. For a query caption, find the top 16 valid bank images by CLIP similarity; their mean is the text-conditioned reference penalty. Reverse the modalities for the image-conditioned penalty. Exclusion rules remove known matching identities/content, without consulting labels.

If `s` is paired cosine, `rT` the text-conditioned penalty and `rI` the image-conditioned penalty, the margins are `m_text = s - rT` and `m_image = s - rI`. The symmetric score is their mean: `b = s - (rT + rI)/2`. With identical neighborhoods, CSLS is `2s - rT - rI = 2b`. This is an established normalization form, not a new scoring principle.

C and D consume directional features through trained heads. Their predictions are not interchangeable with the standalone symmetric ranking score. Full and half-weight directional corrections in the manuscript are fixed analytical comparisons, not VERITE-tuned settings.

## Four endpoints on VERITE

| Endpoint | Comparison | What it can establish |
|---|---|---|
| Miscaptioned ranking | True and misleading captions sharing an image | Which caption receives the higher correspondence score |
| Out-of-context ranking | True and misleading images sharing a caption | Which image receives the higher correspondence score |
| Global discrimination | All truthful versus misleading records | How scores order records across families, using AUC/AP |
| Frozen classifier transfer | Unchanged Fakeddit heads on VERITE | Performance under the declared binary cross-task mapping |

The mapping is true → truthful (1), miscaptioned/out-of-context → misleading pair (0). It does not equate VERITE categories with Fakeddit's distant-supervision fake/real labels. Fakeddit's original reported ROC/PR positive class is fake; VERITE's is truthful. AP means average precision, not trapezoidal PR area.

Repeated images/captions create dependence. The VERITE analysis groups linked content into 336 components and uses paired component resampling. Seeds share the same records and are not independent datasets. Primary, secondary and later exploratory tests retain their separate roles and correction families. See the unchanged [locked protocol](../research_extension/verite/protocol_and_methodology.md) and the manuscript for exact assumptions, seeds and counts.

The [original architecture diagram](../../README.md#models) describes A/B correctly. The table above extends that diagram to C/D and CCR without suggesting that the reference bank supplies factual evidence.
