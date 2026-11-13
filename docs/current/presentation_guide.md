# Presentation and viva guide

## A short opening

“We studied whether a CLIP image–text similarity score helps a BERT and ResNet50 classifier. It helped on our retained Fakeddit subset. We then tested reference-adjusted scores and the frozen classifiers on VERITE. One paired-ranking result improved, but global discrimination and classifier transfer did not show the same consistent benefit. Our contribution is to explain and measure these differences under a fixed evaluation setup.”

## Suggested sequence

1. Define the input pair and the limits of fake/real supervision.
2. Explain BERT, ResNet50 and the single CLIP cosine input using the README diagram.
3. Show the original A/B comparison and its controlled training setup.
4. Explain the reference bank and why symmetric BRAS equals half-CSLS.
5. Separate miscaptioned ranking, out-of-context ranking, global discrimination and frozen transfer before showing their results.
6. Discuss the negative BRAS classifier findings and CCR fallback.
7. End with limitations, reproducibility and what the evidence actually supports.

## Questions to practise

**Why use a scalar CLIP feature?** It tests a specific additional source of information while retaining the same BERT/ResNet inputs and matched heads. It does not isolate factual reasoning from pretrained semantic information.

**What was trained?** The original classifier heads and historical extensions were trained on Fakeddit according to their saved protocols. The encoders and all heads were frozen for VERITE. This editing pass trained nothing.

**Is BRAS new?** It is our working label. Under matched references, the symmetric scalar is half-CSLS. Directional features inside a trained head are a different object, but that distinction does not establish algorithmic novelty.

**Why does miscaptioned ranking improve without global AUC improving?** In a shared-image pair, the image-conditioned reference penalty is constant and cancels. Across families, those constants can change score ordering. The comparisons ask different questions.

**Did CLIP transfer successfully?** Some descriptive metrics improved, but accuracy fell and the primary seed-42 Macro-F1 comparison remained inconclusive. State all three facts together.

**Why retain CCR if it does nothing different?** The saved guard sent every case to B. That is the observed outcome of this policy and an identity control, not evidence for successful adaptive routing or proof that all routing methods fail.

**Why not select the best seed?** Seeds are repeated fits evaluated on the same records. Choosing the best one would hide variation and change the planned comparison.

**Are mismatched pairs fake news?** A mismatch establishes a correspondence problem, not an independently verified false claim. Fakeddit and VERITE have different supervision schemes.

**Can we show VERITE examples?** Follow the provider's access and redistribution terms. Use the aggregate figures in the manuscript for this presentation; do not copy restricted records into public slides.

**What remains before submission?** Author and faculty review, accurate AI-use disclosure, confirmed contributions, applicable ethics and sharing conditions, repository verification, and the unresolved final-version literature check. Improved prose does not resolve those decisions.

For every number shown, name its dataset, endpoint, model/score, seed role and analysis family. Use the retained tables rather than manually rebuilding a more favorable subset. Each team member should be able to trace one result from its report to the responsible code and explain the relevant limitation.
