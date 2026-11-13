# Research gap and contribution

The question is whether a normalization benefit on paired image–caption ranking remains useful for global discrimination and frozen classifier transfer when the encoder, reference bank and records stay fixed. These endpoints are often discussed together, but they answer different questions.

The manuscript's verified related work already establishes the ingredients: VERITE studies unimodal bias and transfer; MUSE separates category behavior in evidence-based systems; CSLS, margin mining and querybank normalization adjust similarities relative to reference examples; FND-CLIP combines CLIP-related information with learned multimodal features. The contribution is therefore an empirical evaluation of known ideas under a common contract. It is not an invention of reference normalization, CLIP fusion or Fakeddit-to-VERITE evaluation.

The study contributes a mathematical comparison of the scores, a separation of four endpoints on the same saved features, and a complete account of frozen classifier seeds and negative results. Symmetric BRAS is half-CSLS for matched references. A benefit on the original miscaptioned ranking test does not establish a global or classifier benefit. The stronger text-conditioned result remains secondary; the later score audit remains exploratory.

Compatible numerical comparisons are limited to the scores and heads actually evaluated here. MUSE, FND-CLIP and evidence-retrieval systems use different architectures, evidence or selection procedures. Their published scores are context, not matched baselines. FND-CLIP implementation claims remain tied to the inspected preprint; the later publication's full text still needs checking before making final-version claims.

This supports manuscript development as a bounded empirical study. Wider generality would require independent evidence across other encoders, reference populations or real-image benchmarks, with decisions fixed before evaluation. The author-embedding NewsCLIPpings study does not supply that replication. No additional experiment is claimed in this editorial revision.

Use the unchanged [reference verification report](../../journal_submission_final/SNCS_Reference_Verification.md) for the sixteen cited works and version boundaries, and the [manuscript](../../manuscript_editorial_20261112/Manuscript_Editorial.docx) for the full argument. The earlier [BRAS gap report](../research_extension/research_gap.md) records an earlier stage; its blocked-access statements are historical, not the current VERITE status.
