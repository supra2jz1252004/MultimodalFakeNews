# CCR pre-training audit

Audit created 2026-11-06T18:00:34.748535+00:00. Preserved inventory: 58,759 files, 4,552,909,479 bytes; SHA-256 9883fe3a8a11ea5dc7996aec9ed1a686a2b8b75b399cc29bd302f68973f703cd. All pre-existing original and BRAS files except Python bytecode are covered. End-of-run rehash required.

Manifest SHA-256 60aeb7e8a04a0296d92e6d5670de1396ae35037a84331c428ab25bf753450edb; 50,000 train/2,500 validation/2,500 historical test. 55,000 unique linked-submission families; one retained row per family. Known family-disjoint folds therefore mostly reduce to row partitions. Event/publisher independence cannot be asserted. Subreddit is not a legitimate inference feature.

All six historical final A/B checkpoint identities, seeds, manifest hashes, architecture and budgets verified from actual checkpoints. BERT768, ResNet2048, cosine1 caches are read using the extension's non-mutating loader. Heads use hidden128, dropout0.3, AdamW0.001/weight-decay0.0001, batch128, max40 epochs, patience7, six CPU threads. Optional text/image-only scores omitted to avoid in-sample training predictions.

The previous 1,000-row balanced same-dataset holdout has already been inspected and must now be called exploratory. BRAS negative findings and corrected paired tables remain unchanged. Historical A/B validation selection prevents claiming independent validation for this gate.

The new code is isolated under src/research_extension/ccr; models/results/docs use matching CCR namespaces. OOF input caches also remain under the CCR result namespace. No original training or feature-writing entry point will run. No external row labels or model scores were inspected before this audit/protocol lock. Reading dataset documentation discloses the official class schema and counts only.
