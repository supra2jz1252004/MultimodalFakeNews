# Reproducibility and research artifact availability

This folder accompanies the SN Computer Science manuscript on CLIP similarity normalization, paired correspondence ranking, global discrimination and frozen classifier transfer on VERITE.

## Publicly available

- [aggregate_results.json](aggregate_results.json): 16 aggregate tables and 101 aggregate rows transcribed, without numerical modification, from existing project documentation. Values are strings to preserve precision and units; source paths and sections are recorded.
- Implementation, evaluation code, requirements, model settings and tests are in the root and source directories.
- [Original Fakeddit documentation](../docs/results_summary.md) and [VERITE research documentation](../docs/research_extension/verite/) give experimental context, hypotheses and limitations.

The JSON is TABLE-DERIVED REPORTING MATERIAL, not original machine-produced result artifacts, rerun evidence, complete reproducibility records, or per-record predictions. Do not pool Fakeddit and VERITE metrics as one dataset.

## Original saved artifacts: research-artifacts-v1

The [versioned release](https://github.com/supra2jz1252004/MultimodalFakeNews/releases/tag/research-artifacts-v1) contains the 68 original artifacts approved in [issue #1](https://github.com/supra2jz1252004/MultimodalFakeNews/issues/1#issuecomment-6873874526):

- 18 project-trained heads: text, image, baseline and CLIP-augmented Fakeddit heads for seeds 42, 123 and 2026, plus BRAS C/D heads for those seeds. Frozen VERITE A/B reuse the original baseline/proposed heads.
- 2 train-only numeric normalization files.
- 48 original aggregate experiment, configuration and provenance artifacts, including saved CCR policy summaries and explicitly exploratory evidence.

These are original saved artifacts, separate from the table-derived JSON above. No models were retrained, tensors rewritten, metrics recomputed or statistical analyses rerun for publication. The original artifact bytes match the reviewed SHA-256 inventory.

Download the [approved ZIP](https://github.com/supra2jz1252004/MultimodalFakeNews/releases/download/research-artifacts-v1/original-artifacts-v1-approved.zip) and [SHA256SUMS.txt](https://github.com/supra2jz1252004/MultimodalFakeNews/releases/download/research-artifacts-v1/SHA256SUMS.txt). The archive is 22,119,122 bytes and contains 72 files: 68 originals plus README, MANIFEST, VERIFY and RELEASE_FILES.

```text
bc269804061aa7aadcab588184208d9c8486d293007f5ca92c344ef5b46a9052  original-artifacts-v1-approved.zip
```

MANIFEST.json lists each original relative path, size and SHA-256. After verifying the ZIP hash and extracting into an empty folder, run `python -B VERIFY.py` with Python 3.11 or newer. It verifies all 71 listed hashes and rejects unexpected files; RELEASE_FILES.json is the 72nd member. Checkpoints and ZIPs are Release assets, not additions to Git history. The corresponding source commit is [a729c2c](https://github.com/supra2jz1252004/MultimodalFakeNews/tree/a729c2c72564233fe57c75cea996c631b0d587e1).

On 14 November 2026, an independent HTTP download without authentication or cookies returned HTTP 200 for the release page and both assets. The downloaded ZIP and checksum file matched their local originals and GitHub's asset digests. ZIP CRC, safe paths, all 68 approved artifact hashes and all 71 release-file hashes passed; no unexpected members were found. These are artifact-integrity checks, not independent scientific replication.

## Excluded files and reproduction limits

This is a **partial reproducibility package**, not a complete replay archive.

- Restricted VERITE images, captions, annotations, record-level labels/predictions/embeddings and private manifests remain excluded. No credentials, dataset archives or private record files are published.
- The exact Fakeddit-derived reference bank and reference identifiers are withheld because they contain source-record material. A replacement bank would change the experiment.
- CCR gate joblib objects and OOF records remain excluded. Saved policy JSON and aggregate summaries do not provide full CCR replay.
- Pretrained encoders and the exact acquired Fakeddit input manifest/images are not bundled. Newly fetched images may differ from those in the original availability-dependent sample.
- Exploratory-v2 summaries and source hashes are included, but unreviewed exploratory executable source remains excluded. These post-hoc summaries are not a complete executable workflow or an independent replication.

Public assets alone therefore do not enable exact full-dataset replay, CCR gate reproduction or independent resampling. For researchers with independently authorized access to the exact private inputs, the release README describes strict checkpoint loading, saved preprocessing and normalization contracts, source locks and unchanged entry points. Do not refit normalization, replace missing inputs or relax frozen checks to claim reproduction.

The project author approved distribution of the listed artifacts. This release does not assign a new license to third-party sources, pretrained weights or datasets. [VERITE access restrictions](https://huggingface.co/datasets/stefpapad/VERITE) remain applicable. Historical reports are preserved; statements that artifacts were unavailable describe their earlier publication status.

## Interpretation

The primary miscaptioned ranking comparison showed a gain for symmetric BRAS, a half-CSLS-equivalent score under matched reference definitions. Primary fixed-head classifier-transfer results were not conclusive; there was no consistent BRAS/CCR transfer benefit or established global discrimination improvement. Exploratory findings are not independently replicated confirmatory results.