# Execution, verification and reproduction

All 1,001 authorized records were encoded with original pinned BERT, ResNet50 and CLIP. CPU reported by Windows registry: `AMD Ryzen 5 5600GT with Radeon Graphics`. Six threads, batches of 16; no GPU or training. Encoder timings (seconds, excluding/including load separately):

| encoder | load_seconds | inference_seconds | truncated_caption_rows |
|---|---|---|---|
| bert | 0.20866 | 45.60625 | 0.00000 |
| clip | 2.32509 | 103.09750 | 0.00000 |
| resnet | 0.42663 | 119.41282 | â€” |

Total extraction invocation: 274.41s. Peak process working set: 1.26GiB (process lifetime, not stage-isolated). BRAS scoring: 0.633s. Frozen head load/inference per seed is recorded in `inference_profile.json`; CCR policy overhead is separate. Cached replays perform no new encoder inference.

24 unit tests passed before extraction, including exact mapping, paired links, archive traversal/symlink/case checks, image decoding/hash mutation, cached resume, cluster randomization, independent weighted metrics, and direct NumPy replay of all 12 frozen A/B/C/D heads across 3 seeds. A Windows ZIP-name test exposed automatic filename normalization; validation now checks original archive names, and a raw-header regression passed before source/protocol lock. Final real-output replay, metric recomputation and preservation receipts accompany this report.

Historical inventory: 59,154 files. Preservation passed, zero changed/missing files. Original input ZIP/CSV, model checkpoints, BRAS/CCR and NewsCLIPpings files are included. New outputs are isolated in `results/research_extension/verite/`; restricted manifests/features stay in `data/external/verite/restricted_cache/`. No shared dependency versions changed.

```powershell
$projectPython = 'python'
Set-Location 'C:\path\to\MultimodalFakeNews'
& $projectPython -B -m unittest src.research_extension.verite.tests -v
if ($LASTEXITCODE -ne 0) { throw "VERITE unit tests failed" }
foreach ($stage in @('audit','lock','extract','evaluate','statistics','plots','report','verify-preservation')) {
    & $projectPython -B run_verite_eval.py $stage
    if ($LASTEXITCODE -ne 0) { throw "VERITE stage failed: $stage" }
}
```

Public source provenance is already pinned in the local audit receipt. No gated access or image download is needed to rerun. Reports/metrics are immutable; changed inputs or code require an explicitly versioned new study, not silent overwrite.
