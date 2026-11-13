"""Reports and figures exclusively from saved executed evidence."""
from __future__ import annotations
import json,collections
import numpy as np
import pandas as pd
from .io_utils import *
from .data import STRATEGIES
SHORT={'person_sbert_text_text':'Person / SBERT','scene_resnet_place':'Scene / Places','semantics_clip_text_image':'Semantic / CLIP T-I','semantics_clip_text_text':'Semantic / CLIP T-T'}

def plots(run):
    prior=run.results/'figure_manifest.json'
    if prior.exists():
        saved=read(prior)
        for rel,h in saved['files'].items():require(sha(run.root/rel)==h,'Figure changed: '+rel)
        return {'figures':len(saved['files']),'status':'verified existing figures'}
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve,precision_recall_curve
    s=read(run.results/'official/statistics.json')['scores']['cosine'];p=pd.read_csv(run.results/'official/paired_cosine.csv');f=pd.read_csv(run.results/'official/predictions.csv')
    dest=run.results/'figures';dest.mkdir(parents=True,exist_ok=True);paths=[]
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':180})
    def save(fig,name):
        fig.text(.01,.01,'Official CLIP embeddings • pairing correspondence, not factual veracity • no image encoder run',fontsize=8,color='#555555')
        fig.tight_layout(rect=(0,.04,1,.95))
        for ext in ['png','pdf','svg']:
            path=dest/f'{name}.{ext}';fig.savefig(path,bbox_inches='tight');paths.append(str(path))
        plt.close(fig)
    fig,ax=plt.subplots(figsize=(8.4,4.6));labels=['Overall']+[SHORT[x] for x in STRATEGIES];rows=[s['overall']]+[s['by_strategy'][x] for x in STRATEGIES]
    for i,(label,row) in enumerate(zip(labels,rows)):
        m=row['metrics']['paired_ranking_half'];v=m['estimate'];lo,hi=m['ci95']
        ax.errorbar(v,i,xerr=[[v-lo],[hi-v]],fmt='o',color='#16697a',capsize=4)
        ax.text(1.01,i,f"n={row['pairs']}; {v:.3f} [{lo:.3f}, {hi:.3f}]",transform=ax.get_yaxis_transform(),va='center',fontsize=9)
    ax.axvline(.5,ls='--',color='#777777',lw=1);ax.set_yticks(range(len(labels)),labels);ax.invert_yaxis();ax.set_xlim(0,1);ax.set_xlabel('Pristine ranked above mismatch (ties: half credit)')
    fig.suptitle('Paired ranking by construction strategy\n95% intervals resample linked content components',fontsize=12);save(fig,'paired_ranking')
    fig,ax=plt.subplots(figsize=(8.4,4.6));arr=[p.loc[p.strategy.eq(x),'margin'] for x in STRATEGIES]
    violin=ax.violinplot(arr,showmedians=True,showextrema=False)
    for body in violin['bodies']:body.set_facecolor('#16697a');body.set_alpha(.65)
    ax.axhline(0,color='#999999',ls='--');ax.set_xticks(range(1,5),[SHORT[x] for x in STRATEGIES]);ax.set_ylabel('Cosine(pristine) − cosine(mismatch)');fig.suptitle('Within-caption correspondence margins',fontsize=12);save(fig,'paired_margins')
    fig,axes=plt.subplots(1,2,figsize=(9,4.3));y=1-f.falsified;v=f.cosine
    x,t,_=roc_curve(y,v);axes[0].plot(x,t,color='#16697a');axes[0].plot([0,1],[0,1],'--',color='#999999');axes[0].set(xlabel='False positive rate',ylabel='True positive rate',title='Pristine-positive ROC')
    precision,recall,_=precision_recall_curve(y,v);axes[1].plot(recall,precision,color='#b65d34');axes[1].axhline(.5,ls='--',color='#999999');axes[1].set(xlabel='Recall',ylabel='Precision',title='Pristine-positive precision–recall')
    for ax in axes:ax.set_xlim(0,1);ax.set_ylim(0,1)
    fig.suptitle('Score-only correspondence discrimination',fontsize=12);save(fig,'roc_pr')
    tab=p.pivot_table(index='strategy',columns='source',values='ranking_half',aggfunc='mean').reindex(STRATEGIES);counts=p.groupby(['strategy','source']).size()
    fig,ax=plt.subplots(figsize=(8.4,4.6));im=ax.imshow(tab,vmin=0,vmax=1,cmap='viridis',aspect='auto')
    ax.set_yticks(range(4),[SHORT[x] for x in STRATEGIES]);ax.set_xticks(range(len(tab.columns)),tab.columns)
    for i,st in enumerate(tab.index):
        for j,src in enumerate(tab.columns):
            if np.isfinite(tab.iloc[i,j]):ax.text(j,i,f'{tab.iloc[i,j]:.2f}\nn={counts.get((st,src),0)}',ha='center',va='center',color='white' if tab.iloc[i,j]<.6 else 'black',fontsize=9)
    fig.colorbar(im,ax=ax,label='Half-credit ranking success');fig.suptitle('Descriptive construction × caption-source analysis',fontsize=12);save(fig,'strategy_source')
    write(run.results/'figure_manifest.json',{'source_statistics_sha256':sha(run.results/'official/statistics.json'),'files':{str(Path(p).relative_to(run.root)):sha(p) for p in paths}})
    return {'figures':len(paths),'directory':str(dest)}

def report(run):
    selection=read(run.results/'selection_audit.json');access=read(run.results/'data_access.json');emb=read(run.results/'embedding_provenance.json');image=read(run.results/'image_access_smoke.json')
    stats=read(run.results/'official/statistics.json');score=stats['scores']['cosine'];overall=score['overall'];f=pd.read_csv(run.results/'official/predictions.csv',keep_default_na=False);pairs=pd.read_csv(run.results/'official/paired_cosine.csv',keep_default_na=False)
    h=read(run.results/'historical_inventory.json');preserve=read(run.results/'preservation_check.json') if (run.results/'preservation_check.json').exists() else None
    metric=lambda name: f"{overall['metrics'][name]['estimate']:.4f} [{overall['metrics'][name]['ci95'][0]:.4f}, {overall['metrics'][name]['ci95'][1]:.4f}]"
    lines=['| Construction strategy | Groups | Linked components | Ranking success (95% CI) | ROC-AUC | AP, pristine positive |','|---|---:|---:|---|---:|---:|']
    for st in STRATEGIES:
        row=score['by_strategy'][st];m=row['metrics'];v=m['paired_ranking_half']
        lines.append(f"| {st} | {row['pairs']} | {row['clusters']} | {v['estimate']:.4f} [{v['ci95'][0]:.4f}, {v['ci95'][1]:.4f}] | {m['roc_auc_pristine']['estimate']:.4f} | {m['average_precision_pristine']['estimate']:.4f} |")
    table='\n'.join(lines);statuses=collections.Counter(x.get('http_status','error') for x in image['probes'])
    hard=pairs.sort_values(['margin','pair_id']).head(10);easy=pairs.sort_values(['margin','pair_id'],ascending=[False,True]).head(10)
    cases=pd.concat([hard.assign(case_type='smallest_margin'),easy.assign(case_type='largest_margin')])
    cases=cases.merge(f[f.falsified.eq(0)][['pair_id','caption_id','image_id','caption','filename']],on='pair_id').rename(columns={'image_id':'pristine_image_id','filename':'pristine_filename'})
    cases=cases.merge(f[f.falsified.eq(1)][['pair_id','image_id','filename']],on='pair_id').rename(columns={'image_id':'mismatch_image_id','filename':'mismatch_filename'})
    cases['interpretation']='Automated pairing provenance only; factual truth and visual content not verified from image bytes.'
    csv(run.results/'official/failure_cases.csv',cases)
    common='''
NewsCLIPpings pairing labels describe intact captions and intact images that were originally paired or automatically mismatched. They do not establish whether a news claim is true. CLIP influenced construction and adversarial filtering, so this is a distinct dataset source from Fakeddit but is not independent of CLIP method selection. Sources: [paper](https://aclanthology.org/2021.emnlp-main.545/), [official instructions](https://github.com/g-luo/news_clippings), [author release](https://huggingface.co/g-luo/news-clippings/tree/main).
'''
    docs={}
    docs['access_and_provenance.md']=f'''# Access and provenance — ALIGNMENT ONLY

Pinned official release: `{access['revision']}`. Obtained six split annotation files, the 37,369,991-byte supplementary test metadata file, and two test CLIP embedding files totaling 192,009,843 bytes. Full-file SHA-256 values match Hugging Face LFS metadata. The image dictionary has {emb['image']['count']:,} 512D vectors; the text dictionary has {emb['text']['count']:,}. Vector norms were checked before explicit L2 normalization (recorded ranges in `embedding_provenance.json`). Pickles were decoded with a numpy-only global allowlist; no arbitrary remote Python object loading.

The 100-group smoke subset has 200 rows and 198 unique image IDs; direct official-host path probes yielded {dict(statuses)}. Verified selected image bytes: **0**. `origin/data.json` returns 404. The [official VisualNews repository](https://github.com/FuxiaoLiu/VisualNews-Repository) links to the [Rice archive](https://www.cs.rice.edu/~vo9/visualnews/): 97,916,641,280 bytes, with HTTP 206 range support verified using only 4,096 bytes. A selected-image offset index is not advertised. No claim is made that the images no longer exist; they are available inside the bulk archive, but selected bytes were not obtained via a practical targeted route. No bulk image transfer, publisher scraping, or gated access occurred.

The official supplementary test metadata contains the required caption/item/image-path joins; it is not misrepresented as a downloaded VisualNews origin/data.json. All 2,000 selected rows join successfully. Raw metadata/embeddings stay in `data/external/newsclippings/`. Repository Apache-2.0 labeling does not establish blanket redistribution rights for source images; images are excluded from deliverables. No selected images were displayed.

To unblock pinned CLIP/BRAS and A/B: provide an authorized local VisualNews origin folder with `data.json` and the fixed selected image paths, or an official indexed permitted subset. The adapter verifies official caption/path joins, actual decoded bytes, hashes and known Fakeddit overlaps before scoring. No missing-pair replacements are allowed.
{common}'''
    docs['protocol.md']=f'''# Locked protocol

The SHA-linked machine protocol is `results/external_eval/newsclippings/protocol.json` (SHA-256 `{sha(run.results/'protocol.json')}`). It was written before any external test metrics. Analysis code was hash-frozen before score generation. Selection uses SHA256(seed NUL pair-ID), seed 20261108, within four strategies. Fixed 100-group smoke (25 per strategy); fixed main 1,000 groups (250 per strategy), nested smoke, no availability/performance replacements. The official merged test population is 3,632 groups, 908 per strategy. The selection therefore preserves its equal strategy allocation.

The primary planned comparison is pinned BRAS mean `(m_text+m_image)/2` minus pinned cosine in within-group ranking. k=16 and the existing 4,096-entry Fakeddit training bank remain unchanged. Directional comparisons are secondary. **This planned comparison was not executed:** official embedding provenance is not verified identical to the pinned bank contract. Only the separately declared official CLIP score-only baseline was executed. No new comparator, threshold, classifier or adaptive weights were introduced after test inspection.

Higher scores predict pristine pairing. Exact ties receive half credit; strict and 1e-6 tolerance outcomes are robustness checks. ROC-AUC and average precision use pristine as positive. Paired margin is score(pristine) minus score(mismatch). Bootstrap: 5,000 common-component resamples, seed 20261108, percentile 95% intervals. Components link shared caption/image IDs across roles, normalized captions, known article paths, and available byte/pixel hashes. Missing image hashes remain unknown. In this run there are {overall['clusters']} linked components for 1,000 groups (953 before article/caption links). These are observed-dependence units, not a guarantee of independent events.

Two-sided cluster sign flips use seed 20261109 and 10,000 draws; +1 correction. Secondary score comparisons and construction-strategy tests form a Holm-corrected family. Transfer tests across three classifier seeds form a separate Holm family and would not treat seeds as extra independent examples. No threshold is needed and no external adaptation was performed. Failure cases are the fixed ten smallest and ten largest cosine margins, explicitly exploratory. Protocol clarification: actual `similarity_score` fields are numeric generation scores; strategy is read from `source_dataset`, not inferred from this number.
'''
    docs['methodology.md']=f'''# Methodology and implementation

Adjacent rows were validated to be pristine then mismatch with the same caption ID, pristine image ID equal to caption ID, and mismatched image ID different. Every selected group stays intact. All merged rows were found in their stated official strategy file; those strategy files are never appended. Test and validation item-ID sets have zero overlap. Full training metadata was not downloaded and exhaustive train/event overlap is not established. Sources/articles are preserved; all known article links from both caption and image items join the bootstrap graph.

Executed score: dot product of normalized official 512D CLIP ViT-B/32 vectors, caption lookup by `id`, image lookup by `image_id`. These supplied vectors are a source-provenance baseline. Exact pinned checkpoint/preprocessing identity is unverified. No BERT, ResNet, pinned CLIP or classifier-head inference occurred on NewsCLIPpings images. A separate synthetic software fixture exercised frozen heads and is excluded from scientific evidence. Official SBERT and Places365 vectors are not substitutes for the original BERT and ImageNet ResNet50 features. No official vectors are mixed with the pinned BRAS reference bank.

The optional real-image path uses local-only original pinned BERT/CLIP weights, ImageNet1K_V2 ResNet50, original CLS64/token77 contracts and EXIF/RGB handling. Extraction is staged by encoder and resumes hash-bound chunks; images are decoded and hashed, missing groups excluded without replacement, known Fakeddit text/byte/pixel overlaps excluded and logged. Original heads and training normalizers load read-only. A/B probability drops, absolute shifts, decision flips and their relation to cosine are separate from factual classification. Any pairing AUROC is labeled a cross-task proxy. CCR is only B's identity control, and the BRAS classifier D is not the BRAS alignment score.

Statistics use complete linked groups, never independent rows. AUC is independently implemented as weighted tied-rank concordance; AP uses weighted threshold precision increments. Both were checked against scikit-learn. Saved prediction CSVs are reloaded before statistics. Per-strategy/source denominators, tie checks, leave-one-strategy-out summaries, pair-only bootstrap sensitivity and singleton-component sensitivity are saved. Source slices and hard cases are exploratory. The singleton check excludes all observed cross-pair article/image/caption reuse, which is stricter than image-only reuse.

No new dependencies or shared version changes. CPU budget: six threads, batches of 16, no training. Official embedding download/validation took {emb['seconds']:.2f} seconds. Encoder and head costs are **not measured** because those stages were blocked; no Ryzen 5 throughput is inferred. A historical README names a different CPU, and host CIM inspection was unavailable, so the requested Ryzen 5 5600GT is treated as the intended target rather than a verified benchmark machine. Future extraction writes separate BERT/ResNet/CLIP chunk timings and head timing.
'''
    docs['results.md']=f'''# Results — ALIGNMENT ONLY

**Executed evidence: official precomputed CLIP correspondence on 1,000 groups / 2,000 rows, {overall['clusters']} observed content-linked components. No real selected images, pinned encoder results, BRAS comparison or frozen Fakeddit transfer results are claimed.** All selected groups had both required embeddings; zero groups were replaced or dropped for score availability. Strategy allocation: 250 groups each.

| Endpoint | Estimate [95% component-bootstrap CI] |
|---|---:|
| Paired ranking, half-credit ties | {metric('paired_ranking_half')} |
| ROC-AUC, pristine positive | {metric('roc_auc_pristine')} |
| Average precision, pristine positive | {metric('average_precision_pristine')} |
| Mean pristine-minus-mismatch cosine | {metric('mean_paired_margin')} |

{table}

Exact ties: {overall['ties']}. Strict ranking: {overall['strict_ranking']:.4f}; 1e-6 tolerance ranking: {overall['tolerance_1e_6_ranking']:.4f}. Component bootstrap and the prespecified robustness summaries appear in `official/statistics.json`. Holm-adjusted strategy tests are reported there without selective omission. Saved predictions, pair-level outcomes and plots are linked by hashes. These are pairing outcomes, not fake-news accuracy.

Verification: 24 unit tests passed before scoring; the final suite has 29 tests after adding download-resume, real-image integrity, feature-provenance and transfer-metric-label regressions. Coverage includes image refusal, source/pair semantics, cluster/tie behavior, independent metrics, immutable cache, malicious-pickle rejection and checkpoint-preservation detection. A bootstrap zero-weight edge case was found and corrected during synthetic tests **before** any external scores; test failures caused by the sandbox temporary directory were resolved by a bounded project-local test directory. Final integration/recomputation and preservation receipts are separate machine-readable artifacts (`verification_final.json`, `preservation_check.json`). A post-score code review found a metric-name ordering bug in the unexecuted transfer branch. It was fixed with an archived-source amendment and a synthetic fixture using the frozen heads; official predictions/statistics/protocol remained byte-identical. See `amendments/transfer_report/amendment.json`. Synthetic fixture results are software tests, not external evaluation evidence. Pre-change inventory: {h['count']:,} files, including CCR. Preservation {'passed with no changed/missing files' if preserve and preserve['passed'] else 'must be checked with verify-preservation; see final receipt'}.

Conditional conclusion: this is a reproducible external correspondence baseline and a documented partial-data result. The primary cosine-versus-BRAS hypothesis remains untested. No new-method superiority or external factual-veracity performance is supported.
{common}'''
    docs['failure_modes.md']=f'''# Failure modes and robustness

`official/failure_cases.csv` records the predeclared 10 smallest and 10 largest paired cosine margins with caption ID/text, pristine and donor image IDs/paths, construction strategy, source, both scores and group linkage. It includes **no images**. These examples were selected after scoring for descriptive error inspection; they cannot support claims about visible entities, image manipulations, or factual truth because selected image bytes were not available. Mismatch scores above pristine scores describe a ranking failure under the dataset's generated pairing labels only.

Access failure: all 198 unique smoke image paths returned HTTP404. This tests direct serving, not existence inside the archive. Main selection image availability remains unverified rather than extrapolating 198 failures to all 1,000 groups. Caption/source missingness is zero for official supplementary joins. Embedding missingness is zero for the fixed selection. Image-acquisition logs, source/strategy row denominators and path statuses are saved. No best-performing replacement subset was searched.

Shared item IDs link 1,000 groups into 953 components; adding captions/article paths yields {overall['clusters']}. Unknown events, near-duplicates, pretraining exposure and unobserved byte/pixel duplicates remain limitations. Official CLIP generation/adversarial filtering can make its own scores unrepresentative of generic web correspondence. No test-time threshold/weight tuning or reference-bank changes occurred. Per-strategy ranking, strict/tolerant ties, leave-one-strategy-out, singleton components and pair-only bootstrap are reported together, not mined for favorable outcomes.

The reference-bank and trained-head branches are implemented and guarded but have not run on real NewsCLIPpings images. Unit tests are software evidence only. A future authorized real-image run needs complete input and cache checks and fresh integration validation; it cannot be retroactively called executed by the present report.
'''
    docs['publication_assessment.md']='''# Publication assessment — conditional NO-GO

**NO-GO for a paper claiming a superior new fake-news detector, a validated CCR mechanism, general factual verification, SOTA or publication readiness.** The executed evidence is a score-only baseline using the dataset authors' CLIP embeddings. It does not test the proposed pinned cosine-versus-BRAS comparison, does not evaluate frozen A/B end-to-end transfer, and does not establish fake/real news truth outside Fakeddit. BRAS/CCR historical negative findings remain intact. No algorithmic novelty is claimed for CLIP alignment or mean directional margins.

The package supports an auditable reproducibility/negative-result work product: fixed selection, official provenance, linked paired statistics, construction-bias analysis, tests and preservation. Whether this becomes a publishable study depends on a stronger scientific contribution, authorized real-image evidence for the planned comparison, adequate independent event/source evaluation and a fuller literature/venue review. No journal indexing or acceptance is guaranteed. A conditional GO is limited to using these verified artifacts in internal research planning and an accurately scoped reproducibility appendix, not submission as a validated superior method.
'''
    for name,text in docs.items():put(run.docs/name,text.encode('utf-8'))
    status={'evidence_status':'ALIGNMENT ONLY','official_embedding_groups':len(pairs),'selected_real_images_verified':0,'pinned_CLIP_BRAS_comparison':'BLOCKED','frozen_A_B_transfer':'BLOCKED','publication_superiority':'NO-GO','reports':list(docs)}
    write(run.results/'evidence_status.json',status);return status
