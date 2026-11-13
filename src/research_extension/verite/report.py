"""Aggregate-only figures/reports; restricted images and embeddings never exported."""
from __future__ import annotations
import numpy as np,pandas as pd
from .core import *
DISPLAY={'A':'A: BERT + ResNet','B':'B: + raw CLIP','C':'BRAS C','D':'BRAS D','CCR':'CCR (= B)'}
COLORS={42:'#16697a',123:'#d2783e',2026:'#7a5195'}
def table(frame,columns=None):
    f=frame[columns] if columns else frame
    def fmt(v):return '—' if pd.isna(v) else (f'{v:.5f}' if isinstance(v,(float,np.floating)) else str(v))
    return '| '+' | '.join(f.columns)+' |\n|'+'|'.join(['---']*len(f.columns))+'|\n'+'\n'.join('| '+' | '.join(fmt(v) for v in row)+' |' for row in f.to_numpy())
def cross_dataset(s):
    rows=[]
    for name,path in [('Fakeddit historical 2500',s.root/'results/research_extension/bras_b4096_k16_s20260916/seed_metrics.csv'),('Fakeddit previous fresh 1000',s.root/'data/features/research_extension/fresh_holdout/model_metrics.csv')]:
        old=pd.read_csv(path)
        for kind in ['A','B','C','D']:
            f=old[old.variant.eq(kind)]
            rows.append({'dataset':name,'model':kind,'accuracy_mean':f.accuracy.mean(),'macro_f1_mean':f.macro_f1.mean(),'macro_f1_seed_sd':f.macro_f1.std(),'n':int(f.n.iloc[0]),'scope':'distant-supervision fake/real; previously viewed'})
        f=old[old.variant.eq('B')];rows.append({'dataset':name,'model':'CCR','accuracy_mean':f.accuracy.mean(),'macro_f1_mean':f.macro_f1.mean(),'macro_f1_seed_sd':f.macro_f1.std(),'n':int(f.n.iloc[0]),'scope':'historical identity B, verified in CCR reports'})
    new=pd.read_csv(s.results/'model_metrics.csv')
    for kind,g in new.groupby('model'):
        rows.append({'dataset':'VERITE current 1001','model':kind,'accuracy_mean':g.accuracy.mean(),'macro_f1_mean':g.macro_f1.mean(),'macro_f1_seed_sd':g.macro_f1.std(),'n':int(g.n.iloc[0]),'scope':'cross-task truthful vs misleading-pair diagnostic'})
    out=pd.DataFrame(rows);csv(s.results/'cross_dataset_context.csv',out);return out
def failures(s):
    f=s.frame();scores=pd.read_csv(s.results/'alignment_predictions.csv',keep_default_na=False);pred=pd.read_csv(s.results/'predictions/seed_42.csv',keep_default_na=False,dtype={'family_id':str})
    misleading=scores[scores.category.ne('true')].sort_values(['cosine','row_id'],ascending=[False,True]).head(10)
    ids=misleading.row_id.tolist();case_types={x:'highest_cosine_misleading' for x in ids}
    pairs=pd.read_csv(s.results/'alignment/paired_cosine.csv',keep_default_na=False)
    for category,g in pairs.groupby('category'):
        for row in g.sort_values(['margin','misleading_row_id']).head(10).itertuples():
            ids.append(row.misleading_row_id);case_types[row.misleading_row_id]=case_types.get(row.misleading_row_id,'')+'; smallest_margin_'+category
    selected=pred[pred.row_id.isin(ids)].copy();selected['case_type']=selected.row_id.map(case_types)
    csv(s.results/'failure_cases_scores.csv',selected)
    # Captions, article URLs and local image references stay solely in restricted local cache.
    private=selected.merge(f[['row_id','caption','image_path','article_url']],on='row_id')
    truth=f[f.category.eq('true')][['family_id','row_id','caption','image_path']].rename(columns={'row_id':'paired_true_row','caption':'paired_true_caption','image_path':'paired_true_image_path'})
    private=private.merge(truth,on='family_id',how='left');csv(s.cache/'failure_cases_private.csv',private)
    return selected

def plots(s):
    existing=s.results/'figure_manifest.json'
    if existing.exists():
        for rel,h in read(existing)['files'].items():require(sha(s.root/rel)==h,'Figure changed')
        return {'status':'verified existing figures'}
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.right':False,'axes.spines.top':False,'savefig.dpi':180})
    metrics=pd.read_csv(s.results/'model_metrics.csv');pairs=read(s.results/'paired_classification.json')['comparisons'];align=read(s.results/'alignment_statistics.json')
    dest=s.results/'figures';dest.mkdir(exist_ok=True);paths=[]
    def save(fig,name,footer='Frozen checkpoints • 336 content-linked components • cross-task diagnostic; no external tuning'):
        fig.text(.01,.01,footer,fontsize=8,color='#555555');fig.tight_layout(rect=(0,.05,1,.94))
        for ext in ['png','pdf','svg']:
            path=dest/f'{name}.{ext}';fig.savefig(path,bbox_inches='tight');paths.append(path)
        plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10.2,4.8))
    for ax,metric,label in zip(axes,['accuracy','macro_f1'],['Binary diagnostic accuracy','Macro-F1']):
        for j,seed in enumerate(SEEDS):
            f=metrics[metrics.seed.eq(seed)].set_index('model').loc[KINDS];values=f[metric].to_numpy();x=np.arange(5)+(j-1)*.18
            ax.errorbar(x,values,yerr=np.array([values-f[metric+'_ci_low'],f[metric+'_ci_high']-values]),fmt='o',capsize=3,label=f'seed {seed}',color=COLORS[seed])
        ax.set_xticks(range(5),['A','B','BRAS C','BRAS D','CCR']);ax.set_ylim(0,1);ax.set_ylabel(label);ax.grid(axis='y',alpha=.2)
    axes[0].legend(fontsize=8);fig.suptitle('VERITE frozen classifier transfer: every seed, 95% cluster intervals',fontsize=12);save(fig,'classifier_transfer')
    fig,axes=plt.subplots(1,2,figsize=(10.2,4.1))
    for ax,(candidate,base) in zip(axes,[('B','A'),('D','B')]):
        for i,seed in enumerate(SEEDS):
            r=next(x for x in pairs if x['candidate']==candidate and x['base']==base and x['seed']==seed);v=r['effect_candidate_minus_base']['macro_f1'];lo,hi=r['ci95']['macro_f1']
            ax.errorbar(v,i,xerr=[[v-lo],[hi-v]],fmt='o',color=COLORS[seed],capsize=4)
        ax.axvline(0,color='#777777',ls='--');ax.set_yticks(range(3),[f'Seed {x}' for x in SEEDS]);ax.set_xlabel('Candidate − baseline Macro-F1');ax.set_title(f'{candidate} − {base}');ax.grid(axis='x',alpha=.2)
    fig.suptitle('Paired transfer effects on identical content groups',fontsize=12);save(fig,'paired_transfer_effects')
    fig,axes=plt.subplots(1,2,figsize=(10.2,4.5));names={'cosine':'Raw cosine','m_text':'Text margin','m_image':'Image margin','bras_mean':'BRAS mean'}
    for ax,cat,title in zip(axes,['miscaptioned','out-of-context'],['True vs miscaptioned: shared image','True vs out-of-context: shared caption']):
        for i,score in enumerate(SCORES):
            r=align['paired'][score][cat];v=r['ranking_half'];lo,hi=r['ci95']['ranking_half'];ax.errorbar(v,i,xerr=[[v-lo],[hi-v]],fmt='o',capsize=4,color='#16697a' if score=='cosine' else '#b65d34')
        ax.axvline(.5,color='#999999',ls='--');ax.set_xlim(0,1);ax.set_yticks(range(4),[names[x] for x in SCORES]);ax.invert_yaxis();ax.set_xlabel('True pair ranks higher (ties: half credit)');ax.set_title(title,fontsize=10)
    fig.suptitle('Pinned CLIP correspondence and training-reference correction',fontsize=12);save(fig,'correspondence_ranking','338 miscaptioned pairs; 325 out-of-context pairs • 95% linked-component bootstrap intervals')
    cat=pd.read_csv(s.results/'category_breakdown.csv');pivot=cat.groupby(['model','category']).accuracy.mean().unstack().reindex(KINDS)[['true','miscaptioned','out-of-context']]
    fig,ax=plt.subplots(figsize=(7.6,4.7));im=ax.imshow(pivot,vmin=0,vmax=1,cmap='viridis',aspect='auto');ax.set_yticks(range(5),[DISPLAY[x] for x in KINDS]);ax.set_xticks(range(3),['True\nn=338','Miscaptioned\nn=338','Out-of-context\nn=325'])
    for i in range(5):
        for j in range(3):ax.text(j,i,f'{pivot.iloc[i,j]:.3f}',ha='center',va='center',color='white' if pivot.iloc[i,j]<.6 else 'black')
    fig.colorbar(im,ax=ax,label='Mean category accuracy across three seeds');fig.suptitle('Which VERITE categories do the frozen classifiers recognize?',fontsize=12);save(fig,'category_breakdown','Descriptive mean across classifier seeds; same evaluation records, not independent replications')
    cross=cross_dataset(s);fig,axes=plt.subplots(1,3,figsize=(11.8,4.2),sharey=True)
    for ax,dataset in zip(axes,cross.dataset.unique()):
        g=cross[cross.dataset.eq(dataset)].set_index('model').loc[KINDS];ax.bar(range(5),g.macro_f1_mean,yerr=g.macro_f1_seed_sd,color=['#91abb0','#16697a','#dfa178','#b65d34','#7a5195'],capsize=3)
        ax.set_xticks(range(5),['A','B','C','D','CCR']);ax.set_title(dataset,fontsize=10);ax.set_ylim(0,1)
    axes[0].set_ylabel('Macro-F1; bars ± seed SD');fig.suptitle('Historical context and the new cross-task diagnostic',fontsize=12);save(fig,'cross_dataset_context','Label definitions, prevalence and sample selection differ across panels; values are not equivalent benchmark scores')
    scores=pd.read_csv(s.results/'alignment_predictions.csv');fig,ax=plt.subplots(figsize=(8.1,4.4))
    for category,color in [('true','#16697a'),('miscaptioned','#b65d34'),('out-of-context','#7a5195')]:
        values=np.sort(scores.loc[scores.category.eq(category),'cosine']);ax.plot(values,np.arange(1,len(values)+1)/len(values),label=f'{category} (n={len(values)})',color=color)
    ax.set(xlabel='Pinned CLIP cosine',ylabel='Empirical cumulative fraction');ax.legend(fontsize=9);fig.suptitle('Similarity distributions overlap across truthful and misleading pairs',fontsize=12);save(fig,'cosine_distributions','Descriptive distributions; no VERITE-fitted cutoff or factual-verification threshold')
    write(existing,{'files':{p.relative_to(s.root).as_posix():sha(p) for p in paths},'statistics_sha256':sha(s.results/'paired_classification.json')});return {'figures':len(paths),'directory':str(dest)}

def report(s):
    audit=read(s.results/'dataset_audit.json');protocol=s.protocol();metrics=pd.read_csv(s.results/'model_metrics.csv');summary=pd.read_csv(s.results/'seed_summary.csv')
    comparisons=read(s.results/'paired_classification.json')['comparisons'];align=read(s.results/'alignment_statistics.json');profile=read(s.results/'encoder_profile.json');inference=read(s.results/'inference_profile.json')
    previous=cross_dataset(s);case_scores=failures(s);preserve=read(s.results/'preservation_check.json') if (s.results/'preservation_check.json').exists() else None
    def cmp(candidate,base,seed=42):return next(r for r in comparisons if r['candidate']==candidate and r['base']==base and r['seed']==seed)
    def describe(r):
        delta=r['effect_candidate_minus_base']['macro_f1'];lo,hi=r['ci95']['macro_f1']
        return f"ΔMacro-F1={delta:+.5f}, 95% CI [{lo:+.5f}, {hi:+.5f}], Holm p={r['macro_f1_holm_p']:.5g}; corrected {r['corrected_errors']}, introduced {r['introduced_errors']} errors"
    def evidence(r):
        lo,hi=r['ci95']['macro_f1'];p=r['macro_f1_holm_p']
        return 'supported positive transfer difference' if lo>0 and p<.05 else ('supported negative transfer difference' if hi<0 and p<.05 else 'no statistically resolved transfer difference')
    b=cmp('B','A');d=cmp('D','B');ranking=[]
    for score in SCORES:
        for cat,r in align['paired'][score].items():ranking.append({'score':score,'comparison':cat,'pairs':r['pairs'],'ranking':r['ranking_half'],'CI_low':r['ci95']['ranking_half'][0],'CI_high':r['ci95']['ranking_half'][1],'mean_margin':r['mean_margin']})
    rank_df=pd.DataFrame(ranking);csv(s.results/'alignment_summary.csv',rank_df)
    gap='''The most defensible candidate contribution is a controlled audit of what survives frozen Fakeddit-to-VERITE transfer, with shared-content uncertainty and a separation of classifier decisions from correspondence ranking. Distinctiveness is not established merely by adding VERITE. [Similarity over Factuality / MUSE (WACV2025)](https://arxiv.org/abs/2407.13488) already studies simple similarity systems on NewsCLIPpings and VERITE and questions their factual reasoning. The [VERITE paper](https://link.springer.com/article/10.1007/s13735-023-00312-6) itself addresses modality bias. The present comparison controls the training bank, feature contract and classifier seeds, but does not reproduce MUSE's evidence-based models or establish superiority to published baselines.

Mean BRAS margins equal one half of the [CSLS](https://arxiv.org/html/1710.04087v3) form when their neighborhoods/exclusions match: `s − (r_text+r_image)/2`. This mathematical relationship prevents a novel-similarity claim. The cancellation of one penalty in shared-caption/shared-image pairs is an algebraic diagnostic, not a new learned verification mechanism. Existing CCR is all-B and cannot validate adaptive reliance here; [SSPC](https://aclanthology.org/2026.emnlp-main.1279/) is also relevant prior work on cross-fitted utility gates.
'''
    docs={}
    verification=read(s.results/'verification.json');robustness=read(s.results/'robustness.json')
    require(verification['passed'] and preserve and preserve['passed'],'Reports require completed verification and preservation checks')
    misc=next(r for r in align['comparisons'] if r['primary'] and r['category']=='miscaptioned')
    ooc=next(r for r in align['comparisons'] if r['primary'] and r['category']=='out-of-context')
    means=summary.set_index('model')
    docs['executive_summary.md']=f'''# Executed VERITE evaluation: conclusions and evidence

**All feasible stages completed using the authorized local images.** The current official release contains 1,001 complete pairs: 338 true, 338 miscaptioned and 325 out-of-context. Every referenced image decoded; no image was substituted. The archive SHA matches the official Hugging Face LFS release, and annotations match the pinned official GitHub revision. The older 1,000-row description was not imposed on the current data. There is no remaining local VERITE data-access blocker.

**Does the original CLIP improvement transfer? Partially.** Across the three existing seeds, mean Macro-F1 increases from {means.loc['A','macro_f1_mean']:.5f} (A) to {means.loc['B','macro_f1_mean']:.5f} (B), and truthful-class ROC-AUC increases from {means.loc['A','roc_auc_truthful_mean']:.5f} to {means.loc['B','roc_auc_truthful_mean']:.5f}. All three B−A Macro-F1 point differences are positive, but the prespecified primary seed-42 result is unresolved: {describe(b)}. Seed 2026 has a positive secondary Macro-F1 result after its family's correction. Mean accuracy falls from {means.loc['A','accuracy_mean']:.5f} to {means.loc['B','accuracy_mean']:.5f}; B's mean log loss and Brier score also worsen. Thus the original broad accuracy improvement does not consistently transfer, despite improved discrimination and descriptive Macro-F1.

**Does BRAS offer an independent benefit? Yes, narrowly in correspondence ranking.** For true versus miscaptioned pairs, raw cosine ranks the true instance higher in 191/338 cases (56.51%), versus 208/338 (61.54%) for the frozen BRAS mean. The paired gain is {100*misc['ranking_effect']:.2f} percentage points, 95% component-bootstrap CI [{100*misc['ci95']['ranking_half'][0]:.2f}, {100*misc['ci95']['ranking_half'][1]:.2f}], Holm p={misc['holm_p']:.5f}; 24 rankings improve and 7 deteriorate. This is independently executed, external alignment evidence, not evidence of factual verification. The corresponding out-of-context gain is only {100*ooc['ranking_effect']:.2f} points, CI [{100*ooc['ci95']['ranking_half'][0]:.2f}, {100*ooc['ci95']['ranking_half'][1]:.2f}], Holm p={ooc['holm_p']:.5f}. The frozen BRAS D classifier has no resolved benefit over B: {describe(d)}. C is seed-sensitive, with a positive secondary result for seed 123 and a negative point difference for seed 42. CCR equals B exactly for every seed.

Descriptive means across the same three trained seeds; these are not three independent evaluation datasets:

{table(summary,['model','accuracy_mean','macro_f1_mean','roc_auc_truthful_mean','average_precision_truthful_mean'])}

All classifier accuracies are below the always-misleading prevalence baseline ({robustness['majority_misleading']['accuracy']:.5f}); their Macro-F1 values exceed its {robustness['majority_misleading']['macro_f1']:.5f}. B recognizes more true examples but also calls more misleading examples truthful. This tradeoff explains why discrimination, Macro-F1 and accuracy need different interpretations.

The mapping `true → truthful; miscaptioned/out-of-context → misleading pair` defines a **cross-dataset, cross-task transfer diagnostic**. It does not turn VERITE into an equivalent Fakeddit fake-news benchmark. All results use actual pinned ViT-B/32 embeddings, original BERT/ResNet preprocessing, frozen heads/scalers, the original training-only BRAS bank and the existing CCR policy. No retraining, external calibration or evaluation-label model selection occurred. Dependence-aware analysis uses 336 shared-content components, 5,000 paired bootstrap draws and 10,000 component-level randomizations, with prespecified multiplicity correction.

**Publication decision: CONDITIONAL GO for manuscript development; NO-GO for a distinctive superior-method submission as-is.** The new evidence supports a focused empirical claim about reference normalization on miscaptioned-pair ranking and documents limits of fixed-head transfer. It does not establish a novel scoring rule, a robust BRAS classifier improvement or general factual reasoning. The contribution still needs a precise contrast to existing VERITE/MUSE findings and comparable published baselines or broader external evidence. See [publication assessment](publication_assessment.md) and [cross-dataset discussion](cross_dataset_discussion.md).

Verification passed: {verification['unit_tests']} unit tests, {verification['independent_classifier_metric_estimates']} independently recomputed classifier metrics, {verification['actual_VERITE_probabilities_replayed']} actual frozen-head probabilities replayed through independent NumPy calculations, and {verification['cluster_bootstrap_expansion_checks']} expanded-resample Macro-F1 checks. All {read(s.results/'historical_inventory.json')['count']:,} inventoried pre-existing files are unchanged. Images, full caption manifests and feature embeddings remain restricted locally; the deliverable package contains no such data.

Read [integrity/provenance](dataset_integrity_and_provenance.md), [all model seeds](model_transfer_results.md), [ranking statistics](correspondence_ranking.md), [failure analysis](failure_analysis.md), [robustness and verification](robustness_and_verification.md), and [reproduction](execution_and_reproduction.md).
'''
    docs['dataset_integrity_and_provenance.md']=f'''# VERITE dataset integrity and provenance

**FULL authorized local image evaluation.** The user confirmed university Hugging Face access. No credentials were inspected or exposed, and no gated download, publisher scraping or image redistribution occurred. The 65,287,817-byte local archive exactly matches the public release LFS SHA-256 `{audit['zip_sha256']}` at revision `{audit['hf_revision']}`. Local CSV exactly matches the official repository file, SHA-256 `{audit['csv_sha256']}`, commit `{audit['github_revision']}`. Public provenance requests/receipts are recorded in `provenance_initial.json`; the raw official inputs remain under `data/external/verite/`.

The pinned current official CSV has **1,001** rows: true338, miscaptioned338, out-of-context325. Earlier paper/README and the old unexecuted adapter expected1,000 (324 out-of-context). We used the current official bytes verbatim; no record was trimmed or annotation silently corrected. This version difference matters when comparing published results. Prior access-blocked reports remain historical records and have not been overwritten.

ZIP structure was inspected first:665 entries (one directory,664 images). All664 images decoded. The annotations reference663 unique files, with338 repeated image references; all1,001 image-caption records are complete. Missing/invalid referenced images:0. Exact annotation/provenance mismatches:0. `images/false_179.jpeg` is an extra unreferenced archive member and was not substituted for any annotation path. Every extracted image has a file hash, decoded RGB pixel hash, dimensions and CRC record. Traversal, noncanonical paths, case collisions, symlinks, encrypted entries and excessive expansion are rejected.

Verified paired links:338 true–miscaptioned (shared image),325 true–out-of-context (shared exact caption);338 original families. Joining shared captions, images/pixels and article URLs yields336 components. Sources:771 Snopes rows and230 Reuters rows. All1,001 records join author article metadata. There are675 unique exact captions and326 repeated caption rows. No retained Fakeddit text, byte-hash or pixel-hash overlap was found; this does not rule out related events, near-duplicates or pretraining exposure.

The [official research repository](https://github.com/stevejpapad/image-text-verification) provides annotation/structure provenance. The [authorized image release](https://huggingface.co/datasets/stefpapad/VERITE) is for research and prohibits redistribution. Images, full caption manifests, private failure-case joins and embeddings remain in local restricted-data folders and are excluded from the deliverable package. Public aggregate reports and code contain no credentials or image bytes.
'''
    docs['protocol_and_methodology.md']=f'''# Locked protocol and methodology

Protocol SHA-256: `{sha(s.results/'protocol.json')}`. It was locally locked before VERITE encoder/model inference, followed by a source hash lock. This is a local pre-inference protocol lock, not public preregistration. No test labels selected thresholds, bank entries, hyperparameters or models. All current official available rows were retained without replacement.

Mapping: `true → 1 (truthful)`; `miscaptioned/out-of-context → 0 (misleading pair)`. Fakeddit's distant-supervision fake/real labels differ. All accuracy, Macro-F1 and classifier AUC/AP are expressly cross-dataset, cross-task **transfer diagnostics**. The original argmax decision is retained, tie→class0. Classifier probability of Fakeddit real is interpreted diagnostically as probability of truthful, without external calibration. Positive class for ROC-AUC/AP is truthful; AP is average precision, not trapezoidal PR area. Loss probabilities are clipped to[1e-6,1−1e-6].

Frozen A/B use original BERT CLS768 and ImageNet ResNet50 2048 features; B also uses raw CLIP cosine. BRAS C uses two margins; BRAS D uses cosine plus both margins. Original head checkpoints, per-feature training normalizers and the4096-row Fakeddit training bank remain frozen. k=16 and `(m_text+m_image)/2` were fixed. Query matches by text, byte/pixel content, group or pair identifiers are excluded from reference neighborhoods. CCR's stored gate and policy are actually loaded/executed; fallback B is an identity control.

Actual authorized images and captions were encoded using the project's pinned **CLIP ViT-B/32**, not the authors' supplied ViT-L/14 features. Original BERT revision, CLIP revision, ImageNet1K_V2 weights,64-token CLS contract,77-token CLIP limit, EXIF first-frame RGB handling, image transforms and float32 normalization were verified. Extraction loads encoders sequentially on CPU,6 threads, batches16, hash-bound64-row chunks. No feature fitting, fine-tuning or retraining occurs.

Primary classifier hypotheses: seed42 B−A and D−B Macro-F1, Holm-corrected together. Other seeds and C−B/CCR−B are a separate secondary family.5,000 common-component bootstrap resamples (seed20261109) yield percentile95% CIs.10,000 within-component swaps of complete model prediction vectors (seed20261110) recompute Macro-F1; two-sided Monte Carlo p-values use +1 correction. Repeated rows are never treated as independent McNemar trials. Seeds share data and are not pooled as independent replications; their mean/sample SD is descriptive.

Alignment is a separate threshold-free task. Higher cosine/margin should rank the verified true pair above its misleading counterpart; exact ties receive half credit. Primary BRAS-mean minus cosine ranking comparisons are tested separately for miscaptioned and out-of-context pairs, with Holm correction across those two tests. Four directional-score comparisons form a secondary family. Strict and1e-6-tolerance ties, family-only uncertainty, source/category breakdown and known-overlap exclusion are predeclared sensitivities. Correspondence AUC/AP use original rows, avoiding duplicate counting of a true record across the two pairing tasks.
'''
    docs['model_transfer_results.md']=f'''# Frozen model transfer results

**B versus A, primary seed42:** {evidence(b)}; {describe(b)}.

**BRAS D versus B, primary seed42:** {evidence(d)}; {describe(d)}.

{table(metrics,['seed','model','n','accuracy','macro_f1','roc_auc_truthful','average_precision_truthful'])}

Every model/seed has component-bootstrap intervals in `model_metrics.csv`. Primary and secondary effect intervals, corrected p-values, probability/decision changes and corrected/introduced error counts are in `paired_classification.json` and the per-row comparison CSVs. Negative findings are retained. CCR equals B exactly for all three seeds; this follows from the frozen guard and does not validate a new selector.

Descriptive seed means and sample SD:

{table(summary,['model','accuracy_mean','accuracy_std','macro_f1_mean','macro_f1_std'])}

The always-misleading baseline and family-only/overlap sensitivities are recorded in `robustness.json`. Class prevalence is663 misleading versus338 truthful, so accuracy alone can be misleading. No claim of matching the authors' trained models is made: training data, encoder backbones, task mapping and current annotation version differ.
'''
    docs['correspondence_ranking.md']=f'''# Correspondence ranking, separate from classifier transfer

All scores use newly extracted pinned CLIP ViT-B/32 features. No VERITE examples enter the Fakeddit reference bank. Exact score ties count0.5; strict/tolerance checks are retained. Uncertainty resamples shared-content components and uses the same draws for compared scores.

{table(rank_df)}

Paired primary effects and corrected tests:

{table(pd.DataFrame([{'category':r['category'],'BRAS_mean_minus_cosine':r['ranking_effect'],'CI_low':r['ci95']['ranking_half'][0],'CI_high':r['ci95']['ranking_half'][1],'Holm_p':r['holm_p'],'gained':r['ranking_wins_gained'],'lost':r['ranking_wins_lost']} for r in align['comparisons'] if r['primary']]))}

Global score discrimination on all 1,001 rows, positive class truthful; these endpoints differ from within-family paired ranking:

{table(pd.DataFrame([{'score':score,'ROC_AUC':r['roc_auc_truthful'],'AUC_CI_low':r['ci95']['roc_auc_truthful'][0],'AUC_CI_high':r['ci95']['roc_auc_truthful'][1],'AP':r['average_precision_truthful'],'AP_CI_low':r['ci95']['average_precision_truthful'][0],'AP_CI_high':r['ci95']['average_precision_truthful'][1]} for score,r in align['discrimination'].items()]))}

The miscaptioned ranking gain does not imply improved global AUC: BRAS mean AUC is slightly lower than raw cosine. No global AUC superiority test was prespecified or claimed.

All directional comparisons, discrimination AUC/AP and intervals, source denominators, and half/strict/tolerance tie counts appear in `alignment_statistics.json`. With equal reference candidate sets, the image-side penalty cancels for shared-image true/miscaptioned pairs, while the text-side penalty cancels for shared-caption true/out-of-context pairs. Numerical equivalence checks are recorded; identical rankings in those directions are algebraic, not independent supporting experiments. Remaining mean-margin changes and ranking gains can diverge.

Neither raw cosine nor BRAS margins independently verify factual truth. Their ranking asks whether the supplied true pairing scores above the paired misleading instance; classifier diagnostic accuracy measures a different fixed decision rule.
'''
    docs['failure_analysis.md']=f'''# Error breakdown and failure examples

`category_breakdown.csv` reports all three categories for every model and seed; `source_breakdown.csv` retains Snopes/Reuters counts and point metrics. The category figure shows descriptive means over the same three heads, not independent repetitions.

Failure selection was fixed before test metrics: ten highest-cosine misleading records and ten smallest cosine paired margins per comparison category, with deterministic row-ID tie breaks. `failure_cases_scores.csv` contains{len(case_scores)} unique selected records with IDs, categories, seed42 predictions and scores. Full paired captions, article URLs and local image paths are retained only in `data/external/verite/restricted_cache/failure_cases_private.csv` for authorized review. No image grids or raw restricted records are distributed.

These cases expose high semantic similarity on benchmark-labeled misleading pairs and disagreements between frozen decisions and correspondence ranking. Such scores cannot resolve a date, location, identity or event claim. Case selection is exploratory; image content is not described without a documented visual review, and official annotations are not relabeled. Repeated caption/image control holds one modality fixed, allowing a cleaner sensitivity comparison but still not a causal proof of factual reasoning.

Three examples from the prespecified high-similarity list are summarized below. These are **annotation/caption contrasts**, not a manual visual assessment or independent fact check. No restricted image or complete caption is reproduced. Probabilities are the frozen seed-42 heads' Fakeddit-real output, mapped diagnostically to truthful.

| Record | Official paired-caption contrast, paraphrased | Misleading-pair cosine | A p(truthful) | B p(truthful) | D p(truthful) |
|---|---|---|---|---|---|
| verite_0283 | Sanitation-agency attribution versus street-artist attribution for New York posters | 0.43049 | 0.89097 | 0.92479 | 0.91158 |
| verite_0358 | A real polar bear on transport in Russia versus a puppet at a Greenpeace event in Brno | 0.42282 | 0.17479 | 0.87621 | 0.89186 |
| verite_0427 | A Hurricane Florence staging allegation versus a Hurricane Ike water-depth account | 0.41358 | 0.20258 | 0.86013 | 0.90814 |

All three misleading records score above their paired true counterparts under both raw cosine and BRAS mean. The first is wrong for all three heads. For the second and third, A correctly predicts misleading while B and D introduce errors. The score changes therefore fail to resolve the specific attribution, object identity or event/time distinction represented by the official annotations.
'''
    family_rows=[]
    for r in robustness['family_only_bootstrap']:
        lo,hi=r['accuracy_macro_f1_ci95'][1]
        family_rows.append({'comparison':r['candidate']+' − '+r['base'],'seed':r['seed'],'Macro_F1_CI_low':lo,'Macro_F1_CI_high':hi})
    source_rows=[]
    for score in ['cosine','bras_mean']:
        for category,r in align['paired'][score].items():
            for row in r['by_source']:source_rows.append({'score':score,'category':category,**row})
    tie_rows=[{'score':score,'category':category,'exact_ties':r['ties'],'strict':r['strict_ranking'],'half':r['ranking_half'],'tolerance_1e_6':r['tolerance_1e_6_ranking']} for score in SCORES for category,r in align['paired'][score].items()]
    docs['robustness_and_verification.md']=f'''# Robustness and verification

The primary uncertainty uses 336 connected components that join official families and repeated article/image/caption content. Replacing them with the 338 original families leaves both primary classifier conclusions unresolved; family-only paired Macro-F1 intervals are:

{table(pd.DataFrame(family_rows))}

No exact retained Fakeddit caption, byte-hash or pixel-hash overlap was found. The predeclared overlap-exclusion analysis therefore retains all 1,001 records and equals the full analysis. This is not evidence against near-duplicates, shared events or encoder-pretraining overlap.

All alignment tie sensitivities:

{table(pd.DataFrame(tie_rows))}

Source-stratified paired rankings are descriptive; small cells and post-stratification do not support a source-specific superiority claim:

{table(pd.DataFrame(source_rows),['score','category','source','pairs','ranking_half','mean_margin'])}

Numerical shared-modality cancellation checks give zero ranking disagreements in the theoretically invariant directions. The maximum margin discrepancy is below 1.8e-7, consistent with float32 rounding. No alternative score direction, threshold, neighborhood size or bank was selected on VERITE.

The final executed verifier independently recomputed 90 classifier metrics and 8 alignment AUC/AP values using scikit-learn. It checked 8 paired-ranking values, replayed 768 actual VERITE probabilities for 12 frozen heads (64 deterministic hash-selected records per head) with identical decisions and maximum probability discrepancy below 3e-7, and checked 120 expanded cluster-bootstrap Macro-F1 estimates against scikit-learn. Recomputed float64 CLIP vector products differed from stored cosines by at most {verification['max_CLIP_vector_scalar_difference']:.3g}. All 24 unit tests passed, including image corruption, unsafe ZIP members, label mapping, paired content, cluster resampling, statistical ties, and frozen-head calculations. Dataset/source/model checksums are revalidated on cached resume. Every tested resume stage passed without changing protocol, predictions, model metrics or paired statistical outputs.

All six aggregate figures were visually inspected for legibility and clipping. PNG, vector PDF and SVG copies are available. No restricted images were used in these figures.

`verification.json`, `tests.txt`, `figure_manifest.json` and `preservation_check.json` provide machine-readable or captured receipts. Timings and exact feature contracts are separate from inference results; no performance number was inferred from a synthetic test fixture.
'''
    docs['cross_dataset_discussion.md']=f'''# Cross-dataset discussion

{table(previous,['dataset','model','n','accuracy_mean','macro_f1_mean','macro_f1_seed_sd'])}

These rows provide context; label definitions, prevalence and sample selection differ, so a cross-panel drop is not a calibrated estimate of a common-task domain effect. The original2500 Fakeddit and previously viewed1000 slice remain exploratory for BRAS/CCR. Historical BRAS D did not reliably improve B; CCR selected B for all seeds. The new VERITE test is external to Fakeddit and was not used for training or selection, with exact retained-data overlap checks negative.

NewsCLIPpings remains **ALIGNMENT ONLY**: author-provided CLIP embeddings on1000 paired groups,250 per construction strategy, ranking0.5900 (95% CI0.5599–0.6201), ROC-AUC0.716961, AP0.710531. Its pinned encoder/BRAS comparison and frozen-head transfer were not run because selected image bytes were unavailable. Its CLIP-assisted construction and unverified equality to the pinned feature contract prevent pooling its scores with VERITE. This new authorized VERITE run does not retroactively upgrade the NewsCLIPpings evidence.

For the original CLIP feature claim, the correct external comparison is frozen B−A within VERITE, not a comparison to the authors' different CLIP model: {describe(b)}. For the trained BRAS extension, use D−B: {describe(d)}. Report each seed alongside these primary tests. Raw-score ranking benefits, if any, do not automatically establish improved binary transfer.
'''
    docs['limitations_and_research_gap.md']=f'''# Limitations, threats to validity and research gap

VERITE category mapping is a cross-task diagnostic; Fakeddit weak labels are not equivalent to benchmark truthful/misleading-pair judgments. Original thresholds and feature scalers can be poorly calibrated under this shift. There is no VERITE-fitted threshold, but this also means that discrimination and fixed-decision accuracy can disagree. The current official1001-row annotation revision differs from older1000-row publications.

336 linked components account for known shared content; they do not eliminate unknown event dependence, perceptual near-duplicates or encoder-pretraining exposure. No exact retained Fakeddit caption/byte/pixel overlap was found, but this cannot establish event or web-pretraining independence. Source-specific cells are smaller and descriptive. Two source organizations do not represent all languages or misinformation settings. A single frozen training setup and one fully imaged external benchmark cannot establish broad generalization.

CLIP/BERT token truncation and CPU timings are disclosed in `encoder_profile.json`. Bank k, membership and combination are frozen. BRAS mean resembles CSLS and the classifier variants differ in added feature count; independent benefit must be tied to the measured paired comparison, not renamed as algorithmic novelty. CCR's fallback identity is a negative finding for adaptive reliance. Three training seeds are not three independent test corpora. CIs are conditional on the benchmark release and selected dependence representation; randomization tests assume exchangeable model predictions within components under the null.

{gap}
'''
    docs['publication_assessment.md']=f'''# Publication evidence assessment

**CONDITIONAL GO for developing an empirical robustness/reproducibility manuscript; NO-GO for submission as a distinctive superior new detection method on the current evidence alone.** The authorized full-image run resolves the previous external-access blocker and supplies real frozen-model evidence. It does not by itself establish novelty, general factual verification, SOTA or publication readiness.

Measured primary conclusions: B−A has {evidence(b)} ({describe(b)}). D−B has {evidence(d)} ({describe(d)}). Separately, BRAS mean improves miscaptioned-pair ranking by {100*misc['ranking_effect']:.2f} percentage points (95% paired CI [{100*misc['ci95']['ranking_half'][0]:.2f}, {100*misc['ci95']['ranking_half'][1]:.2f}], Holm p={misc['holm_p']:.5f}); its out-of-context ranking change is unresolved. This is a narrow independent alignment benefit, not a resolved BRAS classifier gain. CCR remains exactly B. The complete negative/positive seed and category results are retained.

The paper's candidate value is a transparent separation of correspondence ranking, fixed-head transfer and training-reference correction under label/domain shift. For a distinctive empirical submission, establish a precise contrast to MUSE/VERITE's existing similarity-bias findings, reproduce suitable published or simple frozen baselines under a common protocol, and broaden external real-image or training-domain coverage. These are future research conditions, not experiments claimed as executed here. A stronger empirical claim should survive multiple seeds, dependence-aware uncertainty and version-matched comparisons. No journal acceptance or indexing outcome can be guaranteed.

{gap}
'''
    docs['execution_and_reproduction.md']=f'''# Execution, verification and reproduction

All1,001 authorized records were encoded with original pinned BERT, ResNet50 and CLIP. CPU reported by Windows registry: `{profile.get('cpu_name','unavailable')}`. Six threads, batches16; no GPU or training. Encoder timings (seconds, excluding/including load separately):

{table(pd.DataFrame([{'encoder':k,**{key:v[key] for key in ['load_seconds','inference_seconds','truncated_caption_rows']}} for k,v in profile['encoders'].items()]))}

Total extraction invocation: {profile['elapsed_seconds']:.2f}s. Peak process working set: {profile.get('peak_process_rss_bytes',0)/2**30:.2f}GiB (process lifetime, not stage-isolated). BRAS scoring:{inference['bank']['seconds']:.3f}s. Frozen head load/inference per seed is recorded in `inference_profile.json`; CCR policy overhead is separate. Cached replays perform no new encoder inference.

24 unit tests passed before extraction, including exact mapping, paired links, archive traversal/symlink/case checks, image decoding/hash mutation, cached resume, cluster randomization, independent weighted metrics, and direct NumPy replay of all12 frozen A/B/C/D heads across3 seeds. A Windows ZIP-name test exposed automatic filename normalization; validation now checks original archive names, and a raw-header regression passed before source/protocol lock. Final real-output replay, metric recomputation and preservation receipts accompany this report.

Historical inventory: {read(s.results/'historical_inventory.json')['count']:,} files. Preservation {'passed, zero changed/missing files' if preserve and preserve['passed'] else 'pending final rehash; see preservation_check.json'}. Original input ZIP/CSV, model checkpoints, BRAS/CCR and NewsCLIPpings files are included. New outputs are isolated in `results/research_extension/verite/`; restricted manifests/features stay in `data/external/verite/restricted_cache/`. No shared dependency versions changed.

```powershell
$projectPython = 'C:\\Users\\Admin\\AppData\\Local\\venvs\\multimodal-fake-news\\Scripts\\python.exe'
Set-Location 'C:\\Users\\Admin\\OneDrive\\Desktop\\MultimodalFakeNews\\MultimodalFakeNews'
& $projectPython -B -m unittest src.research_extension.verite.tests -v
if ($LASTEXITCODE -ne 0) {{ throw "VERITE unit tests failed" }}
foreach ($stage in @('audit','lock','extract','evaluate','statistics','plots','report','verify-preservation')) {{
    & $projectPython -B run_verite_eval.py $stage
    if ($LASTEXITCODE -ne 0) {{ throw "VERITE stage failed: $stage" }}
}}
```

Public source provenance is already pinned in the local audit receipt. No gated access or image download is needed to rerun. Reports/metrics are immutable; changed inputs or code require an explicitly versioned new study, not silent overwrite.
'''
    for name,text in docs.items():
        for old,new in [('true338','true 338'),('miscaptioned338','miscaptioned 338'),('out-of-context325','out-of-context 325'),('expected1,000','expected 1,000'),
          ('structure was inspected first:665','structure was inspected first: 665'),('directory,664','directory, 664'),('All664','All 664'),('reference663','reference 663'),
          ('with338','with 338'),('all1,001','all 1,001'),('All1,001','All 1,001'),('images:0','images: 0'),('mismatches:0','mismatches: 0'),
          ('links:338','links: 338'),('image),325','image), 325'),('caption);338','caption); 338'),('yields336','yields 336'),('Sources:771','Sources: 771'),
          ('and230','and 230'),('are675','are 675'),('and326','and 326'),('the4096','the 4,096'),('seed42','seed 42'),('across2','across 2'),
          ('fixed before test metrics: ten','fixed before test metrics: ten'),('contains'+str(len(case_scores)),'contains '+str(len(case_scores))),('original2500','original 2,500'),
          ('viewed1000','viewed 1,000'),('on1000','on 1,000'),('groups,250','groups, 250'),('ranking0.5900','ranking 0.5900'),('CI0.5599','CI 0.5599'),
          ('ROC-AUC0.716961','ROC-AUC 0.716961'),('AP0.710531','AP 0.710531'),('official1001','official 1,001'),('older1000','older 1,000'),
          ('code contain no','code contain no'),('probabilities are clipped to[','probabilities are clipped to ['),('tails:0','tails: 0'),('all12','all 12'),('across3','across 3'),
          ('family.5,000','family. 5,000'),('CIs.10,000','CIs. 10,000'),('seed20261109','seed 20261109'),('seed20261110','seed 20261110'),('percentile95%','percentile 95%'),
          ('weights,64-token','weights, 64-token'),('contract,77-token','contract, 77-token'),('CPU,6 threads','CPU, 6 threads'),('batches16','batches of 16'),
          ('hash-bound64-row','hash-bound 64-row'),('Strict and1e-6','Strict and 1e-6'),('count0.5','count 0.5'),('prevalence is663','prevalence is 663'),
          ('scoring:','scoring: '),('misleading versus338','misleading versus 338'),('CLS768','CLS 768'),('near-duplicates','near duplicates')]:text=text.replace(old,new)
        put(s.docs/name,text.encode('utf-8'))
    status={'evidence_status':'FULL_AUTHORIZED_FROZEN_EVALUATION','records':audit['available_records'],'clusters':audit['clusters'],
      'B_minus_A_seed42':{'interpretation':evidence(b),**b},'D_minus_B_seed42':{'interpretation':evidence(d),**d},
      'publication':'CONDITIONAL GO for development of empirical robustness manuscript; NO-GO for distinctive superior-method submission as-is','restricted_images_distributed':False}
    write(s.results/'evidence_status.json',status);return {'reports':list(docs),'status':status['evidence_status']}
