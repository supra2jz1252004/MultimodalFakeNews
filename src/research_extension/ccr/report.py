"""Figures and publication assessment from frozen, saved prediction analyses."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .core import *
from .evaluate import read_prediction

LABELS=['True','Satire / parody','False connection','Imposter content','Manipulated content','Misleading content']
COLORS={'A':'#788492','B':'#20749b','CCR':'#d06635','ridge_candidate':'#7c4eaa','B_temperature':'#41976f','tree':'#ad6787'}
def table(df):
    cols=list(df.columns);lines=['| '+' | '.join(cols)+' |','|'+'|'.join(['---']*len(cols))+'|']
    for row in df.itertuples(index=False,name=None):
        lines.append('| '+' | '.join(f'{x:.6f}' if isinstance(x,(float,np.floating)) else str(x) for x in row)+' |')
    return '\n'.join(lines)
def figure_save(fig,dest,name):
    dest.mkdir(parents=True,exist_ok=True)
    fig.savefig(dest/f'{name}.png',dpi=180,bbox_inches='tight');fig.savefig(dest/f'{name}.pdf',bbox_inches='tight');plt.close(fig)
def report(s):
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.titleweight':'bold'})
    dest=s.path('figures');sets=['original_test','previous_fresh_test'];titles=['Historical Fakeddit (n=2,500)','Previously viewed Fakeddit (n=1,000)']
    chosen=['A','B','B_temperature','ridge_candidate','CCR'];seed_metrics={};pairs={};chunks=[]
    for name,title in zip(sets,titles):
        r=s.path(f'analysis/{name}');df=pd.read_csv(r/'seed_metrics.csv');seed_metrics[name]=df;pairs[name]=read_json(r/'paired_statistics.json')['comparisons']
        comp=pd.read_csv(r/'model_comparison.csv')
        chunks.append(f'## {title}\n\nExploratory diagnostics: these outcomes cannot confirm a method designed after earlier test inspection.\n\n'+table(comp[['variant','accuracy_mean','macro_f1_mean','log_loss_mean','brier_mean','ece_10_mean','b_fraction_mean']]))
    fig,axs=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
    for ax,name,title in zip(axs,sets,titles):
        for j,k in enumerate(chosen):
            v=seed_metrics[name].query('variant == @k').log_loss
            ax.bar(j,v.mean(),yerr=v.std(ddof=1),capsize=4,color=COLORS[k],width=.7)
            ax.scatter(np.full(len(v),j),v,color='black',s=13,zorder=3)
        ax.set_xticks(range(len(chosen)),['A','B','B + temp.','Ridge candidate','CCR'],rotation=20,ha='right');ax.set_ylabel('Log-loss (lower is better)');ax.set_title(title)
    fig.suptitle('Exploratory loss comparison · points are classifier seeds; bars ± seed SD',fontsize=12)
    figure_save(fig,dest,'loss_comparison')
    fig,axs=plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
    for ax,name,title in zip(axs,sets,titles):
        r=pd.read_csv(s.path(f'analysis/{name}/reliability.csv'))
        for k in ['A','B','B_temperature','CCR']:
            rows=r[(r.classifier_seed==42)&(r.variant==k)&(r.bins==10)&(r.n>0)]
            ax.plot(rows.confidence,rows.accuracy,'o-',label=k,color=COLORS[k],alpha=.85)
        ax.plot([.5,1],[.5,1],'--',color='.6',lw=1);ax.set_xlim(.48,1.01);ax.set_ylim(0.35,1.02);ax.set_title(title);ax.set_xlabel('Mean confidence');ax.set_ylabel('Observed accuracy');ax.legend(fontsize=8)
    fig.suptitle('Reliability · seed 42 · 10 equal-width bins · exploratory',fontsize=12);figure_save(fig,dest,'calibration')
    fig,axs=plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
    for ax,name,title in zip(axs,sets,titles):
        r=pd.read_csv(s.path(f'analysis/{name}/risk_coverage.csv'))
        for k in ['A','B','B_temperature','CCR']:
            rows=r[(r.classifier_seed==42)&(r.variant==k)].sort_values('achieved_coverage')
            ax.plot(rows.achieved_coverage,rows.risk,'o-',label=k,color=COLORS[k],alpha=.85)
        ax.set_title(title);ax.set_xlabel('Achieved coverage');ax.set_ylabel('Error rate among retained pairs');ax.legend(fontsize=8);ax.grid(alpha=.2)
    fig.suptitle('Selective prediction · cutoffs fixed on validation · seed 42',fontsize=12);figure_save(fig,dest,'risk_coverage')
    fig,axs=plt.subplots(1,2,figsize=(12,5),layout='constrained')
    for ax,name,title in zip(axs,sets,titles):
        sub=pd.read_csv(s.path(f'analysis/{name}/subgroups.csv'));sub=sub[(sub.classifier_seed==42)&sub.group.str.startswith('label6_')]
        for j,k in enumerate(['A','B','CCR']):
            rows=sub[sub.variant==k].set_index('group').reindex([f'label6_{i}' for i in range(6)]);ypos=np.arange(6)+(j-1)*.19
            ax.errorbar(rows.accuracy,ypos,xerr=[rows.accuracy-rows.accuracy_ci95_low,rows.accuracy_ci95_high-rows.accuracy],fmt='o',ms=4,capsize=2,label=k,color=COLORS[k])
        base=sub[sub.variant=='B'].set_index('group').reindex([f'label6_{i}' for i in range(6)]);labs=[f'{LABELS[i]} (n={int(base.loc[f"label6_{i}","n"]) if pd.notna(base.loc[f"label6_{i}","n"]) else 0})' for i in range(6)]
        ax.set_yticks(range(6),labs);ax.invert_yaxis();ax.set_xlabel('Accuracy; Wilson 95% intervals');ax.set_title(title);ax.legend(fontsize=8);ax.grid(axis='x',alpha=.2)
    fig.suptitle('Six-way categories · descriptive slices · seed 42',fontsize=12);figure_save(fig,dest,'category_panels')
    fig,axs=plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
    for ax,name,title in zip(axs,sets,titles):
        u=pd.read_csv(s.path(f'analysis/{name}/conditional_utility.csv'));u=u[(u.classifier_seed==42)&(u.utility_model=='ridge_candidate')]
        ax.errorbar(u.predicted_mean,u.observed_mean,yerr=[u.observed_mean-u.observed_ci95_low,u.observed_ci95_high-u.observed_mean],fmt='o',capsize=3,color=COLORS['ridge_candidate'])
        lo=min(u.predicted_mean.min(),u.observed_mean.min());hi=max(u.predicted_mean.max(),u.observed_mean.max());ax.plot([lo,hi],[lo,hi],'--',color='.5')
        ax.axhline(0,color='.7',lw=1);ax.set_xlabel('Predicted CE(A) − CE(B)');ax.set_ylabel('Observed mean CE(A) − CE(B)');ax.set_title(title)
    fig.suptitle('Conditional utility · OOF-defined score bins · seed 42 · pointwise 95% intervals',fontsize=12);figure_save(fig,dest,'conditional_utility')
    fig,axs=plt.subplots(1,2,figsize=(10,4),layout='constrained')
    for ax,name,title in zip(axs,sets,titles):
        q=pd.read_csv(s.path(f'analysis/{name}/quartile_prevalence.csv')).query('classifier_seed == 42')
        ax.bar(q.quartile,q.real_fraction,color='#20749b',alpha=.8)
        ax.errorbar(q.quartile,q.real_fraction,yerr=[q.real_fraction-q.prevalence_ci95_low,q.prevalence_ci95_high-q.real_fraction],fmt='none',ecolor='black',capsize=3)
        for r in q.itertuples():ax.text(r.quartile,r.real_fraction+.035,f'{r.real}/{r.n}',ha='center',fontsize=9)
        ax.set_ylim(0,1);ax.set_xticks([1,2,3,4],['Q1','Q2','Q3','Q4']);ax.set_title(title);ax.set_ylabel('Real-label proportion');ax.set_xlabel('Cosine band (training-defined boundaries)')
    fig.suptitle('Observational cosine–label association; not evidence of factual alignment',fontsize=12);figure_save(fig,dest,'cosine_label_prevalence')
    # Validation distribution diagnostic; no calibration correction or retuning.
    shift=[]
    for seed in s.seeds:
        p=read_json(s.models/f'policy_{seed}.json')
        for k,v in p['utility_score_check'].items():shift.append({'classifier_seed':seed,'estimator':k,**v})
    shift_path=s.path('validation/utility_distribution_shift.csv')
    if not shift_path.exists():save_csv(pd.DataFrame(shift),shift_path)
    policies=[read_json(s.models/f'policy_{seed}.json') for seed in s.seeds]
    guard_table=pd.DataFrame([{'classifier_seed':p['classifier_seed'],'guard_passed':p['guard_passed'],'validation_ridge_B_fraction':p['random_b_fraction'],
         'validation_logloss_delta':p['guard']['delta_candidate_minus_base']['log_loss'],'CI_low':p['guard']['ci95']['log_loss'][0],'CI_high':p['guard']['ci95']['log_loss'][1],
         'A_temperature':p['temperatures']['A'],'B_temperature':p['temperatures']['B']} for p in policies])
    primary=[]
    for name in sets:
        for r in pairs[name]:
            if r['classifier_seed']==42 and r['candidate'] in ['CCR','ridge_candidate'] and r['bootstrap_unit']=='row':
                primary.append({'dataset':name,'candidate':r['candidate'],'logloss_delta':r['delta_candidate_minus_base']['log_loss'],'loss_CI_low':r['ci95']['log_loss'][0],
                   'loss_CI_high':r['ci95']['log_loss'][1],'macro_f1_delta':r['delta_candidate_minus_base']['macro_f1'],'corrected':r['corrected'],'new_errors':r['new_errors'],'McNemar_p':r['mcnemar_exact_two_sided_p']})
    audit=read_json(s.project/'results/research_extension/ccr/audit.json');oof=read_json(s.path('oof_complete.json'))
    access=read_json(s.project/'results/research_extension/ccr/external_access.json')
    report_text=f'''# Conditional CLIP Reliance: measured study outcome

**Publication decision: NO-GO for a claimed superior new detection method.** VERITE's full official images are inaccessible without approved institutional access; the user confirmed no approved copy. The preregistered independent endpoint is **not evaluated** (0 external samples). Existing Fakeddit results below are exploratory and cannot replace it. A research/robustness technical report is justified; a journal claim requires independent evidence and a substantially clearer contribution beyond existing utility gating.

## What was run

{len(oof['runs'])} matched OOF A/B head fits, five folds, three classifier seeds42/123/2026, {oof['rows']:,} training examples per seed. Six historical final A/B heads reused. Gate families fixed before training: intercept, ridge alpha100, depth3 tree/minimum500 per leaf. Normalizers use only inner-fit rows; stopping uses only internal validation. The optional historical unimodal scores were excluded because their training-set scores would be in-sample. No encoder fine-tuning or gate fitting on either test set.

The OOF stage recorded {oof['seconds']:.1f} seconds wall time and process-lifetime peak working set {oof.get('peak_process_rss_bytes',0)/2**30:.2f} GiB. Times exclude literature work and prior encoder extraction. Historical inventory covers {audit['preserved_files']:,} files ({audit['preserved_bytes']:,} bytes), including BRAS. See `preservation_check.json` for final rehash.

Protocol SHA-256: `{digest(s.protocol)}` (canonical JSON digest; raw-file SHA is recorded in preregistered_protocol.md). Local pre-training timestamp only, not a publicly registered trial. All policies/checkpoints were frozen before any CCR test inference.

## Validation guard and calibration

{table(guard_table)}

Negative loss delta favors the candidate. CCR uses the candidate only when its validation paired CI upper bound is negative; otherwise it is exactly B. The unguarded ridge candidate remains visible as a secondary ablation. These validation intervals follow threshold selection on the same data and are optimistic diagnostics, not valid independent superiority tests. A/B checkpoint and architecture selection also used this validation set. Distribution summaries are in `validation/utility_distribution_shift.csv`; no test-dependent adjustment was made.

'''+ '\n\n'.join(chunks)+f'''

## Seed-42 paired results against B

{table(pd.DataFrame(primary))}

Every estimate is recomputed from saved prediction CSVs. Loss/Brier deltas are candidate minus B; negative favors the candidate. Accuracy/Macro-F1 deltas use the same direction, so positive favors it.5,000 common-row bootstrap replicates; bootstrap_seed20261106 is separate from classifier_seed. Known Fakeddit families are singletons, so family bootstrap equals row bootstrap; unknown event dependence remains. Exact McNemar uses discordant correctness counts. Per-seed full tables include Brier, accuracy and Macro-F1 intervals. Secondary intervals and slices are descriptive; no test winner selection or pooled-seed sample inflation.

## Calibration, slices and pair sensitivity

Figures include loss comparison, reliability diagrams, fixed-threshold risk/coverage, six-way category accuracy with counts and Wilson intervals, conditional-utility bins and cosine-label prevalence. ECE uses confidence calibration with10 equal-width bins;5/15/20-bin results are saved. Brier is the single real-class squared error, not twice that value. Coverage is achieved test coverage at validation-fixed confidence cutoffs; discarded examples are explicitly counted.

Every subreddit slice includes counts and row-level uncertainty; these are distant-supervision categories, not evidence of source-generalization. Source families are linked-submission components and all retained families have one row. Broader news-event independence is unverified. Train-quartile cosine prevalence is an observational shortcut audit. Differences in class balance affect proper losses/calibration and cannot be causally attributed to cosine.

Synthetic random/hard swaps reuse the exact500-query training-donor selections from BRAS. `stress/summary.csv` and saved pair predictions report cosine/utility changes, routing changes and label-free prediction flips. No truth labels, factual accuracy, or log-loss are assigned to synthetic swaps.

## Previous BRAS result and prior-art correction

BRAS D mean Macro-F1 was0.872889 on historical2500 versus B0.872115, and0.855797 on the previously viewed1000 versus B0.859407. Seed42 D−B Macro-F1 CIs included zero on both. This negative result is preserved, with the corrected classifier-seed/bootstrap-seed tables as authoritative. There was no reliable BRAS advantage.

For identical neighborhoods, `(s-r_image)+(s-r_text)=2s-r_image-r_text`: summed BRAS margins have the CSLS algebraic form. Separate margins and a restricted training-only bank describe an implementation choice, not an invented similarity principle. SSPC already explicitly learns a cross-fitted utility gate in multimodal fake-news detection. CCR studies an incremental scalar-CLIP decision in frozen heads; it does not establish invention of utility gates or factual verification. See the verified-overlap report for access limitations and exact sources.

## Independent evaluation and claims

External status: **{access['status']}**. Public metadata/URLs do not supply the required full authorized image set. No gated data, external scores, or images were fabricated or redistributed. The local authorized-data adapter is supplied but was not run against real VERITE images; it is not evidence of completed external validation. VERITE labels describe truthful/miscaptioned/out-of-context pairs, whereas Fakeddit labels are distant supervision; any future evaluation must be reported as cross-task/domain transfer.

Supported claims: an auditable implementation and controlled exploratory comparison, with preserved negative results and measured routing/calibration tradeoffs. Unsupported claims: generalization to an independent benchmark, improved factual verification, SOTA, first invention, or guaranteed journal/Scopus acceptance. A negative-result paper would need a sharper hypothesis, justified external evaluation or a stronger multi-dataset robustness study, and complete prior-art verification. No journal submission is recommended from the present evidence alone.

## Reproduce

From the project directory, using the existing virtualenv Python:

```powershell
python -m unittest src.research_extension.ccr.tests -v
python -m src.research_extension.ccr smoke
python -m src.research_extension.ccr train
python -m src.research_extension.ccr freeze
python -m src.research_extension.ccr evaluate
python -m src.research_extension.ccr stress
python -m src.research_extension.ccr report
python -m src.research_extension.ccr verify-preservation
# Only after obtaining approved official data; never retune to its labels:
python -m src.research_extension.ccr verite-local --authorized-dir C:\\path\\to\\authorized_VERITE
```

CLI defaults to this project root and writes only CCR namespaces. Completed fold checkpoints and epoch resume states are hash-bound; scientific results never overwrite historical original/BRAS artifacts. The report package excludes all image datasets and contains scripts, policies, predictions, statistics, figures and provenance needed to inspect the results. Large training caches/checkpoints stay in the project.
'''
    (s.docs/'results_summary.md').write_text(report_text,encoding='utf-8')
    return {'report':str(s.docs/'results_summary.md'),'figures':len(list(dest.glob('*.png')))}
