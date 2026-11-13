"""Cluster-aware cross-task classification and paired correspondence statistics."""
from __future__ import annotations
import time
import numpy as np,pandas as pd
from sklearn.metrics import accuracy_score,f1_score,roc_auc_score,average_precision_score,log_loss
from ...external_eval.newsclippings.statistics import auc_formula,ap_formula,holm,signflip
from .core import *
NBOOT=5000;BSEED=20261109;NTEST=10000;TSEED=20261110
METRICS=['accuracy','macro_f1','roc_auc_truthful','average_precision_truthful','log_loss','brier']

def confusion(y,d,w=None):
    y=np.asarray(y,dtype=int);d=np.asarray(d,dtype=int);w=np.ones(len(y)) if w is None else np.asarray(w)
    return np.bincount(y*2+d,weights=w,minlength=4)
def from_confusion(c):
    c=np.asarray(c);tn,fp,fn,tp=np.moveaxis(c,-1,0);total=c.sum(-1)
    d0=2*tn+fp+fn;d1=2*tp+fp+fn
    f0=np.divide(2*tn,d0,out=np.zeros_like(d0,dtype=float),where=d0>0);f1=np.divide(2*tp,d1,out=np.zeros_like(d1,dtype=float),where=d1>0)
    return np.stack([(tn+tp)/total,(f0+f1)/2],axis=-1)
def binary_metrics(y,p,d,w=None):
    y=np.asarray(y);p=np.asarray(p);d=np.asarray(d);w=np.ones(len(y)) if w is None else np.asarray(w)
    require(np.isin(y,[0,1]).all() and np.isin(d,[0,1]).all() and np.isfinite(p).all() and ((p>=0)&(p<=1)).all(),'Invalid binary diagnostic inputs')
    pp=np.clip(p,1e-6,1-1e-6);loss=-(y*np.log(pp)+(1-y)*np.log1p(-pp))
    basic=from_confusion(confusion(y,d,w));den=w.sum()
    return np.r_[basic,auc_formula(y,p,w),ap_formula(y,p,w),w@loss/den,w@((p-y)**2)/den]
def resampling(clusters,reps=NBOOT,seed=BSEED):
    unique,codes=np.unique(np.asarray(clusters).astype(str),return_inverse=True);rng=np.random.default_rng(seed)
    draws=rng.integers(0,len(unique),(reps,len(unique)));weights=np.zeros((reps,len(unique)),np.int32)
    for i in range(reps):weights[i]=np.bincount(draws[i],minlength=len(unique))
    return weights,codes,unique
def boot_binary(y,p,d,weights,codes):
    out=np.empty((len(weights),6),float)
    for i,w in enumerate(weights):out[i]=binary_metrics(y,p,d,w[codes])
    require(np.isfinite(out).all(),'Nonfinite resampled classification metric');return out
def cluster_swap_test(y,a,b,clusters,reps=NTEST,seed=TSEED):
    """Swap entire model decision vectors within components; recompute nonlinear Macro-F1."""
    y=np.asarray(y,int);a=np.asarray(a,int);b=np.asarray(b,int)
    if np.array_equal(a,b):return {'accuracy_p':1.,'macro_f1_p':1.,'draws':reps}
    keys,codes=np.unique(clusters,return_inverse=True);ca=np.zeros((len(keys),4));cb=ca.copy()
    np.add.at(ca,(codes,2*y+a),1);np.add.at(cb,(codes,2*y+b),1)
    total_a=ca.sum(0);total_b=cb.sum(0);delta=cb-ca;observed=from_confusion(total_b)-from_confusion(total_a)
    rng=np.random.default_rng(seed);exceed=np.zeros(2,int)
    for start in range(0,reps,200):
        flips=rng.integers(0,2,(min(200,reps-start),len(keys)));change=flips@delta
        perm=from_confusion(total_a+change)-from_confusion(total_b-change)
        exceed+=(np.abs(perm)>=np.abs(observed)[None,:]-1e-12).sum(0)
    values=(exceed+1)/(reps+1)
    return {'accuracy_p':float(values[0]),'macro_f1_p':float(values[1]),'draws':reps}
def paired_table(frame,links,score):
    indexed=frame.set_index('row_id');use=links[links.complete].copy();t=indexed.loc[use.true_row_id];n=indexed.loc[use.misleading_row_id]
    require(t.category.eq('true').all() and np.array_equal(n.category.to_numpy(),use.category.to_numpy()),'Pair label contract broken')
    require(np.array_equal(t.family_id.to_numpy(),n.family_id.to_numpy()) and np.array_equal(t.cluster_id.to_numpy(),n.cluster_id.to_numpy()),'Paired content grouping broken')
    diff=t[score].to_numpy(float)-n[score].to_numpy(float)
    return pd.DataFrame({'family_id':use.family_id.to_numpy(),'category':use.category.to_numpy(),'cluster_id':t.cluster_id.to_numpy(),'source':t.source.to_numpy(),
      'true_row_id':use.true_row_id.to_numpy(),'misleading_row_id':use.misleading_row_id.to_numpy(),'true_score':t[score].to_numpy(),'misleading_score':n[score].to_numpy(),
      'margin':diff,'ranking_half':(diff>0).astype(float)+.5*(diff==0),'ranking_strict':(diff>0).astype(float),'ranking_tolerance':(diff>1e-6).astype(float)+.5*(np.abs(diff)<=1e-6)})
def boot_pair(p,weights,codes):
    values=p[['ranking_half','margin']].to_numpy();sums=np.zeros((weights.shape[1],2));n=np.bincount(codes,minlength=weights.shape[1]);np.add.at(sums,codes,values)
    return (weights@sums)/(weights@n)[:,None]
def ci(a):return np.quantile(a,[.025,.975],axis=0).T.tolist()
def check_metrics(y,p,d,values):
    clipped=np.clip(p,1e-6,1-1e-6)
    expected=[accuracy_score(y,d),f1_score(y,d,labels=[0,1],average='macro',zero_division=0),roc_auc_score(y,p),average_precision_score(y,p),
       log_loss(y,np.c_[1-clipped,clipped],labels=[0,1]),np.mean((p-y)**2)]
    require(np.allclose(values,expected,rtol=0,atol=1e-12),'Independent sklearn metric verification failed')

def analyze(s):
    s.frozen();verify_sources(s);start=time.perf_counter();manifest=s.frame();manifest=manifest[manifest.available].reset_index(drop=True)
    weights,codes,unique=resampling(manifest.cluster_id);rows=[];categories=[];sources=[];comparisons=[];shifts=[]
    protocol=s.protocol();primary=[(x['candidate'],x['base'],x['seed']) for x in protocol['primary_transfer_comparisons']]
    allpred={};allboots={}
    for seed in SEEDS:
        path=s.results/f'predictions/seed_{seed}.csv';verify(path);f=pd.read_csv(path,keep_default_na=False,dtype={'row_id':str,'family_id':str,'cluster_id':str})
        require(f.row_id.tolist()==manifest.row_id.tolist(),'Stats/model row order differs');allpred[seed]=f;y=f.binary_label.to_numpy()
        boot={}
        for kind in KINDS:
            p=f['p_'+kind].to_numpy();d=f['decision_'+kind].to_numpy();point=binary_metrics(y,p,d);check_metrics(y,p,d,point)
            if kind=='CCR' and np.array_equal(p,f.p_B) and np.array_equal(d,f.decision_B):boot[kind]=boot['B'].copy()
            else:boot[kind]=boot_binary(y,p,d,weights,codes)
            bounds=ci(boot[kind]);record={'seed':seed,'model':kind,'n':len(f),'clusters':len(unique)}
            for j,name in enumerate(METRICS):record[name]=float(point[j]);record[name+'_ci_low']=bounds[j][0];record[name+'_ci_high']=bounds[j][1]
            rows.append(record)
            for cat,g in f.groupby('category'):
                correct=g['decision_'+kind].eq(g.binary_label)
                categories.append({'seed':seed,'model':kind,'category':cat,'n':len(g),'correct':int(correct.sum()),'errors':int((~correct).sum()),'accuracy':float(correct.mean()),'truthful_decision_rate':float(g['decision_'+kind].mean()),'mean_p_truthful':float(g['p_'+kind].mean())})
            for source,g in f.groupby('source'):
                values=binary_metrics(g.binary_label,g['p_'+kind],g['decision_'+kind]);sources.append({'seed':seed,'model':kind,'source':source,'n':len(g),'clusters':g.cluster_id.nunique(),**dict(zip(METRICS,values.tolist()))})
        allboots[seed]=boot
        npz(s.results/f'bootstrap/classification_{seed}.npz',s.binding(),**boot)
        for candidate,base in [('B','A'),('D','B'),('C','B'),('CCR','B')]:
            pa=f['p_'+base].to_numpy();pb=f['p_'+candidate].to_numpy();a=f['decision_'+base].to_numpy();b=f['decision_'+candidate].to_numpy()
            effect=binary_metrics(y,pb,b)-binary_metrics(y,pa,a);bounds=ci(boot[candidate]-boot[base]);tests=cluster_swap_test(y,a,b,f.cluster_id)
            gain=(a!=y)&(b==y);loss=(a==y)&(b!=y)
            record={'seed':seed,'candidate':candidate,'base':base,'primary':(candidate,base,seed) in primary,'n':len(f),'clusters':len(unique),
              'effect_candidate_minus_base':dict(zip(METRICS,effect.tolist())),'ci95':dict(zip(METRICS,bounds)),**tests,
              'corrected_errors':int(gain.sum()),'introduced_errors':int(loss.sum()),'decision_changes':int((a!=b).sum()),'mean_probability_shift':float((pb-pa).mean()),
              'mean_absolute_probability_shift':float(np.abs(pb-pa).mean())}
            comparisons.append(record)
            changes=f[['row_id','category','family_id','cluster_id']].copy();changes['corrected']=gain;changes['introduced']=loss;changes['probability_difference']=pb-pa
            csv(s.results/f'comparisons/{candidate}_minus_{base}_{seed}_rows.csv',changes)
        print('Classifiers analyzed seed',seed,flush=True)
    for is_primary in [True,False]:
        family=[x for x in comparisons if x['primary']==is_primary]
        for r,adjusted in zip(family,holm([x['macro_f1_p'] for x in family])):r['macro_f1_holm_p']=adjusted
    metrics=pd.DataFrame(rows);csv(s.results/'model_metrics.csv',metrics);csv(s.results/'category_breakdown.csv',pd.DataFrame(categories));csv(s.results/'source_breakdown.csv',pd.DataFrame(sources))
    summary=metrics.groupby('model')[METRICS].agg(['mean','std']);summary.columns=['_'.join(x) for x in summary.columns];csv(s.results/'seed_summary.csv',summary.reset_index())
    write(s.results/'paired_classification.json',{'comparisons':comparisons,'primary_family':'seed42 B-A and D-B Macro-F1; Holm across2','secondary_family':'other seed/model comparisons; Holm across10','bootstrap_seed':BSEED,'bootstrap_repetitions':NBOOT,'test_seed':TSEED,'test_repetitions':NTEST})
    alignment(s,weights,codes)
    # Diagnostic majority baseline and predeclared family-only/overlap sensitivity.
    y=manifest.binary_label.to_numpy();d=np.zeros(len(y),int);majority=from_confusion(confusion(y,d))
    robustness={'majority_misleading':{'accuracy':float(majority[0]),'macro_f1':float(majority[1]),'definition':'Always predict misleading; descriptive prevalence baseline, no label-selected model'},'family_only_bootstrap':[],
        'overlap_exclusion_rows':int(manifest.independence_sensitivity_eligible.sum()),'overlap_exclusion_equals_full':bool(manifest.independence_sensitivity_eligible.all())}
    fw,fc,_=resampling(manifest.family_id)
    for candidate,base,seed in primary:
        f=allpred[seed];y=f.binary_label.to_numpy();vals=[]
        for w in fw:
            weight=w[fc];vals.append(from_confusion(confusion(y,f['decision_'+candidate],weight))-from_confusion(confusion(y,f['decision_'+base],weight)))
        robustness['family_only_bootstrap'].append({'candidate':candidate,'base':base,'seed':seed,'accuracy_macro_f1_ci95':ci(np.asarray(vals))})
    if not robustness['overlap_exclusion_equals_full']:
        mask=manifest.independence_sensitivity_eligible.to_numpy();subset=[]
        for seed,f in allpred.items():
            for kind in KINDS:subset.append({'seed':seed,'model':kind,'n':int(mask.sum()),**dict(zip(METRICS,binary_metrics(f.binary_label.to_numpy()[mask],f['p_'+kind].to_numpy()[mask],f['decision_'+kind].to_numpy()[mask]).tolist()))})
        robustness['disjoint_subset_metrics']=subset
    write(s.results/'robustness.json',robustness)
    result={'rows':len(manifest),'clusters':len(unique),'model_rows':len(rows),'paired_comparisons':len(comparisons),'seconds':time.perf_counter()-start,'all_point_metrics_verified_with_sklearn':True}
    if not (s.results/'analysis_profile.json').exists():write(s.results/'analysis_profile.json',result)
    return result

def alignment(s,weights,codes):
    path=s.results/'alignment_predictions.csv';verify(path,s.binding());f=pd.read_csv(path,keep_default_na=False,dtype={'row_id':str,'family_id':str,'cluster_id':str})
    links=pd.read_csv(s.results/'pair_links.csv',keep_default_na=False,dtype={'family_id':str})
    report={'positive_class':'true; higher frozen score means more correspondence','paired':{},'discrimination':{},'comparisons':[],'structural_checks':{}}
    for score in SCORES:
        y=f.binary_label.to_numpy();v=f[score].to_numpy();point=np.array([auc_formula(y,v),ap_formula(y,v)])
        require(np.allclose(point,[roc_auc_score(y,v),average_precision_score(y,v)],atol=1e-12),'Alignment AUC/AP verification failed')
        boot=np.array([[auc_formula(y,v,w[codes]),ap_formula(y,v,w[codes])] for w in weights])
        report['discrimination'][score]={'n':len(f),'roc_auc_truthful':float(point[0]),'average_precision_truthful':float(point[1]),'ci95':dict(zip(['roc_auc_truthful','average_precision_truthful'],ci(boot)))}
        table=paired_table(f,links,score);csv(s.results/f'alignment/paired_{score}.csv',table);report['paired'][score]={}
        for category,g in table.groupby('category'):
            w,c,k=resampling(g.cluster_id);draws=boot_pair(g,w,c);point=g[['ranking_half','margin']].mean().to_numpy()
            report['paired'][score][category]={'pairs':len(g),'clusters':len(k),'ranking_half':float(point[0]),'mean_margin':float(point[1]),
              'ci95':dict(zip(['ranking_half','mean_margin'],ci(draws))),'ties':int(g.margin.eq(0).sum()),'strict_ranking':float(g.ranking_strict.mean()),'tolerance_1e_6_ranking':float(g.ranking_tolerance.mean()),
              'by_source':g.groupby('source').agg(pairs=('ranking_half','size'),ranking_half=('ranking_half','mean'),mean_margin=('margin','mean')).reset_index().to_dict('records')}
            npz(s.results/f'bootstrap/alignment_{score}_{category}.npz',s.binding(),draws=draws)
    for score in SCORES[1:]:
        base=pd.read_csv(s.results/'alignment/paired_cosine.csv',keep_default_na=False);cand=pd.read_csv(s.results/f'alignment/paired_{score}.csv',keep_default_na=False)
        require(base.true_row_id.tolist()==cand.true_row_id.tolist() and base.misleading_row_id.tolist()==cand.misleading_row_id.tolist(),'Unpaired ranking comparison')
        for cat in ['miscaptioned','out-of-context']:
            a=base[base.category.eq(cat)];b=cand[cand.category.eq(cat)];difference=b.ranking_half.to_numpy()-a.ranking_half.to_numpy()
            with np.load(s.results/f'bootstrap/alignment_{score}_{cat}.npz') as t:cb=t['draws']
            with np.load(s.results/f'bootstrap/alignment_cosine_{cat}.npz') as t:ab=t['draws']
            report['comparisons'].append({'score':score,'base':'cosine','category':cat,'primary':score=='bras_mean','pairs':len(a),
              'ranking_effect':float(difference.mean()),'margin_effect':float((b.margin.to_numpy()-a.margin.to_numpy()).mean()),'ci95':dict(zip(['ranking_half','mean_margin'],ci(cb-ab))),
              'cluster_signflip_p':signflip(difference,a.cluster_id.to_numpy(),reps=NTEST,seed=TSEED),
              'ranking_wins_gained':int((difference>0).sum()),'ranking_wins_lost':int((difference<0).sum())})
    for primary in [True,False]:
        family=[r for r in report['comparisons'] if r['primary']==primary]
        for r,adj in zip(family,holm([r['cluster_signflip_p'] for r in family])):r['holm_p']=adj
    for cat,score in [('miscaptioned','m_image'),('out-of-context','m_text')]:
        a=pd.read_csv(s.results/'alignment/paired_cosine.csv');b=pd.read_csv(s.results/f'alignment/paired_{score}.csv');a=a[a.category.eq(cat)];b=b[b.category.eq(cat)]
        report['structural_checks'][cat]={'invariant_direction':score,'maximum_margin_difference':float(np.max(np.abs(a.margin.to_numpy()-b.margin.to_numpy()))),
          'ranking_differences':int((a.ranking_half.to_numpy()!=b.ranking_half.to_numpy()).sum()),'reason':'With unchanged valid references, the neighborhood penalty cancels for the shared image (miscaptioned) or caption (out-of-context).'}
    write(s.results/'alignment_statistics.json',report)
