"""Paired group outcomes, linked-component bootstrap and cluster randomization."""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score,average_precision_score
from .io_utils import *

def auc_formula(y,s,w=None):
    y=np.asarray(y,dtype=int);s=np.asarray(s);w=np.ones(len(y)) if w is None else np.asarray(w)
    valid=w>0;y=y[valid];s=s[valid];w=w[valid]
    order=np.argsort(s,kind='stable');y=y[order];s=s[order];w=w[order]
    ends=np.r_[np.flatnonzero(np.diff(s))+1,len(s)];starts=np.r_[0,ends[:-1]]
    pos=np.add.reduceat(w*y,starts);neg=np.add.reduceat(w*(1-y),starts)
    return float(np.sum(pos*(np.cumsum(neg)-neg+.5*neg))/(pos.sum()*neg.sum()))
def ap_formula(y,s,w=None):
    y=np.asarray(y,dtype=int);w=np.ones(len(y)) if w is None else np.asarray(w)
    s=np.asarray(s);valid=w>0;y=y[valid];s=s[valid];w=w[valid]
    order=np.argsort(-s,kind='stable');ss=s[order];yy=y[order];ww=w[order]
    ends=np.r_[np.flatnonzero(np.diff(ss))+1,len(ss)]-1;tp=np.cumsum(ww*yy)[ends];total=np.cumsum(ww)[ends]
    return float(np.sum(np.diff(np.r_[0,tp])/tp[-1]*(tp/total)))
def paired(frame,score):
    require(frame.groupby('pair_id').falsified.apply(lambda v:sorted(v)==[0,1]).all(),'Incomplete/duplicate pair group')
    p=frame[frame.falsified.eq(0)].set_index('pair_id').sort_index();n=frame[frame.falsified.eq(1)].set_index('pair_id').loc[p.index]
    require((p.caption_id==n.caption_id).all() and (p.cluster_id==n.cluster_id).all(),'Broken paired caption/cluster')
    d=p[score].to_numpy()-n[score].to_numpy()
    return pd.DataFrame({'pair_id':p.index,'cluster_id':p.cluster_id.to_numpy(),'strategy':p.strategy.to_numpy(),
        'source':p.source.to_numpy(),'pristine_score':p[score].to_numpy(),'mismatch_score':n[score].to_numpy(),'margin':d,
        'ranking_half':(d>0).astype(float)+.5*(d==0),'ranking_strict':(d>0).astype(float),
        'ranking_tolerance':(d>1e-6).astype(float)+.5*(np.abs(d)<=1e-6)})
def point(p,w=None):
    w=np.ones(len(p)) if w is None else np.asarray(w);den=w.sum()
    y=np.r_[np.ones(len(p)),np.zeros(len(p))];s=np.r_[p.pristine_score,p.mismatch_score];weights=np.r_[w,w]
    return np.array([np.dot(w,p.ranking_half)/den,auc_formula(y,s,weights),ap_formula(y,s,weights),np.dot(w,p.margin)/den])
METRICS=['paired_ranking_half','roc_auc_pristine','average_precision_pristine','mean_paired_margin']
def bootstrap(p,reps=5000,seed=20261108,cluster='cluster_id'):
    keys,codes=np.unique(p[cluster],return_inverse=True);rng=np.random.default_rng(seed);out=np.empty((reps,4))
    for i in range(reps):
        counts=np.bincount(rng.integers(0,len(keys),len(keys)),minlength=len(keys));out[i]=point(p,counts[codes])
    require(np.isfinite(out).all(),'Nonfinite bootstrap metric');return out
def summary(p,reps=5000,seed=20261108):
    v=point(p);boot=bootstrap(p,reps,seed);bounds=np.quantile(boot,[.025,.975],axis=0)
    y=np.r_[np.ones(len(p)),np.zeros(len(p))];s=np.r_[p.pristine_score,p.mismatch_score]
    require(abs(v[1]-roc_auc_score(y,s))<1e-12 and abs(v[2]-average_precision_score(y,s))<1e-12,'Independent metric verification failed')
    return {'pairs':len(p),'clusters':p.cluster_id.nunique(),'ties':int((p.margin==0).sum()),
      'metrics':{name:{'estimate':float(v[j]),'ci95':bounds[:,j].tolist()} for j,name in enumerate(METRICS)},
      'strict_ranking':float(p.ranking_strict.mean()),'tolerance_1e_6_ranking':float(p.ranking_tolerance.mean()),
      'verified_against_sklearn':True}
def signflip(d,clusters,reps=10000,seed=20261109):
    d=np.asarray(d);keys,codes=np.unique(clusters,return_inverse=True);sums=np.bincount(codes,weights=d)
    observed=abs(d.sum());rng=np.random.default_rng(seed);count=0
    if not np.any(d):return 1.
    for start in range(0,reps,200):
        n=min(200,reps-start);null=np.abs((rng.integers(0,2,(n,len(keys)))*2-1)@sums);count+=int(np.sum(null>=observed-1e-12))
    return (count+1)/(reps+1)
def holm(p):
    p=np.asarray(p);order=np.argsort(p);adj=np.empty(len(p));adj[order]=np.minimum(1,np.maximum.accumulate((len(p)-np.arange(len(p)))*p[order]));return adj.tolist()
def comparisons(tables,reps=5000):
    out=[];base=tables['cosine']
    for name,p in tables.items():
        if name=='cosine':continue
        require(base.pair_id.tolist()==p.pair_id.tolist() and base.cluster_id.tolist()==p.cluster_id.tolist(),'Unpaired score comparison')
        a=bootstrap(base,reps);b=bootstrap(p,reps);delta=point(p)-point(base)
        out.append({'comparison':name+' minus cosine','primary':name=='bras_mean','pairs':len(p),
          'effect':dict(zip(METRICS,delta.tolist())),'ci95':{k:np.quantile(b[:,j]-a[:,j],[.025,.975]).tolist() for j,k in enumerate(METRICS)},
          'ranking_cluster_signflip_p':signflip(p.ranking_half-base.ranking_half,p.cluster_id)})
    return out
def analyze(run,mode='official'):
    run.frozen();path=run.results/f'{mode}/predictions.csv';verify(path,run.binding())
    f=pd.read_csv(path,keep_default_na=False,dtype={'caption_id':str,'image_id':str});p=run.protocol();reps=p['bootstrap']['repetitions']
    scores=[x for x in p['scores'] if x in f];tables={s:paired(f,s) for s in scores};result={'feature_provenance':mode,'positive_class':'pristine','repetitions':reps,'bootstrap_seed':p['bootstrap']['seed'],'scores':{},'comparisons':comparisons(tables,reps),'strategy_ranking_tests':[]}
    for name,table in tables.items():
        csv(run.results/f'{mode}/paired_{name}.csv',table);entry={'overall':summary(table,reps),'by_strategy':{},'by_source':{}}
        for st,g in table.groupby('strategy'):
            entry['by_strategy'][st]=summary(g,reps)
            result['strategy_ranking_tests'].append({'score':name,'strategy':st,'null':'mean half-credit ranking=0.5','p':signflip(g.ranking_half-.5,g.cluster_id)})
        for source,g in table.groupby('source'):entry['by_source'][source]=summary(g,reps)
        entry['robustness']={'leave_one_strategy_out':{st:dict(zip(METRICS,point(table[table.strategy.ne(st)]).tolist())) for st in table.strategy.unique()},
          'pair_only_ranking_ci95':np.quantile(bootstrap(table,reps,cluster='pair_id')[:,0],[.025,.975]).tolist(),
          'single_pair_components':summary(table[table.cluster_id.map(table.cluster_id.value_counts()).eq(1)],reps) if (table.cluster_id.map(table.cluster_id.value_counts()).eq(1)).any() else None}
        result['scores'][name]=entry
    tests=result['strategy_ranking_tests'];secondary=[x for x in result['comparisons'] if not x['primary']]
    values=[x['p'] for x in tests]+[x['ranking_cluster_signflip_p'] for x in secondary]
    for item,adj in zip(tests+secondary,holm(values)):item['holm_adjusted_p']=adj
    result['caveats']=['Dataset generated partly with CLIP; not independent of CLIP method selection.','No factual-veracity inference.','Bootstrap conditions on selected available groups; unknown event/pretraining dependence persists.','Cluster sign-flip tests require exchangeability/symmetry under their null.']
    write(run.results/f'{mode}/statistics.json',result);seal(run.results/f'{mode}/statistics.json',{'predictions_sha256':sha(path),**run.binding()});return result
