from __future__ import annotations
import hashlib,json,datetime
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import expit,logit
from scipy.stats import binomtest
from sklearn.metrics import roc_auc_score
from ..common import (Context,require,sha,digest,read_json,write_json,write_npz,load_npz,
                      manifest,load_historical,fit_scaler,standardize,environment,peak_memory)
from ..statistics import fast_macro_f1

FEATURE_NAMES=['p_A_real','p_B_real','logodds_A','logodds_B','logodds_B_minus_A',
               'margin_A','margin_B','entropy_A','entropy_B','cosine']
KINDS=['A','B','A_temperature','B_temperature','intercept','ridge_candidate','tree',
       'cosine_gate','confidence_gate','entropy_gate','random_gate','CCR']
EPS=1e-6
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
@dataclass
class Study:
    project:Path
    smoke:bool=False
    def __post_init__(self):self.project=Path(self.project).resolve()
    @property
    def results(self):return self.project/'results/research_extension/ccr'/('smoke' if self.smoke else 'full')
    @property
    def models(self):return self.project/'models/research_extension/ccr'/('smoke' if self.smoke else 'full')
    @property
    def docs(self):return self.project/'docs/research_extension/ccr'
    @property
    def protocol(self):return read_json(self.project/'models/research_extension/ccr/protocol.json')
    @property
    def seeds(self):return [42] if self.smoke else self.protocol['seeds']
    @property
    def nfolds(self):return 2 if self.smoke else self.protocol['folds']
    @property
    def reps(self):return 200 if self.smoke else self.protocol['bootstrap']['repetitions']
    def path(self,name):return self.results/name

def source_hashes(s):
    paths=list((s.project/'src/research_extension/ccr').glob('*.py'))
    paths += [s.project/'src/models.py',s.project/'src/research_extension/common.py',
              s.project/'src/research_extension/training.py',s.project/'src/research_extension/statistics.py']
    return {p.relative_to(s.project).as_posix():sha(p) for p in sorted(paths)}
def seal(path,inputs=None):
    write_json(path.with_suffix(path.suffix+'.receipt.json'),{'sha256':sha(path),'inputs':inputs or {},'created_utc':now()},immutable=True)
def verify(path):
    receipt=read_json(path.with_suffix(path.suffix+'.receipt.json'))
    require(sha(path)==receipt['sha256'],f'Artifact changed: {path}')
    return receipt
def save_csv(frame,path,inputs=None):
    path.parent.mkdir(parents=True,exist_ok=True)
    require(not path.exists(),f'Cannot overwrite completed artifact {path}')
    frame.to_csv(path,index=False,float_format='%.17g');seal(path,inputs)
def prob(p):
    p=np.asarray(p,dtype=np.float64)
    require(np.isfinite(p).all() and ((p>=0)&(p<=1)).all(),'Invalid binary probability')
    return np.clip(p,EPS,1-EPS)
def ce(y,p):
    y=np.asarray(y)
    require(np.isin(y,[0,1]).all() and y.shape==np.asarray(p).shape,'Invalid labels/loss shape')
    p=prob(p);return -(y*np.log(p)+(1-y)*np.log1p(-p))
def entropy(p):
    p=prob(p);return -p*np.log(p)-(1-p)*np.log1p(-p)
def gate_features(p_a,p_b,cosine):
    a,b=prob(p_a),prob(p_b);c=np.asarray(cosine,dtype=np.float64).reshape(-1)
    require(a.shape==b.shape==c.shape and a.ndim==1,'Gate input shape mismatch')
    la,lb=logit(a),logit(b)
    z=np.column_stack([a,b,la,lb,lb-la,abs(2*a-1),abs(2*b-1),entropy(a),entropy(b),c])
    require(z.shape[1]==len(FEATURE_NAMES) and np.isfinite(z).all(),'Invalid gate features')
    return z
def uniform(key,seed):return int.from_bytes(hashlib.sha256(f'{seed}\0{key}'.encode()).digest()[:8],'big')/2**64
def fold_assign(groups,nfolds,seed):
    return np.array([int.from_bytes(hashlib.sha256(f'{seed}\0{x}'.encode()).digest()[:8],'big')%nfolds for x in groups],int)
def partition(frame,folds,fold,seed):
    require(frame.split.eq('train').all(),'OOF partitions require official train rows only')
    require(len(frame)==len(folds),'Partition length mismatch')
    held=folds==fold
    internal=np.array([uniform(x,f'{seed}:inner:{fold}')<.1 for x in frame.group_id])
    fit=~held&~internal;val=~held&internal
    for a,b in [(fit,val),(fit,held),(val,held)]:
        require(not set(frame.loc[a,'group_id'])&set(frame.loc[b,'group_id']),'Group leakage in OOF partition')
    require(fit.any() and val.any() and held.any(),'Empty OOF partition')
    return fit,val,held
def train_frame(s):
    f=manifest(Context(s.project,s.project));f=f.loc[f.split.eq('train')].reset_index(drop=True)
    if s.smoke:
        order=np.argsort([uniform(x,20261106) for x in f.sample_id]);f=f.iloc[order[:400]].reset_index(drop=True)
    return f
def split_frame(s,split):
    require(split in ['validation','test'],'Unknown split')
    f=manifest(Context(s.project,s.project));return f.loc[f.split.eq(split)].reset_index(drop=True)

def reliability(y,p,bins):
    p=np.asarray(p);confidence=np.maximum(p,1-p);correct=((p>.5).astype(int)==y)
    bucket=np.minimum((confidence*bins).astype(int),bins-1);out=[]
    for k in range(bins):
        mask=bucket==k;n=int(mask.sum())
        out.append({'bin':k,'lower':k/bins,'upper':(k+1)/bins,'n':n,
                    'confidence':float(confidence[mask].mean()) if n else None,
                    'accuracy':float(correct[mask].mean()) if n else None})
    return out
def scores(y,p):
    y=np.asarray(y);p=np.asarray(p);pred=(p>.5).astype(int)
    out={'n':len(y),'accuracy':float((pred==y).mean()),'macro_f1':float(fast_macro_f1(y,pred)),
         'log_loss':float(ce(y,p).mean()),'brier':float(((p-y)**2).mean()),
         'roc_auc':float(roc_auc_score(y,p)) if len(np.unique(y))==2 else None}
    for bins in [5,10,15,20]:
        out[f'ece_{bins}']=float(sum(r['n']*abs(r['confidence']-r['accuracy']) for r in reliability(y,p,bins) if r['n'])/len(y))
    return out
def paired_stats(y,base,candidate,reps=5000,bootstrap_seed=20261106,clusters=None):
    y=np.asarray(y);a=np.asarray(base);b=np.asarray(candidate)
    require(y.shape==a.shape==b.shape and y.ndim==1 and len(y)>0,'Unpaired prediction arrays')
    dl=ce(y,b)-ce(y,a);db=(b-y)**2-(a-y)**2
    ap=(a>.5).astype(int);bp=(b>.5).astype(int);ac=ap==y;bc=bp==y
    gain=int((~ac&bc).sum());loss=int((ac&~bc).sum())
    point={'log_loss':float(dl.mean()),'brier':float(db.mean()),'macro_f1':float(fast_macro_f1(y,bp)-fast_macro_f1(y,ap)),
           'accuracy':float(bc.mean()-ac.mean())}
    unique,inverse=np.unique(np.arange(len(y)) if clusters is None else clusters,return_inverse=True)
    # Cluster-sum sufficient statistics keep bootstrap bounded and support unequal cluster sizes.
    ncl=len(unique);count=np.bincount(inverse,minlength=ncl)
    dlc=np.bincount(inverse,weights=dl,minlength=ncl);dbc=np.bincount(inverse,weights=db,minlength=ncl)
    confusion=[]
    for pred in [ap,bp]:
        confusion.append(np.stack([np.bincount(inverse,weights=((y==t)&(pred==u)),minlength=ncl) for t,u in [(0,0),(0,1),(1,0),(1,1)]],1))
    rng=np.random.default_rng(bootstrap_seed);dist={k:[] for k in point}
    for start in range(0,reps,100):
        idx=rng.integers(0,ncl,size=(min(100,reps-start),ncl));den=count[idx].sum(1)
        dist['log_loss'].extend((dlc[idx].sum(1)/den).tolist());dist['brier'].extend((dbc[idx].sum(1)/den).tolist())
        ca,cb=[x[idx].sum(1) for x in confusion]
        def mf(c):return .5*(2*c[:,0]/np.maximum(2*c[:,0]+c[:,1]+c[:,2],1)+2*c[:,3]/np.maximum(2*c[:,3]+c[:,1]+c[:,2],1))
        dist['macro_f1'].extend((mf(cb)-mf(ca)).tolist())
        dist['accuracy'].extend(((cb[:,0]+cb[:,3]-ca[:,0]-ca[:,3])/den).tolist())
    return {'n':len(y),'clusters':ncl,'bootstrap_unit':'row' if clusters is None else 'known group/component',
       'bootstrap_seed':bootstrap_seed,'repetitions':reps,'delta_candidate_minus_base':point,
       'ci95':{k:np.quantile(v,[.025,.975]).tolist() for k,v in dist.items()},'corrected':gain,'new_errors':loss,
       'both_correct':int((ac&bc).sum()),'both_wrong':int((~ac&~bc).sum()),
       'mcnemar_exact_two_sided_p':float(binomtest(gain,gain+loss,.5).pvalue) if gain+loss else 1.}
def wilson(correct,n):
    if not n:return [None,None]
    z=1.959963984540054;p=correct/n;d=1+z*z/n
    c=(p+z*z/(2*n))/d;h=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return [float(c-h),float(c+h)]
