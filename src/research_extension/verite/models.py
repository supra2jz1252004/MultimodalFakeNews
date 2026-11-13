"""Read-only checkpoint replay and existing BRAS/CCR policies."""
from __future__ import annotations
import time
import numpy as np,pandas as pd
from .core import *

def alignment_scores(s,f,z):
    from ..reference import margins
    path=s.root/'data/features/research_extension/bras_b4096_k16_s20260916/bank.npz'
    require(sha(path)==read(path.with_suffix('.json'))['sha256'],'Training bank SHA mismatch')
    with np.load(path,allow_pickle=False) as a:bank={k:a[k] for k in a.files}
    bf=pd.DataFrame({k:v for k,v in bank.items() if k not in ['text','image']});q=f.copy()
    q['sample_id']='verite:'+q.row_id;q['group_id']='verite:'+q.family_id
    q['text_key']=q.caption.str.casefold();q['input_hash']=[hashlib.sha256((r.caption+'\0'+r.sha256).encode()).hexdigest() for r in q.itertuples()]
    start=time.perf_counter();m,n,e=margins(z['clip_text'],z['clip_image'],q,bank['text'],bank['image'],bf,k=16,batch_size=64)
    require(np.allclose(m[:,0],z['similarity'][:,0],atol=1e-6),'Pinned scalar/vector cosine disagreement')
    out=f[['row_id','category','binary_label','family_id','cluster_id','source','independence_sensitivity_eligible']].copy()
    out['cosine']=z['similarity'][:,0];out['m_text']=m[:,1];out['m_image']=m[:,2];out['bras_mean']=(m[:,1]+m[:,2])/2
    out['valid_bank_candidates']=n;out['effective_k']=e
    return out,{'seconds':time.perf_counter()-start,'train_only_bank':bool(bf.split.eq('train').all()),'bank_size':len(bf),'rows_with_exclusions':int((n<len(bf)).sum()),'minimum_valid':int(n.min())}

def infer_heads(s,z,bras,seed):
    import torch
    from ..training import make_head
    normal=np.load(s.root/'models/normalization.npz',allow_pickle=False)
    extra=np.load(s.root/'models/research_extension/bras_b4096_k16_s20260916/normalization.npz',allow_pickle=False)
    out={};start=time.perf_counter()
    # Never recompute/fill a scaler from external rows.
    for k in ['text','image','similarity']:
        require(np.array_equal(normal[k+'_mean'],extra[k+'_mean']) and np.array_equal(normal[k+'_std'],extra[k+'_std']),'Historical and BRAS base normalizers differ')
    normalized={k:torch.from_numpy(((z[k]-normal[k+'_mean'])/normal[k+'_std']).astype(np.float32)) for k in ['text','image','similarity']}
    for kind in ['A','B','C','D']:
        if kind in ['A','B']:
            old='baseline' if kind=='A' else 'proposed';path=s.root/f'models/final/{old}_{seed}.pt'
            ck=torch.load(path,map_location='cpu',weights_only=True);require(ck['model']==old and ck['seed']==seed,'Historical checkpoint identity mismatch')
            features=normalized['similarity']
        else:
            path=s.root/f'models/research_extension/bras_b4096_k16_s20260916/{kind}_{seed}.pt';receipt=read(path.with_suffix('.json'))
            require(sha(path)==receipt['sha256'],'BRAS checkpoint hash differs')
            ck=torch.load(path,map_location='cpu',weights_only=True);require(ck['binding']==receipt['binding'] and ck['binding']['variant']==kind and ck['binding']['seed']==seed,'BRAS checkpoint binding mismatch')
            raw=bras[['m_text','m_image']].to_numpy(np.float32) if kind=='C' else bras[['cosine','m_text','m_image']].to_numpy(np.float32)
            features=torch.from_numpy(((raw-extra[kind+'_mean'])/extra[kind+'_std']).astype(np.float32))
        model=make_head(kind,ck['config'],s.root);model.load_state_dict(ck['state_dict']);model.eval().requires_grad_(False)
        with torch.inference_mode():
            logits=torch.cat([model(normalized['text'][i:i+512],normalized['image'][i:i+512],features[i:i+512]) for i in range(0,len(features),512)])
        require(torch.isfinite(logits).all(),'Invalid frozen-head logits')
        out[kind]={'probability':torch.softmax(logits,dim=1)[:,1].numpy(),'decision':logits.argmax(1).numpy(),'logits':logits.numpy()}
    return out,time.perf_counter()-start

def evaluate(s):
    import torch,joblib
    from .features import load,validate_inputs
    from ..ccr.core import Study as CCRStudy
    from ..ccr.fit import verify_frozen,predict_policy
    s.frozen();verify_sources(s);f,z=load(s);validate_inputs(s,f);torch.set_num_threads(6)
    scores,bank_info=alignment_scores(s,f,z);path=s.results/'alignment_predictions.csv';csv(path,scores);seal(path,s.binding())
    frozen_ccr=verify_frozen(CCRStudy(s.root));profile={'bank':bank_info,'heads':{},'CCR':{}}
    for seed in SEEDS:
        out,seconds=infer_heads(s,z,scores,seed);gate=s.root/f'models/research_extension/ccr/full/gate_{seed}.joblib';policy=s.root/f'models/research_extension/ccr/full/policy_{seed}.json'
        require(sha(gate)==frozen_ccr['artifacts'][gate.relative_to(s.root).as_posix()],'CCR gate changed')
        bundle=joblib.load(gate);pol=read(policy);q=f.rename(columns={'row_id':'sample_id'})
        begin=time.perf_counter();pp,routes,utilities=predict_policy(q,out['A']['probability'],out['B']['probability'],z['similarity'][:,0],bundle,pol)
        out['CCR']={'probability':pp['CCR'],'decision':np.where(routes['CCR'],out['B']['decision'],out['A']['decision'])}
        profile['CCR'][str(seed)]={'seconds':time.perf_counter()-begin,'guard_passed':pol['guard_passed'],'equals_B':bool(np.array_equal(pp['CCR'],out['B']['probability'])),'B_route_fraction':float(routes['CCR'].mean())}
        result=scores.copy()
        for kind,values in out.items():
            result['p_'+kind]=values['probability'];result['decision_'+kind]=values['decision']
            if 'logits' in values:
                result['logit0_'+kind]=values['logits'][:,0];result['logit1_'+kind]=values['logits'][:,1]
        dest=s.results/f'predictions/seed_{seed}.csv';csv(dest,result);seal(dest,{**s.binding(),'features_sha256':sha(s.cache/'features.npz')})
        profile['heads'][str(seed)]={'load_and_inference_seconds':seconds,'models':['A','B','C','D']}
    if not (s.results/'inference_profile.json').exists():write(s.results/'inference_profile.json',profile)
    return {'rows':len(f),'seeds':SEEDS,'models':KINDS,'CCR_equals_B':{k:v['equals_B'] for k,v in profile['CCR'].items()}}
