"""Synthetic correspondence tests are never assigned fabricated veracity labels."""
from __future__ import annotations
import time
import numpy as np
import pandas as pd
import torch
from .common import (manifest,stable_rank,require,lock_protocol,digest,sha,write_json,
    read_json,load_npz,load_historical,standardize,peak_memory)
from .features import load_bank,load_generated
from .extraction import load_embeddings
from .reference import valid_candidates,margins
from .training import prepare_data,make_head,verify_completed,binding,predict
from .statistics import metrics

def choose_swaps(text,query,bank_image,bank,seed,hard=False,batch_size=64):
    rng=np.random.default_rng(seed);chosen=[]
    for start in range(0,len(query),batch_size):
        end=min(start+batch_size,len(query));valid=valid_candidates(query.iloc[start:end],bank)
        require(valid.any(1).all(),'No valid swap image available')
        if hard:
            values=text[start:end]@bank_image.T
            chosen.extend(np.argmax(np.where(valid,values,-np.inf),axis=1).tolist())
        else:
            for row in valid: chosen.append(int(rng.choice(np.flatnonzero(row))))
    return np.asarray(chosen,dtype=np.int64)

def ranking_stats(original,changed):
    delta=original-changed
    return {'n':len(delta),'mean_original_minus_swapped':float(delta.mean()),
        'original_higher_fraction':float((delta>1e-7).mean()),'tied_fraction':float((np.abs(delta)<=1e-7).mean()),
        'original_lower_fraction':float((delta < -1e-7).mean())}

def stress_test(ctx,limit=500):
    require(limit>0,'Stress limit must be positive')
    p=lock_protocol(ctx);torch.set_num_threads(p['budget']['threads']);started=time.perf_counter()
    frame,features,labels,feature_receipt=prepare_data(ctx)
    bank,bt,bv,bank_receipt=load_bank(ctx)
    g,_=load_generated(ctx,frame);normalizer=load_npz(ctx.models/'normalization.npz')
    split='validation' if ctx.smoke else 'test'
    candidates=frame[frame.split.eq(split)]
    selected=candidates.iloc[stable_rank(candidates.sample_id.tolist(),ctx.seed)[:limit]]
    indices=selected.index.to_numpy();query=selected.reset_index(drop=True)
    qt,qv=load_embeddings(ctx,query)
    old_indices=torch.tensor(indices,dtype=torch.long)
    saved_models={}
    for kind in ['B','D']:
        saved=verify_completed(ctx,kind,42,binding(ctx,kind,42,feature_receipt))
        model=make_head(kind,p['config'],ctx.project);model.load_state_dict(saved['state_dict']);model.eval()
        saved_models[kind]=model
    baseline_prob={k:predict(m,features,old_indices,k)[0] for k,m in saved_models.items()}
    stressdir=ctx.results/'stress';stressdir.mkdir(parents=True,exist_ok=True)
    evidence={'protocol_sha256':digest(p),'bank_sha256':bank_receipt['sha256'],
        'features_sha256':feature_receipt['sha256'],'query_ids':query.sample_id.tolist(),'limit':limit}
    if (stressdir/'stress.json').exists():
        r=read_json(stressdir/'stress.json');require(r['evidence']==evidence,'Stress configuration changed; use another run ID')
        for rel,h in r['outputs'].items():require(sha(stressdir/rel)==h,f'Stress output changed: {rel}')
        return {'status':'verified existing stress tests','path':str(stressdir/'stress.json')}
    # Reference rows are from the retained training set; load historical ResNet features read-only.
    bank_full=frame.set_index('sample_id').loc[bank.sample_id].reset_index()
    bank_raw=load_historical(ctx,bank_full,('image',))['image']
    records=[];rankings=[]
    for condition,hard in [('random_nonmatching',False),('hard_nonmatching',True)]:
        chosen=choose_swaps(qt,query,bv,bank,ctx.seed,hard,ctx.batch_size)
        donors=bank.iloc[chosen].reset_index(drop=True)
        # Symmetric masks prevent comparing a swapped image to its own bank entry.
        # Apply the same union of query/donor exclusions to both original and swapped pairs.
        original,_,_=margins(qt,qv,query,bt,bv,bank,ctx.top_k,ctx.batch_size,extra_exclusions=donors)
        swapped,counts,_=margins(qt,bv[chosen],query,bt,bv,bank,ctx.top_k,ctx.batch_size,extra_exclusions=donors)
        require(np.isfinite(swapped).all(),'Swap margins missing')
        for j,name in enumerate(['cosine','m_text','m_image']):
            rankings.append({'condition':condition,'feature':name,**ranking_stats(original[:,j],swapped[:,j])})
        rankings.append({'condition':condition,'feature':'sum_margins_csls_form',**ranking_stats(original[:,1:].sum(1),swapped[:,1:].sum(1))})
        details=pd.DataFrame({'sample_id':query.sample_id,'donor_id':donors.sample_id,'synthetic_condition':condition,
            'original_cosine':original[:,0],'swapped_cosine':swapped[:,0],
            'original_m_text':original[:,1],'swapped_m_text':swapped[:,1],
            'original_m_image':original[:,2],'swapped_m_image':swapped[:,2],'valid_candidates':counts})
        with torch.inference_mode():
            image=torch.from_numpy(standardize(bank_raw[chosen],{'mean':normalizer['image_mean'],'std':normalizer['image_std']}))
            for kind,model in saved_models.items():
                x=swapped[:,:1] if kind=='B' else swapped
                extra=torch.from_numpy(standardize(x,{'mean':normalizer[kind+'_mean'],'std':normalizer[kind+'_std']}))
                probability=torch.softmax(model(features['text'][old_indices],image,extra),1).numpy()
                original_prob=baseline_prob[kind]
                records.append({'condition':condition,'variant':kind,'n':len(query),
                    'task':'prediction sensitivity only; swapped pairs have no factual labels',
                    'prediction_flip_fraction':float((probability.argmax(1)!=original_prob.argmax(1)).mean()),
                    'mean_probability_fake_change':float((probability[:,0]-original_prob[:,0]).mean())})
                details[kind+'_original_probability_fake']=original_prob[:,0]
                details[kind+'_swapped_probability_fake']=probability[:,0]
        details.to_csv(stressdir/f'{condition}_pairs.csv',index=False)
    sensitivity=[]
    full=g['bras'][indices]
    # Fixed fitted D head: these are deployment perturbations, not retuned model comparisons.
    configurations=[]
    for size in sorted(set([min(256,len(bank)),min(1024,len(bank)),len(bank)])):
        configurations.append((f'bank_size_{size}',np.arange(size),ctx.top_k))
    for k in [1,4,16,64]: configurations.append((f'top_k_{k}',np.arange(len(bank)),k))
    rng=np.random.default_rng(ctx.seed)
    permutation=rng.permutation(len(bank))
    for frac in [.25,.5,.75,1.]:
        keep=permutation[:int(round(len(bank)*(1-frac)))]
        configurations.append((f'missing_fraction_{frac}',keep,ctx.top_k))
    for name,keep,k in configurations:
        start=time.perf_counter()
        x,n,e=margins(qt,qv,query,bt[keep],bv[keep],bank.iloc[keep],k,ctx.batch_size)
        missing=~np.isfinite(x)
        # Explicit neutral standardized fallback; used only for this missing-bank stress condition.
        x=np.where(missing,normalizer['D_mean'],x).astype(np.float32)
        with torch.inference_mode():
            extra=torch.from_numpy(standardize(x,{'mean':normalizer['D_mean'],'std':normalizer['D_std']}))
            probability=torch.softmax(saved_models['D'](features['text'][old_indices],features['image'][old_indices],extra),1).numpy()
        scored=metrics(query.label.to_numpy(),probability)
        sensitivity.append({'condition':name,'task':'unaltered-pair classification under bank/k perturbation',
            'n':len(query),'bank_size':len(keep),'top_k':k,'min_valid':int(n.min()),'missing_rows':int(missing.any(1).sum()),
            'fallback':'train-mean margins only when zero candidates','accuracy':scored['accuracy'],'macro_f1':scored['macro_f1'],
            'mean_abs_m_text_change':float(np.abs(x[:,1]-full[:,1]).mean()),
            'mean_abs_m_image_change':float(np.abs(x[:,2]-full[:,2]).mean()),
            'prediction_flip_fraction':float((probability.argmax(1)!=baseline_prob['D'].argmax(1)).mean()),
            'feature_seconds':time.perf_counter()-start})
    for column,name in [(1,'remove_text_conditioned_margin'),(2,'remove_image_conditioned_margin')]:
        x=full.copy();x[:,column]=normalizer['D_mean'][column]
        with torch.inference_mode():
            extra=torch.from_numpy(standardize(x,{'mean':normalizer['D_mean'],'std':normalizer['D_std']}))
            probability=torch.softmax(saved_models['D'](features['text'][old_indices],features['image'][old_indices],extra),1).numpy()
        scored=metrics(query.label.to_numpy(),probability)
        sensitivity.append({'condition':name,'task':'post-hoc fixed-head sensitivity; not a separately trained ablation',
            'n':len(query),'accuracy':scored['accuracy'],'macro_f1':scored['macro_f1'],
            'prediction_flip_fraction':float((probability.argmax(1)!=baseline_prob['D'].argmax(1)).mean())})
    pd.DataFrame(rankings).to_csv(stressdir/'correspondence_rankings.csv',index=False)
    pd.DataFrame(records).to_csv(stressdir/'swap_prediction_sensitivity.csv',index=False)
    pd.DataFrame(sensitivity).to_csv(stressdir/'bank_and_direction_sensitivity.csv',index=False)
    r={'status':'completed','scope':'exploratory stress tests; no claim of factual verification','evidence':evidence,
        'outputs':{x.name:sha(x) for x in stressdir.glob('*.csv')},
        'warning':'An original fake-labeled pair can be aligned; a nonmatching swap is not automatically fake. Ranking fractions are not fake-news accuracy.',
        'seconds':time.perf_counter()-started,**peak_memory()}
    write_json(stressdir/'stress.json',r,immutable=True)
    return {'status':'completed','path':str(stressdir/'stress.json'),'n':len(query)}
