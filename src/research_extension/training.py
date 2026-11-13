"""Matched frozen-feature heads, immutable completions and epoch-level resume."""
from __future__ import annotations
import copy
import importlib.util
import random
import time
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from .common import (SEEDS,manifest,require,lock_protocol,digest,sha,read_json,write_json,
    write_npz,load_npz,load_historical,fit_scaler,standardize,peak_memory,environment)
from .features import load_generated

VARIANTS={'A':0,'B':1,'C':2,'D':3,'E':3,'F':1,'G':3,'I':3}

def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)

def original_classifier(project):
    spec=importlib.util.spec_from_file_location('bras_historical_models',project/'src/models.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module.Classifier

class SimilarityHead(nn.Module):
    def __init__(self,width,dimension,dropout):
        super().__init__()
        self.hidden=nn.Linear(width,dimension);self.dropout=nn.Dropout(dropout);self.output=nn.Linear(dimension,2)
    def forward(self,text,image,similarity):
        return self.output(self.dropout(torch.relu(self.hidden(similarity))))

def make_head(kind,config,project):
    require(kind in VARIANTS,f'Unknown variant {kind}')
    width=VARIANTS[kind]
    if kind in ['F','G']:
        return SimilarityHead(width,config['dimension'],config['dropout'])
    base=original_classifier(project)
    model=base('baseline',config['dimension'],config['dropout'])
    if width:
        # The original baseline forward supports similarity via kind='proposed'.
        model.fusion.weight=nn.Parameter(torch.cat([model.fusion.weight,torch.zeros(config['dimension'],width)],1))
        model.fusion.in_features+=width
        model.kind='proposed'
    return model

def extras(kind,generated,raw):
    if kind=='A': return np.zeros((len(raw),0),np.float32)
    if kind in ['B','F']: return raw
    if kind=='C': return generated['bras'][:,1:]
    if kind in ['D','G']: return generated['bras']
    if kind=='E': return np.column_stack([raw,generated['random']]).astype(np.float32)
    if kind=='I': return np.column_stack([raw,generated['unimodal']]).astype(np.float32)
    raise ValueError(kind)

def prepare_data(ctx):
    frame=manifest(ctx); raw=load_historical(ctx,frame)
    generated,receipt=load_generated(ctx,frame)
    train=frame.split.eq('train').to_numpy()
    old=load_npz(ctx.project/'models/normalization.npz')
    scalers={}
    normalized={}
    for key in raw:
        if ctx.smoke:
            scale=fit_scaler(raw[key],train)
        else:
            scale={'mean':old[key+'_mean'],'std':old[key+'_std']}
            computed=fit_scaler(raw[key],train)
            require(np.allclose(computed['mean'],scale['mean'],rtol=0,atol=1e-6) and
                    np.allclose(computed['std'],scale['std'],rtol=0,atol=1e-6),'Historical normalization not equal to train-only fit')
        scalers[key]=scale
        normalized[key]=torch.from_numpy(standardize(raw[key],scale))
    scale_path=ctx.models/'normalization.npz'
    arrays={}
    for kind in VARIANTS:
        x=extras(kind,generated,raw['similarity'])
        scale=fit_scaler(x,train)
        # B and raw column of other variants use exactly the historical scalar normalization.
        if kind!='A' and kind!='C':
            scale['mean'][0]=scalers['similarity']['mean'][0]
            scale['std'][0]=scalers['similarity']['std'][0]
        arrays[kind+'_mean']=scale['mean'];arrays[kind+'_std']=scale['std']
    for key,scale in scalers.items():
        arrays[key+'_mean']=scale['mean'];arrays[key+'_std']=scale['std']
    if scale_path.exists():
        prior=load_npz(scale_path)
        require(set(prior)==set(arrays) and all(np.array_equal(prior[k],v) for k,v in arrays.items()),'Normalizer differs on resume')
    else: write_npz(scale_path,**arrays)
    normalized['variants']={kind:torch.from_numpy(standardize(extras(kind,generated,raw['similarity']),
        {'mean':arrays[kind+'_mean'],'std':arrays[kind+'_std']})) for kind in VARIANTS}
    labels=torch.tensor(frame.label.to_numpy(),dtype=torch.long)
    return frame,normalized,labels,receipt

@torch.inference_mode()
def predict(model,features,indices,kind):
    model.eval()
    chunks=[]
    for idx in indices.split(512):
        z=model(features['text'][idx],features['image'][idx],features['variants'][kind][idx])
        require(torch.isfinite(z).all().item(),'Nonfinite logits')
        chunks.append(z)
    logits=torch.cat(chunks)
    return torch.softmax(logits,dim=1).numpy(),logits

def checkpoint_path(ctx,kind,seed):
    return ctx.models/f'{kind}_{seed}.pt'

def binding(ctx,kind,seed,receipt):
    return {'variant':kind,'seed':seed,'protocol_sha256':digest(lock_protocol(ctx)),
            'features_sha256':receipt['sha256'],'normalization_sha256':sha(ctx.models/'normalization.npz'),
            'historical_model_source_sha256':sha(ctx.project/'src/models.py')}

def save_torch(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.part');torch.save(value,tmp);tmp.replace(path)

def verify_completed(ctx,kind,seed,expected):
    path=checkpoint_path(ctx,kind,seed)
    receipt=read_json(path.with_suffix('.json'))
    require(receipt['sha256']==sha(path),'Checkpoint SHA mismatch')
    saved=torch.load(path,map_location='cpu',weights_only=True)
    require(saved['binding']==expected, 'Checkpoint binding mismatch; use a new run ID')
    return saved

def fit_one(ctx,frame,features,labels,kind,seed,receipt,interrupt_after=None):
    from .statistics import fast_macro_f1
    p=lock_protocol(ctx); cfg=p['config']; budget=p['budget']
    seed_all(seed);torch.set_num_threads(budget['threads'])
    model=make_head(kind,cfg,ctx.project)
    optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['lr'],weight_decay=cfg['weight_decay'])
    train=torch.tensor(np.flatnonzero(frame.split.eq('train')),dtype=torch.long)
    val=torch.tensor(np.flatnonzero(frame.split.eq('validation')),dtype=torch.long)
    generator=torch.Generator().manual_seed(seed)
    bound=binding(ctx,kind,seed,receipt)
    path=checkpoint_path(ctx,kind,seed)
    if path.exists(): return verify_completed(ctx,kind,seed,bound)
    resume=path.with_suffix('.resume.pt')
    best_score,best_loss,best_epoch,stale=-1.,float('inf'),0,0
    best_state=None;history=[];elapsed_before=0.;start_epoch=1
    if resume.exists():
        saved=torch.load(resume,map_location='cpu',weights_only=True)
        require(saved['binding']==bound,'Resume checkpoint configuration mismatch')
        model.load_state_dict(saved['state_dict']);optimizer.load_state_dict(saved['optimizer'])
        generator.set_state(saved['order_rng']);torch.set_rng_state(saved['torch_rng'])
        best_score,best_loss,best_epoch,stale=saved['best_score'],saved['best_loss'],saved['best_epoch'],saved['stale']
        best_state=saved['best_state'];history=saved['history'];elapsed_before=saved['seconds'];start_epoch=saved['epoch']+1
    started=time.perf_counter()
    for epoch in range(start_epoch,budget['epochs']+1):
        if stale>=budget['patience']: break
        model.train();order=train[torch.randperm(len(train),generator=generator)];total=0.
        for idx in order.split(budget['batch_size']):
            optimizer.zero_grad(set_to_none=True)
            z=model(features['text'][idx],features['image'][idx],features['variants'][kind][idx])
            loss=F.cross_entropy(z,labels[idx]);require(torch.isfinite(loss).item(),'Nonfinite training loss')
            loss.backward();optimizer.step();total+=float(loss.detach())*len(idx)
        prob,z=predict(model,features,val,kind);val_loss=float(F.cross_entropy(z,labels[val]))
        score=float(fast_macro_f1(labels[val].numpy(),prob.argmax(1)))
        history.append({'epoch':epoch,'train_loss':total/len(train),'validation_loss':val_loss,'validation_macro_f1':score})
        if score>best_score+1e-12 or (abs(score-best_score)<=1e-12 and val_loss<best_loss-1e-8):
            best_score,best_loss,best_epoch,stale=score,val_loss,epoch,0
            best_state=copy.deepcopy(model.state_dict())
        else: stale+=1
        seconds=elapsed_before+time.perf_counter()-started
        save_torch(resume,{'binding':bound,'state_dict':model.state_dict(),'optimizer':optimizer.state_dict(),
            'order_rng':generator.get_state(),'torch_rng':torch.get_rng_state(),'best_score':best_score,
            'best_loss':best_loss,'best_epoch':best_epoch,'stale':stale,'best_state':best_state,
            'history':history,'epoch':epoch,'seconds':seconds})
        print(f'{kind} seed {seed} epoch {epoch}: val Macro-F1 {score:.6f}; best {best_score:.6f}',flush=True)
        if interrupt_after is not None and epoch==interrupt_after:
            return {'interrupted_for_test':True}
    result={'binding':bound,'state_dict':best_state,'best_epoch':best_epoch,
        'best_validation_macro_f1':best_score,'best_validation_loss':best_loss,'epochs_run':len(history),
        'parameters':sum(q.numel() for q in model.parameters()),'seconds':elapsed_before+time.perf_counter()-started,
        'config':cfg,'budget':budget,'history':history,'origin':'new fit','memory':peak_memory()}
    save_torch(path,result)
    pd.DataFrame(history).to_csv(path.with_suffix('.csv'),index=False)
    write_json(path.with_suffix('.json'),{'sha256':sha(path),'binding':bound},immutable=True)
    # Keep the last resume state as evidence; no historical or completed files are deleted.
    return result

def reuse_historical(ctx,kind,seed,receipt):
    oldkind={'A':'baseline','B':'proposed'}[kind]
    original=ctx.project/f'models/final/{oldkind}_{seed}.pt'
    saved=torch.load(original,map_location='cpu',weights_only=True)
    p=lock_protocol(ctx)
    require(saved['config']==p['config'] and saved['budget']==p['budget'] and saved['seed']==seed,'Historical training settings differ')
    require(saved['manifest_sha256']==p['manifest_sha256'] and saved['normalization_sha256']==p['normalization_sha256'],'Historical provenance mismatch')
    require(saved['model']==oldkind,'Historical model variant mismatch')
    result={**saved,'binding':binding(ctx,kind,seed,receipt),'origin':'reused historical checkpoint; no retraining',
        'original_path':str(original),'original_sha256':sha(original),'seconds':saved['runtime_seconds']}
    path=checkpoint_path(ctx,kind,seed)
    if path.exists(): return verify_completed(ctx,kind,seed,result['binding'])
    save_torch(path,result)
    write_json(path.with_suffix('.json'),{'sha256':sha(path),'binding':result['binding']},immutable=True)
    return result

def train(ctx,variants=None):
    variants=list(VARIANTS) if variants is None else variants
    require(all(x in VARIANTS for x in variants),'Unknown training variants')
    p=lock_protocol(ctx)
    frame,features,labels,receipt=prepare_data(ctx)
    summaries=[]
    for kind in variants:
        for seed in p['seeds']:
            result=reuse_historical(ctx,kind,seed,receipt) if kind in ['A','B'] and not ctx.smoke else fit_one(ctx,frame,features,labels,kind,seed,receipt)
            summaries.append({k:v for k,v in result.items() if k not in ['state_dict','feature_contract','history']})
    record={'scope':'validation-only smoke' if ctx.smoke else 'full training; test not evaluated',
        'runs':summaries,'environment':environment(),'protocol_sha256':digest(p)}
    write_json(ctx.results/'training.json',record)
    return {'runs':len(summaries),'scope':record['scope'],'path':str(ctx.results/'training.json')}

def smoke_test(ctx):
    from .features import build_bank,generate
    require(ctx.smoke,'Smoke context required')
    build_bank(ctx);generate(ctx);train(ctx)
    frame,features,labels,receipt=prepare_data(ctx)
    p=lock_protocol(ctx);val=torch.tensor(np.flatnonzero(frame.split.eq('validation')))
    checks=[]
    for kind in VARIANTS:
        saved=verify_completed(ctx,kind,42,binding(ctx,kind,42,receipt))
        model=make_head(kind,p['config'],ctx.project);model.load_state_dict(saved['state_dict'])
        prob,z=predict(model,features,val,kind)
        from .statistics import fast_macro_f1
        score=float(fast_macro_f1(labels[val].numpy(),prob.argmax(1)))
        require(abs(score-saved['best_validation_macro_f1'])<1e-12,'Reloaded validation metric mismatch')
        checks.append({'variant':kind,'validation_macro_f1':score,'validation_samples':len(val),'parameters':saved['parameters']})
    result={'status':'passed','scope':'256 training / 64 validation only; not scientific test results',
            'checkpoint_reload_checks':checks,'protocol_sha256':digest(p)}
    write_json(ctx.results/'smoke.json',result)
    return result
