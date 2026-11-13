"""Train-only OOF heads, utility fits and validation-only policy freezing."""
from __future__ import annotations
import copy,time
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from scipy.optimize import minimize_scalar
from scipy.special import expit,logit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
import joblib
from .core import *
from ..training import make_head,seed_all,save_torch

@torch.inference_mode()
def infer(model,x,indices=None):
    model.eval();indices=torch.arange(len(x['text'])) if indices is None else torch.as_tensor(indices,dtype=torch.long)
    out=[]
    for idx in indices.split(512):out.append(model(x['text'][idx],x['image'][idx],x['similarity'][idx]))
    logits=torch.cat(out)
    require(torch.isfinite(logits).all().item(),'Nonfinite head logits')
    return torch.softmax(logits,1)[:,1].numpy(),logits

def tensors(raw,scales):return {k:torch.from_numpy(standardize(v,scales[k])) for k,v in raw.items()}
def historical_predictions(s,frame,raw,seed):
    normal=load_npz(s.project/'models/normalization.npz')
    scales={k:{'mean':normal[k+'_mean'],'std':normal[k+'_std']} for k in raw}
    x=tensors(raw,scales);out={}
    for kind,old in [('A','baseline'),('B','proposed')]:
        path=s.project/f'models/final/{old}_{seed}.pt';saved=torch.load(path,map_location='cpu',weights_only=True)
        require(saved['seed']==seed and saved['model']==old,'Wrong historical checkpoint')
        model=make_head(kind,saved['config'],s.project);model.load_state_dict(saved['state_dict'])
        out[kind],_=infer(model,x)
    return out

def cached_train(s):
    f=train_frame(s);path=s.path('inputs/train.npz')
    if path.exists():verify(path);a=load_npz(path)
    else:
        raw=load_historical(Context(s.project,s.project),f)
        write_npz(path,ids=f.sample_id.to_numpy(dtype=str),input_hash=f.input_hash.to_numpy(dtype=str),**raw)
        seal(path,{'manifest_sha256':s.protocol['manifest_sha256'],'historical_inventory_sha256':s.protocol['historical_inventory_sha256']});a=load_npz(path)
    require(a['ids'].tolist()==f.sample_id.tolist() and a['input_hash'].tolist()==f.input_hash.tolist(),'Cached train identity mismatch')
    return f,{k:a[k] for k in ['text','image','similarity']}

def fit_head(s,frame,x,y,fit_mask,val_mask,held_mask,kind,seed,fold,normal_path,roles_path):
    path=s.models/f'oof/fold_{fold}/{kind}_{seed}.pt'
    cfg=s.protocol['head_config'];budget=dict(s.protocol['head_budget'])
    if s.smoke:budget.update(epochs=2,patience=2)
    bound={'protocol_sha256':digest(s.protocol),'train_inputs_sha256':sha(s.path('inputs/train.npz')),
           'normalization_sha256':sha(normal_path),'roles_sha256':sha(roles_path),
           'source':{k:v for k,v in source_hashes(s).items() if k.endswith(('core.py','fit.py','training.py','models.py','common.py','statistics.py'))},
           'seed':seed,'fold':fold,'kind':kind,'smoke':s.smoke}
    if path.exists():
        verify(path);saved=torch.load(path,map_location='cpu',weights_only=True)
        require(saved['binding']==bound,'Completed OOF head provenance changed');return saved
    seed_all(seed);model=make_head(kind,cfg,s.project)
    opt=torch.optim.AdamW(model.parameters(),lr=cfg['lr'],weight_decay=cfg['weight_decay'])
    train=torch.tensor(np.flatnonzero(fit_mask));val=np.flatnonzero(val_mask)
    generator=torch.Generator().manual_seed(seed);best=None;best_f1=-1.;best_loss=float('inf');stale=0;best_epoch=0;history=[]
    resume=path.with_suffix('.resume.pt');start_epoch=1;elapsed=0.
    if resume.exists():
        state=torch.load(resume,map_location='cpu',weights_only=True);require(state['binding']==bound,'OOF resume mismatch')
        model.load_state_dict(state['current']);opt.load_state_dict(state['optimizer'])
        generator.set_state(state['order_rng']);torch.set_rng_state(state['torch_rng'])
        best=state['best'];best_f1=state['best_f1'];best_loss=state['best_loss'];best_epoch=state['best_epoch']
        stale=state['stale'];history=state['history'];start_epoch=state['epoch']+1;elapsed=state['seconds']
    start=time.perf_counter()
    for epoch in range(start_epoch,budget['epochs']+1):
        if stale>=budget['patience']:break
        model.train();order=train[torch.randperm(len(train),generator=generator)];loss_sum=0.
        for idx in order.split(budget['batch_size']):
            opt.zero_grad(set_to_none=True);z=model(x['text'][idx],x['image'][idx],x['similarity'][idx])
            loss=F.cross_entropy(z,y[idx]);require(torch.isfinite(loss).item(),'Nonfinite OOF fit')
            loss.backward();opt.step();loss_sum+=float(loss.detach())*len(idx)
        p,z=infer(model,x,val);vl=float(F.cross_entropy(z,y[val]));vf=float(fast_macro_f1(y[val].numpy(),z.argmax(1).numpy()))
        history.append({'epoch':epoch,'fit_loss':loss_sum/len(train),'inner_validation_loss':vl,'inner_validation_macro_f1':vf})
        if vf>best_f1+1e-12 or (abs(vf-best_f1)<=1e-12 and vl<best_loss-1e-8):
            best=copy.deepcopy(model.state_dict());best_f1,best_loss,best_epoch,stale=vf,vl,epoch,0
        else:stale+=1
        save_torch(resume,{'binding':bound,'current':model.state_dict(),'optimizer':opt.state_dict(),'order_rng':generator.get_state(),'torch_rng':torch.get_rng_state(),
            'best':best,'best_f1':best_f1,'best_loss':best_loss,'best_epoch':best_epoch,'stale':stale,'history':history,'epoch':epoch,'seconds':elapsed+time.perf_counter()-start})
    saved={'binding':bound,'state_dict':best,'best_epoch':best_epoch,'best_inner_validation_macro_f1':best_f1,'history':history,
           'seconds':elapsed+time.perf_counter()-start,'config':cfg,'budget':budget,'fit_rows':int(fit_mask.sum()),'inner_val_rows':int(val_mask.sum()),'held_rows':int(held_mask.sum())}
    save_torch(path,saved);seal(path,bound)
    print(f'OOF fold={fold} seed={seed} {kind}: epochs={len(history)} best={best_epoch} seconds={saved["seconds"]:.1f}',flush=True)
    return saved

def oof(s):
    require((s.docs/'audit.md').exists() and (s.docs/'preregistered_protocol.md').exists(),'Pretraining audit/protocol missing')
    torch.set_num_threads(s.protocol['cpu_threads']);start=time.perf_counter()
    f,raw=cached_train(s);folds=fold_assign(f.group_id,s.nfolds,s.protocol['fold_seed'])
    assignments=f[['sample_id','group_id','split','input_hash']].copy();assignments['fold']=folds
    assignment_path=s.path('fold_assignments.csv')
    if assignment_path.exists():verify(assignment_path);require(pd.read_csv(assignment_path,dtype=str).sample_id.tolist()==f.sample_id.tolist(),'Fold IDs changed')
    else:save_csv(assignments,assignment_path,{'protocol_sha256':digest(s.protocol)})
    y=torch.tensor(f.label.to_numpy(),dtype=torch.long);preds={seed:{k:np.full(len(f),np.nan) for k in ['A','B']} for seed in s.seeds}
    run_summaries=[]
    for fold in range(s.nfolds):
        fit,val,held=partition(f,folds,fold,s.protocol['fold_seed'])
        roles=assignments.copy();roles['role']=np.where(held,'held',np.where(val,'inner_validation','fit'))
        roles_path=s.path(f'partitions/fold_{fold}.csv')
        if roles_path.exists():verify(roles_path);require(pd.read_csv(roles_path).role.tolist()==roles.role.tolist(),'Fold role changed')
        else:save_csv(roles,roles_path)
        scales={k:fit_scaler(v,fit) for k,v in raw.items()}
        normal=s.models/f'oof/fold_{fold}/normalization.npz';arrays={f'{k}_{t}':v for k,sc in scales.items() for t,v in sc.items()}
        if normal.exists():
            verify(normal);old=load_npz(normal);require(all(np.array_equal(old[k],v) for k,v in arrays.items()),'OOF normalizer mismatch')
        else:write_npz(normal,**arrays);seal(normal,{'fit_ids_sha256':digest(f.loc[fit,'sample_id'].tolist()),'roles_sha256':sha(roles_path)})
        x=tensors(raw,scales)
        for seed in s.seeds:
            for kind in ['A','B']:
                saved=fit_head(s,f,x,y,fit,val,held,kind,seed,fold,normal,roles_path)
                model=make_head(kind,saved['config'],s.project);model.load_state_dict(saved['state_dict'])
                preds[seed][kind][held],_=infer(model,x,np.flatnonzero(held))
                run_summaries.append({'fold':fold,'seed':seed,'kind':kind,**{k:saved[k] for k in ['best_epoch','seconds','fit_rows','inner_val_rows','held_rows']}})
        del x
    for seed in s.seeds:
        out=f[['sample_id','group_id','split','label','input_hash']].copy();out['fold']=folds;out['cosine']=raw['similarity'][:,0]
        for k in ['A','B']:require(np.isfinite(preds[seed][k]).all(),'Incomplete OOF coverage');out['p_'+k]=preds[seed][k]
        out['utility']=ce(out.label.to_numpy(),out.p_A.to_numpy())-ce(out.label.to_numpy(),out.p_B.to_numpy())
        path=s.path(f'oof/{seed}.csv')
        if path.exists():verify(path)
        else:save_csv(out,path,{'folds_sha256':sha(assignment_path),'input_sha256':sha(s.path('inputs/train.npz')),
             'heads':{f'{k}_{fold}':sha(s.models/f'oof/fold_{fold}/{k}_{seed}.pt') for fold in range(s.nfolds) for k in ['A','B']}})
    write_json(s.path('oof_complete.json'),{'created_utc':now(),'rows':len(f),'folds':s.nfolds,'runs':run_summaries,'seconds':time.perf_counter()-start,'environment':environment(),**peak_memory()})
    return {'oof_rows':len(f),'head_fits':len(run_summaries)}

def model_scores(bundle,z):
    return {'intercept':np.full(len(z),bundle['intercept']),'ridge_candidate':bundle['ridge'].predict(z),'tree':bundle['tree'].predict(z),
            'cosine_gate':z[:,9],'confidence_gate':.5*(z[:,6]-z[:,5]),'entropy_gate':z[:,7]-z[:,8]}
def choose_threshold(train_score,val_score,y,a,b):
    grid=np.unique(np.r_[-1e30,1e30,0.,np.quantile(train_score,np.linspace(0,1,21))]);records=[]
    for threshold in grid:
        route=val_score>threshold;p=np.where(route,b,a)
        records.append({'threshold':float(threshold),'log_loss':float(ce(y,p).mean()),'b_fraction':float(route.mean())})
    best=min(records,key=lambda r:(r['log_loss'],-r['b_fraction'],r['threshold']))
    return best,records
def predict_policy(frame,p_a,p_b,cosine,bundle,policy):
    z=gate_features(p_a,p_b,cosine);gs=model_scores(bundle,z);out={'A':p_a,'B':p_b}
    routes={'A':np.zeros(len(z),bool),'B':np.ones(len(z),bool)}
    for k in ['A','B']:
        out[k+'_temperature']=expit(logit(prob(out[k]))/policy['temperatures'][k]);routes[k+'_temperature']=routes[k]
    for k,g in gs.items():
        routes[k]=g>policy['thresholds'][k]['threshold'];out[k]=np.where(routes[k],p_b,p_a)
    routes['random_gate']=np.array([uniform(x,'ccr-random-'+str(policy['classifier_seed'])) for x in frame.sample_id])<policy['random_b_fraction']
    out['random_gate']=np.where(routes['random_gate'],p_b,p_a)
    routes['CCR']=routes['ridge_candidate'] if policy['guard_passed'] else routes['B']
    out['CCR']=np.where(routes['CCR'],p_b,p_a)
    return out,routes,gs

def fit_gates(s):
    torch.set_num_threads(s.protocol['cpu_threads']);f=split_frame(s,'validation')
    if s.smoke:f=f.iloc[:100].copy()
    raw=load_historical(Context(s.project,s.project),f);all_paths=[]
    for seed in s.seeds:
        source=s.path(f'oof/{seed}.csv');verify(source);o=pd.read_csv(source,dtype={'sample_id':str,'group_id':str})
        require(o.split.eq('train').all() and o.sample_id.is_unique,'Gate fit data must be unique official train OOF')
        z=gate_features(o.p_A,o.p_B,o.cosine);u=ce(o.label.to_numpy(),o.p_A)-ce(o.label.to_numpy(),o.p_B)
        require(np.allclose(u,o.utility,atol=1e-12),'Saved OOF utility mismatch')
        path=s.models/f'gate_{seed}.joblib';policy_path=s.models/f'policy_{seed}.json'
        if path.exists():verify(path);bundle=joblib.load(path)
        else:
            bundle={'ridge':make_pipeline(StandardScaler(),Ridge(alpha=100.)).fit(z,u),
                    'tree':DecisionTreeRegressor(max_depth=3,min_samples_leaf=20 if s.smoke else 500,random_state=seed).fit(z,u),
                    'intercept':float(u.mean()),'feature_names':FEATURE_NAMES,'oof_sha256':sha(source)}
            path.parent.mkdir(parents=True,exist_ok=True);joblib.dump(bundle,path);seal(path,{'oof_sha256':sha(source),'protocol_sha256':digest(s.protocol)})
        require(bundle['feature_names']==FEATURE_NAMES and bundle['oof_sha256']==sha(source),'Gate input/provenance mismatch')
        p=historical_predictions(s,f,raw,seed);y=f.label.to_numpy();vz=gate_features(p['A'],p['B'],raw['similarity'][:,0])
        ts=model_scores(bundle,z);vs=model_scores(bundle,vz)
        thresholds={};search=[]
        for k in ts:
            best,rows=choose_threshold(ts[k],vs[k],y,p['A'],p['B']);thresholds[k]=best
            search.extend({'variant':k,**r} for r in rows)
        candidate=np.where(vs['ridge_candidate']>thresholds['ridge_candidate']['threshold'],p['B'],p['A'])
        guard=paired_stats(y,p['B'],candidate,s.reps,s.protocol['bootstrap']['seed'])
        temps={}
        for k in ['A','B']:
            result=minimize_scalar(lambda lt:ce(y,expit(logit(prob(p[k]))/np.exp(lt))).mean(),bounds=(np.log(.05),np.log(20.)),method='bounded',options={'xatol':1e-10})
            require(result.success,'Temperature optimizer did not converge');temps[k]=float(np.exp(result.x))
        policy={'classifier_seed':seed,'protocol_sha256':digest(s.protocol),'oof_sha256':sha(source),'gate_sha256':sha(path),
             'temperatures':temps,'thresholds':thresholds,'guard':guard,'guard_passed':guard['ci95']['log_loss'][1]<0.,
             'random_b_fraction':thresholds['ridge_candidate']['b_fraction'],'cosine_quartiles':np.quantile(o.cosine,[.25,.5,.75]).tolist()}
        out,routes,gs=predict_policy(f,p['A'],p['B'],raw['similarity'][:,0],bundle,policy)
        policy['coverage_cutoffs']={k:{str(c):0. if c==1. else float(np.quantile(np.maximum(v,1-v),1-c)) for c in [.5,.6,.7,.8,.9,.95,1.]} for k,v in out.items()}
        policy['utility_score_check']={k:{'oof_mean':float(ts[k].mean()),'oof_sd':float(ts[k].std()),'validation_mean':float(vs[k].mean()),'validation_sd':float(vs[k].std()),
             'validation_utility_mae':float(np.abs(vs[k]-(ce(y,p['A'])-ce(y,p['B']))).mean())} for k in ['intercept','ridge_candidate','tree']}
        write_json(policy_path,policy,immutable=True)
        if not policy_path.with_suffix('.json.receipt.json').exists():seal(policy_path,{'gate_sha256':sha(path),'validation_manifest_ids':digest(f.sample_id.tolist())})
        valout=f[['sample_id','label','group_id','subreddit','label6']].copy();valout['cosine']=raw['similarity'][:,0]
        for k in out:valout['p_'+k]=out[k];valout['use_B_'+k]=routes[k]
        for k in gs:valout['utility_'+k]=gs[k]
        vpath=s.path(f'validation/{seed}.csv')
        if not vpath.exists():save_csv(valout,vpath,{'policy_sha256':sha(policy_path)})
        search_path=s.path(f'validation/threshold_search_{seed}.csv')
        if not search_path.exists():save_csv(pd.DataFrame(search),search_path)
        metrics=[{'variant':k,'classifier_seed':seed,'b_fraction':float(routes[k].mean()),**scores(y,v)} for k,v in out.items()]
        metric_path=s.path(f'validation/metrics_{seed}.csv')
        if not metric_path.exists():save_csv(pd.DataFrame(metrics),metric_path)
        all_paths.extend([path,policy_path,source,vpath])
        print(f'Frozen seed={seed}: ridge validation B fraction={policy["random_b_fraction"]:.4f}; guard_passed={policy["guard_passed"]}; logloss delta={guard["delta_candidate_minus_base"]["log_loss"]:.6f}',flush=True)
    frozen=s.path('frozen_policy.json')
    contract={'protocol_sha256':digest(s.protocol),'sources':source_hashes(s),'artifacts':{p.relative_to(s.project).as_posix():sha(p) for p in all_paths},
         'final_checkpoints':{f'{k}_{seed}':sha(s.project/f'models/final/{k}_{seed}.pt') for k in ['baseline','proposed'] for seed in s.seeds},
         'normalization_sha256':sha(s.project/'models/normalization.npz'),'scope':'frozen before any CCR test inference','smoke':s.smoke}
    if frozen.exists():old=read_json(frozen);require(old['contract']==contract,'Frozen policy changed')
    else:write_json(frozen,{'created_utc':now(),'contract':contract},immutable=True)
    return {'frozen':str(frozen),'seeds':s.seeds}

def verify_frozen(s):
    f=read_json(s.path('frozen_policy.json'))['contract'];require(f['protocol_sha256']==digest(s.protocol),'Protocol changed after freeze')
    require(f['sources']==source_hashes(s),'CCR code changed after freeze; record bug-fix amendment and explicitly refreeze before reevaluation')
    for rel,h in f['artifacts'].items():require(sha(s.project/rel)==h,f'Frozen artifact changed: {rel}')
    for k,h in f['final_checkpoints'].items():require(sha(s.project/f'models/final/{k}.pt')==h,'Historical checkpoint changed')
    require(sha(s.project/'models/normalization.npz')==f['normalization_sha256'],'Historical normalizer changed')
    return f
