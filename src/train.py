"""Validation-only search, matched training, checkpointing and frozen configuration."""
from __future__ import annotations

import copy
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.nn import functional as F

from .feature_extractors import load_features
from .models import Classifier, NAMES, verify_matched_initialization
from .utils import atomic_json, file_hash, log_stage

SEEDS = [42,123,2026]
SEARCH = [
    {'dimension':128,'dropout':.3,'lr':3e-4,'weight_decay':1e-4},
    {'dimension':128,'dropout':.3,'lr':1e-3,'weight_decay':1e-4},
    {'dimension':256,'dropout':.3,'lr':3e-4,'weight_decay':1e-4},
    {'dimension':128,'dropout':.5,'lr':3e-4,'weight_decay':1e-3},
]
BUDGET = {'batch_size':128,'epochs':40,'patience':7,'optimizer':'AdamW',
          'criterion':'validation macro_f1; validation loss tie-break', 'threads':6}


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def normalized_features(root: Path, smoke: bool = False, fit: bool = True):
    frame, raw = load_features(root,smoke)
    suffix = '_smoke' if smoke else ''
    path = root / f'models/normalization{suffix}.npz'
    path.parent.mkdir(parents=True,exist_ok=True)
    train = frame.split.eq('train').to_numpy()
    parameters = {}
    if fit:
        for key,value in raw.items():
            parameters[key+'_mean'] = value[train].mean(axis=0,dtype=np.float64).astype(np.float32)
            parameters[key+'_std'] = np.maximum(value[train].std(axis=0,dtype=np.float64),1e-6).astype(np.float32)
        with path.with_suffix('.part').open('wb') as stream:
            np.savez_compressed(stream,**parameters)
        path.with_suffix('.part').replace(path)
    else:
        with np.load(path,allow_pickle=False) as saved:
            parameters = {key:saved[key] for key in saved.files}
    tensors = {key:torch.from_numpy((value-parameters[key+'_mean'])/parameters[key+'_std']) for key,value in raw.items()}
    assert all(torch.isfinite(x).all() for x in tensors.values())
    labels = torch.tensor(frame.label.to_numpy(),dtype=torch.long)
    return frame,tensors,labels


@torch.inference_mode()
def predict(model, features, labels, indices):
    model.eval()
    logits = torch.cat([model(*(features[k][idx] for k in ['text','image','similarity']))
                        for idx in indices.split(512)])
    assert torch.isfinite(logits).all(), 'Nonfinite classifier logits'
    loss = float(F.cross_entropy(logits,labels[indices]))
    return torch.softmax(logits,dim=1).numpy(),loss


def classification_scores(y, prediction):
    return {'accuracy':float(accuracy_score(y,prediction)),
            'precision':float(precision_score(y,prediction,pos_label=0,zero_division=0)),
            'recall':float(recall_score(y,prediction,pos_label=0,zero_division=0)),
            'f1':float(f1_score(y,prediction,pos_label=0,zero_division=0)),
            'macro_f1':float(f1_score(y,prediction,average='macro',zero_division=0))}


def fit_one(root,frame,features,labels,kind,seed,config,budget,destination,smoke=False):
    started = time.monotonic()
    seed_everything(seed)
    torch.set_num_threads(budget['threads'])
    model = Classifier(kind,config['dimension'],config['dropout'])
    optimizer = torch.optim.AdamW(model.parameters(),lr=config['lr'],weight_decay=config['weight_decay'])
    train_indices = torch.from_numpy(np.flatnonzero(frame.split.eq('train').to_numpy()))
    val_indices = torch.from_numpy(np.flatnonzero(frame.split.eq('validation').to_numpy()))
    best_score,best_loss,best_epoch,stale,state = -1.,float('inf'),0,0,None
    history = []
    generator = torch.Generator().manual_seed(seed)
    for epoch in range(1,budget['epochs']+1):
        model.train()
        shuffled = train_indices[torch.randperm(len(train_indices),generator=generator)]
        total_loss = 0.
        for indices in shuffled.split(budget['batch_size']):
            optimizer.zero_grad(set_to_none=True)
            logits = model(*(features[key][indices] for key in ['text','image','similarity']))
            loss = F.cross_entropy(logits,labels[indices])
            assert torch.isfinite(loss), 'Nonfinite training loss'
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach())*len(indices)
        probability,val_loss = predict(model,features,labels,val_indices)
        scores = classification_scores(labels[val_indices].numpy(),probability.argmax(axis=1))
        history.append({'epoch':epoch,'train_loss':total_loss/len(train_indices),'validation_loss':val_loss,
                        **{'validation_'+key:value for key,value in scores.items()}})
        score = scores['macro_f1']
        if score>best_score+1e-12 or (abs(score-best_score)<=1e-12 and val_loss<best_loss-1e-8):
            best_score,best_loss,best_epoch,stale = score,val_loss,epoch,0
            state = copy.deepcopy(model.state_dict())
        else:
            stale += 1
        if epoch==1 or epoch%5==0 or stale>=budget['patience']:
            print(f'{kind} seed={seed} epoch={epoch} train_loss={history[-1]["train_loss"]:.4f} val_macro_f1={score:.4f} best={best_score:.4f}',flush=True)
        if stale>=budget['patience']:
            break
    suffix = '_smoke' if smoke else ''
    manifest = root / ('data/processed/smoke_manifest.csv' if smoke else 'data/processed/manifest.csv')
    extraction = json.loads((root/f'data/features/extraction{suffix}.json').read_text())
    checkpoint = {'state_dict':state,'model':kind,'seed':seed,'config':config,'budget':budget,
                  'best_epoch':best_epoch,'best_validation_macro_f1':best_score,'best_validation_loss':best_loss,
                  'manifest_sha256':file_hash(manifest),'normalization_sha256':file_hash(root/f'models/normalization{suffix}.npz'),
                  'feature_contract':extraction['contract'],'parameters':sum(p.numel() for p in model.parameters()),
                  'runtime_seconds':time.monotonic()-started,'epochs_run':len(history)}
    destination.parent.mkdir(parents=True,exist_ok=True)
    torch.save(checkpoint,destination.with_suffix('.part'))
    destination.with_suffix('.part').replace(destination)
    pd.DataFrame(history).to_csv(destination.with_suffix('.csv'),index=False)
    return {key:value for key,value in checkpoint.items() if key not in ['state_dict','feature_contract']}


def smoke_test(root: Path):
    started = time.monotonic()
    verify_matched_initialization()
    frame,features,labels = normalized_features(root,True)
    budget = {**BUDGET,'epochs':8,'patience':3}
    results=[]
    for kind in NAMES:
        path=root/f'models/smoke/{kind}_42.pt'
        results.append(fit_one(root,frame,features,labels,kind,42,SEARCH[0],budget,path,True))
        saved=torch.load(path,map_location='cpu',weights_only=True)
        restored=Classifier(kind,SEARCH[0]['dimension'],SEARCH[0]['dropout'])
        restored.load_state_dict(saved['state_dict'])
        val_indices=torch.from_numpy(np.flatnonzero(frame.split.eq('validation').to_numpy()))
        probability,loss=predict(restored,features,labels,val_indices)
        from .evaluate import metrics, paired
        scores=metrics(labels[val_indices].numpy(),probability)
        assert abs(scores['macro_f1']-saved['best_validation_macro_f1'])<1e-12
        assert abs(loss-saved['best_validation_loss'])<1e-6
        null=paired(labels[val_indices].numpy(),probability.argmax(axis=1),probability.argmax(axis=1),100,42)
        assert null['mcnemar_exact_two_sided_p']==1. and all(value==0. for value in null['differences_proposed_minus_baseline'].values())
        results[-1]['reloaded_validation_metrics']=scores
    atomic_json(root/'results/metrics/smoke.json',{'status':'passed','scope':'pipeline verification; validation only; not final results',
                'runtime_seconds':time.monotonic()-started,'checkpoint_reload_and_paired_null_checks':'passed','runs':results})
    log_stage(root,'End-to-end smoke test',f'Command: `python run_experiments.py smoke`. Four frozen-feature heads trained on 300 samples and selected on 50 validation samples. '
              f'No test metrics examined. Matching C/D initialization passed. Actual validation results and runtimes: results/metrics/smoke.json. '
              f'Training runtime {time.monotonic()-started:.1f}s. This verifies execution, not research performance.')


def search(root: Path):
    started=time.monotonic()
    verify_matched_initialization()
    frame,features,labels = normalized_features(root)
    digest = file_hash(root/'data/processed/manifest.csv')
    directory=root/'models/search'
    results=[]
    for i,config in enumerate(SEARCH):
        for kind in ['baseline','proposed']:
            path=directory/f'config{i}_{kind}_42.pt'
            if path.exists():
                saved=torch.load(path,map_location='cpu',weights_only=True)
                assert saved['manifest_sha256']==digest and saved['config']==config and saved['budget']==BUDGET
                assert saved['model']==kind and saved['seed']==42
                assert saved['normalization_sha256']==file_hash(root/'models/normalization.npz')
                result={k:v for k,v in saved.items() if k not in ['state_dict','feature_contract']}
            else:
                result=fit_one(root,frame,features,labels,kind,42,config,BUDGET,path)
            results.append({'config_index':i,**result})
    table=pd.DataFrame([{k:v for k,v in row.items() if k not in ['config','budget']} for row in results])
    table.to_csv(root/'results/metrics/validation_search.csv',index=False)
    scores=table.groupby('config_index').best_validation_macro_f1.mean()
    chosen=int(scores.idxmax())
    lock={'config':SEARCH[chosen],'budget':BUDGET,'seeds':SEEDS,'search_seed':42,'candidates':SEARCH,
          'selected_config_index':chosen,'selection_rule':'highest mean validation Macro-F1 across matched baseline and proposed; earliest configuration breaks ties',
          'selected_validation_mean_macro_f1':float(scores.loc[chosen]),'manifest_sha256':digest,
          'primary_paired_seed':42,'positive_class':{'label':0,'name':'fake'},
          'test_use':'No test predictions or metrics used for selection. All listed models/seeds and paired analyses prespecified.',
          'bootstrap_repetitions':5000,'bootstrap_seed':20260916,
          'normalization':'Per-coordinate z-score for all features fit only on training rows; standard deviation floor 1e-6',
          'runtime_seconds':time.monotonic()-started}
    atomic_json(root/'models/locked_config.json',lock)
    log_stage(root,'Validation-only configuration lock',
              f'Command: `python run_experiments.py search`. Eight fits: four prespecified configurations each for C/D, seed 42, identical budgets. '
              f'Chosen configuration {chosen}: {SEARCH[chosen]}, mean C/D validation Macro-F1 {scores.loc[chosen]:.6f}. '
              f'All final models use this same configuration. Final seeds {SEEDS}; no test metrics inspected. Runtime {time.monotonic()-started:.1f}s. '
              'See models/locked_config.json and results/metrics/validation_search.csv.')


def train(root: Path, kind: str = 'all'):
    lock_path=root/'models/locked_config.json'
    if not lock_path.exists():
        raise RuntimeError('Run validation-only search before final training')
    lock=json.loads(lock_path.read_text())
    assert lock['manifest_sha256']==file_hash(root/'data/processed/manifest.csv')
    frame,features,labels=normalized_features(root,fit=False)
    selected=list(NAMES) if kind=='all' else [kind]
    summaries=[]
    for name in selected:
        for seed in lock['seeds']:
            path=root/f'models/final/{name}_{seed}.pt'
            if path.exists():
                saved=torch.load(path,map_location='cpu',weights_only=True)
                assert saved['manifest_sha256']==lock['manifest_sha256'] and saved['config']==lock['config'] and saved['budget']==lock['budget']
                assert saved['model']==name and saved['seed']==seed
                assert saved['normalization_sha256']==file_hash(root/'models/normalization.npz')
                print(f'Verified completed checkpoint: {path.name}',flush=True)
                result={k:v for k,v in saved.items() if k not in ['state_dict','feature_contract']}
            else:
                result=fit_one(root,frame,features,labels,name,seed,lock['config'],lock['budget'],path)
            summaries.append(result)
    atomic_json(root/f'results/metrics/training_{kind}.json',summaries)
    log_stage(root,'Final head training',f'Command: `python run_experiments.py train --model {kind}`. Models {selected}, seeds {lock["seeds"]}. '
              'Same locked optimizer, architecture, batch size, epoch budget, early stopping and validation criterion. '
              'Best validation checkpoints and actual histories saved under models/final. Test not used for training or checkpoint selection.')
