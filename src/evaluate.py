"""Locked test evaluation, paired inference and reproducible error tables."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import binomtest
from sklearn.metrics import (auc, average_precision_score, confusion_matrix, f1_score,
                             precision_recall_curve, roc_auc_score)

from .models import Classifier, NAMES
from .train import classification_scores, normalized_features, predict
from .utils import atomic_json, file_hash, log_stage


def metrics(y,probability):
    pred=probability.argmax(axis=1)
    result=classification_scores(y,pred)
    precision,recall,_=precision_recall_curve(1-y,probability[:,0])
    result.update(weighted_f1=float(f1_score(y,pred,average='weighted',zero_division=0)),
                  roc_auc=float(roc_auc_score(1-y,probability[:,0])),
                  pr_auc=float(auc(recall,precision)),
                  average_precision=float(average_precision_score(1-y,probability[:,0])),
                  false_positives=int(((y==1)&(pred==0)).sum()),
                  false_negatives=int(((y==0)&(pred==1)).sum()))
    return result


def fast_f1(y,pred):
    tp=((y==0)&(pred==0)).sum(axis=-1)
    tn=((y==1)&(pred==1)).sum(axis=-1)
    fp=((y==1)&(pred==0)).sum(axis=-1)
    fn=((y==0)&(pred==1)).sum(axis=-1)
    fake=2*tp/np.maximum(2*tp+fp+fn,1)
    real=2*tn/np.maximum(2*tn+fp+fn,1)
    return fake,(fake+real)/2


def paired(y,baseline,proposed,repetitions,seed):
    b=(baseline==y)
    p=(proposed==y)
    gained=int((~b&p).sum())
    lost=int((b&~p).sum())
    pvalue=float(binomtest(gained,gained+lost,.5,alternative='two-sided').pvalue) if gained+lost else 1.
    bf,bm=fast_f1(y,baseline)
    pf,pm=fast_f1(y,proposed)
    point={'accuracy':float(p.mean()-b.mean()),'f1':float(pf-bf),'macro_f1':float(pm-bm)}
    bootstrap={key:[] for key in point}
    rng=np.random.default_rng(seed)
    for start in range(0,repetitions,100):
        indices=rng.integers(0,len(y),size=(min(100,repetitions-start),len(y)))
        yy,bb,pp=y[indices],baseline[indices],proposed[indices]
        bf1,bmacro=fast_f1(yy,bb)
        pf1,pmacro=fast_f1(yy,pp)
        bootstrap['accuracy'].extend(((pp==yy).mean(axis=1)-(bb==yy).mean(axis=1)).tolist())
        bootstrap['f1'].extend((pf1-bf1).tolist())
        bootstrap['macro_f1'].extend((pmacro-bmacro).tolist())
    report={'test_samples':len(y),'baseline_wrong_proposed_correct':gained,
            'baseline_correct_proposed_wrong':lost,'both_correct':int((b&p).sum()),'both_wrong':int((~b&~p).sum()),
            'mcnemar_exact_two_sided_p':pvalue,'differences_proposed_minus_baseline':point,
            'relative_f1_improvement_percent':float(100*(pf-bf)/bf) if bf else None,
            'relative_macro_f1_improvement_percent':float(100*(pm-bm)/bm) if bm else None,
            'bootstrap_repetitions':repetitions,'bootstrap_seed':seed,
            'paired_bootstrap_95_percentile_intervals':{key:np.quantile(value,[.025,.975]).tolist() for key,value in bootstrap.items()},
            'scope':'Conditional on this test subset and trained checkpoints. Resample the same test indices for both models. Training randomness summarized separately; repeated seeds are not independent test samples.'}
    return report


def evaluate(root: Path):
    started=time.monotonic()
    lock=json.loads((root/'models/locked_config.json').read_text())
    assert lock['manifest_sha256']==file_hash(root/'data/processed/manifest.csv')
    for kind in NAMES:
        for seed in lock['seeds']:
            assert (root/f'models/final/{kind}_{seed}.pt').is_file(), 'Complete all prespecified training runs before test evaluation'
    frame,features,labels=normalized_features(root,fit=False)
    torch.set_num_threads(lock['budget']['threads'])
    indices=torch.from_numpy(np.flatnonzero(frame.split.eq('test').to_numpy()))
    test=frame.iloc[indices.numpy()].copy().reset_index(drop=True)
    y=labels[indices].numpy()
    predictions=root/'results/predictions'
    predictions.mkdir(parents=True,exist_ok=True)
    records,outputs=[],{}
    for kind in NAMES:
        for seed in lock['seeds']:
            path=root/f'models/final/{kind}_{seed}.pt'
            saved=torch.load(path,map_location='cpu',weights_only=True)
            assert saved['manifest_sha256']==lock['manifest_sha256'] and saved['config']==lock['config']
            assert saved['model']==kind and saved['seed']==seed and saved['budget']==lock['budget']
            assert saved['normalization_sha256']==file_hash(root/'models/normalization.npz')
            extraction=json.loads((root/'data/features/extraction.json').read_text())
            assert saved['feature_contract']==extraction['contract']
            model=Classifier(kind,lock['config']['dimension'],lock['config']['dropout'])
            model.load_state_dict(saved['state_dict'],strict=True)
            probability,loss=predict(model,features,labels,indices)
            scored=metrics(y,probability)
            records.append({'model':NAMES[kind],'kind':kind,'seed':seed,'test_samples':len(test),
                            **scored,'test_loss':loss,'best_epoch':saved['best_epoch'],
                            'best_validation_macro_f1':saved['best_validation_macro_f1'],
                            'parameters':saved['parameters'],'training_seconds':saved['runtime_seconds']})
            output=pd.DataFrame({'sample_id':test.sample_id,'true_label':y,'predicted_label':probability.argmax(axis=1),
                                 'probability_fake':probability[:,0],'probability_real':probability[:,1],
                                 'confidence':probability.max(axis=1)})
            output.to_csv(predictions/f'{kind}_{seed}.csv',index=False)
            outputs[kind,seed]=output
            print(f'{kind} seed={seed}: {json.dumps(scored)}',flush=True)
    table=pd.DataFrame(records)
    table.to_csv(root/'results/metrics/seed_metrics.csv',index=False)
    names=['accuracy','precision','recall','f1','macro_f1','weighted_f1','roc_auc','pr_auc','average_precision','false_positives','false_negatives']
    summary=[]
    for kind,name in NAMES.items():
        group=table[table.kind.eq(kind)]
        summary.append({'model':name,'kind':kind,'seeds':len(group),'aggregation':'mean across prespecified seeds; std is sample standard deviation',
                        **{key:float(group[key].mean()) for key in names},
                        **{key+'_std':float(group[key].std(ddof=1)) for key in names}})
    pd.DataFrame(summary).to_csv(root/'results/metrics/model_comparison.csv',index=False)
    statistics={}
    for seed in lock['seeds']:
        statistics[str(seed)]=paired(y,outputs['baseline',seed].predicted_label.to_numpy(),
                                      outputs['proposed',seed].predicted_label.to_numpy(),
                                      lock['bootstrap_repetitions'],lock['bootstrap_seed'])
    atomic_json(root/'results/metrics/paired_comparison.json',{'primary_seed':lock['primary_paired_seed'],
                'alpha':.05,'other_seeds':'descriptive sensitivity checks; no cherry-picking or pooling repeated test samples',
                'results':statistics})
    seed=lock['primary_paired_seed']
    base,prop=outputs['baseline',seed],outputs['proposed',seed]
    errors=test[['sample_id','text','image_path','label','label6','subreddit']].rename(columns={'label':'true_label'})
    for prefix,output in [('baseline',base),('proposed',prop),
                          ('text',outputs['text',seed]),('image',outputs['image',seed])]:
        errors[prefix+'_prediction']=output.predicted_label
        errors[prefix+'_confidence']=output.confidence
        errors[prefix+'_probability_fake']=output.probability_fake
    from .feature_extractors import load_features
    _,raw=load_features(root)
    errors['similarity']=raw['similarity'][indices.numpy(),0]
    correct_b=base.predicted_label.eq(y)
    correct_p=prop.predicted_label.eq(y)
    errors['category']=np.select([~correct_b&correct_p,correct_b&~correct_p,correct_b&correct_p],
                                 ['Baseline wrong / Proposed correct','Baseline correct / Proposed wrong','Both correct'],default='Both wrong')
    errors['probability_fake_change']=errors.proposed_probability_fake-errors.baseline_probability_fake
    train_sim=raw['similarity'][frame.split.eq('train').to_numpy(),0]
    q25,q75=np.quantile(train_sim,[.25,.75])
    errors['similarity_band']=np.select([errors.similarity.lt(q25),errors.similarity.gt(q75)],['Low (below train Q1)','High (above train Q3)'],default='Middle')
    errors.to_csv(predictions/'error_analysis.csv',index=False)
    for column,filename in [('similarity_band','error_band_metrics'),('label6','error_subtype_metrics')]:
        rows=[]
        for value,group in errors.groupby(column):
            row={column:value,'samples':len(group),'fake_samples':int(group.true_label.eq(0).sum()),
                 'mean_similarity':float(group.similarity.mean())}
            for kind in NAMES:
                row[kind+'_accuracy']=float(group[kind+'_prediction'].eq(group.true_label).mean())
            row['proposed_minus_baseline_accuracy']=row['proposed_accuracy']-row['baseline_accuracy']
            rows.append(row)
        pd.DataFrame(rows).to_csv(root/f'results/metrics/{filename}.csv',index=False)
    examples=[]
    for category,group in errors.groupby('category'):
        ordered=group.sort_values(['similarity','sample_id'])
        positions=np.unique(np.linspace(0,len(ordered)-1,min(8,len(ordered)),dtype=int))
        examples.append(ordered.iloc[positions])
    pd.concat(examples).to_csv(predictions/'error_examples.csv',index=False)
    errors.groupby(['category','similarity_band']).agg(count=('sample_id','size'),mean_similarity=('similarity','mean'),
                mean_fake_probability_change=('probability_fake_change','mean')).to_csv(root/'results/metrics/error_summary.csv')
    pd.crosstab(errors.category,errors.subreddit).to_csv(root/'results/metrics/error_sources.csv')
    report={'status':'complete','manifest_sha256':lock['manifest_sha256'],'config_sha256':file_hash(root/'models/locked_config.json'),
            'primary_seed':seed,'seeds':lock['seeds'],'test_samples':len(test),'positive_class':'fake (0)',
            'runtime_seconds':time.monotonic()-started,'bootstrap_unit':'one retained source/content-disjoint sample'}
    atomic_json(root/'results/metrics/evaluation.json',report)
    primary=statistics[str(seed)]
    log_stage(root,'Locked test evaluation and paired comparison',
              f'Command: `python run_experiments.py evaluate`. {len(test)} test pairs, four models × {len(lock["seeds"])} seeds. '
              f'Actual metrics: results/metrics/seed_metrics.csv and model_comparison.csv (mean and sample SD). '
              f'Primary seed {seed}: proposed-minus-baseline differences {primary["differences_proposed_minus_baseline"]}; '
              f'exact McNemar p={primary["mcnemar_exact_two_sided_p"]:.6g}. Paired bootstrap uses {lock["bootstrap_repetitions"]} common-index resamples. '
              f'Error analysis includes all four correctness categories, unimodal predictions, train-defined similarity bands and descriptive six-way subtype slices. Evaluation runtime {time.monotonic()-started:.1f}s. '
              'Confidence scores are uncalibrated softmax probabilities, not factual-truth guarantees.')
