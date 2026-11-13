"""Fixed exploratory evaluation, paired inference and subgroup counts."""
from __future__ import annotations
import time
import numpy as np
import pandas as pd
import torch
from .common import require,lock_protocol,digest,sha,read_json,write_json,peak_memory,LABEL6,manifest
from .training import VARIANTS,prepare_data,make_head,predict,binding,verify_completed,checkpoint_path
from .statistics import metrics,paired,holm

METRICS=['accuracy','macro_f1','weighted_f1','fake_precision','fake_recall','fake_f1','roc_auc','pr_auc','average_precision']

def group_masks(frame,cosine,quartiles):
    masks={}
    bands=np.searchsorted(quartiles,cosine,side='right')+1
    for q in range(1,5):masks[f'cosine_Q{q}']=bands==q
    for label,name in LABEL6.items():masks[f'label6_{label}_{name}']=frame.label6.to_numpy()==label
    masks['high_similarity_fake']=(frame.label.to_numpy()==0)&(cosine>=quartiles[2])
    masks['low_similarity_real']=(frame.label.to_numpy()==1)&(cosine<=quartiles[0])
    return masks,bands

def evaluate(ctx,variants=None):
    variants=list(VARIANTS) if variants is None else variants
    require(all(x in VARIANTS for x in variants),'Unknown variants')
    require(set(VARIANTS).issubset(variants),'Complete every prespecified variant before test evaluation; no partial best-model selection')
    p=lock_protocol(ctx);torch.set_num_threads(p['budget']['threads'])
    split='validation' if ctx.smoke else 'test'
    frame,features,labels,receipt=prepare_data(ctx)
    checkpoints={}
    # No test predictions until every prescribed checkpoint validates.
    for kind in variants:
        for seed in p['seeds']:
            checkpoints[kind,seed]=verify_completed(ctx,kind,seed,binding(ctx,kind,seed,receipt))
    evidence={'protocol_sha256':digest(p),'features_sha256':receipt['sha256'],
        'checkpoints':{f'{k}_{s}':sha(checkpoint_path(ctx,k,s)) for k,s in checkpoints}}
    completed=ctx.results/'evaluation.json'
    if completed.exists():
        old=read_json(completed);require(old['evidence']==evidence,'Evaluation evidence changed')
        for rel,h in old['outputs'].items():require(sha(ctx.results/rel)==h,f'Changed evaluation output {rel}')
        return {'status':'verified existing evaluation','path':str(completed)}
    start=time.perf_counter();idx=torch.tensor(np.flatnonzero(frame.split.eq(split)),dtype=torch.long)
    test=frame.iloc[idx.numpy()].reset_index(drop=True);y=labels[idx].numpy()
    from .features import load_generated
    generated,_=load_generated(ctx,frame)
    cosine=generated['bras'][:,0];quartiles=np.quantile(cosine[frame.split.eq('train')],[.25,.5,.75])
    masks,bands=group_masks(test,cosine[idx.numpy()],quartiles)
    outdir=ctx.results/'predictions';outdir.mkdir(parents=True,exist_ok=True)
    records=[];groups=[];predictions={};confusions={}
    for kind in variants:
        for seed in p['seeds']:
            saved=checkpoints[kind,seed];model=make_head(kind,p['config'],ctx.project)
            model.load_state_dict(saved['state_dict'])
            # Warm-up is timed separately; inference timings exclude encoder/feature loading.
            predict(model,features,idx[:min(32,len(idx))],kind)
            started=time.perf_counter();prob,_=predict(model,features,idx,kind);seconds=time.perf_counter()-started
            score=metrics(y,prob);confusions[f'{kind}_{seed}']=score.pop('confusion_matrix')
            records.append({'variant':kind,'seed':seed,**score,'best_epoch':saved['best_epoch'],
                'parameters':saved['parameters'],'training_seconds':saved['seconds'],
                'inference_seconds':seconds,'inference_ms_per_pair':1000*seconds/len(y),
                'checkpoint_sha256':evidence['checkpoints'][f'{kind}_{seed}'],'origin':saved['origin']})
            pred=prob.argmax(1);predictions[kind,seed]=pred
            output=pd.DataFrame({'sample_id':test.sample_id,'label':y,'label6':test.label6,
                'prediction':pred,'probability_fake':prob[:,0],'probability_real':prob[:,1],
                'cosine':cosine[idx.numpy()],'quantile':bands,
                'm_text':generated['bras'][idx.numpy(),1],'m_image':generated['bras'][idx.numpy(),2]})
            output.to_csv(outdir/f'{kind}_{seed}.csv',index=False)
            for name,mask in masks.items():
                if not mask.any():
                    groups.append({'variant':kind,'seed':seed,'group':name,'n':0,'small_group':True})
                    continue
                s=metrics(y[mask],prob[mask]);s.pop('confusion_matrix')
                if s['single_class']:s['macro_f1']=None
                groups.append({'variant':kind,'seed':seed,'group':name,**s,'correct':int((pred[mask]==y[mask]).sum())})
            print(f'{kind} seed={seed} {split} Macro-F1={score["macro_f1"]:.6f}',flush=True)
    table=pd.DataFrame(records);table.to_csv(ctx.results/'seed_metrics.csv',index=False)
    summary=[]
    for kind in variants:
        sub=table[table.variant.eq(kind)]
        summary.append({'variant':kind,'n_seeds':len(sub),'parameters':int(sub.parameters.iloc[0]),
            **{m:float(sub[m].mean()) for m in METRICS},
            **{m+'_sd':float(sub[m].std(ddof=1)) if len(sub)>1 else None for m in METRICS}})
    pd.DataFrame(summary).to_csv(ctx.results/'model_comparison.csv',index=False)
    pd.DataFrame(groups).to_csv(ctx.results/'subgroup_metrics.csv',index=False)
    write_json(ctx.results/'confusion_matrices.json',{'order':['Fake (0)','Real (1)'],'matrices':confusions})
    comparisons=[p['primary_comparison']]+p['secondary_comparisons'];stats=[]
    for a,b in comparisons:
        if a not in variants or b not in variants:continue
        for seed in p['seeds']:
            st=paired(y,predictions[a,seed],predictions[b,seed],p['bootstrap_repetitions'],p['bootstrap_seed'])
            st.pop('seed',None)
            stats.append({'comparison':f'{b}_minus_{a}','seed':seed,'role':'primary' if [a,b]==p['primary_comparison'] and seed==42 else 'secondary/exploratory',**st})
            stats[-1]['bootstrap_seed']=p['bootstrap_seed']
            base=predictions[a,seed]==y;new=predictions[b,seed]==y
            error=test[['sample_id','label','label6','text','image_path']].copy()
            error['baseline_prediction']=predictions[a,seed];error['comparison_prediction']=predictions[b,seed]
            error['error_type']=np.select([~base&new,base&~new,base&new],['corrected','new_error','both_correct'],default='both_wrong')
            error.to_csv(outdir/f'errors_{b}_minus_{a}_{seed}.csv',index=False)
    secondary=[s for s in stats if s['seed']==42 and s['role']!='primary']
    for s,adjusted in zip(secondary,holm([s['mcnemar_exact_two_sided_p'] for s in secondary])):s['mcnemar_holm_secondary_p']=adjusted
    write_json(ctx.results/'paired_comparisons.json',{'primary':'D minus B, seed 42, Macro-F1','statistics':stats,
        'multiplicity':'Primary prespecified; secondary seed-42 McNemar Holm family; all bootstrap intervals marginal, exploratory',
        'seed_variation':'Three fitted heads share one evaluation set. No pooling of repeated rows.'})
    distributions=[]
    for split_name in frame.split.unique():
        for column in ['label','label6']:
            for value in sorted(frame[column].unique()):
                use=frame.split.eq(split_name)&frame[column].eq(value)
                for j,feature in enumerate(['cosine','m_text','m_image']):
                    x=generated['bras'][use,j]
                    if not len(x):continue
                    distributions.append({'split':split_name,'label_column':column,'label':int(value),'feature':feature,
                        'n':len(x),'mean':float(x.mean()),'sd':float(x.std(ddof=1)) if len(x)>1 else None,
                        **{f'q{q}':float(np.quantile(x,q/100)) for q in [5,25,50,75,95]}})
    pd.DataFrame(distributions).to_csv(ctx.results/'similarity_distributions.csv',index=False)
    write_json(ctx.results/'quantile_thresholds.json',{'fit_split':'train','quartiles':quartiles.tolist(),'rule':'Q bins use searchsorted side=right; extreme slices use inclusive q25/q75'})
    output_files=[p for p in ctx.results.rglob('*') if p.is_file() and p.name not in ['evaluation.json','training.json','smoke.json'] and 'plots' not in p.parts]
    result={'status':'completed','scope':'validation smoke only' if ctx.smoke else 'exploratory previously inspected Fakeddit test',
        'evidence':evidence,'outputs':{q.relative_to(ctx.results).as_posix():sha(q) for q in output_files},
        'seconds':time.perf_counter()-start,'test_samples':len(y),**peak_memory()}
    write_json(completed,result,immutable=True)
    return {'status':'completed','path':str(completed),'samples':len(y)}


def recompute_paired(ctx, fresh_holdout=False):
    """Rebuild labeled paired statistics from saved predictions without refitting."""
    from .statistics import paired,holm
    from .common import read_json,write_json,sha,digest
    import numpy as np
    import pandas as pd
    if fresh_holdout:
        from .fresh_holdout import _load_fresh,root as fresh_root
        frame=_load_fresh(ctx); outdir=fresh_root(ctx); pred_dir=outdir/'predictions'
        source_eval=outdir/'evaluation.json'; source_pairs=outdir/'paired_comparisons.json'
    else:
        frame=manifest(ctx)
        frame=frame[frame.split.eq('test')].reset_index(drop=True)
        outdir=ctx.results;pred_dir=outdir/'predictions'
        source_eval=outdir/'evaluation.json';source_pairs=outdir/'paired_comparisons.json'
    require(source_eval.is_file() and source_pairs.is_file(),'Completed evaluation artifacts are required')
    protocol=read_json(ctx.models/'protocol.json')
    require(len(frame)>0,'No evaluation rows found')
    y=frame.label.to_numpy(dtype=int);predictions={};hashes={}
    for kind in VARIANTS:
        for seed in protocol['seeds']:
            path=pred_dir/f'{kind}_{seed}.csv'
            require(path.is_file(),f'Missing saved prediction file: {path}')
            table=pd.read_csv(path,keep_default_na=False,dtype={'sample_id':str})
            require(table.sample_id.tolist()==frame.sample_id.tolist(),f'Prediction IDs/order differ: {path.name}')
            require(np.array_equal(table.label.to_numpy(dtype=int),y),f'Prediction labels differ: {path.name}')
            predictions[kind,seed]=table.prediction.to_numpy(dtype=int)
            require(np.isin(predictions[kind,seed],[0,1]).all(),f'Invalid predictions: {path.name}')
            hashes[path.name]=sha(path)
    pairs=[protocol['primary_comparison']]+protocol['secondary_comparisons']
    stats=[]
    for a,b in pairs:
        for seed in protocol['seeds']:
            values=paired(y,predictions[a,seed],predictions[b,seed],
                          protocol['bootstrap_repetitions'],protocol['bootstrap_seed'])
            values.pop('seed',None)
            role='primary' if [a,b]==protocol['primary_comparison'] and seed==42 else 'secondary/exploratory'
            stats.append({'comparison':f'{b}_minus_{a}','seed':seed,'role':role,**values,
                'bootstrap_seed':protocol['bootstrap_seed']})
    secondary=[s for s in stats if s['seed']==42 and s['role']!='primary']
    adjusted=holm([s['mcnemar_exact_two_sided_p'] for s in secondary])
    for item,pv in zip(secondary,adjusted):item['mcnemar_holm_secondary_p']=pv
    output=outdir/'paired_comparisons_corrected.json'
    write_json(output,{'primary':'D minus B, classifier seed 42, Macro-F1',
        'scope':'fresh official-test subset' if fresh_holdout else 'previously inspected test subset',
        'statistics':stats,'seed_field':'classifier/checkpoint seed',
        'bootstrap_seed_field':'bootstrap resampling seed',
        'multiplicity':'Primary prespecified; secondary seed-42 McNemar Holm family; bootstrap intervals marginal',
        'source_paired_sha256':sha(source_pairs),'prediction_hashes':hashes,
        'protocol_sha256':digest(protocol)})
    receipt={'status':'completed','output':'paired_comparisons_corrected.json',
        'output_sha256':sha(output),'source_evaluation_sha256':sha(source_eval),
        'source_paired_sha256':sha(source_pairs),'prediction_hashes':hashes,
        'protocol_sha256':digest(protocol),'rows':len(frame),'classifier_seeds':protocol['seeds']}
    write_json(outdir/'paired_comparisons_corrected.receipt.json',receipt)
    return receipt
