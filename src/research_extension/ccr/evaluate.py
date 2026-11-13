"""Frozen-policy prediction and statistics calculated only from saved predictions."""
from __future__ import annotations
import joblib
import numpy as np
import pandas as pd
import torch
from .core import *
from .fit import historical_predictions,predict_policy,verify_frozen

def dataset(s,name):
    if name=='original_test':
        frame=split_frame(s,'test');raw=load_historical(Context(s.project,s.project),frame)
        return frame,raw,{'manifest':s.protocol['manifest_sha256'],'historical_inventory':s.protocol['historical_inventory_sha256']}
    require(name=='previous_fresh_test','Unsupported evaluation set')
    root=s.project/'data/features/research_extension/fresh_holdout';mp=root/'manifest.csv';fp=root/'features/vectors.npz'
    require(sha(mp)==read_json(root/'manifest.json')['sha256'],'Fresh manifest changed')
    require(sha(fp)==read_json(root/'features/vectors.json')['sha256'],'Fresh feature hash changed')
    f=pd.read_csv(mp,keep_default_na=False,dtype={'sample_id':str,'group_id':str});a=load_npz(fp)
    require(f.sample_id.tolist()==a['ids'].tolist() and f.input_hash.tolist()==a['input_hash'].tolist(),'Fresh feature row mismatch')
    raw={'text':a['text'],'image':a['image'],'similarity':np.einsum('ij,ij->i',a['clip_text'],a['clip_image'])[:,None]}
    return f,raw,{'manifest':sha(mp),'vectors':sha(fp)}

def save_predictions(s,name,frame,raw,evidence):
    verify_frozen(s);torch.set_num_threads(s.protocol['cpu_threads']);paths=[]
    for seed in s.seeds:
        path=s.path(f'predictions/{name}_{seed}.csv');paths.append(path)
        if path.exists():verify(path);continue
        bundle=joblib.load(s.models/f'gate_{seed}.joblib');policy=read_json(s.models/f'policy_{seed}.json')
        base=historical_predictions(s,frame,raw,seed)
        if name in ['original_test','previous_fresh_test']:
            oldroot=s.project/('results/research_extension/bras_b4096_k16_s20260916/predictions' if name=='original_test' else 'data/features/research_extension/fresh_holdout/predictions')
            for k in ['A','B']:
                old=pd.read_csv(oldroot/f'{k}_{seed}.csv',dtype={'sample_id':str}).set_index('sample_id').loc[frame.sample_id]
                require(np.array_equal(old.label.to_numpy(),frame.label.to_numpy()),'Historical test labels differ')
                require(np.allclose(old.probability_real,base[k],rtol=0,atol=2e-6),'Historical prediction replication failed')
        out,routes,gs=predict_policy(frame,base['A'],base['B'],raw['similarity'][:,0],bundle,policy)
        keep=[k for k in ['sample_id','label','group_id','subreddit','label6','cluster_id','official_label'] if k in frame]
        table=frame[keep].copy();table['cosine']=raw['similarity'][:,0]
        for k in KINDS:table['p_'+k]=out[k];table['use_B_'+k]=routes[k]
        for k in gs:table['utility_'+k]=gs[k]
        save_csv(table,path,{'dataset':evidence,'policy':sha(s.models/f'policy_{seed}.json'),'frozen_policy':sha(s.path('frozen_policy.json'))})
    return paths

def read_prediction(path):
    verify(path);f=pd.read_csv(path,dtype={'sample_id':str,'group_id':str,'cluster_id':str},keep_default_na=False)
    require(f.sample_id.is_unique and f.label.isin([0,1]).all(),'Prediction identity/label invalid')
    for k in KINDS:require(np.isfinite(f['p_'+k]).all(),'Incomplete predictions')
    return f

def analyze(s,name):
    records=[];paired=[];reliabilities=[];risk=[];subgroups=[];quartiles=[];utility=[]
    evidence={}
    for seed in s.seeds:
        path=s.path(f'predictions/{name}_{seed}.csv');f=read_prediction(path);evidence[path.name]=sha(path)
        policy=read_json(s.models/f'policy_{seed}.json');y=f.label.to_numpy();base=f.p_B.to_numpy()
        cls=f.cluster_id.to_numpy() if 'cluster_id' in f else f.group_id.to_numpy()
        same_clusters=len(np.unique(cls))==len(f)
        q=np.searchsorted(policy['cosine_quartiles'],f.cosine.to_numpy(),side='right')+1
        for band in range(1,5):
            mask=q==band;n=int(mask.sum());real=int(f.loc[mask,'label'].sum())
            quartiles.append({'dataset':name,'classifier_seed':seed,'quartile':band,'n':n,'real':real,'real_fraction':real/n if n else None,
                'unique_families':int(f.loc[mask,'group_id'].nunique()) if 'group_id' in f else None,'prevalence_ci95_low':wilson(real,n)[0],'prevalence_ci95_high':wilson(real,n)[1]})
        masks={f'cosine_Q{k}':q==k for k in range(1,5)}
        if 'label6' in f:
            for k in range(6):masks[f'label6_{k}']=f.label6.eq(k).to_numpy()
        if 'subreddit' in f:
            for k in sorted(f.subreddit.unique()):masks['subreddit_'+str(k)]=f.subreddit.eq(k).to_numpy()
        if 'official_label' in f:
            for k in sorted(f.official_label.unique()):masks['official_'+k]=f.official_label.eq(k).to_numpy()
        for kind in KINDS:
            p=f['p_'+kind].to_numpy();pred=(p>.5).astype(int)
            records.append({'dataset':name,'variant':kind,'classifier_seed':seed,'b_fraction':float(f['use_B_'+kind].mean()),**scores(y,p)})
            for bins in [5,10,15,20]:
                reliabilities.extend({'dataset':name,'variant':kind,'classifier_seed':seed,'bins':bins,**r} for r in reliability(y,p,bins))
            for target,cut in policy['coverage_cutoffs'][kind].items():
                mask=np.maximum(p,1-p)>=cut;n=int(mask.sum());wrong=int((pred[mask]!=y[mask]).sum())
                risk.append({'dataset':name,'variant':kind,'classifier_seed':seed,'validation_target_coverage':float(target),'frozen_cutoff':cut,
                  'retained_n':n,'achieved_coverage':float(mask.mean()),'risk':wrong/n if n else None,
                  'risk_ci95_low':wilson(wrong,n)[0],'risk_ci95_high':wilson(wrong,n)[1]})
            if kind!='B':
                st=paired_stats(y,base,p,s.reps,s.protocol['bootstrap']['seed'])
                paired.append({'dataset':name,'classifier_seed':seed,'base':'B','candidate':kind,**st})
                if not same_clusters:
                    paired.append({'dataset':name,'classifier_seed':seed,'base':'B','candidate':kind,**paired_stats(y,base,p,s.reps,s.protocol['bootstrap']['seed'],cls)})
            if kind in ['A','B','CCR','ridge_candidate','tree','B_temperature']:
                for group,mask in masks.items():
                    n=int(mask.sum())
                    if not n:continue
                    correct=int((pred[mask]==y[mask]).sum());bc=(base[mask]>.5).astype(int)==y[mask];pc=pred[mask]==y[mask]
                    interval=wilson(correct,n);difference=pc.astype(float)-bc.astype(float)
                    se=float(difference.std(ddof=1)/np.sqrt(n)) if n>1 else None
                    subgroups.append({'dataset':name,'classifier_seed':seed,'variant':kind,'group':group,'n':n,'real_n':int(y[mask].sum()),
                      'unique_families':int(len(np.unique(cls[mask]))),'accuracy':correct/n,'accuracy_ci95_low':interval[0],'accuracy_ci95_high':interval[1],
                      'log_loss':float(ce(y[mask],p[mask]).mean()),'brier':float(((p[mask]-y[mask])**2).mean()),
                      'corrected_vs_B':int((~bc&pc).sum()),'new_errors_vs_B':int((bc&~pc).sum()),'accuracy_delta_vs_B':float(difference.mean()),
                      'paired_delta_normal_ci95_low':max(-1.,float(difference.mean())-1.96*se) if se is not None else None,
                      'paired_delta_normal_ci95_high':min(1.,float(difference.mean())+1.96*se) if se is not None else None,
                      'uncertainty_note':'Wilson row accuracy; approximate paired-normal delta; small/correlated slices descriptive only'})
        observed=ce(y,f.p_A.to_numpy())-ce(y,f.p_B.to_numpy())
        for kind in ['intercept','ridge_candidate','tree']:
            oof=pd.read_csv(s.path(f'oof/{seed}.csv'));bundle=joblib.load(s.models/f'gate_{seed}.joblib')
            from .fit import model_scores
            train_score=model_scores(bundle,gate_features(oof.p_A,oof.p_B,oof.cosine))[kind]
            edges=np.unique(np.quantile(train_score,np.linspace(0,1,11)));bucket=np.searchsorted(edges[1:-1],f['utility_'+kind],side='right')
            for band in np.unique(bucket):
                mask=bucket==band;n=int(mask.sum());obs=observed[mask];se=float(obs.std(ddof=1)/np.sqrt(n)) if n>1 else None
                utility.append({'dataset':name,'classifier_seed':seed,'utility_model':kind,'bin':int(band),'n':n,
                     'predicted_mean':float(f.loc[mask,'utility_'+kind].mean()),'observed_mean':float(obs.mean()),
                     'observed_ci95_low':float(obs.mean())-1.96*se if se is not None else None,'observed_ci95_high':float(obs.mean())+1.96*se if se is not None else None,
                     'note':'Train-OOF score bins; pointwise mean normal CI; descriptive, not calibrated probabilities'})
    dest=s.path(f'analysis/{name}');dest.mkdir(parents=True,exist_ok=True)
    outputs={'seed_metrics':records,'reliability':reliabilities,'risk_coverage':risk,'subgroups':subgroups,'quartile_prevalence':quartiles,'conditional_utility':utility}
    for key,rows in outputs.items():
        path=dest/f'{key}.csv'
        if not path.exists():save_csv(pd.DataFrame(rows),path,{'saved_predictions':evidence})
        else:verify(path)
    pairs=dest/'paired_statistics.json'
    data={'comparison_direction':'candidate minus B; lower loss/Brier better, higher Macro-F1/accuracy better','classifier_seed_distinct_from_bootstrap_seed':True,
          'known_group_count_equals_rows':same_clusters,'scope':'Exploratory previously viewed Fakeddit' if name!='verite' else 'Preregistered cross-task/domain transfer; cluster bootstrap primary',
          'prediction_hashes':evidence,'comparisons':paired}
    write_json(pairs,data,immutable=True)
    if not pairs.with_suffix('.json.receipt.json').exists():seal(pairs,{'saved_predictions':evidence})
    summary=pd.DataFrame(records).groupby('variant',sort=False).agg({k:['mean','std'] for k in ['accuracy','macro_f1','log_loss','brier','ece_10','b_fraction']})
    summary.columns=['_'.join(x) for x in summary.columns];summary=summary.reset_index()
    if not (dest/'model_comparison.csv').exists():save_csv(summary,dest/'model_comparison.csv',{'seed_metrics':sha(dest/'seed_metrics.csv')})
    write_json(dest/'complete.json',{'created_utc':now(),'prediction_hashes':evidence,'rows':len(f),'seeds':s.seeds,
               'bootstrap_repetitions':s.reps,'source_sha256':sha(s.project/'src/research_extension/ccr/evaluate.py')})
    print(f'Analyzed {name}: {len(f)} saved rows/seed, {len(paired)} paired comparisons',flush=True)
    return {'dataset':name,'rows_per_seed':len(f),'paired_comparisons':len(paired)}

def evaluate(s):
    require(not s.smoke,'Scientific test evaluation disabled in smoke mode');verify_frozen(s);out=[]
    for name in ['original_test','previous_fresh_test']:
        f,raw,evidence=dataset(s,name);save_predictions(s,name,f,raw,evidence);del raw
        out.append(analyze(s,name))
    return out

def repeated_historical(ctx,frame,kinds):
    """The historical loader requires unique IDs; restore intentional donor reuse."""
    unique=frame.drop_duplicates('sample_id').reset_index(drop=True)
    values=load_historical(ctx,unique,kinds)
    order=pd.Index(unique.sample_id).get_indexer(frame.sample_id)
    require((order>=0).all(),'Donor feature alignment failed')
    return {k:v[order] for k,v in values.items()}

def stress(s):
    require(not s.smoke,'No test swaps in smoke');verify_frozen(s);torch.set_num_threads(s.protocol['cpu_threads'])
    old=s.project/'results/research_extension/bras_b4096_k16_s20260916/stress';full=manifest(Context(s.project,s.project)).set_index('sample_id');records=[]
    for condition in ['random_nonmatching','hard_nonmatching']:
        pair_path=old/f'{condition}_pairs.csv';pairs=pd.read_csv(pair_path,dtype={'sample_id':str,'donor_id':str})
        query=full.loc[pairs.sample_id].reset_index();donor=full.loc[pairs.donor_id].reset_index()
        require(query.split.eq('test').all() and donor.split.eq('train').all(),'Swap query/donor split violation')
        require(not query.group_id.eq(donor.group_id).any(),'Swap self/family donor')
        raw=load_historical(Context(s.project,s.project),query);donor_image=repeated_historical(Context(s.project,s.project),donor,('image',))['image']
        swapped={'text':raw['text'],'image':donor_image,'similarity':pairs.swapped_cosine.to_numpy(dtype=np.float32)[:,None]}
        for seed in s.seeds:
            policy=read_json(s.models/f'policy_{seed}.json');bundle=joblib.load(s.models/f'gate_{seed}.joblib')
            original=historical_predictions(s,query,raw,seed);changed=historical_predictions(s,query,swapped,seed)
            po,ro,go=predict_policy(query,original['A'],original['B'],raw['similarity'][:,0],bundle,policy)
            pc,rc,gc=predict_policy(query,changed['A'],changed['B'],swapped['similarity'][:,0],bundle,policy)
            details=pairs[['sample_id','donor_id','original_cosine','swapped_cosine']].copy()
            # Deliberately no labels in synthetic rows or loss/accuracy metrics.
            for k in ['A','B','CCR','ridge_candidate']:
                details['original_p_'+k]=po[k];details['swapped_p_'+k]=pc[k]
                details['original_use_B_'+k]=ro[k];details['swapped_use_B_'+k]=rc[k]
                records.append({'condition':condition,'classifier_seed':seed,'variant':k,'n':len(query),
                  'prediction_flip_fraction':float(((pc[k]>.5)!=(po[k]>.5)).mean()),'mean_p_real_shift':float((pc[k]-po[k]).mean()),
                  'routing_flip_fraction':float((rc[k]!=ro[k]).mean()),'mean_ridge_utility_shift':float((gc['ridge_candidate']-go['ridge_candidate']).mean()),
                  'mean_cosine_shift':float((swapped['similarity']-raw['similarity']).mean()),'scope':'label-free synthetic pair sensitivity; not factual accuracy'})
            details['original_ridge_utility']=go['ridge_candidate'];details['swapped_ridge_utility']=gc['ridge_candidate']
            path=s.path(f'stress/{condition}_{seed}.csv')
            if not path.exists():save_csv(details,path,{'donor_selection_from_BRAS':sha(pair_path),'frozen_policy':sha(s.path('frozen_policy.json'))})
    path=s.path('stress/summary.csv')
    if not path.exists():save_csv(pd.DataFrame(records),path)
    return {'conditions':2,'queries_each':len(query),'seeds':s.seeds}
