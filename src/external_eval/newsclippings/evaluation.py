"""Separate official-score and pinned-real-image evaluation; no fitting APIs."""
from __future__ import annotations
import gc,time,platform,os
from pathlib import Path
import numpy as np
import pandas as pd
from .io_utils import *
from .data import *
from .acquire import enriched

def source_contract(run):
    names=['io_utils.py','data.py','acquire.py','statistics.py','evaluation.py']
    return {f'src/external_eval/newsclippings/{name}':sha(run.root/f'src/external_eval/newsclippings/{name}') for name in names}
def freeze_analysis(run):
    path=run.results/'analysis_code_freeze.json';obj={'protocol_sha256':sha(run.results/'protocol.json'),'sources':source_contract(run)}
    if path.exists() and read(path)!=obj:
        amendment=read(run.results/'amendments/transfer_report/amendment.json')
        require(amendment['after_sources']==obj['sources'] and amendment['before_freeze_sha256']==sha(path),'Unrecorded post-score analysis change')
        path=run.results/'analysis_code_freeze_transfer_fix.json'
    write(path,obj);return obj
def load_features(run,mode):
    p=run.data/f'{mode}_features.npz';receipt=verify(p);bound=receipt['binding']
    require(bound['protocol_sha256']==run.binding()['protocol_sha256'],'Feature protocol mismatch')
    if mode=='official':
        require(bound.get('provenance')=='official_precomputed_CLIP_ViT-B32_not_project_pinned_encoder' and bound['selection_sha256']==sha(run.results/'selection_manifest.csv'),'Wrong official feature provenance')
    else:
        require(bound.get('feature_provenance')=='pinned_local_real_images' and bound['manifest_sha256']==sha(run.results/'local_manifest.csv') and bound['encoder_contract_sha256']==sha(run.root/'data/features/extraction.json'),'Pinned features require original full-image encoder provenance')
    with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def alignment(run,mode):
    run.frozen();freeze_analysis(run)
    f=enriched(run) if mode=='official' else pd.read_csv(run.results/'local_manifest.csv',keep_default_na=False,dtype={'caption_id':str,'image_id':str})
    z=load_features(run,mode);ids=z['ids'].tolist();f=f.set_index('row_id').loc[ids].reset_index()
    for name in ['clip_text','clip_image']:
        require(z[name].shape==(len(f),512) and np.isfinite(z[name]).all() and np.allclose(np.linalg.norm(z[name],axis=1),1,atol=2e-5),'Invalid normalized CLIP cache')
    f['cosine']=np.einsum('ij,ij->i',z['clip_text'],z['clip_image'],dtype=np.float32)
    if mode=='pinned':
        receipt=read(run.results/'local_input_receipt.json');require_images(f,receipt['authorized_root'])
        from ...research_extension.reference import margins
        p=run.root/'data/features/research_extension/bras_b4096_k16_s20260916/bank.npz';require(sha(p)==read(p.with_suffix('.json'))['sha256'],'BRAS bank changed')
        with np.load(p,allow_pickle=False) as a:bank={k:a[k] for k in a.files}
        bf=pd.DataFrame({k:v for k,v in bank.items() if k not in ['text','image']});q=f.copy()
        q['sample_id']='newsclippings:'+q.row_id;q['group_id']='newsclippings:'+q.cluster_id
        q['input_hash']=[digest([r.caption,r.sha256]) for r in q.itertuples()]
        # Existing bank's exact casefold key contract, independently of normalized text clustering.
        q['text_key']=q.caption.str.casefold()
        m,n,k=margins(z['clip_text'].astype(np.float32),z['clip_image'].astype(np.float32),q,bank['text'],bank['image'],bf,16,64)
        require(np.isfinite(m).all() and np.allclose(m[:,0],f.cosine,atol=1e-6),'BRAS score contract failed')
        f['m_text']=m[:,1];f['m_image']=m[:,2];f['bras_mean']=(m[:,1]+m[:,2])/2;f['valid_references']=n
    f['feature_provenance']=mode;path=run.results/f'{mode}/predictions.csv';csv(path,f);seal(path,run.binding())
    return {'mode':mode,'rows':len(f),'pairs':f.pair_id.nunique(),'clusters':f.cluster_id.nunique(),'scores':[k for k in run.protocol()['scores'] if k in f]}

def local_manifest(run,root):
    root=Path(root).resolve();path=root/'data.json';require(path.is_file(),'Authorized origin/data.json is missing')
    obj=read(path);require(isinstance(obj,list),'Expected official VisualNews list metadata')
    lookup={str(x['id']):x for x in obj};require(len(lookup)==len(obj),'Duplicate VisualNews IDs')
    selected=pd.read_csv(run.results/'selection_manifest.csv',keep_default_na=False,dtype={'caption_id':str,'image_id':str})
    f=join_metadata(selected.drop(columns=['source','image_source','filename','sha256','pixel_sha256','acquisition_status','cluster_id']),lookup)
    official=read(run.data/'metadata/test.json')
    for row in f.itertuples():
        require(row.metadata_join and row.caption==official[row.caption_id]['caption'],'Local caption does not match official supplementary metadata')
        canonical=lambda x:str(x).replace('\\','/').removeprefix('./').removeprefix('visual_news/').removeprefix('origin/')
        require(canonical(row.filename)==canonical(official[row.image_id]['image_path']),'Local image-ID/path join differs from official')
    f['filename']=f.filename.map(canonical)
    # Preserve author article/source provenance for clustering, including the mismatched image's article.
    joined=enriched(run).set_index('row_id')
    for col in ['source','image_source','article_key','image_article_key']:f[col]=f.row_id.map(joined[col])
    f=validate_images(f,root)
    known=pd.read_csv(run.root/'data/processed/manifest.csv',keep_default_na=False)
    fresh=run.root/'data/features/research_extension/fresh_holdout/manifest.csv'
    if fresh.exists():known=pd.concat([known,pd.read_csv(fresh,keep_default_na=False)],ignore_index=True)
    clean=lambda s:s.str.replace(r'\s+',' ',regex=True).str.strip().str.casefold()
    overlaps=f.text_key.isin(set(clean(known.text)))
    for col in ['sha256','pixel_sha256']:overlaps|=f[col].ne('')&f[col].isin(set(known[col]))
    f['known_fakeddit_overlap']=overlaps
    eligible=f.acquisition_status.eq('verified')&~overlaps
    valid=f.groupby('pair_id').apply(lambda g:bool(eligible.loc[g.index].all()),include_groups=False)
    f['complete_group_retained']=f.pair_id.isin(valid[valid].index)
    csv(run.results/'local_acquisition_audit.csv',f);use=f[f.complete_group_retained].copy()
    require(len(use)>0,'No complete authorized nonoverlapping groups; no substitution allowed')
    use['cluster_id']=linked_clusters(use);csv(run.results/'local_manifest.csv',use)
    receipt={'authorized_root':str(root),'local_metadata_sha256':sha(path),'selected_pairs':1000,'retained_pairs':use.pair_id.nunique(),
      'manifest_sha256':sha(run.results/'local_manifest.csv'),'protocol_sha256':sha(run.results/'protocol.json'),'overlapping_rows':int(overlaps.sum()),
      'missingness':f.groupby(['strategy','source','acquisition_status']).size().reset_index(name='rows').to_dict('records')}
    write(run.results/'local_input_receipt.json',receipt);seal(run.results/'local_manifest.csv',receipt);return use

def extract_local(run,root):
    import torch
    from PIL import Image,ImageOps
    from transformers import AutoTokenizer,BertModel,CLIPProcessor,CLIPModel
    from torchvision.models import resnet50,ResNet50_Weights
    from ...feature_extractors import BERT,BERT_REVISION,CLIP,CLIP_REVISION
    run.frozen();f=local_manifest(run,root);require_images(f,root)
    torch.set_num_threads(6);torch.use_deterministic_algorithms(True)
    weights=Path(torch.hub.get_dir())/'checkpoints/resnet50-11ad3fa6.pth';require(weights.is_file(),'Original cached ResNet weights missing')
    binding={**run.binding(),'manifest_sha256':sha(run.results/'local_manifest.csv'),
      'encoder_contract_sha256':sha(run.root/'data/features/extraction.json'),'resnet_weights_sha256':sha(weights),'feature_provenance':'pinned_local_real_images'}
    report={'encoders':{},'batch_size':16,'threads':6,'cpu_env':os.environ.get('PROCESSOR_IDENTIFIER','unknown'),'platform':platform.platform(),'cuda_used':False,'real_image_rows':len(f)}
    results={};start_all=time.perf_counter()
    for kind in ['bert','resnet','clip']:
        chunk_size=128;pending=[];parts=[]
        for start in range(0,len(f),chunk_size):
            rows=f.iloc[start:start+chunk_size];dest=run.data/f'pinned_chunks/{kind}_{start:06d}.npz'
            bound={**binding,'row_ids':rows.row_id.tolist()}
            if dest.exists():verify(dest,bound)
            else:pending.append((start,dest,bound))
            parts.append(dest)
        start_load=time.perf_counter();model=processor=None;inference=0.
        if pending:
            if kind=='bert':
                processor=AutoTokenizer.from_pretrained(BERT,revision=BERT_REVISION,local_files_only=True)
                model=BertModel.from_pretrained(BERT,revision=BERT_REVISION,local_files_only=True)
            elif kind=='resnet':
                processor=ResNet50_Weights.IMAGENET1K_V2.transforms();model=resnet50(weights=None)
                model.load_state_dict(torch.load(weights,map_location='cpu',weights_only=True));model.fc=torch.nn.Identity()
            else:
                processor=CLIPProcessor.from_pretrained(CLIP,revision=CLIP_REVISION,use_fast=False,local_files_only=True)
                model=CLIPModel.from_pretrained(CLIP,revision=CLIP_REVISION,local_files_only=True)
            model.eval().requires_grad_(False)
        load_seconds=time.perf_counter()-start_load
        for start,dest,bound in pending:
            rows=f.iloc[start:start+chunk_size];values={};begin=time.perf_counter()
            for i in range(0,len(rows),16):
                batch=rows.iloc[i:i+16];images=[]
                try:
                    if kind!='bert':
                        for r in batch.itertuples():
                            require(sha(Path(root)/r.filename)==r.sha256,'Image changed during extraction')
                            with Image.open(Path(root)/r.filename) as im:
                                im.seek(0);images.append(ImageOps.exif_transpose(im).convert('RGB'))
                    with torch.inference_mode():
                        if kind=='bert':out={'text':model(**processor(batch.caption.tolist(),return_tensors='pt',padding=True,truncation=True,max_length=64)).last_hidden_state[:,0,:]}
                        elif kind=='resnet':out={'image':model(torch.stack([processor(im) for im in images]))}
                        else:
                            raw=model(**processor(text=batch.caption.tolist(),images=images,return_tensors='pt',padding=True,truncation=True,max_length=77))
                            out={'clip_text':torch.nn.functional.normalize(raw.text_embeds,dim=-1),'clip_image':torch.nn.functional.normalize(raw.image_embeds,dim=-1)}
                        for k,v in out.items():values.setdefault(k,[]).append(v.numpy().astype(np.float32))
                finally:
                    for im in images:im.close()
            elapsed=time.perf_counter()-begin;inference+=elapsed
            arrays={k:np.concatenate(v) for k,v in values.items()};npz(dest,bound,**arrays)
            write(str(dest)+'.timing.json',{'inference_seconds':elapsed,'rows':len(rows),'kind':kind})
            print(f'{kind} {start+len(rows)}/{len(f)}',flush=True)
        for dest in parts:
            with np.load(dest,allow_pickle=False) as a:
                for k in a.files:results.setdefault(k,[]).append(a[k])
        report['encoders'][kind]={'load_seconds_this_invocation':load_seconds,'inference_seconds_this_invocation':inference,'new_chunks':len(pending),'total_chunks':len(parts),
            'total_saved_chunk_inference_seconds':sum(read(str(p)+'.timing.json')['inference_seconds'] for p in parts)}
        del model,processor;gc.collect()
    arrays={k:np.concatenate(v) for k,v in results.items()};arrays['similarity']=np.einsum('ij,ij->i',arrays['clip_text'],arrays['clip_image'])[:,None]
    npz(run.data/'pinned_features.npz',binding,ids=f.row_id.to_numpy(dtype=str),**arrays)
    report['total_seconds_this_invocation']=time.perf_counter()-start_all
    if not (run.results/'encoder_profile.json').exists():write(run.results/'encoder_profile.json',report)
    return report

def transfer(run,root=None):
    import torch
    from ...models import Classifier
    from .statistics import signflip,holm
    run.frozen();require((run.results/'local_manifest.csv').exists(),'Frozen Fakeddit transfer blocked: no verified local image manifest')
    receipt=read(run.results/'local_input_receipt.json');root=root or receipt['authorized_root']
    f=pd.read_csv(run.results/'local_manifest.csv',keep_default_na=False,dtype={'caption_id':str,'image_id':str});require_images(f,root)
    z=load_features(run,'pinned');require(z['ids'].tolist()==f.row_id.tolist(),'Transfer feature/row mismatch')
    for k,d in [('text',768),('image',2048),('similarity',1)]:require(k in z and z[k].shape==(len(f),d),'Full BERT/ResNet/CLIP pipeline required')
    freeze_analysis(run);normal=np.load(run.root/'models/normalization.npz',allow_pickle=False)
    x={k:torch.from_numpy(((z[k]-normal[k+'_mean'])/np.maximum(normal[k+'_std'],1e-6)).astype(np.float32)) for k in ['text','image','similarity']}
    torch.set_num_threads(6);start=time.perf_counter();head_seconds=0.;report={'task':'Fakeddit probability sensitivity to correspondence; not external fake-news accuracy','seeds':[]}
    for seed in [42,123,2026]:
        out=f.copy();out['cosine']=z['similarity'][:,0]
        head_start=time.perf_counter()
        for name,kind in [('A','baseline'),('B','proposed')]:
            ck=torch.load(run.root/f'models/final/{kind}_{seed}.pt',map_location='cpu',weights_only=True)
            require(ck['seed']==seed and ck['model']==kind,'Checkpoint identity mismatch')
            model=Classifier(kind,ck['config']['dimension'],ck['config']['dropout']);model.load_state_dict(ck['state_dict']);model.eval().requires_grad_(False)
            with torch.inference_mode():
                logits=torch.cat([model(x['text'][i:i+512],x['image'][i:i+512],x['similarity'][i:i+512]) for i in range(0,len(f),512)])
            out['p_'+name]=torch.softmax(logits,dim=1)[:,1].numpy();out['decision_'+name]=logits.argmax(1).numpy()
        head_seconds+=time.perf_counter()-head_start
        policy=read(run.root/f'models/research_extension/ccr/full/policy_{seed}.json');require(not policy['guard_passed'],'CCR is not identity B for this policy')
        out['p_CCR_identity']=out.p_B
        path=run.results/f'transfer/predictions_{seed}.csv';csv(path,out);seal(path,run.binding())
        p=out[out.falsified.eq(0)].set_index('pair_id').sort_index();n=out[out.falsified.eq(1)].set_index('pair_id').loc[p.index]
        effects=pd.DataFrame({'pair_id':p.index,'cluster_id':p.cluster_id.to_numpy(),'strategy':p.strategy.to_numpy(),'cosine_drop':p.cosine.to_numpy()-n.cosine.to_numpy()})
        from sklearn.metrics import roc_auc_score
        for name in ['A','B']:
            effects['drop_'+name]=p['p_'+name].to_numpy()-n['p_'+name].to_numpy();effects['absolute_shift_'+name]=np.abs(effects['drop_'+name])
            effects['flip_'+name]=(p['decision_'+name].to_numpy()!=n['decision_'+name].to_numpy()).astype(int)
        d=effects.drop_B-effects.drop_A;keys,codes=np.unique(effects.cluster_id,return_inverse=True);rng=np.random.default_rng(20261108);boot=[]
        metric_names=['drop_A','drop_B','absolute_shift_A','absolute_shift_B','flip_A','flip_B']
        values=effects[metric_names].to_numpy()
        for _ in range(5000):
            w=np.bincount(rng.integers(0,len(keys),len(keys)),minlength=len(keys))[codes];boot.append(w@values/w.sum())
        boot=np.array(boot);corr=np.corrcoef(effects.cosine_drop,d)[0,1]
        report['seeds'].append({'classifier_seed':seed,'pairs':len(p),'clusters':len(keys),'means':dict(zip(metric_names,values.mean(0).tolist())),
           'ci95':dict(zip(metric_names,np.quantile(boot,[.025,.975],axis=0).T.tolist())),
           'B_minus_A_signed_probability_drop':float(d.mean()),'drop_difference_ci95':np.quantile(boot[:,1]-boot[:,0],[.025,.975]).tolist(),
           'cluster_signflip_p':signflip(d,effects.cluster_id),'cosine_drop_relation_to_B_minus_A_drop_pearson':float(corr) if np.isfinite(corr) else None,
           'pairing_status_auc_cross_task_proxy':{k:float(roc_auc_score(1-out.falsified,out['p_'+k])) for k in ['A','B']},
           'by_strategy':effects.groupby('strategy')[list(effects.columns[4:])].mean().to_dict('index')})
        csv(run.results/f'transfer/paired_{seed}.csv',effects)
    for row,adj in zip(report['seeds'],holm([x['cluster_signflip_p'] for x in report['seeds']])):row['holm_adjusted_p']=adj
    report['head_loading_and_inference_seconds']=head_seconds
    report['head_inference_and_analysis_seconds']=time.perf_counter()-start;report['note']='Seeds reuse the same groups and are not independent sample replication. CCR identity is descriptive.'
    write(run.results/'transfer/statistics.json',report);return report
