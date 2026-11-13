"""Read-only provenance audit and historical-artifact preservation checks."""
from __future__ import annotations
import time
import html
import unicodedata
from urllib.parse import urlsplit
import numpy as np
import pandas as pd
import torch
from .common import (manifest,sha,require,read_json,write_json,load_npz,load_historical,
    fit_scaler,environment,lock_protocol,peak_memory,LABEL6)

def historical_paths(project):
    paths=[]
    for rel in ['src','docs','models','results','data/processed','data/features']:
        paths.extend(p for p in (project/rel).rglob('*') if p.is_file() and 'research_extension' not in p.parts
            and '__pycache__' not in p.parts and p.name!='research_extension_audit.md')
    paths.extend(project/x for x in ['README.md','demo.py','run_experiments.py','requirements.txt'])
    return sorted(paths)


def audit_fresh_holdout(ctx, retained, official, audited_groups):
    base=ctx.cache/'fresh_holdout';path=base/'manifest.csv';receipt_path=base/'manifest.json'
    if not path.exists() and not receipt_path.exists():
        return {'status':'not acquired'}
    require(path.is_file() and receipt_path.is_file(),'Fresh holdout manifest/receipt is incomplete')
    receipt=read_json(receipt_path);require(receipt['sha256']==sha(path),'Fresh holdout manifest SHA mismatch')
    require(receipt['spec']['retained_manifest_sha256']==sha(ctx.project/'data/processed/manifest.csv'),
            'Fresh holdout was selected against another retained manifest')
    fresh=pd.read_csv(path,keep_default_na=False,dtype={'sample_id':str,'group_id':str})
    require(fresh.sample_id.is_unique and len(fresh)==receipt['samples'],'Fresh holdout IDs/count invalid')
    joined=fresh.merge(official,on='sample_id',how='left',validate='one_to_one')
    require(joined.official_binary.notna().all() and joined.label.eq(joined.official_binary).all()
        and joined.label6.eq(joined.official_sixway).all() and joined.original_split.eq(joined.official_split).all(),
        'Fresh holdout official IDs/labels/split mismatch')
    groups=fresh.merge(audited_groups,on='sample_id',how='left',suffixes=('','_audited'),validate='one_to_one')
    require(groups.group_id.eq(groups.group_id_audited).all() and groups.eligible.all(),
        'Fresh holdout source-family audit mismatch')
    fresh['text_key_check']=[unicodedata.normalize('NFKC',str(x)).casefold() for x in fresh.text]
    retained_text=set(retained.text_key.astype(str))
    retained_groups=set(retained.group_id.astype(str));retained_ids=set(retained.sample_id.astype(str))
    retained_hashes=set(retained.sha256.astype(str));retained_pixels=set(retained.pixel_sha256.astype(str))
    retained_urls=set()
    for value in retained.image_url.astype(str):
        parsed=urlsplit(html.unescape(value));retained_urls.add(parsed.netloc.casefold()+parsed.path)
    fresh_urls=set()
    for value in fresh.image_url.astype(str):
        parsed=urlsplit(html.unescape(value));fresh_urls.add(parsed.netloc.casefold()+parsed.path)
    within={'sample_id':int(fresh.sample_id.duplicated().sum()),
        'group_id':int(fresh.group_id.duplicated().sum()),
        'normalized_text':int(pd.Series(fresh.text_key_check).duplicated().sum()),
        'image_url':len(fresh)-len(fresh_urls),
        'file_sha256':int(fresh.sha256.duplicated().sum()),
        'pixel_sha256':int(fresh.pixel_sha256.duplicated().sum())}
    require(not any(within.values()),f'Fresh holdout contains internal duplicates: {within}')
    overlaps={'sample_id':len(set(fresh.sample_id)&retained_ids),
        'group_id':len(set(fresh.group_id.astype(str))&retained_groups),
        'normalized_text':len(set(fresh.text_key_check)&retained_text),
        'image_url':len(fresh_urls&retained_urls),
        'file_sha256':len(set(fresh.sha256.astype(str))&retained_hashes),
        'pixel_sha256':len(set(fresh.pixel_sha256.astype(str))&retained_pixels)}
    require(not any(overlaps.values()),f'Fresh holdout overlaps retained dataset: {overlaps}')
    for row in fresh.itertuples():
        image=base/'images'/row.image_path
        require(image.is_file() and sha(image)==row.sha256,f'Fresh holdout image changed: {row.sample_id}')
    counts={str(int(k)):int(v) for k,v in fresh.label.value_counts().items()}
    require(counts==receipt['class_counts'],'Fresh holdout class counts differ from receipt')
    vectors=base/'features/vectors.npz';vectors_receipt=base/'features/vectors.json'
    evaluation=base/'evaluation.json';corrected=base/'paired_comparisons_corrected.receipt.json'
    vector_state={'status':'not extracted'}
    if vectors.exists() or vectors_receipt.exists():
        require(vectors.is_file() and vectors_receipt.is_file(),'Fresh vector artifact/receipt is incomplete')
        require(read_json(vectors_receipt)['sha256']==sha(vectors),'Fresh vector artifact SHA mismatch')
        vector_state={'status':'passed','sha256':sha(vectors)}
    return {'status':'passed','samples':len(fresh),'class_counts':counts,'manifest_sha256':sha(path),
        'overlap_with_retained':overlaps,'within_holdout_duplicates':within,
        'source':'unused official Fakeddit public-test metadata rows',
        'same_dataset_as_training':True,'cross_dataset_evaluation':False,
        'vectors':vector_state,
        'evaluation':({'status':'completed','sha256':sha(evaluation)} if evaluation.is_file()
            else {'status':'not evaluated'}),
        'corrected_paired_statistics_receipt':({'status':'present','sha256':sha(corrected)} if corrected.is_file()
            else {'status':'not written'})}

def audit(ctx,verify_images=False,verify_preservation=False):
    start=time.perf_counter();dest=ctx.output/'results/research_extension/audit'
    inventory_path=dest/'original_hashes.json'
    if inventory_path.exists():
        inventory=read_json(inventory_path)
        mismatch=[rel for rel,h in inventory.items() if not (ctx.project/rel).is_file() or sha(ctx.project/rel)!=h]
        require(not mismatch,f'Historical artifacts changed: {mismatch[:10]}')
    else:
        require(not verify_preservation,'No original inventory to verify')
        inventory={p.relative_to(ctx.project).as_posix():sha(p) for p in historical_paths(ctx.project)}
        write_json(inventory_path,inventory,immutable=True)
    if verify_preservation:
        result={'status':'passed','historical_files':len(inventory),'seconds':time.perf_counter()-start,
            'meaning':'Every inventoried original file is byte-identical; no historical metrics/checkpoints overwritten'}
        write_json(dest/'preservation_check.json',result)
        return result
    frame=manifest(ctx,scope=False);checks={}
    for key in ['sample_id','group_id','sha256','pixel_sha256','text_key']:
        duplicate=int(frame[key].duplicated().sum())
        cross=int(frame.groupby(key).split.nunique().gt(1).sum())
        require(duplicate==0 and cross==0,f'Retained duplicate/source overlap: {key}')
        checks[key]={'duplicates':duplicate,'cross_split':cross}
    sources=read_json(ctx.project/'data/raw/fakeddit/sources.json')
    official=[]
    for item in sources['files']:
        path=ctx.project/'data/raw/fakeddit'/item['file']
        require(sha(path)==item['sha256'],f'Official metadata SHA mismatch: {path}')
        part=pd.read_csv(path,sep='\t',usecols=['id','2_way_label','6_way_label'],dtype={'id':str})
        part['split']=item['split'];official.append(part)
    original=pd.concat(official).rename(columns={'id':'sample_id','2_way_label':'official_binary','6_way_label':'official_sixway','split':'official_split'})
    joined=frame.merge(original,on='sample_id',how='left',validate='one_to_one')
    require(joined.official_binary.notna().all() and joined.label.eq(joined.official_binary).all() and
        joined.label6.eq(joined.official_sixway).all() and joined.split.eq(joined.official_split).all(),'Official ID/label/split mismatch')
    saved_groups=pd.read_csv(ctx.project/'data/processed/metadata_audit.csv.gz',dtype={'sample_id':str,'group_id':str})
    joined=frame.merge(saved_groups,on='sample_id',suffixes=('','_audited'),validate='one_to_one')
    require(len(joined)==len(frame) and joined.group_id.eq(joined.group_id_audited).all() and joined.eligible.all(),'Source-family/eligibility mismatch')
    raw=load_historical(ctx,frame);normalizer=load_npz(ctx.project/'models/normalization.npz')
    for key,value in raw.items():
        scale=fit_scaler(value,frame.split.eq('train').to_numpy())
        require(np.allclose(scale['mean'],normalizer[key+'_mean'],atol=1e-6,rtol=0) and
            np.allclose(scale['std'],normalizer[key+'_std'],atol=1e-6,rtol=0),'Historical scaler does not match train-only fit')
    del raw
    lock=read_json(ctx.project/'models/locked_config.json');extraction=read_json(ctx.project/'data/features/extraction.json')
    checkpoint_records=[]
    for path in sorted((ctx.project/'models/final').glob('*.pt')):
        c=torch.load(path,map_location='cpu',weights_only=True)
        require(c['manifest_sha256']==lock['manifest_sha256'] and c['normalization_sha256']==sha(ctx.project/'models/normalization.npz'),'Historical checkpoint provenance mismatch')
        require(c['config']==lock['config'] and c['budget']==lock['budget'] and c['feature_contract']==extraction['contract'],'Historical checkpoint contract mismatch')
        checkpoint_records.append({'path':path.relative_to(ctx.project).as_posix(),'sha256':sha(path),
            **{k:c[k] for k in ['model','seed','parameters','best_epoch']}})
    require(len(checkpoint_records)==12,'Expected 12 historical final checkpoints')
    images={'status':'not rehashed in this command','checked':0}
    if verify_images:
        for r in frame.itertuples():
            path=ctx.project/r.image_path
            require(path.is_file() and sha(path)==r.sha256,f'Missing/changed image {r.sample_id}')
        images={'status':'passed','checked':len(frame)}
    # Existence-only independent-evaluation availability check; no new labels selected by performance.
    retained=set(frame.sample_id)
    official_test=original[original.official_split.eq('test')]
    unused=[sid for sid in official_test.sample_id if sid not in retained]
    cached_ids={p.stem for p in (ctx.project/'data/raw/fakeddit/images').glob('*') if p.is_file()}
    local_unused=sorted(set(unused)&cached_ids)
    fresh_holdout=audit_fresh_holdout(ctx,frame,original,saved_groups)
    result={'status':'passed','manifest_sha256':lock['manifest_sha256'],'counts':frame.split.value_counts().to_dict(),
        'duplicates':checks,'official_labels_and_splits':'verified by one-to-one ID join against SHA-verified official TSVs',
        'source_families':'match original metadata-audit connected groups; independent event identity not established',
        'normalization':'recomputed and matched using training rows only','label6_mapping':LABEL6,
        'clip_vectors_cached_historically':False,'historical_files':len(inventory),'checkpoints':checkpoint_records,
        'images':images,'git_directory_present':(ctx.project/'.git').exists(),
        'unused_official_test_metadata_rows':len(unused),'local_unused_test_image_ids':local_unused,
        'fresh_holdout':fresh_holdout,
        'external_benchmark_evaluation':'Not run; no compatible external benchmark or validated label crosswalk is present.',
        'environment':environment(),'seconds':time.perf_counter()-start,**peak_memory()}
    write_json(dest/'audit.json',result)
    lock_protocol(ctx)
    return {'status':'passed','historical_files':len(inventory),'rows':len(frame),'path':str(dest/'audit.json')}
