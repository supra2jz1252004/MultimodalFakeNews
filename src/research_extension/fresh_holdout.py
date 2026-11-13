"""Acquire a small, disjoint Fakeddit official-test holdout and evaluate frozen heads."""
from __future__ import annotations

import hashlib
import html
import json
import threading
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from pathlib import Path
from urllib.parse import urlsplit

import numpy as np
import pandas as pd
import requests
import torch
from PIL import Image, ImageOps
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .common import (CLIP_ID, CLIP_REVISION, digest, load_npz, manifest,
                     peak_memory, read_json, require, sha, standardize,
                     write_json, write_npz)

FRESH_SEED = 20261106
FRESH_SCOPE = "previously unused official Fakeddit public-test metadata rows"
THREAD = threading.local()


def root(ctx):
    return ctx.cache / 'fresh_holdout'


def _clean_text(value):
    return ' '.join(unicodedata.normalize('NFKC', str(value)).split())


def _image_key(value):
    parsed = urlsplit(html.unescape(str(value)))
    return parsed.netloc.casefold() + parsed.path


def _candidate_rows(ctx):
    raw = ctx.project / 'data/raw/fakeddit/multimodal_test_public.tsv'
    audit = ctx.project / 'data/processed/metadata_audit.csv.gz'
    retained_path = ctx.project / 'data/processed/manifest.csv'
    require(raw.is_file() and audit.is_file() and retained_path.is_file(),
            'Official test metadata, its audited groups, and the retained manifest are required')
    retained = pd.read_csv(retained_path, keep_default_na=False, dtype=str)
    metadata = pd.read_csv(raw, sep='\t', usecols=['id','clean_title','image_url','linked_submission_id',
        'hasImage','2_way_label','6_way_label','subreddit','created_utc'], dtype=str, keep_default_na=False)
    metadata = metadata.rename(columns={'id':'sample_id','clean_title':'text',
        '2_way_label':'label','6_way_label':'label6'})
    metadata['label'] = pd.to_numeric(metadata.label, errors='raise').astype(int)
    metadata['label6'] = pd.to_numeric(metadata.label6, errors='raise').astype(int)
    require(metadata.label.isin([0,1]).all() and metadata.label6.isin(range(6)).all(),
            'Official binary/six-way labels are invalid')
    require((metadata.label.eq(1) == metadata.label6.eq(0)).all(), 'Official label mappings disagree')
    metadata['text'] = metadata.text.map(_clean_text)
    metadata['text_key'] = metadata.text.str.casefold()
    metadata['image_key'] = metadata.image_url.map(_image_key)
    audit_frame = pd.read_csv(audit, dtype={'sample_id':str,'group_id':str})
    metadata = metadata.merge(audit_frame, on='sample_id', how='left', validate='one_to_one')
    require(metadata.split.eq('test').all() and metadata.group_id.notna().all(),
            'Official test group audit could not be joined one-to-one')
    eligible = metadata.eligible.astype(str).str.casefold().eq('true')
    metadata = metadata[eligible & metadata.hasImage.str.casefold().eq('true')
                        & metadata.text.ne('') & metadata.image_url.str.startswith(('http://','https://'))]
    metadata = metadata[~metadata.sample_id.isin(set(retained.sample_id))]
    for col in ['group_id','text_key','image_key']:
        if col == 'group_id':
            used = set(retained.group_id)
        elif col == 'text_key':
            used = set(retained.text.map(_clean_text).str.casefold())
        else:
            used = set(retained.image_url.map(_image_key))
        metadata = metadata[~metadata[col].isin(used)]
    metadata = metadata.sort_values('sample_id').reset_index(drop=True)
    metadata['rank'] = metadata.sample_id.map(lambda sid: hashlib.sha256(
        f'{FRESH_SEED}\0{sid}'.encode()).hexdigest())
    return metadata.sort_values(['label','rank','sample_id']).reset_index(drop=True), retained


def _download(row, destination):
    sid = str(row['sample_id'])
    result = {'sample_id':sid,'status':'failed','reason':'','retrieved_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
    try:
        if not hasattr(THREAD, 'session'):
            THREAD.session = requests.Session()
            THREAD.session.headers['User-Agent'] = 'Mozilla/5.0 (compatible; academic-dataset-retrieval/1.0)'
            retry = Retry(total=1, backoff_factor=.3, status_forcelist=[429,500,502,503,504],
                          allowed_methods=['GET'], raise_on_status=False,
                          respect_retry_after_header=False)
            THREAD.session.mount('https://', HTTPAdapter(max_retries=retry))
            THREAD.session.mount('http://', HTTPAdapter(max_retries=retry))
        url = html.unescape(str(row['image_url']))
        if url.startswith('http://i.imgur.com/'):
            url = 'https://' + url[len('http://'):]
        started = time.monotonic()
        with THREAD.session.get(url, stream=True, timeout=(8,20)) as response:
            if response.status_code != 200:
                result['reason'] = f'http_{response.status_code}'
                return result
            blocks=[]; total=0
            for block in response.iter_content(65536):
                if time.monotonic()-started > 90:
                    result['reason']='transfer_over_90_seconds'; return result
                total += len(block)
                if total > 15*1024*1024:
                    result['reason']='download_over_15_mib'; return result
                blocks.append(block)
            content=b''.join(blocks)
        with Image.open(BytesIO(content)) as opened:
            opened.verify()
        with Image.open(BytesIO(content)) as opened:
            opened.seek(0)
            rgb=ImageOps.exif_transpose(opened).convert('RGB')
            rgb.load()
            pixels=hashlib.sha256(f'{rgb.width}x{rgb.height}:'.encode()+rgb.tobytes()).hexdigest()
            gray=np.asarray(rgb.convert('L').resize((9,8)))
            bits=(gray[:,1:] > gray[:,:-1]).ravel()
            dhash=f'{int("".join(bits.astype(int).astype(str)),2):016x}'
            fmt=opened.format
            width,height=rgb.size
            extrema=rgb.getextrema()
            uniform=all(low==high for low,high in extrema)
            rgb.close()
        ext={'JPEG':'.jpg','PNG':'.png','WEBP':'.webp','GIF':'.gif','BMP':'.bmp','TIFF':'.tif'}.get(fmt)
        if ext is None:
            result['reason']='unsupported_image_format'; return result
        file_hash=hashlib.sha256(content).hexdigest()
        path=destination/f'{sid}{ext}'
        temp=path.with_suffix(path.suffix+'.part')
        temp.write_bytes(content); temp.replace(path)
        result.update(status='ok',reason='',image_path=path.name,sha256=file_hash,pixel_sha256=pixels,
            original_sha256=file_hash,width=width,height=height,format=fmt,bytes=len(content),
            dhash=dhash,uniform_rgb=uniform,resolved_url=response.url)
    except requests.RequestException as exc:
        result['reason']=type(exc).__name__
    except (OSError,ValueError,Image.DecompressionBombError) as exc:
        result['reason']='image_decode_'+type(exc).__name__
    return result


def prepare(ctx, per_class=500, workers=12):
    """Fetch only a disjoint balanced slice of unused official-test examples."""
    require(per_class > 0 and workers > 0, 'Holdout count and workers must be positive')
    dest=root(ctx); images=dest/'images'; images.mkdir(parents=True,exist_ok=True)
    manifest_path=dest/'manifest.csv'; receipt_path=dest/'manifest.json'
    spec={'scope':FRESH_SCOPE,'selection_seed':FRESH_SEED,'per_class':per_class,
          'official_test_metadata_sha256':sha(ctx.project/'data/raw/fakeddit/multimodal_test_public.tsv'),
          'retained_manifest_sha256':sha(ctx.project/'data/processed/manifest.csv'),
          'metadata_group_audit_sha256':sha(ctx.project/'data/processed/metadata_audit.csv.gz'),
          'group_exclusions':['all retained train/validation/test linked-submission families',
              'all retained normalized texts and image URLs','within-holdout groups, texts, URLs and downloaded content hashes'],
          'image_selection':'Stable SHA-256 rank of official sample IDs; label-stratified 0=fake, 1=real',
          'download_limit':'Only selected candidates, maximum 15 MiB per image; no dataset archive download'}
    if manifest_path.exists():
        require(read_json(receipt_path)['spec']==spec,'Fresh-holdout protocol differs; choose a new output namespace')
        require(read_json(receipt_path)['sha256']==sha(manifest_path),'Fresh-holdout manifest hash mismatch')
        existing=pd.read_csv(manifest_path,keep_default_na=False)
        for row in existing.itertuples():
            path=dest/'images'/row.image_path
            require(path.is_file() and sha(path)==row.sha256,'Fresh-holdout image changed or is missing')
        return {'status':'verified existing holdout','samples':len(existing),'manifest_sha256':sha(manifest_path)}
    candidates,retained=_candidate_rows(ctx)
    counts={int(k):0 for k in [0,1]}; selected=[]; decisions=[]
    existing_ids=set(retained.sample_id.astype(str))
    used_groups=set(retained.group_id.astype(str))
    used_text=set(retained.text.map(_clean_text).str.casefold())
    used_urls=set(retained.image_url.map(_image_key))
    used_files=set(retained.sha256.astype(str)); used_pixels=set(retained.pixel_sha256.astype(str))
    download_log=dest/'download_records.jsonl'; records={}
    if download_log.exists():
        for line in download_log.read_text(encoding='utf-8').splitlines():
            if line.strip():
                rec=json.loads(line); records[rec['sample_id']]=rec
    attempted_this_run=set()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for label in (0,1):
            part=candidates[candidates.label.eq(label)]
            pending=part.to_dict('records')
            for offset in range(0,len(pending),64):
                if counts[label]>=per_class: break
                batch=pending[offset:offset+64]
                futures={}
                for row in batch:
                    sid=str(row['sample_id']); cached=records.get(sid)
                    if cached and cached.get('status')=='ok':
                        path=images/cached['image_path']
                        require(path.is_file() and sha(path)==cached['sha256'],
                                f'Cached fresh image changed or is missing: {sid}')
                        continue
                    if sid in attempted_this_run: continue
                    attempted_this_run.add(sid)
                    futures[pool.submit(_download,row,images)]=sid
                for future in as_completed(futures):
                    rec=future.result(); records[rec['sample_id']]=rec
                if futures:
                    with download_log.open('a',encoding='utf-8') as stream:
                        for sid in futures.values():
                            stream.write(json.dumps(records[sid],ensure_ascii=False)+'\n')
                with download_log.open('a',encoding='utf-8') as stream:
                    for row in batch:
                        sid=str(row['sample_id'])
                        rec=records.get(sid)
                        if not rec or rec.get('status')!='ok':
                            decisions.append({'sample_id':sid,'label':label,'decision':rec.get('reason','not_attempted') if rec else 'not_attempted'})
                            continue
                        checks=[('existing_id',sid in existing_ids),('source_family',str(row['group_id']) in used_groups),
                            ('normalized_text',row['text_key'] in used_text),('image_url',row['image_key'] in used_urls),
                            ('duplicate_file_hash',rec['sha256'] in used_files),('duplicate_pixel_hash',rec['pixel_sha256'] in used_pixels),
                            ('blank_image',bool(rec['uniform_rgb']))]
                        reason=next((name for name,matched in checks if matched),'')
                        if counts[label]>=per_class and not reason: reason='quota_reached'
                        decisions.append({'sample_id':sid,'label':label,'decision':reason or 'selected'})
                        if reason:
                            continue
                        item={**row,**rec,'original_split':'test','split':'fresh_test',
                              'input_hash':hashlib.sha256((str(row['text'])+'\0'+rec['sha256']).encode()).hexdigest()}
                        item['image_path']=rec['image_path']
                        selected.append(item); counts[label]+=1
                        existing_ids.add(sid);used_groups.add(str(row['group_id']))
                        used_text.add(str(row['text_key']));used_urls.add(str(row['image_key']))
                        used_files.add(rec['sha256']);used_pixels.add(rec['pixel_sha256'])
                print(f'Fresh test label {label}: {counts[label]}/{per_class}; examined {offset+len(batch)}/{len(part)} official candidates',flush=True)
    if any(counts[label] != per_class for label in (0,1)):
        write_json(dest/'partial.json',{'spec':spec,'counts':counts,'requested_per_class':per_class,
            'candidate_counts':candidates.label.value_counts().to_dict(),'download_records':len(records),
            'reason':'Candidate pool/download success exhausted before exact quotas; rerun to retry failed image URLs.'})
        raise RuntimeError(f'Could not fill balanced fresh holdout quotas: {counts}; no quota reduction applied')
    frame=pd.DataFrame(selected).sort_values(['label','sample_id']).reset_index(drop=True)
    require(frame.sample_id.is_unique and frame.label.value_counts().to_dict()=={0:per_class,1:per_class},
            'Fresh holdout IDs or class counts are invalid')
    require(not set(frame.sample_id.astype(str)) & set(retained.sample_id.astype(str)),
            'Fresh holdout overlaps retained IDs')
    require(not set(frame.group_id.astype(str)) & set(retained.group_id.astype(str)),
            'Fresh holdout overlaps retained source families')
    require(not set(frame.text_key.astype(str)) & set(retained.text.map(_clean_text).str.casefold()),
            'Fresh holdout overlaps retained normalized text')
    require(not set(frame.image_key.astype(str)) & set(retained.image_url.map(_image_key)),
            'Fresh holdout overlaps retained image URLs')
    keep=['sample_id','original_split','split','label','label6','text','image_url','image_path','group_id',
        'linked_submission_id','subreddit','created_utc','text_key','image_key','input_hash','sha256','pixel_sha256',
        'original_sha256','retrieved_utc','resolved_url','width','height','format','bytes','dhash']
    frame=frame[keep]
    frame.to_csv(manifest_path,index=False)
    pd.DataFrame(decisions).to_csv(dest/'selection_audit.csv',index=False)
    write_json(receipt_path,{'spec':spec,'sha256':sha(manifest_path),'samples':len(frame),
        'class_counts':{str(k):int(v) for k,v in frame.label.value_counts().items()},
        'source_family_overlap_with_retained':0,'text_overlap_with_retained':0,
        'image_url_overlap_with_retained':0,'retained_manifest_sha256':spec['retained_manifest_sha256'],
        'download_attempts':len(records),'selected_image_bytes':int(frame.bytes.sum())},immutable=True)
    return {'status':'completed','samples':len(frame),'manifest_sha256':sha(manifest_path),
        'class_counts':counts,'path':str(manifest_path)}


def _load_fresh(ctx):
    path=root(ctx)/'manifest.csv'; receipt=read_json(root(ctx)/'manifest.json')
    require(receipt['sha256']==sha(path),'Fresh holdout manifest hash mismatch')
    frame=pd.read_csv(path,keep_default_na=False,dtype={'sample_id':str,'group_id':str})
    require(frame.sample_id.is_unique and frame.split.eq('fresh_test').all(),'Fresh holdout IDs/split invalid')
    return frame


def extract(ctx, batch_size=16):
    """Extract frozen BERT, ResNet and CLIP vectors for fresh holdout rows only."""
    from torch.nn import functional as F
    from torchvision.models import ResNet50_Weights, resnet50
    from transformers import AutoTokenizer, BertModel, CLIPModel, CLIPProcessor
    from ..feature_extractors import BERT, BERT_REVISION

    require(batch_size>0,'Batch size must be positive')
    torch.set_num_threads(ctx.threads);torch.use_deterministic_algorithms(True)
    frame=_load_fresh(ctx); dest=root(ctx)/'features';dest.mkdir(parents=True,exist_ok=True)
    old_extract=read_json(ctx.project/'data/features/extraction.json')
    bert_max=old_extract['contract']['bert_max_length']
    contract={'schema':1,'samples_manifest_sha256':sha(root(ctx)/'manifest.csv'),
        'bert':BERT,'bert_revision':BERT_REVISION,'bert_max_length':bert_max,
        'bert_pooling':'last_hidden_state[:,0,:]','resnet':'torchvision ResNet50 IMAGENET1K_V2 fc=Identity',
        'resnet_preprocessing':'weights.transforms(): resize 232, center crop 224, ImageNet mean/std',
        'clip':CLIP_ID,'clip_revision':CLIP_REVISION,'clip_max_length':77,
        'clip_preprocessing':'CLIPProcessor use_fast=False; pinned revision; projected L2-normalized vectors',
        'image_decode':'first frame; EXIF transpose; RGB','dtype':'float32','threads':ctx.threads}
    write_json(dest/'contract.json',contract,immutable=True)
    chunk=dest/'vectors.npz'; receipt_path=dest/'vectors.json'
    if chunk.exists():
        r=read_json(receipt_path);require(r['sha256']==sha(chunk) and r['contract']==contract,
            'Fresh embedding artifact differs from its contract')
        arrays=load_npz(chunk);require(arrays['ids'].tolist()==frame.sample_id.tolist(),
            'Fresh embedding row ordering differs')
        return {'status':'verified existing vectors','samples':len(frame),'sha256':sha(chunk)}
    try:
        tokenizer=AutoTokenizer.from_pretrained(BERT,revision=BERT_REVISION,local_files_only=True)
        bert=BertModel.from_pretrained(BERT,revision=BERT_REVISION,local_files_only=True).eval().requires_grad_(False)
        clip_processor=CLIPProcessor.from_pretrained(CLIP_ID,revision=CLIP_REVISION,use_fast=False,local_files_only=True)
        clip=CLIPModel.from_pretrained(CLIP_ID,revision=CLIP_REVISION,local_files_only=True).eval().requires_grad_(False)
        weights=ResNet50_Weights.IMAGENET1K_V2
        weights_file=Path(torch.hub.get_dir())/'checkpoints'/Path(urlsplit(weights.url).path).name
        require(weights_file.is_file(),f'Pinned local ResNet weights not found: {weights_file}; no download attempted')
        resnet=resnet50(weights=None)
        state=torch.load(weights_file,map_location='cpu',weights_only=True)
        resnet.load_state_dict(state);resnet.fc=torch.nn.Identity();resnet.eval().requires_grad_(False)
        resnet_transform=weights.transforms()
    except Exception as exc:
        raise RuntimeError(f'Pinned local frozen encoders unavailable; no model download attempted: {exc}') from exc
    text_out=[]; image_out=[]; clip_text=[]; clip_image=[]; started=time.perf_counter()
    for start in range(0,len(frame),batch_size):
        batch=frame.iloc[start:start+batch_size]; images=[]
        try:
            for row in batch.itertuples():
                path=root(ctx)/'images'/row.image_path
                require(path.is_file() and sha(path)==row.sha256,f'Fresh holdout image changed: {row.sample_id}')
                with Image.open(path) as source:
                    source.seek(0);images.append(ImageOps.exif_transpose(source).convert('RGB'))
            text_tokens=tokenizer(batch.text.tolist(),return_tensors='pt',padding=True,
                truncation=True,max_length=bert_max)
            clip_inputs=clip_processor(text=batch.text.tolist(),images=images,return_tensors='pt',
                padding=True,truncation=True,max_length=77)
            with torch.inference_mode():
                btxt=bert(**text_tokens).last_hidden_state[:,0,:]
                imgs=resnet(torch.stack([resnet_transform(im) for im in images]))
                output=clip(**clip_inputs)
                ct=F.normalize(output.text_embeds,dim=-1);ci=F.normalize(output.image_embeds,dim=-1)
            text_out.append(btxt.cpu().numpy().astype(np.float32))
            image_out.append(imgs.cpu().numpy().astype(np.float32))
            clip_text.append(ct.cpu().numpy().astype(np.float32));clip_image.append(ci.cpu().numpy().astype(np.float32))
        finally:
            for im in images: im.close()
        if (start//batch_size+1)%8==0 or start+len(batch)==len(frame):
            print(f'Fresh frozen features {min(start+len(batch),len(frame))}/{len(frame)}; {time.perf_counter()-started:.1f}s',flush=True)
    arrays={'ids':frame.sample_id.to_numpy(dtype=str),'input_hash':frame.input_hash.to_numpy(dtype=str),
        'text':np.concatenate(text_out),'image':np.concatenate(image_out),
        'clip_text':np.concatenate(clip_text),'clip_image':np.concatenate(clip_image)}
    for name in ['text','image','clip_text','clip_image']:
        require(arrays[name].dtype==np.float32 and np.isfinite(arrays[name]).all(),f'Nonfinite fresh {name} embeddings')
    require(arrays['text'].shape==(len(frame),768) and arrays['image'].shape==(len(frame),2048)
        and arrays['clip_text'].shape==arrays['clip_image'].shape==(len(frame),512),'Fresh vector dimensions invalid')
    require(np.allclose(np.linalg.norm(arrays['clip_text'],axis=1),1,atol=2e-5)
        and np.allclose(np.linalg.norm(arrays['clip_image'],axis=1),1,atol=2e-5),'Fresh CLIP vectors are not normalized')
    write_npz(chunk,**arrays)
    r={'contract':contract,'sha256':sha(chunk),'samples':len(frame),
        'clip_cosine_min':float(np.einsum('ij,ij->i',arrays['clip_text'],arrays['clip_image']).min()),
        'clip_cosine_max':float(np.einsum('ij,ij->i',arrays['clip_text'],arrays['clip_image']).max()),
        'seconds':time.perf_counter()-started,**peak_memory()}
    write_json(receipt_path,r,immutable=True)
    return {'status':'completed','samples':len(frame),'sha256':r['sha256'],'seconds':r['seconds']}


def evaluate(ctx):
    """Evaluate every prespecified checkpoint on a new, disjoint official-test subset."""
    from .features import load_bank, random_control, unimodal_control
    from .reference import margins
    from .training import (VARIANTS,binding,make_head,predict,prepare_data,
                          verify_completed,extras)
    from .statistics import metrics,paired,holm
    from .evaluation import METRICS

    frame=_load_fresh(ctx); vectors_path=root(ctx)/'features/vectors.npz'
    receipt=read_json(root(ctx)/'features/vectors.json')
    require(receipt['sha256']==sha(vectors_path),'Fresh embedding hash mismatch')
    arrays=load_npz(vectors_path);require(arrays['ids'].tolist()==frame.sample_id.tolist(),
        'Fresh embedding/manifest order mismatch')
    full,train_features,train_labels,feature_receipt=prepare_data(ctx)
    p=read_json(ctx.models/'protocol.json')
    saved={}
    for kind in VARIANTS:
        for seed in p['seeds']:
            saved[kind,seed]=verify_completed(ctx,kind,seed,binding(ctx,kind,seed,feature_receipt))
    bank,bt,bv,_=load_bank(ctx)
    query=frame.copy(); query['text_key']=query.text.str.casefold()
    generated,valid_count,effective=margins(arrays['clip_text'],arrays['clip_image'],query,bt,bv,bank,
        k=p['top_k'],batch_size=ctx.batch_size)
    cosine=np.einsum('ij,ij->i',arrays['clip_text'],arrays['clip_image'])
    require(np.allclose(cosine,generated[:,0],rtol=0,atol=1e-6),'Fresh cosine/margin mismatch')
    # Match the project's frozen training normalization and never refit on fresh rows.
    normalizers=load_npz(ctx.models/'normalization.npz')
    text=standardize(arrays['text'],{'mean':normalizers['text_mean'],'std':normalizers['text_std']})
    image=standardize(arrays['image'],{'mean':normalizers['image_mean'],'std':normalizers['image_std']})
    raw={'similarity':cosine.reshape(-1,1)}
    random=random_control(frame.sample_id,ctx.seed)
    unimodal=unimodal_control(arrays['clip_text'],arrays['clip_image'],ctx.seed)
    gen={'bras':generated,'random':random,'unimodal':unimodal}
    variant_features={}
    for kind in VARIANTS:
        x=extras(kind,gen,raw['similarity'])
        variant_features[kind]=torch.from_numpy(standardize(x,{'mean':normalizers[kind+'_mean'],
            'std':normalizers[kind+'_std']}))
    features={'text':torch.from_numpy(text),'image':torch.from_numpy(image),'variants':variant_features}
    y=frame.label.to_numpy(dtype=int);idx=torch.arange(len(frame),dtype=torch.long)
    results=[];predictions={};confusion={};outdir=root(ctx)/'predictions';outdir.mkdir(parents=True,exist_ok=True)
    for kind in VARIANTS:
        for seed in p['seeds']:
            model=make_head(kind,p['config'],ctx.project);model.load_state_dict(saved[kind,seed]['state_dict'])
            prob,_=predict(model,features,idx,kind)
            scores=metrics(y,prob);confusion[f'{kind}_{seed}']=scores.pop('confusion_matrix')
            pred=prob.argmax(1);predictions[kind,seed]=pred
            results.append({'variant':kind,'seed':seed,**scores,'checkpoint_sha256':sha(ctx.models/f'{kind}_{seed}.pt')})
            pd.DataFrame({'sample_id':frame.sample_id,'label':y,'label6':frame.label6,
                'prediction':pred,'probability_fake':prob[:,0],'probability_real':prob[:,1],
                'clip_cosine':cosine,'m_text':generated[:,1],'m_image':generated[:,2],
                'valid_references':valid_count,'effective_k':effective}).to_csv(outdir/f'{kind}_{seed}.csv',index=False)
    table=pd.DataFrame(results);table.to_csv(root(ctx)/'model_metrics.csv',index=False)
    summary=[]
    for kind in VARIANTS:
        part=table[table.variant.eq(kind)]
        summary.append({'variant':kind,'n_seeds':len(part),
            **{metric:float(part[metric].mean()) for metric in METRICS},
            **{metric+'_sd':float(part[metric].std(ddof=1)) for metric in METRICS}})
    pd.DataFrame(summary).to_csv(root(ctx)/'model_comparison.csv',index=False)
    comparison=[]
    for a,b in p['secondary_comparisons']:
        if (a,b)==tuple(p['primary_comparison']): continue
        comparison.append((a,b,'secondary exploratory'))
    comparison.append((*p['primary_comparison'],'prespecified primary'))
    stats=[]
    for a,b,role in comparison:
        for seed in p['seeds']:
            res=paired(y,predictions[a,seed],predictions[b,seed],p['bootstrap_repetitions'],p['bootstrap_seed'])
            res.pop('seed',None)
            stats.append({'comparison':f'{b}_minus_{a}','seed':seed,'role':role,
                'positive_class':0,'metric':'Macro-F1','interpretation':'fresh official-test subset; source-disjoint from retained project manifest',
                **res,'bootstrap_seed':p['bootstrap_seed']})
    secondary=[s for s in stats if s['seed']==42 and s['role']!='prespecified primary']
    for item,pv in zip(secondary,holm([s['mcnemar_exact_two_sided_p'] for s in secondary])):
        item['mcnemar_holm_secondary_p']=pv
    write_json(root(ctx)/'paired_comparisons.json',{'primary':'D minus B, seed 42, Macro-F1',
        'statistics':stats,'scope':FRESH_SCOPE,
        'multiplicity':'Primary comparison prespecified; all other comparisons exploratory; no model selection on this set'})
    write_json(root(ctx)/'confusion_matrices.json',{'order':['Fake (0)','Real (1)'],'matrices':confusion})
    write_json(root(ctx)/'evaluation.json',{'status':'completed','scope':FRESH_SCOPE,
        'manifest_sha256':sha(root(ctx)/'manifest.csv'),'vectors_sha256':receipt['sha256'],
        'bank_sha256':sha(ctx.features/'bank.npz'),'protocol_sha256':digest(p),
        'n':len(frame),'class_counts':{str(k):int(v) for k,v in frame.label.value_counts().items()},
        'all_prespecified_checkpoints_verified':True,'models_not_tuned_on_fresh_holdout':True,
        'limitations':['Same dataset/source as training (Fakeddit); disjoint records are not cross-dataset validation.',
            'Official test metadata is used for the new records; this is a fresh subset of the public-test pool, not a new collection.',
            'Image retrieval availability can induce a selection bias; report failures and final sample counts.']})
    return {'status':'completed','samples':len(frame),'path':str(root(ctx)/'evaluation.json'),
        'primary_comparison':'D minus B; seed 42; paired bootstrap on fresh rows'}
