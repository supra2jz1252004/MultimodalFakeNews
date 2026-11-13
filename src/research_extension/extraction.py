"""Resumable extraction of missing CLIP vectors only; local pinned weights."""
from __future__ import annotations
import time
import numpy as np
from .common import (CLIP_ID,CLIP_REVISION,require,sha,digest,read_json,write_json,
    write_npz,load_npz,manifest,load_historical,peak_memory)
from .reference import check_embeddings

def contract(ctx):
    import torch
    import transformers
    return {'schema':1,'model':CLIP_ID,'revision':CLIP_REVISION,'dimension':512,
        'dtype':'float32','normalized':True,'max_length':77,'use_fast':False,
        'decode':'first frame; EXIF transpose; RGB','torch':str(torch.__version__),
        'transformers':transformers.__version__,
        'manifest_sha256':sha(ctx.project/'data/processed/manifest.csv')}

def directory(ctx):
    return ctx.cache/'clip'/digest(contract(ctx))[:16]

def index_cache(ctx):
    directory_path=directory(ctx)
    index={}
    for path in sorted(directory_path.glob('chunk_*.npz')):
        receipt=path.with_suffix('.json')
        # A committed chunk always has a hash receipt. An interrupted write is not reusable.
        require(receipt.exists(),f'Missing receipt for {path}; quarantine incomplete chunk before resuming')
        require(read_json(receipt)['sha256']==sha(path),f'Corrupt CLIP chunk: {path}')
        a=load_npz(path)
        require(a['text'].shape==a['image'].shape==(len(a['ids']),512),'Bad CLIP dimensions')
        check_embeddings(a['text'],'cached CLIP text'); check_embeddings(a['image'],'cached CLIP image')
        for i,sid in enumerate(a['ids']):
            sid=str(sid)
            require(sid not in index,f'Duplicate CLIP ID: {sid}')
            index[sid]=(path,i,str(a['input_hash'][i]),str(a['splits'][i]))
    return index

def load_embeddings(ctx, frame):
    index=index_cache(ctx)
    missing=[sid for sid in frame.sample_id if sid not in index]
    require(not missing,f'{len(missing)} CLIP embeddings missing; first IDs {missing[:5]}. Run extract-clip (resumable).')
    t=np.empty((len(frame),512),np.float32); v=np.empty_like(t)
    needed={}
    for i,r in enumerate(frame.itertuples()):
        path,j,h,s=index[r.sample_id]
        require(h==r.input_hash and s==r.split,f'CLIP input hash/split mismatch: {r.sample_id}')
        needed.setdefault(path,[]).append((i,j))
    for path,positions in needed.items():
        a=load_npz(path)
        ii,jj=zip(*positions); t[list(ii)]=a['text'][list(jj)]; v[list(ii)]=a['image'][list(jj)]
    return t,v

def extract(ctx,max_new=None,batch_size=16):
    import torch
    from torch.nn import functional as F
    from PIL import Image,ImageOps
    from transformers import CLIPModel,CLIPProcessor
    require(batch_size>0 and (max_new is None or max_new>0),'Extraction sizes must be positive')
    torch.set_num_threads(ctx.threads)
    torch.use_deterministic_algorithms(True)
    frame=manifest(ctx)
    c=contract(ctx); dest=directory(ctx)
    write_json(dest/'contract.json',c,immutable=True)
    index=index_cache(ctx)
    for r in frame.itertuples():
        if r.sample_id in index:
            require(index[r.sample_id][2:]==(r.input_hash,r.split),f'Stale CLIP entry {r.sample_id}')
    pending=frame[~frame.sample_id.isin(index)]
    n_missing=len(pending)
    if max_new is not None: pending=pending.iloc[:max_new]
    print(f'CLIP: {len(frame)-n_missing} verified, {n_missing} pending; extracting {len(pending)}',flush=True)
    started=time.perf_counter()
    if len(pending):
        try:
            processor=CLIPProcessor.from_pretrained(CLIP_ID,revision=CLIP_REVISION,use_fast=False,local_files_only=True)
            model=CLIPModel.from_pretrained(CLIP_ID,revision=CLIP_REVISION,local_files_only=True).eval().requires_grad_(False)
        except Exception as exc:
            raise RuntimeError(f'Pinned CLIP model/processor unavailable locally; no download attempted: {exc}') from exc
        old=load_historical(ctx,pending,('similarity',))['similarity'][:,0]
        next_chunk=max([int(p.stem.split('_')[1]) for p in dest.glob('chunk_*.npz')],default=-1)+1
        for start in range(0,len(pending),128):
            rows=pending.iloc[start:start+128]
            all_t=[]; all_v=[]
            for b in range(0,len(rows),batch_size):
                batch=rows.iloc[b:b+batch_size]; images=[]
                try:
                    for r in batch.itertuples():
                        path=ctx.project/r.image_path
                        require(path.is_file() and sha(path)==r.sha256,f'Missing/changed image {r.sample_id}')
                        with Image.open(path) as source:
                            source.seek(0); images.append(ImageOps.exif_transpose(source).convert('RGB'))
                    inputs=processor(text=batch.text.tolist(),images=images,return_tensors='pt',padding=True,truncation=True,max_length=77)
                    with torch.inference_mode():
                        output=model(**inputs)
                        t=F.normalize(output.text_embeds,dim=-1).cpu().numpy().astype(np.float32)
                        v=F.normalize(output.image_embeds,dim=-1).cpu().numpy().astype(np.float32)
                    all_t.append(t); all_v.append(v)
                finally:
                    for im in images: im.close()
            text=np.concatenate(all_t); image=np.concatenate(all_v)
            cosine=np.einsum('ij,ij->i',text,image)
            delta=float(np.max(np.abs(cosine-old[start:start+len(rows)])))
            require(delta<=2e-5,f'Re-extracted cosine differs from historical scalar by {delta}; investigate encoder drift')
            path=dest/f'chunk_{next_chunk:06d}.npz'
            write_npz(path,ids=rows.sample_id.to_numpy(dtype=str),input_hash=rows.input_hash.to_numpy(dtype=str),
                      splits=rows.split.to_numpy(dtype=str),text=text,image=image)
            write_json(path.with_suffix('.json'),{'sha256':sha(path),'rows':len(rows),'cosine_max_abs_error':delta,
                'contract_sha256':digest(c),'batch_size':batch_size,'threads':ctx.threads,
                'elapsed_seconds_this_invocation':time.perf_counter()-started},immutable=True)
            next_chunk+=1
            done=start+len(rows); elapsed=time.perf_counter()-started
            write_json(dest/'progress.json',{'verified_before':len(frame)-n_missing,'new_completed':done,
                'remaining_in_scope':n_missing-done,'scope':'smoke' if ctx.smoke else 'full','seconds':elapsed,**peak_memory()})
            print(f'CLIP {done}/{len(pending)} new; {elapsed:.1f}s; {done/max(elapsed,1e-6):.2f} pairs/s',flush=True)
    result={'scope':'smoke validation pipeline' if ctx.smoke else 'full retained manifest',
        'reused':len(frame)-n_missing,'extracted':len(pending),'remaining':n_missing-len(pending),
        'seconds':time.perf_counter()-started,'contract_sha256':digest(c),**peak_memory()}
    log=dest/'extractions.jsonl'
    import json
    with log.open('a',encoding='utf-8') as f: f.write(json.dumps(result)+'\n')
    return result
