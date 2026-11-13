"""Immutable bank manifests, generated features and explicit randomized controls."""
from __future__ import annotations
import hashlib
import time
import numpy as np
import pandas as pd
from .common import (manifest,lock_protocol,require,digest,sha,write_json,read_json,
                     write_npz,load_npz,peak_memory,load_historical)
from .extraction import load_embeddings,contract
from .reference import select_bank,margins

def build_bank(ctx):
    frame=manifest(ctx); p=lock_protocol(ctx)
    idx=select_bank(frame,ctx.bank_size,ctx.seed)
    bank=frame.iloc[idx].reset_index(drop=True)
    t,v=load_embeddings(ctx,bank)
    metadata={'protocol_sha256':digest(p),'clip_contract_sha256':digest(contract(ctx)),
              'ids':bank.sample_id.tolist(),'input_hashes':bank.input_hash.tolist(),
              'selection':'SHA256(seed NUL ID) sorted, unique known family/text/exact image/pair; train only',
              'bank_size':len(bank),'seed':ctx.seed}
    path=ctx.features/'bank.npz'
    if path.exists():
        receipt=read_json(ctx.features/'bank.json')
        require(receipt['metadata']==metadata and receipt['sha256']==sha(path),'Existing reference bank changed')
        return {'status':'verified existing bank','sha256':receipt['sha256'],
                'bank_size':metadata['bank_size'],'protocol_sha256':metadata['protocol_sha256']}
    write_npz(path,text=t,image=v,**{key:bank[key].to_numpy(dtype=str) for key in ['sample_id','input_hash','split','group_id','text_key','sha256','pixel_sha256']})
    receipt={'metadata':metadata,'sha256':sha(path)}
    write_json(ctx.features/'bank.json',receipt,immutable=True)
    bank[['sample_id','split','group_id','input_hash','sha256','pixel_sha256']].to_csv(ctx.features/'reference_ids.csv',index=False)
    return {'status':'created','sha256':receipt['sha256'],
            'bank_size':metadata['bank_size'],'protocol_sha256':metadata['protocol_sha256']}

def load_bank(ctx):
    path=ctx.features/'bank.npz'; r=read_json(ctx.features/'bank.json')
    require(r['sha256']==sha(path),'Reference-bank hash mismatch')
    require(r['metadata']['protocol_sha256']==digest(lock_protocol(ctx)), 'Reference-bank protocol differs')
    a=load_npz(path)
    bank=pd.DataFrame({k:v for k,v in a.items() if k not in ['text','image']})
    require(bank.split.eq('train').all(),'Reference bank contains evaluation samples')
    return bank,a['text'],a['image'],r

def random_control(ids, seed):
    result=[]
    for sid in ids:
        key=int.from_bytes(hashlib.sha256(f'E\0{seed}\0{sid}'.encode()).digest()[:8],'little')
        result.append(np.random.default_rng(key).standard_normal(2))
    return np.asarray(result,dtype=np.float32)

def unimodal_control(text,image,seed):
    rng=np.random.default_rng(seed)
    wt=rng.standard_normal(text.shape[1]).astype(np.float32); wt/=np.linalg.norm(wt)
    wi=rng.standard_normal(image.shape[1]).astype(np.float32); wi/=np.linalg.norm(wi)
    return np.column_stack([text@wt,image@wi]).astype(np.float32)

def generate(ctx):
    start=time.perf_counter(); frame=manifest(ctx); p=lock_protocol(ctx)
    bank,bt,bv,receipt=load_bank(ctx)
    path=ctx.features/'features.npz'; record=ctx.features/'features.json'
    config={'protocol_sha256':digest(p),'bank_sha256':receipt['sha256'],
            'ids':frame.sample_id.tolist(),'input_hashes':frame.input_hash.tolist()}
    if path.exists():
        r=read_json(record)
        require(r['config']==config and r['sha256']==sha(path),'Generated features changed or configuration differs')
        return {'status':'verified existing generated features','sha256':r['sha256'],
                'shape':r['shape'],'min_valid_candidates':r['min_valid_candidates'],
                'seconds':r['seconds']}
    t,v=load_embeddings(ctx,frame)
    x,n,k=margins(t,v,frame,bt,bv,bank,ctx.top_k,ctx.batch_size)
    old=load_historical(ctx,frame,('similarity',))['similarity']
    require(np.allclose(x[:,:1],old,atol=2e-5,rtol=0),'Cached cosine mismatch')
    # Retain the exact historical scalar in all variants. Margins use the measured paired dot.
    x[:,0]=old[:,0]
    require(np.isfinite(x).all(),'No valid reference candidates; abort training rather than fabricate margins')
    write_npz(path,ids=frame.sample_id.to_numpy(dtype=str),bras=x,valid_count=n,effective_k=k,
        random=random_control(frame.sample_id,ctx.seed),unimodal=unimodal_control(t,v,ctx.seed))
    r={'config':config,'sha256':sha(path),'seconds':time.perf_counter()-start,
       'shape':list(x.shape),'min_valid_candidates':int(n.min()),'batch_size':ctx.batch_size,**peak_memory()}
    write_json(record,r,immutable=True)
    pd.DataFrame({'sample_id':frame.sample_id,'split':frame.split,'label':frame.label,'label6':frame.label6,
        'cosine':x[:,0],'m_text':x[:,1],'m_image':x[:,2],'valid_count':n,'effective_k':k}).to_csv(ctx.features/'features.csv',index=False)
    return {'status':'created','sha256':r['sha256'],'shape':r['shape'],
            'min_valid_candidates':r['min_valid_candidates'],'seconds':r['seconds'],
            'batch_size':r['batch_size']}

def load_generated(ctx,frame):
    r=read_json(ctx.features/'features.json'); path=ctx.features/'features.npz'
    require(r['sha256']==sha(path),'Feature hash mismatch')
    require(r['config']['protocol_sha256']==digest(lock_protocol(ctx)),'Features belong to another protocol')
    _,_,_,bank=load_bank(ctx)
    require(r['config']['bank_sha256']==bank['sha256'],'Features belong to another bank')
    a=load_npz(path)
    require(a['ids'].tolist()==frame.sample_id.tolist() and r['config']['input_hashes']==frame.input_hash.tolist(),'Feature sample ordering/input hash mismatch')
    return a,r
