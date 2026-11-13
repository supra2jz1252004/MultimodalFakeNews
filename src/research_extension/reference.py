"""Training-only reference sampling and bidirectional float32 top-k margins."""
from __future__ import annotations
import numpy as np
from .common import require, stable_rank

EXCLUDE_KEYS=('sample_id','group_id','input_hash','text_key','sha256','pixel_sha256')

def select_bank(frame, size=4096, seed=20260916):
    require(size>0, 'bank size must be positive')
    train=frame[frame.split.eq('train')]
    require(len(train)>0, 'Reference bank requires training samples')
    # Sorted hash order is invariant to the manifest's incidental row order.
    order=stable_rank(train.sample_id.tolist(),seed)
    used={k:set() for k in EXCLUDE_KEYS}
    chosen=[]
    for position in order:
        row=train.iloc[position]
        if any(str(row[k]) and str(row[k]) in used[k] for k in EXCLUDE_KEYS if k in train): continue
        chosen.append(train.index[position])
        for k in EXCLUDE_KEYS:
            if k in train: used[k].add(str(row[k]))
        if len(chosen)==size: break
    require(len(chosen)==size, f'Only {len(chosen)} independent training references for requested bank size {size}')
    return np.asarray(chosen,dtype=np.int64)

def valid_candidates(query, bank):
    valid=np.ones((len(query),len(bank)),dtype=bool)
    for key in EXCLUDE_KEYS:
        if key not in query or key not in bank: continue
        q=query[key].astype(str).to_numpy()[:,None]
        b=bank[key].astype(str).to_numpy()[None,:]
        valid &= ~((q==b)&(q!='')&(b!=''))
    return valid

def check_embeddings(x, name):
    require(x.ndim==2 and x.dtype==np.float32 and np.isfinite(x).all(), f'{name}: expected finite float32 matrix')
    require(np.allclose(np.linalg.norm(x,axis=1),1.,atol=2e-5), f'{name}: embeddings must be L2-normalized')

def top_mean(scores, valid, k):
    require(k>0,'top-k must be positive')
    counts=valid.sum(1)
    effective=np.minimum(counts,k)
    out=np.full(len(scores),np.nan,dtype=np.float32)
    if scores.shape[1]:
        values=np.where(valid,scores,-np.inf)
        width=min(k,scores.shape[1])
        top=np.partition(values,values.shape[1]-width,axis=1)[:,-width:]
        total=np.where(np.isfinite(top),top,0).sum(1,dtype=np.float32)
        np.divide(total,effective,out=out,where=effective>0)
    return out,counts,effective

def margins(text,image,query,bank_text,bank_image,bank,k=16,batch_size=64,extra_exclusions=None):
    require(len(text)==len(image)==len(query), 'Query length mismatch')
    require(len(bank_text)==len(bank_image)==len(bank), 'Bank length mismatch')
    require(batch_size>0 and k>0, 'Positive batch size and k required')
    require(bank.split.eq('train').all(), 'Non-training reference entry')
    for x,name in [(text,'text'),(image,'image'),(bank_text,'bank text'),(bank_image,'bank image')]: check_embeddings(x,name)
    require(text.shape[1]==image.shape[1]==bank_text.shape[1]==bank_image.shape[1], 'Embedding dimensions differ')
    absolute=np.einsum('ij,ij->i',text,image,dtype=np.float32)
    result=np.empty((len(text),3),dtype=np.float32); result[:,0]=absolute
    counts=np.empty(len(text),np.int64); effective=np.empty(len(text),np.int64)
    for start in range(0,len(text),batch_size):
        end=min(start+batch_size,len(text)); sl=slice(start,end)
        valid=valid_candidates(query.iloc[sl],bank)
        if extra_exclusions is not None:
            require(len(extra_exclusions)==len(query),'Extra exclusion rows mismatch')
            valid &= valid_candidates(extra_exclusions.iloc[sl],bank)
        mt,n,e=top_mean(text[sl]@bank_image.T,valid,k)
        mi,_,_=top_mean(image[sl]@bank_text.T,valid,k)
        result[sl,1]=absolute[sl]-mt
        result[sl,2]=absolute[sl]-mi
        counts[sl]=n; effective[sl]=e
    return result,counts,effective
