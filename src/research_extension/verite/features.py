"""Pinned original CPU encoders, sequential loading and immutable resumable chunks."""
from __future__ import annotations
import time,gc,os,platform
import numpy as np,pandas as pd
from pathlib import Path
from .core import *
from .audit import decode_hash

def validate_inputs(s,f):
    require(f.available.all(),'Feature extraction requires verified real image bytes for every included record')
    for row in f.drop_duplicates('image_path').itertuples():
        p=(s.images/row.image_path).resolve();require(p.is_relative_to(s.images.resolve()) and p.is_file(),'Missing/unsafe authorized image')
        actual=decode_hash(p);require(actual['sha256']==row.sha256 and actual['pixel_sha256']==row.pixel_sha256,'Image bytes/pixels changed')

def extract(s):
    import torch,transformers
    from PIL import Image,ImageOps
    from transformers import AutoTokenizer,BertModel,CLIPProcessor,CLIPModel
    from torchvision.models import resnet50,ResNet50_Weights
    from ...feature_extractors import BERT,BERT_REVISION,CLIP,CLIP_REVISION,cache_contract
    p=s.frozen();lock_sources(s);f=s.frame();f=f[f.available].reset_index(drop=True);validate_inputs(s,f)
    require(cache_contract(64)==p['encoder_contract'],'Current runtime encoder contract differs from original pinned contract')
    weights=Path(torch.hub.get_dir())/'checkpoints/resnet50-11ad3fa6.pth';require(weights.is_file(),'Original cached ResNet weights not available')
    bound={**s.binding(),'manifest_sha256':sha(s.cache/'manifest.csv'),'encoder_contract':p['encoder_contract'],
      'resnet_weight_sha256':sha(weights),'feature_provenance':'original_pinned_encoders_from_authorized_VERITE_bytes',
      'source_lock_sha256':sha(s.results/'analysis_source_lock.json')}
    destination=s.cache/'features.npz'
    if destination.exists():verify(destination,bound);return read(s.results/'encoder_profile.json')
    torch.set_num_threads(6);torch.use_deterministic_algorithms(True)
    report={'rows':len(f),'threads':6,'batch_size':16,'device':'cpu','processor_environment':os.environ.get('PROCESSOR_IDENTIFIER','unknown'),
      'platform':platform.platform(),'encoders':{},'torch':str(torch.__version__),'transformers':transformers.__version__,'original_preprocessing_contract_verified':True}
    if os.name=='nt':
        import winreg
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,r'HARDWARE\DESCRIPTION\System\CentralProcessor\0') as key:report['cpu_name']=winreg.QueryValueEx(key,'ProcessorNameString')[0].strip()
    full={};all_start=time.perf_counter()
    for kind in ['bert','resnet','clip']:
        parts=[];pending=[]
        for start in range(0,len(f),64):
            rows=f.iloc[start:start+64];path=s.cache/f'chunks/{kind}_{start:06d}.npz';binding={**bound,'row_ids':rows.row_id.tolist()}
            parts.append(path)
            if path.exists():verify(path,binding)
            else:pending.append((start,path,binding))
        load_start=time.perf_counter();model=processor=None
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
        load_seconds=time.perf_counter()-load_start
        token_lengths=[]
        if pending and kind in ['bert','clip']:
            tok=processor if kind=='bert' else processor.tokenizer
            token_lengths=[len(x) for x in tok(f.caption.tolist(),truncation=False)['input_ids']]
        for start,path,binding in pending:
            rows=f.iloc[start:start+64];result={};begin=time.perf_counter()
            for j in range(0,len(rows),16):
                batch=rows.iloc[j:j+16];images=[]
                try:
                    if kind!='bert':
                        for row in batch.itertuples():
                            require(sha(s.images/row.image_path)==row.sha256,'Image changed during encoder extraction')
                            with Image.open(s.images/row.image_path) as source:
                                source.seek(0);images.append(ImageOps.exif_transpose(source).convert('RGB'))
                    with torch.inference_mode():
                        if kind=='bert':
                            result_batch={'text':model(**processor(batch.caption.tolist(),return_tensors='pt',padding=True,truncation=True,max_length=64)).last_hidden_state[:,0,:]}
                        elif kind=='resnet':result_batch={'image':model(torch.stack([processor(im) for im in images]))}
                        else:
                            out=model(**processor(text=batch.caption.tolist(),images=images,return_tensors='pt',padding=True,truncation=True,max_length=77))
                            t=torch.nn.functional.normalize(out.text_embeds,dim=-1);v=torch.nn.functional.normalize(out.image_embeds,dim=-1)
                            similarity=(t*v).sum(-1,keepdim=True)
                            require(torch.allclose(similarity[:,0],torch.nn.functional.cosine_similarity(out.text_embeds,out.image_embeds),atol=1e-6),'Cosine verification failed')
                            result_batch={'clip_text':t,'clip_image':v,'similarity':similarity}
                        for key,value in result_batch.items():
                            array=value.numpy().astype(np.float32);require(np.isfinite(array).all(),'Nonfinite encoder output');result.setdefault(key,[]).append(array)
                finally:
                    for im in images:im.close()
            elapsed=time.perf_counter()-begin;npz(path,binding,**{k:np.concatenate(v) for k,v in result.items()})
            write(str(path)+'.timing.json',{'rows':len(rows),'inference_seconds':elapsed,'kind':kind});print(kind,min(start+64,len(f)),'/',len(f),f'{elapsed:.2f}s',flush=True)
        for path in parts:
            with np.load(path,allow_pickle=False) as a:
                for key in a.files:full.setdefault(key,[]).append(a[key])
        report['encoders'][kind]={'load_seconds':load_seconds,'new_chunks':len(pending),'chunks':len(parts),
          'inference_seconds':sum(read(str(path)+'.timing.json')['inference_seconds'] for path in parts),
          'token_limit':64 if kind=='bert' else (77 if kind=='clip' else None),
          'truncated_caption_rows':int(sum(n>(64 if kind=='bert' else 77) for n in token_lengths)) if token_lengths else None}
        del model,processor;gc.collect()
    arrays={k:np.concatenate(v) for k,v in full.items()};npz(destination,bound,ids=f.row_id.to_numpy(dtype=str),**arrays)
    report['elapsed_seconds']=time.perf_counter()-all_start
    from ..common import peak_memory
    report.update(peak_memory());write(s.results/'encoder_profile.json',report);return report

def load(s):
    s.frozen();verify_sources(s);path=s.cache/'features.npz';receipt=verify(path)
    require(receipt['binding']['protocol_sha256']==s.binding()['protocol_sha256'] and receipt['binding']['manifest_sha256']==sha(s.cache/'manifest.csv'),'Features have wrong input provenance')
    require(receipt['binding']['feature_provenance']=='original_pinned_encoders_from_authorized_VERITE_bytes','Author features cannot substitute for pinned encoder outputs')
    with np.load(path,allow_pickle=False) as a:z={k:a[k] for k in a.files}
    f=s.frame();f=f[f.available].reset_index(drop=True);require(z['ids'].tolist()==f.row_id.tolist(),'Feature IDs/order mismatch')
    dims={'text':768,'image':2048,'clip_text':512,'clip_image':512,'similarity':1}
    for key,d in dims.items():require(z[key].shape==(len(f),d) and np.isfinite(z[key]).all(),'Bad feature shape/dtype')
    return f,z
