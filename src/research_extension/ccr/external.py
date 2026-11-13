"""Official VERITE access receipt and optional authorized-local evaluation.

No gated download, credential handling, terms acceptance, or image redistribution.
"""
from __future__ import annotations
from pathlib import Path
import re,hashlib
import numpy as np
import pandas as pd
import torch
from .core import *
from .fit import verify_frozen

def blocked(s):
    record={'created_utc':now(),'status':'blocked_access','primary_endpoint':'not evaluated','independent_samples_evaluated':0,
       'user_confirmation':'User explicitly confirmed: No approved copy; document the blocker.',
       'official_repository':'https://github.com/stevejpapad/image-text-verification',
       'official_image_release':'https://huggingface.co/datasets/stefpapad/VERITE',
       'public_metadata':'https://zenodo.org/records/10474450',
       'access_conditions':'Official institutional/university Hugging Face email verification; research use only; no redistribution; user liability terms.',
       'metadata_vs_images':'Zenodo public metadata contains URLs, not the full image archive. Original URLs are documented as frequently unavailable. No partial URL subset substitutes for the preregistered full benchmark.',
       'action':'No gated image request/download, no acceptance of terms, no request submitted to authors, no row labels or scores inspected. No external data included in deliverables.',
       'unblocking':'Obtain authorized official images and matching official VERITE.csv locally, then run verite-local once using the frozen policy. Do not tune to its outcomes.',
       'task_caveat':'Truthful vs miscaptioned/out-of-context differs from Fakeddit distant supervision; this is cross-task/domain transfer.'}
    path=s.project/'results/research_extension/ccr/external_access.json'
    if not path.exists():write_json(path,record,immutable=True);seal(path)
    return read_json(path)

def prepare_local(s,directory):
    """Caller must possess authorized original files. Full official population only."""
    verify_frozen(s);root=Path(directory).resolve();metadata=root/'VERITE.csv'
    require(metadata.is_file(),'Authorized folder must contain official VERITE.csv and images/')
    table=pd.read_csv(metadata,keep_default_na=False)
    require({'caption','image_path','label'}<=set(table),'Expected official caption,image_path,label columns')
    labels={'true':1,'miscaptioned':0,'out-of-context':0}
    require(len(table)==1000 and set(table.label)==set(labels),'Full official 1000-row VERITE table required')
    require(table.label.value_counts().to_dict()=={'true':338,'miscaptioned':338,'out-of-context':324},'Unexpected official category counts')
    # Verify the local label/text/path table against the public official GitHub table.
    # This public metadata request does not obtain gated images.
    import requests,io
    url='https://raw.githubusercontent.com/stevejpapad/image-text-verification/master/VERITE/VERITE.csv'
    response=requests.get(url,timeout=45);response.raise_for_status();official=pd.read_csv(io.BytesIO(response.content),keep_default_na=False)
    cols=['caption','image_path','label']
    require(table[cols].astype(str).equals(official[cols].astype(str)),'Local metadata is not the official row order/content; no evaluation performed')
    from PIL import Image,ImageOps
    f=pd.DataFrame({'sample_id':[f'verite_{i:04d}' for i in range(len(table))],'text':table.caption,'label':table.label.map(labels),
        'official_label':table.label,'image_path':table.image_path,'group_id':[re.sub(r'^(true|false)_','',Path(x).stem) for x in table.image_path]})
    filehash=[];pixelhash=[]
    for rel in f.image_path:
        p=(root/rel).resolve();require(p.is_relative_to(root) and p.is_file(),f'Authorized image missing or unsafe path: {rel}')
        filehash.append(sha(p))
        with Image.open(p) as source:
            im=ImageOps.exif_transpose(source).convert('RGB')
            pixelhash.append(hashlib.sha256(f'{im.width}x{im.height}:'.encode()+im.tobytes()).hexdigest());im.close()
    f['sha256']=filehash;f['pixel_sha256']=pixelhash
    # Article, caption, byte, and pixel-linked components, computed without outcomes.
    parent=np.arange(len(f))
    def find(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return i
    for column in ['group_id','text','sha256','pixel_sha256']:
        seen={}
        for i,key in enumerate(f[column]):
            if key in seen:parent[find(i)]=find(seen[key])
            else:seen[key]=i
    f['cluster_id']=[str(find(i)) for i in range(len(f))]
    f['input_hash']=[hashlib.sha256((r.text+'\0'+r.sha256).encode()).hexdigest() for r in f.itertuples()]
    known=manifest(Context(s.project,s.project))
    fresh_path=s.project/'data/features/research_extension/fresh_holdout/manifest.csv'
    if fresh_path.exists():known=pd.concat([known,pd.read_csv(fresh_path,keep_default_na=False)],ignore_index=True)
    require(not set(f.sha256)&set(known.sha256),'External images overlap previously used Fakeddit bytes; independent endpoint halted')
    require(not set(f.pixel_sha256)&set(known.pixel_sha256),'External images overlap previously used Fakeddit pixels; independent endpoint halted')
    clean=lambda values:values.str.replace(r'\s+',' ',regex=True).str.strip().str.casefold()
    require(not set(clean(f.text))&set(clean(known.text)),'External captions overlap previously used Fakeddit; independent endpoint halted')
    dest=s.path('verite_local');dest.mkdir(parents=True,exist_ok=True)
    path=dest/'manifest.csv'
    if not path.exists():save_csv(f,path,{'authorized_root':str(root),'official_metadata_url':url,'metadata_sha256':hashlib.sha256(response.content).hexdigest(),'local_csv_sha256':sha(metadata)})
    else:verify(path)
    return root,f

def extract_local(s,root,f,batch_size=16):
    from PIL import Image,ImageOps
    from transformers import AutoTokenizer,BertModel,CLIPProcessor,CLIPModel
    from torchvision.models import resnet50,ResNet50_Weights
    from torch.nn import functional as F
    from ...feature_extractors import BERT,BERT_REVISION
    from ..common import CLIP_ID,CLIP_REVISION
    path=s.path('verite_local/features.npz')
    if path.exists():
        verify(path);a=load_npz(path);require(a['ids'].tolist()==f.sample_id.tolist(),'External feature order changed')
        return {k:a[k] for k in ['text','image','similarity']}
    torch.set_num_threads(6);torch.use_deterministic_algorithms(True)
    tok=AutoTokenizer.from_pretrained(BERT,revision=BERT_REVISION,local_files_only=True)
    bert=BertModel.from_pretrained(BERT,revision=BERT_REVISION,local_files_only=True).eval().requires_grad_(False)
    proc=CLIPProcessor.from_pretrained(CLIP_ID,revision=CLIP_REVISION,use_fast=False,local_files_only=True)
    clip=CLIPModel.from_pretrained(CLIP_ID,revision=CLIP_REVISION,local_files_only=True).eval().requires_grad_(False)
    w=ResNet50_Weights.IMAGENET1K_V2;weights=Path(torch.hub.get_dir())/'checkpoints'/'resnet50-11ad3fa6.pth'
    require(weights.exists(),'Cached ResNet weights missing');resnet=resnet50(weights=None)
    resnet.load_state_dict(torch.load(weights,map_location='cpu',weights_only=True));resnet.fc=torch.nn.Identity();resnet.eval().requires_grad_(False)
    result={k:[] for k in ['text','image','similarity']}
    for start in range(0,len(f),batch_size):
        rows=f.iloc[start:start+batch_size];images=[]
        try:
            for r in rows.itertuples():
                require(sha(root/r.image_path)==r.sha256,'Authorized external image changed')
                with Image.open(root/r.image_path) as source:images.append(ImageOps.exif_transpose(source).convert('RGB'))
            with torch.inference_mode():
                txt=bert(**tok(rows.text.tolist(),return_tensors='pt',padding=True,truncation=True,max_length=64)).last_hidden_state[:,0,:]
                vis=resnet(torch.stack([w.transforms()(im) for im in images]))
                output=clip(**proc(text=rows.text.tolist(),images=images,return_tensors='pt',padding=True,truncation=True,max_length=77))
                cosine=(F.normalize(output.text_embeds,dim=-1)*F.normalize(output.image_embeds,dim=-1)).sum(-1,keepdim=True)
            for k,v in zip(result,[txt,vis,cosine]):result[k].append(v.numpy().astype(np.float32))
        finally:
            for im in images:im.close()
        print(f'Authorized VERITE extraction {min(start+batch_size,len(f))}/{len(f)}',flush=True)
    raw={k:np.concatenate(v) for k,v in result.items()}
    write_npz(path,ids=f.sample_id.to_numpy(dtype=str),**raw);seal(path,{'manifest':sha(s.path('verite_local/manifest.csv')),
           'encoder_contract':read_json(s.project/'data/features/extraction.json')['contract']})
    return raw

def evaluate_local(s,directory):
    require(not s.smoke,'External evaluation unavailable in smoke mode');root,f=prepare_local(s,directory)
    raw=extract_local(s,root,f)
    from .evaluate import save_predictions,analyze
    save_predictions(s,'verite',f,raw,{'manifest':sha(s.path('verite_local/manifest.csv')),'features':sha(s.path('verite_local/features.npz'))})
    return analyze(s,'verite')
