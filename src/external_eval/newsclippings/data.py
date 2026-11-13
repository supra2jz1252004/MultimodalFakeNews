"""Official pair semantics, deterministic selection and content-linked grouping."""
from __future__ import annotations
import hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd
from .io_utils import *
STRATEGIES=['person_sbert_text_text','scene_resnet_place','semantics_clip_text_image','semantics_clip_text_text']
SEED=20261108

def parse_pairs(obj,split='test',strategy=None):
    require(split in ('test','val','train'),'Unknown official split')
    a=obj['annotations'];require(len(a)%2==0,'Odd annotation count');rows=[];seen=set()
    for j in range(0,len(a),2):
        p,n=a[j:j+2]
        require(p['falsified'] in (False,0) and n['falsified'] in (True,1),'Pristine/mismatch order invalid')
        require(p['id']==n['id'] and p['id']==p['image_id'] and n['id']!=n['image_id'],'Invalid shared-caption pair')
        st=strategy or obj['source_datasets'][str(p['source_dataset'])]
        require(strategy or p['source_dataset']==n['source_dataset'],'Mixed construction strategy in pair')
        group=f"{st}:{p['id']}:{n['image_id']}";require(group not in seen,'Duplicate annotated pair');seen.add(group)
        for k,r in enumerate([p,n]):
            rows.append({'row_id':f'{split}:{j+k}','pair_id':group,'caption_id':str(r['id']),'image_id':str(r['image_id']),
              'falsified':int(r['falsified']),'strategy':st,'split':split,'official_row':j+k,
              'generation_similarity':r.get('similarity_score','')})
    f=pd.DataFrame(rows);require(f.row_id.is_unique,'Duplicate row IDs');return f
def select(f,n,seed=SEED):
    """Equal strategy allocation; fixed before asset availability or scores, no replacement."""
    require(n>0 and n%len(STRATEGIES)==0,'Pair budget must divide evenly across four strategies')
    ids=[]
    for st in STRATEGIES:
        candidates=sorted(f.loc[f.strategy.eq(st),'pair_id'].unique(),key=lambda x:hashlib.sha256(f'{seed}\0{x}'.encode()).hexdigest())
        ids.extend(candidates[:n//len(STRATEGIES)])
    order={x:i for i,x in enumerate(ids)}
    out=f[f.pair_id.isin(ids)].copy();out['selection_order']=out.pair_id.map(order)
    return out.sort_values(['selection_order','falsified']).reset_index(drop=True)
def linked_clusters(f):
    """Connect caption/image item IDs across roles and exact text/byte/pixel identities."""
    pairs=sorted(f.pair_id.unique());parent={p:p for p in pairs};seen={}
    def find(x):
        while parent[x]!=x:parent[x]=parent[parent[x]];x=parent[x]
        return x
    def link(p,key):
        if key in seen:
            a,b=find(p),find(seen[key]);parent[max(a,b)]=min(a,b)
        else:seen[key]=p
    for r in f.to_dict('records'):
        for col in ['caption_id','image_id']:link(r['pair_id'],('visualnews_item',str(r[col])))
        for col in ['text_key','sha256','pixel_sha256','article_key','image_article_key']:
            value=str(r.get(col,''))
            if value and value!='unknown':link(r['pair_id'],('article' if col.endswith('article_key') else col,value))
    return f.pair_id.map(lambda p:find(p))
def join_metadata(f,lookup):
    out=f.copy();records=[]
    for r in out.to_dict('records'):
        c=lookup.get(str(r['caption_id']),{});im=lookup.get(str(r['image_id']),{})
        records.append({'caption':c.get('caption',''),'filename':im.get('image_path',''),
          'source':c.get('source','unknown'),'image_source':im.get('source','unknown'),
          'metadata_join':bool(c and im),'article_key':c.get('full_article_path',c.get('article_url',c.get('url',''))),
          'image_article_key':im.get('full_article_path',im.get('article_url',im.get('url','')))})
    extra=pd.DataFrame(records);out=pd.concat([out.reset_index(drop=True),extra],axis=1)
    out['text_key']=out.caption.str.replace(r'\s+',' ',regex=True).str.strip().str.casefold()
    return out
def validate_images(f,root):
    from PIL import Image,ImageOps
    root=Path(root).resolve();out=f.copy();records=[]
    for row in out.itertuples():
        record={'acquisition_status':'missing','sha256':'','pixel_sha256':'','width':0,'height':0}
        p=(root/row.filename).resolve()
        try:
            require(bool(row.filename) and p.is_relative_to(root),'Missing/unsafe image path')
            if p.is_file():
                with Image.open(p) as im:
                    im.seek(0);rgb=ImageOps.exif_transpose(im).convert('RGB');rgb.load()
                    record.update(acquisition_status='verified',sha256=sha(p),pixel_sha256=hashlib.sha256(f'{rgb.width}x{rgb.height}:'.encode()+rgb.tobytes()).hexdigest(),width=rgb.width,height=rgb.height)
                    rgb.close()
        except Exception as e:record['acquisition_status']='invalid: '+type(e).__name__
        records.append(record)
    for k in records[0]:out[k]=[r[k] for r in records]
    out['cluster_id']=linked_clusters(out);return out
def require_images(f,root):
    require({'acquisition_status','sha256','caption','filename'}<=set(f),'Full-image manifest absent')
    require(f.acquisition_status.eq('verified').all() and f.caption.str.len().gt(0).all(),'Full-image inference requires decoded real images and verified captions')
    for r in f.itertuples():
        p=(Path(root)/r.filename).resolve();require(p.is_relative_to(Path(root).resolve()) and sha(p)==r.sha256,'Image changed/missing')

def prepare(run):
    access=read(run.results/'data_access.json');raw=run.data/'data/merged_balanced/test.json'
    full=parse_pairs(read(raw));selected=select(full,1000);smoke=select(full,100)
    # Metadata-only selection: no scores or image availability have been inspected.
    selected['source']='unknown';selected['image_source']='unknown';selected['filename']='';selected['sha256']='';selected['pixel_sha256']=''
    selected['acquisition_status']='not_acquired';selected['cluster_id']=linked_clusters(selected)
    selected['smoke']=selected.pair_id.isin(smoke.pair_id);csv(run.results/'selection_manifest.csv',selected)
    csv(run.results/'smoke_manifest.csv',selected[selected.smoke])
    val=parse_pairs(read(run.data/'data/merged_balanced/val.json'),'val')
    audit={'official_rows':len(full),'official_pairs':full.pair_id.nunique(),'selected_pairs':selected.pair_id.nunique(),'smoke_pairs':smoke.pair_id.nunique(),
      'population_pairs_by_strategy':full[full.falsified.eq(0)].strategy.value_counts().to_dict(),
      'selected_pairs_by_strategy':selected[selected.falsified.eq(0)].strategy.value_counts().to_dict(),
      'shared_caption_pair_contract_verified':True,'all_pristine_ids_equal_image_ids':True,
      'test_val_shared_item_ids':len((set(full.caption_id)|set(full.image_id))&(set(val.caption_id)|set(val.image_id))),
      'test_train_overlap':'not checked: full train annotations not required/downloaded; official split designation retained',
      'selected_item_components':selected.cluster_id.nunique(),'unknown_events':'No claim of complete event/publisher independence',
      'strategy_file_membership':{}}
    for st in STRATEGIES:
        other=parse_pairs(read(run.data/f'data/{st}/test.json'),strategy=st)
        lookup=set(zip(other.caption_id,other.image_id,other.falsified))
        sub=full[full.strategy.eq(st)]
        audit['strategy_file_membership'][st]={'rows':len(other),'merged_rows':len(sub),'merged_missing':sum(tuple(x) not in lookup for x in sub[['caption_id','image_id','falsified']].to_numpy())}
        require(audit['strategy_file_membership'][st]['merged_missing']==0,'Merged/strategy membership mismatch')
    write(run.results/'selection_audit.json',audit)
    frozen={}
    for glob in ['models/final/*.pt','models/normalization.npz','models/research_extension/bras_b4096_k16_s20260916/[CD]_*.pt','models/research_extension/ccr/full/policy_*.json','models/research_extension/ccr/full/gate_*.joblib','data/features/extraction.json','data/features/research_extension/bras_b4096_k16_s20260916/bank.npz','data/features/research_extension/bras_b4096_k16_s20260916/bank.json','src/models.py','src/research_extension/reference.py']:
        for p in run.root.glob(glob):frozen[p.relative_to(run.root).as_posix()]=sha(p)
    protocol={'schema':1,'locked_before_test_scores':True,'created_utc':now(),'hf_revision':access['revision'],'task':'image-caption correspondence; not factual veracity',
      'selection_seed':SEED,'planned_pairs':1000,'smoke_pairs':100,'strategy_design':'250 pairs each of four strategies; equal-strategy estimand, not population-weighted',
      'missing_policy':'No replacements. Complete groups only for each supported endpoint; disclose every excluded group and strategy/source denominators.',
      'tie_handling':'exact floating point tie receives 0.5; strict and 1e-6-tolerance sensitivity also reported',
      'positive_class':'pristine (falsified=0), higher score means better correspondence',
      'scores':['cosine','m_text','m_image','bras_mean'],'bras_mean':'(m_text+m_image)/2; k=16, original train-only 4096 Fakeddit bank',
      'primary_comparison':'bras_mean minus cosine within-group ranking half-credit; two-sided cluster sign-flip test',
      'secondary_comparisons':['m_text minus cosine','m_image minus cosine'],'multiplicity':'Holm across all secondary score comparisons and construction-strategy ranking tests within each feature provenance; transfer seeds Holm separately',
      'bootstrap':{'repetitions':5000,'seed':20261108,'unit':'connected components of shared caption/image item IDs, known article, normalized caption and image byte/pixel hashes; same resample for all scores','ci':'percentile 95%'},
      'permutation':{'repetitions':10000,'seed':20261109,'unit':'cluster; two-sided paired sign flips of summed group differences; +1 Monte Carlo correction'},
      'robustness':['strict versus half-credit ties','1e-6 tolerance ties','leave-one-strategy-out descriptive','pair-only versus linked-component bootstrap','no-image-reuse subset descriptive'],
      'hard_cases':'10 smallest and 10 largest official cosine paired margins, deterministic pair-ID tie break; exploratory, no factual labels or image redistribution',
      'official_embeddings':'Separate score-only provenance baseline. Never substitute for pinned model outputs or mix with pinned BRAS bank without exact contract verification.',
      'full_image_endpoint':'Frozen A/B three seeds 42,123,2026; unchanged training normalizers/argmax; probability and decision shifts; pairing-AUROC expressly cross-task proxy; CCR descriptive B identity only when guard is false.',
      'no_external_training':True,'cpu_threads':6,'batch_size':16,'metadata_sha256':sha(raw),'selection_sha256':sha(run.results/'selection_manifest.csv'),
      'blueprint_sha256':sha(run.root/'docs/Path_B_NewsCLIPpings_Codex_Blueprint.md'),'historical_inventory_sha256':sha(run.results/'historical_inventory.json'),'frozen_assets':frozen}
    path=run.results/'protocol.json'
    if path.exists():
        prior=read(path);protocol['created_utc']=prior['created_utc']
    write(path,protocol);seal(path,{'selection_sha256':protocol['selection_sha256'],'metadata_sha256':protocol['metadata_sha256']})
    seal(run.results/'selection_manifest.csv',{'protocol_sha256':sha(path)})
    return audit
