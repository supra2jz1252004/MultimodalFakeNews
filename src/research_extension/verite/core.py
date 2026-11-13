"""Additive VERITE evaluation: immutable contracts, no training or old-file writes."""
from __future__ import annotations
from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
from ...external_eval.newsclippings.io_utils import require,sha,digest,now,read,write,put,csv,npz,seal,verify
SEEDS=[42,123,2026]
KINDS=['A','B','C','D','CCR']
SCORES=['cosine','m_text','m_image','bras_mean']
LABELS={'true':1,'miscaptioned':0,'out-of-context':0}
class Study:
    def __init__(self,project):
        self.root=Path(project).resolve();self.data=self.root/'data/external/verite'
        self.results=self.root/'results/research_extension/verite';self.docs=self.root/'docs/research_extension/verite'
        self.cache=self.data/'restricted_cache';self.images=self.data/'authorized_extracted'
        for p in [self.results,self.docs,self.cache]:p.mkdir(parents=True,exist_ok=True)
    def protocol(self):
        p=self.results/'protocol.json';verify(p);return read(p)
    def binding(self):return {'protocol_sha256':sha(self.results/'protocol.json')}
    def frozen(self):
        p=self.protocol()
        for rel,h in p['frozen_assets'].items():require(sha(self.root/rel)==h,'Frozen asset changed: '+rel)
        require(sha(self.data/'images.zip')==p['images_zip_sha256'],'Authorized archive changed')
        require(sha(self.data/'VERITE.csv')==p['annotations_sha256'],'Official annotations changed')
        return p
    def frame(self):
        path=self.cache/'manifest.csv';verify(path)
        return pd.read_csv(path,keep_default_na=False,dtype={'row_id':str,'family_id':str,'cluster_id':str})
def canonical_text(s):return ' '.join(str(s).casefold().split())
def clusters(frame):
    """Union article families, shared normalized captions and decoded/byte image identity."""
    ids=frame.row_id.tolist();parent={x:x for x in ids};seen={}
    def find(x):
        while parent[x]!=x:parent[x]=parent[parent[x]];x=parent[x]
        return x
    for row in frame.to_dict('records'):
        for k in ['family_id','article_url','text_key','image_path','sha256','pixel_sha256']:
            value=str(row.get(k,''))
            if not value:continue
            key=(k,value)
            if key in seen:
                a,b=find(row['row_id']),find(seen[key]);parent[max(a,b)]=min(a,b)
            else:seen[key]=row['row_id']
    return frame.row_id.map(find)
def scientific_sources(s):
    return {p.relative_to(s.root).as_posix():sha(p) for p in sorted((s.root/'src/research_extension/verite').glob('*.py')) if p.name not in ['report.py','cli.py','tests.py','__init__.py']}
def lock_sources(s):
    obj={'protocol_sha256':sha(s.results/'protocol.json'),'sources':scientific_sources(s)}
    write(s.results/'analysis_source_lock.json',obj);return obj
def verify_sources(s):
    require(read(s.results/'analysis_source_lock.json')['sources']==scientific_sources(s),'Analysis source changed after lock')
def preservation(s):
    inv=read(s.results/'historical_inventory.json');bad=[]
    for rel,record in inv['files'].items():
        p=s.root/rel
        if not p.is_file() or p.stat().st_size!=record['bytes'] or sha(p)!=record['sha256']:bad.append(rel)
    result={'passed':not bad,'checked_files':len(inv['files']),'changed_or_missing':bad,'inventory_sha256':sha(s.results/'historical_inventory.json')}
    write(s.results/'preservation_check.json',result);require(not bad,'Historical files changed');return result
