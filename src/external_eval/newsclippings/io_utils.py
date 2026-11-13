"""Immutable, hash-linked artifacts; bounded official downloads, no implicit bulk fetch."""
from __future__ import annotations
import hashlib,json,time,platform,sys
from pathlib import Path
from datetime import datetime,timezone
import numpy as np

def require(ok,message):
    if not ok: raise ValueError(message)
def now(): return datetime.now(timezone.utc).isoformat()
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''): h.update(b)
    return h.hexdigest()
def digest(obj): return hashlib.sha256(json.dumps(obj,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))
def put(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists(): require(path.read_bytes()==data,f'Immutable artifact differs: {path}');return
    temp=path.with_suffix(path.suffix+'.part');temp.write_bytes(data);temp.replace(path)
def write(path,obj): put(path,(json.dumps(obj,indent=2,sort_keys=True,allow_nan=False)+'\n').encode())
def csv(path,frame): put(path,frame.to_csv(index=False,lineterminator='\n',float_format='%.12g').encode())
def seal(path,binding):
    write(str(path)+'.receipt.json',{'sha256':sha(path),'binding':binding})
def verify(path,binding=None):
    r=read(str(path)+'.receipt.json');require(sha(path)==r['sha256'],f'Hash mismatch: {path}')
    if binding is not None:require(r['binding']==binding,f'Cache input contract changed: {path}')
    return r
def npz(path,binding,**arrays):
    path=Path(path)
    if path.exists():
        verify(path,binding)
        with np.load(path,allow_pickle=False) as old:
            require(set(old.files)==set(arrays) and all(np.array_equal(old[k],v) for k,v in arrays.items()),'Cache differs')
        return
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.with_suffix('.part').open('wb') as f:np.savez_compressed(f,**arrays)
    path.with_suffix('.part').replace(path);seal(path,binding)
def download(url,path,expected_sha=None,max_bytes=50_000_000):
    """Resume only validated 206 ranges. Existing final files require a receipt/hash."""
    import requests
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        if expected_sha:require(sha(path)==expected_sha,'Downloaded file changed')
        else:verify(path,{'url':url})
        return path
    part=path.with_suffix(path.suffix+'.part');start=part.stat().st_size if part.exists() else 0
    require(start<=max_bytes,'Oversized partial download')
    headers={'Range':f'bytes={start}-'} if start else {}
    with requests.get(url,headers=headers,stream=True,timeout=(15,60)) as r:
        r.raise_for_status()
        if start and r.status_code==200:start=0
        if start:require(r.status_code==206 and r.headers.get('Content-Range','').startswith(f'bytes {start}-'),'Invalid resume range')
        require(start+int(r.headers.get('Content-Length','0'))<=max_bytes,'Download exceeds declared cap')
        n=start
        with part.open('ab' if start else 'wb') as f:
            for b in r.iter_content(2**18):
                n+=len(b);require(n<=max_bytes,'Download exceeds streamed byte cap');f.write(b)
    if expected_sha:require(sha(part)==expected_sha,'Official LFS SHA256 mismatch')
    part.replace(path);seal(path,{'url':url});return path

class Run:
    def __init__(self,project):
        self.root=Path(project).resolve();self.data=self.root/'data/external/newsclippings'
        self.results=self.root/'results/external_eval/newsclippings';self.docs=self.root/'docs/external_eval/newsclippings'
        for p in [self.data,self.results,self.docs]:p.mkdir(parents=True,exist_ok=True)
    def protocol(self):
        p=self.results/'protocol.json';verify(p);return read(p)
    def binding(self):
        self.protocol();return {'protocol_sha256':sha(self.results/'protocol.json')}
    def frozen(self):
        p=self.protocol()
        for rel,h in p['frozen_assets'].items():require(sha(self.root/rel)==h,'Frozen asset changed: '+rel)
        return p
def preservation(run):
    inventory=read(run.results/'historical_inventory.json');bad=[]
    for rel,record in inventory['files'].items():
        p=run.root/rel
        if not p.exists() or p.stat().st_size!=record['bytes'] or sha(p)!=record['sha256']:bad.append(rel)
    report={'checked_files':len(inventory['files']),'changed_or_missing':bad,'passed':not bad,'inventory_sha256':sha(run.results/'historical_inventory.json')}
    write(run.results/'preservation_check.json',report);require(not bad,'Historical files changed');return report
