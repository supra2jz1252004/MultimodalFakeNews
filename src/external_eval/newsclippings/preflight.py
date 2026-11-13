"""Initial bounded official access audit, resumable without altering prior receipts."""
def preflight(run):
    from .io_utils import read
    existing=run.results/'data_access.json'
    if existing.exists():return read(existing)
    import json,hashlib,shutil,time,sys
    from pathlib import Path
    from datetime import datetime,timezone
    import requests
    P=run.root
    R=P/'results/external_eval/newsclippings';D=P/'data/external/newsclippings'
    R.mkdir(parents=True,exist_ok=True);D.mkdir(parents=True,exist_ok=True)
    def sha(p):
     h=hashlib.sha256()
     with p.open('rb') as f:
      for x in iter(lambda:f.read(2**20),b''):h.update(x)
     return h.hexdigest()
    def save(p,x):
     text=json.dumps(x,indent=2,sort_keys=True)+'\n'
     if p.exists():return json.loads(p.read_text())
     p.parent.mkdir(parents=True,exist_ok=True);p.write_text(text,encoding='utf-8');return x
    inventory=R/'historical_inventory.json'
    if not inventory.exists():
     start=time.monotonic();files={}
     for p in sorted(P.rglob('*')):
      if not p.is_file() or '__pycache__' in p.parts or p.suffix=='.pyc' or p.is_relative_to(R) or p.is_relative_to(D):continue
      files[p.relative_to(P).as_posix()]={'sha256':sha(p),'bytes':p.stat().st_size}
     save(inventory,{'created_utc':datetime.now(timezone.utc).isoformat(),'files':files,'count':len(files),'total_bytes':sum(x['bytes'] for x in files.values()),'seconds':time.monotonic()-start,'exclusions':['Python bytecode','new external_eval/newsclippings results/data']})
     print('inventory',len(files),flush=True)
    s=requests.Session();s.headers['User-Agent']='MultimodalFakeNews-Research-Audit/1.0'
    def get(url,name,limit=10_000_000):
     p=D/name
     if p.exists():return p
     with s.get(url,stream=True,timeout=(15,45)) as r:
      r.raise_for_status()
      if int(r.headers.get('Content-Length','0'))>limit:raise ValueError('size cap')
      buf=bytearray()
      for b in r.iter_content(65536):
       buf.extend(b)
       if len(buf)>limit:raise ValueError('stream size cap')
     p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(buf);return p
    api=get('https://huggingface.co/api/models/g-luo/news-clippings?blobs=true','hf_api.json')
    meta=json.loads(api.read_text());rev=meta['sha'];print('revision',rev,flush=True)
    info={'created_utc':datetime.now(timezone.utc).isoformat(),'revision':rev,'disk_free_bytes':shutil.disk_usage(P).free,'hf_license':meta.get('cardData',{}).get('license'),'files':[],'probes':[]}
    for f in meta.get('siblings',[]):
     n=f['rfilename']
     if (n.startswith('data/') and n.endswith(('test.json','val.json'))) or ('clip_' in n and n.endswith('_test.pkl')) or n.startswith('metadata/'):
      info['files'].append(f)
    for strategy in ['merged_balanced','person_sbert_text_text','scene_resnet_place','semantics_clip_text_image','semantics_clip_text_text']:
     for split in (['test','val'] if strategy=='merged_balanced' else ['test']):
      n=f'data/{strategy}/{split}.json';url=f'https://huggingface.co/g-luo/news-clippings/resolve/{rev}/{n}'
      try:
       p=get(url,n);obj=json.loads(p.read_text());print(n,p.stat().st_size,{k:(len(v) if isinstance(v,list) else str(v)[:150]) for k,v in obj.items()},flush=True)
       info['probes'].append({'url':url,'path':str(p),'sha256':sha(p),'bytes':p.stat().st_size,'status':'downloaded'})
      except Exception as e:info['probes'].append({'url':url,'error':str(e)})
    for url in ['https://www.cs.rice.edu/~vo9/visualnews/origin/data.json','https://www.cs.rice.edu/~vo9/visualnews/origin.tar']:
     try:
      r=s.head(url,timeout=(15,30),allow_redirects=True);record={'url':url,'method':'HEAD','status':r.status_code,'headers':dict(r.headers)};info['probes'].append(record);print('HEAD',url,r.status_code,r.headers.get('Content-Length'),flush=True)
     except Exception as e:info['probes'].append({'url':url,'error':str(e)})
    url='https://www.cs.rice.edu/~vo9/visualnews/origin.tar'
    try:
     with s.get(url,headers={'Range':'bytes=0-4095'},stream=True,timeout=(15,30)) as r:
      record={'url':url,'method':'GET Range bytes=0-4095','status':r.status_code,'headers':dict(r.headers)}
      if r.status_code==206 and r.headers.get('Content-Range','').startswith('bytes 0-4095/'):
       b=r.raw.read(4096);(D/'origin_first_4096.bin').write_bytes(b);record['bytes_read']=len(b);record['first_header_name']=b[:100].split(b'\0')[0].decode(errors='replace')
      else:record['bytes_read']=0;record['action']='Closed without downloading response body; range unsupported.'
      info['probes'].append(record);print('RANGE',record,flush=True)
    except Exception as e:info['probes'].append({'url':url,'method':'range','error':str(e)})
    save(R/'data_access.json',info)
    return info
