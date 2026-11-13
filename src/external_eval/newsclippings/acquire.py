"""Bounded access audit and explicit official embedding fallback."""
from __future__ import annotations
import io,pickle,time
import numpy as np
import pandas as pd
from .io_utils import *
from .data import *

def asset(run,name):
    api=read(run.data/'hf_api.json');f=next(x for x in api['siblings'] if x['rfilename']==name)
    require(f['size']<110_000_000,'Asset exceeds per-file budget')
    return download(f"https://huggingface.co/g-luo/news-clippings/resolve/{api['sha']}/{name}",run.data/name,f['lfs']['sha256'],f['size'])
def smoke(run):
    run.frozen();f=pd.read_csv(run.results/'smoke_manifest.csv',keep_default_na=False,dtype={'caption_id':str,'image_id':str})
    require(f.pair_id.nunique()==100 and len(f)==200,'Expected fixed 100-pair smoke')
    again=select(parse_pairs(read(run.data/'data/merged_balanced/test.json')),100)
    require(f.row_id.tolist()==again.row_id.tolist(),'Nonreproducible smoke selection')
    path=asset(run,'metadata/test.json');meta=read(path)
    # Official supplementary metadata is not assumed to equal VisualNews origin/data.json.
    report={'pairs':100,'rows':200,'strategy_counts':f[f.falsified.eq(0)].strategy.value_counts().to_dict(),'deterministic_ids':True,
       'metadata_sha256':sha(path),'metadata_type':type(meta).__name__,'no_test_scores_computed':True,
       'visualnews_direct_metadata_http':404,'archive_bytes':97916641280,'archive_range_supported':True,
       'status':'PARTIAL','image_paths_are_not_image_bytes':True}
    print('Supplementary metadata shape',type(meta).__name__,len(meta),'sample',str(meta)[:1200],flush=True)
    if isinstance(meta,list):report['metadata_first_record_keys']=list(meta[0])
    elif isinstance(meta,dict):report['metadata_top_keys']=list(meta)[:15]
    write(run.results/'metadata_smoke.json',report)
    image_availability(run,meta)
    return report

def enriched(run):
    f=pd.read_csv(run.results/'selection_manifest.csv',keep_default_na=False,dtype={'caption_id':str,'image_id':str})
    meta=read(run.data/'metadata/test.json')
    require(isinstance(meta,dict) and all(str(v['id'])==str(k) for k,v in meta.items()),'Supplementary ID contract invalid')
    f=f.drop(columns=['source','image_source','filename'])
    f=join_metadata(f,meta);require(f.metadata_join.all() and f.caption.str.len().gt(0).all(),'Caption/image supplementary join incomplete')
    # ID joins verified against author metadata; not falsely claimed as a downloaded origin/data.json.
    f['cluster_id']=linked_clusters(f);csv(run.results/'joined_manifest.csv',f)
    seal(run.results/'joined_manifest.csv',{'selection_sha256':sha(run.results/'selection_manifest.csv'),'metadata_sha256':sha(run.data/'metadata/test.json')})
    return f

def image_availability(run,meta):
    import requests,concurrent.futures
    f=enriched(run);smoke=f[f.smoke.astype(str).str.lower().eq('true')]
    path=run.results/'image_access_smoke.json'
    if path.exists():return read(path)
    def probe(row):
        rel=row.filename.removeprefix('visual_news/');url='https://www.cs.rice.edu/~vo9/visualnews/'+rel
        try:
            r=requests.head(url,allow_redirects=True,timeout=(10,20))
            return {'image_id':row.image_id,'filename':row.filename,'url':url,'http_status':r.status_code,'verified_image_bytes':False,'method':'HEAD'}
        except requests.RequestException as e:return {'image_id':row.image_id,'url':url,'error':str(e),'verified_image_bytes':False}
    unique=smoke.drop_duplicates('image_id')
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:probes=list(pool.map(probe,unique.itertuples()))
    result={'smoke_pairs':100,'rows':len(smoke),'unique_images_probed':len(probes),'verified_selected_images':0,
      'supplementary_caption_image_joins':len(f),'source_rows':f.groupby(['strategy','source']).size().reset_index(name='rows').to_dict('records'),
      'clusters_after_known_article_and_caption_links':f.cluster_id.nunique(),'probes':probes,
      'blocker':'Selected image bytes not obtained: direct official per-image URLs are not an advertised distribution and probes record their actual statuses. The available 97,916,641,280-byte tar lacks a published per-file offset index. No full archive download or publisher scraping was attempted.',
      'unblocking':'Provide an authorized local VisualNews origin/data.json and the fixed selected image paths, or an official indexed subset. Archive range support alone does not locate selected images.'}
    write(path,result);return result

class NumpyOnly(pickle.Unpickler):
    """No arbitrary Python globals; accepts only numpy numeric array reconstruction."""
    def find_class(self,module,name):
        allowed={('numpy','ndarray'):np.ndarray,('numpy','dtype'):np.dtype,
          ('numpy.core.multiarray','_reconstruct'):np._core.multiarray._reconstruct,
          ('numpy._core.multiarray','_reconstruct'):np._core.multiarray._reconstruct,
          ('numpy.core.multiarray','scalar'):np._core.multiarray.scalar,
          ('numpy._core.multiarray','scalar'):np._core.multiarray.scalar}
        require((module,name) in allowed,f'Unsafe/unsupported pickle global: {module}.{name}')
        return allowed[(module,name)]
def load_vectors(path):
    with Path(path).open('rb') as stream:obj=NumpyOnly(stream).load()
    require(isinstance(obj,dict),'Expected official ID-array dictionary');out={};norms=[]
    for k,v in obj.items():
        require(isinstance(k,(int,str,np.integer)) and isinstance(v,np.ndarray),'Invalid embedding dictionary')
        require(v.shape in [(512,),(1,512)] and v.dtype.kind=='f' and np.isfinite(v).all(),'Invalid CLIP vector')
        vec=v.reshape(512).astype(np.float32);norm=float(np.linalg.norm(vec));require(norm>1e-10,'Zero CLIP vector')
        out[str(k)]=vec/norm;norms.append(norm)
    return out,{'count':len(out),'dimension':512,'original_norm_min':min(norms),'original_norm_max':max(norms),'original_norm_median':float(np.median(norms)),
      'normalized_before_cosine':True,'source_sha256':sha(path)}
def official_features(run):
    run.frozen();f=enriched(run)
    binding={**run.binding(),'selection_sha256':sha(run.results/'selection_manifest.csv'),'provenance':'official_precomputed_CLIP_ViT-B32_not_project_pinned_encoder'}
    dest=run.data/'official_features.npz'
    if dest.exists():verify(dest,binding);return read(run.results/'embedding_provenance.json')
    start=time.perf_counter();tv,ta=load_vectors(asset(run,'embeddings/clip_text_embeddings/clip_text_embeddings_test.pkl'))
    iv,ia=load_vectors(asset(run,'embeddings/clip_image_embeddings/clip_image_embeddings_test.pkl'))
    avail=f.caption_id.isin(tv)&f.image_id.isin(iv);complete=f.groupby('pair_id').apply(lambda g:bool(avail.loc[g.index].all()),include_groups=False)
    keep=f.pair_id.isin(complete[complete].index);use=f[keep]
    require(len(use)>0,'No complete groups supported by official embeddings')
    npz(dest,binding,ids=use.row_id.to_numpy(dtype=str),clip_text=np.stack([tv[x] for x in use.caption_id]),clip_image=np.stack([iv[x] for x in use.image_id]))
    audit=f[['row_id','pair_id','strategy','source']].copy();audit['embedding_available']=avail;audit['complete_group_retained']=keep
    csv(run.results/'embedding_availability.csv',audit)
    report={'evidence_status':'ALIGNMENT ONLY','source':'https://github.com/g-luo/news_clippings#embeddings','image':ia,'text':ta,
       'selected_pairs':1000,'complete_pairs':use.pair_id.nunique(),'missing_pairs':int((~complete).sum()),'seconds':time.perf_counter()-start,
       'contract':'Authors identify CLIP ViT-B/32, 512D; exact weight revision, image decoding, processor version, truncation and runtime not pinned in supplied embedding release.',
       'not_executed':['project pinned CLIP encoder','pinned BRAS cross-score comparison','BERT/ResNet Fakeddit transfer'],
       'bras_blocker':'Official embeddings are not verified to match the existing pinned Fakeddit bank feature contract; no mixing of feature provenances.',
       'no_real_selected_image_bytes':True,'normalization':'L2 normalization of each supplied vector before dot product; original norm ranges saved.'}
    write(run.results/'embedding_provenance.json',report);return report
