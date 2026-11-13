"""Verify exact official revision and archive, then extract bounded regular files safely."""
from __future__ import annotations
import re,stat,zipfile,hashlib,os
from pathlib import Path,PurePosixPath,PureWindowsPath
import numpy as np,pandas as pd
from PIL import Image,ImageOps
from .core import *

def zip_members(z):
    require(len(z.infolist())<10000,'Unexpectedly large archive member count');seen=set();members=[];total=0
    for info in z.infolist():
        name=info.orig_filename
        require(name==info.filename,'Noncanonical ZIP name')
        require('\\' not in name and ':' not in name and '\x00' not in name,'Unsafe ZIP name')
        rel=PurePosixPath(name)
        require(not rel.is_absolute() and not PureWindowsPath(name).drive and '..' not in rel.parts,'Unsafe ZIP path traversal')
        require(all(p not in ('','.') and p.rstrip('. ')==p for p in rel.parts),'Ambiguous ZIP path')
        require(all(p.split('.')[0].upper() not in {'CON','PRN','AUX','NUL',*(f'COM{i}' for i in range(1,10)),*(f'LPT{i}' for i in range(1,10))} for p in rel.parts),'Reserved Windows ZIP path')
        require(name.casefold() not in seen,'Case-colliding archive names');seen.add(name.casefold())
        mode=info.external_attr>>16;require(not stat.S_ISLNK(mode),'ZIP symlink forbidden');require(not info.flag_bits&1,'Encrypted archive member')
        require(info.file_size<50_000_000 and info.file_size<=max(1,info.compress_size)*1000,'Implausible image member size/compression')
        if not info.is_dir():
            require(rel.suffix.lower() in ['.jpg','.jpeg','.png','.webp'],'Non-image archive payload');members.append(info)
        total+=info.file_size
    require(total<500_000_000,'Unexpected archive expansion');return members
def decode_hash(path):
    with Image.open(path) as source:
        source.seek(0);im=ImageOps.exif_transpose(source).convert('RGB');im.load()
        result={'sha256':sha(path),'pixel_sha256':hashlib.sha256(f'{im.width}x{im.height}:'.encode()+im.tobytes()).hexdigest(),'width':im.width,'height':im.height}
        im.close();return result
def extract_archive(archive,dest):
    dest=Path(dest).resolve();dest.mkdir(parents=True,exist_ok=True);rows=[]
    with zipfile.ZipFile(archive) as z:
        infos=zip_members(z)
        for info in infos:
            target=(dest/Path(*PurePosixPath(info.filename).parts)).resolve();require(target.is_relative_to(dest),'ZIP target escapes destination')
            content=z.read(info) # verifies ZIP CRC before committing any member
            require(len(content)==info.file_size,'Truncated ZIP member')
            put(target,content)
            try:record={'archive_member':info.filename,'bytes':info.file_size,'crc32':info.CRC,'decode_status':'verified',**decode_hash(target)}
            except Exception as exc:record={'archive_member':info.filename,'bytes':info.file_size,'decode_status':'invalid','error':type(exc).__name__+': '+str(exc)}
            rows.append(record)
    return pd.DataFrame(rows)
def annotation_frame(table):
    require({'caption','image_path','label'}<=set(table),'Missing annotation columns')
    require(set(table.label)==set(LABELS),'Unknown VERITE label');require(table.caption.str.len().gt(0).all(),'Empty caption')
    f=table[['caption','image_path','label']].copy();f.insert(0,'row_id',[f'verite_{i:04d}' for i in range(len(f))]);f=f.rename(columns={'label':'category'})
    f['binary_label']=f.category.map(LABELS)
    matched=f.image_path.str.extract(r'^images/(?:true|false)_(\d+)\.(?:jpg|jpeg|png)$',expand=False)
    require(matched.notna().all(),'Unrecognized official image-family path; inspect instead of assuming')
    f['family_id']=matched;f['text_key']=f.caption.map(canonical_text)
    return f
def pairing_checks(f):
    errors=[];ambiguous=[];counts={'miscaptioned':0,'out-of-context':0};links=[]
    for family,g in f.groupby('family_id',sort=True):
        t=g[g.category.eq('true')];m=g[g.category.eq('miscaptioned')];o=g[g.category.eq('out-of-context')]
        if len(t)!=1 or len(m)!=1 or len(o)>1:errors.append({'family':family,'reason':'unexpected category multiplicity'});continue
        t=t.iloc[0];m=m.iloc[0]
        if t.image_path!=m.image_path:errors.append({'family':family,'reason':'true/miscaptioned do not share image'})
        if t.text_key==m.text_key:ambiguous.append({'family':family,'reason':'identical true/miscaptioned normalized caption'})
        for cat,part in [('miscaptioned',g[g.category.eq('miscaptioned')]),('out-of-context',o)]:
            if not len(part):continue
            n=part.iloc[0]
            if cat=='out-of-context' and t.caption!=n.caption:errors.append({'family':family,'reason':'true/out-of-context do not share exact caption'})
            if cat=='out-of-context' and t.image_path==n.image_path:ambiguous.append({'family':family,'reason':'identical true/out-of-context image path'})
            links.append({'family_id':family,'category':cat,'true_row_id':t.row_id,'misleading_row_id':n.row_id});counts[cat]+=1
    require(not errors,'Official paired structure mismatch: '+str(errors[:5]))
    return {'errors':errors,'ambiguous':ambiguous,'paired_counts':counts},pd.DataFrame(links)
def audit(s):
    initial=read(s.results/'provenance_initial.json');hf=read(s.data/'provenance/hf_metadata.json')
    entry=next(x for x in hf['siblings'] if x['rfilename']=='images.zip')
    require(sha(s.data/'images.zip')==entry['lfs']['sha256'] and (s.data/'images.zip').stat().st_size==entry['size'],'Local ZIP is not the official authorized release')
    official=s.data/'provenance/VERITE.csv';require(sha(s.data/'VERITE.csv')==sha(official),'Local CSV is not byte-identical to pinned official repository annotation')
    original=pd.read_csv(s.data/'VERITE.csv',keep_default_na=False);f=annotation_frame(original)
    pairing,links=pairing_checks(f);images=extract_archive(s.data/'images.zip',s.images)
    csv(s.cache/'image_integrity.csv',images)
    require(images.archive_member.is_unique,'Duplicate ZIP paths')
    fields=['archive_member','sha256','pixel_sha256','width','height','decode_status']
    f=f.merge(images[fields],left_on='image_path',right_on='archive_member',how='left',validate='many_to_one')
    f['available']=f.decode_status.eq('verified');missing=f[~f.available]
    # Join family/source provenance from the pinned author article metadata without fetching publisher sites.
    articles=pd.read_csv(s.data/'provenance/VERITE_articles.csv',keep_default_na=False)
    require('id' in articles and articles.id.astype(str).is_unique,'Official article IDs are not unique')
    a=articles.assign(family_id=articles.id.astype(str)).set_index('family_id')
    urlcol='snopes_url' if 'snopes_url' in a else 'article_url';require(urlcol in a,'Unknown article provenance schema')
    f['article_url']=f.family_id.map(a[urlcol]).fillna('')
    f['article_metadata_join']=f.family_id.isin(a.index)
    f['source']=f.article_url.map(lambda x:'reuters' if 'reuters.' in x else ('snopes' if 'snopes.' in x else 'unknown'))
    f['cluster_id']=clusters(f)
    known=pd.read_csv(s.root/'data/processed/manifest.csv',keep_default_na=False)
    fresh=pd.read_csv(s.root/'data/features/research_extension/fresh_holdout/manifest.csv',keep_default_na=False)
    fresh['split']='previous_fresh_test';known=pd.concat([known,fresh],ignore_index=True)
    for key,values in [('text_key',known.text.map(canonical_text)),('sha256',known.sha256),('pixel_sha256',known.pixel_sha256)]:
        f['overlap_'+key]=f[key].isin(set(values))&f[key].fillna('').ne('')
    f['known_fakeddit_overlap']=f[['overlap_text_key','overlap_sha256','overlap_pixel_sha256']].any(axis=1)
    affected=set(f.loc[f.known_fakeddit_overlap,'cluster_id']);f['independence_sensitivity_eligible']=~f.cluster_id.isin(affected)
    # Complete records only, with fixed missingness; pairs require both records. No replacement images.
    for k in ['sha256','pixel_sha256','archive_member','decode_status']:f[k]=f[k].fillna('')
    csv(s.cache/'manifest.csv',f);seal(s.cache/'manifest.csv',{'csv_sha256':sha(official),'zip_sha256':sha(s.data/'images.zip')})
    links['complete']=links.true_row_id.isin(set(f[f.available].row_id))&links.misleading_row_id.isin(set(f[f.available].row_id))
    csv(s.results/'pair_links.csv',links)
    report={'status':'FULL_LOCAL_DATA' if len(missing)==0 else 'PARTIAL_LOCAL_DATA','official_csv_rows':len(f),'label_counts':f.category.value_counts().to_dict(),
      'available_records':int(f.available.sum()),'missing_or_invalid_records':len(missing),'missing_row_ids':missing.row_id.tolist(),
      'unique_referenced_images':f.image_path.nunique(),'duplicate_image_reference_rows':int(f.image_path.duplicated().sum()),
      'unique_captions':f.caption.nunique(),'repeated_caption_rows':int(f.caption.duplicated().sum()),'families':f.family_id.nunique(),'clusters':f.cluster_id.nunique(),
      'zip_image_files':len(images),'all_archive_images_decoded':bool(images.decode_status.eq('verified').all()),'unreferenced_archive_members':sorted(set(images.archive_member)-set(f.image_path)),
      'unique_byte_hashes_referenced':f.sha256.nunique(),'unique_pixel_hashes_referenced':f.pixel_sha256.nunique(),
      'file_and_annotation_mismatches':0,'csv_matches_official_bytes':True,'zip_matches_official_LFS_sha256':True,
      'csv_sha256':sha(official),'zip_sha256':sha(s.data/'images.zip'),'github_revision':initial['github_revision'],'hf_revision':hf['sha'],
      'article_metadata_joined_rows':int(f.article_metadata_join.sum()),'sources':f.source.value_counts().to_dict(),'pairing':pairing,
      'complete_pair_counts':links[links.complete].category.value_counts().to_dict(),'known_fakeddit_overlap_rows':int(f.known_fakeddit_overlap.sum()),
      'known_fakeddit_overlap_components':len(affected),'independence_sensitivity_rows':int(f.independence_sensitivity_eligible.sum()),
      'release_caveat':'Current official CSV is evaluated verbatim. Legacy README/paper counts may differ; no silent trimming to an older count.',
      'privacy':'Local authorized images and embeddings only. No credentials read, no image upload/display/redistribution.'}
    write(s.results/'dataset_audit.json',report);return report

def protocol(s):
    audit=read(s.results/'dataset_audit.json');require(audit['csv_matches_official_bytes'] and audit['zip_matches_official_LFS_sha256'],'Provenance not verified')
    assets={}
    patterns=['models/final/baseline_*.pt','models/final/proposed_*.pt','models/normalization.npz',
      'models/research_extension/bras_b4096_k16_s20260916/[CD]_*.pt','models/research_extension/bras_b4096_k16_s20260916/[CD]_*.json','models/research_extension/bras_b4096_k16_s20260916/normalization.npz',
      'models/research_extension/ccr/full/gate_*.joblib','models/research_extension/ccr/full/policy_*.json','models/research_extension/ccr/protocol.json','results/research_extension/ccr/full/frozen_policy.json',
      'data/features/extraction.json','data/features/research_extension/bras_b4096_k16_s20260916/bank.npz','data/features/research_extension/bras_b4096_k16_s20260916/bank.json',
      'src/models.py','src/feature_extractors.py','src/research_extension/training.py','src/research_extension/reference.py','src/research_extension/common.py','src/research_extension/ccr/*.py',
      'src/external_eval/newsclippings/io_utils.py','src/external_eval/newsclippings/statistics.py']
    for pattern in patterns:
        for p in s.root.glob(pattern):assets[p.relative_to(s.root).as_posix()]=sha(p)
    obj={'schema':1,'created_utc':now(),'locked_before_VERITE_inference':True,'task':'cross-dataset cross-task binary transfer diagnostic; not Fakeddit-equivalent fake-news labels',
      'mapping':LABELS,'positive_class':'truthful (true=1); misleading pair=0','population':'All records in the pinned current official CSV; no selection by model outputs',
      'official_rows':audit['official_csv_rows'],'expected_counts':audit['label_counts'],'seed_models':SEEDS,'models':KINDS,
      'BRAS_classifier':'D (raw cosine plus two margins) primary; C (two margins) secondary; frozen existing heads and scalers only',
      'CCR':'Load and execute saved gate/policy; current guard fallback is B; identity control only',
      'threshold':'Original two-logit argmax; tie class0; no external calibration, threshold search, fine-tuning or new model selection',
      'primary_transfer_comparisons':[{'candidate':'B','base':'A','seed':42},{'candidate':'D','base':'B','seed':42}],
      'primary_transfer_metric':'Macro-F1; two-sided within-component model-prediction swap randomization; Holm correction across the two primary tests',
      'secondary_transfer_comparisons':'Same comparisons at seeds123/2026 plus C-B and CCR-B all seeds; Holm within this secondary family',
      'classification_metrics':['accuracy','macro_f1','roc_auc_truthful','average_precision_truthful','log_loss','brier'],
      'alignment_scores':SCORES,'alignment_combination':'bras_mean=(m_text+m_image)/2; no learned weights; existing train-only 4096 bank, k=16',
      'alignment_primary':'bras_mean minus raw cosine half-credit ranking in true-vs-miscaptioned and true-vs-out-of-context pairs; Holm across two tests',
      'alignment_secondary':'directional margins vs cosine; Holm across 4 directional/category tests; same shared pairs and component draws',
      'ties':'exact score ties receive0.5; strict and 1e-6-tolerance ranking are descriptive sensitivity checks',
      'uncertainty':{'bootstrap_repetitions':5000,'bootstrap_seed':20261109,'cluster_unit':'Connected components of original family/article URL/shared normalized caption/image path/file hash/pixel hash','ci':'percentile95%; common-component resamples for every compared model'},
      'tests':{'randomization_repetitions':10000,'seed':20261110,'p_value':'two-sided +1 Monte Carlo correction; swap all predictions within each component','no_row_independent_McNemar':'Repeated content violates its independent-pair assumption'},
      'robustness':['family-only bootstrap sensitivity','exclude components overlapping known Fakeddit text/bytes/pixels','per-category and per-source breakdown','strict and tolerance alignment ties','unaltered majority-negative accuracy/Macro-F1 diagnostic'],
      'failures':'Ten misleading records with highest raw cosine; ten smallest true-minus-misleading paired cosine margins per category; deterministic ID tie break; exploratory and locally logged only',
      'budget':{'threads':6,'batch_size':16,'feature_chunk':64,'device':'cpu'},
      'encoder_contract':read(s.root/'data/features/extraction.json')['contract'],'annotations_sha256':audit['csv_sha256'],'images_zip_sha256':audit['zip_sha256'],
      'manifest_sha256':sha(s.cache/'manifest.csv'),'pair_links_sha256':sha(s.results/'pair_links.csv'),'historical_inventory_sha256':sha(s.results/'historical_inventory.json'),'frozen_assets':assets}
    path=s.results/'protocol.json'
    if path.exists():obj['created_utc']=read(path)['created_utc']
    write(path,obj);seal(path,{'manifest_sha256':obj['manifest_sha256'],'inventory_sha256':obj['historical_inventory_sha256']});return {'locked':True,'sha256':sha(path),'rows':obj['official_rows']}
