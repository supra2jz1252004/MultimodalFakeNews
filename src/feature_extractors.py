"""Frozen CPU encoders and sample-addressed, resumable feature chunks."""
from __future__ import annotations

import gc
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageOps
from torch.nn import functional as F
from torchvision.models import ResNet50_Weights, resnet50
from transformers import AutoTokenizer, BertModel, CLIPModel, CLIPProcessor

from .utils import atomic_json, file_hash, log_stage

BERT = 'google-bert/bert-base-uncased'
BERT_REVISION = '86b5e0934494bd15c9632b12f734a8a67f723594'
CLIP = 'openai/clip-vit-base-patch32'
CLIP_REVISION = '3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268'
DIMS = {'text': 768, 'image': 2048, 'similarity': 1}


def read_manifest(root: Path, smoke: bool = False) -> pd.DataFrame:
    name = 'smoke_manifest.csv' if smoke else 'manifest.csv'
    frame = pd.read_csv(root / 'data/processed' / name, keep_default_na=False,
                        dtype={'sample_id': str, 'group_id': str})
    assert frame.sample_id.is_unique and frame.label.isin([0, 1]).all()
    assert set(frame.split) == {'train', 'validation', 'test'}
    frame['input_hash'] = [hashlib.sha256((r.text+'\0'+r.sha256).encode()).hexdigest()
                           for r in frame.itertuples()]
    return frame


def token_audit(root: Path, frame: pd.DataFrame, smoke: bool) -> tuple:
    tokenizer = AutoTokenizer.from_pretrained(BERT, revision=BERT_REVISION)
    clip_processor = CLIPProcessor.from_pretrained(CLIP, revision=CLIP_REVISION, use_fast=False)
    train = frame[frame.split.eq('train')]
    lengths = {}
    for name, tok in [('bert', tokenizer), ('clip', clip_processor.tokenizer)]:
        lengths[name] = np.array([len(x) for x in tok(train.text.tolist(), truncation=False)['input_ids']])
    p99 = np.quantile(lengths['bert'], .99)
    max_length = next((n for n in [64,128,256,512] if n >= p99),512)
    report = {'scope': 'training only', 'train_samples': len(train), 'bert_max_length': max_length,
              'choice': 'Smallest of 64/128/256/512 covering training 99th percentile; dynamic batch padding.',
              'clip_max_length': 77}
    for name, values in lengths.items():
        limit = max_length if name=='bert' else 77
        report[name] = {'median': float(np.median(values)), 'p90': float(np.quantile(values,.9)),
                        'p95': float(np.quantile(values,.95)), 'p99': float(np.quantile(values,.99)),
                        'maximum': int(values.max()), 'truncated': int((values>limit).sum())}
    suffix = '_smoke' if smoke else ''
    atomic_json(root / f'data/processed/token_lengths{suffix}.json', report)
    pd.DataFrame({'sample_id': train.sample_id, **lengths}).to_csv(root / f'data/processed/token_lengths{suffix}.csv',index=False)
    print('Training token lengths:', json.dumps(report), flush=True)
    return tokenizer, clip_processor, max_length


def cache_contract(max_length: int) -> dict:
    return {'schema': 1, 'bert': BERT, 'bert_revision': BERT_REVISION,
            'bert_max_length': max_length, 'bert_pooling': 'last_hidden_state[:,0,:]',
            'resnet': 'torchvision ResNet50 IMAGENET1K_V2; fc=Identity',
            'resnet_preprocessing': 'weights.transforms(): resize 232, center crop 224, ImageNet mean/std',
            'clip': CLIP, 'clip_revision': CLIP_REVISION, 'clip_max_length': 77,
            'clip_preprocessing': 'CLIPProcessor use_fast=False; resize shortest edge 224, center crop 224, CLIP mean/std',
            'image_decode': 'first frame, EXIF orientation, RGB; no blank substitution or augmentation',
            'similarity': 'per-row dot of L2-normalized projected embeddings, checked against cosine_similarity',
            'dtype': 'float32', 'dimensions': DIMS,
            'torch': torch.__version__}


def cache_directory(root: Path, contract: dict) -> Path:
    digest = hashlib.sha256(json.dumps(contract,sort_keys=True).encode()).hexdigest()[:16]
    return root / 'data/features' / digest


def chunk_index(directory: Path, kind: str) -> dict:
    index = {}
    for path in sorted(directory.glob(f'{kind}_*.npz')):
        with np.load(path, allow_pickle=False) as part:
            arrays={key:part[key] for key in part.files}
            assert arrays['features'].shape == (len(arrays['ids']), DIMS[kind])
            assert np.isfinite(arrays['features']).all(), f'Nonfinite cache: {path}'
            assert all(len(arrays[key])==len(arrays['ids']) for key in ['input_hash','splits','labels'])
            for i, sid in enumerate(arrays['ids']):
                assert str(sid) not in index, f'Duplicate cached ID: {sid}'
                index[str(sid)] = {'path': path, 'row': i, 'input_hash': str(arrays['input_hash'][i]),
                                   'split': str(arrays['splits'][i]), 'label': int(arrays['labels'][i])}
    return index


def open_rgb(path: Path) -> Image.Image:
    with Image.open(path) as original:
        original.seek(0)
        return ImageOps.exif_transpose(original).convert('RGB')


@torch.inference_mode()
def encode(kind: str, rows: pd.DataFrame, root: Path, processor, model, max_length: int) -> np.ndarray:
    texts = rows.text.tolist()
    if kind == 'text':
        tokens = processor(texts,return_tensors='pt',padding=True,truncation=True,max_length=max_length)
        result = model(**tokens).last_hidden_state[:,0,:]
    else:
        images = [open_rgb(root / path) for path in rows.image_path]
        try:
            if kind == 'image':
                result = model(torch.stack([processor(im) for im in images]))
            else:
                inputs = processor(text=texts,images=images,return_tensors='pt',padding=True,
                                   truncation=True,max_length=77)
                output = model(**inputs)
                text = F.normalize(output.text_embeds,dim=-1)
                image = F.normalize(output.image_embeds,dim=-1)
                assert text.shape == image.shape == (len(rows),512)
                result = (text*image).sum(dim=-1,keepdim=True)
                explicit = F.cosine_similarity(output.text_embeds,output.image_embeds,dim=-1)
                assert torch.allclose(result[:,0],explicit,atol=1e-6,rtol=1e-5)
                assert torch.allclose(result[:,0],(text@image.T).diag(),atol=1e-6,rtol=1e-5)
                assert (result.abs() <= 1.00001).all()
        finally:
            for im in images:
                im.close()
    array = result.cpu().numpy().astype(np.float32)
    assert array.shape == (len(rows),DIMS[kind]) and np.isfinite(array).all()
    return array


def extract(root: Path, smoke: bool = False, batch_size: int = 16, threads: int = 6) -> None:
    started = time.monotonic()
    torch.set_num_threads(threads)
    frame = read_manifest(root,smoke)
    for row in frame.itertuples():
        assert file_hash(root / row.image_path)==row.sha256, f'Changed image: {row.sample_id}'
    tokenizer, clip_processor, max_length = token_audit(root,frame,smoke)
    contract = cache_contract(max_length)
    directory = cache_directory(root,contract)
    directory.mkdir(parents=True,exist_ok=True)
    atomic_json(directory / 'contract.json',contract)
    suffix = '_smoke' if smoke else ''
    report = {'samples':len(frame),'batch_size':batch_size,'threads':threads,'cuda_available':torch.cuda.is_available(),
              'contract':contract,'cache_directory':directory.relative_to(root).as_posix(),'encoders':{}}
    for kind in DIMS:
        index = chunk_index(directory,kind)
        for row in frame.itertuples():
            if row.sample_id in index:
                prior = index[row.sample_id]
                assert prior['input_hash']==row.input_hash and prior['split']==row.split and prior['label']==row.label
        pending = frame[~frame.sample_id.isin(index)]
        print(f'{kind}: {len(frame)-len(pending)} verified cached, {len(pending)} pending',flush=True)
        begin = time.monotonic()
        if len(pending):
            if kind=='text':
                processor, model = tokenizer, BertModel.from_pretrained(BERT,revision=BERT_REVISION)
            elif kind=='image':
                weights = ResNet50_Weights.IMAGENET1K_V2
                processor, model = weights.transforms(), resnet50(weights=weights)
                model.fc = torch.nn.Identity()
            else:
                processor, model = clip_processor, CLIPModel.from_pretrained(CLIP,revision=CLIP_REVISION)
            model.eval().requires_grad_(False)
            chunk_number = max([int(p.stem.rsplit('_',1)[1]) for p in directory.glob(f'{kind}_*.npz')],default=-1)+1
            for start in range(0,len(pending),128):
                chunk = pending.iloc[start:start+128]
                features = np.concatenate([encode(kind,chunk.iloc[j:j+batch_size],root,processor,model,max_length)
                                           for j in range(0,len(chunk),batch_size)])
                path = directory / f'{kind}_{chunk_number:06}.npz'
                temporary = path.with_suffix('.part')
                with temporary.open('wb') as stream:
                    np.savez_compressed(stream,ids=chunk.sample_id.to_numpy(dtype=str),
                                        splits=chunk.split.to_numpy(dtype=str),labels=chunk.label.to_numpy(dtype=np.int64),
                                        input_hash=chunk.input_hash.to_numpy(dtype=str),features=features)
                temporary.replace(path)
                chunk_number += 1
                completed = min(start+len(chunk),len(pending))
                elapsed = time.monotonic()-begin
                print(f'{kind}: {completed}/{len(pending)} new rows, {elapsed:.1f}s, {completed/max(elapsed,.001):.2f} samples/s',flush=True)
            del model, processor
            gc.collect()
        report['encoders'][kind] = {'new_samples':len(pending),'reused_samples':len(frame)-len(pending),
                                    'runtime_seconds':time.monotonic()-begin}
        atomic_json(root/f'data/features/extraction{suffix}.json',report)
    report['runtime_seconds'] = time.monotonic()-started
    report['cache_bytes'] = sum(p.stat().st_size for p in directory.glob('*.npz'))
    atomic_json(root/f'data/features/extraction{suffix}.json',report)
    load_features(root,smoke)
    history=root/'data/features/extraction_history.jsonl'
    if not history.exists():
        previous=[json.loads(p.read_text()) for p in (root/'data/features').glob('extraction*.json')
                  if p.name!=f'extraction{suffix}.json']
        history.write_text(''.join(json.dumps(record)+'\n' for record in previous),encoding='utf-8')
    with history.open('a',encoding='utf-8') as stream:
        stream.write(json.dumps(report)+'\n')
    log_stage(root,'Frozen feature extraction'+(' smoke' if smoke else ''),
              f'Command: `python run_experiments.py extract-features {"--smoke" if smoke else ""} --batch-size {batch_size} --threads {threads}`. '
              f'{len(frame):,} pairs, float32 BERT 768 + ResNet50 2048 + CLIP cosine 1. '
              f'BERT max_length {max_length}; train-only token statistics in data/processed/token_lengths{suffix}.json. '
              f'Runtime {report["runtime_seconds"]:.1f}s; cache {report["cache_bytes"]:,} bytes. '
              'All shapes, finite values, IDs, labels, original split membership, input hashes and paired cosine assertions passed. '
              'Encoder contracts and per-stage runtime saved in data/features; no classifier metrics yet.')


def load_features(root: Path, smoke: bool = False) -> tuple[pd.DataFrame, dict]:
    frame = read_manifest(root,smoke)
    suffix = '_smoke' if smoke else ''
    report = json.loads((root/f'data/features/extraction{suffix}.json').read_text())
    directory = root / report['cache_directory']
    values, readable = {}, []
    for kind in DIMS:
        index = chunk_index(directory,kind)
        features = np.empty((len(frame),DIMS[kind]),dtype=np.float32)
        needed = {}
        for i,row in enumerate(frame.itertuples()):
            record = index[row.sample_id]
            assert record['input_hash']==row.input_hash and record['label']==row.label and record['split']==row.split
            needed.setdefault(record['path'],[]).append((i,record['row']))
            readable.append({'sample_id':row.sample_id,'split':row.split,'label':row.label,'kind':kind,
                             'chunk':record['path'].relative_to(root).as_posix(),'row':record['row']})
        for path, positions in needed.items():
            with np.load(path,allow_pickle=False) as part:
                chunk_features=part['features']
                targets,sources=zip(*positions)
                features[list(targets)] = chunk_features[list(sources)]
        assert np.isfinite(features).all()
        values[kind] = features
    pd.DataFrame(readable).to_csv(root/f'data/features/index{suffix}.csv',index=False)
    return frame, values
