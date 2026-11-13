"""Atomic extension outputs, explicit contracts, and read-only historical adapters."""
from __future__ import annotations
import hashlib
import json
import os
import platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd

SEEDS = [42, 123, 2026]
LABEL6 = {0:'True', 1:'Satire/Parody', 2:'False Connection', 3:'Imposter Content',
          4:'Manipulated Content', 5:'Misleading Content'}
CLIP_ID = 'openai/clip-vit-base-patch32'
CLIP_REVISION = '3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268'

def require(condition, message):
    if not condition:
        raise ValueError(message)

def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()

def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def write_json(path, value, immutable=False):
    path = Path(path)
    if immutable and path.exists():
        require(read_json(path) == value, f'Immutable contract differs: {path}; use a new run ID')
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.part')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    temporary.replace(path)

def write_npz(path, **values):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix('.part').open('wb') as f:
        np.savez_compressed(f, **values)
    path.with_suffix('.part').replace(path)

def load_npz(path):
    with np.load(path, allow_pickle=False) as a:
        return {k:a[k] for k in a.files}

def stable_rank(ids, seed):
    return sorted(range(len(ids)), key=lambda i: (hashlib.sha256(f'{seed}\0{ids[i]}'.encode()).digest(), str(ids[i])))

@dataclass
class Context:
    project: Path
    output: Path
    run_id: str = 'bras_b4096_k16_s20260916'
    bank_size: int = 4096
    top_k: int = 16
    seed: int = 20260916
    threads: int = 6
    batch_size: int = 64
    smoke: bool = False

    def __post_init__(self):
        self.project, self.output = Path(self.project).resolve(), Path(self.output).resolve()
        require(self.run_id and all(c.isalnum() or c in '_-' for c in self.run_id), 'Unsafe run ID')
        require(self.bank_size > 0 and self.top_k > 0 and self.threads > 0 and self.batch_size > 0, 'Sizes must be positive')

    @property
    def name(self): return ('smoke_' if self.smoke else '') + self.run_id
    @property
    def cache(self): return self.output/'data/features/research_extension'
    @property
    def models(self): return self.output/'models/research_extension'/self.name
    @property
    def results(self): return self.output/'results/research_extension'/self.name
    @property
    def docs(self): return self.output/'docs/research_extension'
    @property
    def features(self): return self.cache/self.name

def manifest(ctx, scope=True):
    path = ctx.project/'data/processed/manifest.csv'
    f = pd.read_csv(path, keep_default_na=False, dtype={'sample_id':str,'group_id':str,'linked_submission_id':str})
    require(f.sample_id.is_unique and f.label.isin([0,1]).all(), 'Invalid IDs/binary labels')
    require(set(f.split) == {'train','validation','test'}, 'Missing official splits')
    require(f.label6.isin(range(6)).all() and (f.label.eq(1)==f.label6.eq(0)).all(), 'Six-way/binary label mismatch')
    require(f.group_id.ne('').all(), 'Missing source family')
    lock = read_json(ctx.project/'models/locked_config.json')
    require(lock['manifest_sha256'] == sha(path), 'Manifest differs from historical lock')
    f['input_hash'] = [hashlib.sha256((r.text+'\0'+r.sha256).encode()).hexdigest() for r in f.itertuples()]
    f['text_key'] = f.text.str.casefold()
    if ctx.smoke and scope:
        selected=[]
        for split, count in [('train',256),('validation',64)]:
            part=f[f.split.eq(split)]
            selected.extend(part.iloc[stable_rank(part.sample_id.tolist(),ctx.seed)[:count]].index.tolist())
        f=f.loc[selected]
    return f.reset_index(drop=True)

def load_historical(ctx, frame, kinds=('text','image','similarity')):
    """Reuses old chunks without calling the old index-writing load_features."""
    report=read_json(ctx.project/'data/features/extraction.json')
    directory=ctx.project/report['cache_directory']
    wanted={str(r.sample_id):(i,r) for i,r in enumerate(frame.itertuples())}
    values={}
    for kind in kinds:
        width={'text':768,'image':2048,'similarity':1}[kind]
        array=np.empty((len(frame),width),dtype=np.float32)
        seen=set()
        for path in sorted(directory.glob(f'{kind}_*.npz')):
            a=load_npz(path)
            require(a['features'].shape==(len(a['ids']),width) and a['features'].dtype==np.float32, f'Invalid shape/dtype: {path}')
            for j,sid in enumerate(a['ids']):
                sid=str(sid)
                if sid not in wanted: continue
                require(sid not in seen, f'Duplicate cached ID {sid}')
                i,r=wanted[sid]
                require((str(a['input_hash'][j]),str(a['splits'][j]),int(a['labels'][j]))==(r.input_hash,r.split,r.label), f'Stale cache {sid}')
                array[i]=a['features'][j]
                seen.add(sid)
        require(len(seen)==len(frame) and np.isfinite(array).all(), f'Missing/nonfinite historical {kind}')
        values[kind]=array
    return values

def fit_scaler(x, train):
    require(np.asarray(train,dtype=bool).any(), 'No training rows for normalization')
    fit=x[np.asarray(train,dtype=bool)]
    require(np.isfinite(fit).all(), 'Training features have missing values')
    return {'mean':fit.mean(0,dtype=np.float64).astype(np.float32),
            'std':np.maximum(fit.std(0,dtype=np.float64),1e-6).astype(np.float32)}

def standardize(x, scaler):
    y=(x-scaler['mean'])/scaler['std']
    require(np.isfinite(y).all(), 'Nonfinite normalized values')
    return y.astype(np.float32)

def peak_memory():
    """Process lifetime OS high-water mark, not a stage-isolated allocation estimate."""
    if os.name == 'nt':
        import ctypes
        from ctypes import wintypes
        class Counters(ctypes.Structure):
            _fields_=[('cb',wintypes.DWORD),('PageFaultCount',wintypes.DWORD)]+[(n,ctypes.c_size_t) for n in ['PeakWorkingSetSize','WorkingSetSize','QuotaPeakPagedPoolUsage','QuotaPagedPoolUsage','QuotaPeakNonPagedPoolUsage','QuotaNonPagedPoolUsage','PagefileUsage','PeakPagefileUsage']]
        c=Counters(); c.cb=ctypes.sizeof(c)
        kernel=ctypes.WinDLL('kernel32'); kernel.GetCurrentProcess.restype=wintypes.HANDLE
        query=ctypes.WinDLL('psapi').GetProcessMemoryInfo
        query.argtypes=[wintypes.HANDLE,ctypes.POINTER(Counters),wintypes.DWORD]
        query.restype=wintypes.BOOL
        ok=query(kernel.GetCurrentProcess(),ctypes.byref(c),c.cb)
        return {'peak_process_rss_bytes':int(c.PeakWorkingSetSize) if ok else None,'memory_scope':'process lifetime OS peak working set'}
    import resource
    return {'peak_process_rss_bytes':int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024)), 'memory_scope':'process lifetime OS peak RSS'}

def environment():
    from importlib.metadata import version
    return {'python':sys.version,'platform':platform.platform(),
            'packages':{k:version(k) for k in ['numpy','pandas','torch','transformers','scikit-learn','scipy','matplotlib']}}

def protocol(ctx):
    old=read_json(ctx.project/'models/locked_config.json')
    return {'schema':1,'method':'BRAS working name; exploratory Fakeddit extension',
        'manifest_sha256':sha(ctx.project/'data/processed/manifest.csv'),
        'historical_lock_sha256':sha(ctx.project/'models/locked_config.json'),
        'normalization_sha256':sha(ctx.project/'models/normalization.npz'),
        'config':old['config'],'budget':{**old['budget'],**({'epochs':2,'patience':2} if ctx.smoke else {})},
        'seeds':[42] if ctx.smoke else SEEDS,'bank_size':ctx.bank_size,'top_k':ctx.top_k,'bank_seed':ctx.seed,
        'selection':'fixed historical validation-selected configuration; no extension search',
        'primary_metric':'macro_f1','primary_comparison':['B','D'],'primary_seed':42,
        'secondary_comparisons':[['A','B'],['C','D'],['E','D'],['F','G'],['G','D'],['I','D']],
        'bootstrap_repetitions':5000,'bootstrap_seed':20260916,'alpha':0.05,
        'positive_class':0,'labels':{'0':'fake','1':'real'},'smoke':ctx.smoke,
        'interpretation':'paired row bootstrap conditional on this previously inspected subset; training seeds are not independent test datasets',
        'control_E':'two independent deterministic standard-normal features keyed by ID; no labels or evaluation distribution used',
        'control_I':'raw cosine plus fixed random projections of CLIP text (one) and CLIP image (one); same input dimension as D',
        'missing_candidates':'mean of min(k,n_valid); n_valid=0 is NaN and full training aborts',
        'quantiles':'train-only quartiles; high-fake >= train q75; low-real <= train q25',
        'stress':'synthetic correspondence rankings and prediction shifts only; no fake-news accuracy assigned to swaps'}

def lock_protocol(ctx):
    value=protocol(ctx)
    write_json(ctx.models/'protocol.json',value,immutable=True)
    return value
