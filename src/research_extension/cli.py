from __future__ import annotations
import argparse
import json
import sys
import time
import traceback
from pathlib import Path
from .common import Context,write_json,peak_memory

def parser():
    p=argparse.ArgumentParser(description='Auditable, CPU-only BRAS extension. Historical project is read-only; outputs are isolated.')
    sub=p.add_subparsers(dest='stage',required=True)
    for stage in ['audit','extract-clip','build-reference-bank','generate-features','smoke','train','evaluate',
                  'stress-test','plots','prepare-fresh-holdout','extract-fresh-holdout','evaluate-fresh-holdout',
                  'recompute-paired','all']:
        s=sub.add_parser(stage,help={'all':'Audit, extract, smoke, generate, train, evaluate, stress and plot (may take hours)',
            'smoke':'Real-data train/validation pipeline check; never evaluates test',
            'extract-clip':'Resume only missing CLIP vectors using local pinned weights'}.get(stage,stage))
        s.add_argument('--project-root',type=Path,default=Path(__file__).resolve().parents[2],help='Existing project containing src, data and models (read only)')
        s.add_argument('--output-root',type=Path,default=Path(__file__).resolve().parents[2],help='Extension package/artifact root; no historical files are overwritten')
        s.add_argument('--run-id',default='bras_b4096_k16_s20260916',help='Immutable configuration namespace; change when tuning')
        s.add_argument('--bank-size',type=int,default=4096)
        s.add_argument('--top-k',type=int,default=16)
        s.add_argument('--seed',type=int,default=20260916,help='Bank/control seed, not classifier seeds')
        s.add_argument('--threads',type=int,default=6)
        s.add_argument('--batch-size',type=int,default=64,help='Bounded query-by-bank batch size')
        s.add_argument('--scope',choices=['full','smoke'],default='full',help='Smoke uses deterministic 256 training and 64 validation rows')
        if stage in ['extract-clip','all']:
            s.add_argument('--encoder-batch-size',type=int,default=16)
        if stage=='extract-clip': s.add_argument('--max-new',type=int,help='Bound this invocation; completed chunks are resumable')
        if stage in ['train','evaluate']:
            s.add_argument('--variants',default='A,B,C,D,E,F,G,I',help='Comma-separated variants; I is the extra CLIP-unimodal control')
        if stage=='audit':
            s.add_argument('--verify-images',action='store_true',help='Hash every retained image; potentially slow')
            s.add_argument('--verify-preservation',action='store_true',help='Verify the initial historical inventory')
        if stage=='stress-test':
            s.add_argument('--limit',type=int,default=500,help='Deterministic maximum query count; no label-based selection')
        if stage=='prepare-fresh-holdout':
            s.add_argument('--per-class',type=int,default=500,help='Exact fake and real fresh-test counts; no silent quota reduction')
            s.add_argument('--download-workers',type=int,default=12)
        if stage=='extract-fresh-holdout':
            s.add_argument('--encoder-batch-size',type=int,default=16)
        if stage=='recompute-paired':
            s.add_argument('--dataset-scope',choices=['retained-test','fresh-holdout'],default='retained-test')
    return p

def main():
    args=parser().parse_args()
    smoke=args.scope=='smoke' or args.stage=='smoke'
    bank_size=64 if args.stage=='smoke' and args.bank_size==4096 else args.bank_size
    ctx=Context(args.project_root,args.output_root,args.run_id,bank_size,args.top_k,args.seed,args.threads,args.batch_size,smoke)
    started=time.perf_counter()
    logdir=ctx.output/'results/research_extension/logs'; logdir.mkdir(parents=True,exist_ok=True)
    record={'argv':sys.argv,'stage':args.stage,'started_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'project_root':str(ctx.project),'output_root':str(ctx.output)}
    try:
        if args.stage=='audit':
            from .audit import audit
            result=audit(ctx,args.verify_images,args.verify_preservation)
        elif args.stage=='extract-clip':
            from .extraction import extract
            result=extract(ctx,args.max_new,args.encoder_batch_size)
        elif args.stage=='build-reference-bank':
            from .features import build_bank
            result=build_bank(ctx)
        elif args.stage=='generate-features':
            from .features import generate
            result=generate(ctx)
        elif args.stage=='smoke':
            from .training import smoke_test
            result=smoke_test(ctx)
        elif args.stage=='train':
            from .training import train
            result=train(ctx,args.variants.split(','))
        elif args.stage=='evaluate':
            from .evaluation import evaluate
            result=evaluate(ctx,args.variants.split(','))
        elif args.stage=='stress-test':
            from .stress import stress_test
            result=stress_test(ctx,args.limit)
        elif args.stage=='plots':
            from .reporting import plots
            result=plots(ctx)
        elif args.stage=='prepare-fresh-holdout':
            from .fresh_holdout import prepare
            result=prepare(ctx,args.per_class,args.download_workers)
        elif args.stage=='extract-fresh-holdout':
            from .fresh_holdout import extract
            result=extract(ctx,args.encoder_batch_size)
        elif args.stage=='evaluate-fresh-holdout':
            from .fresh_holdout import evaluate
            result=evaluate(ctx)
        elif args.stage=='recompute-paired':
            from .evaluation import recompute_paired
            result=recompute_paired(ctx,args.dataset_scope=='fresh-holdout')
        else:
            from .audit import audit
            from .extraction import extract
            from .features import build_bank,generate
            from .training import smoke_test,train
            from .evaluation import evaluate
            from .stress import stress_test
            from .reporting import plots
            from dataclasses import replace
            audit(ctx,False,False)
            small=replace(ctx,smoke=True,bank_size=64)
            extract(small,None,args.encoder_batch_size)
            smoke_test(small)
            extract(ctx,None,args.encoder_batch_size)
            build_bank(ctx); generate(ctx); train(ctx); evaluate(ctx); stress_test(ctx,500)
            result=plots(ctx)
            audit(ctx,False,True)
        record['status']='completed'; record['result']=result
        print(json.dumps(result,indent=2,default=str),flush=True)
    except Exception as exc:
        record['status']='failed';record['error']=f'{type(exc).__name__}: {exc}';record['traceback']=traceback.format_exc()
        raise
    finally:
        record['seconds']=time.perf_counter()-started; record.update(peak_memory())
        with (logdir/'commands.jsonl').open('a',encoding='utf-8') as f: f.write(json.dumps(record,default=str)+'\n')
