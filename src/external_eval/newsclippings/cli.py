"""Staged CPU evaluation with explicit provenance modes and no fitting stages."""
import argparse,json
from pathlib import Path
from .io_utils import Run,read,preservation,require

def main():
    parser=argparse.ArgumentParser(description='Independent NewsCLIPpings correspondence. Pairing status is not factual veracity. Additive immutable artifacts; no training.')
    parser.add_argument('stage',choices=['preflight','prepare','smoke','extract','evaluate-alignment','evaluate-transfer','statistics','plots','report','verify-preservation'])
    parser.add_argument('--project',type=Path,default=Path(__file__).resolve().parents[3])
    parser.add_argument('--mode',choices=['official','pinned'],default='official',help='Official: separate score-only embeddings. Pinned: authorized image encoder pipeline.')
    parser.add_argument('--visualnews-root',type=Path,help='Authorized local VisualNews origin folder containing official data.json and images')
    args=parser.parse_args();run=Run(args.project)
    if args.stage=='preflight':
        from .preflight import preflight
        result=preflight(run)
    elif args.stage=='prepare':
        from .data import prepare
        result=prepare(run)
    elif args.stage=='smoke':
        from .acquire import smoke
        result=smoke(run)
    elif args.stage=='extract':
        if args.mode=='official':
            from .acquire import official_features
            result=official_features(run)
        else:
            from .evaluation import extract_local
            require(args.visualnews_root,'Pinned extraction needs --visualnews-root with authorized real files')
            result=extract_local(run,args.visualnews_root)
    elif args.stage=='evaluate-alignment':
        from .evaluation import alignment
        result=alignment(run,args.mode)
    elif args.stage=='evaluate-transfer':
        from .evaluation import transfer
        result=transfer(run,args.visualnews_root)
    elif args.stage=='statistics':
        from .statistics import analyze
        result=analyze(run,args.mode)
    elif args.stage in ['plots','report']:
        from .reporting import plots,report
        result=(plots if args.stage=='plots' else report)(run)
    else:result=preservation(run)
    print(json.dumps(result,indent=2,default=str))
