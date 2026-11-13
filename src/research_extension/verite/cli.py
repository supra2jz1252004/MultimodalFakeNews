import argparse,json
from pathlib import Path
from .core import Study,preservation
def main():
    parser=argparse.ArgumentParser(description='Authorized local VERITE frozen-model transfer and correspondence. No training or factual-checker claim.')
    parser.add_argument('stage',choices=['audit','lock','extract','evaluate','statistics','plots','report','verify-preservation'])
    parser.add_argument('--project',type=Path,default=Path(__file__).resolve().parents[3]);args=parser.parse_args();s=Study(args.project)
    if args.stage in ['audit','lock']:
        from .audit import audit,protocol
        result=(audit if args.stage=='audit' else protocol)(s)
    elif args.stage=='extract':
        from .features import extract
        result=extract(s)
    elif args.stage=='evaluate':
        from .models import evaluate
        result=evaluate(s)
    elif args.stage=='statistics':
        from .statistics import analyze
        result=analyze(s)
    elif args.stage in ['plots','report']:
        from .report import plots,report
        result=(plots if args.stage=='plots' else report)(s)
    else:result=preservation(s)
    print(json.dumps(result,indent=2,default=str))
