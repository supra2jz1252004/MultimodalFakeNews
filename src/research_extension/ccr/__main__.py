from pathlib import Path
import argparse,json
from .core import *

def main():
    parser=argparse.ArgumentParser(description='Additive CCR experiment; historical files are read-only.')
    parser.add_argument('stage',choices=['smoke','train','freeze','evaluate','stress','report','access-status','verify-preservation','verite-local'])
    parser.add_argument('--project-root',type=Path,default=Path(__file__).resolve().parents[3]);parser.add_argument('--authorized-dir',type=Path)
    args=parser.parse_args();s=Study(args.project_root,args.stage=='smoke')
    if args.stage in ['smoke','train','freeze']:
        from .fit import oof,fit_gates
        if args.stage=='smoke':result={'oof':oof(s),'gates':fit_gates(s)}
        elif args.stage=='train':result=oof(s)
        else:result=fit_gates(s)
    elif args.stage=='evaluate':
        from .evaluate import evaluate
        result=evaluate(s)
    elif args.stage=='stress':
        from .evaluate import stress
        result=stress(s)
    elif args.stage=='report':
        from .report import report
        result=report(s)
    elif args.stage=='access-status':
        from .external import blocked
        result=blocked(s)
    elif args.stage=='verite-local':
        from .external import evaluate_local
        require(args.authorized_dir is not None,'An authorized local directory is required; no gated download is attempted')
        result=evaluate_local(s,args.authorized_dir)
    else:
        from concurrent.futures import ThreadPoolExecutor
        inv=read_json(s.project/'results/research_extension/ccr/historical_inventory.json')
        def check(item):
            rel,rec=item;p=s.project/rel
            return None if p.exists() and p.stat().st_size==rec['bytes'] and sha(p)==rec['sha256'] else rel
        with ThreadPoolExecutor(max_workers=4) as executor:changed=[x for x in executor.map(check,inv['files'].items()) if x]
        result={'status':'passed' if not changed else 'failed','files_checked':len(inv['files']),'changed':changed,'checked_utc':now()}
        write_json(s.project/'results/research_extension/ccr/preservation_check.json',result)
        require(not changed,'Historical artifact preservation failed')
    print(json.dumps(result,indent=2,default=str),flush=True)
if __name__=='__main__':main()
