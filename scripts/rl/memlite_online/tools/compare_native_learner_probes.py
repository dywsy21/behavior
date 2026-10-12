"""Compare the entire registered parent/control/candidate native probe arms."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

REPO=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(REPO/'scripts/rl/memlite_online/code'))
from recovery_corpus import file_sha
from recovery_native_probe_comparison import compare_native_probes, ARMS


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs',type=Path,nargs=3,required=True,help='parent/control/candidate service receipts')
    p.add_argument('--audits',type=Path,nargs=3,required=True,help='same-order full independent physical audits')
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists() or subprocess.check_output(['git','status','--porcelain'],cwd=REPO,text=True).strip():
        raise ValueError('Fresh output and clean frozen analysis source required')
    configs=[REPO/f'configs/recovery_sft/a800_native_learner_{arm}_probes_heartbeat_v2.json' for arm in ARMS]
    read=lambda path:json.loads(path.read_text())
    result=compare_native_probes([read(p) for p in configs],[read(p/'result.json') for p in a.runs],
        [read(p) for p in a.audits],[[json.loads(x) for x in (p/'episodes.jsonl').read_text().splitlines()] for p in a.runs],
        [file_sha(p) for p in configs],repo=REPO)
    result['source_commit']=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()
    result['inputs']={arm:{name:dict(path=str(path.resolve()),sha256=file_sha(path)) for name,path in
        [('config',configs[i]),('service',a.runs[i]/'result.json'),('episodes',a.runs[i]/'episodes.jsonl'),('audit',a.audits[i])]}
        for i,arm in enumerate(ARMS)}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x') as stream:stream.write(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('pairs','inputs')}))


if __name__=='__main__':main()
