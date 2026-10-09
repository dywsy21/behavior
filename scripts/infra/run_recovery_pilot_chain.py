"""Sequential, fail-closed H0 -> L0 -> H1 pilot after explicit data acceptance.

Each child has its own pinned ticket, independent W&B run and cumulative wall
supervisor. This never launches RL, simulation evaluation, or data collection.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

REPO=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(REPO/'src'))
from g05.utils.training.stage1_runtime import atomic_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('recipe','corpus','preparation','output'):p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    completed={};current=None
    def run(command):
        subprocess.run([sys.executable,*map(str,command)],cwd=REPO,check=True)
    def record(status,**extra):
        atomic_json(a.output/'chain.json',dict(status=status,current_component=current,
            completed=completed,updated_unix=time.time(),source_commit=subprocess.check_output(
                ['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),**extra))
    try:
        for component in ('H0','L0','H1'):
            current=component;record('PREPARING_TICKET')
            ticket=a.output/(component+'-ticket.json')
            command=['scripts/rl/memlite_online/tools/make_recovery_pilot_ticket.py',
                '--recipe',a.recipe,'--corpus',a.corpus,'--processor-audit',a.preparation/'processor.json',
                '--component',component,'--output',ticket]
            if component=='H0': command+=['--feature-cache',a.preparation/'feature-cache','--feature-audit',a.preparation/'feature-audit.json']
            if component=='H1': command+=['--feedback-directory',completed['H0']['artifacts_directory']]
            run(command);record('RUNNING')
            run(['scripts/infra/launch_memlite_recovery.py','--ticket',ticket,'--component',component,'--output',a.output/component])
            result=json.loads((a.output/component/'result.json').read_text())
            expected='trained_calibration_and_oof_completed' if component=='H0' else 'completed_finite_schedule'
            if result['status']!=expected:raise RuntimeError('Component paused or incomplete; do not silently continue')
            completed[component]=result;record('COMPONENT_COMPLETED')
        record('COMPLETED_NO_RUNTIME_DEPLOYMENT')
    except BaseException as exc:
        record('STOPPED_ERROR',error=f'{type(exc).__name__}: {exc}')
        raise


if __name__=='__main__':main()
