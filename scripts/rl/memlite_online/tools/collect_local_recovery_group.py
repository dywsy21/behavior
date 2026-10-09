"""Reuse scene initialization for one task, never policy memory or instances.

Each case retains separate source, complete snapshots, RGB, clock and outcome
receipts. Unexpected simulator errors stop the group; there is no blind retry.
"""
import argparse
import json
from pathlib import Path
import sys
import time

from collect_local_grasp_recovery import main as collect_case
from common import atomic_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sources',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--cases',nargs='+',required=True);a=p.parse_args()
    tasks={json.loads((a.sources/c/'manifest.json').read_text())['task'] for c in a.cases}
    if len(tasks)!=1:raise ValueError('Warm scene group must have one exact task')
    a.output.mkdir(parents=True,exist_ok=True)
    identity=next(iter(tasks));path=a.output/(identity+'-group.json')
    if path.exists() or any((a.output/c).exists() for c in a.cases):raise FileExistsError('Existing group / case evidence')
    session=[];results=[];started=time.monotonic()
    try:
        for case in a.cases:
            atomic_json(path,dict(status='running',current_case=case,finished_cases=results,cases=a.cases,quota=None))
            collect_case(['--proposal',str(a.sources/case),'--output',str(a.output/case)],shared_session=session)
            results.append(dict(case=case,status=json.loads((a.output/case/'status.json').read_text())['status']))
        atomic_json(path,dict(status='finished',finished_cases=results,cases=a.cases,seconds=time.monotonic()-started,
                             new_policy_ledger_every_branch=True,quota=None))
    except BaseException as error:
        atomic_json(path,dict(status='failed',finished_cases=results,cases=a.cases,error=repr(error)))
        raise
    finally:
        if session:session[0].__exit__(None,None,None)


if __name__=='__main__':main()
