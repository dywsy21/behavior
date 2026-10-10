"""Run an explicit source inventory on idle RTX GPUs; retain every attempt.

No training, arbitrary wall-clock/reset quota, or automatic deletion. The
inventory is a coverage batch, not a global collection limit. Native failures
are recorded and do not silently turn into successful samples or retries.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
import json
import os
from pathlib import Path
import queue
import shutil
import signal
import subprocess
import sys
import threading
import time

REPO=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(REPO/'scripts/eval/memlite_sft100'))
sys.path.insert(0,str(REPO/'scripts/rl/memlite_online/code'))
from common import atomic_json,sha256
from recovery_gpu_ownership import owns_auxiliary,owns_short_skill_auxiliary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sources',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--gpus',nargs='+',type=int,required=True)
    p.add_argument('--previous-collection',type=Path,action='append',default=[])
    p.add_argument('--peer-collection',type=Path,action='append',default=[])
    p.add_argument('--short-skill-peer-prefix',type=Path,action='append',default=[],
        help='Exact sibling episode-directory prefix for an owned running short-skill collector')
    p.add_argument('--short-skill-peer-config',type=Path)
    p.add_argument('--short-skill-peer-port',type=int)
    p.add_argument('--skip-case',action='append',default=[])
    p.add_argument('--skip-reason')
    p.add_argument('--priority-task',nargs='*',default=[])
    p.add_argument('--diversify-fault-timing',action='store_true')
    p.add_argument('--reference-category-binding',action='store_true')
    p.add_argument('--reference-scene-inventory',action='store_true')
    p.add_argument('--concurrent-loads',type=int,default=2)
    p.add_argument('--warm-groups',action='store_true');a=p.parse_args()
    if a.warm_groups:
        raise ValueError('Warm groups quarantined: non-TRO scene joints can survive instance load. Use independent cold instances.')
    if a.reference_scene_inventory and not a.reference_category_binding:
        raise ValueError('Scene inventory requires the original physical reference binding gate')
    if not 1<=a.concurrent_loads<=len(a.gpus):raise ValueError('Invalid scene loading concurrency')
    if a.output.exists():raise FileExistsError(a.output)
    if len(set(a.gpus))!=len(a.gpus):raise ValueError('Duplicate GPU')
    if a.skip_case and not a.skip_reason:raise ValueError('Explicit skipped-source reason required')
    if subprocess.check_output(['git','status','--porcelain'],cwd=REPO,text=True).strip():raise ValueError('Dirty source')
    short_peer_binding=None
    if a.short_skill_peer_prefix:
        if (a.short_skill_peer_config is None or a.short_skill_peer_port is None
                or not 1<=a.short_skill_peer_port<=65535):
            raise ValueError('Short-skill peers require their exact config and port')
        peer_config=json.loads(a.short_skill_peer_config.read_text())
        if peer_config.get('schema')!='short_skill_rl_a800_v1':raise ValueError('Unknown peer recipe')
        short_peer_binding=dict(config_sha256=sha256(a.short_skill_peer_config),
            cases=set(peer_config['cases']),port=a.short_skill_peer_port)
        if any(not p.name.endswith('-episode') or any(c in p.name for c in '*?[]') for p in a.short_skill_peer_prefix):
            raise ValueError('Use explicit episode prefix paths, never arbitrary globs')
    elif a.short_skill_peer_config is not None or a.short_skill_peer_port is not None:
        raise ValueError('Peer recipe/port must bind explicit episode prefixes')
    for gpu in a.gpus:
        running=subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-compute-apps=pid,used_gpu_memory',
            '--format=csv,noheader,nounits'],text=True).strip()
        for line in running.splitlines():
            pid,memory=line.split(',');pid=int(pid);memory=float(memory);allowed=False
            for peer in a.peer_collection:
                for path in peer.glob('*/status.json'):
                    receipt=json.loads(path.read_text())
                    if receipt.get('pid')!=pid:continue
                    try:
                        argv=Path(f'/proc/{pid}/cmdline').read_bytes().decode().split('\0')
                        environ=dict(v.split('=',1) for v in Path(f'/proc/{pid}/environ').read_bytes().decode().split('\0') if '=' in v)
                    except FileNotFoundError:continue
                    allowed=allowed or owns_auxiliary(pid,memory,gpu,receipt,path.parent,argv,environ)
            for prefix in a.short_skill_peer_prefix:
                for path in prefix.parent.glob(prefix.name+'-*/status.json'):
                    receipt=json.loads(path.read_text())
                    if receipt.get('pid')!=pid:continue
                    try:
                        argv=Path(f'/proc/{pid}/cmdline').read_bytes().decode().split('\0')
                        environ=dict(v.split('=',1) for v in Path(f'/proc/{pid}/environ').read_bytes().decode().split('\0') if '=' in v)
                    except FileNotFoundError:continue
                    allowed=allowed or owns_short_skill_auxiliary(pid,memory,gpu,receipt,path.parent,
                        argv,environ,**short_peer_binding)
            if not allowed:raise RuntimeError(f'GPU {gpu} has unowned/non-auxiliary PID {pid}; do not stop it')
    inventory=json.loads((a.sources/'manifest.json').read_text())
    if inventory.get('status','complete')!='complete':raise ValueError('Source export is not complete')
    sources=inventory['cases']
    if not set(a.skip_case)<={r['directory'] for r in sources}:raise ValueError('Unknown skipped source')
    for case in sources:
        if sha256(a.sources/case['directory']/'manifest.json')!=case['manifest_sha256']:raise ValueError('Changed inventory')
    a.output.mkdir(parents=True);jobs=queue.Queue();lock=threading.Lock();loading=threading.Semaphore(a.concurrent_loads)
    stopping=threading.Event()
    # Stop scheduling on a user/operator stop, but allow active evidence writers
    # to finish their current source. No SIGINT-triggered launch of more jobs.
    for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,lambda *_:stopping.set())
    def should_stop():
        if (a.output/'STOP').exists() or shutil.disk_usage(a.output).free<100*2**30:stopping.set()
        return stopping.is_set()
    rows=[];pending=[]
    for case in sources:
        if case['directory'] in a.skip_case:
            rows.append(dict(case=case['directory'],status='explicitly_skipped',reason=a.skip_reason));continue
        priors=[root/case['directory'] for root in a.previous_collection
                if (root/case['directory']/'result.json').exists() or (root/case['directory']/'status.json').exists()]
        if len(priors)>1:raise ValueError('Multiple original attempts; cannot silently select a lucky retry')
        previous=priors[0] if priors else None
        prior_receipt=(previous/'result.json' if previous is not None and (previous/'result.json').exists()
                       else previous/'status.json' if previous is not None and (previous/'status.json').exists() else None)
        if prior_receipt is not None:
            result=json.loads(prior_receipt.read_text())
            if result['proposal_sha256']!=case['manifest_sha256']:raise ValueError('Reused case source changed')
            rows.append(dict(case=case['directory'],reused_directory=str(previous),
                             receipt_sha256=sha256(prior_receipt),status='retained_original_attempt',original_status=result['status']))
        else:pending.append(case)
    priority={task:i for i,task in enumerate(a.priority_task)}
    pending.sort(key=lambda case:priority.get(json.loads((a.sources/case['directory']/'manifest.json').read_text())['task'],len(priority)))
    atomic_json(a.output/'reused_cases.json',rows)
    if a.warm_groups:
        groups=defaultdict(list)
        for case in pending:
            task=json.loads((a.sources/case['directory']/'manifest.json').read_text())['task']
            groups[task].append(case['directory'])
        for task,cases in groups.items():jobs.put(dict(directory=task,cases=cases))
    else:
        for case in pending:jobs.put(case)
    def worker(gpu):
        while True:
            if should_stop():return
            try:case=jobs.get_nowait()
            except queue.Empty:return
            name=case['directory'];out=a.output/(case['cases'][0] if a.warm_groups else name)
            log=(a.output/(name+'.log')).open('x')
            env=dict(os.environ,EVAL_GPU=str(gpu),EVAL_SOURCE=str(REPO/'scripts/eval/memlite_sft100'))
            loading.acquire();released=False
            try:
                if should_stop():
                    jobs.put(case);return
                command=([str(REPO/'scripts/rl/memlite_online/tools/collect_local_recovery_group.py'),
                    '--sources',str(a.sources),'--output',str(a.output),'--cases',*case['cases']] if a.warm_groups else
                    [str(REPO/'scripts/rl/memlite_online/tools/collect_local_grasp_recovery.py'),
                     '--proposal',str(a.sources/name),'--output',str(out)])
                proposal=json.loads((a.sources/name/'manifest.json').read_text())
                if proposal['schema']=='recovery_expert_skill_proposal_v1':
                    if a.reference_scene_inventory:raise ValueError('New scene-inventory option is GRASP-only')
                    if proposal['skill_verb'] in ('PLACE_IN','PLACE_ON'):
                        command[0]=str(REPO/'scripts/rl/memlite_online/tools/collect_placement_curriculum.py')
                        command.extend(['--manifest-sha256',case['manifest_sha256']])
                        if a.reference_category_binding:
                            raise ValueError('Placement requires exact original bindings; category inference not validated')
                    else:
                        command[0]=str(REPO/'scripts/rl/memlite_online/tools/collect_local_articulation_recovery.py')
                        if a.reference_category_binding:command.append('--reference-category-binding')
                else:
                    if a.diversify_fault_timing:command.append('--diversify-fault-timing')
                    if a.reference_category_binding:command.append('--reference-category-binding')
                    if a.reference_scene_inventory:command.append('--reference-scene-inventory')
                proc=subprocess.Popen(['bash',str(REPO/'scripts/eval/memlite_sft100/launch_sim.sh'),*command],stdout=log,stderr=subprocess.STDOUT,
                    stdin=subprocess.DEVNULL,env=env,cwd=REPO)
                while proc.poll() is None:
                    if not released and (out/'status.json').exists():
                        state=json.loads((out/'status.json').read_text())
                        if state['status']!='loading':loading.release();released=True
                    time.sleep(1)
                receipt=a.output/(name+'-group.json') if a.warm_groups else out/'status.json'
                state=json.loads(receipt.read_text()) if receipt.exists() else None
                result=out/'result.json'
                row=dict(case=name,gpu=gpu,pid=proc.pid,returncode=proc.returncode,status=state,
                    closed_result_sha256=sha256(result) if result.exists() else None,
                    evidence_state='closed_attempt' if result.exists() else 'incomplete_or_failed_not_approved')
                with lock:
                    rows.append(row);atomic_json(a.output/'status.json',dict(status='collecting',completed=len(rows),
                        inventory_cases=len(sources),quota=None,results=rows))
            finally:
                if not released:loading.release()
                log.close();jobs.task_done()
    with ThreadPoolExecutor(max_workers=len(a.gpus)) as pool:list(pool.map(worker,a.gpus))
    done_status='inventory_finished_with_explicit_skips' if a.skip_case else 'inventory_attempts_finished'
    atomic_json(a.output/'result.json',dict(status='stopped_with_pending_sources' if not jobs.empty() else done_status,results=rows,
        pending_sources=jobs.qsize(),
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
        source_inventory_sha256=sha256(a.sources/'manifest.json'),training_approved=False,quota=None))


if __name__=='__main__':main()
