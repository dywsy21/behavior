"""Sequence the two explicitly authorized public runs without overlapping GPUs.

This is a job dependency, not a new evaluation budget or an automatic retry.
Unclassified failures stop the sequence for diagnosis. An explicitly registered
continuation can finish with disclosed execution errors, never fabricated scores.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dependency_ready(state,summary,*,continuation=False):
    if state.get('status') in ('needs_diagnosis','failed_no_retry','failed'):
        raise RuntimeError('MEM evaluation needs diagnosis; no automatic reroll or GPU takeover')
    if continuation:
        if state.get('status') not in ('completed','completed_with_execution_errors'):return False
        counts=[summary.get(k) for k in ('completed','execution_errors','coverage','unaccounted','missing')]
        if (any(type(n) is not int or n<0 for n in counts) or counts[0]+counts[1]!=1000
                or counts[2]!=1000 or counts[3]!=0 or counts[4]!=counts[1]
                or summary.get('status')!=state['status'] or summary.get('policy_changed') is not False
                or summary.get('optimizer_updates')!=0
                or (counts[1]>0 and (state['status']!='completed_with_execution_errors'
                    or summary.get('official_full_q_score') is not None or summary.get('official_full_sr') is not None))
                or (counts[1]==0 and state['status']!='completed')):
            raise ValueError('Continuation must account for every case and explicitly disclose missing official scores')
        return True
    if state.get('status')!='completed':return False
    if summary.get('status')!='complete' or summary.get('completed')!=1000 or summary.get('missing')!=0:
        raise ValueError('MEM completion must include all prescribed1000 cases')
    return True


def existing_preflight(job,source):
    """Explicitly validate, never rerun, both completed target-source gates."""
    for n in (2,4):
        path=job/'preflight'/f'batch{n}'/'status.json'
        if not path.is_file() or json.loads(path.read_text()).get('status')!='completed':
            raise ValueError('Both actual preflights must finish before using --completed-preflight')
    code=('import json,sys; sys.path.insert(0,sys.argv[2]+"/scripts/eval"); '
          'import champion_full_eval as c; job,m=c.registration(sys.argv[1]); '
          'print(json.dumps(c.select_preflight(job,m)))')
    result=json.loads(subprocess.check_output([sys.executable,'-c',code,str(job),str(source)],
        cwd=source,env=os.environ|dict(PYTHONDONTWRITEBYTECODE='1'),text=True))
    if set(result['evidence'])!={'2','4'}:raise ValueError('Missing completed 2/4 evidence')
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('mem-job','champion-job','champion-source'):p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--mem-continuation',action='store_true',help='Account for diagnosed execution errors explicitly, not as official scores')
    p.add_argument('--completed-preflight',action='store_true',help='Verify and reuse both already-completed same-source TRAIN gates; do not rerun them')
    a=p.parse_args();mem=a.mem_job.resolve();job=a.champion_job.resolve();source=a.champion_source.resolve()
    mm=json.loads((mem/'manifest.json').read_text());cm=json.loads((job/'manifest.json').read_text())
    mem_schema='memlite_final_explicit_continuation_v1' if a.mem_continuation else 'memlite_final_public100_run_v1'
    if (mm['schema']!=mem_schema or cm['schema']!='champion_final_public100_run_v1'
            or mem==job or (job/'status.json').exists() or (job/'STOP').exists()
            or subprocess.check_output(['git','rev-parse','HEAD'],cwd=source,text=True).strip()!=cm['source_commit']
            or subprocess.check_output(['git','status','--porcelain'],cwd=source,text=True).strip()):
        raise ValueError('Two original registered runs and exact clean champion source required')
    preflight=existing_preflight(job,source) if a.completed_preflight else None
    if preflight is None:
        for n in (2,4):
            if (job/'preflight'/f'batch{n}').exists():raise ValueError('Never silently reuse or reroll prior queue attempts')
    queue=job/'queue';queue.mkdir(exist_ok=False)
    lock=(queue/'lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    bindings={str(x/'manifest.json'):sha(x/'manifest.json') for x in (mem,job)}
    if preflight:
        bindings.update({r['path']:r['sha256'] for r in preflight['evidence'].values()})
    state=dict(status='waiting_for_memlite',pid=os.getpid(),started=time.time(),bindings=bindings,
        mem_job=str(mem),champion_job=str(job),champion_source=str(source),phases=[],automatic_retries=False,
        mem_continuation=a.mem_continuation,existing_preflight=preflight,
        queue_source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=Path(__file__).resolve().parents[2],text=True).strip())
    def publish():
        state['updated']=time.time();tmp=queue/'status.json.tmp'
        with tmp.open('w') as f:json.dump(state,f,indent=2);f.flush();os.fsync(f.fileno())
        os.replace(tmp,queue/'status.json')
    def unchanged():
        if any(sha(p)!=v for p,v in bindings.items()):raise ValueError('Registered run changed while queued')
        if (queue/'STOP').exists() or (job/'STOP').exists():raise InterruptedError('Requested queue stop')
    try:
        publish()
        while True:
            unchanged()
            ms=json.loads((mem/'status.json').read_text());summary_path=mem/'summary.json'
            summary=json.loads(summary_path.read_text()) if summary_path.exists() else {}
            if dependency_ready(ms,summary,continuation=a.mem_continuation):
                state['mem_execution_errors']=summary.get('execution_errors',0);break
            time.sleep(30)
        if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
            raise RuntimeError('GPU owner changed after MEM; no displacement')
        script=source/'scripts/eval/champion_full_eval.py'
        phases=[('preflight_batch2',['worker','--gpu','0','--num-envs','2','--preflight']),
                ('preflight_batch4',['worker','--gpu','0','--num-envs','4','--preflight']),
                ('public1000',['supervise'])]
        if preflight:phases=phases[-1:]
        env=os.environ.copy();env['PYTHONDONTWRITEBYTECODE']='1'
        for name,extra in phases:
            unchanged();argv=[sys.executable,str(script),*extra,'--job',str(job)]
            phase=dict(name=name,started=time.time(),argv=argv,status='running')
            state['phases'].append(phase);state['status']=name;publish()
            with (queue/(name+'.log')).open('x') as log:
                result=subprocess.run(argv,cwd=source,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
            phase.update(returncode=result.returncode,finished=time.time(),status='completed' if result.returncode==0 else 'failed')
            publish()
            if result.returncode:raise RuntimeError('Champion phase failed; preserve evidence: '+name)
        result=json.loads((job/'status.json').read_text())
        if result['status']!='completed':raise ValueError('Missing full champion terminal result')
        state['status']=('both_internal_evaluations_finished_with_disclosed_mem_errors'
            if state.get('mem_execution_errors') else 'both_public_runs_completed_not_submitted')
    except BaseException as error:
        state.update(status='needs_diagnosis',error=repr(error));raise
    finally:
        state['finished']=time.time();publish();lock.close()


if __name__=='__main__':main()
