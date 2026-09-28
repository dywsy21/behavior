"""Select one saved checkpoint on fixed held-out FM loss, then test two scenes.

No training. Reuses the completed 40k/100k comparisons; never changes old runs.
"""
from __future__ import annotations

import argparse
import asyncio
from collections import Counter
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import traceback

import eval_g05_100k as core

ROOT = Path('/mnt/nvme_tmp/robodojo_g05_best_eval_20260928/v1')
PREVIOUS = Path('/mnt/nvme_tmp/robodojo_g05_100k_eval_20260928/v4')
STEPS = tuple(range(10000, 100001, 10000))
REUSE = {
    40000: '1ec9db7258d03ea3d84bd8cca079b76938547c0b3574f5922519fc2d5ee405b6',
    100000: 'a17091ad803b68b77744674a0ae483f82409287e08cd0e230164982373113464',
}
SIM_TASKS = (0, 1)
WALL_SECONDS = 4500
SELECTION_SECONDS = 1200


def configure(output=ROOT):
    core.OUT = output
    core.RUNTIME = Path('/mnt/nvme_tmp/robodojo_sim_runtime_20260925/eval_g05_best_20260928_v1')
    core.PORT = 8932
    core.MODEL_CALL_LIMIT = sum((core.LIMITS[t]+15)//16 for t in SIM_TASKS)


def folder(step):
    if step not in STEPS: raise ValueError('Unregistered checkpoint')
    return PREVIOUS if step in REUSE else ROOT/'candidates'/str(step)


def locators(rows):
    return [(r['task'], r['episode'], r['index'], r['fraction']) for r in rows]


def ranked(rows):
    if not rows or len({r['step'] for r in rows}) != len(rows):
        raise ValueError('Missing or duplicate candidate')
    if any(not math.isfinite(r[k]) or r[k] < 0 for r in rows for k in ('fm_loss', 'ce_loss')):
        raise ValueError('Invalid candidate loss')
    # Primary objective is the deployed FM head, not an arbitrary CE+FM sum.
    return sorted(rows, key=lambda r:(r['fm_loss'],r['ce_loss'],r['step']))


def read_candidate(step):
    base = folder(step)
    result = json.loads((base/f'offline_{step}.json').read_text())
    trace = base/f'offline_{step}.jsonl'
    rows = [json.loads(line) for line in trace.read_text().splitlines()]
    digest = core.sha(trace)
    if digest != result['fixed_window_rows_sha256'] or (step in REUSE and digest != REUSE[step]):
        raise ValueError('Candidate rows changed')
    if result['step'] != step or result['samples'] != 100 or len(rows) != 100:
        raise ValueError('Incomplete fixed-window candidate')
    if len({r['episode'] for r in rows}) != 50 or sorted(Counter(r['task'] for r in rows).values()) != [20]*5:
        raise ValueError('Changed original held-out coverage')
    for key in ('fm_loss','ce_loss'):
        mean=sum(row['losses'][key] for row in rows)/len(rows)
        if not math.isfinite(mean) or not math.isclose(mean,result['mean'][key],rel_tol=1e-12,abs_tol=1e-12):
            raise ValueError('Candidate summary does not match its raw losses')
    equivalent = json.loads((base/f'fm_equivalence_{step}.json').read_text())
    if equivalent != {'max_abs_difference':0.0, 'identical':True}:
        raise ValueError('FM-only equivalence gate failed')
    receipt = json.loads((base/f'load_{step}.json').read_text())
    if receipt['checkpoint'] != str(core.checkpoint(step)) or receipt['checkpoint_bytes'] != core.checkpoint(step).stat().st_size:
        raise ValueError('Checkpoint identity changed')
    return result, rows, receipt, digest


def choose_best():
    reference, ref_rows, ref_receipt, _ = read_candidate(40000)
    ranking = []
    for step in STEPS:
        result, rows, receipt, digest = read_candidate(step)
        if locators(rows) != locators(ref_rows): raise ValueError('Non-paired evaluation windows')
        if any(receipt[k] != ref_receipt[k] for k in ('config_sha256','stats_sha256','action_tokenizer_sha256','dtype')):
            raise ValueError('Training preprocessing/precision differs between candidates')
        ranking.append(dict(step=step, checkpoint=str(core.checkpoint(step)),
            fm_loss=result['mean']['fm_loss'], ce_loss=result['mean']['ce_loss'],
            per_task=result['per_task'], rows_sha256=digest, source_commit=result['source_commit'],
            reused=step in REUSE, evidence=str(folder(step))))
    ranking = ranked(ranking)
    selection = dict(selected_step=ranking[0]['step'], primary_metric='fixed_100_window_fm_loss',
        tie_breaks=['ce_loss','earlier_step'], candidates=ranking, paired_locators=True,
        unique_heldout_episodes=50, windows_per_task=20, seed_scheme=reference['seed_scheme'],
        new_windows=800, reused_windows=200, full_validation=False,
        selection_split='original_task_stratified_5_percent_development_validation',
        source_commit=core.commit(), training_updates=0)
    core.save(ROOT/'selection.json',selection)
    return selection


def spawn(children, mode, log_name, *, step=None, task=None, gpu=0):
    command=[sys.executable,str(Path(__file__).resolve()),mode]
    if step is not None: command += ['--step',str(step)]
    if task is not None:
        command[0]=core.SIM_PY
        command += ['--task',str(task)]
        env=core.simulation_env(task)
    else:
        env=core.model_env(gpu)
    with (ROOT/log_name).open('x') as log:
        process=subprocess.Popen(command,env=env,cwd=core.REPO,stdin=subprocess.DEVNULL,
            stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    children.append(process)
    core.save(ROOT/(log_name+'.launch.json'),dict(pid=process.pid,mode=mode,step=step,task=task,gpu=gpu))
    return process


def supervise():
    configure()
    children=[]
    start=time.monotonic()
    state=dict(status='running',source_commit=core.commit(),training_updates=0,
               started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
    try:
        # Validate existing evidence before consuming any new GPU work.
        for step in REUSE: read_candidate(step)
        pending=[s for s in STEPS if s not in REUSE]
        active={}
        while pending or active:
            if time.monotonic()-start > SELECTION_SECONDS: raise TimeoutError('Selection budget exhausted')
            for gpu in range(4):
                if gpu not in active and pending:
                    step=pending.pop(0)
                    active[gpu]=(step,spawn(children,'candidate',f'candidate_{step}.log',step=step,gpu=gpu))
            for gpu,(step,process) in list(active.items()):
                code=process.poll()
                if code is not None:
                    if code: raise RuntimeError('Candidate evaluation failed: '+str(step))
                    read_candidate(step)
                    del active[gpu]
            time.sleep(2)
        selection=choose_best()
        state['selected_step']=selection['selected_step']
        actor=spawn(children,'serve-selected','serve_selected.log',gpu=0)
        ready_deadline=min(start+WALL_SECONDS,time.monotonic()+600)
        while not (ROOT/'ready.json').exists():
            if actor.poll() is not None: raise RuntimeError('Selected actor failed')
            if time.monotonic()>ready_deadline: raise TimeoutError('Actor load deadline')
            time.sleep(2)
        for task in SIM_TASKS:
            remaining=WALL_SECONDS-(time.monotonic()-start)
            if remaining<60: raise TimeoutError('Total 75-minute budget exhausted')
            process=spawn(children,'simulate',f'sim_{task}.log',task=task,gpu=3)
            try: code=process.wait(timeout=min(2400,remaining))
            except subprocess.TimeoutExpired:
                core.stop_owned(process); raise
            if code or not (ROOT/f'task_{task}/result.json').exists():
                raise RuntimeError('Selected simulator did not complete: '+str(task))
        state['status']='completed'
    except BaseException as error:
        state.update(status='failed',error=repr(error),traceback=traceback.format_exc())
        raise
    finally:
        for process in reversed(children): core.stop_owned(process)
        state.update(seconds=time.monotonic()-start,owned_returncodes={str(p.pid):p.returncode for p in children})
        core.save(ROOT/'supervisor.json',state)


def launch():
    configure()
    revision=core.commit()
    used=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True)
    if len(used.splitlines())!=4 or any(int(x)>512 for x in used.splitlines()):
        raise RuntimeError('Expected idle GPUs after stopping the superseded run')
    with socket.socket() as connection: connection.bind(('127.0.0.1',core.PORT))
    subprocess.run([core.MODEL_PY,'-c','from torchcodec.decoders import VideoDecoder'],
                   env=core.model_env(0),check=True,timeout=60)
    for step in STEPS:
        if not core.checkpoint(step).is_file(): raise FileNotFoundError(core.checkpoint(step))
    for step in REUSE: read_candidate(step)
    ROOT.mkdir(parents=True,exist_ok=False)
    core.save(ROOT/'launch.json',dict(owner='Codex/EVAL-G05-BEST',source_commit=revision,
        steps=STEPS,reused_steps=list(REUSE),new_fixed_window_evaluations=800,
        primary_metric='fm_loss',selection_seconds=SELECTION_SECONDS,wall_seconds=WALL_SECONDS,
        simulator_tasks=SIM_TASKS,control_limits=[core.LIMITS[t] for t in SIM_TASKS],
        max_model_calls=core.MODEL_CALL_LIMIT,training_updates=0,automatic_retry=False))
    with (ROOT/'supervisor.log').open('x') as log:
        process=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'supervise'],
            env=core.model_env(0),cwd=core.REPO,stdin=subprocess.DEVNULL,stdout=log,
            stderr=subprocess.STDOUT,start_new_session=True)
    core.save(ROOT/'supervisor_launch.json',dict(pid=process.pid))
    print(json.dumps({'output':str(ROOT),'pid':process.pid}),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('launch','supervise','candidate','serve-selected','simulate'))
    parser.add_argument('--step',type=int,choices=STEPS)
    parser.add_argument('--task',type=int,choices=SIM_TASKS)
    args=parser.parse_args()
    configure()
    if args.mode=='candidate':
        if args.step is None or args.step in REUSE: raise ValueError('Do not recompute completed candidates')
        directory=folder(args.step);directory.mkdir(parents=True,exist_ok=False)
        configure(directory)
        cfg,policy,processor=core.load_model(args.step)
        core.offline(args.step,cfg,policy,processor)
    elif args.mode=='serve-selected':
        selection=json.loads((ROOT/'selection.json').read_text())
        step=selection['selected_step']
        read_candidate(step)
        cfg,policy,processor=core.load_model(step)
        asyncio.run(core.serve(cfg,policy,processor,step=step))
    elif args.mode=='simulate':
        if args.task is None: raise ValueError('Task required')
        core.simulate(args.task)
    elif args.mode=='supervise': supervise()
    else: launch()


if __name__=='__main__':main()
