"""One original-reset VLM episode using the unchanged H75/H77 harness."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import secrets
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request

import launch_h73 as synchronous
from launch_h69 import belongs_to_session, stop_owned_group
from launch_h44 import tree_bytes

gate = synchronous.gate
REPO = Path(__file__).resolve().parents[2]
ROOT = Path('/mnt/nvme_tmp/robodojo_agentic_20260925/h79_synchronous_actor_v1')
RUNTIME = Path('/mnt/nvme_tmp/robodojo_sim_runtime_20260925/h79_synchronous_actor_v1')
VLM_PYTHON = Path('/mnt/sdc1/robodojo/GalaxeaVLA/.venv/bin/python')
MODEL = Path('/mnt/sdc1/robodojo/behavior_dev/semantic_agent_v2_20260917/models/Qwen3.8-27B')
REVISION = '1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0'
DIGEST = '7bad2e9021b6a48dd298e571b213c19e763453b8c593ba871afbc404f0c765f8'
PORT = 8979
MAX_CALLS = 215
WALL_SECONDS = 3600
PREREQUISITES = (
    ('h75_render_batch_v1', 0, 'd1cc900f093ed37d0078f5776652a0344b010c05e6404660a2010cf8d7b24bab'),
    ('h77_task3_gate_v1', 3, '530904075e8f47654ef36399867d25661e64e60ede0843a2a77ffbd0552d7863'),
)


def read(path): return json.loads(path.read_text())


def sha(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''): value.update(block)
    return value.hexdigest()


def write(name, value):
    path = ROOT / name
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w') as stream:
        json.dump(value, stream, indent=2); stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, path)


def configure():
    base = synchronous.configure()
    gate.ROOT, gate.RUNTIME = ROOT, RUNTIME
    base.OUTPUT, base.RUNTIME, base.ENTRYPOINT = ROOT, RUNTIME, Path(__file__).resolve()
    synchronous.ORIGINAL_VALIDATE = validate_actor_basic
    return base


def model_identity():
    specification = REPO / 'configs/semantic_robot/h79_model_identity.json'
    expected = read(specification)
    if expected['path'] != str(MODEL) or expected['revision'] != REVISION:
        raise ValueError('Wrong registered reference model')
    actual = {p.name for p in MODEL.iterdir() if p.name != '.cache'}
    if actual != set(expected['files']): raise ValueError('Unexpected model root entries')
    for name, record in expected['files'].items():
        path = MODEL / name
        if (not path.is_file() or path.is_symlink() or path.stat().st_size != record['bytes'] or
                sha(path) != record['sha256']): raise ValueError('Model file changed: ' + name)
    receipt = read(MODEL / 'download_receipt.json')
    if receipt['state'] != 'complete' or receipt['revision'] != REVISION:
        raise ValueError('Original model download identity changed')
    return {'spec_sha256': sha(specification), **expected}


def check_resources(current, baseline=None, sessions=None, *, released=False):
    uuids = gate.scene.supervisor.GPU_UUIDS
    if set(current) != set(uuids): raise ValueError('Physical GPU set changed')
    sessions = sessions or {}
    for index, uuid in enumerate(uuids):
        row = current[uuid]
        owned = {}
        for kind, sid in sessions.items():
            owned[kind] = [p for p in row['processes'] if belongs_to_session(p['pid'], sid)]
            cap = (65536 if index == 2 else 0) if kind == 'model' else (24576 if index == 3 else 512)
            if sum(p['used_mib'] for p in owned[kind]) > cap:
                raise RuntimeError('Own ' + kind + ' allocation cap')
            if released and owned[kind]: raise RuntimeError('Owned GPU session not released')
        own_ids = {p['pid'] for group in owned.values() for p in group}
        if index in (2, 3):
            if any(p['pid'] not in own_ids for p in row['processes']):
                raise RuntimeError('Allocated GPU has an unknown process; stop only ours')
            if baseline is not None and row['used_mib'] - baseline[uuid]['used_mib'] > (66048 if index == 2 else 24576):
                raise RuntimeError('Allocated GPU total delta cap')
        required = (73728 if index == 2 else 32768 if index == 3 else 8192) if baseline is None else 8192
        if row['free_mib'] < required: raise RuntimeError('GPU shared headroom')


def preflight(base):
    code = gate.identity(base)
    from run_v2 import implementation_digest
    if implementation_digest() != DIGEST: raise ValueError('H75/H77 harness implementation changed')
    for name, task, digest in PREREQUISITES:
        folder = ROOT.parent / name
        result, ended = read(folder / 'gate/result.json'), read(folder / 'supervisor.json')
        if (sha(folder / 'gate/result.json') != digest or result['task'] != task or
                result['implementation_digest'] != DIGEST or result['gate_ok'] is not True or
                ended['status'] != 'completed' or not ended.get('after_exit')):
            raise ValueError('Original two-pose prerequisites changed')
    previous = read(Path('/mnt/nvme_tmp/robodojo_vlm_visual_20260925/h78_presence_v1/supervisor.json'))
    if previous.get('status') not in ('completed', 'failed') or not previous.get('after_exit'):
        raise RuntimeError('H78 still running; never overlap its exclusive GPU2')
    before = base.snapshot(); check_resources(before)
    return code, before


def commands(base):
    # The directory name "gate" is retained only to reuse the unchanged native
    # journal validator; --mode agent and the manifest determine the experiment.
    actor = gate.command(base)
    for key, value in {'--mode':'agent', '--max-decisions':'96', '--max-controls':'3072',
                       '--max-seconds':'2400'}.items(): actor[actor.index(key)+1] = value
    actor += ['--uri', f'http://127.0.0.1:{PORT}', '--expected-revision', REVISION]
    for name, _, _ in PREREQUISITES:
        actor += ['--gate-result', str(ROOT.parent / name / 'gate/result.json')]
    model = [str(VLM_PYTHON), str(REPO / 'scripts/semantic_robot/serve_v2.py'),
             '--model', str(MODEL), '--revision', REVISION, '--output', str(ROOT / 'server'),
             '--port', str(PORT), '--max-calls', str(MAX_CALLS), '--structured-planning']
    return {'model': model, 'actor': actor}


def expected_actor_args(base):
    value = gate.expected_args(base)
    value.update(mode='agent', max_decisions=96, max_controls=3072, max_seconds=2400,
                 uri=f'http://127.0.0.1:{PORT}', expected_revision=REVISION,
                 gate_result=[str(ROOT.parent / n / 'gate/result.json') for n, _, _ in PREREQUISITES])
    return value


def model_environment(base):
    env, _ = base.environment()
    env.update(CUDA_VISIBLE_DEVICES=base.GPU_UUIDS[2],
        PYTHONPATH='/mnt/sdc1/robodojo/behavior_dev/semantic_structured_20260919/deps_817f944:'
                   '/mnt/sdc1/robodojo/behavior_dev/semantic_agent_20260917/deps:' + str(REPO/'src'),
        OMP_NUM_THREADS='4', MKL_NUM_THREADS='4')
    return env


def check_health(health, code, *, initial=False):
    expected = {'protocol':'semantic-v2', 'model':str(MODEL), 'revision':REVISION,
                'code_commit':code, 'max_calls':MAX_CALLS, 'visible_devices':gate.scene.supervisor.GPU_UUIDS[2],
                'dtype':'bfloat16', 'training_updates':0, 'thinking':False,
                'transformers':'5.7.0', 'torch':'2.7.1+cu128', 'image_max_side':640,
                'wrist_no_upsampling':True, 'max_images':9, 'backend':'transformers-sdpa'}
    if any(type(health.get(k)) is not type(v) or health[k] != v for k, v in expected.items()):
        raise ValueError('Actual model health/source/decoder mismatch')
    from semantic_robot.v2.structured_planning import schema_digest, DECODER_COMMIT
    if (health.get('planning_schema_sha256') != schema_digest() or
            health.get('structured_planning_schemas') != ['task_plan_v1','recovery_v1'] or
            health.get('structured_decoder',{}).get('commit') != DECODER_COMMIT or
            health.get('finite_choice_kinds') != ['act','ground','reference']):
        raise ValueError('Structured decoder identity changed')
    if type(health.get('calls')) is not int or not 0 <= health['calls'] <= MAX_CALLS or (initial and health['calls'] != 0):
        raise ValueError('Wrong model call budget/initial state')


def get_health():
    with urllib.request.urlopen(f'http://127.0.0.1:{PORT}/health', timeout=2) as response:
        return json.load(response)


def validate_actor_basic(result, digest):
    expected = {'status':'complete', 'task':0, 'prefix_controls':0, 'diagnostic_replay_controls':0,
                'implementation_digest':DIGEST, 'native_profile':'a100_full_v1', 'synchronous_io_v1':True,
                'harness':'grounded', 'contact_geometry':False, 'press_finger_asset_sha256':gate.ASSET_SHA,
                'full_task_success_rate_claim':False, 'gate_ok':False}
    expected.update({flag.replace('-','_'):True for flag in gate.FLAGS if flag != 'structured-planning'})
    if digest != DIGEST or any(type(result.get(k)) is not type(v) or result[k] != v for k,v in expected.items()):
        raise ValueError('Wrong actor source/profile or hidden warm start')
    for key, lower, upper in (('controls',1,3072), ('model_calls',1,MAX_CALLS)):
        if type(result.get(key)) is not int or not lower <= result[key] <= upper:
            raise ValueError('Actor exceeded ' + key)
    if (not isinstance(result.get('decisions'),list) or not 1 <= len(result['decisions']) <= 96 or
            type(result.get('official_success')) is not bool or type(result.get('terminal')) is not bool or
            not isinstance(result.get('final_goal_status'),dict) or not isinstance(result.get('stop_reason'),str) or
            type(result.get('wall_s')) not in (int,float) or not math.isfinite(result['wall_s']) or
            not 0 <= result['wall_s'] <= 2405):
        raise ValueError('Incomplete actor result/terminal receipt')


def validate_completion(base, code, final_health):
    gate.raise_worker_failure({})
    manifest = read(ROOT / 'gate/manifest.json')
    expected = {'code_commit':code, 'implementation_digest':DIGEST, 'task':0, 'instance':138,
                'task_name':'turning_on_radio', 'split':'train', 'seed':0, 'training_updates':0,
                'actor_scene_truth':False, 'prefix_is_expert_not_agent':False, 'diagnostic_replay_requested':False,
                'args':expected_actor_args(base)}
    if any(json.dumps(manifest.get(k),sort_keys=True) != json.dumps(v,sort_keys=True) for k,v in expected.items()):
        raise ValueError('Actor exact manifest mismatch')
    check_health(manifest['model_identity'],code,initial=True)
    check_health(final_health,code)
    result = read(ROOT / 'gate/result.json')
    synchronous.validate_result(result,DIGEST)
    ledger = [json.loads(line) for line in (ROOT / 'server/calls.jsonl').read_text().splitlines()]
    if (not ledger or any('error' in r for r in ledger) or
            [r.get('call') for r in ledger] != list(range(1,len(ledger)+1)) or
            result['model_calls'] != len(ledger) or final_health['calls'] != len(ledger)):
        raise ValueError('Model call ledger/actor count mismatch')
    native = read(ROOT / 'gate/native_profile.json')
    events = [(r['name'],r['instance'],r['status']) for r in native['official_api_events']]
    if events != [('reset',None,'completed'),('load_task_instance',138,'completed'),('reset',None,'completed')]:
        raise ValueError('Wrong original reset/load sequence')
    return result


def launch(base):
    code, before = preflight(base)
    model = model_identity()
    with socket.socket() as sock:
        if sock.connect_ex(('127.0.0.1',PORT)) == 0: raise RuntimeError('Reserved service port occupied')
    if ROOT.exists() or RUNTIME.exists(): raise FileExistsError('Never reuse a submitted episode')
    ROOT.mkdir(parents=True)
    env, folders = base.environment()
    for path in folders: path.mkdir(parents=True,exist_ok=False)
    env['H52_LAUNCH_TOKEN'] = secrets.token_hex(24)
    receipt = {'status':'reserved', 'source_commit':code, 'output':str(ROOT), 'runtime':str(RUNTIME),
               'utc':datetime.now(timezone.utc).isoformat(), 'commands':commands(base),
               'model_identity':model, 'gpu_before':before, 'wall_seconds':WALL_SECONDS,
               'cleanup_seconds':60, 'token_sha256':hashlib.sha256(env['H52_LAUNCH_TOKEN'].encode()).hexdigest()}
    write('launch.json',receipt)
    with (ROOT/'supervisor.log').open('x') as log:
        child = subprocess.Popen([str(base.PYTHON),str(Path(__file__).resolve()),'--supervise'],
            cwd=REPO,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    receipt.update(status='supervisor_started',supervisor_pid=child.pid)
    write('launch.json',receipt); print(json.dumps(receipt),flush=True)


def supervise(base):
    code,before = preflight(base)
    base.claim_stage('supervisor',code)
    launch_receipt = read(ROOT/'launch.json')
    if launch_receipt['commands'] != commands(base): raise ValueError('Launch commands changed')
    start = time.monotonic(); children = {}; failure = None
    receipt = {'status':'starting', 'source_commit':code, 'supervisor_pid':os.getpid(),
               'samples':[], 'baseline':before, 'wall_seconds':WALL_SECONDS}
    def interrupted(sig,_): raise InterruptedError('H79 signal '+str(sig))
    previous = {sig:signal.signal(sig,interrupted) for sig in (signal.SIGTERM,signal.SIGINT)}
    def monitor():
        if time.monotonic()-start >= WALL_SECONDS: raise TimeoutError('H79 total wall budget')
        current = base.snapshot(); check_resources(current,before,{k:p.pid for k,p in children.items()})
        if shutil.disk_usage('/mnt/nvme_tmp').free < 80*1024**3: raise RuntimeError('Disk reserve')
        sizes = {'run':tree_bytes(ROOT,8), 'runtime':tree_bytes(RUNTIME,16)}
        receipt['samples'].append({'elapsed':time.monotonic()-start,'gpus':current,'bytes':sizes})
        write('supervisor.json',receipt)
    def spawn(kind,env,cores):
        if not cores <= os.sched_getaffinity(0): raise ValueError('Registered CPU cores unavailable')
        oldmask = signal.pthread_sigmask(signal.SIG_BLOCK,set(previous))
        try:
            def prepare():
                os.sched_setaffinity(0,cores); signal.pthread_sigmask(signal.SIG_UNBLOCK,set(previous))
            with (ROOT/(kind+'.log')).open('x') as log:
                children[kind] = subprocess.Popen(commands(base)[kind],cwd=REPO,env=env,
                    stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,
                    restore_signals=True,preexec_fn=prepare)
            receipt[kind+'_pid'] = children[kind].pid
        finally: signal.pthread_sigmask(signal.SIG_SETMASK,oldmask)
    try:
        # Verify all model files again inside the total wall budget before use.
        if model_identity() != launch_receipt['model_identity']: raise ValueError('Model changed after reservation')
        spawn('model',model_environment(base),{48,49,50,51}); receipt['status']='model_loading'
        while True:
            monitor()
            if children['model'].poll() is not None: raise RuntimeError('Model exited before readiness')
            if time.monotonic()-start > 300: raise TimeoutError('Model readiness budget')
            try: health = get_health()
            except (urllib.error.URLError,TimeoutError,ConnectionError): time.sleep(2); continue
            check_health(health,code,initial=True); receipt['model_health']=health; break
        spawn('actor',base.environment()[0],{72,73,74,75}); receipt['status']='actor_running'
        while children['actor'].poll() is None:
            monitor()
            if children['model'].poll() is not None: raise RuntimeError('Model died during actor episode')
            time.sleep(2)
        gate.raise_worker_failure(receipt)
        if children['actor'].returncode != 0: raise RuntimeError('Actor exit '+str(children['actor'].returncode))
        final_health = get_health(); result = validate_completion(base,code,final_health); monitor()
        receipt.update(status='completed', final_health=final_health,
                       official_success=result['official_success'], controls=result['controls'],
                       decisions=len(result['decisions']), stop_reason=result['stop_reason'])
    except BaseException as error:
        failure = error; receipt.update(status='failed',error=repr(error))
    finally:
        cleanup=time.monotonic()
        for kind in ('actor','model'):
            if kind not in children: continue
            try: stop_owned_group(children[kind])
            except BaseException as error:
                failure=failure or error; receipt.update(status='failed',**{kind+'_cleanup_error':repr(error)})
        try:
            receipt['after_exit']=base.snapshot()
            check_resources(receipt['after_exit'],before,{k:p.pid for k,p in children.items()},released=True)
        except BaseException as error:
            failure=failure or error; receipt.update(status='failed',after_exit_error=repr(error))
        if time.monotonic()-cleanup > 60:
            failure=failure or TimeoutError('Cleanup budget'); receipt.update(status='failed',cleanup_overrun=True)
        receipt.update(seconds=time.monotonic()-start,exit_codes={k:p.returncode for k,p in children.items()})
        write('supervisor.json',receipt)
        for sig,handler in previous.items(): signal.signal(sig,handler)
    if failure: raise failure


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__); mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--launch',action='store_true'); mode.add_argument('--supervise',action='store_true')
    args=parser.parse_args(); base=configure(); (launch if args.launch else supervise)(base)
