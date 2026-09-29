"""JEV-04: new planning contract, two new gates, one additional original-start episode.

Each stage is explicitly launched once in a new private runtime. No automatic
retry, training, shared-source edits, or termination of another session.
GPU1 is shared only with the recorded small, idle cross-device contexts of
teammates; our observer and simulator are both capped on physical GPU1.
"""
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
import time
import urllib.error
import urllib.request

import launch_h79 as common
from launch_h69 import belongs_to_session, stop_owned_group
from launch_h44 import tree_bytes
from native_full_profile import configure_gpu

gate, sync = common.gate, common.synchronous
REPO = Path(__file__).resolve().parents[2]
STAGES = ('gate0','gate3','episode1')
RUN_PARENT = Path('/mnt/nvme_tmp/robodojo_agentic_20260925')
RUNTIME_PARENT = Path('/mnt/nvme_tmp/robodojo_sim_runtime_20260925')
KEY_FILE = Path('/mnt/sdc1/robodojo/.config/behavior/credentials/typesafe.key')
PORT = 8986
PLANNING_VALIDATION = RUN_PARENT / 'jev_plan_20260929_api_v1/result.json'
OBSERVER_CALLS, JEV_CALLS = 151, 192
DECISIONS, CONTROLS = 64, 2048
FLAGS = tuple(f for f in sync.ORIGINAL_FLAGS if f != 'structured-planning') + ('synchronous-io-v1',)
STAGE = None
ROOT = RUNTIME = None


def read(path): return json.loads(path.read_text())


def stage_root(stage): return RUN_PARENT / ('jev_plan_20260929_' + stage + '_v1')


def configure(stage):
    global STAGE, ROOT, RUNTIME
    if stage not in STAGES: raise ValueError('Unregistered stage')
    STAGE, ROOT = stage, stage_root(stage)
    RUNTIME = RUNTIME_PARENT / ROOT.name
    base = sync.configure()
    gate.ROOT, gate.RUNTIME, gate.FLAGS = ROOT, RUNTIME, FLAGS
    base.OUTPUT, base.RUNTIME, base.ENTRYPOINT = ROOT, RUNTIME, Path(__file__).resolve()
    configure_gpu(base, 1)
    common.ROOT, common.RUNTIME = ROOT, RUNTIME
    sync.ORIGINAL_VALIDATE = validate_basic
    return base


def is_actor(): return STAGE.startswith('episode')


def commands(base):
    actor = gate.command(base)
    changes = {'--gpu':'1', '--task':'3' if STAGE == 'gate3' else '0'}
    if is_actor(): changes.update({'--mode':'agent','--max-decisions':str(DECISIONS),'--max-controls':str(CONTROLS),'--max-seconds':'2400'})
    for flag, value in changes.items(): actor[actor.index(flag)+1] = value
    if is_actor():
        actor += ['--uri',f'http://127.0.0.1:{PORT}','--expected-revision',common.REVISION,
            '--controller','jev','--jev-task-plan',str(REPO/'configs/semantic_robot/jev_task0_plan.json'),
            '--typesafe-key-file',str(KEY_FILE),'--jev-max-calls',str(JEV_CALLS),'--jev-timeout','20']
        for stage in ('gate0','gate3'): actor += ['--gate-result',str(stage_root(stage)/'gate/result.json')]
    model = [str(common.VLM_PYTHON),str(REPO/'scripts/semantic_robot/serve_v2.py'),
        '--model',str(common.MODEL),'--revision',common.REVISION,'--output',str(ROOT/'server'),
        '--port',str(PORT),'--max-calls',str(OBSERVER_CALLS)]
    return {'actor':actor, **({'model':model} if is_actor() else {})}


def expected_args(base):
    result = dict(uri='http://127.0.0.1:8907', expected_revision=None, gate_result=[],
        replay_prefix_spec=None, contact_geometry=False, budget_profile='pilot', structured_planning=False,
        controller='vlm', jev_task_plan=None, typesafe_key_file=None, jev_max_calls=104, jev_timeout=20.)
    integer = {'task','gpu','prefix','max_decisions','max_controls','max_seconds','odometry_substep_controls','jev_max_calls'}
    parts = iter(commands(base)['actor'][2:])
    for flag in parts:
        key = flag[2:].replace('-','_')
        value = True if flag[2:] in FLAGS else next(parts)
        if key == 'gate_result': result[key].append(value); continue
        result[key] = int(value) if key in integer else float(value) if key == 'jev_timeout' else value
    return result


def check_resources(current, baseline=None, sessions=None, *, released=False):
    uuids = gate.scene.supervisor.GPU_UUIDS
    if set(current) != set(uuids): raise ValueError('Unknown physical GPU set')
    for index, uuid in enumerate(uuids):
        row = current[uuid]
        own_ids = set()
        for kind, sid in (sessions or {}).items():
            owned = [p for p in row['processes'] if belongs_to_session(p['pid'],sid)]
            own_ids.update(p['pid'] for p in owned)
            cap = (59392 if index == 1 else 0) if kind == 'model' else (14336 if index == 1 else 512)
            if sum(p['used_mib'] for p in owned) > cap: raise RuntimeError('Own '+kind+' GPU cap exceeded')
            if released and owned: raise RuntimeError('Own GPU session not released')
        if index == 1:
            if baseline is None:
                if row['used_mib'] > 1024 or any(p['used_mib'] > 256 for p in row['processes']):
                    raise RuntimeError('GPU1 is not free except small existing cross-device contexts')
                if row['free_mib'] < 78000: raise RuntimeError('GPU1 initial reserve')
            else:
                existing = {p['pid']:p['used_mib'] for p in baseline[uuid]['processes']}
                for p in row['processes']:
                    if p['pid'] not in own_ids and (p['pid'] not in existing or p['used_mib'] > existing[p['pid']]+64):
                        raise RuntimeError('GPU1 external allocation changed; stop only ours')
        if row['free_mib'] < 8192: raise RuntimeError('Shared GPU reserve below 8GiB')


def check_health(health, code, *, initial=False):
    expected = dict(protocol='semantic-v2', model=str(common.MODEL), revision=common.REVISION,
        code_commit=code, max_calls=OBSERVER_CALLS, visible_devices=gate.scene.supervisor.GPU_UUIDS[1],
        dtype='bfloat16', training_updates=0, thinking=False, transformers='5.7.0',torch='2.7.1+cu128',
        image_max_side=640, wrist_no_upsampling=True,max_images=9,backend='transformers-sdpa',
        finite_choice_kinds=['act','ground','reference'])
    if any(type(health.get(k)) is not type(v) or health[k] != v for k,v in expected.items()):
        raise ValueError('Observer model/source/decoder mismatch')
    if type(health.get('calls')) is not int or not 0 <= health['calls'] <= OBSERVER_CALLS or (initial and health['calls'] != 0):
        raise ValueError('Observer call budget/initial state mismatch')


def health():
    with urllib.request.urlopen(f'http://127.0.0.1:{PORT}/health', timeout=2) as response: return json.load(response)


def prerequisites(digest):
    required = ['gate0'] if STAGE == 'gate3' else ['gate0','gate3'] if is_actor() else []
    receipts = {}
    for stage in required:
        folder = stage_root(stage)
        result, ended = read(folder/'gate/result.json'), read(folder/'supervisor.json')
        if (ended.get('status') != 'completed' or not ended.get('after_exit') or
                result.get('implementation_digest') != digest or result.get('physical_gpu') != 1):
            raise ValueError('Prior stage has not passed with this implementation/GPU')
        if stage.startswith('gate') and (result.get('gate_ok') is not True or result.get('task') != int(stage[-1])):
            raise ValueError('Wrong prerequisite engineering gate')
        receipts[stage] = {'result_sha256':common.sha(folder/'gate/result.json'),
                           'supervisor_sha256':common.sha(folder/'supervisor.json')}
    return receipts


def planning_qualification():
    from probe_jev_planning import digest, qualification
    from semantic_robot.v2.jev_policy import load_task_plan, planning_request
    value = read(PLANNING_VALIDATION)
    goals, identity = load_task_plan(REPO/'configs/semantic_robot/jev_task0_plan.json', 0)
    templates = [planning_request(goals, identity, identity['official_instruction'], prefix) for prefix in ([], [0])]
    repeats = value.get('repetitions')
    if (value.get('schema') != 'jev04-planning-validation-v1' or value.get('model') != 'jev-1.13.0'
            or value.get('qualified') is not True or value.get('dry_run') is not False or value.get('error')
            or type(repeats) is not int or repeats < 10 or not qualification(value.get('rows', []), repeats)
            or value.get('contract_sha256') != identity['sha256']
            or value.get('official_instruction') != identity['official_instruction']
            or value.get('request_template_sha256') != [digest(dict(state=s, questions=q)) for s, q in templates]
            or value.get('policy_sha256') != common.sha(REPO/'src/semantic_robot/v2/jev_policy.py')
            or value.get('client_sha256') != common.sha(REPO/'src/semantic_robot/v2/jev_client.py')
            or value.get('probe_sha256') != common.sha(REPO/'scripts/semantic_robot/probe_jev_planning.py')
            or value.get('api_requests') != 5+2*repeats or value.get('validated_responses') != 5+2*repeats
            or any(type(value.get(k)) is not int or value[k] != 0 for k in ('new_controls','new_resets','training_updates'))):
        raise ValueError('Exact-input planning API validation not qualified for this source')
    folder = PLANNING_VALIDATION.parent
    if any(value.get(name+'_sha256') != common.sha(folder/(name+'.jsonl')) for name in ('calls','trials')):
        raise ValueError('Planning validation trace changed')
    ledger = [json.loads(line) for line in (folder/'calls.jsonl').read_text().splitlines()]
    validate_jev_ledger(ledger, dict(jev_requests=value['api_requests'],
        jev_input_tokens=value['input_tokens'],jev_output_tokens=value['output_tokens']))
    trials = [json.loads(line) for line in (folder/'trials.jsonl').read_text().splitlines()]
    if [trial['summary'] for trial in trials] != value['rows']:
        raise ValueError('Planning validation summary omits or changes trials')
    choices = [r['selections'].get('next_goal') for r in ledger if r['event'] == 'validated']
    if choices[3:] != ['goal_0','goal_1']*repeats + ['abstain','abstain']:
        raise ValueError('Planning validation differs from durable model choices')
    return {'path':str(PLANNING_VALIDATION),'sha256':common.sha(PLANNING_VALIDATION),
            'repetitions':repeats,'official_instruction_sha256':identity['official_instruction_sha256']}


def preflight(base):
    if 'TYPESAFE_API_KEY' in os.environ:
        raise ValueError('Use only the private key file; inherited TypeSafe credential environment forbidden')
    code = sync.identity(base)
    from run_v2 import implementation_digest
    digest = implementation_digest()
    prior = prerequisites(digest)
    prior['planning_validation'] = planning_qualification()
    before = base.snapshot(); check_resources(before)
    if is_actor():
        from semantic_robot.v2.jev_client import load_key
        load_key(KEY_FILE)  # Validate owner/mode, never store or print the value.
        with socket.socket() as sock:
            if sock.connect_ex(('127.0.0.1',PORT)) == 0: raise RuntimeError('Observer port occupied')
    return code,digest,prior,before


def validate_wall(value,actor):
    if type(value) not in (int,float) or not math.isfinite(value) or not 0 <= value <= (2405 if actor else 1205):
        raise ValueError('Action-phase wall budget exceeded or missing')


def validate_jev_ledger(rows,result):
    attempts=[r for r in rows if r.get('event')=='attempt']
    valid=[r for r in rows if r.get('event')=='validated']
    n=result['jev_requests']
    if (len(rows)!=2*n or [(r.get('call'),r.get('event')) for r in rows] !=
            [(i,event) for i in range(1,n+1) for event in ('attempt','validated')] or
            any(r.get('model')!='jev-1.13.0' for r in rows) or len(attempts)!=n or len(valid)!=n or
            sum(r['usage']['input_tokens'] for r in valid)!=result['jev_input_tokens'] or
            sum(r['usage']['output_tokens'] for r in valid)!=result['jev_output_tokens']):
        raise ValueError('Jev durable request/response accounting mismatch')


def validate_basic(result,digest):
    exact = dict(status='complete',task=3 if STAGE == 'gate3' else 0,physical_gpu=1,
        prefix_controls=0,diagnostic_replay_controls=0,implementation_digest=digest,
        native_profile='a100_full_v1',harness='grounded',contact_geometry=False,
        controller='jev' if is_actor() else 'vlm',press_finger_asset_sha256=gate.ASSET_SHA,
        gate_ok=not is_actor(),full_task_success_rate_claim=False,gate_failures=[])
    exact.update({f.replace('-','_'):True for f in FLAGS})
    if any(type(result.get(k)) is not type(v) or result[k] != v for k,v in exact.items()):
        raise ValueError('Incomplete result or wrong source/profile/controller')
    validate_wall(result.get('wall_s'),is_actor())
    decisions = result.get('decisions',[])
    minimum_decisions = 0 if is_actor() and result.get('stop_reason') == 'JEV_PLAN_ABSTAINED' else 1
    if (not isinstance(decisions,list) or not minimum_decisions <= len(decisions) <= (DECISIONS if is_actor() else 24) or
            type(result.get('controls')) is not int or not 1 <= result['controls'] <= (CONTROLS if is_actor() else 1536)):
        raise ValueError('Control/decision budget mismatch')
    executed = [r for r in decisions if r.get('accepted_before_motion')]
    if any(r.get('jev_freshness_passed') is not True for r in executed):
        raise ValueError('Executed without frozen-state/fresh-render proof')
    if not is_actor():
        if len(decisions)!=24 or result.get('model_calls') != 0: raise ValueError('Incomplete gate')
    elif (type(result.get('official_success')) is not bool or type(result.get('terminal')) is not bool or
            not isinstance(result.get('final_goal_status'),dict) or not isinstance(result.get('stop_reason'),str) or
            not 0 <= result.get('jev_requests',-1) <= JEV_CALLS or
            not 0 <= result.get('observer_calls',-1) <= OBSERVER_CALLS or
            result.get('jev_requests_without_validated_response') != 0):
        raise ValueError('Incomplete official outcome/API accounting')
    if is_actor() and result['stop_reason'] == 'JEV_DECISION_STATE_CHANGED':
        raise ValueError('Frozen-state contract failed; abort evaluation expansion')
    if is_actor():
        from semantic_robot.v2.jev_control import validate_actor_ownership
        validate_actor_ownership(result)


def validate_completion(base,code,digest):
    gate.raise_worker_failure({})
    manifest = read(ROOT/'gate/manifest.json')
    task = 3 if STAGE == 'gate3' else 0
    exact = dict(code_commit=code,implementation_digest=digest,task=task,instance=242 if task else 138,
        task_name='cleaning_up_plates_and_food' if task else 'turning_on_radio',split='train',seed=0,
        training_updates=0,actor_scene_truth=False,prefix_is_expert_not_agent=False,
        diagnostic_replay_requested=False,native_profile='a100_full_v1',args=expected_args(base))
    if any(json.dumps(manifest.get(k),sort_keys=True) != json.dumps(v,sort_keys=True) for k,v in exact.items()):
        raise ValueError('Exact stage manifest mismatch')
    if is_actor():
        # The combined identity nests the unchanged perception-only identity.
        identity = manifest['model_identity']
        if (identity.get('controller') != 'jev' or identity.get('model') != 'jev-1.13.0'
                or identity.get('decision_scope') != 'jev_all_strategy_choices_v2'):
            raise ValueError('Wrong Jev identity')
        check_health(identity['observer_only'],code,initial=True)
    elif manifest['model_identity'] is not None: raise ValueError('Engineering gate used a model')
    result = read(ROOT/'gate/result.json'); sync.validate_result(result,digest)
    if is_actor():
        final = health(); check_health(final,code)
        ledger = [json.loads(x) for x in (ROOT/'server/calls.jsonl').read_text().splitlines()]
        if (any('error' in r or r.get('kind') not in ('observe','ground') for r in ledger) or
                [r.get('call') for r in ledger] != list(range(1,len(ledger)+1)) or
                result['observer_calls'] != len(ledger) or final['calls'] != len(ledger) or
                result['model_calls'] != result['observer_calls']+result['jev_requests']):
            raise ValueError('Perception-only ledger/accounting mismatch')
        jev_ledger = [json.loads(x) for x in (ROOT/'gate/jev_calls.jsonl').read_text().splitlines()]
        validate_jev_ledger(jev_ledger,result)
        from semantic_robot.v2.jev_control import validate_actor_ownership
        ownership = validate_actor_ownership(result, jev_ledger,
            [json.loads(x) for x in (ROOT/'gate/steps.jsonl').read_text().splitlines()])
        common.write('decision_ownership.json', ownership)
    native = read(ROOT/'gate/native_profile.json')
    if (native.get('physical_gpu') != 1 or
        [(r['name'],r['instance'],r['status']) for r in native['official_api_events']] !=
        [('reset',None,'completed'),('load_task_instance',242 if task else 138,'completed'),('reset',None,'completed')]):
        raise ValueError('Wrong GPU or original-start reset sequence')
    return result


def launch(base):
    code,digest,prior,before = preflight(base)
    if ROOT.exists() or ROOT.is_symlink() or RUNTIME.exists() or RUNTIME.is_symlink():
        raise FileExistsError('Never reuse a submitted stage')
    ROOT.mkdir(parents=True)
    env,folders = base.environment()
    prepared = common.prepare_runtime(folders)
    env['H52_LAUNCH_TOKEN'] = secrets.token_hex(24)
    receipt = dict(status='reserved',source_commit=code,implementation_digest=digest,stage=STAGE,
        utc=datetime.now(timezone.utc).isoformat(),output=str(ROOT),runtime=str(RUNTIME),
        token_sha256=hashlib.sha256(env['H52_LAUNCH_TOKEN'].encode()).hexdigest(),
        runtime_preparation=prepared,commands=commands(base),prerequisites=prior,gpu_before=before,
        wall_seconds=3600 if is_actor() else 1800,cleanup_seconds=60,training_updates=0)
    common.write('launch.json',receipt)
    with (ROOT/'supervisor.log').open('x') as log:
        child = subprocess.Popen([str(base.PYTHON),str(Path(__file__).resolve()),'--stage',STAGE,'--supervise'],
            cwd=REPO,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    receipt.update(status='supervisor_started',supervisor_pid=child.pid)
    common.write('launch.json',receipt)
    print(json.dumps({k:receipt[k] for k in ('status','stage','source_commit','supervisor_pid','output','wall_seconds')}),flush=True)


def supervise(base):
    code,digest,prior,before = preflight(base)
    base.claim_stage('supervisor',code)
    reserved = read(ROOT/'launch.json')
    if reserved['stage'] != STAGE or reserved['commands'] != commands(base) or reserved['prerequisites'] != prior:
        raise ValueError('Stage/command/prerequisite changed since reservation')
    common.check_runtime_preparation(reserved['runtime_preparation'])
    start = time.monotonic(); children = {}; failure = None
    receipt = dict(status='starting',source_commit=code,implementation_digest=digest,stage=STAGE,
        supervisor_pid=os.getpid(),baseline=before,wall_seconds=reserved['wall_seconds'],samples=[])
    def interrupted(sig,_): raise InterruptedError('JEV-04 signal '+str(sig))
    previous = {sig:signal.signal(sig,interrupted) for sig in (signal.SIGTERM,signal.SIGINT)}
    def monitor():
        if time.monotonic()-start >= receipt['wall_seconds']: raise TimeoutError('Total stage wall budget')
        current=base.snapshot(); check_resources(current,before,{k:p.pid for k,p in children.items()})
        if shutil.disk_usage('/mnt/nvme_tmp').free < 80*1024**3: raise RuntimeError('Disk reserve')
        sizes = {'run':tree_bytes(ROOT,8),'runtime':tree_bytes(RUNTIME,16)}
        receipt['samples'].append(dict(elapsed=time.monotonic()-start,gpus=current,bytes=sizes))
        common.write('supervisor.json',receipt)
    def spawn(kind,env,cores):
        if not cores <= os.sched_getaffinity(0): raise ValueError('CPU cores unavailable')
        old = signal.pthread_sigmask(signal.SIG_BLOCK,set(previous))
        try:
            def prepare():
                os.sched_setaffinity(0,cores); signal.pthread_sigmask(signal.SIG_UNBLOCK,set(previous))
            with (ROOT/(kind+'.log')).open('x') as log:
                children[kind] = subprocess.Popen(commands(base)[kind],cwd=REPO,env=env,
                    stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,preexec_fn=prepare)
            receipt[kind+'_pid'] = children[kind].pid
        finally: signal.pthread_sigmask(signal.SIG_SETMASK,old)
    try:
        if is_actor():
            receipt['model_identity'] = common.model_identity()
            env = common.model_environment(base);env['CUDA_VISIBLE_DEVICES']=base.GPU_UUIDS[1]
            spawn('model',env,{48,49,50,51});receipt['status']='model_loading'
            while True:
                monitor()
                if children['model'].poll() is not None: raise RuntimeError('Observer exited before ready')
                if time.monotonic()-start > 300: raise TimeoutError('Observer readiness budget')
                try: value=health()
                except (urllib.error.URLError,TimeoutError,ConnectionError): time.sleep(2);continue
                check_health(value,code,initial=True);receipt['model_health']=value;break
        spawn('actor',base.environment()[0],{88,89,90,91});receipt['status']='actor_running'
        while children['actor'].poll() is None:
            monitor()
            if is_actor() and children['model'].poll() is not None: raise RuntimeError('Observer died during episode')
            time.sleep(5)
        gate.raise_worker_failure(receipt)
        if children['actor'].returncode != 0: raise RuntimeError('Actor exit '+str(children['actor'].returncode))
        result=validate_completion(base,code,digest);monitor()
        receipt.update(status='completed',official_success=result['official_success'],controls=result['controls'],
            decisions=len(result['decisions']),stop_reason=result['stop_reason'])
        if is_actor():
            receipt['control_integration_validated'] = read(ROOT/'decision_ownership.json')['control_integration_validated']
    except BaseException as error:
        failure=error;receipt.update(status='failed',error=repr(error))
    finally:
        cleanup_started = time.monotonic()
        for kind in ('actor','model'):
            if kind not in children: continue
            try: stop_owned_group(children[kind])
            except BaseException as error:
                failure=failure or error;receipt.update(status='failed',**{kind+'_cleanup_error':repr(error)})
        try:
            receipt['after_exit']=base.snapshot()
            check_resources(receipt['after_exit'],before,{k:p.pid for k,p in children.items()},released=True)
        except BaseException as error:
            failure=failure or error;receipt.update(status='failed',after_exit_error=repr(error))
        receipt['cleanup_seconds'] = time.monotonic()-cleanup_started
        if receipt['cleanup_seconds'] > 60:
            failure=failure or TimeoutError('Cleanup budget exceeded')
            receipt.update(status='failed',cleanup_overrun=True)
        receipt.update(seconds=time.monotonic()-start,exit_codes={k:p.returncode for k,p in children.items()})
        common.write('supervisor.json',receipt)
        for sig,handler in previous.items(): signal.signal(sig,handler)
    if failure: raise failure


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=STAGES,required=True)
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--launch',action='store_true');mode.add_argument('--supervise',action='store_true')
    args=parser.parse_args();base=configure(args.stage)
    (launch if args.launch else supervise)(base)
