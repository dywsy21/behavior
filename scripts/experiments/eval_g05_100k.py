"""Bounded evaluation of the teammate's September 23 no-memory G05 run.

Never trains or changes the original run. Model and simulator live in separate
processes/environments. This is a development evaluation, not a leaderboard run.
"""
from __future__ import annotations

import argparse
import asyncio
from collections import Counter, defaultdict
from contextlib import contextmanager
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import traceback

REPO = Path(__file__).resolve().parents[2]
RUN = Path('/mnt/sdc1/robodojo/outputs/g05/r1pro/behavior5_nomem_bs8_4gpu_20260923T135319Z')
OUT = Path('/mnt/nvme_tmp/robodojo_g05_100k_eval_20260928/v4')
RUNTIME = Path('/mnt/nvme_tmp/robodojo_sim_runtime_20260925/eval_g05_100k_20260928_v4')
MODEL_PY = '/mnt/sdc1/robodojo/GalaxeaVLA/.venv/bin/python'
SIM_PY = '/mnt/sdc1/xhz/miniconda3/envs/behavior/bin/python'
PORT = 8931
SEED = 17
MODEL_CALL_LIMIT = 460
TASKS = ('turning_on_radio', 'picking_up_trash', 'putting_away_Halloween_decorations',
         'cleaning_up_plates_and_food', 'can_meat')
LIMITS = (3224, 1024, 1024, 1024, 1024)
ADAPTER = Path('/mnt/sdc1/robodojo/behavior_dev/GalaxeaVLA_memlite_coordination_dev_20260908/sim_runtime/production_native_oracle_low_v1')
WINDOW = Path('/mnt/sdc1/robodojo/behavior_dev/direct_execution_L6rqZ6_20260910/c1_windows_v2_matched/c1v2-matched-t0-train-e121-f448-grasp/window.json')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''): h.update(b)
    return h.hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f: json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)


def commit():
    if subprocess.check_output(['git', '-C', str(REPO), 'status', '--porcelain'], text=True).strip():
        raise RuntimeError('Evaluation requires clean, frozen source')
    return subprocess.check_output(['git', '-C', str(REPO), 'rev-parse', 'HEAD'], text=True).strip()


def checkpoint(step):
    if type(step) is not int or step not in range(10000, 100001, 10000):
        raise ValueError('Only the ten saved checkpoints in this run')
    return RUN / f'checkpoints/step_{step}.pt'


def choose_windows(starts, ends, episode_ids):
    """Two deterministic interior windows per original held-out episode."""
    if len(starts) != len(ends) or len(starts) != len(episode_ids): raise ValueError('Episode map mismatch')
    rows = []
    for start, end, episode in zip(starts, ends, episode_ids):
        start, end, episode = int(start), int(end), int(episode)
        if end-start < 64: raise ValueError('Episode too short for registered windows')
        for fraction in (0.25, 0.75):
            rows.append(dict(index=start+int((end-start-32)*fraction), episode=episode, fraction=fraction))
    if len({r['index'] for r in rows}) != len(rows): raise ValueError('Duplicate windows')
    return rows


def vector_chunk(action):
    import numpy as np
    keys = ('base_qvel', 'trunk_qpos', 'left_arm', 'left_gripper', 'right_arm', 'right_gripper')
    dims = (3, 4, 7, 1, 7, 1)
    arrays = []
    for key, dim in zip(keys, dims):
        x = action[key]
        if hasattr(x, 'detach'): x = x.detach().cpu().numpy()
        x = np.asarray(x, dtype=np.float32)
        if x.shape == (1, 32, dim): x = x[0]
        if x.shape != (32, dim) or not np.isfinite(x).all(): raise ValueError('Invalid full FM part: '+key)
        arrays.append(x)
    return np.concatenate(arrays, axis=-1)


def load_model(step):
    sys.path[:0] = [str(REPO/'scripts'), str(REPO/'src'), str(REPO)]
    os.chdir(REPO)
    import torch
    from omegaconf import OmegaConf
    from g05.utils.config.config_resolvers import register_default_resolvers
    from g05.utils.checkpoint import checkpoint_utils
    from g05.utils.common.pytorch_utils import set_global_seed
    import serve_policy
    register_default_resolvers()
    set_global_seed(SEED)
    torch.set_num_threads(2)
    cfg = OmegaConf.load(RUN/'.hydra/config.yaml')
    cfg.output_dir = str(OUT)
    cfg.exp_name = 'g05_100k_eval_20260928'
    cfg.ckpt_path, cfg.run_dir = str(checkpoint(step)), str(RUN)
    cfg.model.model_arch.hf_processor_path = '/mnt/sdc1/robodojo/checkpoints/G05/qwen3_5_2b_base_processor'
    cfg.tokenizer.vq_config.ckpt_dir = str(RUN/'action_tokenizer.pt')
    # Preserve the training recipe's fp32 parameters + bf16 autocast. A full
    # bf16 weight cast is valid for serving but not this fused CE implementation.
    cfg.model.model_weights_to_bf16 = False
    cfg.model.use_torch_compile = False
    if (cfg.model.model_arch._target_ != 'g05.models.g05.g05_policy_qwen35.G05PolicyQwen35'
            or cfg.data.obs_size != 1 or cfg.data.val_split_mode != 'task_stratified'
            or cfg.data.val_set_proportion != .05 or cfg.model.model_arch.predict_cot):
        raise ValueError('Unexpected training configuration')
    original = checkpoint_utils.load_state_dict_safely
    receipts = []
    def strict_load(*args, **kwargs):
        kwargs['return_info'] = True
        info = original(*args, **kwargs)
        receipt = {k:v for k,v in info.items() if k != 'model'}
        receipts.append(receipt)
        if any(receipt.get(k) for k in ('truly_missing', 'partial_loaded_keys', 'mismatched_keys', 'unexpected_keys')):
            raise RuntimeError('Non-exact checkpoint load: '+str(receipt))
        return info['model']
    checkpoint_utils.load_state_dict_safely = strict_load
    try: policy, processor = serve_policy.setup(cfg, device='cuda:0')
    finally: checkpoint_utils.load_state_dict_safely = original
    save(OUT/f'load_{step}.json', dict(source_commit=commit(), checkpoint=str(checkpoint(step)),
        checkpoint_bytes=checkpoint(step).stat().st_size, load_receipts=receipts,
        config_sha256=sha(RUN/'.hydra/config.yaml'), stats_sha256=sha(RUN/'dataset_stats.json'),
        action_tokenizer_sha256=sha(RUN/'action_tokenizer.pt'), dtype='original_fp32_weights_bf16_autocast'))
    return cfg, policy, processor


def offline(step, cfg, policy, processor):
    import torch
    from g05.utils.data.processor_utils import instantiate_dataset
    from g05.utils.data.data_utils import collate_fn_pad_sequences
    from g05.utils.common.pytorch_utils import dict_apply, set_global_seed
    dataset = instantiate_dataset(cfg, is_training_set=False)
    dataset.set_processor(processor)
    if len(dataset.datasets) != 1: raise ValueError('Expected one five-task dataset')
    child = dataset.datasets[0]
    episodes = child._active_episode_indices
    if len(episodes) != 50: raise ValueError('Original 50 held-out episodes required')
    windows = choose_windows(child.episode_data_index['from'], child.episode_data_index['to'], episodes)
    rows = []
    with (OUT/f'offline_{step}.jsonl').open('x', buffering=1) as trace:
        for number, window in enumerate(windows):
            sample = dataset[window['index']]
            # The existing collator pops the nested samples field in place.
            task = sample['samples']['command']
            batch = collate_fn_pad_sequences([sample], padding_input_id=processor.pad_token_id)
            batch = dict_apply(batch, lambda t: t.to('cuda:0') if isinstance(t, torch.Tensor) else t)
            set_global_seed(19000+number)
            with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                _, loss = policy(batch)
            values = {k:float(v) for k,v in loss.items()}
            if not all(torch.isfinite(torch.tensor(v)) for v in values.values()): raise ValueError('Nonfinite eval loss')
            row = dict(window, task=task, losses=values)
            rows.append(row); trace.write(json.dumps(row, ensure_ascii=False)+'\n')
            print('OFFLINE', step, number+1, values, flush=True)
    groups = defaultdict(list)
    for row in rows: groups[row['task']].append(row)
    if len(groups) != 5 or any(len(v) != 20 for v in groups.values()):
        raise ValueError('Five real task identities with 20 windows each required: '+str({k:len(v) for k,v in groups.items()}))
    def means(group):
        return {key:sum(row['losses'][key] for row in group)/len(group) for key in rows[0]['losses']}
    result = dict(step=step, samples=len(rows), episodes=list(map(int,episodes)), seed_scheme='19000+window_number',
        samples_per_task={k:len(v) for k,v in groups.items()}, mean=means(rows),
        per_task={k:means(v) for k,v in groups.items()}, source_commit=commit(),
        fixed_window_rows_sha256=sha(OUT/f'offline_{step}.jsonl'), full_eval=False, training_updates=0)
    save(OUT/f'offline_{step}.json', result)
    # A single real held-out input establishes that omitting auxiliary AR
    # generation leaves the already-generated FM output identical.
    from copy import deepcopy
    batch = collate_fn_pad_sequences([dataset[windows[0]['index']]], padding_input_id=processor.pad_token_id)
    batch = dict_apply(batch, lambda t: t.to('cuda:0') if isinstance(t,torch.Tensor) else t)
    set_global_seed(SEED)
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
        full = policy.predict_action(deepcopy(batch))['action'].detach().cpu()
        policy.discrete_action = False
        set_global_seed(SEED)
        try: fm_only = policy.predict_action(deepcopy(batch))['action'].detach().cpu()
        finally: policy.discrete_action = True
    delta = float((full-fm_only).abs().max())
    save(OUT/f'fm_equivalence_{step}.json', dict(max_abs_difference=delta, identical=torch.equal(full,fm_only)))
    if not torch.equal(full,fm_only): raise ValueError('FM-only optimization changed selected actions')
    return dataset


async def serve(cfg, policy, processor, step=100000):
    import torch
    import numpy as np
    import websockets
    from g05.models.g05.inferencer import PolicyInferencer
    from g05.utils.websocket import packb, unpackb
    from g05.utils.common.pytorch_utils import set_global_seed
    from serve_policy import build_obs_dict
    inferencer = PolicyInferencer(policy, processor, device='cuda:0')
    # FM is generated BEFORE AR and is the selected action in this dual-head
    # configuration. Confirm equality before omitting the unused AR diagnostic.
    policy.discrete_action = False
    identity = dict(checkpoint=str(checkpoint(step)), checkpoint_sha256=sha(checkpoint(step)), checkpoint_step=step,
        source_commit=commit(), action_source='fm', obs_steps=1, predicted_steps=32, execute_steps=16,
        predict_cot=False, memlite=False, robot_action_dim=23, port=PORT)
    save(OUT/'identity.json', identity)
    calls = 0
    async def handler(ws):
        nonlocal calls
        await ws.send(packb(identity))
        async for message in ws:
            request = unpackb(message)
            if set(request) == {'reset'}:
                set_global_seed(SEED); await ws.send(packb({'reset':True})); continue
            if set(request) != {'observation'}: raise ValueError('Only public observation request allowed')
            raw = request['observation']
            if set(raw) != {'images','state','task','embodiment_type','frequency'} or float(raw['frequency']) != 30:
                raise ValueError('Deployment observation allowlist violated')
            if calls >= MODEL_CALL_LIMIT: raise RuntimeError('Registered model-call budget exhausted')
            start = time.monotonic()
            with torch.inference_mode(): action = inferencer.infer([build_obs_dict(raw,processor)])[0]
            chunk = vector_chunk(action)
            calls += 1
            await ws.send(packb(dict(chunk=chunk, call=calls, inference_seconds=time.monotonic()-start)))
    async with websockets.serve(handler, '127.0.0.1', PORT, max_size=64<<20, ping_timeout=None):
        save(OUT/'ready.json', dict(pid=os.getpid(), identity=identity))
        await asyncio.Future()


def simulate(task):
    if task not in range(5): raise ValueError('Task outside preregistration')
    sys.path[:0] = [str(REPO/'scripts/semantic_robot'), str(REPO/'src'), str(REPO)]
    # Import the private A100 profile before inserting the legacy adapter path.
    import native_full_profile
    sys.path.insert(0,str(ADAPTER))
    from native_oracle_low_v1 import official_factory as factory
    import numpy as np
    import imageio.v2 as imageio
    from PIL import Image, ImageDraw
    from websockets.sync.client import connect
    from g05.utils.websocket import packb, unpackb
    from semantic_robot.v2.synchronous_io import native_adapter
    from semantic_robot.v2.onboard import OnboardRGBD
    out = OUT/f'task_{task}'
    out.mkdir(exist_ok=False)
    runtime = RUNTIME/f'task_{task}'
    window = replace(factory.load_official_oracle_window(WINDOW), task_name=TASKS[task],
                     official_mode='public_test', instance_id=301, seed=0, max_steps=None)
    save(out/'manifest.json', dict(task=TASKS[task], task_id=task, instance=301, split='public_test',
        development_instance=True, environment_seed=0, policy_seed=SEED, control_limit=LIMITS[task],
        expert_prefix=0, oracle_skill=0, source_commit=commit(), renderer='a100_full_v1_pathtracing',
        image_distribution_changed_vs_old_a4=True, training_updates=0, camera_history=1,
        actor_inputs=['task','three_current_RGB','proprio'], action_start=0, execute_steps=16))
    controls = calls = 0
    last_info = {}
    success = terminal = False
    start = time.monotonic()
    trace = (out/'steps.jsonl').open('x',buffering=1)
    io_trace = (out/'io.jsonl').open('x',buffering=1)
    video = io = None
    def array(x): return x.detach().cpu().numpy() if hasattr(x,'detach') else np.asarray(x)
    def write_io(row): io_trace.write(json.dumps(row,allow_nan=False)+'\n')
    with native_full_profile.session(factory, window, gpu=3, output=out, runtime=runtime) as session:
        try:
            session.reset()
            import omnigibson as og
            evaluator = session.evaluator
            env = evaluator.env
            reader = OnboardRGBD(env)
            io = native_adapter(og.sim, reader.sensors, write_io)
            video = imageio.get_writer(out/'rollout.mp4', fps=30/16, codec='libx264', quality=7,
                                       macro_block_size=None)
            with connect(f'ws://127.0.0.1:{PORT}', max_size=64<<20, ping_timeout=None) as ws:
                identity = unpackb(ws.recv(timeout=30))
                if identity != json.loads((OUT/'identity.json').read_text()): raise ValueError('Wrong model server')
                ws.send(packb({'reset':True}))
                if unpackb(ws.recv(timeout=30)) != {'reset':True}: raise ValueError('Model reset not acknowledged')
                while controls < LIMITS[task] and not terminal:
                    capture = io.synchronize()
                    images, receipts = {}, {}
                    for view,sensor in reader.sensors.items():
                        obs,_ = sensor.get_obs()
                        rgb = array(obs['rgb'])[...,:3].copy()
                        depth = array(obs['depth_linear']).copy()
                        images[view+'_rgb'] = rgb.transpose(2,0,1)
                        receipts[view] = dict(native_time=capture[view],
                            rgb_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),
                            depth_sha256=hashlib.sha256(depth.tobytes()).hexdigest())
                    io.verify_read(receipts)
                    # Current proprio comes from the same stopped physical state.
                    obs,_ = env.get_obs()
                    evaluator.obs = evaluator._preprocess_obs(obs)
                    raw = factory.behavior_obs_to_native_low(evaluator.obs, factory.load_task_instructions(factory.DEFAULT_TASKS_PATH))
                    raw['images'] = images
                    raw = {k:raw[k] for k in ('images','state','task','embodiment_type','frequency')}
                    before = io.clock()
                    canvas = Image.new('RGB',(960,768),'black')
                    canvas.paste(Image.fromarray(images['head_rgb'].transpose(1,2,0)).resize((720,720)),(0,48))
                    for i,key in enumerate(('left_wrist_rgb','right_wrist_rgb')):
                        canvas.paste(Image.fromarray(images[key].transpose(1,2,0)).resize((240,240)),(720,48+i*240))
                    ImageDraw.Draw(canvas).text((8,12),f'G05-Qwen3.5 step {identity.get("checkpoint_step",100000)} | {TASKS[task]} | control {controls} | FM',fill='white')
                    video.append_data(np.asarray(canvas))
                    if controls == 0:
                        canvas.save(out/'initial.png')
                        save(out/'observation_contract.json', dict(keys=sorted(raw),
                            images={k:list(v.shape) for k,v in raw['images'].items()},
                            state={k:list(v.shape) for k,v in raw['state'].items()}, task=raw['task'], frequency=raw['frequency']))
                    ws.send(packb({'observation':raw}))
                    reply = unpackb(ws.recv(timeout=180))
                    if io.clock() != before: raise RuntimeError('Physics advanced during policy inference')
                    chunk = np.asarray(reply['chunk'],dtype=np.float32)
                    if chunk.shape != (32,23) or not np.isfinite(chunk).all(): raise ValueError('Bad full control chunk')
                    calls += 1
                    for action in chunk[:min(16,LIMITS[task]-controls)]:
                        with io.control(render_requested=False):
                            obs,_,terminated,truncated,info = env.step(action,n_render_iterations=1)
                        controls += 1
                        last_info = info
                        success = bool(info.get('done',{}).get('success',False))
                        terminal = bool(terminated or truncated)
                        held = {arm:(obj.name if obj is not None else None)
                                for arm,obj in env.robots[0]._ag_obj_in_hand.items()}
                        trace.write(json.dumps(dict(control=controls, call=calls, action=action.tolist(),
                            success=success, terminated=bool(terminated), truncated=bool(truncated),
                            held=held, inference_seconds=reply['inference_seconds'], clock=io.clock()),allow_nan=False)+'\n')
                        if terminal: break
                    print('ROLLOUT',task,controls,success,held,flush=True)
                result = dict(task=TASKS[task], instance=301, controls=controls, model_calls=calls,
                    official_success=success, env_terminated_or_truncated=terminal,
                    reason='environment_done' if terminal else 'registered_control_budget',
                    max_controls=LIMITS[task], seconds=time.monotonic()-start, status='completed',
                    full_task_success_rate_claim=False, local_partial_episode=task!=0,
                    source_commit=commit(), checkpoint_sha256=identity['checkpoint_sha256'])
                canvas.save(out/'last_policy_observation.png')
                video.close(); video=None
                io.close(); io=None
                save(out/'result.json',result)
        except BaseException as error:
            save(out/'failure.json',dict(error=repr(error),traceback=traceback.format_exc(),controls=controls,calls=calls))
            raise
        finally:
            if video is not None: video.close()
            if io is not None:
                if sys.exc_info()[0] is not None: io.abandon_read_after_failure()
                io.close()
            trace.close(); io_trace.close()


def model_env(gpu):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONDONTWRITEBYTECODE='1',
               PYTHONUNBUFFERED='1', OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1',
               HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false')
    env['PYTHONPATH'] = str(REPO/'src')+':'+str(REPO)
    npp = '/mnt/sdc1/robodojo/GalaxeaVLA/.venv/lib/python3.10/site-packages/nvidia/npp/lib'
    env['LD_LIBRARY_PATH'] = ':'.join([npp,*[p for p in env.get('LD_LIBRARY_PATH','').split(':') if p and p!=npp]])
    return env


def simulation_env(task):
    sys.path.insert(0,str(REPO/'scripts/semantic_robot'))
    import probe_simulator_startup as base
    runtime = RUNTIME/f'task_{task}'
    runtime.mkdir(parents=True,exist_ok=False)
    env = dict(os.environ)
    env.pop('CUDA_VISIBLE_DEVICES',None)
    env.update(base.FIXED_ENV)
    env.update({k:str(runtime/v) for k,v in base.ROUTES.items()})
    for v in set(base.ROUTES.values())|{'portable','cache','data'}: (runtime/v).mkdir(parents=True,exist_ok=True)
    # Avoid inherited Kit screenshot permissions; no old runtime is changed.
    (runtime/'portable/data/documents/Kit/shared/screenshots').mkdir(parents=True,mode=0o700)
    env.update(OMNIGIBSON_GPU_ID='3', OMNIGIBSON_HEADLESS='1', OMNIGIBSON_NO_OMNI_LOGS='0',
               BEHAVIOR_ACTION_STEPS='1', MEMLITE_SIM_TRACE_PATH='', PYTHONPATH=str(REPO/'src')+':'+str(REPO),
               LD_LIBRARY_PATH='/mnt/sdc1/xhz/miniconda3/envs/behavior/lib/python3.11/site-packages/pymeshlab/lib')
    return env


def stop_owned(process):
    if process.poll() is not None: return
    os.killpg(process.pid,signal.SIGTERM)
    try: process.wait(timeout=20)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid,signal.SIGKILL); process.wait(timeout=10)


def supervise():
    children=[]
    start=time.monotonic()
    result={'source_commit':commit(),'status':'running','started_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
    def child(mode,step,gpu):
        log=(OUT/f'{mode}_{step}.log').open('x')
        proc=subprocess.Popen([MODEL_PY,str(Path(__file__).resolve()),mode,'--step',str(step)],
            env=model_env(gpu),cwd=REPO,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        children.append(proc); log.close(); return proc
    try:
        baseline=child('offline',40000,1)
        actor=child('serve',100000,0)
        save(OUT/'children.json',dict(baseline_pid=baseline.pid,actor_pid=actor.pid))
        deadline=time.monotonic()+1800
        while not (OUT/'ready.json').exists():
            if actor.poll() is not None: raise RuntimeError('Actor/offline validation failed')
            if baseline.poll() not in (None,0): raise RuntimeError('Paired offline baseline failed')
            if time.monotonic()>deadline: raise TimeoutError('30-minute offline/actor preparation budget exhausted')
            time.sleep(3)
        for task in range(5):
            remaining=7200-(time.monotonic()-start)
            if remaining<60: raise TimeoutError('Total two-hour evaluation budget exhausted')
            env=simulation_env(task)
            with (OUT/f'sim_{task}.log').open('x') as log:
                proc=subprocess.Popen([SIM_PY,str(Path(__file__).resolve()),'simulate','--task',str(task)],
                    env=env,cwd=REPO,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            children.append(proc)
            save(OUT/f'sim_{task}_launch.json',dict(pid=proc.pid,task=task,max_wall_seconds=min(3600,remaining)))
            try: code=proc.wait(timeout=min(3600,remaining))
            except subprocess.TimeoutExpired:
                stop_owned(proc); raise
            if code or not (OUT/f'task_{task}/result.json').exists(): raise RuntimeError('Simulator did not complete: '+str(task))
        if baseline.wait(timeout=30): raise RuntimeError('Baseline offline failed')
        result['status']='completed'
    except BaseException as error:
        result.update(status='failed',error=repr(error),traceback=traceback.format_exc())
        raise
    finally:
        for process in reversed(children): stop_owned(process)
        result.update(seconds=time.monotonic()-start,owned_returncodes={str(p.pid):p.returncode for p in children})
        save(OUT/'supervisor.json',result)


def launch():
    revision=commit()
    used=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True)
    if len(used.splitlines())!=4 or any(int(x)>512 for x in used.splitlines()): raise RuntimeError('Expected currently idle GPUs')
    with socket.socket() as sock: sock.bind(('127.0.0.1',PORT))
    subprocess.run([MODEL_PY,'-c','from torchcodec.decoders import VideoDecoder; print("CPU decoder import passed")'],
                   env=model_env(0),check=True,timeout=60)
    OUT.mkdir(parents=True,exist_ok=False)
    save(OUT/'launch.json',dict(source_commit=revision,owner='Codex/EVAL-G05-100K',
        model_run=str(RUN),steps=[40000,100000],fixed_eval_windows=100,
        heldout_episodes=50,tasks=list(TASKS),control_limits=list(LIMITS),total_control_limit=sum(LIMITS),
        model_call_limit=460,wall_seconds=7200,training_updates=0,automatic_retry=False))
    with (OUT/'supervisor.log').open('x') as log:
        process=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'supervise'],cwd=REPO,
            env=model_env(0),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    save(OUT/'supervisor_launch.json',dict(pid=process.pid))
    print(json.dumps({'output':str(OUT),'pid':process.pid}),flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=('offline','serve','simulate','launch','supervise'))
    p.add_argument('--step', type=int, default=100000)
    p.add_argument('--task', type=int, default=0)
    args = p.parse_args()
    if args.mode in ('offline','serve'):
        cfg, policy, processor = load_model(args.step)
        offline(args.step,cfg,policy,processor)
        if args.mode == 'serve': asyncio.run(serve(cfg,policy,processor,step=args.step))
    elif args.mode == 'simulate': simulate(args.task)
    elif args.mode == 'launch': launch()
    else: supervise()


if __name__ == '__main__': main()
