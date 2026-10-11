"""Frozen current high/low/observer, one fresh full episode per process/GPU."""
import argparse
import asyncio
from dataclasses import asdict
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO/'src'), str(REPO/'scripts'),
    str(REPO/'scripts/rl/memlite_online/code'), str(REPO/'scripts/eval/memlite_sft100')]
from recovery_corpus import file_sha
from recovery_process_cache import isolated_cache_env
from common import atomic_json


async def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--case', required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    cfg = json.loads(a.config.read_text()); case = cfg['cases'][a.case]
    if cfg['schema'] != 'fresh_full_episode_readonly_v1' or a.output.exists():
        raise ValueError('Explicit fresh full-episode recipe and new output required')
    if subprocess.check_output(['git','status','--porcelain'], cwd=REPO, text=True).strip():
        raise ValueError('Clean frozen source required')
    gpu = os.environ.get('CUDA_VISIBLE_DEVICES')
    if gpu != str(case['model_gpu']) or subprocess.check_output(
            ['nvidia-smi','-i',gpu,'--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
        raise ValueError('Do not use an occupied or unregistered model GPU')
    a.output.mkdir(parents=True)
    os.environ.update(isolated_cache_env(a.output))
    spec = json.loads((REPO/cfg['models_config']).read_text()); root = Path(spec['root'])
    status = dict(status='validating', pid=os.getpid(), case=a.case, config_sha256=file_sha(a.config),
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
        optimizer_updates=0, started=time.time())
    atomic_json(a.output/'status.json', status)
    for key in ('planner','low','observer_backbone','observer_adapter'):
        if file_sha(root/spec[key]['path']) != spec[key]['sha256']:
            raise ValueError('Changed fixed model: '+key)
    from g05.utils.memlite_causal_session import CausalModelIdentity, CausalSessionIdentity
    from recovery_observer_pilot_runtime import validate_pilot_receipts, LoadedPilotObserver
    identities = CausalModelIdentity(**{k:spec[k]['sha256'] for k in
        ('planner','low','observer_backbone','observer_adapter')},
        **{k+'_normalization':v for k,v in spec['normalization_sha256'].items()})
    accepted = validate_pilot_receipts(spec, identities, REPO)
    import torch
    import websockets
    from recovery_cuda_startup import initialize_cuda_backend
    initialize_cuda_backend(0)
    from g05.models.g05.qwen35 import vision
    from g05.utils.training.stage1_model import configuration,restore_model,make_processor
    from g05.models.kv_cache import SparseKVCache
    from sparse_cache import indexed_cache_freeze
    from train_memlite_recovery import parameter_digest
    from recovery_causal_inference import CausalPlannerInference
    from skill_policy_adapter import SingleFrameSkillAdapter
    from direct_a4_flow import A4DirectPPO
    from full_episode_session import FullEpisodeSession
    from wire import packb, unpackb
    torch.set_num_threads(2); torch.manual_seed(cfg['policy_seed'])
    vision._flash_attn_varlen = None; vision._flash_attn_backend = None
    def forbidden(*args, **kwargs):
        raise RuntimeError('Video rollout cannot perform optimizer updates')
    torch.optim.Adam.step = torch.optim.AdamW.step = forbidden
    names = json.loads((root/spec['expert_release']/'manifest.json').read_text())['task_names']
    models, configs, restores = {}, {}, {}
    for branch,key in [('low','low'),('high','planner')]:
        configs[branch] = configuration(root, branch, names)
        if file_sha(configs[branch]['stats_path']) != getattr(identities,key+'_normalization'):
            raise ValueError('Wrong normalization for '+branch)
        saved = torch.load(root/spec[key]['path'],map_location='cpu',mmap=True,weights_only=False)
        model, restores[branch] = restore_model(configs[branch],branch,state=saved['model_state_dict'])
        del saved; gc.collect()
        models[branch] = model.requires_grad_(False).eval().cuda()
    low = models['low']; low.model.action_expert.float()
    adapter = SingleFrameSkillAdapter(make_processor(configs['low'],False),configs['low'])
    high = models['high']
    planner = CausalPlannerInference(high,make_processor(configs['high'],False),configs['high'],
        models=identities, loaded_planner_sha256=identities.planner,
        cache_context=lambda:indexed_cache_freeze(high.model.ar_helper,SparseKVCache))
    observer = LoadedPilotObserver(dict(models=spec,calibrated=accepted),identities,names,
        parameter_digest=parameter_digest)
    before = {k:parameter_digest(v,frozen=True) for k,v in models.items()}
    # Reuse the exact read-only FP32 Euler velocity function, NOT its PPO
    # constructor, optimizer, critic, replay, exploration or update code.
    velocity_owner = SimpleNamespace(policy=low, fm=low.model.fm_helper)
    pad = torch.zeros(27,dtype=torch.bool,device='cuda'); pad[[7,8,17,18]]=True

    @torch.inference_mode()
    def sample(observation, goal, step):
        prepared, anchor = adapter.prepare(observation,goal)
        batch = adapter.collate([prepared]); fm = low.model.fm_helper
        with torch.random.fork_rng(devices=[0]):
            torch.manual_seed(cfg['policy_seed']+1000003*(step//16))
            with adapter._branch_context():
                state = low.prefill(batch['samples'],batch['pixel_values'])
                dummy = torch.zeros(1,fm.horizon_steps,fm.action_dim,device='cuda',dtype=torch.float32)
                action = fm._sample_noise(dummy,torch.float32,[batch['samples'][0].get('embodiment')])
                action[...,pad]=0
                velocity = A4DirectPPO._velocity_fn(velocity_owner,state)
                pi = str(fm.time_convention)=='pi_convention'; n=int(fm.num_inference_steps)
                for i in range(n):
                    times=torch.full((1,),(n-i)/n if pi else i/n,device='cuda',dtype=torch.float32)
                    action=action+(-1. if pi else 1.)/n*velocity(action,times)
                    action[...,pad]=0
        return adapter.postprocess(action,batch,anchor)

    identity = CausalSessionIdentity(a.output.name,case['task'].replace('_',' '),cfg['instance_id'],
        f"{a.case}-{cfg['instance_id']}-seed{cfg['policy_seed']}",identities)
    def emit(kind, value):
        with (a.output/'events.jsonl').open('a') as f:
            f.write(json.dumps(dict(value,kind=kind,episode=identity.episode),allow_nan=False)+'\n')
    session = None; used = False; done = asyncio.Event(); gate = asyncio.Lock()

    async def handler(socket):
        nonlocal session, used
        if used:
            await socket.close(code=1013,reason='Exactly one fresh episode, no retry'); return
        used=True
        await socket.send(packb(dict(kind='fresh_full_episode_v1',config_sha256=file_sha(a.config),
            case=a.case,identity=asdict(identity))))
        try:
            async for payload in socket:
                request=unpackb(payload); op=request.pop('op')
                async with gate:
                    if request.pop('identity') != asdict(identity):
                        raise ValueError('Foreign episode/model/task identity')
                    if op=='begin' and not request and session is None:
                        session=FullEpisodeSession(identity,calibration=accepted['calibration'],planner=planner,
                            low=adapter,observer_factory=observer.bind,emit=emit)
                        result=dict(control_step=0)
                    elif op=='infer' and set(request)=={'observation'} and session is not None:
                        result=await asyncio.to_thread(session.infer,identity,request['observation'],sample)
                    elif op=='ack' and set(request)=={'token','controls'} and session is not None:
                        result=session.acknowledge(identity,request['token'],request['controls'])
                    elif op=='finish' and set(request)=={'observation'} and session is not None:
                        result=await asyncio.to_thread(session.close,identity,request['observation'])
                        after={k:await asyncio.to_thread(parameter_digest,v,frozen=True) for k,v in models.items()}
                        if before!=after or any(p.grad is not None or p.requires_grad for m in models.values() for p in m.parameters()):
                            raise ValueError('Read-only models changed')
                        verified=await asyncio.to_thread(observer.finish)
                        status.update(status='completed',finished=time.time(),result=result,
                            model_identity=asdict(identities),weights_unchanged=True,before=before,after=after,**verified)
                        atomic_json(a.output/'result.json',status)
                    else:
                        raise ValueError('Unknown, malformed or uninitialized request')
                    status.update(control_step=session.session.feedback.control_step,updated=time.time())
                    atomic_json(a.output/'status.json',status)
                    await socket.send(packb(dict(ok=True,result=result)))
                    if op=='finish':done.set(); return
            raise ConnectionError('Client disconnected before complete episode finish')
        except BaseException as e:
            status.update(status='failed_no_retry',error=repr(e),finished=time.time())
            atomic_json(a.output/'status.json',status); traceback.print_exc()
            try:await socket.send(packb(dict(ok=False,error=repr(e))))
            except Exception:pass
            done.set()

    async with websockets.serve(handler,'127.0.0.1',case['port'],compression=None,max_size=64<<20,ping_timeout=None):
        status.update(status='ready',model_identity=asdict(identities),restoration=restores,loaded=time.time())
        atomic_json(a.output/'status.json',status)
        await done.wait()


if __name__=='__main__': asyncio.run(main())
