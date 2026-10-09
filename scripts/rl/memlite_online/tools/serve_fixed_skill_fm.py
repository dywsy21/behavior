"""A800 native-FM paired diagnostics through loopback SSH; no optimization."""
import argparse
import asyncio
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
import time

REPO=Path(__file__).resolve().parents[4]
sys.path[:0]=[str(REPO/'src'),str(REPO/'scripts/eval/memlite_sft100'),str(Path(__file__).resolve().parents[1]/'code')]
from common import atomic_json
from recovery_corpus import file_sha
from fixed_skill_protocol import FixedSkillSession, low_prefix, validate_rgb_proprio
from wire import packb, unpackb


async def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--port',type=int,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    if subprocess.check_output(['git','status','--porcelain'],cwd=REPO,text=True).strip():raise ValueError('Dirty source')
    cfg=json.loads(a.config.read_text());root=Path(cfg['root'])
    gpu=os.environ.get('CUDA_VISIBLE_DEVICES')
    if gpu not in tuple(map(str,range(8))) or subprocess.check_output(
            ['nvidia-smi','-i',gpu,'--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
        raise ValueError('Select one verified idle A800 GPU')
    import numpy as np
    import torch
    import websockets
    from g05.utils.training.stage1_model import configuration, make_processor, restore_model
    from g05.models.g05.inferencer import PolicyInferencer
    from g05.models.g05.qwen35 import vision
    vision._flash_attn_varlen=None;vision._flash_attn_backend=None
    torch.set_num_threads(2)
    def forbid(*args,**kwargs):raise RuntimeError('No optimizer in fixed-skill evaluation')
    torch.optim.Adam.step=forbid;torch.optim.AdamW.step=forbid
    a.output.mkdir(parents=True)
    receipt=dict(status='loading',pid=os.getpid(),source_commit=subprocess.check_output(
        ['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),config_sha256=file_sha(a.config),
        optimizer_steps=0,planner_used=False,requests=0,models=cfg['models'])
    atomic_json(a.output/'status.json',receipt)
    manifest=json.loads((root/cfg['expert_release']/'manifest.json').read_text())
    config=configuration(root,'low',manifest['task_names'])
    if file_sha(config['stats_path'])!=cfg['stats_sha256']:raise ValueError('Changed action/state normalizer')
    processor=make_processor(config,False);models={}
    for name,item in cfg['models'].items():
        path=root/item['path']
        if file_sha(path)!=item['sha256']:raise ValueError('Unpinned low checkpoint')
        state=torch.load(path,map_location='cpu',mmap=True,weights_only=False)
        model,restored=restore_model(config,'low',state=state['model_state_dict'])
        model.requires_grad_(False).eval().cuda();models[name]=model
        atomic_json(a.output/(name+'-restoration.json'),restored)
        del state;gc.collect()
    gate=asyncio.Lock();stopped=asyncio.Event();episodes=set()

    @torch.no_grad()
    def infer(session,request,case):
        validate_rgb_proprio(request['images'],request['proprio'])
        raw=dict(task=case['task'].replace('_',' '),idx=0,embodiment='galaxea_r1pro',frequency=30,
            image_is_pad=torch.zeros(1,dtype=torch.bool),state_is_pad=torch.zeros(1,dtype=torch.bool),
            images={k:torch.from_numpy(v.copy())[None] for k,v in request['images'].items()},
            state={m['key']:torch.from_numpy(request['proprio'][m['start_index']:m['start_index']+m['raw_shape']].copy())[None]
                   for m in config['raw_shape']['state']})
        # Preserve absolute measured joint anchors before transforms/normalizers
        # touch their working copy (especially trunk/relative arm targets).
        anchor={k:raw['state'][k][-1:].clone()[None] for k in ('left_arm','right_arm','trunk_qpos')}
        prepared=processor._process_tensors(raw)
        prepared['samples']=low_prefix(processor.samples_builder,prepared,task=raw['task'],
            parent_goal=case['parent_goal'],semantic_bundle=case['semantic_bundle'])
        shape={'action':{k:torch.zeros(1,w) for k,w in
                       {'left_arm':7,'left_gripper':1,'right_arm':7,'right_gripper':1,'lower_body':7}.items()}}
        pad=processor.action_state_merger.forward(shape)['action_dim_is_pad']
        if torch.where(pad)[0].tolist()!=[7,8,17,18]:raise ValueError('Lost real base/trunk action dimensions')
        prepared['action_dim_is_pad']=pad
        for key in ('action','gt_action','action_is_pad'):prepared.pop(key,None)
        def move(value):
            if isinstance(value,torch.Tensor):return value.cuda()
            if isinstance(value,dict):return {k:move(v) for k,v in value.items()}
            if isinstance(value,list):return [move(v) for v in value]
            return value
        batch=move(PolicyInferencer._collate([prepared],padding_input_id=processor.pad_token_id))
        policy=models[session.bound['model']]
        if not policy.continuous_action or policy.discrete_action or policy.predict_cot:raise ValueError('Not native FM')
        # Reset the generator per case/seed/chunk. Interleaved sockets/models
        # cannot consume one another's noise stream; paired runs share noise.
        chunk=(request['control_step']-case['start_control'])//16
        with torch.random.fork_rng(devices=[0]):
            torch.manual_seed(session.bound['seed']+1000003*chunk)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                result=policy.forward_inference(samples=batch['samples'],pixel_values=batch['pixel_values'],
                                               action_dim_is_pad=batch['action_dim_is_pad'])
        if result.get('selected_action_source')!='fm':raise ValueError('Unexpected action source')
        cpu={k:v.cpu() for k,v in batch.items() if isinstance(v,torch.Tensor)}
        cpu.update(action=result['action'].cpu(),selected_action_source='fm')
        groups=PolicyInferencer._postprocess_single(cpu,0,processor,raw_state_anchor=anchor)
        order=[('base_qvel',3),('trunk_qpos',4),('left_arm',7),('left_gripper',1),('right_arm',7),('right_gripper',1)]
        action=torch.cat([torch.as_tensor(groups[k]).reshape(32,w) for k,w in order],-1)[:16].float().numpy()
        if action.shape!=(16,23) or not np.isfinite(action).all():raise ValueError('Invalid real-23 action chunk')
        session.acknowledge_chunk()
        return dict(action_chunk=action,episode=request['episode'],context_id=request['context_id'],
                    control_step=request['control_step'],model_sha256=cfg['models'][session.bound['model']]['sha256'])

    async def handler(socket):
        session=FixedSkillSession(cfg['cases'],cfg['models'])
        await socket.send(packb(dict(protocol='fixed_skill_fm_v1',models=cfg['models'],optimizer_steps=0)))
        try:
            async for payload in socket:
                request=unpackb(payload)
                async with gate:
                    if request.get('op')=='begin':
                        if request.get('episode') in episodes:raise ValueError('Reused episode ID')
                        response=session.begin(request);episodes.add(request['episode'])
                    else:
                        case=session.check_action(request);t=time.monotonic()
                        response=await asyncio.to_thread(infer,session,request,case)
                        elapsed=time.monotonic()-t;receipt['requests']+=1
                        with (a.output/'requests.jsonl').open('a') as f:
                            f.write(json.dumps(dict(case=session.bound['case'],model=session.bound['model'],
                                episode=request['episode'],context_id=request['context_id'],control_step=request['control_step'],
                                seed=session.bound['seed'],seconds=elapsed))+'\n')
                        atomic_json(a.output/'status.json',receipt)
                    await socket.send(packb(response))
        except websockets.ConnectionClosed:pass
        except Exception as error:
            with (a.output/'errors.jsonl').open('a') as f:f.write(json.dumps(dict(error=repr(error),session=session.bound))+'\n')
            await socket.send(packb(dict(error=repr(error))))
            await socket.close(code=1011,reason='Invalid fixed-skill request')

    async def watcher():
        while not (a.output/'STOP').exists():await asyncio.sleep(2)
        async with gate:
            # The inference-only no-grad path never constructs a trainer. Full
            # tensor comparison establishes no accidental in-place mutation.
            try:
                for name,model in models.items():
                    state=torch.load(root/cfg['models'][name]['path'],map_location='cpu',mmap=True,weights_only=False)['model_state_dict']
                    actual=model.state_dict()
                    if set(state)!=set(actual) or any(not torch.equal(v.detach().cpu(),state[k]) for k,v in actual.items()):
                        raise ValueError('Evaluation changed model weights')
                receipt.update(status='finished',weights_unchanged=True)
            except Exception as error:receipt.update(status='failed',error=repr(error))
            finally:
                atomic_json(a.output/'status.json',receipt);stopped.set()
    async with websockets.serve(handler,'127.0.0.1',a.port,compression=None,max_size=16<<20,ping_timeout=None):
        receipt.update(status='ready');atomic_json(a.output/'status.json',receipt)
        watcher_task=asyncio.create_task(watcher());await stopped.wait();watcher_task.cancel()


if __name__=='__main__':asyncio.run(main())
