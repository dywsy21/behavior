"""One verified champion parent, isolated2026 episode, no optimization."""
import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

REPO=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(REPO/'src'),str(REPO/'scripts/eval/memlite_sft100')]
from common import atomic_json
from behavior_champion.inference import file_sha, checked_sources

STATUS_PATH=None


async def main():
    global STATUS_PATH
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    a=ap.parse_args();cfg=json.loads(a.config.read_text())
    if a.output.exists() or cfg['schema']!='champion2026_readonly_full_episode_v1':
        raise ValueError('Fresh registered champion inference output required')
    if subprocess.check_output(['git','status','--porcelain'],cwd=REPO,text=True).strip():
        raise ValueError('Use a clean frozen checkout')
    gpu=str(cfg['model_gpu'])
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=gpu or subprocess.check_output(
        ['nvidia-smi','-i',gpu,'--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
        raise ValueError('Registered model GPU must be idle')
    root=Path(cfg['a800_root']);reference=root/cfg['reference'];openpi=root/cfg['openpi_reference']
    checked_sources(reference,openpi,REPO)
    verified=root/cfg['verified']
    if file_sha(verified)!=cfg['verified_sha256']:
        raise ValueError('Changed public-asset verification receipt')
    receipt=json.loads(verified.read_text())
    if receipt['revision']!=cfg['revision'] or receipt['status']!='all_selected_assets_verified':
        raise ValueError('Wrong public parent')
    a.output.mkdir(parents=True)
    STATUS_PATH=a.output/'status.json'
    status=dict(status='validating_assets',pid=os.getpid(),started=time.time(),optimizer_updates=0,
        config_sha256=file_sha(a.config),asset_receipt_sha256=cfg['verified_sha256'],
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip())
    atomic_json(a.output/'status.json',status)
    asset_root=(root/cfg['asset_root']).resolve()
    for row in receipt['files']:
        path=(asset_root/row['path']).resolve()
        if not path.is_relative_to(asset_root) or path.stat().st_size!=row['size'] or file_sha(path)!=row['sha256']:
            raise ValueError('Changed public checkpoint asset '+row['path'])
    os.environ.update(WANDB_MODE='disabled',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
        XLA_PYTHON_CLIENT_PREALLOCATE='false',JAX_COMPILATION_CACHE_DIR=str(a.output/'jax-cache'))
    sys.path[:0]=[str(reference/'src'),str(openpi/'src')]
    import numpy as np
    import jax
    import flax.nnx as nnx
    import websockets
    from behavior_champion.inference import ChampionRuntime
    from behavior_champion.session import ChampionSession
    from b1k.shared.correction_rules import apply_correction_rules, check_gripper_variation
    from wire import packb,unpackb
    layout=json.loads((REPO/cfg['layout']).read_text())['layout']
    model_task=layout['dataset_to_model'][str(cfg['task_id'])]
    if model_task not in cfg['trained_model_tasks']:
        raise ValueError('This specialist was not trained for the requested task')
    stages=layout['num_stages'][:cfg['model_task_count']]
    status.update(status='loading_model',assets_rehashed=len(receipt['files']))
    atomic_json(a.output/'status.json',status)
    runtime=ChampionRuntime(root/cfg['checkpoint'],stages)
    def parameter_digest():
        digest=hashlib.sha256()
        for path,value in nnx.state(runtime.policy._model,nnx.Param).flat_state().items():
            array=np.asarray(value.value)
            digest.update(repr((path,array.shape,str(array.dtype))).encode());digest.update(array.tobytes())
        return digest.hexdigest()
    before=parameter_digest()
    def make_session(episode,task):
        mt=layout['dataset_to_model'][str(task)]
        if mt not in cfg['trained_model_tasks']:raise ValueError('Specialist/task mismatch')
        return ChampionSession(episode=episode,task=task,model_task=mt,num_stages=stages[mt],
            seed=cfg['policy_seed'],predictor=runtime.infer,correct_actions=apply_correction_rules,
            gripper_variation=check_gripper_variation,apply_legacy_rules=cfg['apply_legacy_rules'])
    # Target-free actual-model session isolation, with two TRAIN images. Never
    # infer success or tune a checkpoint from these engineering repetitions.
    probe=root/cfg['probe'];probe_meta=json.loads((probe/'result.json').read_text())
    if file_sha(probe/'native.npz')!=probe_meta['native_npz_sha256']:
        raise ValueError('Changed registered actual-observation probe')
    native=np.load(probe/'native.npz',allow_pickle=False)
    arrays={key:native[key] for key in ('task_id','state61','base_0_rgb','left_wrist_0_rgb','right_wrist_0_rgb')}
    observations={}
    for task in (0,1):
        index=np.where(arrays['task_id']==task)[0]
        if len(index)!=1:raise ValueError('Missing exact task probe')
        i=int(index[0])
        observations[task]=dict(proprio=arrays['state61'][i],images={
            name:arrays[key][i].transpose(2,0,1) for name,key in
            [('head_rgb','base_0_rgb'),('left_wrist_rgb','left_wrist_0_rgb'),('right_wrist_rgb','right_wrist_0_rgb')]})
    def call_and_ack(session,obs):
        result=session.infer(obs)
        session.ack(token=result['token'],controls=[dict(control_step=session.control_step+i,
            simulator_apply_ack=True,action_executed_raw23=x.tolist()) for i,x in enumerate(result['action_chunk'])])
        # This is an ENGINEERING clock replay, explicitly not physical ACKs.
        return result
    status.update(status='actual_model_isolation_preflight');atomic_json(a.output/'status.json',status)
    expected={}
    for task in (0,1):
        s=make_session('preflight-'+str(task),task);s.begin()
        expected[task]=[call_and_ack(s,observations[task]) for _ in range(3)]
    pair={task:make_session('preflight-'+str(task),task) for task in (0,1)}
    for s in pair.values():s.begin()
    for turn in range(3):
        for task in (1,0):
            actual=call_and_ack(pair[task],observations[task]);wanted=expected[task][turn]
            np.testing.assert_array_equal(actual['action_chunk'],wanted['action_chunk'])
            if actual['display']!=wanted['display'] or actual['token']!=wanted['token']:
                raise ValueError('Model sessions changed under interleaving')
    atomic_json(a.output/'preflight.json',dict(status='actual_model_interleaving_exact',
        tasks=[0,1],calls=12,physical_controls=0,engineering_clock_replay_only=True,
        optimizer_updates=0,parameters_before=before))
    del native,arrays,observations,pair,expected
    done=asyncio.Event();accepted=False
    identity=hashlib.sha256((status['config_sha256']+status['asset_receipt_sha256']+before).encode()).hexdigest()
    (a.output/'observations').mkdir()
    async def handler(socket):
        nonlocal accepted
        if accepted:await socket.close(code=1008,reason='One fixed episode per process');return
        accepted=True
        session=make_session(cfg['case']+'-301-seed'+str(cfg['policy_seed']),cfg['task_id'])
        await socket.send(packb(dict(kind='champion2026_episode_v1',identity=identity,
            case=cfg['case'],config_sha256=status['config_sha256'],chunk_size=20)))
        try:
            async for payload in socket:
                request=unpackb(payload)
                if request.pop('identity',None)!=identity:raise ValueError('Foreign model/session identity')
                op=request.pop('op',None)
                if op=='begin' and not request:result=session.begin()
                elif op=='infer' and set(request)=={'observation'}:
                    obs=request['observation'];started=time.monotonic()
                    result=session.infer(obs)
                    name=f'{session.predictions:06d}.npz';path=a.output/'observations'/name
                    np.savez_compressed(path,proprio=obs['proprio'],**obs['images'])
                    with (a.output/'events.jsonl').open('a') as stream:
                        stream.write(json.dumps(dict(kind='infer',token=result['token'],display=result['display'],
                            observation=name,observation_sha256=file_sha(path),
                            action_sha256=hashlib.sha256(result['action_chunk'].tobytes()).hexdigest(),
                            seconds=time.monotonic()-started),allow_nan=False)+'\n')
                elif op=='ack' and set(request)=={'token','controls'}:result=session.ack(**request)
                elif op=='finish' and not request:
                    result=session.finish();after=parameter_digest()
                    if after!=before:raise ValueError('Read-only model parameters changed')
                    status.update(status='completed',finished=time.time(),result=result,
                        parameters_before=before,parameters_after=after,weights_unchanged=True)
                    atomic_json(a.output/'result.json',status)
                else:raise ValueError('Unknown or malformed request')
                status.update(control_step=session.control_step,updated=time.time())
                atomic_json(a.output/'status.json',status)
                await socket.send(packb(dict(ok=True,result=result)))
                if op=='finish':done.set();return
            raise ConnectionError('Disconnected before official episode finish')
        except BaseException as e:
            status.update(status='failed_no_retry',error=repr(e),finished=time.time())
            atomic_json(a.output/'status.json',status);traceback.print_exc()
            try:await socket.send(packb(dict(ok=False,error=repr(e))))
            except Exception:pass
            done.set()
    async with websockets.serve(handler,'127.0.0.1',cfg['port'],compression=None,max_size=64<<20,ping_timeout=None):
        status.update(status='ready',identity=identity,parameters_before=before,loaded=time.time())
        atomic_json(a.output/'status.json',status)
        await done.wait()


if __name__=='__main__':
    try:asyncio.run(main())
    except BaseException as error:
        if STATUS_PATH is not None and STATUS_PATH.exists():
            failed=json.loads(STATUS_PATH.read_text())
            failed.update(status='failed_no_retry',error=repr(error),finished=time.time())
            atomic_json(STATUS_PATH,failed)
        raise
