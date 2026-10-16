"""Verified champion policy through the unmodified official websocket protocol.

One environment per connection; reset has no reply; no physical ACK is invented.
The registered loopback protocol probe is not a submission or formal training.
"""
import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

REPO=Path(__file__).resolve().parents[2]
ENTRY_REPO=REPO
sys.path[:0]=[str(REPO/'src'),str(REPO/'scripts/eval/memlite_sft100')]
from common import atomic_json


async def run(args):
    global REPO
    REPO=ENTRY_REPO
    cfg=json.loads(args.config.read_text())
    recovery_job=getattr(args,'recovery_job',None)
    admission=None
    if recovery_job is not None:
        import champion_worker_recovery as recovery
        _,registration=recovery.registration(recovery_job,int(cfg['model_gpu']))
        REPO=Path(registration['runtime_source'])
        sys.path.insert(0,str(REPO/'src'))
        recovery.check_model_request(recovery_job,registration,args.config,args.output)
    from behavior_champion.inference import file_sha,checked_sources
    if cfg['schema']!='champion2026_official_websocket_v1' or args.output.exists():
        raise ValueError('Registered official-protocol config and fresh run required')
    if cfg['host']!='127.0.0.1':
        raise ValueError('This engineering entry is loopback-only, not public submission hosting')
    if subprocess.check_output(['git','status','--porcelain'],cwd=REPO,text=True).strip():
        raise ValueError('Clean frozen source required')
    gpu=str(cfg['model_gpu'])
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=gpu:
        raise ValueError('Registered model GPU must be idle')
    if recovery_job is not None:
        from champion_peer_admission import check
        admission=check(registration['parent'],registration['parent_manifest']['sha256'],int(gpu),
                        recoveries=registration.get('peer_recoveries', []))
    elif subprocess.check_output(['nvidia-smi','-i',gpu,'--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
        raise ValueError('Registered model GPU must be idle')
    root=Path(cfg['a800_root']);reference=root/cfg['reference'];openpi=root/cfg['openpi_reference']
    checked_sources(reference,openpi,REPO)
    from behavior_champion.policy_assets import verify_policy_assets
    checked=verify_policy_assets(root,REPO,cfg)
    stages=checked['layout']['num_stages'][:cfg['model_task_count']]
    mapping=checked['layout']['dataset_to_model']
    allowed=cfg['allowed_task_ids']
    if (not allowed or len(set(allowed))!=len(allowed) or any(type(t) is not int or str(t) not in mapping
            or mapping[str(t)] not in cfg['trained_model_tasks'] for t in allowed)):
        raise ValueError('Task scope must be covered by the verified model')
    if cfg['num_inference_steps']!=20:
        raise ValueError('Only the verified20-step champion runtime is supported')
    args.output.mkdir(parents=True)
    status=dict(status='loading',pid=os.getpid(),started=time.time(),optimizer_updates=0,
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
        config_sha256=file_sha(args.config),asset_receipt_sha256=cfg['verified_sha256'],
        physical_execution_evidence=False,allowed_task_ids=allowed,memory_profiling=args.profile_memory)
    if recovery_job is not None:
        # Provenance distinguishes this admission-only entry from the unchanged
        # actor modules imported from the original frozen policy checkout.
        admission=check(registration['parent'],registration['parent_manifest']['sha256'],int(gpu),
                        recoveries=registration.get('peer_recoveries', []))
        status.update(launcher_source_commit=registration['source_commit'],
                      recovery_manifest_sha256=file_sha(recovery_job/'manifest.json'),
                      registered_peer_admission=admission)
    atomic_json(args.output/'status.json',status)
    memory_profile = None
    if args.profile_memory:
        from behavior_champion.memory_profile import ProcessMemoryProfile
        memory_profile = ProcessMemoryProfile(args.output,gpu)
        memory_profile.start()
    os.environ.update(WANDB_MODE='disabled',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
        XLA_PYTHON_CLIENT_PREALLOCATE='false',JAX_COMPILATION_CACHE_DIR=str(args.output/'jax-cache'))
    sys.path[:0]=[str(reference/'src'),str(openpi/'src')]
    import numpy as np
    import flax.nnx as nnx
    from b1k.shared.correction_rules import apply_correction_rules,check_gripper_variation
    from behavior_champion.inference import ChampionRuntime
    from behavior_champion.session import ChampionSession
    from behavior_champion.official import OfficialPolicyServer
    from wire import packb,unpackb
    runtime=ChampionRuntime(root/cfg['checkpoint'],stages,expected_model=checked['expected_model'])
    def memory_snapshot(label):
        if memory_profile is not None:
            import jax
            memory_profile.record_jax(label,jax.devices()[0].memory_stats())
    memory_snapshot('model_loaded')
    def digest():
        sha=hashlib.sha256()
        for path,value in nnx.state(runtime.policy._model,nnx.Param).flat_state().items():
            array=np.asarray(value.value);sha.update(repr((path,array.shape,str(array.dtype))).encode())
            sha.update(array.tobytes())
        return sha.hexdigest()
    before=digest()
    def factory(episode,task):
        if task not in allowed:raise ValueError('Task outside registered model scope')
        model_task=mapping[str(task)]
        return ChampionSession(episode=episode,task=task,model_task=model_task,num_stages=stages[model_task],
            seed=cfg['policy_seed'],predictor=runtime.infer,correct_actions=apply_correction_rules,
            gripper_variation=check_gripper_variation,
            apply_legacy_rules=cfg['apply_legacy_rules'] and model_task<50,clock_kind='policy_delivery')
    def journal(row):
        with (args.output/'events.jsonl').open('a') as stream:
            stream.write(json.dumps(dict(time=time.time(),**row),allow_nan=False)+'\n')
        if row['kind']=='delivered' and row.get('generated_display') is not None:
            memory_snapshot('prediction_delivered')
    server=OfficialPolicyServer(factory,metadata=dict(policy='champion2026',action_chunk_size=20,
        action_dimension=23,environment_batch_size=1,allowed_task_ids=allowed,
        source_commit=status['source_commit'],config_sha256=status['config_sha256'],
        clock_kind='policy_delivery',physical_execution_evidence=False),
        packb=packb,unpackb=unpackb,journal=journal,max_connections=cfg['max_connections'])
    stop=asyncio.Event()
    for sig in (signal.SIGTERM,signal.SIGINT):
        asyncio.get_running_loop().add_signal_handler(sig,stop.set)
    def ready(_):
        if memory_profile is not None:memory_profile.phase='serving'
        status.update(status='ready',parameters_before=before,loaded=time.time())
        atomic_json(args.output/'status.json',status)
    try:
        await server.run(host=cfg['host'],port=cfg['port'],stop=stop,ready=ready)
        after=digest()
        if after!=before:raise ValueError('Read-only policy parameters changed')
        if memory_profile is not None:
            memory_snapshot('server_closed')
            status['memory_profile']=memory_profile.finish()
            atomic_json(args.output/'memory-profile.json',status['memory_profile'])
        status.update(status='closed',finished=time.time(),parameters_after=after,weights_unchanged=True)
        atomic_json(args.output/'result.json',status)
    except BaseException as error:
        status.update(status='failed_no_retry',error=repr(error),finished=time.time())
        raise
    finally:atomic_json(args.output/'status.json',status)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--profile-memory',action='store_true',help='Read-only per-PID NVML sampling plus JAX allocator statistics; not a hardware-limit test')
    parser.add_argument('--recovery-job',type=Path,help='Explicit isolated-worker registration; admits only verified live peer-simulator auxiliary contexts')
    asyncio.run(run(parser.parse_args()))
