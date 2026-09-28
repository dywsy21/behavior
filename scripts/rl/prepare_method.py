"""Register E2 once; never overwrites an E1 run or active runtime."""
from copy import deepcopy
import json
from pathlib import Path
import os
import numpy as np
import pyarrow.parquet as pq
from common import OUT,RUNTIME,PREVIOUS_RUNTIME,PARENT,PARENT_SHA,DATA,commit,sha,save
from g05.rl.protocol import EVAL_INSTANCES,EVAL_SEEDS,EVAL_LIMIT,validate_worker

E1=Path('/mnt/nvme_tmp/robodojo_g05_rl_20260928/e1_fp32')
DELTA=E1/'rl_batch_001_updates_0010.pt'
DELTA_SHA='4ccbe449c350807b96a473841cfa1cd1ab46aae6e59b0365b7066c8b2b8a1d36'
INSTANCE_ROOT=Path('/mnt/sdc1/xhz/BEHAVIOR2026/BEHAVIOR-1K/datasets/2026-challenge-task-instances/scene_test/public/house_double_floor_lower/json/house_double_floor_lower_task_turning_on_radio_instances')


def main():
    if (not OUT.name.startswith('e2_') or OUT.parent!=E1.parent or OUT.exists() or RUNTIME.exists()
            or not RUNTIME.name.startswith('rl_g05_50k_e2_')
            or PREVIOUS_RUNTIME.name!='rl_g05_50k_e1_fp32'):
        raise ValueError('New explicitly registered E2 output/runtime required')
    old=json.loads((E1/'manifest.json').read_text())
    ended=json.loads((E1/'supervisor.json').read_text())
    pids=json.loads((E1/'pids.json').read_text())
    if ended['status']!='completed': raise ValueError('E1 not complete')
    for pid in [ended['supervisor'],pids['learner'],*pids['simulators']]:
        file=Path(f'/proc/{pid}/stat')
        if file.exists() and file.read_text().rsplit(')',1)[1].split()[0]!='Z':
            raise ValueError('Old runtime is still in use')
    if sha(PARENT)!=PARENT_SHA or sha(DELTA)!=DELTA_SHA: raise ValueError('Weights changed')
    metadata=pq.read_table(DATA/'meta/episodes/chunk-000/file-000.parquet').to_pylist()
    heldout={(r['task_index'],r['task_instance_id']) for r in metadata if r['episode_index']%200>=190}
    workers=deepcopy(old['workers'])
    for spec in workers:
        spec['evaluation_only']=False
        validate_worker(spec,evaluation=False)
        if (spec['task_id'],spec['instance']) in heldout: raise ValueError('Held-out TRAIN instance')
        if sha(spec['actions'])!=spec['actions_sha256']: raise ValueError('Changed TRAIN actions')
        actions=np.load(spec['actions'],allow_pickle=False)
        if actions.ndim!=2 or actions.shape[1]!=23 or not np.isfinite(actions).all():
            raise ValueError('Invalid real controls')
    evaluation=[]
    for w,instance in enumerate(EVAL_INSTANCES):
        file=INSTANCE_ROOT/f'house_double_floor_lower_task_turning_on_radio_0_{instance}_template-tro_state.json'
        spec=dict(worker=w,gpu=w+2,task='turning_on_radio',task_id=0,split='public_test',
                  instance=instance,seed=0,prefix_controls=0,evaluation_only=True,
                  instance_file=str(file),instance_file_sha256=sha(file))
        validate_worker(spec,evaluation=True); evaluation.append(spec)
    manifest=dict(source_commit=commit(),entry='method',parent=str(PARENT),parent_sha256=PARENT_SHA,
        resume_checkpoint=str(DELTA),resume_sha256=DELTA_SHA,workers=workers,
        sim_pools=dict(baseline=evaluation,training=workers,final=deepcopy(evaluation)),
        max_controls=100000,max_active_wall_seconds=43200,max_training_controls=50000,
        max_training_seconds=25200,final_eval_reserved_seconds=10800,
        final_eval_reserved_controls=6*EVAL_LIMIT,max_batches=16,max_new_actor_updates=400,
        max_actor_updates=410,episode_controls=1024,critic_steps_per_batch=2,
        learning_rates=dict(action_expert=1e-8,noise=1e-7,critic=1e-4),
        backtracking_scales=[1.,.5,.25,.125,.0625,.03125],bounded_backtracking=True,
        ae_precision='float32',eval_seeds=list(EVAL_SEEDS),eval_control_limit=EVAL_LIMIT,
        train_policy_rng='resume saved Torch/CUDA and learner Python RNG',
        evaluation_initial_noise='independent per-worker generator, same seed per paired episode',
        bc_weight=.1,clip=.05,target_path_kl=.01,gamma_control=.9998,lambda_chunk=.95,
        previous_runtime=str(PREVIOUS_RUNTIME),runtime=str(RUNTIME),
        hypothesis='fresh on-policy training improves fixed original-reset complete-task development success rate',
        training_curriculum='first terminal -96; three successive TRAIN successes at one prefix moves it 96 earlier, max gap768',
        benchmark_receipt=old['benchmark_receipt'],repeat_throughput_benchmark=False)
    OUT.mkdir(parents=True,exist_ok=False); save(OUT/'manifest.json',manifest)
    print(json.dumps(dict(run=str(OUT),source=manifest['source_commit'],max_controls=100000,
                         eval_cases=[(i,s) for i in EVAL_INSTANCES for s in EVAL_SEEDS])))


if __name__=='__main__': main()
