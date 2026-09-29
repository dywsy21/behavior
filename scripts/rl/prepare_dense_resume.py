"""Continue this E3 without wall limits; retain the entire original step ledger."""
from copy import deepcopy
import json
from pathlib import Path
import shutil

import pyarrow.parquet as pq

from common import OUT,RUNTIME,PREVIOUS_RUNTIME,DATA,PARENT,PARENT_SHA,commit,save,sha
from g05.rl.dense_recipe import validate_recipe
from g05.rl.dense_resume import resume_state,reconcile_operator_stop
from g05.rl.protocol import validate_worker
from g05.rl.recovery import count_physical_steps
from g05.rl.time_limits import NO_TRAINING_WALL
from stop_dense_at_boundary import PRIOR,SOURCE_SHA


def main():
    if (OUT!=PRIOR.parent/'e3_no_wall_v1' or OUT.exists() or RUNTIME.exists()
            or RUNTIME!=Path('/mnt/nvme_tmp/robodojo_sim_runtime_20260925/rl_g05_50k_e3_no_wall_v1')
            or PREVIOUS_RUNTIME!=Path('/mnt/nvme_tmp/robodojo_sim_runtime_20260925/rl_g05_50k_e3_dense_v1/training')):
        raise ValueError('Expected a new registered continuation, not the active source/run')
    if shutil.disk_usage(OUT.parent).free<400*(1<<30):
        raise ValueError('Insufficient reserved checkpoint space')
    inputs={}
    def read(path):
        inputs[str(path)]=sha(path)
        return json.loads(path.read_text())
    old=read(PRIOR/'manifest.json'); validate_recipe(old)
    request=read(PRIOR/'operator_restart_request.json')
    ended=read(PRIOR/'supervisor.json'); pids=read(PRIOR/'pids.json')
    if old['source_commit']!=SOURCE_SHA or ended['status']!='failed':
        raise ValueError('Original E3 has not stopped for the requested restart')
    owned={ended['supervisor'],ended['learner']}
    for group in pids['history']: owned.update([group['learner'],*group['simulators']])
    for pid in owned:
        path=Path(f'/proc/{pid}/stat')
        if path.exists() and path.read_text().rsplit(')',1)[1].split()[0]!='Z':
            raise ValueError('Previous learner/runtime is still alive')
    status=read(PRIOR/'status.json'); failure=read(PRIOR/'failure.json')
    if (request['learner_pid']!=ended['learner'] or request['source_commit']!=SOURCE_SHA
            or failure['error']!="SystemExit('Owned E3 stopped by signal 15')"):
        raise ValueError('Old run stopped for a different reason; do not silently restart')
    closed=read(PRIOR/'training/closed.json')
    counts=[]; policy_controls=0
    for worker in (0,1):
        path=PRIOR/f'training/worker_{worker}/steps.jsonl'
        inputs[str(path)]=sha(path)
        rows=[json.loads(line) for line in path.read_text().splitlines()]
        counts.append(count_physical_steps(rows,allowed_phases={'expert_prefix','policy'}))
        worker_close=read(PRIOR/f'training/worker_{worker}_closed.json')
        if (not worker_close['closed'] or worker_close['total_controls']!=counts[-1]
                or rows[-1]['phase']!='expert_prefix'):
            raise ValueError('Missing exact physical close receipt or stop after prefix boundary')
        policy_controls+=sum(r['phase']=='policy' for r in rows)
    unresolved=read(PRIOR/'unresolved_training.json') if (PRIOR/'unresolved_training.json').exists() else {}
    nudge=read(PRIOR/'operator_close_nudge.json') if (PRIOR/'operator_close_nudge.json').exists() else {}
    reconciliation=reconcile_operator_stop(closed,counts,unresolved,
        operator_nudge=(nudge.get('pid')==request['learner_pid'] and nudge.get('signal')=='SIGTERM'
                        and nudge.get('no_actor_update_or_new_control_dispatch') is True))
    batches=[read(p) for p in sorted(PRIOR.glob('train_batch_*.json'))]
    checkpoint=Path(batches[-1]['checkpoint']); receipt=read(checkpoint.with_suffix('.json'))
    if (str(checkpoint)!=request['checkpoint'] or sha(checkpoint)!=request['checkpoint_sha256']
            or receipt['sha256']!=request['checkpoint_sha256'] or sha(PARENT)!=PARENT_SHA
            or status['actor_updates']!=receipt['actor_updates']
            or status['critic_updates']!=receipt['critic_updates']
            or len(batches)!=request['completed_batches']
            or policy_controls!=sum(b['policy_controls'] for b in batches)):
        raise ValueError('Lost an update/rollout or wrong parent/delta at restart')
    progress=resume_state(old,batches,receipt,sum(counts))
    metadata=pq.read_table(DATA/'meta/episodes/chunk-000/file-000.parquet').to_pylist()
    heldout={(r['task_index'],r['task_instance_id']) for r in metadata if r['episode_index']%200>=190}
    workers=deepcopy(old['workers'])
    for spec in workers:
        spec['prefix_controls']=progress['prefixes'][str(spec['worker'])]
        validate_worker(spec,evaluation=False)
        if sha(spec['actions'])!=spec['actions_sha256'] or (spec['task_id'],spec['instance']) in heldout:
            raise ValueError('TRAIN source identity/split changed')
    for spec in old['sim_pools']['final']:
        validate_worker(spec,evaluation=True)
        if sha(spec['instance_file'])!=spec['instance_file_sha256']:
            raise ValueError('Fixed evaluation instance changed')
    for path,expected in old['audit_input_sha256'].items():
        if sha(path)!=expected: raise ValueError('Prior sealed evidence changed: '+path)
    manifest=deepcopy(old)
    manifest.update(source_commit=commit(),continued_from=str(PRIOR),
        resume_checkpoint=str(checkpoint),resume_sha256=receipt['sha256'],
        max_training_seconds=None,max_active_wall_seconds=None,time_limit_override=NO_TRAINING_WALL,
        training_resume=progress,prior_active_seconds=ended['seconds'],
        workers=workers,sim_pools=dict(training=workers,final=deepcopy(old['sim_pools']['final'])),
        previous_runtime=str(PREVIOUS_RUNTIME),runtime=str(RUNTIME),
        audit_input_sha256={**old['audit_input_sha256'],**inputs})
    validate_recipe(manifest)
    OUT.mkdir(parents=True,exist_ok=False)
    save(OUT/'manifest.json',manifest)
    save(OUT/'prior_control_audit.json',dict(worker_controls=counts,controls=sum(counts),
        policy_controls=policy_controls,completed_batches=len(batches),all_updates_preserved=True,
        checkpoint=str(checkpoint),sha256=receipt['sha256'],training_resume=progress,
        stop_reconciliation=reconciliation))
    print(json.dumps(dict(run=str(OUT),source=manifest['source_commit'],training_resume=progress,
                         max_training_seconds=None,max_active_wall_seconds=None,new_gpu_processes=0)))


if __name__=='__main__': main()
