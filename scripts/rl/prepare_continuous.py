"""Prepare E4 ONLY after E3 finishes its registered six-pair evaluation."""
from copy import deepcopy
import json
from pathlib import Path

import pyarrow.parquet as pq

from common import OUT,RUNTIME,PREVIOUS_RUNTIME,DATA,PARENT,PARENT_SHA,commit,save,sha
from g05.rl.continuous import AUTHORIZATION,CAPS,MIN_FREE_BYTES,check_storage
from g05.rl.continuous_handoff import (PRIOR_SOURCE,dependency_state,owned_pids,live_pids,check_closed_pool)
from g05.rl.dense_recipe import validate_recipe
from g05.rl.dense_resume import resume_state
from g05.rl.protocol import validate_worker
from g05.rl.recovery import count_physical_steps

PRIOR=Path('/mnt/nvme_tmp/robodojo_g05_rl_20260928/e3_render_resume_v1')
EXPECTED_OUT=PRIOR.parent/'e4_continuous_v1'
EXPECTED_RUNTIME=Path('/mnt/nvme_tmp/robodojo_sim_runtime_20260925/rl_g05_50k_e4_continuous_v1')
EXPECTED_PREVIOUS=EXPECTED_RUNTIME.parent/'rl_g05_50k_e3_render_resume_v1/training'


def main():
    if (OUT!=EXPECTED_OUT or RUNTIME!=EXPECTED_RUNTIME or PREVIOUS_RUNTIME!=EXPECTED_PREVIOUS
            or OUT.exists() or RUNTIME.exists()):
        raise ValueError('Expected fresh E4 output/private runtime, never overwrite a previous run')
    ready=dependency_state(PRIOR)
    if not ready['ready'] or live_pids(owned_pids(PRIOR)):
        raise ValueError('E3 dependency still running')
    source=commit(); inputs={}
    def read(path,lines=False):
        inputs[str(path)]=sha(path)
        value=path.read_text()
        return [json.loads(line) for line in value.splitlines()] if lines else json.loads(value)
    old=read(PRIOR/'manifest.json'); validate_recipe(old)
    if old['source_commit']!=PRIOR_SOURCE: raise ValueError('Wrong E3 source')
    for path,expected in old['audit_input_sha256'].items():
        if sha(path)!=expected: raise ValueError('Previous sealed evidence changed: '+path)
    for name in ('supervisor.json','status.json','pids.json','training_result.json',
                 'frozen_final_selection.json','result.json','evaluations.jsonl'):
        read(PRIOR/name,lines=name.endswith('.jsonl'))
    pool_counts={}
    for pool,phases in [('training',{'expert_prefix','policy'}),('final',{'eval_rl_fp32'})]:
        counts=[]
        for worker in (0,1):
            path=PRIOR/f'{pool}/worker_{worker}/steps.jsonl'
            inputs[str(path)]=sha(path)
            with path.open() as stream:
                counts.append(count_physical_steps((json.loads(line) for line in stream),allowed_phases=phases))
            # Connections enter the IPC dictionary in nondeterministic arrival
            # order. Validate each named worker receipt, not its list position.
            if read(PRIOR/f'{pool}/worker_{worker}_closed.json')['total_controls']!=counts[-1]:
                raise ValueError('Named worker close receipt differs from physical log')
        pool_counts[pool]=check_closed_pool(read(PRIOR/f'{pool}/closed.json'),counts)
    if (old['training_resume']['controls']+pool_counts['training']!=ready['training_controls']
            or pool_counts['final']!=ready['evaluation_controls']):
        raise ValueError('Physical TRAIN/final logs disagree with dependency counters')
    directories=[PRIOR.parent/'e3_dense_v1',PRIOR.parent/'e3_no_wall_v1',PRIOR]
    batches=[read(path) for directory in directories for path in sorted(directory.glob('train_batch_*.json'))]
    checkpoint=Path(ready['checkpoint']); receipt=read(checkpoint.with_suffix('.json'))
    if (str(checkpoint)!=batches[-1]['checkpoint'] or receipt['sha256']!=ready['sha256']
            or sha(checkpoint)!=ready['sha256'] or receipt['parent_sha256']!=PARENT_SHA
            or sha(PARENT)!=PARENT_SHA or receipt['controls']!=ready['training_controls']):
        raise ValueError('Parent/delta/last completed update mismatch')
    initial=read(directories[0]/'manifest.json'); validate_recipe(initial)
    progress=resume_state(initial,batches,receipt,ready['training_controls'],
                          continuation_authorization=AUTHORIZATION)
    metadata=pq.read_table(DATA/'meta/episodes/chunk-000/file-000.parquet').to_pylist()
    heldout={(r['task_index'],r['task_instance_id']) for r in metadata if r['episode_index']%200>=190}
    workers=deepcopy(old['workers'])
    for spec in workers:
        spec['prefix_controls']=progress['prefixes'][str(spec['worker'])]
        validate_worker(spec,evaluation=False)
        if (spec['task_id'],spec['instance']) in heldout: raise ValueError('Held-out TRAIN source')
        for name in ('actions','source_parquet','annotation'):
            if sha(spec[name])!=spec[name+'_sha256']: raise ValueError('TRAIN source changed')
    for spec in old['sim_pools']['final']:
        validate_worker(spec,evaluation=True)
        if sha(spec['instance_file'])!=spec['instance_file_sha256']: raise ValueError('Eval source changed')
    manifest=deepcopy(old)
    manifest.update({key:None for key in CAPS})
    manifest.update(source_commit=source,experiment='RL-G05-50K-E4',training_limit_override=AUTHORIZATION,
        continuation_reason=AUTHORIZATION,continued_from=str(PRIOR),
        hypothesis='test whether continued on-policy learning improves earlier takeover and full-reset success without a total training budget',
        resume_checkpoint=str(checkpoint),resume_sha256=ready['sha256'],training_resume=progress,
        prior_evaluation_controls=ready['evaluation_controls'],min_free_disk_bytes=MIN_FREE_BYTES,
        checkpoint_retention='retain_all_no_automatic_deletion',
        prior_active_seconds=old['prior_active_seconds']+json.loads((PRIOR/'supervisor.json').read_text())['seconds'],
        workers=workers,sim_pools=dict(training=workers,final=deepcopy(old['sim_pools']['final'])),
        previous_runtime=str(PREVIOUS_RUNTIME),runtime=str(RUNTIME),
        audit_input_sha256={**old['audit_input_sha256'],**inputs})
    validate_recipe(manifest); check_storage(manifest,OUT.parent)
    OUT.mkdir(parents=True,exist_ok=False)
    save(OUT/'manifest.json',manifest)
    save(OUT/'prior_control_audit.json',dict(training_resume=progress,pool_controls=pool_counts,
        dependency=ready,eval_rollouts_used_for_training=False,all_updates_preserved=True,
        cumulative_physical_controls=ready['training_controls']+ready['evaluation_controls']))
    print(json.dumps(dict(run=str(OUT),source=source,training_resume=progress,unlimited_training=True)))


if __name__=='__main__': main()
