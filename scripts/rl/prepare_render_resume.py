"""One registered E3 continuation from173; no new budget or replay of partial PPO."""
from copy import deepcopy
import json
from pathlib import Path
import shutil

import pyarrow.parquet as pq

from common import OUT,RUNTIME,PREVIOUS_RUNTIME,DATA,PARENT,PARENT_SHA,commit,save,sha
from g05.rl.dense_recipe import validate_recipe
from g05.rl.dense_resume import resume_state
from g05.rl.protocol import validate_worker
from g05.rl.render_resume import audit_render_failure,reconcile_render_stop

PRIOR = Path('/mnt/nvme_tmp/robodojo_g05_rl_20260928/e3_no_wall_v1')
FIRST = PRIOR.parent/'e3_dense_v1'
SOURCE = '25d868ebad96f85608aa069d4355801afae35ac3'
DELTA_SHA = 'daa37a25399a7f40b30bdbc445354eec1acbeb89d24370f825ae680b5f34c3fc'


def main():
    if (OUT != PRIOR.parent/'e3_render_resume_v1' or OUT.exists() or RUNTIME.exists()
            or RUNTIME != Path('/mnt/nvme_tmp/robodojo_sim_runtime_20260925/rl_g05_50k_e3_render_resume_v1')
            or PREVIOUS_RUNTIME != Path('/mnt/nvme_tmp/robodojo_sim_runtime_20260925/rl_g05_50k_e3_no_wall_v1/training')):
        raise ValueError('Expected new registered output and private runtime')
    if shutil.disk_usage(OUT.parent).free < 400*(1<<30):
        raise ValueError('Insufficient checkpoint reserve')
    inputs={}
    def read(path, lines=False):
        inputs[str(path)]=sha(path)
        text=path.read_text()
        return [json.loads(line) for line in text.splitlines()] if lines else json.loads(text)
    old=read(PRIOR/'manifest.json');validate_recipe(old)
    ended=read(PRIOR/'supervisor.json');pids=read(PRIOR/'pids.json')
    if old['source_commit']!=SOURCE or ended['source_commit']!=SOURCE or ended['status']!='failed':
        raise ValueError('Unexpected previous run identity/status')
    owned={ended['supervisor'],ended['learner']}
    for group in pids['history']:owned.update([group['learner'],*group['simulators']])
    for pid in owned:
        path=Path(f'/proc/{pid}/stat')
        if path.exists() and path.read_text().rsplit(')',1)[1].split()[0]!='Z':
            raise ValueError('Previous process is still alive')
    for path,expected in old['audit_input_sha256'].items():
        if sha(path)!=expected:raise ValueError('Previous sealed evidence changed: '+path)
    inherited=read(PRIOR/'prior_control_audit.json')
    if inherited['controls']!=old['training_resume']['controls'] or inherited['controls']!=8458:
        raise ValueError('Inherited E3 accounting changed')
    status=read(PRIOR/'status.json')
    audits=[]
    for worker in (0,1):
        base=PRIOR/f'training/worker_{worker}'
        audits.append(audit_render_failure(read(base/'steps.jsonl',True),read(base/'io.jsonl',True),
                                          read(base/'primary_failure.json')))
    reconciliation=reconcile_render_stop(read(PRIOR/'training/closed.json'),
        [a['controls'] for a in audits],read(PRIOR/'unresolved_training.json'),status,inherited['controls'])
    batches=[read(p) for directory in (FIRST,PRIOR) for p in sorted(directory.glob('train_batch_*.json'))]
    checkpoint=Path(batches[-1]['checkpoint']);receipt=read(checkpoint.with_suffix('.json'))
    if (checkpoint!=PRIOR/'rl_batch_014_updates_0173.pt' or receipt['sha256']!=DELTA_SHA
            or sha(checkpoint)!=DELTA_SHA or receipt['parent_sha256']!=PARENT_SHA or sha(PARENT)!=PARENT_SHA
            or status['actor_updates']!=receipt['actor_updates'] or status['critic_updates']!=receipt['critic_updates']
            or status['batches']!=len(batches) or len(batches)!=14):
        raise ValueError('Wrong parent/delta or a completed update/batch lost')
    initial=read(FIRST/'manifest.json');validate_recipe(initial)
    progress=resume_state(initial,batches,receipt,reconciliation['controls'])
    partial=sum(a['incomplete_controls'] for a in audits)
    partial_policy=sum(a['incomplete_policy_controls'] for a in audits)
    policy_controls=inherited['policy_controls']+sum(a['policy_controls'] for a in audits)
    if (progress['controls']!=55740 or receipt['controls']+partial!=progress['controls']
            or partial_policy!=1728 or policy_controls!=sum(b['policy_controls'] for b in batches)+partial_policy
            or progress['prefixes']!={'0':884,'1':1096}
            or progress['actor_updates']!=173 or progress['critic_updates']!=72
            or any(a['incomplete_episode']!=12 for a in audits)):
        raise ValueError('Unaccounted partial rollout or unexpected restored curriculum')
    metadata=pq.read_table(DATA/'meta/episodes/chunk-000/file-000.parquet').to_pylist()
    heldout={(r['task_index'],r['task_instance_id']) for r in metadata if r['episode_index']%200>=190}
    workers=deepcopy(old['workers'])
    for spec in workers:
        spec['prefix_controls']=progress['prefixes'][str(spec['worker'])]
        validate_worker(spec,evaluation=False)
        if (spec['task_id'],spec['instance']) in heldout:raise ValueError('Held-out TRAIN source')
        for name in ('actions','source_parquet','annotation'):
            if sha(spec[name])!=spec[name+'_sha256']:raise ValueError('TRAIN source changed')
    for spec in old['sim_pools']['final']:
        validate_worker(spec,evaluation=True)
        if sha(spec['instance_file'])!=spec['instance_file_sha256']:raise ValueError('Eval instance changed')
    manifest=deepcopy(old)
    manifest.update(source_commit=commit(),continued_from=str(PRIOR),
        resume_checkpoint=str(checkpoint),resume_sha256=DELTA_SHA,training_resume=progress,
        prior_active_seconds=old['prior_active_seconds']+ended['seconds'],
        render_completion_retries=2,
        continuation_reason='user_20260930_restart_after_render_failure',
        workers=workers,sim_pools=dict(training=workers,final=deepcopy(old['sim_pools']['final'])),
        previous_runtime=str(PREVIOUS_RUNTIME),runtime=str(RUNTIME),
        audit_input_sha256={**old['audit_input_sha256'],**inputs})
    validate_recipe(manifest)
    OUT.mkdir(parents=True,exist_ok=False)
    save(OUT/'manifest.json',manifest)
    save(OUT/'prior_control_audit.json',dict(**reconciliation,worker_audits=audits,
        policy_controls=policy_controls,discarded_partial_policy_controls=partial_policy,
        completed_batches=len(batches),all_updates_preserved=True,training_resume=progress,
        checkpoint=str(checkpoint),sha256=DELTA_SHA))
    print(json.dumps(dict(run=str(OUT),source=manifest['source_commit'],training_resume=progress,
        render_completion_retries=2,remaining_train_controls=80000-progress['controls'],new_gpu_processes=0)))


if __name__=='__main__':main()
