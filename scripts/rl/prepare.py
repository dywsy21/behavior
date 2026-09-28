"""Extract two original TRAIN curriculum prefixes; no held-out labels used."""
import json
import math
import os
import subprocess
import numpy as np
import pyarrow.parquet as pq
from common import DATA, RAW, OUT, PREVIOUS, TEMPLATE, PARENT, PARENT_SHA, REPO, commit, sha, save


def main():
    OUT.mkdir(parents=True,exist_ok=False)
    source=commit()
    if PREVIOUS.is_dir():
        previous=json.loads((PREVIOUS/'manifest.json').read_text())
        supervisor=json.loads((PREVIOUS/'supervisor.json').read_text())
        status=json.loads((PREVIOUS/'status.json').read_text())
        pids=(json.loads((PREVIOUS/'pids.json').read_text()) if (PREVIOUS/'pids.json').is_file()
              else dict(learner=supervisor['learner'],simulators=[]))
        for pid in [supervisor['supervisor'],pids['learner'],*pids['simulators']]:
            # Zombie supervisors cannot hold CUDA/runtime resources.
            path=f'/proc/{pid}/stat'
            if os.path.exists(path) and open(path).read().rsplit(')',1)[1].split()[0]!='Z':
                raise ValueError('Previous experiment process still alive')
        if supervisor['status']!='failed' or status['pending_controls'] or status['actor_updates']:
            raise ValueError('Only this pre-update failed integration may be continued')
        if previous['parent_sha256']!=PARENT_SHA or sha(PARENT)!=PARENT_SHA:
            raise ValueError('Parent changed')
        for worker in previous['workers']:
            if sha(worker['actions'])!=worker['actions_sha256']: raise ValueError('TRAIN actions changed')
            worker['policy_seed']=1700
        used=previous.get('prior_controls',0)+status['controls']
        elapsed=previous.get('prior_active_seconds',0.)+supervisor['seconds']
        previous.update(source_commit=source,max_controls=10000-used,
            max_active_wall_seconds=7200-math.ceil(elapsed),policy_rng='shared torch/CUDA seed1700',
            continued_from=str(PREVIOUS),prior_controls=used,prior_active_seconds=elapsed)
        receipt=PREVIOUS/'throughput.json'
        if receipt.is_file():
            result=json.loads(receipt.read_text())
            if result['serial']['controls']!=256 or result['parallel']['controls']!=256:
                raise ValueError('Unexpected matched-workload receipt')
            # Reuse only if the complete simulator-side implementation is unchanged.
            simulator_paths=['scripts/rl/sim_worker.py','scripts/semantic_robot/native_rl_profile.py',
                'scripts/semantic_robot/rl_reset_boundary.py','scripts/semantic_robot/shared_og_startup.py',
                'src/semantic_robot/v2']
            subprocess.run(['git','-C',str(REPO),'diff','--exit-code',supervisor['source_commit'],source,
                            '--',*simulator_paths],check=True)
            previous['benchmark_receipt']=dict(path=str(receipt),sha256=sha(receipt),
                source_commit=supervisor['source_commit'],simulator_implementation_unchanged=True)
        save(OUT/'manifest.json',previous)
        print('CONTINUATION',previous['max_controls'],previous['max_active_wall_seconds'],flush=True)
        return
    if sha(PARENT) != PARENT_SHA: raise ValueError('Parent checkpoint changed')
    meta=pq.read_table(DATA/'meta/episodes/chunk-000/file-000.parquet').to_pylist()
    holdout={r['task_instance_id'] for r in meta if int(r['episode_index'])%200 >=190}
    entries=[]
    # Two source episodes, each 96 controls BEFORE the first recorded terminal.
    # The simulator must independently confirm the replayed start is nonterminal.
    for worker,episode in enumerate((0,121)):
        row=next(r for r in meta if r['episode_index']==episode)
        if row['task_instance_id'] in holdout or row['task_index'] != 0: raise ValueError('Split leakage')
        path=DATA/f'data/chunk-{row["data/chunk_index"]:03d}/file-{row["data/file_index"]:03d}.parquet'
        rows=pq.read_table(path,filters=[('episode_index','=',episode)],
            columns=['frame_index','action','next.reward','next.terminated','next.truncated']).to_pylist()
        rows.sort(key=lambda r:r['frame_index'])
        if [r['frame_index'] for r in rows] != list(range(row['length'])): raise ValueError('Non-contiguous demo')
        done=next(r['frame_index'] for r in rows if r['next.terminated'])
        if rows[done]['next.reward'] != 1 or done<224: raise ValueError('No recorded successful first terminal')
        prefix=done-96
        if any(r['next.terminated'] or r['next.truncated'] for r in rows[:prefix]): raise ValueError('Terminal prefix')
        actions=np.asarray([r['action'] for r in rows],dtype=np.float32)
        if actions.shape != (row['length'],23) or not np.isfinite(actions).all(): raise ValueError('Invalid demo controls')
        file=OUT/f'demo_{episode}.npy'; np.save(file,actions)
        annotation=RAW/row['annotation_path']
        entry=dict(worker=worker,gpu=worker+2,task='turning_on_radio',task_id=0,split='train',episode=episode,
            instance=int(row['task_instance_id']),seed=0,policy_seed=1700+worker,first_recorded_terminal=done,
            prefix_controls=prefix,actions=str(file),actions_sha256=sha(file),source_parquet=str(path),
            source_parquet_sha256=sha(path),annotation=str(annotation),annotation_sha256=sha(annotation),
            metadata=row, actor_oracle_fields=False)
        entries.append(entry)
    manifest=dict(source_commit=source,parent=str(PARENT),parent_sha256=PARENT_SHA,workers=entries,
        max_controls=10000,max_active_wall_seconds=7200,rollout_chunks=128,chunk_controls=16,
        episode_controls=512,critic_warmup_batches=1,ppo_epochs=2,minibatch=8,clip=.05,target_path_kl=.01,
        learning_rates=dict(action_expert=1e-6,noise=1e-5,critic=1e-4),bc_weight=.1,
        noise=dict(initial=.01,low=.003,high=.03),gamma_control=.9998,lambda_chunk=.95,
        reward='once-only official success +1',snapshot_reset=False,legacy_window_template=str(TEMPLATE),
        template_purpose='robot config only; no old oracle label admitted',
        claim='bounded integration, not method success-rate evaluation')
    save(OUT/'manifest.json',manifest)
    print(json.dumps({k:v for k,v in manifest.items() if k!='workers'},indent=2))
    print('CURRICULUM',[(r['episode'],r['instance'],r['prefix_controls'],r['first_recorded_terminal']) for r in entries])


if __name__=='__main__': main()
