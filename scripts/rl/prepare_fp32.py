"""One remaining-budget online batch, FP32 AE collection AND PPO scoring."""
import json
import math
from pathlib import Path
import subprocess
from common import PREVIOUS,OUT,REPO,PARENT,PARENT_SHA,commit,sha,save


def main():
    manifest=json.loads((PREVIOUS/'manifest.json').read_text())
    supervisor=json.loads((PREVIOUS/'supervisor.json').read_text())
    status=json.loads((PREVIOUS/'status.json').read_text())
    for pid in (supervisor['supervisor'],supervisor['learner']):
        path=Path(f'/proc/{pid}/stat')
        if path.exists() and path.read_text().rsplit(')',1)[1].split()[0]!='Z':
            raise ValueError('Numerical probe still active')
    if supervisor['status']!='completed' or status['controls'] or status['actor_updates']:
        raise ValueError('Numerical probe did not complete cleanly')
    final=json.loads((PREVIOUS/'final_restore.json').read_text())
    if final['accepted_updates'] or final['saved_actor']: raise ValueError('Unexpected probe actor')
    if sha(PARENT)!=PARENT_SHA: raise ValueError('Parent changed')
    for w in manifest['workers']:
        if sha(w['actions'])!=w['actions_sha256']: raise ValueError('TRAIN controls changed')
    source=commit(); receipt=manifest['benchmark_receipt']
    if sha(receipt['path'])!=receipt['sha256']: raise ValueError('Changed simulation evidence')
    subprocess.run(['git','-C',str(REPO),'diff','--exit-code',receipt['source_commit'],source,'--',
        'scripts/rl/sim_worker.py','scripts/semantic_robot/native_rl_profile.py',
        'scripts/semantic_robot/rl_reset_boundary.py','scripts/semantic_robot/shared_og_startup.py',
        'src/semantic_robot/v2'],check=True)
    elapsed=manifest['prior_active_seconds']+supervisor['seconds']
    remaining=10000-manifest['prior_controls']; wall=7200-math.ceil(elapsed)
    if remaining<sum(w['prefix_controls'] for w in manifest['workers'])+128 or wall<1200:
        raise ValueError('Insufficient original budget for one fresh on-policy batch')
    manifest.update(source_commit=source,continued_from=str(PREVIOUS),prior_active_seconds=elapsed,
        max_active_wall_seconds=wall,max_controls=remaining,entry='learner',ae_precision='float32',
        bounded_backtracking=True,max_batches=1,max_actor_updates=12,
        learning_rates=dict(action_expert=1e-8,noise=1e-7,critic=1e-4),
        old_bfloat16_paths_used_for_training=False,critic_initialization='fresh, same torch seed1700',
        precision_amendment='VLM stays bfloat16; AE sampling/scoring/deployment FP32, TF32 disabled; fresh paths only')
    manifest['previous_bf16_artifacts']={key:manifest.pop(key) for key in ('source_rollout','source_checkpoint')}
    manifest['update_amendment'].update(max_accepted_updates=12,source='fresh FP32 on-policy batch only')
    OUT.mkdir(parents=True,exist_ok=False); save(OUT/'manifest.json',manifest)
    print(json.dumps(dict(output=str(OUT),prior_controls=manifest['prior_controls'],prior_seconds=elapsed,
                         remaining_controls=remaining,remaining_seconds=wall,ae_precision='float32',max_batches=1)))


if __name__=='__main__': main()
