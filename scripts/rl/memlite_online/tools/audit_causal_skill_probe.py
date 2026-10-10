"""Independently bind joint planner logs to every real archived input/action.

Run AFTER the unchanged physical success/reset/archive auditor. Neither audit
authorizes SFT labels, RL gradients, observer deployment or full-task success.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

REPO=Path(__file__).resolve().parents[4]
sys.path[:0]=[str(REPO/'src'),str(REPO/'scripts/rl/memlite_online/code')]
from recovery_corpus import file_sha
from causal_skill_probe_audit import audit_joint_episode
from skill_observation_archive import load_observation
from skill_training_protocol import observation_hash,read_only_recipe
from g05.utils.memlite_causal_session import CausalModelIdentity


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('config','service-result','journal','histories','physical-audit','output'):
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    cfg=json.loads(a.config.read_text());service=json.loads(a.service_result.read_text())
    audit=json.loads(a.physical_audit.read_text());history=json.loads(a.histories.read_text())
    journal=[json.loads(s) for s in a.journal.read_text().splitlines()]
    spec=cfg['causal_planner'];model_path=REPO/spec['models_config']
    if (not read_only_recipe(cfg) or file_sha(model_path)!=spec['models_config_sha256']
            or file_sha(a.histories)!=spec['inputs_manifest_sha256']):
        raise ValueError('Changed bound read-only config/model/history inputs')
    model_cfg=json.loads(model_path.read_text())
    models=CausalModelIdentity(**{k:model_cfg[k]['sha256'] for k in
        ('planner','low','observer_backbone','observer_adapter')},
        **{k+'_normalization':v for k,v in model_cfg['normalization_sha256'].items()})
    expected={(key,seed) for key in cfg['cases'] for seed in cfg['evaluation_seeds']}
    jobs=service['jobs'];by_id={j['id']:j for j in jobs}
    if (service['status']!='completed_read_only_skill_probes' or service['error'] is not None
            or service['evaluation_only'] is not True or service['optimizer_updates']!=0
            or service['actor_updates']!=0 or service['active_episodes']!=0
            or service['config_sha256']!=file_sha(a.config)
            or service['policy_identity_sha256']!=cfg['model']['sha256']
            or service['joint_model_identity']!=asdict(models)
            or service['high_before_sha256']!=service['high_after_sha256']
            or service['actor_before_sha256']!=service['actor_after_sha256']
            or service['frozen_before_sha256']!=service['frozen_after_sha256']
            or service['observer_predictions']!=0 or service['observer_feedback_mode']!='shadow_unknown_v1'
            or service['scoring']!='original_restored_skill_endpoint_not_RL_reward'
            or set(service['finish_acknowledged'])!=set(cfg['cases'])
            or len(jobs)!=len(expected) or len(by_id)!=len(expected)
            or {(j['case'],j['seed']) for j in jobs}!=expected or any(j['status']!='done' for j in jobs)):
        raise ValueError('Incomplete/non-read-only service, changed model or missing final ACK')
    if (audit['status']!='machine_control_reward_reset_audit_passed'
            or audit['config_sha256']!=file_sha(a.config) or audit['policy_sha256']!=cfg['model']['sha256']
            or audit['optimizer_authorized'] is not False or audit['whole_task_sr'] is not False
            or audit['baseline_protocol_checked'] is not True
            or len(audit['episodes'])!=len(expected)
            or {(r['case'],r['seed']) for r in audit['episodes']}!=expected
            or {e['job']['id'] for e in journal}!=set(by_id)):
        raise ValueError('Require complete matching independent physical-control audit and journal')
    histories={r['case']:r for r in history['rows']};results=[]
    for row in audit['episodes']:
        directory=Path(row['run']);job_id=row['job_id']
        if (file_sha(directory/'controls.jsonl')!=row['controls_sha256']
                or file_sha(directory/'result.json')!=row['result_sha256']
                or file_sha(directory/'observations/manifest.json')!=row['observation_archive']['manifest_sha256']
                or file_sha(directory/'observations/observations.jsonl')!=row['observation_archive']['observations_sha256']):
            raise ValueError('Source controls/observations changed after physical audit')
        controls=[json.loads(s) for s in (directory/'controls.jsonl').read_text().splitlines()]
        records=[json.loads(s) for s in (directory/'observations/observations.jsonl').read_text().splitlines()]
        observed={r['control_step']:observation_hash(load_observation(directory/'observations',r)) for r in records}
        entries=[e for e in journal if e['job']['id']==job_id]
        job=dict(by_id[job_id],status='active')
        result=audit_joint_episode(entries,models=models,history_row=histories[row['case']],
            case=cfg['cases'][row['case']],job=job,controls=controls,observation_hashes=observed)
        if result['applied_controls']!=row['actual_controls']:
            raise ValueError('Journal and physical audit disagree on actual controls')
        results.append(result)
    actual=service['causal_planner_audit']
    for key in ('generations','reuses','applied_controls'):
        if actual[key]!=sum(r[key] for r in results):raise ValueError('Wrong aggregate causal '+key)
    if actual['active_episodes'] or actual['completed_episodes']!=len(expected) or actual['observer_predictions']:
        raise ValueError('Unclosed episode or invented observer result')
    result=dict(status='all_joint_observations_intents_actual_actions_and_clocks_replayed',episodes=results,
        config_sha256=file_sha(a.config),service_result_sha256=file_sha(a.service_result),
        journal_sha256=file_sha(a.journal),histories_sha256=file_sha(a.histories),
        physical_audit_sha256=file_sha(a.physical_audit),whole_task_sr=False,
        optimizer_updates=0,observer_predictions=0,policy_promotion_authorized=False,
        original_media_owner_review_still_required=True)
    a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__=='__main__':main()
