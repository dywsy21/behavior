"""Assemble byte-bound pilot evidence after explicit original-image review.

No model fitting, launch authorization, runtime calibration, or new labels.
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
from recovery_corpus import file_sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('release','run-root','start-root','start-review','owner','remote-root','output'):
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    owner=json.loads(a.owner.read_text());review_path=a.start_review/'review.json'
    if (owner['schema']!='recovery_start_owner_review_v1'
            or file_sha(review_path)!=owner['review_sha256'] or not owner['reviewer']):
        raise ValueError('Unbound explicit start owner review')
    review=json.loads(review_path.read_text())
    candidates={row['case']:row for row in review['candidates']}
    if set(owner['approved_cases'])!=set(candidates):raise ValueError('Missing or expanded reviewed start scope')
    for sheet in review['sheets']:
        if file_sha(a.start_review/sheet['path'])!=sheet['sha256']:raise ValueError('Changed reviewed original media')
    admission=a.release/'local-admission-v2/admission.json';admission_sha=file_sha(admission)
    rows=[json.loads(line) for line in (admission.parent/'action.jsonl').read_text().splitlines()]
    groups={row['candidate']['source_group']:row['candidate']['split'] for row in rows}
    import torch
    allowed_context={'task_name','parent_goal','memory','previous_intent','issued_skills_semantic_json',
        'served_controls','intent_started_control_step','attempt_number','known_previous_outcome'}
    accepted=[]
    for case in owner['approved_cases']:
        row=candidates[case];directory=a.start_root/Path(row['remote_directory']).relative_to(a.remote_root)
        result=directory/'result.json';snapshot=directory/'failure-start.pt'
        if file_sha(result)!=row['receipt_sha256'] or file_sha(snapshot)!=row['snapshot_sha256']:
            raise ValueError('Changed cold evidence bytes')
        receipt=json.loads(result.read_text())
        if (receipt['status']!='accepted_failure_start' or groups.get(row['source_group'])!=row['split']
                or receipt['actor_oracle_inputs'] or receipt['optimizer_steps']!=0):
            raise ValueError('Unapproved start or cross-source split')
        for camera,checksum in row['images'].items():
            if file_sha(directory/(camera+'.png'))!=checksum:raise ValueError('Changed actual failure RGB')
        state=torch.load(snapshot,map_location='cpu',weights_only=False)
        context=state['actor_observable_context']
        if (set(context)!=allowed_context or context['known_previous_outcome']!='UNKNOWN'
                or context['served_controls']!=32 or context['intent_started_control_step']!=0
                or context['attempt_number']!=1 or state['source_group']!=row['source_group']
                or state['source_branch_sha256']!=row['source_branch_sha256']
                or state['metadata']['policy_state_included'] is not False):
            raise ValueError('Privileged, future, or cross-branch actor context')
        accepted.append(dict(case=case,source_group=row['source_group'],split=row['split'],
            source_branch_sha256=row['source_branch_sha256'],source_snapshot_sha256=receipt['source_snapshot_sha256'],
            verified_route='cold_full_seed_restore_then_recorded_fault_32_then_measured_failure_gate',
            derived_failure_snapshot_cold_load_certified=False,receipt=dict(path=str(result),sha256=file_sha(result)),
            snapshot=dict(path=str(snapshot),sha256=file_sha(snapshot)),
            recorded_fault_controls=32,continued_stable_grasp_controls=receipt['continued_stable_grasp_controls'],
            position_error_m=receipt['failed_target_position_error_m']))
    if {row['split'] for row in accepted}!={'train','dev'}:raise ValueError('No independent split coverage')
    a.output.mkdir(parents=True)
    starts=dict(schema='recovery_start_acceptance_v1',status='passed',admission_sha256=admission_sha,
        accepted=accepted,rejected=review['rejected'],manual_review_complete=True,
        owner_sha256=file_sha(a.owner),review_sha256=file_sha(review_path),actor_oracle_inputs=False,
        formal_training=False,scope='Only local GRASP pilot; cold seed+fault route, not universal snapshot determinism')
    path=a.output/'start-acceptance.json';path.write_text(json.dumps(starts,indent=2)+'\n')
    paths=dict(full_read=a.release/'local-corpus-v3/full-read-audit.json',
        processor=a.run_root/'accepted-processor-v2.json',features=a.run_root/'accepted-feature-audit-v2.json',
        start_states=path,transfer=a.run_root/'accepted-transfer-v2.json')
    preflight=dict(schema='recovery_accepted_data_preflight_v1',admission_sha256=admission_sha,
        optimizer_steps=0,formal_training_authorized=False,
        checks={key:dict(path=str(value.resolve()),sha256=file_sha(value)) for key,value in paths.items()})
    path=a.output/'accepted-data-preflight.json';path.write_text(json.dumps(preflight,indent=2)+'\n')
    print(json.dumps(dict(status='assembled',accepted_starts=len(accepted),rejected_starts=len(review['rejected']),
        preflight_sha256=file_sha(path),formal_training_authorized=False)))


if __name__=='__main__':main()
