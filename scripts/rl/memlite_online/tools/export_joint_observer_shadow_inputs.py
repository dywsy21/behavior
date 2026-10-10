"""Export every native boundary of already independently audited TRAIN probes.

No fresh rollout, truth labels, approval, sampling or model selection. All four
closed jobs are retained, with observable actual ACKs rather than offered
actions masquerading as applied actions.
"""
import argparse
import json
from pathlib import Path
import shutil
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
from recovery_corpus import file_sha
from skill_observation_archive import load_observation


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('config','causal-audit','physical-audit','journal','histories','output'):
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    cfg=json.loads(a.config.read_text());audit=json.loads(a.causal_audit.read_text())
    physical=json.loads(a.physical_audit.read_text());histories=json.loads(a.histories.read_text())
    events=[json.loads(s) for s in a.journal.read_text().splitlines()]
    if (audit['status']!='all_joint_observations_intents_actual_actions_and_clocks_replayed'
            or audit['config_sha256']!=file_sha(a.config)
            or audit['physical_audit_sha256']!=file_sha(a.physical_audit)
            or audit['journal_sha256']!=file_sha(a.journal)
            or audit['histories_sha256']!=file_sha(a.histories)
            or len(audit['episodes'])!=4 or len(physical['episodes'])!=4):
        raise ValueError('Require the full closed four-job independent audit, never a favorable subset')
    ids={r['job_id'] for r in audit['episodes']}
    if ids!={r['job_id'] for r in physical['episodes']} or ids!={r['job']['id'] for r in events}:
        raise ValueError('Different source jobs')
    by_case={r['case']:r for r in histories['rows']};items=[]
    for episode in physical['episodes']:
        case=cfg['cases'][episode['case']];source=Path(episode['run'])
        if (case['original_split']!='train' or case['recovery_split']!='train'
                or file_sha(source/'controls.jsonl')!=episode['controls_sha256']
                or file_sha(source/'result.json')!=episode['result_sha256']
                or file_sha(source/'observations/observations.jsonl')!=episode['observation_archive']['observations_sha256']
                or file_sha(source/'observations/manifest.json')!=episode['observation_archive']['manifest_sha256']):
            raise ValueError('Only unchanged audited TRAIN sources')
        records=[json.loads(s) for s in (source/'observations/observations.jsonl').read_text().splitlines()]
        controls=[json.loads(s) for s in (source/'controls.jsonl').read_text().splitlines()]
        selected=[e for e in events if e['job']['id']==episode['job_id']]
        subdir=episode['case']+'-seed'+str(episode['seed'])
        for row in records:load_observation(source/'observations',row)
        acknowledgements=[dict(control_step=r['control_step']-1,
            simulator_apply_ack=r['simulator_apply_ack'],action_executed_raw23=r['action_executed_raw23']) for r in controls]
        items.append(dict(directory=subdir,case=episode['case'],task=case['task'].replace('_',' '),
            instance=case['instance_id'],job_id=episode['job_id'],seed=episode['seed'],records=records,
            issued_prefix=by_case[episode['case']]['issued_prefix'],events=selected,actual_acks=acknowledgements,
            original_source=str(source),source_result_sha256=episode['result_sha256'],
            source_controls_sha256=episode['controls_sha256']))
    a.output.mkdir(parents=True)
    for row in items:
        destination=a.output/row['directory'];destination.mkdir()
        for record in row['records']:
            shutil.copyfile(Path(row['original_source'])/'observations'/record['file'],destination/record['file'])
            load_observation(destination,record)
    result=dict(schema='audited_joint_observer_shadow_inputs_v1',
        role='TRAIN_engineering_only_never_SFT_calibration_or_test',rows=items,
        source_config_sha256=file_sha(a.config),causal_audit_sha256=file_sha(a.causal_audit),
        physical_audit_sha256=file_sha(a.physical_audit),journal_sha256=file_sha(a.journal),
        physical_controls_during_export=0,whole_task_sr=False)
    (a.output/'manifest.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(status='all_four_shadow_sequences_exported',manifest_sha256=file_sha(a.output/'manifest.json'),
        observations=sum(len(r['records']) for r in items),actual_controls=sum(len(r['actual_acks']) for r in items))))


if __name__=='__main__':main()
