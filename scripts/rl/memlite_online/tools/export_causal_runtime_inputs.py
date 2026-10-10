"""Export three closed TRAIN initial observations for GPU engineering QA only."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import shutil
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
from recovery_corpus import file_sha
from skill_observation_archive import load_observation


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True);p.add_argument('--runs',type=Path,nargs=3,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    cfg=json.loads(a.config.read_text());rows=[]
    if a.output.exists():raise FileExistsError(a.output)
    expected_policy=cfg['resume']['sha256'];seen=set()
    for directory in a.runs:
        result=json.loads((directory/'result.json').read_text())
        process=json.loads((directory.parent/'result.json').read_text());job=result['job'];case=cfg['cases'][job['case']]
        archive=directory/'observations';header=json.loads((archive/'manifest.json').read_text())
        record=json.loads((archive/'observations.jsonl').read_text().splitlines()[0])
        if (job['case'] in seen or job['phase']!='train' or job['round']!=1
                or case['original_split']!='train' or case['recovery_split']!='train'
                or process['status']!='completed_single_episode' or process['config_sha256']!=file_sha(a.config)
                or result['policy_sha256']!=expected_policy or header['policy_sha256']!=expected_policy
                or header['identity']!=result['identity'] or header['identity']['episode']!=job['id']
                or record['control_step']!=case['start_control']
                or file_sha(directory/'controls.jsonl')!=result['controls_sha256']):
            raise ValueError('Only exact closed first-TRAIN observations from the registered three cases')
        load_observation(archive,record);seen.add(job['case'])
        rows.append(dict(case=job['case'],task=case['task'].replace('_',' '),instance=case['instance_id'],
            source_directory=str(directory),source_episode=job['id'],source_policy_sha256=expected_policy,
            source_result_sha256=file_sha(directory/'result.json'),source_config_sha256=file_sha(a.config),
            source_archive_manifest_sha256=file_sha(archive/'manifest.json'),source_observation=record,
            record=dict(record,file=job['case']+'.npz')))
    if seen!=set(cfg['cases']):raise ValueError('Must preserve all three TRAIN mechanism fixtures')
    a.output.mkdir(parents=True)
    for row in rows:
        shutil.copyfile(Path(row['source_directory'])/'observations'/row['source_observation']['file'],a.output/row['record']['file'])
        load_observation(a.output,row['record'])
    manifest=dict(schema='causal_runtime_initial_inputs_v1',rows=rows,
        role='TRAIN_engineering_only_never_SFT_calibration_or_test',
        physical_controls_during_test=0,rollout_success_measurement=False)
    (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(dict(status='three_observable_inputs_exported',manifest_sha256=file_sha(a.output/'manifest.json'))))


if __name__=='__main__':main()
