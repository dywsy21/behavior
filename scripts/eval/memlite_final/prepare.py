"""Register a new final-policy run; no inheritance of old public results."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent;REPO=HERE.parents[2]
sys.path[:0]=[str(REPO/'scripts/rl/memlite_online/code'),str(HERE.parent/'memlite_sft100'),str(HERE)]
from assets import verify_assets
from common import ROOT,DATA,OFFICIAL,atomic_json,sha256,load_official_task_names


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('config','pack','job'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--manifest-sha256',required=True)
    a=p.parse_args();job=a.job.resolve()
    if (job.parent!=ROOT/'runs' or not job.name.startswith('memlite_final100_') or job.exists()
            or subprocess.check_output(['git','status','--porcelain'],cwd=REPO,text=True).strip()):
        raise ValueError('New registered final run and frozen source required')
    cfg,assets,_configs=verify_assets(a.pack,a.manifest_sha256,a.config)
    pinned=json.loads((HERE.parent/'memlite_sft100/official_manifest.json').read_text())
    for path,sha in pinned['files'].items():
        if sha256(OFFICIAL/path)!=sha:raise ValueError('Official source changed')
    task_csv=DATA/'2026-challenge-task-instances/metadata/B100_task_misc.csv'
    names=load_official_task_names(task_csv)
    old=json.loads((ROOT/'runs/large_scale73h_trainonly_20261007/manifest.json').read_text())
    horizons={t['task']:t['official_horizon_steps'] for g in old['groups'] for t in g['tasks']}
    if set(horizons)!=set(names.values()):raise ValueError('Incomplete official scheduling horizons')
    public=list((DATA/'2026-challenge-task-instances/scene_test/public').rglob('*-tro_state.json'))
    cases=[]
    for task_id,task in sorted(names.items()):
        for index in range(10):
            instance=index+301;suffix=f'_task_{task}_0_{instance}_template-tro_state.json'
            files=[f for f in public if f.name.endswith(suffix)]
            if len(files)!=1:raise ValueError('Missing/ambiguous public task instance')
            cases.append(dict(task_id=task_id,task=task,public_index=index,instance_id=instance,rollout_id=0,
                instance_path=str(files[0]),instance_sha256=sha256(files[0])))
    groups=[dict(gpu=i,tasks=[],estimated_controls=0) for i in range(8)]
    for task_id,task in sorted(names.items(),key=lambda r:(-horizons[r[1]],r[0])):
        group=min(groups,key=lambda g:g['estimated_controls'])
        group['tasks'].append(dict(task=task,task_id=task_id));group['estimated_controls']+=10*horizons[task]
    for group in groups:group['tasks'].sort(key=lambda r:(horizons[r['task']],r['task_id']))
    job.mkdir()
    for name in ('workers','parts','submission','preflight'):(job/name).mkdir()
    manifest=dict(schema='memlite_final_public100_run_v1',status='prepared_not_started',created=time.time(),
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
        config_path=str(a.config.resolve()),config_sha256=sha256(a.config),pack=str(a.pack.resolve()),
        pack_manifest_sha256=a.manifest_sha256,models=assets['models'],official_commit=pinned['commit'],
        tasks=list(names.values()),cases=cases,episodes=1000,groups=groups,nominal_horizons=horizons,
        task_csv_sha256=sha256(task_csv),inherited_results=0,optimizer_updates=0,submission_authorized=False,
        automatic_retries=False,actual_ack_clock=True,public_test301_is_development=True,
        batch_size='awaiting_train_engineering_measurement',policy_seed=17,simulator_seed=0)
    atomic_json(job/'manifest.json',manifest)
    shutil.copyfile(HERE.parent/'memlite_sft100/rgb_wrapper.py',job/'submission/rgb_wrapper.py')
    shutil.copyfile(OFFICIAL/'omnigibson/eval/r1pro.yaml',job/'submission/r1pro.yaml')
    print(json.dumps(dict(job=str(job),registered_cases=len(cases),manifest_sha256=sha256(job/'manifest.json'))))


if __name__=='__main__':main()
