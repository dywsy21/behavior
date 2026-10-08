"""Seal the 1,000 public cases and collect reproducibility/submission metadata."""
import argparse
import csv
import json
from pathlib import Path
import shutil
import subprocess
import time
from common import (ROOT, SOURCE, REPO, DATA, OFFICIAL, OFFICIAL_COMMIT, CHECKPOINTS, G05, G05_COMMIT,
                    expected_cases, sha256, atomic_json)


def main():
    p=argparse.ArgumentParser();p.add_argument('--job',type=Path,required=True)
    args=p.parse_args();job=args.job.resolve()
    if job.parent!=ROOT/'runs' or not job.name.startswith('sft100_full_') or job.exists():
        raise ValueError('A new explicit SFT evaluation run is required')
    commit=subprocess.check_output(['git','-C',str(REPO),'rev-parse','HEAD'],text=True).strip()
    if subprocess.check_output(['git','-C',str(REPO),'status','--porcelain'],text=True).strip():
        raise ValueError('Frozen evaluation checkout is dirty')
    if (subprocess.check_output(['git','-C',str(G05),'rev-parse','HEAD'],text=True).strip()!=G05_COMMIT
        or subprocess.check_output(['git','-C',str(G05),'status','--porcelain'],text=True).strip()):
        raise ValueError('G0.5 dependency is not the sealed clean checkpoint-compatible revision')
    checks={}
    for side,(name,expected) in CHECKPOINTS.items():
        path=ROOT/'models/stage1'/side/name
        actual=sha256(path)
        if actual!=expected:raise ValueError('SFT checkpoint hash mismatch: '+side)
        config=ROOT/'configs'/(side+'_model.local.json')
        c=json.loads(config.read_text())
        checks[side]=dict(path=str(path),sha256=actual,bytes=path.stat().st_size,
            config=str(config),config_sha256=sha256(config),
            stats_path=c['stats_path'],stats_sha256=sha256(c['stats_path']),
            model_class=c['arch']['_target_'])
    csv_path=DATA/'2026-challenge-task-instances/metadata/B100_task_misc.csv'
    tasks=[r['Task'] for r in csv.DictReader(csv_path.open())]
    cases=expected_cases(tasks)
    all_public=list((DATA/'2026-challenge-task-instances/scene_test/public').rglob('*-tro_state.json'))
    source_cases=[]
    for task,instance,rollout in sorted(cases):
        suffix=f'_task_{task}_0_{instance}_template-tro_state.json'
        paths=[f for f in all_public if f.name.endswith(suffix)]
        if len(paths)!=1:raise ValueError('Missing/ambiguous public instance '+str((task,instance)))
        source_cases.append(dict(task=task,instance_id=instance,public_index=instance-301,
            rollout_id=rollout,instance_path=str(paths[0]),instance_sha256=sha256(paths[0])))
    old=json.loads((ROOT/'runs/large_scale73h_trainonly_20261007/manifest.json').read_text())
    horizons={t['task']:t['official_horizon_steps'] for g in old['groups'] for t in g['tasks']}
    groups=[dict(gpu=i,tasks=[],estimated_control_steps=0) for i in range(8)]
    for task in sorted(tasks,key=lambda t:horizons[t],reverse=True):
        group=min(groups,key=lambda g:g['estimated_control_steps'])
        group['tasks'].append(task);group['estimated_control_steps']+=10*(horizons[task]+1)
    for g in groups:g['tasks'].sort(key=lambda t:horizons[t])
    official_hashes={str(f.relative_to(OFFICIAL)):sha256(f) for f in OFFICIAL.rglob('*')
                     if f.is_file() and f.suffix in ('.py','.yaml','.json','.kit')}
    expected_official=json.loads((SOURCE/'official_manifest.json').read_text())
    if expected_official['commit']!=OFFICIAL_COMMIT or official_hashes!=expected_official['files']:
        raise ValueError('Official simulator files differ from the upstream Git tag')
    job.mkdir();(job/'workers').mkdir();(job/'tasks').mkdir();package=job/'submission';package.mkdir()
    manifest=dict(kind='native_sft_100task_public_once',status='prepared_not_started',created=time.time(),
        owner='Codex/EVAL-SFT100-10383',source_commit=commit,source_path=str(SOURCE),
        official_tag='v3.9.3-post2',official_commit=OFFICIAL_COMMIT,official_source=str(OFFICIAL),
        g05_source=str(G05),g05_commit=G05_COMMIT,
        official_files=official_hashes,checkpoints=checks,task_csv_sha256=sha256(csv_path),
        tasks=tasks,groups=groups,cases=source_cases,episodes=1000,num_envs=2,
        optimizer_steps=0,exploration_transition_noise=False,native_initial_fm_noise=True,
        fm_steps=10,predicted_horizon=32,executed_horizon=16,action_start=0,
        planner_period_controls=128,planner_temperature=0,policy_seed=17,simulator_seed=0,
        nominal_horizons_for_scheduling=horizons,timeout='stock 1.5x mean-human length; no override',
        automatic_retries=False,train_recovery_collection=False,submission_authorized=False,
        safety=dict(max_gpu_count=8,loader_concurrency=2,ram_stop_anonymous_gib=210,
                    disk_reserve_gib=150,initialization_timeout_seconds=1800,
                    no_progress_timeout_seconds=1800),
        development_notice='public_test301 was previously used for project diagnostics; this is not a blind test',
        source_urls=['https://behavior.stanford.edu/challenge/evaluation.html',
                     'https://behavior.stanford.edu/challenge/submission.html'])
    atomic_json(job/'manifest.json',manifest)
    shutil.copy2(SOURCE/'rgb_wrapper.py',package/'rgb_wrapper.py')
    shutil.copy2(OFFICIAL/'omnigibson/eval/r1pro.yaml',package/'r1pro.yaml')
    atomic_json(package/'submission_checklist.json',dict(
        metrics_json=dict(required=1000,collected=0,do_not_edit=True),
        videos=dict(required=1000,collected=0,cameras=['head','left_wrist','right_wrist'],
                    do_not_edit=True,hosted_video_link=None),
        wrapper=dict(path='rgb_wrapper.py',sha256=sha256(package/'rgb_wrapper.py')),
        robot=dict(path='r1pro.yaml',sha256=sha256(package/'r1pro.yaml'),unchanged_official=True),
        readme='generated with final inventory; exact command/source/weights/config required',
        serving=dict(docker_image=None,required_single_gpu_vram_gb=24,validated_24gb=False,
                     alternative_ip=None,ip_route_required_ports=50,exposed_ports=[]),
        portal='https://behavior-1k-2026-challenge-leaderboard.hf.space',
        team_identity=None,model_display_name=None,contact=None,open_source_disclosure_permission=None,
        note='Portal currently inaccessible during audit; required portal fields must be reconfirmed. No upload/submission.'))
    print(json.dumps(dict(job=str(job),cases=len(cases),checkpoints_verified=True,source_commit=commit)))


if __name__=='__main__':main()
