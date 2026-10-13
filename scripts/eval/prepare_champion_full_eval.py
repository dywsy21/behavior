"""Register terminal formal EMA only; never substitute an intermediate model."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent;REPO=HERE.parents[1]
sys.path[:0]=[str(REPO/'src'),str(HERE/'memlite_sft100')]
from common import ROOT,DATA,OFFICIAL,atomic_json,sha256,load_official_task_names
from behavior_champion.policy_assets import verify_policy_assets


def require_terminal_export(terminal,export):
    if (terminal.get('status') not in ('completed_target_updates','saved_at_cumulative_time_limit')
            or terminal.get('completed_updates',0)<=0
            or terminal['completed_updates']!=export.get('completed_updates')
            or terminal.get('checkpoint')!=export.get('source_checkpoint')
            or export.get('schema')!='champion-ema-export-v2' or export.get('weights')!='EMA'
            or export.get('purpose')!='CHAMPION100_FORMAL_SFT' or export.get('usable_for_deployment') is not True
            or export.get('restored_by_author_loader_exact') is not True):
        raise ValueError('Need the exact completed formal-training terminal EMA, never a selected interim checkpoint')


def policy_config(checkpoint,export_sha,completed,*,gpu):
    return dict(schema='champion2026_official_websocket_v1',owner='Codex SUBMISSION-FULL-EVAL',
        a800_root=str(ROOT),reference='code/champion-reference-20261013-v1',
        openpi_reference='code/openpi-reference-20261013-v1',checkpoint=str(checkpoint.relative_to(ROOT)),
        verified=str((checkpoint/'export.json').relative_to(ROOT)),verified_sha256=export_sha,
        checkpoint_kind='formal_ema_v2',completed_updates=completed,trained_model_tasks=list(range(100)),
        model_task_count=100,allowed_task_ids=list(range(100)),policy_seed=17,model_gpu=gpu,
        host='127.0.0.1',port=20000+gpu,max_connections=4,apply_legacy_rules=True,num_inference_steps=20,
        purpose='full_original_public_evaluation_not_external_submission')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--export-sha256',required=True)
    p.add_argument('--terminal',type=Path,required=True);p.add_argument('--terminal-sha256',required=True)
    p.add_argument('--job',type=Path,required=True)
    a=p.parse_args();job=a.job.resolve();checkpoint=a.checkpoint.resolve()
    if (job.parent!=ROOT/'runs' or not job.name.startswith('champion_final100_') or job.exists()
            or not checkpoint.is_relative_to(ROOT/'runs')
            or sha256(a.terminal)!=a.terminal_sha256 or sha256(checkpoint/'export.json')!=a.export_sha256
            or subprocess.check_output(['git','status','--porcelain'],cwd=REPO,text=True).strip()):
        raise ValueError('New registered run, exact terminal receipts and clean frozen code required')
    terminal=json.loads(a.terminal.read_text());export=json.loads((checkpoint/'export.json').read_text())
    require_terminal_export(terminal,export)
    configs=[policy_config(checkpoint,a.export_sha256,terminal['completed_updates'],gpu=i) for i in range(8)]
    verify_policy_assets(ROOT,REPO,configs[0])
    pinned=json.loads((HERE/'memlite_sft100/official_manifest.json').read_text())
    for path,sha in pinned['files'].items():
        if sha256(OFFICIAL/path)!=sha:raise ValueError('Official source changed')
    task_csv=DATA/'2026-challenge-task-instances/metadata/B100_task_misc.csv'
    names=load_official_task_names(task_csv)
    prior=json.loads((ROOT/'runs/large_scale73h_trainonly_20261007/manifest.json').read_text())
    horizons={t['task']:t['official_horizon_steps'] for g in prior['groups'] for t in g['tasks']}
    if set(horizons)!=set(names.values()):raise ValueError('Incomplete official scheduling metadata')
    public=list((DATA/'2026-challenge-task-instances/scene_test/public').rglob('*-tro_state.json'))
    cases=[]
    for task_id,task in sorted(names.items()):
        for index in range(10):
            instance=index+301
            matches=[f for f in public if f.name.endswith(f'_task_{task}_0_{instance}_template-tro_state.json')]
            if len(matches)!=1:raise ValueError('Missing/ambiguous prescribed public case')
            cases.append(dict(task_id=task_id,task=task,public_index=index,instance_id=instance,rollout_id=0,
                instance_path=str(matches[0]),instance_sha256=sha256(matches[0])))
    groups=[dict(gpu=i,tasks=[],estimated_controls=0) for i in range(8)]
    for task_id,task in sorted(names.items(),key=lambda r:(-horizons[r[1]],r[0])):
        group=min(groups,key=lambda g:g['estimated_controls'])
        group['tasks'].append(dict(task_id=task_id,task=task));group['estimated_controls']+=10*horizons[task]
    for group in groups:group['tasks'].sort(key=lambda r:(horizons[r['task']],r['task_id']))
    job.mkdir()
    for name in ('workers','parts','submission','preflight','configs'):(job/name).mkdir()
    config_receipts=[]
    for i,cfg in enumerate(configs):
        path=job/'configs'/f'gpu_{i}.json';atomic_json(path,cfg)
        config_receipts.append(dict(gpu=i,path=str(path),sha256=sha256(path)))
    manifest=dict(schema='champion_final_public100_run_v1',status='prepared_not_started',created=time.time(),
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
        checkpoint=str(checkpoint),export_sha256=a.export_sha256,terminal_path=str(a.terminal.resolve()),
        terminal_sha256=a.terminal_sha256,completed_training_updates=terminal['completed_updates'],
        official_commit=pinned['commit'],configs=config_receipts,tasks=list(names.values()),cases=cases,groups=groups,
        nominal_horizons=horizons,task_csv_sha256=sha256(task_csv),episodes=1000,inherited_results=0,
        optimizer_updates=0,automatic_retries=False,submission_authorized=False,policy_seed=17,simulator_seed=0,
        public_test301_is_development=True,
        model_python='/home/tione/notebook/baselines/StarVLA/.venv/bin/python',
        model_overlay=str(ROOT/'runs/champion_final_preparation_20261013/deps-v1'),
        transport='one_original_protocol_connection_per_slot',server_clock_kind='policy_delivery',
        actual_simulator_clocks_recorded_separately=True,preflight_required=True)
    atomic_json(job/'manifest.json',manifest)
    shutil.copyfile(HERE/'memlite_sft100/rgb_wrapper.py',job/'submission/rgb_wrapper.py')
    shutil.copyfile(OFFICIAL/'omnigibson/eval/r1pro.yaml',job/'submission/r1pro.yaml')
    print(json.dumps(dict(job=str(job),cases=1000,manifest_sha256=sha256(job/'manifest.json'))))


if __name__=='__main__':main()
