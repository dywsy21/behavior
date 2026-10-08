"""New run, immutable inherited results, explicit mixed-engine provenance."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import time
from common import ROOT,REPO,SOURCE,G05,G05_COMMIT,OFFICIAL,sha256,atomic_json
from resume_plan import remaining_parts


def main():
    p=argparse.ArgumentParser();p.add_argument('--prior',type=Path,required=True)
    p.add_argument('--benchmark',type=Path,required=True);p.add_argument('--job',type=Path,required=True)
    p.add_argument('--mixed-audit',type=Path,required=True)
    p.add_argument('--num-envs',type=int,choices=[2,4],required=True);a=p.parse_args()
    if a.job.parent!=ROOT/'runs' or a.job.exists():raise ValueError('New explicit resume run required')
    if a.prior.parent!=ROOT/'runs' or a.benchmark.parent!=ROOT/'runs':raise ValueError('Unregistered evidence path')
    old=json.loads((a.prior/'manifest.json').read_text())
    receipt=a.prior/'stopped_inventory_20261008_speed.json';sealed=json.loads(receipt.read_text())
    if (sealed['job']!=str(a.prior) or sealed['manifest_sha256']!=sha256(a.prior/'manifest.json') or
            not sealed['weights_unchanged_all_8'] or not (a.prior/'STOP').exists()):
        raise ValueError('Prior stopped inventory is not verified')
    bench=json.loads((a.benchmark/'manifest.json').read_text())
    benchmark_status=json.loads((a.benchmark/'status.json').read_text())
    qa=json.loads((a.benchmark/'numerical_audit/audit.json').read_text())
    report=json.loads((a.benchmark/'report.json').read_text())
    if (benchmark_status['status']!='completed' or not qa['engineering_gate_passed'] or
            not qa['verification']['weights_unchanged'] or not report['numerical_gate']):
        raise ValueError('GPU batching/numerical/task-isolation gates have not passed')
    isolation=a.benchmark/'isolation/smoke/gpu_1'
    checked=json.loads((isolation/'status.json').read_text())
    sessions=[json.loads(line) for line in (isolation/'session_begin.jsonl').read_text().splitlines()]
    if (checked['status']!='completed' or not checked['verification']['weights_unchanged'] or
            [s['task'] for s in sessions]!=['turning_on_radio','clean_a_keyboard'] or
            any(not all(s[k] for k in ('all_memories_empty','independent_ledgers','contexts_cleared'))
                for s in sessions) or sessions[1]['requests_before']<=sessions[0]['requests_before']):
        raise ValueError('Actual persistent-model cross-task isolation did not pass')
    if not any(r['task']=='clean a keyboard' and r['engineering_gate_passed'] for r in qa['results']):
        raise ValueError('Missing fresh replay of the second-task outputs')
    mixed=json.loads((a.mixed_audit/'audit.json').read_text())
    if (mixed['kind']!='diagnostic_mixed_causal_train_rows_not_rollout' or
        not mixed['engineering_gate_passed'] or not mixed['verification']['weights_unchanged'] or
        len(set(mixed['clocks']))<2):
        raise ValueError('Unequal-history slot QA did not pass')
    for configuration,expected_count in [('serial2',4),('batch2',4),('batch4',4),('isolation',4)]:
        paths=sorted((a.benchmark/configuration).glob('smoke_tasks/*/json/*.json'))
        if len(paths)!=expected_count:raise ValueError('Missing TRAIN benchmark metrics')
        for path in paths:
            record=json.loads(path.read_text());video=path.parent.parent/'videos'/(path.stem+'.mp4')
            probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0',
                '-show_entries','stream=nb_frames,width,height,r_frame_rate','-of','json',str(video)],text=True))['streams'][0]
            if int(probe['nb_frames'])!=record['steps'] or probe['r_frame_rate']!='30/1' or (
                    probe['width'],probe['height'])!=(672,448):
                raise ValueError('TRAIN benchmark media/step mismatch')
    candidate=report['configurations'][f'batch{a.num_envs}']
    if candidate['control_steps_per_second']<=report['configurations']['serial2']['control_steps_per_second']:
        raise ValueError('Selected batching failed to improve actual throughput')
    commit=subprocess.check_output(['git','-C',str(REPO),'rev-parse','HEAD'],text=True).strip()
    if subprocess.check_output(['git','-C',str(REPO),'status','--porcelain'],text=True).strip():
        raise ValueError('Resume source must be clean and frozen')
    components=['common.py','native_engine.py','batched_engine.py','batch_core.py','sparse_cache.py',
                'serve.py','wire.py','rgb_wrapper.py','run_task.py','launch_sim.sh','official_manifest.json']
    for name in components:
        expected=subprocess.check_output(['git','-C',str(REPO),'show',
            bench['source_commit']+':scripts/eval/memlite_sft100/'+name])
        if (SOURCE/name).read_bytes()!=expected:
            raise ValueError('Candidate runtime differs from GPU-tested source: '+name)
    if (subprocess.check_output(['git','-C',str(G05),'rev-parse','HEAD'],text=True).strip()!=G05_COMMIT or
            subprocess.check_output(['git','-C',str(G05),'status','--porcelain'],text=True).strip()):
        raise ValueError('G0.5 dependency changed')
    official={str(f.relative_to(OFFICIAL)):sha256(f) for f in OFFICIAL.rglob('*')
              if f.is_file() and f.suffix in ('.py','.yaml','.json','.kit')}
    if official!=old['official_files']:raise ValueError('Official simulator changed')
    for side,pin in old['checkpoints'].items():
        for pk,hk in [('path','sha256'),('config','config_sha256'),('stats_path','stats_sha256')]:
            if sha256(pin[pk])!=pin[hk]:raise ValueError('Model/config/stats changed: '+side+'/'+pk)
    for case in old['cases']:
        if sha256(case['instance_path'])!=case['instance_sha256']:raise ValueError('Public case changed')
    parts=remaining_parts(old['tasks'],sealed['records'],old['nominal_horizons_for_scheduling'],a.num_envs)
    a.job.mkdir()
    for name in ('workers','parts','tasks','submission','claims'):(a.job/name).mkdir()
    inherited=[]
    for row in sealed['records']:
        task=a.job/'tasks'/row['task']
        for kind in ('json','videos'):(task/kind).mkdir(parents=True,exist_ok=True)
        local={**row}
        for field,kind in [('metrics','json'),('video','videos')]:
            original=Path(row[field]);target=task/kind/original.name
            if sha256(original)!=row[field+'_sha256']:raise ValueError('Inherited original changed')
            shutil.copy2(original,target)
            if sha256(target)!=row[field+'_sha256']:raise ValueError('Inherited copy mismatch')
            local[field]=str(target)
        inherited.append(local|dict(source_commit=old['source_commit'],original_metrics=row['metrics'],
                                    original_video=row['video'],inference_mode='serial',num_envs=2))
    atomic_json(a.job/'inherited_inventory.json',inherited)
    old_commands=[dict(path=str(path),command=json.loads(path.read_text()))
                  for path in sorted((a.prior/'tasks').glob('*/command.json'))]
    atomic_json(a.job/'prior_commands.json',old_commands)
    smoke=a.benchmark/f'batch{a.num_envs}/smoke/gpu_1/status.json'
    protocols=[dict(source_commit=old['source_commit'],inference_mode='serial',num_envs=2,
                    completed_cases=len(inherited)),
               dict(source_commit=commit,inference_mode='batch',max_num_envs=a.num_envs,
                    remaining_cases=1000-len(inherited),tail_envs=[1,2])]
    manifest=old|dict(kind='native_sft100_admin_resume',created=time.time(),status='prepared_not_started',
        source_commit=commit,source_path=str(SOURCE),num_envs=a.num_envs,inference_mode='batch',
        owner='Codex/EVAL-BATCH-SPEED-10383',parts=parts,prior_job=str(a.prior),benchmark=str(a.benchmark),
        benchmark_report_sha256=sha256(a.benchmark/'report.json'),
        numerical_audit_sha256=sha256(a.benchmark/'numerical_audit/audit.json'),
        cross_task_isolation_sha256=sha256(isolation/'session_begin.jsonl'),
        mixed_history_audit=dict(path=str(a.mixed_audit/'audit.json'),sha256=sha256(a.mixed_audit/'audit.json')),
        smoke_evidence=dict(path=str(smoke),sha256=sha256(smoke)),
        inherited_inventory_sha256=sha256(a.job/'inherited_inventory.json'),
        administrative_stop_receipt=dict(path=str(receipt),sha256=sha256(receipt)),
        interrupted_attempts=sealed['interrupted_attempts'],evaluation_protocols=protocols,
        numerical_notice='Batching changes BF16 arithmetic; not bitwise policy equivalence. No outcome-based selection.',
        noise_protocol='same native FM distribution; scalar-shaped draws in row order, seed17 per instance batch',
        automatic_retries=False)
    atomic_json(a.job/'manifest.json',manifest)
    for name in ('rgb_wrapper.py','r1pro.yaml','submission_checklist.json'):
        shutil.copy2(a.prior/'submission'/name,a.job/'submission'/name)
    print(json.dumps(dict(job=str(a.job),inherited=len(inherited),remaining=1000-len(inherited),parts=len(parts),
                         source_commit=commit)))


if __name__=='__main__':main()
