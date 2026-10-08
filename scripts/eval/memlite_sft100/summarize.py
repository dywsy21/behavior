"""Report coverage, task-macro final Q and full success without altering outputs."""
import argparse
import json
from pathlib import Path
import time
import zipfile
from common import aggregate, atomic_json, sha256


def summarize(job,final=False):
    manifest=json.loads((job/'manifest.json').read_text())
    records=[];files=[]
    for path in sorted((job/'tasks').glob('*/json/*.json')):
        record=json.loads(path.read_text())
        video=path.parent.parent/'videos'/(path.stem+'.mp4')
        if not video.is_file() or video.stat().st_size<1000:
            raise ValueError('Metrics without matching recorded video: '+str(path))
        records.append(record)
        files.append(dict(metrics=str(path),video=str(video),video_bytes=video.stat().st_size,
                          metrics_sha256=sha256(path),**({ 'video_sha256':sha256(video)} if final else {})))
    report=aggregate(manifest['tasks'],records)
    report.update(updated=time.time(),source_commit=manifest['source_commit'],
                  checkpoints=manifest['checkpoints'],rollout_files=files,
                  blind_test=False,reason=manifest['development_notice'])
    atomic_json(job/'summary.json',report)
    checklist=json.loads((job/'submission/submission_checklist.json').read_text())
    checklist['metrics_json']['collected']=len(records);checklist['videos']['collected']=len(files)
    atomic_json(job/'submission/submission_checklist.json',checklist)
    if final:
        atomic_json(job/'submission/rollout_inventory.json',files)
        lines=[
            '# MEM-Lite Stage-1 SFT evaluation',
            '', 'This is a reproducibility bundle, not an automatic challenge submission.',
            f"Source commit: {manifest['source_commit']}",
            f"Official simulator: {manifest['official_tag']} / {manifest['official_commit']}",
            'Public indices 0–9 (actual IDs 301–310), one rollout each; default 1.5x human timeout.',
            'RGBOnlyFullResWrapper, unchanged bundled r1pro.yaml; 23D raw commands.',
            'One frame / 3 RGB cameras; high planner greedy every128controls; native FM10 steps;',
            'predict32 / execute16 from index0, policy seed17, simulator seed0; no PPO exploration/update.',
            'The provided run_task.py wraps stock BatchedEvaluator.run in five two-env batches.',
            'Exact per-task shell command and safe environment overrides are in task_commands.json.',
            'Original rollout JSON and MP4 files have not been edited. Missing outputs are not rerolled.',
            'Weights and config/stats SHA values are in model_provenance.json.',
            'Docker image or external IP with >=50 ports: NOT YET PROVIDED.',
            'Single24GB policy serving: pending measured acceptance; video hosting URL: pending.',
            'Public301 has prior diagnostic exposure; do not call this a blind test.',
        ]
        commands=[json.loads(p.read_text()) for p in sorted((job/'tasks').glob('*/command.json'))]
        package=job/'submission/results.zip'
        if package.exists():raise ValueError('Refusing to overwrite an existing final submission bundle')
        with zipfile.ZipFile(package,'x',compression=zipfile.ZIP_DEFLATED) as z:
            z.writestr('README.md','\n'.join(lines)+'\n')
            z.writestr('model_provenance.json',json.dumps(manifest['checkpoints'],indent=2))
            z.writestr('task_commands.json',json.dumps(commands,indent=2))
            for name in ['rgb_wrapper.py','r1pro.yaml','submission_checklist.json','rollout_inventory.json']:
                z.write(job/'submission'/name,name)
            for row in files:z.write(row['metrics'],'metrics/'+Path(row['metrics']).name)
        report['submission_zip']=str(package);report['submission_zip_sha256']=sha256(package)
        atomic_json(job/'summary.json',report)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--job',type=Path,required=True);p.add_argument('--final',action='store_true')
    a=p.parse_args();r=summarize(a.job,a.final)
    print(json.dumps({k:r[k] for k in ['status','completed','expected','official_q_score','official_sr']}))
