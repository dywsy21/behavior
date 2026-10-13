"""Count every prescribed case; archive official bytes without rewriting them."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import zipfile

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'memlite_sft100'))
from common import aggregate,atomic_json,sha256


def reproduction_readme(manifest):
    return f'''# MEM-Lite full public evaluation

This archive contains original evaluator metrics, wrapper/robot configuration,
and reproducibility metadata. It is NOT a completed model-delivery submission.

- Source repository: https://github.com/dywsy21/behavior.git
- Frozen source commit: `{manifest['source_commit']}`
- Official evaluator commit: `{manifest['official_commit']}`
- Policy config SHA256: `{manifest['config_sha256']}`
- Exact model asset manifest SHA256: `{manifest['pack_manifest_sha256']}`
- Cases: all 100 tasks, public indices 0–9 (instances 301–310), rollout 0.
- Seeds: simulator 0; policy 17. No checkpoint selection using these results.
- Timeout: unchanged official 1.5 × mean human trajectory length.
- Observation: three onboard RGB streams and original proprioception only.
- Planning/feedback state is separate for each episode and slot. Action clocks
  advance only after the official simulator step returns successfully.
- High-level refresh: every 128 applied controls. Low: FP32 FM10, predict 32,
  execute first 16; real 23 controls with the fixed model 27-D mapping.
- Observer: separately bound frozen backbone/adapter; calibration is GRASP-only.
  Unsupported skills remain UNKNOWN. Model predictions are not physical truth.

## Reproduction

Use the frozen commit, the immutable assets named in `run-manifest.json`, and
the registered model/simulator Python environments from `policy-config.json`.
The asset SHA refers to the complete pack, not just a checkpoint filename.
Recreate a NEW run directory; do not overwrite or reroll the archived run.

```bash
python scripts/eval/memlite_final/prepare.py --config POLICY_CONFIG --pack MODEL_PACK --manifest-sha256 {manifest['pack_manifest_sha256']} --job NEW_RUN
python scripts/eval/memlite_final/worker.py --job NEW_RUN --gpu 0 --num-envs 2 --preflight
python scripts/eval/memlite_final/worker.py --job NEW_RUN --gpu 0 --num-envs 4 --preflight
python scripts/eval/memlite_final/supervise.py --job NEW_RUN
```

Each preflight consumes the same radio/keyboard TRAIN indices 1–4 with 256
controls. Batch size is chosen by actual end-to-end controls/s before public
evaluation. See `accepted-preflight.json` for measured choice and exact evidence.
The original official evaluator owns physics, scores, terminal conditions and
video writing; the accompanying transport adds execution receipts only.

## Outputs and remaining delivery requirements

`json/` contains all 1,000 original JSON files without editing. The 1,000 original
30fps MP4 files are stored separately; `inventory.json` binds video paths, SHA256,
frame counts and original metrics SHA256. Captions/diagnostic videos are NOT used.
`summary.json` reports final official Q and binary full-task success, not peak Q
or local skill completion. Previously inspected public instance 301 is a dev case.

The host has 72GiB-class RTX GPUs. This does not prove single-24GB compatibility.
Docker 24GB acceptance OR at least 50 hosted policy ports, external video upload,
team declarations and final submission are separate pending items. No public
upload or official submission is authorized or claimed by this archive.
'''


def report(job, *, final=False):
    job=Path(job);manifest=json.loads((job/'manifest.json').read_text())
    records=[];inventory=[]
    for part in sorted((job/'parts').glob('*')):
        for metric in sorted((part/'json').glob('*.json')):
            row=json.loads(metric.read_text());records.append(row)
            video=part/'videos'/(metric.stem+'.mp4')
            item=dict(task=row['task'],instance_id=row['instance_id'],rollout_id=row['rollout_id'],
                metrics=str(metric),video=str(video),video_exists=video.is_file())
            if final:
                if not video.is_file() or video.stat().st_size<1000:raise ValueError('Missing original video '+str(video))
                probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0',
                    '-show_entries','stream=nb_frames,r_frame_rate,width,height','-of','json',str(video)],text=True))
                stream=probe['streams'][0]
                if int(stream['nb_frames'])!=row['steps'] or stream['r_frame_rate']!='30/1':
                    raise ValueError('Official video/control count mismatch '+str(video))
                item.update(metrics_sha256=sha256(metric),video_sha256=sha256(video),video_bytes=video.stat().st_size,video_probe=stream)
            inventory.append(item)
    summary=aggregate(manifest['tasks'],records)
    summary.update(config_sha256=manifest['config_sha256'],source_commit=manifest['source_commit'],
        original_outputs_unmodified=True,optimizer_updates=0,final_model_submission_ready=False)
    atomic_json(job/'summary.json',summary)
    if final:
        if summary['status']!='complete':raise ValueError('Cannot seal an incomplete evaluation as 1000/1000')
        atomic_json(job/'inventory.json',inventory)
        if sha256(manifest['config_path'])!=manifest['config_sha256']:
            raise ValueError('Final policy config identity changed')
        readme=job/'submission/README.md'
        with readme.open('x') as stream:stream.write(reproduction_readme(manifest))
        target=job/'submission/evaluation_outputs.zip'
        with zipfile.ZipFile(target,'x',compression=zipfile.ZIP_DEFLATED) as archive:
            for item in inventory:archive.write(item['metrics'],'json/'+Path(item['metrics']).name)
            for name in ('rgb_wrapper.py','r1pro.yaml'):archive.write(job/'submission'/name,name)
            # Transport is part of reproducibility, not an edited metric or a
            # hidden privileged actor feature. Final endpoint/container QA is
            # deliberately still a separate unfulfilled submission requirement.
            for name in ('client.py','lifecycle.py','run_task.py'):archive.write(HERE/name,'transport/'+name)
            archive.write(HERE.parent/'full_episode_client.py','transport/full_episode_client.py')
            archive.write(HERE.parent/'memlite_sft100/wire.py','transport/wire.py')
            archive.write(job/'manifest.json','run-manifest.json')
            archive.write(manifest['config_path'],'policy-config.json')
            archive.write(job/'accepted-preflight.json','accepted-preflight.json')
            archive.write(job/'inventory.json','inventory.json')
            archive.write(job/'summary.json','summary.json')
            archive.write(readme,'README.md')
        atomic_json(job/'submission/package-status.json',dict(status='evaluation_outputs_only_not_final_model_delivery',
            metrics=1000,videos=1000,zip_path=str(target),zip_sha256=sha256(target),video_inventory=str(job/'inventory.json'),
            videos_uploaded=False,docker_24gb_validated=False,external_endpoint_50_ports_validated=False,
            team_declarations_confirmed=False,officially_submitted=False))
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--job',type=Path,required=True);p.add_argument('--final',action='store_true')
    a=p.parse_args();r=report(a.job,final=a.final)
    print(json.dumps({k:r[k] for k in ('status','completed','missing','official_q_score','official_sr')}))
