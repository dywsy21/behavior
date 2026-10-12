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
        atomic_json(job/'submission/package-status.json',dict(status='evaluation_outputs_only_not_final_model_delivery',
            metrics=1000,videos=1000,zip_path=str(target),zip_sha256=sha256(target),video_inventory=str(job/'inventory.json'),
            videos_uploaded=False,docker_24gb_validated=False,external_endpoint_50_ports_validated=False,
            team_declarations_confirmed=False,officially_submitted=False))
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--job',type=Path,required=True);p.add_argument('--final',action='store_true')
    a=p.parse_args();r=report(a.job,final=a.final)
    print(json.dumps({k:r[k] for k in ('status','completed','missing','official_q_score','official_sr')}))
