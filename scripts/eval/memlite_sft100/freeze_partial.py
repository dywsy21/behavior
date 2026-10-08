"""Seal stopped evaluation outputs; do not edit metrics, videos, or attempts."""
import argparse
import json
from pathlib import Path
import subprocess
import time
from common import aggregate, atomic_json, sha256


def freeze(job, output):
    if output.exists():
        raise ValueError('A frozen receipt must not be overwritten')
    status=json.loads((job/'status.json').read_text())
    if not (job/'STOP').exists() or status['status'] not in ('needs_diagnosis','failed'):
        raise ValueError('Only a stopped evaluation may be sealed')
    workers=[json.loads(p.read_text()) for p in sorted((job/'workers').glob('gpu_*/status.json'))]
    if len(workers)!=8 or any(not w.get('verification',{}).get('weights_unchanged') for w in workers):
        raise ValueError('Need all eight post-stop weight verification receipts')
    manifest=json.loads((job/'manifest.json').read_text())
    rows=[];records=[]
    for path in sorted((job/'tasks').glob('*/json/*.json')):
        record=json.loads(path.read_text())
        name=f"{record['task']}_{record['instance_id']}_{record['rollout_id']}"
        if path.stem != name:
            raise ValueError('Official filename/identity mismatch')
        video=path.parent.parent/'videos'/(name+'.mp4')
        probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0',
            '-show_entries','stream=codec_name,width,height,nb_frames,duration,r_frame_rate',
            '-of','json',str(video)],text=True))['streams'][0]
        if (int(probe['nb_frames'])!=record['steps'] or probe['r_frame_rate']!='30/1' or
            (probe['width'],probe['height'])!=(672,448) or
            abs(float(probe['duration'])-record['steps']/30)>.04):
            raise ValueError('Official media/step mismatch: '+name)
        records.append(record)
        rows.append(dict(task=record['task'],instance_id=record['instance_id'],rollout_id=record['rollout_id'],
            metrics=str(path),metrics_sha256=sha256(path),video=str(video),video_sha256=sha256(video),
            video_bytes=video.stat().st_size,probe=probe,steps=record['steps']))
    report=aggregate(manifest['tasks'],records)
    completed={(r['task'],r['instance_id']) for r in rows}
    interrupted=[]
    for path in sorted((job/'tasks').glob('*/attempts/*.json')):
        attempt=json.loads(path.read_text())
        if attempt['status']!='completed':
            missing=[i for i in attempt['instance_ids'] if (attempt['task'],i) not in completed]
            interrupted.append(dict(path=str(path),sha256=sha256(path),original_status=attempt['status'],
                                    task=attempt['task'],instance_ids_without_complete_outputs=missing))
    receipt=dict(kind='administrative_stop_inventory',created=time.time(),job=str(job),
        manifest_sha256=sha256(job/'manifest.json'),source_commit=manifest['source_commit'],
        reason='User authorized evaluation speed optimization; not outcome-based selection',
        records=rows,completed=len(rows),partial_summary=report,interrupted_attempts=interrupted,
        original_outputs_unmodified=True,weights_unchanged_all_8=True,
        restart_rule='Only uncompleted cases; keep administrative interruptions in provenance; never overwrite')
    atomic_json(output,receipt)
    return receipt


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--job',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    r=freeze(a.job,a.output)
    print(json.dumps(dict(completed=r['completed'],interrupted_attempts=len(r['interrupted_attempts']),
                         receipt=str(a.output),sha256=sha256(a.output))))
