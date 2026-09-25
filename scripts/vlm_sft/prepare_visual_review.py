"""Bounded original-resolution visual review candidates, NEVER training labels.

No simulator, model, action/state array or privileged object annotation is used.
The later manual review/label release must be separate from image extraction.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

REPO=Path(__file__).resolve().parents[2]
ROOT=Path('/mnt/sdc1/robodojo/datasets/behavior2026_g05_tasks_0_4')
COUNTS=REPO/'configs/vlm_sft/h09r_train_feasibility_counts.json'
COUNTS_SHA='d94850ebfeadeef8b28f618e543ebae7cb7a18bd127c5d79ba6f4c18c0f015e3'
QUARANTINE=Path('/mnt/sdc1/robodojo/datasets/memlite_skill_annotations_task0_4_v6_formal_a_composite_release_v2_20260909/meta/quarantine_ranges.parquet')
LATER_PROTECTED={(0,138),(3,242),(1,1),(1,71)}
PRIOR_STATE_TRAINING={(1,114),(1,192)}
CAMERAS={'head':'zed_link_camera_0','left_wrist':'left_realsense_link_camera_0',
         'right_wrist':'right_realsense_link_camera_0'}
QUERIES={0:'the radio',1:'the wastebasket',3:'the plate holding food'}
BUDGET={'cpu_ids':[48,49],'wall_seconds':240,'outer_wall_seconds':270,'images':108,'bytes':1024**3}


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def atomic_json(path,value):
    temporary=path.with_name(path.name+'.partial')
    with temporary.open('x') as f:
        json.dump(value,f,indent=2,allow_nan=False);f.write('\n')
        f.flush();os.fsync(f.fileno())
    temporary.replace(path)


def validate_collection(output,*,require_seal=True):
    """A directory or even a manifest alone is never a completed collection."""
    manifest=output/'review_manifest.json'
    if (output/'failure.json').exists():raise ValueError('Failed/partial visual collection')
    result=json.loads(manifest.read_text())
    if (result.get('schema')!='h76-visual-review-candidates-v1' or
            result.get('status')!='IMAGES_ONLY_REVIEW_PENDING' or result.get('training_eligible') is not False or
            result.get('image_count')!=108 or len(result.get('rows',[]))!=36):
        raise ValueError('Incomplete visual review manifest')
    if sha(COUNTS)!=COUNTS_SHA:raise ValueError('Source counts identity changed')
    sources=select_sources(json.loads(COUNTS.read_text()))
    if result.get('sources')!=sources or result.get('counts_sha256')!=COUNTS_SHA:
        raise ValueError('Visual source selection changed')
    started=json.loads((output/'extraction_started.json').read_text())
    if started!={'status':'RUNNING_REVIEW_CANDIDATES_ONLY','counts_sha256':COUNTS_SHA,
                'sources':sources,'training_eligible':False,'budget':BUDGET}:
        raise ValueError('Missing or altered visual start receipt')
    wall=result.get('wall_seconds');size=result.get('image_bytes')
    if (type(wall) not in (int,float) or not math.isfinite(wall) or not 0<=wall<=BUDGET['wall_seconds'] or
            type(size) is not int or not 0<size<=BUDGET['bytes'] or
            any(type(result.get(k)) is not int or result[k]!=0 for k in ('new_model_calls','new_controls','training_updates'))):
        raise ValueError('Visual preparation budget/zero-action contract changed')
    expected_states={(s['task'],s['instance'],s['episode'],f):s['candidate_split']
                     for s in sources for f in frames_for(s,[])}
    seen_states=set()
    expected=set();total=0
    groups={}
    for row in result['rows']:
        if (row.get('manual_review')!='PENDING' or row.get('training_eligible') is not False or
                set(row.get('images',{}))!=set(CAMERAS) or set(row.get('image_receipts',{}))!=set(CAMERAS)):
            raise ValueError('Missing views or premature training release')
        group=(row['task'],row['instance']);split=row['candidate_split']
        state=(*group,row['episode'],row['frame'])
        if (state in seen_states or expected_states.get(state)!=split or row.get('query')!=QUERIES[row['task']] or
                row.get('id')!=f't{row["task"]}_i{row["instance"]}_e{row["episode"]}_f{row["frame"]:06d}'):
            raise ValueError('Wrong or duplicated fixed visual source frame')
        seen_states.add(state)
        if split not in ('visual_train','visual_validation') or groups.get(group,split)!=split:
            raise ValueError('Visual source split mismatch')
        groups[group]=split
        for view,relative in row['images'].items():
            path=Path(relative)
            if path.parent!=Path('images') or path.suffix!='.png' or relative in expected:
                raise ValueError('Invalid/duplicated visual image path')
            expected.add(relative)
            if sha(output/path)!=row['image_receipts'][view]['png_sha256']:
                raise ValueError('Visual image content changed')
            total+=(output/path).stat().st_size
    actual={p.relative_to(output).as_posix() for p in (output/'images').iterdir()}
    if actual!=expected or total!=result['image_bytes'] or len(groups)!=12 or seen_states!=set(expected_states):
        raise ValueError('Partial or unexpected visual images')
    if (sum(s=='visual_train' for s in groups.values())!=9 or
            sum(s=='visual_validation' for s in groups.values())!=3 or
            set(groups) & (LATER_PROTECTED|PRIOR_STATE_TRAINING)):
        raise ValueError('Protected or incorrectly grouped visual split')
    if require_seal:
        seal=json.loads((output/'complete.json').read_text())
        if seal!={'status':'IMAGES_COMPLETE_REVIEW_PENDING','manifest_sha256':sha(manifest),
                  'image_count':108,'image_bytes':total,'training_eligible':False}:
            raise ValueError('Missing/invalid visual completion seal')
    return result


def select_sources(counts):
    excluded={tuple(x) for rows in counts['exclusions'].values() for x in rows}|LATER_PROTECTED|PRIOR_STATE_TRAINING
    chosen=[]
    for task in QUERIES:
        candidates=[r for r in counts['sources'] if r['task']==task and r['cohort']=='additional_train'
                    and (task,r['instance']) not in excluded]
        groups=[r['instance'] for r in candidates]
        if len(set(groups))!=len(groups) or len(candidates)<4:
            raise ValueError('Four unique unprotected TRAIN sources per task required')
        candidates.sort(key=lambda r:hashlib.sha256(f'h76-visible-source-v1:{task}:{r["instance"]}'.encode()).hexdigest())
        for index,source in enumerate(candidates[:4]):
            chosen.append({**source,'candidate_split':'visual_validation' if index==3 else 'visual_train'})
    return chosen


def frames_for(source, quarantine):
    n=source['frames']
    if type(n) is not int or n<30:raise ValueError('Invalid source frame count')
    frames=[0,n//2,9*n//10]
    for frame in frames:
        if any(r['episode_index']==source['episode'] and r['frame_start']<=frame<=r['frame_end'] for r in quarantine):
            raise ValueError('Pre-registered visual frame is quarantined; do not silently substitute')
    return frames


def choose_frame(container, stream, timestamp):
    # Same half-frame tolerance as the existing expert image extractor.
    container.seek(int(timestamp/stream.time_base),stream=stream,backward=True)
    for frame in container.decode(stream):
        if frame.pts is None:continue
        actual=float(frame.pts*stream.time_base)
        if actual>=timestamp-1/60:
            if abs(actual-timestamp)>1/60+1e-4:raise ValueError('Decoded video frame is not aligned')
            return frame,actual
    raise ValueError('Requested source image is missing')


def prepare(output):
    started=time.monotonic()  # Includes optional dependency imports.
    import av
    import pyarrow.parquet as pq
    if sha(COUNTS)!=COUNTS_SHA:raise ValueError('Source counts identity changed')
    counts=json.loads(COUNTS.read_text())
    sources=select_sources(counts)
    if sha(QUARANTINE)!=counts['source_identity']['quarantine_sha256']:
        raise ValueError('Quarantine identity changed')
    quarantine=pq.read_table(QUARANTINE).to_pylist()
    metadata={}
    for path,digest in counts['source_identity']['episode_meta_sha256'].items():
        if sha(path)!=digest:raise ValueError('Episode metadata identity changed')
        for row in pq.read_table(path).to_pylist():
            episode=row['episode_index']
            if episode in metadata:raise ValueError('Duplicate episode metadata')
            metadata[episode]=row
    if subprocess.check_output(['git','-C',str(REPO),'status','--porcelain'],text=True).strip():
        raise ValueError('Use an immutable clean Git source')
    if output.exists():raise FileExistsError('Never overwrite a review collection')
    if shutil.disk_usage(output.parent).free<80*1024**3:raise RuntimeError('80GiB disk reserve required')
    os.sched_setaffinity(0,{48,49})
    output.mkdir();(output/'images').mkdir()
    rows=[];total_bytes=0
    def bounded():
        if time.monotonic()-started>240:raise TimeoutError('Visual review preparation budget exceeded')
        if total_bytes>1024**3:raise RuntimeError('Visual review exceeds 1GiB budget')
    try:
        atomic_json(output/'extraction_started.json',{'status':'RUNNING_REVIEW_CANDIDATES_ONLY',
            'counts_sha256':COUNTS_SHA,'sources':sources,'training_eligible':False,
            'budget':BUDGET})
        for source in sources:
            bounded()
            meta=metadata[source['episode']]
            if (meta['task_index']!=source['task'] or meta['task_instance_id']!=source['instance'] or
                    meta['length']!=source['frames']):raise ValueError('Source episode/group identity changed')
            selected=[]
            for frame in frames_for(source,quarantine):
                key=f't{source["task"]}_i{source["instance"]}_e{source["episode"]}_f{frame:06d}'
                selected.append({'id':key,'task':source['task'],'instance':source['instance'],
                    'episode':source['episode'],'frame':frame,'candidate_split':source['candidate_split'],
                    'query':QUERIES[source['task']],'images':{},'image_receipts':{},
                    'manual_review':'PENDING','training_eligible':False})
            for view,camera in CAMERAS.items():
                stem='videos/observation.rgb.'+camera
                path=ROOT/f'{stem}/chunk-{meta[stem+"/chunk_index"]:03d}/file-{meta[stem+"/file_index"]:03d}.mp4'
                stat=path.stat();identity=(stat.st_size,stat.st_mtime_ns)
                with av.open(str(path)) as container:
                    stream=container.streams.video[0];stream.codec_context.thread_count=2
                    if abs(float(stream.average_rate)-30)>1e-6:raise ValueError('Original 30Hz source required')
                    for row in selected:
                        bounded()
                        timestamp=float(meta[stem+'/from_timestamp'])+row['frame']/30
                        if not math.isfinite(timestamp) or timestamp<0:raise ValueError('Invalid video time')
                        frame,actual=choose_frame(container,stream,timestamp)
                        image=frame.to_image().convert('RGB')  # No resize, crop, montage or overlays.
                        rel=f'images/{row["id"]}_{view}.png';target=output/rel;image.save(target)
                        total_bytes+=target.stat().st_size
                        row['images'][view]=rel
                        row['image_receipts'][view]={'video':str(path),'video_bytes':stat.st_size,
                            'video_mtime_ns':stat.st_mtime_ns,'requested_timestamp_s':timestamp,
                            'actual_timestamp_s':actual,'source_frame_pts':frame.pts,
                            'source_time_base':str(stream.time_base),'resolution':list(image.size),
                            'png_sha256':sha(target),'raw_resolution_preserved':True}
                now=path.stat()
                if (now.st_size,now.st_mtime_ns)!=identity:raise ValueError('Source video changed during decode')
            rows.extend(selected)
            print(json.dumps({'episode':source['episode'],'states':len(rows),'images':len(rows)*3}),flush=True)
        bounded()
        if len(rows)!=36:raise ValueError('Exact 36-state/108-image review budget required')
        result={'schema':'h76-visual-review-candidates-v1','status':'IMAGES_ONLY_REVIEW_PENDING',
            'code_commit':subprocess.check_output(['git','-C',str(REPO),'rev-parse','HEAD'],text=True).strip(),
            'counts_sha256':COUNTS_SHA,'sources':sources,'protected_later_groups':[list(x) for x in sorted(LATER_PROTECTED)],
            'prior_state_training_groups_excluded':[list(x) for x in sorted(PRIOR_STATE_TRAINING)],
            'selection':'4 additional_train instances/task; hash h76-visible-source-v1; first3 train,last1 visual_validation',
            'frame_selection':'0, length//2, 9*length//10; no quarantine replacement',
            'rows':rows,'image_count':108,'image_bytes':total_bytes,'wall_seconds':time.monotonic()-started,
            'training_eligible':False,'new_model_calls':0,'new_controls':0,'training_updates':0}
        atomic_json(output/'review_manifest.json',result)
        validate_collection(output,require_seal=False)
        bounded()
        atomic_json(output/'complete.json',{'status':'IMAGES_COMPLETE_REVIEW_PENDING',
            'manifest_sha256':sha(output/'review_manifest.json'),'image_count':108,'image_bytes':total_bytes,
            'training_eligible':False})
        bounded()  # A post-seal overrun creates failure.json; consumers reject it.
        return result
    except BaseException as error:
        try:
            atomic_json(output/'failure.json',{'error':repr(error),'completed_states':len(rows),
                'image_bytes':total_bytes,'training_eligible':False,'wall_seconds':time.monotonic()-started})
        except BaseException as secondary:
            print('Visual review failure evidence also failed: '+repr(secondary),file=sys.stderr)
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args();result=prepare(args.output)
    print(json.dumps({k:result[k] for k in ('status','image_count','image_bytes','wall_seconds')}))
