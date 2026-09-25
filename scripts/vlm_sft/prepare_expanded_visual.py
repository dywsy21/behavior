"""H80 instance-separated RAW candidates; no teacher, labels, or train release."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
import json
import hashlib
from fractions import Fraction
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time

from prepare_visual_review import (REPO, ROOT, COUNTS, COUNTS_SHA, QUARANTINE,
    CAMERAS, LATER_PROTECTED, PRIOR_STATE_TRAINING, sha, atomic_json, select_sources, choose_frame)

VERSION = 'h80-expanded-raw-candidates-v1'
BUDGET = {'seconds':7200, 'outer_seconds':7230, 'cleanup_seconds':30,
          'max_images':42240, 'max_bytes':20*1024**3, 'image_bytes':int(19.5*1024**3),
          'min_free_bytes':80*1024**3, 'cpu_ids':list(range(48,56)), 'workers':4,
          'train_groups_per_task':80, 'validation_groups_per_task':8, 'test_groups_per_task':8,
          'train_frames_per_group':32, 'eval_frames_per_group':16, 'minimum_spacing_frames':30}
TASKS = tuple(range(5))
TASKS_SHA = '80bddeab4c6c4ab6944ba932ce63c48fdc9a1220f90a1f558382334a19ef5921'


def grouped_selection(episodes, counts):
    """Choose groups before viewing pixels, teacher answers, or model scores."""
    protected = {tuple(v) for group in counts['exclusions'].values() for v in group} | LATER_PROTECTED
    known_train = {(r['task'],r['instance']) for r in counts['sources'] if r['cohort']=='h09_train'} | PRIOR_STATE_TRAINING
    for row in select_sources(counts):
        group = (row['task'],row['instance'])
        if row['candidate_split']=='visual_validation': protected.add(group)
        else: known_train.add(group)
    for task in TASKS:
        rows = sorted((r for r in episodes if r['task_index']==task),key=lambda r:r['episode_index'])
        if len(rows)!=200 or len({r['episode_index'] for r in rows})!=200:
            raise ValueError('Original 200 episodes/task changed')
        protected.update((task,r['task_instance_id']) for r in rows[-10:])
    sources=[]
    for task in TASKS:
        pool=sorted((r for r in episodes if r['task_index']==task and (task,r['task_instance_id']) not in protected),
                    key=lambda r:r['episode_index'])
        unique={}
        for row in pool: unique.setdefault(row['task_instance_id'],row)
        def ordered(tag, rows):
            return sorted(rows,key=lambda r:hashlib.sha256(f'h80:{tag}:41:{task}:{r["task_instance_id"]}'.encode()).hexdigest())
        holdouts=ordered('holdout',(r for r in unique.values() if (task,r['task_instance_id']) not in known_train))[:16]
        held={r['task_instance_id'] for r in holdouts}
        training=ordered('training',(r for r in unique.values() if r['task_instance_id'] not in held))[:80]
        if len(holdouts)!=16 or len(training)!=80: raise ValueError('Insufficient independent source groups')
        for split,rows in (('validation',holdouts[:8]),('test',holdouts[8:]),('train',training)):
            for row in rows:
                sources.append({'split':split,'task':task,'instance':row['task_instance_id'],
                                'episode':row['episode_index'],'frames':row['length']})
    sources.sort(key=lambda r:r['episode'])
    groups=[(s['task'],s['instance']) for s in sources]
    if len(groups)!=480 or len(set(groups))!=480 or set(groups)&protected:
        raise ValueError('Source-instance leakage')
    if any((s['task'],s['instance']) in known_train for s in sources if s['split']!='train'):
        raise ValueError('Old training instance entered new heldout set')
    return sources,{'protected_groups':[list(v) for v in sorted(protected)],
                    'historical_train_groups':[list(v) for v in sorted(known_train)]}


def sample_frames(source, quarantine):
    n=source['frames'];count=32 if source['split']=='train' else 16
    if type(n) is not int or n<30: raise ValueError('Invalid source length')
    intervals=[(r['frame_start'],r['frame_end']) for r in quarantine if r['episode_index']==source['episode']]
    allowed=[f for f in range(n) if not any(lo<=f<=hi for lo,hi in intervals)]
    if not allowed: raise ValueError('Entire selected source quarantined; do not substitute')
    # Quantiles span the released timeline; round-to-even is deterministic.
    proposed=sorted({allowed[round(i*(len(allowed)-1)/(count-1))] for i in range(count)})
    selected=[]
    for frame in proposed:
        if not selected or frame-selected[-1]>=BUDGET['minimum_spacing_frames']: selected.append(frame)
    if len(selected)<2: raise ValueError('Too little non-quarantined temporal coverage')
    return selected


def source_identity():
    import pyarrow.parquet as pq
    if sha(ROOT/'meta/tasks.jsonl')!=TASKS_SHA: raise ValueError('Original task metadata changed')
    if sha(COUNTS)!=COUNTS_SHA: raise ValueError('Historical exclusion manifest changed')
    counts=json.loads(COUNTS.read_text());identity=counts['source_identity']
    if sha(QUARANTINE)!=identity['quarantine_sha256']: raise ValueError('Quarantine changed')
    review=REPO/'configs/vlm_sft/h76_parent_visual_review_v1.json'
    if sha(review)!='3159ed61cb56778343e360e282a9fbb68b64e5dd7d770d0346397fa1380dd136':
        raise ValueError('Historical visual holdout/review changed')
    episodes=[]
    for path,digest in identity['episode_meta_sha256'].items():
        if sha(Path(path))!=digest: raise ValueError('Original source metadata changed')
        episodes.extend(pq.read_table(path).to_pylist())
    sources,exclusions=grouped_selection(episodes,counts)
    quarantine=pq.read_table(QUARANTINE).to_pylist()
    for source in sources: source['selected_frames']=sample_frames(source,quarantine)
    return sources,{r['episode_index']:r for r in episodes}, {
        'counts_sha256':COUNTS_SHA,'quarantine_sha256':identity['quarantine_sha256'],
        'episode_meta_sha256':identity['episode_meta_sha256'],
        'parent_review_sha256':sha(review),'tasks_sha256':TASKS_SHA,**exclusions}


def difference_hash(image):
    import numpy as np
    from PIL import Image
    pixels=np.asarray(image.convert('L').resize((17,16),Image.Resampling.BILINEAR))
    return np.packbits(pixels[:,1:]>pixels[:,:-1]).tobytes().hex()


def validate_output(output, *, require_seal=True):
    import av
    from PIL import Image
    output=Path(output)
    if (output/'failure.json').exists(): raise ValueError('Failed or partial collection is not releasable')
    expected_top={'source_plan.json','images.jsonl','manifest.json','images'}
    if require_seal: expected_top.add('complete.json')
    if output.is_symlink() or not output.is_dir(): raise ValueError('RAW package must be a real directory')
    if {p.name for p in output.iterdir()}!=expected_top:
        raise ValueError('Missing or unauthorized RAW package members')
    for path in output.iterdir():
        if path.is_symlink() or (not path.is_dir() if path.name=='images' else not path.is_file()):
            raise ValueError('RAW package contains linked or non-regular evidence')
    if any(p.is_symlink() or not p.is_file() for p in (output/'images').iterdir()):
        raise ValueError('RAW images must be flat, regular files')
    source=read_json(output/'source_plan.json');receipt=read_json(output/'manifest.json')
    source_keys={'schema','code_commit','budget','sources','source_identity','training_eligible','input_modalities','labels_created'}
    receipt_keys={'schema','status','code_commit','source_plan_sha256','images_manifest_sha256','states','states_by_split',
                  'image_count','image_bytes','unique_exact_pixel_images','wall_seconds','training_eligible',
                  'new_model_calls','new_controls','training_updates'}
    if (set(source)!=source_keys or set(receipt) not in (receipt_keys,receipt_keys|{'extraction_seconds','validation_seconds'}) or
            source.get('labels_created') is not False or source.get('input_modalities')!=['raw_rgb'] or
            receipt.get('schema')!=VERSION or receipt.get('code_commit')!=source.get('code_commit') or
            source.get('schema')!=VERSION or source.get('budget')!=BUDGET or
            source.get('training_eligible') is not False or receipt.get('training_eligible') is not False or
            receipt.get('status')!='RAW_COMPLETE_LABELS_PENDING' or
            receipt.get('source_plan_sha256')!=sha(output/'source_plan.json')):
        raise ValueError('Missing immutable RAW-only provenance')
    if require_seal or 'extraction_seconds' in receipt:
        times=[receipt.get(k) for k in ('extraction_seconds','validation_seconds','wall_seconds')]
        if (any(type(v) not in (int,float) or not math.isfinite(v) or v<0 for v in times) or
                not math.isclose(times[0]+times[1],times[2],rel_tol=1e-9,abs_tol=1e-7)):
            raise ValueError('Final timing does not include extraction and validation')
    expected_sources,metadata,expected_identity=source_identity()
    if source.get('sources')!=expected_sources or source.get('source_identity')!=expected_identity:
        raise ValueError('Source plan differs from fixed original instances/splits/quarantine')
    if (any(type(receipt.get(k)) is not int or receipt[k]!=0 for k in ('new_model_calls','new_controls','training_updates')) or
            type(receipt.get('wall_seconds')) not in (int,float) or not math.isfinite(receipt['wall_seconds']) or
            not 0<receipt['wall_seconds']<=BUDGET['seconds']): raise ValueError('Collection budget/label contract changed')
    expected={f't{s["task"]}_i{s["instance"]}_e{s["episode"]}_f{f:06d}':(s,f)
              for s in source['sources'] for f in s['selected_frames']}
    rows=[json.loads(line) for line in (output/'images.jsonl').read_text().splitlines()]
    if len(rows)!=len(expected) or len({r['id'] for r in rows})!=len(rows): raise ValueError('Missing or duplicated states')
    files=set();total=0;hashes=[];image_count=0;video_headers={}
    for row in rows:
        s,f=expected[row['id']]
        if (set(row)!={'id','task','instance','episode','split','frame','images','image_receipts','training_eligible','manual_review'} or
                any(row.get(k)!=s[k] for k in ('task','instance','episode','split')) or row.get('frame')!=f or
                row.get('training_eligible') is not False or row.get('manual_review')!='PENDING' or
                set(row.get('images',{}))!=set(CAMERAS) or set(row.get('image_receipts',{}))!=set(CAMERAS)):
            raise ValueError('Wrong source group/frame or premature training release')
        for view,name in row['images'].items():
            expected_name=f'images/{row["id"]}_{view}.png'
            if name!=expected_name or name in files: raise ValueError('Wrong image path')
            files.add(name);p=output/name;item=row['image_receipts'][view]
            if set(item)!={'video','video_bytes','video_mtime_ns','source_frame','source_frame_pts','source_time_base',
                            'requested_timestamp_s','actual_timestamp_s','resolution','bytes','png_sha256',
                            'raw_pixels_sha256','difference_hash_256','raw_resolution_preserved'}:
                raise ValueError('RAW receipt contains missing or unauthorized fields')
            if (type(item['source_frame_pts']) is not int or item['source_frame_pts']<0 or
                    type(item['source_time_base']) is not str or Fraction(item['source_time_base'])<=0 or
                    abs(float(item['source_frame_pts']*Fraction(item['source_time_base']))-item['actual_timestamp_s'])>1e-9):
                raise ValueError('Decoded PTS/time-base does not bind actual timestamp')
            if p.is_symlink() or sha(p)!=item['png_sha256'] or p.stat().st_size!=item['bytes']:
                raise ValueError('Extracted image changed')
            with Image.open(p) as im:
                rgb=im.convert('RGB')
                if (list(im.size)!=item['resolution'] or hashlib.sha256(rgb.tobytes()).hexdigest()!=item['raw_pixels_sha256'] or
                        difference_hash(rgb)!=item['difference_hash_256']):
                    raise ValueError('Wrong decoded pixels/perceptual duplicate metadata')
            meta=metadata[s['episode']];stem='videos/observation.rgb.'+CAMERAS[view]
            video=ROOT/f'{stem}/chunk-{meta[stem+"/chunk_index"]:03d}/file-{meta[stem+"/file_index"]:03d}.mp4'
            stat=video.stat();expected_time=float(meta[stem+'/from_timestamp'])+f/30
            if str(video) not in video_headers:
                with av.open(str(video)) as container:
                    stream=container.streams.video[0]
                    video_headers[str(video)]=(str(stream.time_base),float(stream.average_rate))
            time_base,fps=video_headers[str(video)]
            if (item['video']!=str(video) or item['video_bytes']!=stat.st_size or item['video_mtime_ns']!=stat.st_mtime_ns or
                    item['source_time_base']!=time_base or abs(fps-30)>1e-6 or
                    abs(item['requested_timestamp_s']-expected_time)>1e-7):
                raise ValueError('Image source video or original timestamp changed')
            if (item['resolution']!=([720,720] if view=='head' else [480,480]) or
                    item['source_frame']!=f or item.get('raw_resolution_preserved') is not True or
                    not math.isfinite(item['actual_timestamp_s']) or not math.isfinite(item['requested_timestamp_s']) or
                    abs(item['actual_timestamp_s']-item['requested_timestamp_s'])>1/60+1e-4):
                raise ValueError('Changed source frame/image alignment')
            total+=item['bytes'];image_count+=1;hashes.append(item['raw_pixels_sha256'])
    if files!={p.relative_to(output).as_posix() for p in (output/'images').iterdir()}:
        raise ValueError('Extra or missing RAW files')
    counts=dict(Counter(r['split'] for r in rows))
    if (receipt['states']!=len(rows) or receipt['states_by_split']!=counts or receipt['image_count']!=image_count or
            receipt['image_bytes']!=total or receipt['unique_exact_pixel_images']!=len(set(hashes)) or
            image_count>BUDGET['max_images'] or total>BUDGET['image_bytes'] or
            sha(output/'images.jsonl')!=receipt['images_manifest_sha256']): raise ValueError('Incorrect collection counts')
    if sum(p.stat().st_size for p in output.rglob('*') if p.is_file())>=BUDGET['max_bytes']:
        raise ValueError('Artifact budget exceeded')
    if require_seal and read_json(output/'complete.json')!={'status':'RAW_COMPLETE_LABELS_PENDING',
            'manifest_sha256':sha(output/'manifest.json'),'training_eligible':False}:
        raise ValueError('Missing final RAW seal')
    return receipt


def read_json(path): return json.loads(path.read_text())


def prepare(output):
    started=time.monotonic();sources,metadata,identity=source_identity()
    if subprocess.check_output(['git','-C',str(REPO),'status','--porcelain'],text=True).strip():
        raise ValueError('Clean immutable source required')
    commit=subprocess.check_output(['git','-C',str(REPO),'rev-parse','HEAD'],text=True).strip()
    if output.exists(): raise FileExistsError('Never reuse a candidate collection')
    if shutil.disk_usage(output.parent).free<BUDGET['min_free_bytes']: raise RuntimeError('Disk reserve')
    cores=set(BUDGET['cpu_ids'])
    if not cores<=os.sched_getaffinity(0): raise ValueError('Registered CPU cores unavailable')
    os.sched_setaffinity(0,cores);output.mkdir();(output/'images').mkdir()
    plan={'schema':VERSION,'code_commit':commit,'budget':BUDGET,'sources':sources,'source_identity':identity,
          'training_eligible':False,'input_modalities':['raw_rgb'],'labels_created':False}
    atomic_json(output/'source_plan.json',plan)
    total=0;image_count=0;lock=threading.Lock();stopped=threading.Event();completed=[]
    def bounded():
        if stopped.is_set(): raise RuntimeError('Collection stopping after original error')
        if time.monotonic()-started>=BUDGET['seconds']: raise TimeoutError('H80 collection wall budget')
        if shutil.disk_usage(output).free<BUDGET['min_free_bytes']: raise RuntimeError('Disk reserve')
    def extract(source):
        import av
        nonlocal total,image_count
        bounded();meta=metadata[source['episode']]
        rows=[]
        for frame in source['selected_frames']:
            key=f't{source["task"]}_i{source["instance"]}_e{source["episode"]}_f{frame:06d}'
            rows.append({'id':key,**{k:source[k] for k in ('task','instance','episode','split')},'frame':frame,
                         'images':{},'image_receipts':{},'training_eligible':False,'manual_review':'PENDING'})
        for view,camera in CAMERAS.items():
            stem='videos/observation.rgb.'+camera
            path=ROOT/f'{stem}/chunk-{meta[stem+"/chunk_index"]:03d}/file-{meta[stem+"/file_index"]:03d}.mp4'
            stat=path.stat()
            with av.open(str(path)) as container:
                stream=container.streams.video[0];stream.codec_context.thread_count=2
                if abs(float(stream.average_rate)-30)>1e-6: raise ValueError('Original 30Hz source required')
                for row in rows:
                    bounded();timestamp=float(meta[stem+'/from_timestamp'])+row['frame']/30
                    if not math.isfinite(timestamp) or timestamp<0: raise ValueError('Invalid video timestamp')
                    frame,actual=choose_frame(container,stream,timestamp);image=frame.to_image().convert('RGB')
                    if image.size!=((720,720) if view=='head' else (480,480)): raise ValueError('RAW resolution changed')
                    buffer=BytesIO();image.save(buffer,format='PNG');png=buffer.getvalue()
                    rel=f'images/{row["id"]}_{view}.png'
                    with lock:
                        if total+len(png)>BUDGET['image_bytes'] or image_count+1>BUDGET['max_images']:
                            raise RuntimeError('RAW count/byte budget')
                        total+=len(png);image_count+=1
                    with (output/rel).open('xb') as stream_out: stream_out.write(png)
                    row['images'][view]=rel
                    row['image_receipts'][view]={'video':str(path),'video_bytes':stat.st_size,'video_mtime_ns':stat.st_mtime_ns,
                        'source_frame':row['frame'],'source_frame_pts':frame.pts,'source_time_base':str(stream.time_base),
                        'requested_timestamp_s':timestamp,'actual_timestamp_s':actual,'resolution':list(image.size),
                        'bytes':len(png),'png_sha256':hashlib.sha256(png).hexdigest(),
                        'raw_pixels_sha256':hashlib.sha256(image.tobytes()).hexdigest(),'difference_hash_256':difference_hash(image),
                        'raw_resolution_preserved':True}
            after=path.stat()
            if (after.st_size,after.st_mtime_ns)!=(stat.st_size,stat.st_mtime_ns): raise ValueError('Source video changed')
        return rows
    executor=None
    try:
        executor=ThreadPoolExecutor(max_workers=BUDGET['workers'])
        futures={executor.submit(extract,s):s for s in sources}
        with (output/'images.jsonl').open('x',buffering=1) as ledger:
            for future in as_completed(futures):
                bounded();rows=future.result();completed.extend(rows)
                for row in rows: ledger.write(json.dumps(row,separators=(',',':'))+'\n')
                ledger.flush()
                progress={'completed_groups':sum(1 for f in futures if f.done() and not f.cancelled()),
                          'recorded_states':len(completed),'image_files_written':image_count,'image_bytes':total,
                          'seconds':time.monotonic()-started,'training_eligible':False}
                print(json.dumps(progress),flush=True)
        executor.shutdown(wait=True);executor=None;bounded()
        result={'schema':VERSION,'status':'RAW_COMPLETE_LABELS_PENDING','code_commit':commit,
                'source_plan_sha256':sha(output/'source_plan.json'),'images_manifest_sha256':sha(output/'images.jsonl'),
                'states':len(completed),'states_by_split':dict(Counter(r['split'] for r in completed)),
                'image_count':image_count,'image_bytes':total,
                'unique_exact_pixel_images':len({x['raw_pixels_sha256'] for r in completed for x in r['image_receipts'].values()}),
                'wall_seconds':time.monotonic()-started,'training_eligible':False,
                'new_model_calls':0,'new_controls':0,'training_updates':0}
        atomic_json(output/'manifest.json',result)
        validate_output(output,require_seal=False);bounded()
        result['extraction_seconds']=result['wall_seconds']
        result['wall_seconds']=time.monotonic()-started
        result['validation_seconds']=result['wall_seconds']-result['extraction_seconds']
        atomic_json(output/'manifest.json',result)
        atomic_json(output/'complete.json',{'status':'RAW_COMPLETE_LABELS_PENDING',
                    'manifest_sha256':sha(output/'manifest.json'),'training_eligible':False});bounded()
        return result
    except BaseException as error:
        stopped.set()
        # Quiesce all writers before reporting terminal file counts. An external
        # process-group timeout handles a native decoder stuck in this join.
        secondary=[]
        if executor is not None:
            try: executor.shutdown(wait=True,cancel_futures=True)
            except BaseException as cleanup_error: secondary.append(repr(cleanup_error))
            executor=None
        try:
            actual=list((output/'images').iterdir())
            atomic_json(output/'failure.json',{'error':repr(error),'recorded_states':len(completed),
                        'image_count':len(actual),'image_bytes':sum(p.stat().st_size for p in actual),
                        'wall_seconds':time.monotonic()-started,'secondary_errors':secondary,
                        'training_eligible':False,'automatic_retry':False})
        except BaseException as receipt_error:
            print(json.dumps({'primary_error':repr(error),'failure_receipt_error':repr(receipt_error),
                              'secondary_errors':secondary}),file=sys.stderr,flush=True)
        raise
    finally:
        stopped.set()
        if executor is not None: executor.shutdown(wait=True,cancel_futures=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args(); print(json.dumps(prepare(args.output)),flush=True)
