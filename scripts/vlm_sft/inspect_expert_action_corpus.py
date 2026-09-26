"""TRAIN-only, deterministic 50-window image/action review; never an actor input.

Exports original current/future RGB and human-only contact sheets. Does not
approve labels, infer outcomes, invoke a model, train, or change the corpus.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import textwrap
import time

import numpy as np

from audit_expert_action_capacity import snapshot
from common import CAMERAS
from prepare import REPO, ROOT
from prepare_full_annotation import packed, strict_json, forbid_source_output
from prepare_visual_review import choose_frame

SCHEMA='h85-action-human-review-v1'
STRATA=('base','torso','left','right','both','gripper','mixed','early','middle','late')


def command_activity(row):
    a=np.asarray(row['expert_action'],float);s=np.asarray(row['observation_state'],float)
    if a.shape!=(16,23) or s.shape!=(61,) or not np.isfinite(a).all() or not np.isfinite(s).all():raise ValueError('Invalid review arrays')
    flags={'base':bool(np.abs(a[:,:3]).max()>.05),
           'torso':bool(np.abs(a[:,3:7]-s[53:57]).max()>.015),
           'left':bool(np.abs(a[:,7:14]-s[3:10]).max()>.03),
           'right':bool(np.abs(a[:,15:22]-s[28:35]).max()>.03),
           # Sampling stratum is a target SWITCH within the window, not
           # persistent CLOSE, physical finger motion, or grasp success.
           'gripper':bool(np.ptp(a[:,[14,22]],axis=0).max()>.2)}
    flags['both']=flags['left'] and flags['right']
    flags['mixed']=sum(flags[k] for k in ('base','torso','left','right','gripper'))>=2
    return flags


def choose_row(rows, requested):
    if not rows:raise ValueError('Empty review source')
    if requested in ('early','middle','late'):
        chosen=rows[round((len(rows)-1)*{'early':0.,'middle':.5,'late':1.}[requested])]
        return chosen,False
    matches=[r for r in rows if command_activity(r)[requested]]
    candidates=matches or rows
    chosen=min(candidates,key=lambda r:hashlib.sha256(('h85-review-row:41:'+r['id']).encode()).hexdigest())
    return chosen,not bool(matches)


def select_shards(manifest):
    selected=[]
    for task in range(5):
        pool=[s for s in manifest['shards'] if s['split']=='train' and s['task']==task and s['samples']>0]
        pool.sort(key=lambda s:hashlib.sha256(f'h85-review-group:41:{task}:{s["instance"]}'.encode()).hexdigest())
        if len(pool)<10:raise ValueError('Not enough independent TRAIN review groups')
        selected.extend((s,stratum) for s,stratum in zip(pool[:10],STRATA))
    if len({(s['task'],s['instance']) for s,_ in selected})!=50:raise ValueError('Duplicate review groups')
    return selected


def guard_output(output, corpus, selected):
    forbidden=[ROOT,corpus,REPO,*[Path(s['source_parquet']).parent for s,_ in selected],
               *[Path(v['path']).parent for s,_ in selected for v in s['videos'].values()]]
    # The subset uses read-only aliases into the official dataset. Protect the
    # actual data root too, not only the individual aliased chunk directory.
    for shard,_ in selected:
        actual=Path(shard['source_parquet']).resolve()
        if actual.parent.parent.name=='data':forbidden.append(actual.parents[2])
    forbid_source_output(output,*forbidden)


def main():
    import av
    import pyarrow as pa
    import pyarrow.parquet as pq
    from PIL import Image,ImageDraw
    pa.set_cpu_count(1);pa.set_io_thread_count(1)
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--corpus',type=Path,required=True);parser.add_argument('--manifest-sha256',required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    started=time.monotonic();pins={}
    if subprocess.check_output(['git','-C',str(REPO),'status','--porcelain'],text=True).strip():raise ValueError('Clean source required')
    commit=subprocess.check_output(['git','-C',str(REPO),'rev-parse','HEAD'],text=True).strip()
    for name in ('inspect_expert_action_corpus.py','audit_expert_action_capacity.py','prepare_visual_review.py','prepare_full_annotation.py','common.py'):
        snapshot(REPO/'scripts/vlm_sft'/name,pins)
    manifest=strict_json(snapshot(args.corpus/'manifest.json',pins,args.manifest_sha256))
    if ((args.corpus/'failure.json').exists() or manifest['status']!='INTERMEDIATE_NATIVE_ACTION_CORPUS_NOT_SFT_RELEASE'
            or manifest['training_eligible'] is not False):raise ValueError('Not an intact intermediate corpus')
    selected=select_shards(manifest)
    guard_output(args.output,args.corpus,selected)
    if args.output.exists():raise FileExistsError('Preserve prior review')
    if shutil.disk_usage(args.output.parent).free<80*1024**3:raise ValueError('Disk reserve')
    os.sched_setaffinity(0,set(range(50,56)))
    args.output.mkdir();(args.output/'images').mkdir();(args.output/'human_only_sheets').mkdir()
    launch={'schema':SCHEMA,'code_commit':commit,'pid':os.getpid(),'manifest_sha256':args.manifest_sha256,
        'max_seconds':900,'max_images':450,'max_bytes':512*1024**2,'model_calls':0,'training_updates':0,'controls':0,
        'training_eligible':False,'human_review':'PENDING'}
    launch['strata_note']='Command-based sampling only; gripper means target changes within 16 ticks, not holding or any persistent close command.'
    (args.output/'launch.json').write_bytes(packed(launch)+b'\n')
    results=[];written=0
    try:
        for shard,stratum in selected:
            if time.monotonic()-started>850:raise TimeoutError('Review extraction wall budget')
            shard_path=args.corpus/shard['shard']
            content=snapshot(shard_path,pins,shard['shard_sha256'])
            rows=pq.read_table(pa.BufferReader(content),use_threads=False).to_pylist()
            row,fallback=choose_row(rows,stratum)
            if (row['source_split']!='train' or row['episode_index']!=shard['episode'] or
                    row['task_instance_id']!=shard['instance'] or row['task_index']!=shard['task']):raise ValueError('Wrong source identity')
            f=row['frame_index'];frames=[f,f+8,f+16]
            source=Path(shard['source_parquet']);stat=source.stat()
            if (stat.st_size,stat.st_mtime_ns)!=(shard['source_bytes'],shard['source_mtime_ns']):raise ValueError('Original arrays changed')
            future=pq.read_table(source,filters=[('episode_index','=',shard['episode']),('frame_index','in',frames)],
                columns=['frame_index','timestamp','observation.state'],use_threads=False).sort_by('frame_index')
            states=np.asarray(future['observation.state'].to_pylist(),np.float32)
            if (states.shape!=(3,61) or future['frame_index'].to_pylist()!=frames or
                    not np.array_equal(states[0],np.asarray(row['observation_state'],np.float32)) or
                    not np.allclose(future['timestamp'].to_numpy(),np.asarray(frames)/30,atol=1e-4,rtol=0)):raise ValueError('Review future/current clock mismatch')
            images={};receipts=[]
            for view,camera in CAMERAS.items():
                video=shard['videos'][view];path=Path(video['path']);stat=path.stat()
                if (stat.st_size,stat.st_mtime_ns)!=(video['bytes'],video['mtime_ns']):raise ValueError('Original RGB changed')
                with av.open(str(path)) as container:
                    stream=container.streams.video[0];stream.codec_context.thread_count=2
                    for offset in (0,8,16):
                        target=video['episode_start_timestamp_s']+(f+offset)/30
                        frame,actual=choose_frame(container,stream,target);rgb=frame.to_image().convert('RGB')
                        if list(rgb.size)!=video['resolution']:raise ValueError('Wrong original RGB resolution')
                        relative=f'images/{row["id"]}_plus{offset:02d}_{view}.png';dest=args.output/relative
                        rgb.save(dest,format='PNG');data=dest.read_bytes();written+=len(data)
                        if written>512*1024**2:raise RuntimeError('Review disk budget')
                        receipts.append({'view':view,'offset':offset,'source_frame':f+offset,'requested_timestamp_s':target,
                            'actual_timestamp_s':actual,'path':relative,'bytes':len(data),'png_sha256':hashlib.sha256(data).hexdigest(),
                            'pixels_sha256':hashlib.sha256(rgb.tobytes()).hexdigest(),'resolution':list(rgb.size)})
                        images[(offset,view)]=rgb
            a=np.asarray(row['expert_action']);activity=command_activity(row)
            dp={arm:(states[-1,sl]-states[0,sl]).round(4).tolist() for arm,sl in [('left',slice(17,20)),('right',slice(42,45))]}
            header=[f'HUMAN REVIEW ONLY | {row["id"]} | requested={stratum} | fallback={fallback}',
                *textwrap.wrap(row['active_instruction'],145),
                f'command activity={json.dumps(activity,separators=(",",":"))}',
                f'expert EEF change base metres={dp}',
                f'base command mean={a[:,:3].mean(0).round(3).tolist()} grips first/last={a[[0,-1]][:,[14,22]].round(3).tolist()}']
            top=max(125,len(header)*16+12);sheet=Image.new('RGB',(960,top+3*342),'white');draw=ImageDraw.Draw(sheet)
            for index,line in enumerate(header):draw.text((5,5+16*index),line,fill='black')
            for iy,offset in enumerate((0,8,16)):
                for ix,view in enumerate(CAMERAS):
                    image=images[(offset,view)].resize((320,320),Image.Resampling.LANCZOS)
                    y=top+iy*342;draw.text((ix*320+4,y+3),f'{view} t+{offset}/30s',fill='black');sheet.paste(image,(ix*320,y+22))
            rel=f'human_only_sheets/{row["id"]}.jpg';dest=args.output/rel;sheet.save(dest,quality=94)
            data=dest.read_bytes();written+=len(data)
            if written>512*1024**2:raise RuntimeError('Review disk budget')
            results.append({'id':row['id'],'task':shard['task'],'instance':shard['instance'],'episode':shard['episode'],
                'frame':f,'split':'train','requested_stratum':stratum,'fallback':fallback,'command_activity':activity,
                'expert_delta_eef_base_m':dp,'row':row,'original_states_for_human_only':states.tolist(),
                'images':receipts,'sheet':rel,'sheet_sha256':hashlib.sha256(data).hexdigest(),'manual_review':'PENDING'})
            print(json.dumps({'review_cases':len(results),'original_images':len(results)*9,'bytes':written}),flush=True)
        for path,pin in pins.items():snapshot(path,{},pin)
        for shard,_ in selected:
            stat=Path(shard['source_parquet']).stat()
            if (stat.st_size,stat.st_mtime_ns)!=(shard['source_bytes'],shard['source_mtime_ns']):raise ValueError('Original arrays changed before seal')
            for video in shard['videos'].values():
                stat=Path(video['path']).stat()
                if (stat.st_size,stat.st_mtime_ns)!=(video['bytes'],video['mtime_ns']):raise ValueError('Original RGB changed before seal')
        for case in results:
            snapshot(args.output/case['sheet'],{},case['sheet_sha256'])
            for receipt in case['images']:snapshot(args.output/receipt['path'],{},receipt['png_sha256'])
        elapsed=time.monotonic()-started
        if len(results)!=50 or elapsed>900:raise ValueError('Review incomplete or over budget')
        result={**launch,'status':'DECODED_HUMAN_REVIEW_PENDING','wall_seconds':elapsed,'cases':results,
            'original_images':450,'bytes':written,'input_sha256':pins,'outcome_labels_created':False}
        (args.output/'manifest.json').write_bytes(packed(result)+b'\n')
    except BaseException as exc:
        (args.output/'failure.json').write_bytes(packed({'error':repr(exc),'cases':len(results),'bytes':written})+b'\n')
        raise


if __name__=='__main__':main()
