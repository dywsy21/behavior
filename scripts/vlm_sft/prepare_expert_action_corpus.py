"""H85 lossless action corpus, preserving all 23 native controls.

This is an intermediate corpus, NOT an approved micro-action SFT release.
No direction projection, model-generated actions, or invented outcome labels.
RGB stays in immutable source videos; frame references are part of each shard.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import threading
import time

import numpy as np

from audit_expert_action_capacity import (ROOT, RAW, PLAN, COUNTS, LABELS, RELEASE, REPO,
    HORIZON, snapshot, index_metadata, released_windows)
from common import CAMERAS, skill_text
from prepare_full_annotation import packed, strict_json, forbid_source_output

SCHEMA = 'h85-expert-native23-corpus-v1'
MAX_SECONDS = 1800
MAX_WINDOWS = 350000
MAX_BYTES = 20 * 1024**3
MAX_WORKERS = 4


def bound_identity(raw, counts_bytes):
    identity=strict_json(counts_bytes)['source_identity']
    inherited=raw['source_identity']
    if (hashlib.sha256(counts_bytes).hexdigest()!=inherited['counts_sha256'] or
            any(identity[k]!=inherited[k] for k in ('quarantine_sha256','episode_meta_sha256'))):
        raise ValueError('H80 grouping and expert-source identity disagree')
    return identity


def validate_video_window(offset, frames, start, duration):
    if (not np.isfinite([offset,start,duration]).all() or duration<=0 or offset<0 or
            offset<start-1/60 or offset+(frames-1)/30>start+duration-1/60+1e-6):
        raise ValueError('Referenced RGB time lies outside source video')


def select_corpus_sources(plan, reviews):
    protected = {tuple(pair) for r in reviews for pair in r['protect_groups_in_future_student_releases']}
    selected = [s for s in plan['sources'] if (s['task'], s['instance']) not in protected]
    groups = [(s['task'], s['instance']) for s in selected]
    episodes = [s['episode'] for s in selected]
    if (len(groups) != len(set(groups)) or len(episodes) != len(set(episodes)) or
            any(s['split'] not in ('train','validation','test') or s['task'] not in range(5) for s in selected)):
        raise ValueError('Source grouping or task scope mismatch')
    if Counter(s['split'] for s in selected) != {'train':386,'validation':40,'test':40}:
        raise ValueError('Frozen H80 minus 14 calibration groups required')
    return sorted(selected, key=lambda s:s['episode'])


def arrays_from_source(table, source):
    states = np.asarray(table['observation.state'].to_pylist(), dtype=np.float32)
    actions = np.asarray(table['action'].to_pylist(), dtype=np.float32)
    frames = table['frame_index'].to_numpy()
    timestamps = table['timestamp'].to_numpy()
    n = source['frames']
    if (states.shape != (n,61) or actions.shape != (n,23) or
            not np.array_equal(frames,np.arange(n)) or
            not np.allclose(timestamps,np.arange(n)/30,atol=1e-4,rtol=0) or
            not np.isfinite(states).all() or not np.isfinite(actions).all()):
        raise ValueError('Nonfinite, wrong-shape or misaligned source arrays')
    if np.max(np.abs(actions[:,[0,1,2,14,22]])) > 1.00002:
        raise ValueError('Native normalized base/gripper command out of range')
    return states, actions


def make_rows(source, states, actions, label_rows, quarantines, task):
    """Observation at t pairs with actual commands [t,t+16), never t+history."""
    frames, rejected = released_windows(label_rows, len(states), quarantines)
    low = {r['frame_index']:r for r in label_rows if r['memlite_branch']=='low'}
    rows=[]
    for frame in frames:
        rows.append({'id':f't{source["task"]}_i{source["instance"]}_e{source["episode"]}_f{frame:06d}',
                     'source_split':source['split'],'task_index':source['task'],
                     'task_instance_id':source['instance'],'episode_index':source['episode'],
                     'frame_index':frame, 'timestamp_s':frame/30., 'task':task,
                     'active_instruction':skill_text(low[frame]['active_skills_semantic_json']),
                     'observation_state':states[frame].tolist(),
                     'expert_action':actions[frame:frame+HORIZON].tolist()})
    return rows, rejected


def verify_payload(table, rows, states, actions):
    """Parquet round-trip must preserve float32 targets, not merely shapes."""
    actual = table.to_pylist()
    if len(actual)!=len(rows):raise ValueError('Shard row count changed')
    for got,want in zip(actual,rows):
        f=want['frame_index']
        if (got['id']!=want['id'] or got['frame_index']!=f or got['timestamp_s']!=f/30. or
                got['task']!=want['task'] or got['active_instruction']!=want['active_instruction'] or
                any(got[k]!=want[k] for k in ('source_split','task_index','task_instance_id','episode_index')) or
                got.get('image_references')!=want.get('image_references') or
                not np.array_equal(np.asarray(got['observation_state'],np.float32),states[f]) or
                not np.array_equal(np.asarray(got['expert_action'],np.float32),actions[f:f+HORIZON])):
            raise ValueError('Action/observation payload differs from exact source')


def build_episode(source, meta, label_bytes, quarantines, tasks, output, deadline, stopped):
    import av
    import pyarrow as pa
    import pyarrow.parquet as pq
    if stopped.is_set() or time.monotonic()>deadline:raise TimeoutError('Corpus stopping or wall budget')
    path=ROOT/f'data/chunk-{meta["data/chunk_index"]:03d}/file-{meta["data/file_index"]:03d}.parquet'
    before=path.stat()
    table=pq.read_table(path,filters=[('episode_index','=',source['episode'])],
                        columns=['frame_index','timestamp','action','observation.state'],use_threads=False).sort_by('frame_index')
    states,actions=arrays_from_source(table,source)
    columns=['frame_index','active_skills_semantic_json','low_action_supervision_mask',
             'action_horizon_end','segment_end','memlite_branch','source_kind']
    labels=pq.read_table(pa.BufferReader(label_bytes),filters=[('episode_index','=',source['episode'])],
                         columns=columns,use_threads=False).to_pylist()
    rows,rejected=make_rows(source,states,actions,labels,quarantines,tasks[source['task']])
    videos={}
    for view,camera in CAMERAS.items():
        stem='videos/observation.rgb.'+camera
        video=ROOT/f'{stem}/chunk-{meta[stem+"/chunk_index"]:03d}/file-{meta[stem+"/file_index"]:03d}.mp4'
        stat=video.stat();offset=float(meta[stem+'/from_timestamp'])
        with av.open(str(video)) as container:
            stream=container.streams.video[0]
            expected=(720,720) if view=='head' else (480,480)
            if (abs(float(stream.average_rate)-30)>1e-6 or (stream.width,stream.height)!=expected or
                    not np.isfinite(offset) or offset<0):raise ValueError('Video layout/clock changed')
            if stream.duration is None:raise ValueError('Video duration unavailable')
            start=float((stream.start_time or 0)*stream.time_base)
            duration=float(stream.duration*stream.time_base)
            validate_video_window(offset,source['frames'],start,duration)
            # Source references are not a claim of RGB decode validation.
            videos[view]={'path':str(video),'resolved_path':str(video.resolve()),'bytes':stat.st_size,
                'mtime_ns':stat.st_mtime_ns,'episode_start_timestamp_s':offset,'fps':30,
                'resolution':list(expected),'stream_start_s':start,'stream_duration_s':duration,'decode_validation':'PENDING'}
    for row in rows:
        row['image_references']=[{'view':view,'video_path':video['path'],
            'episode_start_timestamp_s':video['episode_start_timestamp_s'],
            'requested_timestamp_s':video['episode_start_timestamp_s']+row['timestamp_s']} for view,video in videos.items()]
    schema=pa.schema([('id',pa.string()),('source_split',pa.string()),('task_index',pa.int64()),
        ('task_instance_id',pa.int64()),('episode_index',pa.int64()),('frame_index',pa.int64()),('timestamp_s',pa.float64()),
        ('task',pa.string()),('active_instruction',pa.string()),
        ('image_references',pa.list_(pa.struct([('view',pa.string()),('video_path',pa.string()),
             ('episode_start_timestamp_s',pa.float64()),('requested_timestamp_s',pa.float64())]),3)),
        ('observation_state',pa.list_(pa.float32(),61)),
        ('expert_action',pa.list_(pa.list_(pa.float32(),23),HORIZON))])
    shard=output/'shards'/source['split']/f'episode_{source["episode"]:06d}.parquet'
    if stopped.is_set() or time.monotonic()>deadline:raise TimeoutError('Corpus stopping or wall budget')
    pq.write_table(pa.Table.from_pylist(rows,schema=schema),shard,compression='zstd',row_group_size=128)
    verify_payload(pq.read_table(shard,use_threads=False),rows,states,actions)
    after=path.stat()
    if (after.st_size,after.st_mtime_ns)!=(before.st_size,before.st_mtime_ns):raise ValueError('Source changed during extraction')
    counts=Counter(r['active_instruction'].split(';',1)[0] for r in rows)
    return {**{k:source[k] for k in ('task','instance','episode','split','frames')},
        'source_parquet':str(path),'source_bytes':before.st_size,'source_mtime_ns':before.st_mtime_ns,
        'read_arrays_sha256':hashlib.sha256(states.tobytes()+actions.tobytes()).hexdigest(),
        'shard':str(shard.relative_to(output)),'shard_sha256':hashlib.sha256(shard.read_bytes()).hexdigest(),
        'shard_bytes':shard.stat().st_size,'samples':len(rows),'rejected':dict(rejected),'skills':dict(counts),
        'videos':videos,'training_eligible':False}


def main():
    import pyarrow as pa
    import pyarrow.parquet as pq
    pa.set_cpu_count(1);pa.set_io_thread_count(1)
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();started=time.monotonic();deadline=started+1750
    if subprocess.check_output(['git','-C',str(REPO),'status','--porcelain'],text=True).strip():raise ValueError('Clean source required')
    commit=subprocess.check_output(['git','-C',str(REPO),'rev-parse','HEAD'],text=True).strip()
    pins={}
    for name in ('prepare_expert_action_corpus.py','audit_expert_action_capacity.py','common.py','prepare.py','prepare_full_annotation.py'):
        snapshot(REPO/'scripts/vlm_sft'/name,pins)
    plan=strict_json(snapshot(PLAN,pins))
    snapshot(RAW/'manifest.json',pins,plan['metadata_sha256']['manifest.json'])
    raw=strict_json(snapshot(RAW/'source_plan.json',pins,plan['metadata_sha256']['source_plan.json']))
    reviews=[strict_json(snapshot(REPO/name,pins,pin)) for name,pin in plan['review_files'].items()]
    selected=select_corpus_sources(raw,reviews)
    identity=bound_identity(raw,snapshot(COUNTS,pins))
    labels=snapshot(LABELS,pins,identity['labels_sha256'])
    meta=[]
    for path,pin in identity['episode_meta_sha256'].items():
        meta.extend(pq.read_table(pa.BufferReader(snapshot(path,pins,pin)),use_threads=False).to_pylist())
    metadata=index_metadata(meta,selected)
    quarantine=pq.read_table(pa.BufferReader(snapshot(RELEASE/'quarantine_ranges.parquet',pins,identity['quarantine_sha256'])),use_threads=False).to_pylist()
    invalid={}
    for row in quarantine:invalid.setdefault(row['episode_index'],[]).append((row['frame_start'],row['frame_end']))
    task_bytes=snapshot(ROOT/'meta/tasks.jsonl',pins,plan['metadata_sha256']['tasks.jsonl'])
    tasks={r['task_index']:r['task'] for r in map(strict_json,task_bytes.splitlines())}
    forbid_source_output(args.output,ROOT,RAW,LABELS.parent,REPO)
    if args.output.exists():raise FileExistsError('Never overwrite an existing corpus')
    if shutil.disk_usage(args.output.parent).free<80*1024**3:raise ValueError('Disk reserve below 80GiB')
    cores=set(range(48,56))
    if not cores<=os.sched_getaffinity(0):raise ValueError('Registered CPU cores unavailable')
    os.sched_setaffinity(0,cores)
    args.output.mkdir();(args.output/'shards').mkdir()
    for split in ('train','validation','test'):(args.output/'shards'/split).mkdir()
    launch={'schema':SCHEMA,'code_commit':commit,'utc':datetime.now(timezone.utc).isoformat(),
        'pid':os.getpid(),'input_sha256':pins,'selected_sources':selected,'horizon':HORIZON,'stride':HORIZON,
        'fps':30,'max_seconds':MAX_SECONDS,'max_windows':MAX_WINDOWS,'max_bytes':MAX_BYTES,
        'workers':MAX_WORKERS,'training_eligible':False,'new_model_calls':0,'training_updates':0,'controls':0}
    (args.output/'launch.json').write_bytes(packed(launch)+b'\n')
    shards=[];total_samples=0;total_bytes=0;stopped=threading.Event()
    try:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            futures=[pool.submit(build_episode,s,metadata[s['episode']],labels,invalid.get(s['episode'],[]),tasks,args.output,deadline,stopped) for s in selected]
            try:
                for future in as_completed(futures):
                    row=future.result();shards.append(row);total_samples+=row['samples'];total_bytes+=row['shard_bytes']
                    if total_samples>MAX_WINDOWS or total_bytes>MAX_BYTES or time.monotonic()>deadline:raise RuntimeError('Corpus budget')
                    print(json.dumps({'episodes':len(shards),'samples':total_samples,'bytes':total_bytes,'seconds':time.monotonic()-started}),flush=True)
            except BaseException:
                stopped.set()
                for future in futures:future.cancel()
                raise
        for path,pin in pins.items():snapshot(path,{},pin)
        expected={(s['task'],s['instance'],s['episode'],s['split'],s['frames']) for s in selected}
        actual={(s['task'],s['instance'],s['episode'],s['split'],s['frames']) for s in shards}
        if len(shards)!=466 or expected!=actual:raise ValueError('Incomplete source cohort')
        split_counts=Counter();task_counts=Counter()
        for row in shards:
            stat=Path(row['source_parquet']).stat()
            if (stat.st_size,stat.st_mtime_ns)!=(row['source_bytes'],row['source_mtime_ns']):raise ValueError('Source changed before final seal')
            for video in row['videos'].values():
                stat=Path(video['path']).stat()
                if (stat.st_size,stat.st_mtime_ns)!=(video['bytes'],video['mtime_ns']):raise ValueError('Source video changed')
            split_counts[row['split']]+=row['samples'];task_counts[f'{row["split"]}/task{row["task"]}']+=row['samples']
        elapsed=time.monotonic()-started
        if elapsed>MAX_SECONDS:raise TimeoutError('Final corpus wall budget')
        manifest={**launch,'status':'INTERMEDIATE_NATIVE_ACTION_CORPUS_NOT_SFT_RELEASE','shards':sorted(shards,key=lambda s:s['episode']),
            'samples':total_samples,'shard_bytes':total_bytes,'samples_by_split':dict(split_counts),'samples_by_task':dict(task_counts),
            'wall_seconds':elapsed,'source_action_roundtrip':'ALL_FLOAT32_TARGETS_AND_CURRENT_STATES_EXACT',
            'rgb_decode_validation':'PENDING','parent_visual_review':'PENDING','vlm_output_protocol':'NOT_FROZEN',
            'three_hour_training_capacity':'NOT_YET_MEASURED','outcome_labels_created':False}
        (args.output/'manifest.json').write_bytes(packed(manifest)+b'\n')
    except BaseException as exc:
        (args.output/'failure.json').write_bytes(packed({'error':repr(exc),'completed_episodes':len(shards),
            'samples':total_samples,'wall_seconds':time.monotonic()-started})+b'\n')
        raise


if __name__=='__main__':main()
