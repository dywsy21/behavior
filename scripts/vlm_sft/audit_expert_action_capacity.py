"""H85 bounded TRAIN-only action capacity audit; not a training-data release.

Measures the legacy single-motion projection and amplitude compatibility against
same-state expert windows. Does not fabricate executed native macros or success.
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
import subprocess
import time

import numpy as np
from common import (GRIPS, JOINTS, POSITIONS, TOKENS,
                    classify_window, skill_text)
from prepare import ROOT, LABELS, RELEASE, REPO
from prepare_full_annotation import strict_json, checked_bytes, packed, forbid_source_output

SCHEMA = 'h85-expert-action-capacity-v1'
RAW = Path('/mnt/nvme_tmp/robodojo_vlm_visual_20260925/h80_expanded_raw_v1/raw')
PLAN = REPO / 'configs/vlm_sft/h84_full_annotation_plan_v1.json'
COUNTS = REPO / 'configs/vlm_sft/h09r_train_feasibility_counts.json'
HORIZON = 16


def snapshot(path, pins, expected=None):
    """Parse these exact verified bytes; never verify then reopen to parse."""
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError(f'Original regular file required: {path}')
    content = path.read_bytes()
    digest = hashlib.sha256(content).hexdigest()
    if expected is not None and digest != expected:
        raise ValueError(f'Source SHA mismatch: {path}')
    if str(path) in pins and pins[str(path)] != digest:
        raise ValueError(f'Source changed: {path}')
    pins[str(path)] = digest
    return content


def index_metadata(rows, sources):
    metadata = {}
    for row in rows:
        episode = row['episode_index']
        if episode in metadata:
            raise ValueError(f'Duplicate episode metadata: {episode}')
        metadata[episode] = row
    for source in sources:
        row = metadata.get(source['episode'])
        if row is None or any(row[key] != source[value] for key, value in (
                ('episode_index', 'episode'), ('task_index', 'task'),
                ('task_instance_id', 'instance'), ('length', 'frames'))):
            raise ValueError(f'Selected source metadata identity mismatch: {source}')
    return metadata


def verify_completion(pins, rows, start, selected):
    """An altered input or an over-budget run cannot acquire a COMPLETE seal."""
    for path, digest in pins.items():
        checked_bytes(path, digest)
    for row in rows:
        stat = Path(row['source_parquet']).stat()
        if (stat.st_size, stat.st_mtime_ns) != (row['source_bytes'], row['source_mtime_ns']):
            raise ValueError('Source parquet changed before sealing')
    elapsed = time.monotonic() - start
    if elapsed > 900:
        raise TimeoutError('H85 final wall budget')
    keys = ('task', 'instance', 'episode', 'frames')
    expected = {tuple(s[k] for k in keys) for s in selected}
    actual = {tuple(r[k] for k in keys) for r in rows}
    if (len(rows) != 20 or len(selected) != 20 or len(expected) != 20 or actual != expected or
            any(r.get('training_eligible') is not False for r in rows)):
        raise ValueError('Incomplete or mismatched source cohort')
    return elapsed


def selected_sources(plan, reviews):
    excluded = {tuple(pair) for review in reviews
                for pair in review['protect_groups_in_future_student_releases']}
    available = [s for s in plan['sources'] if s['split'] == 'train' and (s['task'], s['instance']) not in excluded]
    selected = []
    for task in range(5):
        pool = sorted((s for s in available if s['task'] == task),
                      key=lambda s: hashlib.sha256(f'h85-action-audit:41:{task}:{s["instance"]}'.encode()).hexdigest())
        if len(pool) < 4:
            raise ValueError('Not enough eligible TRAIN sources')
        selected.extend(pool[:4])
    if len({(s['task'], s['instance']) for s in selected}) != 20:
        raise ValueError('Repeated source group')
    return selected


def released_windows(rows, length, invalid):
    """Explicit low-branch rows only; never last-wins a duplicate annotation."""
    low = {}
    for row in rows:
        if row['source_kind'] != 'original_demo':
            raise ValueError('Non-original action annotation')
        if row['memlite_branch'] != 'low':
            continue
        frame = row['frame_index']
        if frame in low:
            raise ValueError('Ambiguous duplicate low action annotation')
        low[frame] = row
    selected, rejected = [], Counter()
    for frame in range(0, length - HORIZON, HORIZON):
        row = low.get(frame)
        if (row is None or not row['low_action_supervision_mask'] or
                min(row['action_horizon_end'], row['segment_end']) <= frame + HORIZON or
                any(lo <= frame + HORIZON and hi >= frame for lo, hi in invalid)):
            rejected['not_released_same_skill'] += 1
            continue
        try:
            skill_text(row['active_skills_semantic_json'])
        except ValueError:
            rejected['unbound_instruction'] += 1
            continue
        selected.append(frame)
    return selected, rejected


def endpoint_compatible(token, evidence):
    """Necessary diagnostic only, NOT evidence of native-servo equivalence."""
    if token is None or token.startswith(('BASE_', 'TORSO_')) or token == 'HOLD':
        return None
    part, move = token.split('_', 1)
    if move in ('OPEN', 'CLOSE'):
        return None  # command label is not a calibrated native terminal opening
    arms = ('left', 'right') if part == 'BOTH' else (part.lower(),)
    vectors = {'FORWARD': (1,0,0), 'BACK': (-1,0,0), 'LEFT': (0,1,0),
               'RIGHT': (0,-1,0), 'UP': (0,0,1), 'DOWN': (0,0,-1)}
    if move in vectors:
        required = np.array(vectors[move]) * .01
        return all(np.linalg.norm(np.asarray(evidence['delta_eef_base_m'][arm])-required) <= .003 for arm in arms)
    axis, sign = move.split('_')
    required = np.eye(3)[('ROLL','PITCH','YAW').index(axis)] * np.deg2rad(3) * (1 if sign == 'PLUS' else -1)
    return all(np.linalg.norm(np.asarray(evidence['delta_rotation_base_rad'][arm])-required) <= np.deg2rad(1) for arm in arms)


def scan_episode(source, meta, quarantines, deadline, labels_snapshot):
    import pyarrow as pa
    import pyarrow.parquet as pq
    if time.monotonic() > deadline:
        raise TimeoutError('H85 CPU audit budget')
    episode = source['episode']
    path = ROOT / f'data/chunk-{meta["data/chunk_index"]:03d}/file-{meta["data/file_index"]:03d}.parquet'
    before = path.stat()
    data = pq.read_table(path, filters=[('episode_index','=',episode)],
                         columns=['frame_index','timestamp','action','observation.state'], use_threads=False).sort_by('frame_index')
    states = np.asarray(data['observation.state'].to_pylist(), dtype=np.float32)
    actions = np.asarray(data['action'].to_pylist(), dtype=np.float32)
    timestamps = data['timestamp'].to_numpy()
    if (states.shape != (source['frames'],61) or actions.shape != (len(states),23) or
            not np.array_equal(data['frame_index'].to_numpy(),np.arange(len(states))) or
            not np.allclose(timestamps,np.arange(len(states))/30,atol=1e-4,rtol=0)):
        raise ValueError('Source dimensions, frames or 30Hz timestamps changed')
    columns = ['frame_index','active_skills_semantic_json','low_action_supervision_mask',
               'action_horizon_end','segment_end','memlite_branch','source_kind']
    labels = pq.read_table(pa.BufferReader(labels_snapshot), filters=[('episode_index','=',episode)],
                           columns=columns, use_threads=False).to_pylist()
    frames, rejects = released_windows(labels, len(states), quarantines)
    counts, tokens, activity = Counter(), Counter(), Counter()
    examples = {}
    for frame in frames:
        if time.monotonic() > deadline:
            raise TimeoutError('H85 CPU audit budget')
        window = states[frame:frame+HORIZON+1]
        token, evidence = classify_window(window, actions[frame:frame+HORIZON], actions[frame-1] if frame else None)
        counts['released_same_state_windows'] += 1
        moving = [arm for arm, sl in POSITIONS.items() if np.linalg.norm(window[:,sl]-window[0,sl],axis=1).max() >= .006]
        torso = np.ptp(window[:,JOINTS['torso']],axis=0).max() >= .01
        gripper = any(np.ptp(window[:,sl].mean(axis=1)) >= .005 for sl in GRIPS.values())
        base = np.abs(actions[frame:frame+HORIZON,:3]).max() >= .04
        activity[f'arms{len(moving)}_torso{int(torso)}_gripper{int(gripper)}_base{int(base)}'] += 1
        if token not in TOKENS:
            reason = evidence.get('reject','executor_contract_mismatch')
            rejects[reason] += 1
            examples.setdefault('reject/'+reason,frame)
            continue
        tokens[token] += 1
        counts['legacy_direction_candidate'] += 1
        counts['base_candidate' if token.startswith('BASE_') else 'hold_candidate' if token=='HOLD' else 'manipulation_candidate'] += 1
        compatibility = endpoint_compatible(token,evidence)
        if compatibility is not None:
            counts['endpoint_compatible' if compatibility else 'endpoint_scale_mismatch'] += 1
        examples.setdefault('candidate/'+token,frame)
    after = path.stat()
    if (before.st_size,before.st_mtime_ns) != (after.st_size,after.st_mtime_ns):
        raise ValueError('Source parquet changed during read')
    return {'task':source['task'],'instance':source['instance'],'episode':episode,'frames':len(states),
            'source_parquet':str(path),'source_bytes':before.st_size,'source_mtime_ns':before.st_mtime_ns,
            'arrays_sha256':hashlib.sha256(states.tobytes()+actions.tobytes()).hexdigest(),
            'selected_label_rows_sha256':hashlib.sha256(packed(labels)).hexdigest(),
            'counts':dict(counts),'tokens':dict(tokens),'rejects':dict(rejects),'activity':dict(activity),
            'first_example_frames':examples,'training_eligible':False}


def main():
    import pyarrow as pa
    import pyarrow.parquet as pq
    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    start=time.monotonic(); deadline=start+850
    if subprocess.check_output(['git','-C',str(REPO),'status','--porcelain'],text=True).strip():
        raise ValueError('Clean immutable audit source required')
    code=subprocess.check_output(['git','-C',str(REPO),'rev-parse','HEAD'],text=True).strip()
    pins={}
    for name in ('audit_expert_action_capacity.py','common.py','prepare.py','prepare_full_annotation.py'):
        snapshot(REPO/'scripts/vlm_sft'/name,pins)
    plan=strict_json(snapshot(PLAN,pins))
    snapshot(RAW/'manifest.json',pins,plan['metadata_sha256']['manifest.json'])
    raw_plan=strict_json(snapshot(RAW/'source_plan.json',pins,plan['metadata_sha256']['source_plan.json']))
    reviews=[strict_json(snapshot(REPO/name,pins,pin)) for name,pin in plan['review_files'].items()]
    selected=selected_sources(raw_plan,reviews)
    identity=strict_json(snapshot(COUNTS,pins))['source_identity']
    labels_snapshot=snapshot(LABELS,pins,identity['labels_sha256'])
    quarantine_path=RELEASE/'quarantine_ranges.parquet'
    quarantine_snapshot=snapshot(quarantine_path,pins,identity['quarantine_sha256'])
    metadata_rows=[]
    for path,pin in identity['episode_meta_sha256'].items():
        content=snapshot(path,pins,pin)
        metadata_rows.extend(pq.read_table(pa.BufferReader(content),use_threads=False).to_pylist())
    metadata=index_metadata(metadata_rows,selected)
    invalid={}
    for row in pq.read_table(pa.BufferReader(quarantine_snapshot),use_threads=False).to_pylist():
        invalid.setdefault(row['episode_index'],[]).append((row['frame_start'],row['frame_end']))
    forbid_source_output(args.output,ROOT,RAW,LABELS.parent)
    if args.output.exists():raise FileExistsError('Preserve existing audit')
    args.output.mkdir(parents=True)
    launch={'schema':SCHEMA,'status':'RUNNING','code_commit':code,'pid':os.getpid(),
            'utc':datetime.now(timezone.utc).isoformat(),'selected_sources':selected,
            'plan_sha256':pins[str(PLAN)],'source_identity':identity,'input_sha256':pins,'workers':4,'max_seconds':900,
            'training_eligible':False,'new_model_calls':0,'training_updates':0,'controls':0}
    (args.output/'launch.json').write_bytes(packed(launch)+b'\n')
    rows=[]
    try:
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures=[pool.submit(scan_episode,s,metadata[s['episode']],invalid.get(s['episode'],[]),deadline,labels_snapshot) for s in selected]
            for future in as_completed(futures):
                row=future.result();rows.append(row)
                print(json.dumps({'episodes':len(rows),'task':row['task'],'episode':row['episode'],'counts':row['counts']}),flush=True)
        elapsed=verify_completion(pins,rows,start,selected)
        result={**launch,'status':'COMPLETE_CAPACITY_AUDIT_NOT_TRAINING_DATA','rows':sorted(rows,key=lambda r:r['episode']),
                'wall_seconds':elapsed,'code_sha256':pins[str(Path(__file__).resolve())],
                'common_sha256':pins[str(REPO/'scripts/vlm_sft/common.py')],
                'scope':'Legacy direction and necessary endpoint scale diagnostics; not actual native-servo execution or success labels'}
        (args.output/'result.json').write_bytes(packed(result)+b'\n')
    except BaseException as exc:
        (args.output/'failure.json').write_bytes(packed({'error':repr(exc),'completed_episodes':len(rows),'wall_seconds':time.monotonic()-start})+b'\n')
        raise


if __name__=='__main__':main()
