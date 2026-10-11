"""CPU-only real RGB/proprio/action probe; not a new training corpus.

One preselected TRAIN episode per task. Preserve native-resolution RGB and raw
61/23-D arrays; normalization/resize/tokenization happen in the champion env.
Unlike the MEM-Lite skill dataset, action targets may cross skill boundaries.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

from behavior_champion.release import AcceptedRelease


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--release', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--task-count', type=int, choices=(50, 100), default=50)
    args = ap.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('Explicit CPU-only environment required')
    import numpy as np
    import pyarrow.dataset as ds
    import torch
    from g05.data.lerobot.datasets.video_utils import decode_video_frames_torchcodec
    torch.set_num_threads(1)
    accepted = AcceptedRelease(args.release)
    manifest = accepted.manifest
    root = Path(manifest['raw_root'])
    selected = {}
    for _, episode in accepted.episodes('train'):
        task = int(episode['row']['task_index'])
        if episode['split'] == 'train' and task < args.task_count and task not in selected:
            selected[task] = episode
    if set(selected) != set(range(args.task_count)):
        raise ValueError('Missing TRAIN task')
    args.output.mkdir(parents=True)
    images = {'base_0_rgb': [], 'left_wrist_0_rgb': [], 'right_wrist_0_rgb': []}
    cameras = dict(base_0_rgb='zed_link_camera_0', left_wrist_0_rgb='left_realsense_link_camera_0',
                   right_wrist_0_rgb='right_realsense_link_camera_0')
    states, actions, identities = [], [], []
    started = time.monotonic()
    for task, ep in sorted(selected.items()):
        row = ep['row']
        frame = int(row['length'])//3
        frame -= (frame - ep['phase']) % 16
        if frame < 0 or frame + 30 > row['length']:
            raise ValueError('Probe requires full real 30-step horizon')
        data = root/f"data/chunk-{row['data/chunk_index']:03d}/file-{row['data/file_index']:03d}.parquet"
        table = ds.dataset(data, format='parquet').to_table(
            columns=['frame_index','timestamp','action','observation.state'],
            filter=(ds.field('episode_index') == row['episode_index']) &
                   (ds.field('frame_index') >= frame) & (ds.field('frame_index') < frame+30),
            use_threads=False).to_pylist()
        if [x['frame_index'] for x in table] != list(range(frame, frame+30)):
            raise ValueError('Missing/reordered frames')
        if not np.allclose([x['timestamp'] for x in table], np.arange(frame, frame+30)/30, atol=.4/30, rtol=0):
            raise ValueError('Timestamp mismatch')
        action = np.asarray([x['action'] for x in table], dtype=np.float32)
        state = np.asarray(table[0]['observation.state'], dtype=np.float32)
        if action.shape != (30, 23) or state.shape != (61,) or not np.isfinite(action).all() or not np.isfinite(state).all():
            raise ValueError('Native action/state contract mismatch')
        for model_key, sensor in cameras.items():
            key = 'videos/observation.rgb.'+sensor
            video = root/f"{key}/chunk-{row[key+'/chunk_index']:03d}/file-{row[key+'/file_index']:03d}.mp4"
            timestamp = row[key+'/from_timestamp'] + frame/30
            decoded = decode_video_frames_torchcodec(video, [timestamp], tolerance_s=.4/30, device='cpu')
            if decoded.ndim != 4 or decoded.shape[:2] != (1,3) or not torch.isfinite(decoded).all():
                raise ValueError('Invalid raw RGB')
            image = decoded[0].mul(255).round().to(torch.uint8).permute(1,2,0).numpy()
            images[model_key].append(image)
        states.append(state); actions.append(action)
        identities.append(dict(task=task, episode=row['episode_index'], raw_episode=row['raw_episode_id'],
                               instance=row['task_instance_id'], frame=frame, length=row['length'], phase=ep['phase']))
        print(json.dumps(dict(completed=len(identities), task=task, frame=frame)), flush=True)
    arrays = dict(state61=np.stack(states), actions23=np.stack(actions),
                  task_id=np.array(list(sorted(selected)), dtype=np.int32),
                  frame=np.array([x['frame'] for x in identities], dtype=np.int32),
                  length=np.array([x['length'] for x in identities], dtype=np.int32))
    arrays.update({key: np.stack(value) for key, value in images.items()})
    np.savez(args.output/'native.npz', **arrays)
    with (args.output/'native.npz').open('rb') as stream:
        digest = hashlib.sha256()
        while block := stream.read(8*1024*1024):
            digest.update(block)
    result = dict(status='real_train_probe_prepared_not_training_or_effect_evaluation',
                  selection='first admitted TRAIN episode per task, one-third episode frame on original stride phase',
                  preserved_source_quarantines=dict(accepted.excluded_splits),
                  identities=identities, seconds=time.monotonic()-started, native_npz_sha256=digest.hexdigest(),
                  source_manifest_sha256=hashlib.sha256((args.release/'manifest.json').read_bytes()).hexdigest(),
                  image_shapes={k:list(v[0].shape) for k,v in images.items()}, action_horizon=30,
                  skill_boundary_clipping=False, pixels_resized=False,
                  warning='A cached mixed-task batch tests model compute, NOT end-to-end shuffled-loader throughput or policy success.')
    (args.output/'result.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('status','seconds','native_npz_sha256')}))


if __name__ == '__main__':
    main()
