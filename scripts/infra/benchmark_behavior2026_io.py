"""Bounded, CPU-only RGB/action I/O probe across all 100 tasks.

800 windows total, no model/labels/training or dataset mutation. These grouped
episode reads are an I/O proxy, not the production shuffled MEM-Lite loader.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
import os
from pathlib import Path
import random
import subprocess
import time

ROOT = Path('/data/workspace/wsy/behavior2026/datasets/2026-challenge-demos/datasets/fduTristin--2026-challenge-demos/snapshots/master')
KEYS = ('observation.rgb.zed_link_camera_0', 'observation.rgb.left_realsense_link_camera_0',
        'observation.rgb.right_realsense_link_camera_0')


def anchors(length, stride=1):
    if not isinstance(length, int) or length < 64:
        raise ValueError('Need at least 64 real episode frames')
    if isinstance(stride, bool) or stride not in (1, 16):
        raise ValueError('Only registered observation strides 1 and 16 are allowed')
    slots = (length - 32) // stride + 1
    if slots < 4:
        raise ValueError('Need four distinct full-horizon observation starts')
    selected = []
    for i, fraction in enumerate((.15, .35, .60, .85)):
        previous = selected[-1] if selected else -1
        selected.append(max(previous + 1, min(int((slots - 1) * fraction), slots - (4 - i))))
    return [index * stride for index in selected]


def schedule(root, stride=1):
    import pyarrow.parquet as pq
    info = json.loads((root / 'meta/info.json').read_text())
    if (info['fps'] != 30 or info['data_path'] != 'data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet'
            or info['video_path'] != 'videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4'):
        raise ValueError('Unexpected metadata clock/path contract')
    groups = {}
    total_frames = total_episodes = candidate_starts = full_horizon_starts = 0
    for path in sorted((root / 'meta/episodes').glob('chunk-*/file-*.parquet')):
        for row in pq.read_table(path).to_pylist():
            groups.setdefault(int(row['task_index']), []).append(row)
            total_episodes += 1
            length = int(row['length'])
            total_frames += length
            candidate_starts += (length + stride - 1) // stride
            full_horizon_starts += max(0, (length - 32) // stride + 1)
    if set(groups) != set(range(100)) or total_episodes != 20000 or total_frames != 210916774:
        raise ValueError('Expected the fixed 100-task/20k metadata corpus')
    rng = random.Random(73)
    selected = []
    for task, group in sorted(groups.items()):
        row = rng.choice(sorted(group, key=lambda row: int(row['episode_index'])))
        selected.append(dict(root=str(root), row=row, frames=anchors(int(row['length']), stride)))
    return selected, dict(tasks=100, episodes=total_episodes, raw_frames=total_frames, seed=73,
                          observation_start_stride=stride, raw_candidate_starts=candidate_starts,
                          full_horizon_candidate_starts=full_horizon_starts,
                          counts_before_holdout_and_skill_filter=True)


def worker_initialize():
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)


def read_episode(job):
    import numpy as np
    import pyarrow.dataset as ds
    import torch
    from g05.data.lerobot.datasets.video_utils import decode_video_frames_torchcodec
    started = time.perf_counter()
    root, row, frames = Path(job['root']), job['row'], job['frames']
    episode, task = int(row['episode_index']), int(row['task_index'])
    data = root / f"data/chunk-{int(row['data/chunk_index']):03d}/file-{int(row['data/file_index']):03d}.parquet"
    ranges = None
    for frame in frames:
        condition = (ds.field('frame_index') >= frame) & (ds.field('frame_index') < frame + 32)
        ranges = condition if ranges is None else ranges | condition
    table = ds.dataset(data, format='parquet').to_table(
        columns=['frame_index', 'action', 'observation.state'],
        filter=(ds.field('episode_index') == episode) & ranges, use_threads=False)
    values = {int(x['frame_index']): x for x in table.to_pylist()}
    for frame in frames:
        actions = np.asarray([values[i]['action'] for i in range(frame, frame+32)], dtype=np.float32)
        state = np.asarray(values[frame]['observation.state'], dtype=np.float32)
        if actions.shape != (32, 23) or state.shape != (61,) or not np.isfinite(actions).all() or not np.isfinite(state).all():
            raise ValueError('Raw action/state reader contract failed')
    resized_checks = []
    for key in KEYS:
        stem = 'videos/' + key
        path = root / f"{stem}/chunk-{int(row[stem+'/chunk_index']):03d}/file-{int(row[stem+'/file_index']):03d}.mp4"
        for frame in frames:
            timestamp = float(row[stem+'/from_timestamp']) + frame / 30.
            # Match the actual LeRobot backend: a new approximate-seek
            # decoder per camera/sample, actual stream FPS and its timestamp
            # assertion. Do not amortize one exact decoder over four samples
            # and mistake that for the existing training reader.
            pixels = decode_video_frames_torchcodec(path, [timestamp], tolerance_s=.4/30, device='cpu')
            if pixels.ndim != 4 or pixels.shape[:2] != (1, 3) or pixels.dtype != torch.float32:
                raise ValueError('Expected float32 RGB BCHW from the real reader')
            resized = torch.nn.functional.interpolate(pixels,
                         size=(256, 256), mode='bilinear', align_corners=False, antialias=True)
            resized_checks.append(float(resized.mean()))
    if torch.cuda.is_initialized():
        raise RuntimeError('The I/O probe must not initialize CUDA')
    return dict(task=task, episode=episode, frames=frames, windows=len(frames),
                seconds=time.perf_counter()-started, rgb_resized_mean=resized_checks,
                raw_action_dim=23, state_dim=61, cuda_initialized=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=32, choices=(8, 16, 32))
    parser.add_argument('--stride', type=int, default=1, choices=(1, 16))
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    if subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True).strip():
        raise RuntimeError('Do not overlap I/O probe with the GPU compute measurements')
    args.output.mkdir(parents=True, exist_ok=False)
    jobs, corpus = schedule(ROOT, args.stride)
    results = []
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context('spawn'),
                             initializer=worker_initialize) as pool:
        for pass_index in range(2):
            started = time.perf_counter()
            rows = list(pool.map(read_episode, jobs, chunksize=1))
            elapsed = time.perf_counter() - started
            if {row['task'] for row in rows} != set(range(100)) or sum(row['windows'] for row in rows) != 400:
                raise RuntimeError('All-task I/O coverage incomplete')
            result = dict(pass_index=pass_index, wall_seconds=elapsed,
                          windows=400, windows_per_second=400/elapsed, rows=rows)
            results.append(result)
            with (args.output / f'pass{pass_index}.json').open('x') as stream:
                json.dump(result, stream, indent=2, allow_nan=False)
            print(json.dumps({key:value for key,value in result.items() if key!='rows'}), flush=True)
    report = dict(status='complete', scope='real per-sample TorchCodec backend; grouped-episode raw Parquet/action/state read plus resize proxy',
                  production_memlite_loader=False, labels_used=False, model_updates=0,
                  first_pass_includes_worker_startup=True, second_pass_is_disk_warm_not_guaranteed_decoder_warm=True,
                  workers=args.workers, windows=800, corpus=corpus,
                  passes=[{key:value for key,value in result.items() if key!='rows'} for result in results])
    with (args.output / 'result.json').open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)


if __name__ == '__main__':
    main()
