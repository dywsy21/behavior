"""Native LeRobot V3 RGB reader, independent of the MEM-Lite skill targets.

The release supplies the protected split, whole-source quarantines, episode
identity and fixed stride phase. Every frame of an admitted episode is indexed;
there is no skill-boundary truncation or automatic quarantine readmission.
Pixels stay native uint8. The champion's own transforms run downstream.
"""
from collections import OrderedDict
from pathlib import Path

import numpy as np


from .release import AcceptedRelease, RELEASE_SHA256
CAMERAS = {
    'base_0_rgb': 'zed_link_camera_0',
    'left_wrist_0_rgb': 'left_realsense_link_camera_0',
    'right_wrist_0_rgb': 'right_realsense_link_camera_0',
}


def action_indices(frame, length, horizon=30):
    if not 0 <= frame < length or horizon != 30:
        raise ValueError('Expected a valid native frame and champion 30-step horizon')
    # Repeat only the episode's last real action, as LeRobot delta queries do.
    # A skill boundary is deliberately not an argument to this function.
    return np.minimum(np.arange(frame, frame + horizon), length - 1)


class NativeV3Dataset:
    def __init__(self, release, split='train', *, manifest_sha256=RELEASE_SHA256):
        self.release = Path(release)
        if split not in ('train', 'eval'):
            raise ValueError('Explicit original train/eval split required')
        accepted = AcceptedRelease(release, manifest_sha256=manifest_sha256)
        self.root = Path(accepted.manifest['raw_root'])
        self.split = split
        self.episodes = []
        self.excluded_episode_counts = dict(accepted.excluded_splits)
        for _, ep in accepted.episodes(split):
            row = ep['row']
            length, phase = int(row['length']), int(ep['phase'])
            if not 0 <= phase < 16 or length <= phase:
                raise ValueError('Invalid original fixed stride phase')
            self.episodes.append(dict(row=row, phase=phase, split=ep['split']))
        counts = [(e['row']['length'] - 1 - e['phase']) // 16 + 1 for e in self.episodes]
        if not counts:
            raise ValueError('Empty selected split')
        self.offsets = np.r_[0, np.cumsum(counts)].astype(np.int64)
        self._arrays = OrderedDict()
        self._decoders = OrderedDict()

    def __len__(self):
        return int(self.offsets[-1])

    def locate(self, index):
        index = int(index)
        if not 0 <= index < len(self):
            raise IndexError(index)
        serial = int(np.searchsorted(self.offsets, index, side='right') - 1)
        ep = self.episodes[serial]
        frame = ep['phase'] + 16 * (index - int(self.offsets[serial]))
        return ep, frame

    def _raw_arrays(self, row):
        import pyarrow.dataset as ds
        eid = int(row['episode_index'])
        if eid not in self._arrays:
            path = self.root / f"data/chunk-{row['data/chunk_index']:03d}/file-{row['data/file_index']:03d}.parquet"
            table = ds.dataset(path, format='parquet').to_table(
                columns=['frame_index', 'timestamp', 'action', 'observation.state'],
                filter=ds.field('episode_index') == eid, use_threads=False).combine_chunks()
            frames = table['frame_index'].to_numpy()
            if not np.array_equal(frames, np.arange(row['length'])):
                raise ValueError(f'Noncontiguous/reordered episode {eid}')
            if not np.allclose(table['timestamp'].to_numpy(), frames / 30, atol=.4/30, rtol=0):
                raise ValueError(f'Incorrect action timestamps in episode {eid}')
            arrays = {}
            for key, dim in (('action', 23), ('observation.state', 61)):
                values = table[key].chunk(0).values.to_numpy(zero_copy_only=False)
                array = values.reshape(-1, dim).astype(np.float32, copy=False)
                if array.shape != (row['length'], dim) or not np.isfinite(array).all():
                    raise ValueError(f'Invalid {key} in episode {eid}')
                arrays[key] = array
            self._arrays[eid] = arrays
            if len(self._arrays) > 2:
                self._arrays.popitem(last=False)
        self._arrays.move_to_end(eid)
        return self._arrays[eid]

    def _image(self, row, frame, sensor):
        from torchcodec.decoders import VideoDecoder
        key = 'videos/observation.rgb.' + sensor
        path = self.root / f"{key}/chunk-{row[key+'/chunk_index']:03d}/file-{row[key+'/file_index']:03d}.mp4"
        name = str(path)
        if name not in self._decoders:
            self._decoders[name] = VideoDecoder(path, device='cpu', seek_mode='approximate',
                                                 num_ffmpeg_threads=1)
            if len(self._decoders) > 3:
                self._decoders.popitem(last=False)
        self._decoders.move_to_end(name)
        decoder = self._decoders[name]
        fps = float(decoder.metadata.average_fps)
        if abs(fps - 30) > 1e-3:
            raise ValueError('Native video FPS changed')
        requested = float(row[key + '/from_timestamp']) + frame / 30
        video_frame = round(requested * fps)
        decoded = decoder.get_frames_at(indices=[video_frame])
        if abs(float(decoded.pts_seconds[0]) - video_frame / fps) >= .4/30:
            raise ValueError('Video seek failed timestamp alignment')
        rgb = decoded.data[0].permute(1, 2, 0).numpy()
        if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[-1] != 3:
            raise ValueError('Expected native uint8 RGB')
        return rgb

    def raw(self, index, *, images=True):
        ep, frame = self.locate(index)
        row = ep['row']
        arrays = self._raw_arrays(row)
        indices = action_indices(frame, int(row['length']))
        result = dict(state61=arrays['observation.state'][frame].copy(),
                      actions23=arrays['action'][indices].copy(),
                      identity=dict(index=int(index), episode=int(row['episode_index']),
                                    task=int(row['task_index']), frame=frame,
                                    length=int(row['length']), phase=int(ep['phase']),
                                    instance=int(row['task_instance_id']),
                                    raw_episode=int(row['raw_episode_id']), split=ep['split'],
                                    real_action_steps=min(30, int(row['length']) - frame)))
        if images:
            result.update({name: self._image(row, frame, sensor) for name, sensor in CAMERAS.items()})
        return result

    def __getitem__(self, index):
        return self.raw(index)


def collate_native(rows):
    # Preserve integer pixels and source identities; no normalizer or target text.
    return dict(identity=[r['identity'] for r in rows],
                **{key: np.stack([r[key] for r in rows])
                   for key in ('state61', 'actions23', *CAMERAS)})


def worker_init(_):
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['JAX_PLATFORMS'] = 'cpu'
    import torch
    torch.set_num_threads(1)
