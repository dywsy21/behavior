"""Read current source RGB lazily, without storing new image copies or futures.

The caller must SHA-pin the corpus manifest/shard before passing its row/video
metadata. Each reader owns at most three containers and is worker-local.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from prepare_visual_review import choose_frame

VIEWS = ('head', 'left_wrist', 'right_wrist')


def validate_reference(row, shard, view):
    f = row['frame_index']
    if (type(f) is not int or f < 0 or f + 16 >= shard['frames']
            or row['timestamp_s'] != f / 30 or row['source_split'] != shard['split']
            or any(row[k] != shard[s] for k, s in (('task_index', 'task'),
                       ('task_instance_id', 'instance'), ('episode_index', 'episode')))
            or row['id'] != f't{shard["task"]}_i{shard["instance"]}_e{shard["episode"]}_f{f:06d}'):
        raise ValueError('Current row/shard identity or clock mismatch')
    refs = row['image_references']
    if len(refs) != 3 or {r['view'] for r in refs} != set(VIEWS):
        raise ValueError('Exactly three unique camera references required')
    reference = next(r for r in refs if r['view'] == view)
    video = shard['videos'][view]
    target = video['episode_start_timestamp_s'] + f / 30
    if (reference['video_path'] != video['path']
            or reference['episode_start_timestamp_s'] != video['episode_start_timestamp_s']
            or not np.isfinite([target, reference['requested_timestamp_s']]).all()
            or abs(reference['requested_timestamp_s'] - target) > 1e-9
            or video['fps'] != 30 or video['resolution'] != ([720, 720] if view == 'head' else [480, 480])
            or not video['stream_start_s'] - 1 / 60 <= target <= video['stream_start_s'] + video['stream_duration_s']):
        raise ValueError('Current video binding/time/size mismatch')
    return video, target


def video_identity(video):
    path = Path(video['path'])
    stat = path.stat()
    if (str(path.resolve(strict=True)) != video['resolved_path']
            or (stat.st_size, stat.st_mtime_ns) != (video['bytes'], video['mtime_ns'])):
        raise ValueError('Pinned source video file changed')
    return path


class CurrentVideoReader:
    def __init__(self): self.opened = {}

    def close(self):
        for container, _, _, _ in self.opened.values(): container.close()
        self.opened.clear()

    def __enter__(self): return self
    def __exit__(self, *args): self.close()

    def read(self, row, shard):
        import av
        images = {}; receipts = []
        try:
            for view in VIEWS:
                video, target = validate_reference(row, shard, view)
                path = video_identity(video)
                # The manifest pins the resolved path and original metadata.
                # If a new shard names the same path with a different identity,
                # do not silently reuse an old open descriptor.
                identity = (video['resolved_path'], video['bytes'], video['mtime_ns'])
                old = self.opened.get(view)
                if old is not None and (old[2] != str(path) or old[3] != identity):
                    old[0].close(); del self.opened[view]; old = None
                if old is None:
                    container = av.open(str(path))
                    stream = container.streams.video[0]; stream.codec_context.thread_count = 1
                    self.opened[view] = (container, stream, str(path), identity)
                container, stream, _, _ = self.opened[view]
                if abs(float(stream.average_rate) - 30) > 1e-6:
                    raise ValueError('Source video rate changed')
                frame, actual = choose_frame(container, stream, target)
                if not np.isfinite(actual) or abs(actual - target) > 1 / 60 + 1e-6:
                    raise ValueError('Actual decoded frame is not the current observation')
                image = frame.to_image().convert('RGB')
                if list(image.size) != video['resolution']:
                    image.close(); raise ValueError('Decoded camera resolution changed')
                images[view] = image
                video_identity(video)
                receipts.append({'view': view, 'source_frame': row['frame_index'],
                                 'requested_timestamp_s': target, 'actual_timestamp_s': actual})
            return images, receipts
        except BaseException:
            for image in images.values(): image.close()
            raise
