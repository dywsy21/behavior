"""Current-frame image loading from externally SHA-pinned audit receipts.

The caller must first verify the row and receipts' enclosing source manifest.
Bindings are private data checks and are never serialized into an actor prompt.
"""
from __future__ import annotations

import hashlib
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image

from audit_expert_action_capacity import snapshot

VIEWS = ('head', 'left_wrist', 'right_wrist')


def checked_current_pngs(row, receipts, root):
    """Reject future/wrong-view/wrong-size pixels even under legal actor keys."""
    frame = row['frame_index']
    if type(frame) is not int or frame < 0 or row['timestamp_s'] != frame / 30:
        raise ValueError('Current source row clock required')
    if any(type(row[k]) is not int or row[k] < 0 for k in ('task_index', 'task_instance_id', 'episode_index')):
        raise ValueError('Current source row identity required')
    expected_id = f't{row["task_index"]}_i{row["task_instance_id"]}_e{row["episode_index"]}_f{frame:06d}'
    if row['id'] != expected_id: raise ValueError('Current source row ID mismatch')
    refs = row['image_references']
    if len(refs) != 3 or {r['view'] for r in refs} != set(VIEWS):
        raise ValueError('Three unique current video references required')
    references = {r['view']: r for r in refs}
    if len(receipts) != 3 or {r['view'] for r in receipts} != set(VIEWS):
        raise ValueError('Three unique current PNG receipts required')
    root = Path(root).resolve()
    images = {}
    for r in receipts:
        view = r['view']; reference = references[view]
        wanted = reference['episode_start_timestamp_s'] + frame / 30
        times = [wanted, reference['requested_timestamp_s'], r['requested_timestamp_s'], r['actual_timestamp_s']]
        if (not np.isfinite(times).all() or wanted < 0 or r['offset'] != 0 or r['source_frame'] != frame
                or abs(reference['requested_timestamp_s'] - wanted) > 1e-9
                or abs(r['requested_timestamp_s'] - wanted) > 1e-9
                or abs(r['actual_timestamp_s'] - wanted) > 1 / 60 + 1e-6):
            raise ValueError('PNG is not from this current observation time')
        relative = Path(r['path'])
        if r['path'] != f'images/{expected_id}_plus00_{view}.png':
            raise ValueError('Receipt image belongs to another case/view/time')
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Unsafe source-image path')
        path = (root / relative).resolve()
        if not path.is_relative_to(root): raise ValueError('Source image escaped its pinned root')
        data = snapshot(path, {}, r['png_sha256'])
        if len(data) != r['bytes']: raise ValueError('Source PNG byte count changed')
        size = (720, 720) if view == 'head' else (480, 480)
        with Image.open(BytesIO(data)) as original:
            if original.format != 'PNG' or original.size != size or list(size) != r['resolution']:
                raise ValueError('Original camera image format/resolution changed')
            rgb = original.convert('RGB')
            if hashlib.sha256(rgb.tobytes()).hexdigest() != r['pixels_sha256']:
                raise ValueError('Original camera pixels changed')
        images[view] = rgb
    return {view: images[view] for view in VIEWS}
