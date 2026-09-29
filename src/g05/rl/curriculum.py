"""Automatic curriculum admission; no human-release files or waiting loops."""
from hashlib import sha256
from math import isfinite
import numpy as np
from g05.rl.protocol import validate_worker


def check_curriculum(spec, row, prefix):
    validate_worker(spec, evaluation=False)
    if (type(prefix) is not int or not 0 <= prefix < spec['first_recorded_terminal']
            or row['episode_controls'] != prefix or row['success'] or row['terminal']
            or type(row['episode']) is not int or row['episode'] < 0):
        raise ValueError('Invalid live TRAIN curriculum state')
    raw = row['observation']
    if set(raw) != {'images', 'state', 'task', 'embodiment_type', 'frequency'}:
        raise ValueError('Actor observation fields changed or privileged input leaked')
    if not isinstance(raw['task'], str) or not raw['task'].strip() or float(raw['frequency']) != 30:
        raise ValueError('Missing task or changed control clock')

    def check_state(value):
        if isinstance(value, dict):
            if not value:
                raise ValueError('Empty state mapping')
            for child in value.values(): check_state(child)
        else:
            array = np.asarray(value)
            if not array.size or not np.isfinite(array).all():
                raise ValueError('Nonfinite or missing proprioception')
    check_state(raw['state'])
    if set(raw['images']) != {'head_rgb', 'left_wrist_rgb', 'right_wrist_rgb'}:
        raise ValueError('Three original RGB cameras required')
    cameras = {}
    for name, value in raw['images'].items():
        array = np.asarray(value)
        if (array.ndim != 3 or array.shape[0] != 3 or min(array.shape[1:]) < 32
                or array.dtype != np.uint8 or not np.isfinite(array).all()):
            raise ValueError('Invalid native RGB shape/type')
        cameras[name] = dict(shape=list(array.shape), sha256=sha256(array.tobytes()).hexdigest(),
                             varying=bool(array.max() > array.min()))
    if not cameras['head_rgb']['varying']:
        raise ValueError('Head camera is blank')
    reward = row.get('reward_state')
    if (not isinstance(reward, dict) or reward['goal_success'] or reward['geometry_targets'] < 1
            or not isfinite(reward['potential']) or not 0 <= reward['potential'] <= 1):
        raise ValueError('Missing or terminal dense-reward state')
    clock=row.get('clock',{})
    if (set(clock)!={'simulation_time','physics_index'} or not isfinite(clock['simulation_time'])
            or clock['simulation_time']<0 or type(clock['physics_index']) is not int
            or clock['physics_index']<0):
        raise ValueError('Missing or invalid native observation clock')
    return dict(accepted=True, mode='automatic', worker=spec['worker'], instance=spec['instance'],
                episode=row['episode'], prefix=prefix, cameras=cameras, clock=row['clock'],
                reward_state=reward, human_review_required=False, human_signature_created=False)
