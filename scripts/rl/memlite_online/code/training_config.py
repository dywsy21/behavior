"""Explicit, immutable-at-start settings for this repaired continuation."""
import json
import math
import os
from pathlib import Path

DEFAULTS = dict(actor_lr=1e-7, critic_lr=1e-4, target_kl=0.02,
                max_clip_fraction=0.25, recorder_quota_gib=25.0,
                disk_reserve_gib=100.0, recovery_pre_controls=128,
                recovery_post_controls=384, recovery_max_controls=1024,
                recovery_max_clips_per_episode=8, recovery_stall_controls=256)


def load_training_config():
    path = os.environ.get('RL_TRAINING_CONFIG')
    data = json.loads(Path(path).read_text()) if path else dict(DEFAULTS)
    if set(data) != set(DEFAULTS):
        raise ValueError('Unknown or missing repaired training setting')
    for key, value in data.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError('Invalid training setting: ' + key)
    if data['actor_lr'] > 1e-6 or data['target_kl'] > 0.05 or data['max_clip_fraction'] > 0.5:
        raise ValueError('Outside reviewed trust-region bounds')
    for key in ('recovery_pre_controls','recovery_post_controls','recovery_max_controls',
                'recovery_max_clips_per_episode','recovery_stall_controls'):
        if type(data[key]) is not int: raise ValueError(key + ' must be integer')
    if data['recovery_max_controls'] < data['recovery_pre_controls'] + data['recovery_post_controls']:
        raise ValueError('Clip limit is shorter than configured context')
    return data
