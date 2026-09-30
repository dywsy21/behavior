"""Explicit user-authorized unlimited TRAIN; still fail closed on real faults."""
from itertools import count
import shutil

AUTHORIZATION = 'user_20260930_continue_after_e3_without_training_budget'
CAPS = ('max_controls', 'max_training_controls', 'max_batches',
        'max_new_actor_updates', 'max_actor_updates')
MIN_FREE_BYTES = 100*(1<<30)
CHECKPOINT_RESERVE_BYTES = 8*(1<<30)


def continuous_mode(manifest):
    value = manifest.get('training_limit_override')
    if value is None:
        return False
    if value != AUTHORIZATION:
        raise ValueError('Unknown continuous-training authorization')
    if (manifest.get('entry') != 'method_dense'
            or any(key not in manifest or manifest[key] is not None for key in CAPS)
            or manifest.get('max_training_seconds') is not None
            or manifest.get('max_active_wall_seconds') is not None
            or manifest.get('min_free_disk_bytes') != MIN_FREE_BYTES
            or manifest.get('checkpoint_retention') != 'retain_all_no_automatic_deletion'
            or type(manifest.get('prior_evaluation_controls')) is not int
            or manifest['prior_evaluation_controls'] < 0):
        raise ValueError('Continuous TRAIN must remove every budget and retain accounting/storage guards')
    return True


def batch_indices(manifest, first):
    return count(first) if continuous_mode(manifest) else range(first,manifest['max_batches'])


def check_storage(manifest, directory):
    if continuous_mode(manifest):
        free=shutil.disk_usage(directory).free
        if free < manifest['min_free_disk_bytes']+CHECKPOINT_RESERVE_BYTES:
            raise RuntimeError(f'Continuous TRAIN storage guard: {free} bytes free; no checkpoint deleted')
