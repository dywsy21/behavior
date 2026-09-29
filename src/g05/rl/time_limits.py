"""Wall-clock policy, separate from control/update budgets and IPC timeouts."""
from math import inf, isfinite

NO_TRAINING_WALL = 'user_20260929_remove_training_wall'


def active_wall_limit(manifest):
    value = manifest['max_active_wall_seconds']
    if value is None:
        if (manifest.get('entry') != 'method_dense'
                or manifest.get('time_limit_override') != NO_TRAINING_WALL
                or manifest.get('max_training_seconds') is not None):
            raise ValueError('Unlimited wall time requires explicit registered E3 authorization')
    elif not isinstance(value, (int, float)) or not isfinite(value) or value <= 0:
        raise ValueError('Invalid active wall-clock limit')
    return value


def training_deadline(manifest, *, started, training_started):
    wall = active_wall_limit(manifest)
    if wall is None:
        return inf
    limit = manifest['max_training_seconds']
    if not isinstance(limit, (int, float)) or not isfinite(limit) or limit <= 0:
        raise ValueError('Invalid training wall-clock limit')
    return min(training_started + limit,
               started + wall - manifest['final_eval_reserved_seconds'])
