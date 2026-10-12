"""Reserve all planned retained snapshots without deleting any prior artifact."""
import shutil


# The measured 302-leaf checkpoint is 54,489,569,924 bytes. Round up beyond a
# 10% serialization margin. EMA is about 12GiB. Keep 256GiB for other disk use.
FULL_CHECKPOINT_BYTES = 56 << 30
EMA_EXPORT_BYTES = 15 << 30
RESERVE_BYTES = 256 << 30


def storage_plan(*, completed, updates, checkpoint_every):
    if (any(type(value) is not int for value in (completed, updates, checkpoint_every))
            or not 0 <= completed <= updates or checkpoint_every <= 0):
        raise ValueError('Invalid remaining checkpoint schedule')
    periodic = updates // checkpoint_every - completed // checkpoint_every
    final_extra = int(completed < updates and updates % checkpoint_every != 0)
    # One additional arbitrary-step signal/time-limit save, including step0.
    snapshots = periodic + final_extra + 1
    required = snapshots * FULL_CHECKPOINT_BYTES + EMA_EXPORT_BYTES + RESERVE_BYTES
    return dict(remaining_periodic_snapshots=periodic, extra_final_snapshot=final_extra,
                emergency_snapshots=1, planned_new_snapshots=snapshots,
                conservative_full_checkpoint_bytes=FULL_CHECKPOINT_BYTES,
                ema_export_bytes=EMA_EXPORT_BYTES, reserve_bytes=RESERVE_BYTES,
                required_free_bytes=required, deletes_existing_artifacts=False)


def check_storage(directory, *, completed, updates, checkpoint_every):
    result = storage_plan(completed=completed, updates=updates, checkpoint_every=checkpoint_every)
    result['observed_free_bytes'] = shutil.disk_usage(directory).free
    if result['observed_free_bytes'] < result['required_free_bytes']:
        raise OSError('Insufficient free disk for all retained checkpoints, EMA and reserve; '
                      'no files deleted and no optimizer started')
    return result
