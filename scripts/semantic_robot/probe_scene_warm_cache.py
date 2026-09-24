"""H59: one H57 scene check with frozen private cache copies, same 600s budget."""
import argparse
import faulthandler
import hashlib
import json
from pathlib import Path
import time

import probe_scene_reduced_cameras as reduced
import probe_scene_startup as scene
from shared_cached_runtime import copy_frozen, digest_manifest, CACHE_TREES


SOURCE = Path('/mnt/nvme_tmp/robodojo_sim_runtime_20260924/h57_reduced_cameras_v1')
OLD_RUN = Path('/mnt/nvme_tmp/robodojo_agentic_20260924/h57_reduced_cameras_v1')
OLD_COMMIT = '73c11236c36cf63b789a158050ac89a1b38b483f'
RECEIPTS = {
    'launch.json':'0be0c4848f111cb28e5fe364fd65ae90803ab50e75b7153fde69a29f7ea71497',
    'supervisor.json':'24b4ac54ac5fbae55c3eb936c971585364030dbac379cf890fae74037410d4c2',
    'worker.json':'ee9123998043d18c832acfcaabd86717b0f341f9bbdad820e405b864afc46a1a',
}
CACHE_SHA = '730cba12ec6aec361ce22ef8ccbef88ef31798a69cb0bb1ab0e8653867845f5c'
CACHE_FILES, CACHE_BYTES = 4170, 7715899668


def source_stopped():
    for name,digest in RECEIPTS.items():
        path = OLD_RUN/name
        if path.resolve() != path or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError('Frozen stopped-run receipt mismatch')
    launch = json.loads((OLD_RUN/'launch.json').read_text())
    supervisor = json.loads((OLD_RUN/'supervisor.json').read_text())
    if (launch['source_commit'] != OLD_COMMIT or launch['runtime'] != str(SOURCE) or
            supervisor['status'] != 'failed' or supervisor['exit_code'] != -15 or
            supervisor['supervisor_pid'] != 3503152 or supervisor['worker_pid'] != 3503159):
        raise ValueError('Only the archived H57 runtime may seed this probe')
    if any(Path('/proc',str(pid)).exists() for pid in (3503152,3503159)):
        raise ValueError('Source runtime may still be in use; never copy it')


def prepare_cache(record):
    base = scene.supervisor
    source_stopped()
    started = time.monotonic()
    record.update(phase='copying_frozen_private_cache',cache_seed_verified=False)
    base.write('worker.json',record)
    rows = copy_frozen(SOURCE,base.RUNTIME,expected_sha=CACHE_SHA,
                      expected_count=CACHE_FILES,expected_bytes=CACHE_BYTES)
    source_stopped()
    base.write('cache_seed_manifest.json',{'source':str(SOURCE),'source_commit':OLD_COMMIT,
        'source_receipts':RECEIPTS,'rows':rows,'sha256':digest_manifest(rows)})
    record.update(cache_seed_verified=True,cache_seed={'source':str(SOURCE),
        'source_commit':OLD_COMMIT,'files':len(rows),'bytes':sum(r['bytes'] for r in rows),
        'sha256':digest_manifest(rows),'seconds':time.monotonic()-started,
        'independent_copies':True,'cache_hit_not_established':True},phase='cache_seed_copied')
    base.write('worker.json',record)


def configure_profile():
    reduced.configure_profile()
    scene.WORKER_PREPARE = prepare_cache
    base = scene.supervisor
    base.ENTRYPOINT = Path(__file__).resolve()
    base.OUTPUT = Path('/mnt/nvme_tmp/robodojo_agentic_20260925/h59_warm_cache_v1')
    base.RUNTIME = Path('/mnt/nvme_tmp/robodojo_sim_runtime_20260925/h59_warm_cache_v1')
    base.SUCCESS_FIELDS = {**base.SUCCESS_FIELDS,'cache_seed_verified':True}
    base.BUDGET_DETAILS = {**base.BUDGET_DETAILS,'cold_runtime':False,
        'cache_source_commit':OLD_COMMIT,'cache_manifest_sha256':CACHE_SHA,
        'cache_seed_bytes':CACHE_BYTES,'cache_trees':list(CACHE_TREES),
        'cache_copy_in_600s_activity_budget':True,'traceback_interval_seconds':60}


def worker():
    # Read-only stack snapshots explain a silent initialization without extending it.
    faulthandler.dump_traceback_later(60,repeat=True)
    try:scene.scene_worker()
    finally:faulthandler.cancel_dump_traceback_later()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    for mode in ('launch','supervise','worker'):group.add_argument('--'+mode,action='store_true')
    args = parser.parse_args(); configure_profile()
    if args.launch:
        source_stopped()
        scene.supervisor.launch()
    elif args.supervise:scene.supervisor.supervise()
    else:worker()
