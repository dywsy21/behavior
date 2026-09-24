"""H57: three lower-resolution RGB-D cameras, unchanged H56 resource/physics limits."""
import argparse
from pathlib import Path

import probe_scene_compatible_cameras as compatible
import probe_scene_startup as scene


def configure_profile():
    compatible.configure_profile()
    scene.CAMERA_RESOLUTION_PROFILE = 'shared_v1'
    base = scene.supervisor
    base.ENTRYPOINT = Path(__file__).resolve()
    base.OUTPUT = Path('/mnt/nvme_tmp/robodojo_agentic_20260924/h57_reduced_cameras_v1')
    base.RUNTIME = Path('/mnt/nvme_tmp/robodojo_sim_runtime_20260924/h57_reduced_cameras_v1')
    base.SUCCESS_FIELDS = {**base.SUCCESS_FIELDS, 'input_resolution_changed': True,
                           'camera_intrinsics_verified': True}
    base.BUDGET_DETAILS = {**base.BUDGET_DETAILS,
        'initial_camera_resolution': 'head512/wrists320', 'input_resolution_changed': True,
        'actor_camera_resolution_changed': True,
        'camera_resolution_profile': 'shared_v1', 'configured_camera_pixels': 466944,
        'camera_views_removed': 0}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    for mode in ('launch', 'supervise', 'worker'):
        group.add_argument('--' + mode, action='store_true')
    args = parser.parse_args()
    configure_profile()
    (scene.supervisor.launch if args.launch else scene.supervisor.supervise if args.supervise else scene.scene_worker)()
