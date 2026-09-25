"""H56: H55 early RGB-D cameras with the bounded PathTracing/OptiX profile."""
import argparse
from pathlib import Path

import probe_scene_preconfigured_cameras as cameras
import probe_scene_startup as scene
import shared_pathtracing as pathtracing


def configure_profile():
    cameras.configure_profile()
    scene.PATH_TRACING = True
    scene.CAMERA_PATH_TRACING_ALLOWED = True
    base = scene.supervisor
    base.ENTRYPOINT = Path(__file__).resolve()
    base.OUTPUT = Path('/mnt/nvme_tmp/robodojo_agentic_20260924/h56_compatible_cameras_v1')
    base.RUNTIME = Path('/mnt/nvme_tmp/robodojo_sim_runtime_20260924/h56_compatible_cameras_v1')
    base.PROFILE_SETTINGS = pathtracing.SETTINGS.copy()
    base.PROFILE_APP_CONFIG = pathtracing.APP_CONFIG.copy()
    base.RUNTIME_SETTINGS = {**base.RUNTIME_SETTINGS, **base.PROFILE_SETTINGS}
    base.SUCCESS_FIELDS = {**base.SUCCESS_FIELDS, 'pathtracing_profile_verified': True}
    base.BUDGET_DETAILS = {**base.BUDGET_DETAILS, 'renderer_profile_changed': True,
                          'renderer': 'PathTracing', 'image_distribution_changed': True,
                          'spp': 4, 'total_spp': 16}
    base.DEPENDENCIES = {**base.DEPENDENCIES, **pathtracing.INSTALLED_DEPENDENCIES,
        scene.OG/'omnigibson/__init__.py': 'c7867c236051fd8994c75973284b5e88b2637c5b8ab1fece9c6e29bdbd9ca03f',
        scene.OG/'omnigibson/envs/env_base.py': 'ab3e0cefd46a583a8fa9e8ccd42e5cc29590f0b9efddfb0a6f34dab74e4f6c7c'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    for mode in ('launch', 'supervise', 'worker'):
        group.add_argument('--' + mode, action='store_true')
    args = parser.parse_args()
    configure_profile()
    (scene.supervisor.launch if args.launch else scene.supervisor.supervise if args.supervise else scene.scene_worker)()
