"""H53b: same original scene and resource caps, no unused spectator camera."""
import argparse
from pathlib import Path

import probe_scene_startup as scene


def configure_profile():
    scene.configure_profile()
    scene.DISABLE_VIEWER = True
    base = scene.supervisor
    base.ENTRYPOINT = Path(__file__).resolve()
    base.OUTPUT = Path('/mnt/nvme_tmp/robodojo_agentic_20260924/h53b_no_viewer_v1')
    base.RUNTIME = Path('/mnt/nvme_tmp/robodojo_sim_runtime_20260924/h53b_no_viewer_v1')
    base.SUCCESS_FIELDS = {**base.SUCCESS_FIELDS, 'render_viewer_camera': False, 'viewer_camera_absent': True}
    base.BUDGET_DETAILS = {**base.BUDGET_DETAILS, 'spectator_cameras': 0, 'actor_camera_resolution_changed': False}
    base.DEPENDENCIES = {**base.DEPENDENCIES,
        scene.OG/'omnigibson/macros.py': '5e8ebb1a00fa3e864c5a4984f126442e91b6653f5869977ecddbf4039ea73228'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    for mode in ('launch', 'supervise', 'worker'):
        group.add_argument('--' + mode, action='store_true')
    args = parser.parse_args()
    configure_profile()
    (scene.supervisor.launch if args.launch else scene.supervisor.supervise if args.supervise else scene.scene_worker)()
