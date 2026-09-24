"""H55: original renderer, final RGB-D sizes at camera creation, no live resize."""
import argparse
from pathlib import Path

import probe_scene_without_viewer as no_viewer
import probe_scene_startup as scene


def configure_profile():
    no_viewer.configure_profile()
    scene.PRECONFIGURE_CAMERAS = True
    base = scene.supervisor
    base.ENTRYPOINT = Path(__file__).resolve()
    base.OUTPUT = Path('/mnt/nvme_tmp/robodojo_agentic_20260924/h55_preconfigured_cameras_v1')
    base.RUNTIME = Path('/mnt/nvme_tmp/robodojo_sim_runtime_20260924/h55_preconfigured_cameras_v1')
    base.SUCCESS_FIELDS = {**base.SUCCESS_FIELDS, 'preconfigured_cameras_verified': True}
    base.BUDGET_DETAILS = {**base.BUDGET_DETAILS, 'initial_camera_resolution': 'final head720/wrists480',
                          'live_camera_reconfiguration': False, 'renderer_profile_changed': False}
    base.DEPENDENCIES = {**base.DEPENDENCIES,
        scene.OG/'omnigibson/robots/robot.py': '998826feedcf8c28208681a809cbf4500d3da0cbf1f6d07a7384627e09726271',
        scene.OG/'omnigibson/sensors/vision_sensor.py': 'a445dc5d73ea2d02fa0a87b1cf688abe52b1f213d9263b3e2cfd1fc1e46b9e79',
        scene.OG/'omnigibson/sensors/__init__.py': 'a4be79249a1fb2f7dc2bdf55e3b44755d2fff2fc2ecf28639a9db752f7ec08f9',
        scene.OG/'omnigibson/utils/python_utils.py': '2d6ba4bc43b07a4e266f3b7048de2cd72efcb39f817acadeafb9488577a88869',
        scene.OG/'omnigibson/eval/utils/eval_utils.py': '9b7ea80a54c3f21608a9cf0cacdfbc49d9e9816957034f3b55d5c72a08a0ff8d',
        scene.OG/'omnigibson/eval/wrappers/rgbd_full_res_wrapper.py': 'f1d108f19282ff13f5130102b5bbeb59312084961ac090d49d32b5a41ad8cb86',
        scene.OG/'omnigibson/envs/env_wrapper.py': '52fb65a92cf79f977e24546dd61e717835a44d926dcc1fd7772df53becb681ef'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    for mode in ('launch', 'supervise', 'worker'):
        group.add_argument('--' + mode, action='store_true')
    args = parser.parse_args()
    configure_profile()
    (scene.supervisor.launch if args.launch else scene.supervisor.supervise if args.supervise else scene.scene_worker)()
