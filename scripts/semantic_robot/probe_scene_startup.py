"""H53: one original task0 session, RGB-D capture, no actor/model or training."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import replace
import hashlib
import importlib
import inspect
import os
from pathlib import Path
import signal
import sys
import time
from types import SimpleNamespace

import probe_simulator_startup as supervisor
from shared_og_startup import private_og_startup

OG = Path('/mnt/sdc1/xhz/BEHAVIOR2026/BEHAVIOR-1K/OmniGibson')
OG_SOURCE = OG / 'omnigibson/simulator.py'
ADAPTER = Path('/mnt/sdc1/robodojo/behavior_dev/GalaxeaVLA_memlite_coordination_dev_20260908/sim_runtime/production_native_oracle_low_v1')
FACTORY = ADAPTER / 'native_oracle_low_v1/official_factory.py'
EVAL_ROOT = Path('/mnt/sdc1/robodojo/behavior_eval')
ROBOT_CONFIG = OG / 'omnigibson/eval/r1pro.yaml'
WINDOW = Path('/mnt/sdc1/robodojo/behavior_dev/direct_execution_L6rqZ6_20260910/c1_windows_v2_matched/c1v2-matched-t0-train-e121-f448-grasp/window.json')
ROBOT_SHA = 'a98fa8a81472adfecfb642778d4d7a01e20131be7f70824e0ae095d665cdbb93'
ICON = OG.parent / 'docs/assets/OmniGibson_logo.png'
INSTALLED_ICON = supervisor.ISAAC / 'apps/OmniGibson_logo.png'
ICON_SHA = '415480a53ed07ee3b909b4a3510f0aa44c4f44aa23296713ebc07af5a461ec23'
EXTRA_DEPENDENCIES = {
    OG_SOURCE: 'd800c2832f24962c440c4781ebfdb4ea78c74aac37d2c74ee7894ae430f1e2a9',
    FACTORY: '5492910dac0a75aac9e8ca4e71f5b762aeff4e129c37d6ed4d8c995c312e5f70',
    WINDOW: '93731a793ed550bae225e148d872b97d42e331a709e2fd0cd33bd1ccf76ccbd8',
    ADAPTER / 'native_oracle_low_v1/runtime.py': '61beb65dbe0dc5a6e7bb952a60a9d790c9b2eb51bd546e2ecc53a61fb662c447',
    EVAL_ROOT / 'run_behavior_eval_chunked.py': '8feddedc46ffca3cb63bd80b4c237e9a157a3c8c7929c74b74c5dbc51ec524c5',
    EVAL_ROOT / 'memlite_sim_trace.py': 'd2f9eb7df66c982aac0a15c34a8768a75215370bb72d178b0ce596bc72b0bef5',
    OG / 'omnigibson/eval/evaluator.py': 'eb121fff3214107ec08c9eb0428b7d898ab7972b197b77ea1a873383d844b288',
    ROBOT_CONFIG: ROBOT_SHA,
    ICON: ICON_SHA, INSTALLED_ICON: ICON_SHA,
}
DISABLE_VIEWER = False  # H53 default; the separate H53b CLI opts in.
PATH_TRACING = False  # H54 is a separate, image-distribution-changing profile.
PRECONFIGURE_CAMERAS = False  # H55 moves final config before sensor creation.
CAMERA_PATH_TRACING_ALLOWED = False  # Only the separately registered H56 opts in.
CAMERA_RESOLUTION_PROFILE = 'full_v1'  # H57 alone selects shared_v1 before creation.
WORKER_PREPARE = None  # H59 only: copied stopped-run caches inside the worker budget.


def configure_profile():
    """Called only by this immutable CLI, never as an import side effect."""
    global DISABLE_VIEWER, PATH_TRACING, PRECONFIGURE_CAMERAS, CAMERA_PATH_TRACING_ALLOWED, CAMERA_RESOLUTION_PROFILE, WORKER_PREPARE
    DISABLE_VIEWER = False
    PATH_TRACING = False
    PRECONFIGURE_CAMERAS = False
    CAMERA_PATH_TRACING_ALLOWED = False
    CAMERA_RESOLUTION_PROFILE = 'full_v1'
    WORKER_PREPARE = None
    supervisor.PROFILE_SETTINGS = {}
    supervisor.PROFILE_APP_CONFIG = {}
    supervisor.RUNTIME_SETTINGS = {**supervisor.SETTINGS, **supervisor.GPU_SETTINGS}
    supervisor.ENTRYPOINT = Path(__file__).resolve()
    supervisor.OUTPUT = Path('/mnt/nvme_tmp/robodojo_agentic_20260924/h53_original_scene_v1')
    supervisor.RUNTIME = Path('/mnt/nvme_tmp/robodojo_sim_runtime_20260924/h53_original_scene_v1')
    supervisor.WALL_SECONDS = 600
    supervisor.SUCCESS_FIELDS = {'phase': 'scene_observation_complete', 'rgbd_captures': 1,
        'session_constructions': 1, 'explicit_final_resets': 1,
        'official_api_resets': 2, 'load_frozen_instance_calls': 1,
        'onboard_render_calls': 4, 'depth_views': 3,
        'startup_launch_calls': 1, 'app_constructions': 1, 'verified_copy_noops': 2,
        'shared_install_writes': 0, 'robot_controls': 0, 'expert_prefix_controls': 0,
        'old_policy_prefix_controls': 0, 'model_calls': 0, 'training_steps': 0}
    supervisor.BUDGET_DETAILS = {'session_constructions': 1, 'official_api_resets': 2,
        'load_frozen_instance_calls': 1, 'observation_render_barrier_calls': 4,
        'rgbd_captures': 1, 'robot_controls': 0, 'expert_prefix_controls': 0,
        'old_policy_prefix_controls': 0, 'model_calls': 0, 'training_steps': 0,
        'initialization_internal_steps': 'unchanged official reset/settling; wall bounded'}
    supervisor.DEPENDENCIES = {**supervisor.DEPENDENCIES, **EXTRA_DEPENDENCIES}
    supervisor.FIXED_ENV = {**supervisor.FIXED_ENV, 'OMNIGIBSON_GPU_ID': '3',
        'OMNIGIBSON_HEADLESS': '1', 'OMNIGIBSON_NO_OMNI_LOGS': '0', 'BEHAVIOR_ACTION_STEPS': '1',
        'MEMLITE_SIM_TRACE_PATH': ''}


def validate_window(window):
    if (window.task_name != 'turning_on_radio' or window.official_mode != 'train' or
            type(window.instance_id) is not int or window.instance_id != 138 or
            type(window.seed) is not int or window.seed != 0 or window.robot_config_sha256 != ROBOT_SHA):
        raise ValueError('Exact original task0 TRAIN138/seed0/robot required')
    if hashlib.sha256(Path(window.robot_config_path).read_bytes()).hexdigest() != ROBOT_SHA:
        raise ValueError('Installed robot config changed')


def configure_viewer_before_launch(gm, og, record):
    if not DISABLE_VIEWER:
        return
    if og.app is not None or og.sim is not None:
        raise ValueError('Viewer selection must precede any simulator/application')
    # Official process-local macro; never edit installed macros or resize/remove
    # a live sensor (which can invalidate the initialized physics handles).
    with gm.unlocked():
        gm.RENDER_VIEWER_CAMERA = False
    if gm.RENDER_VIEWER_CAMERA is not False:
        raise ValueError('Viewer disable setting was not applied')
    record['render_viewer_camera'] = False
    supervisor.write('worker.json', record)


def validate_viewer_after_scene(gm, sim, record):
    if DISABLE_VIEWER:
        if gm.RENDER_VIEWER_CAMERA is not False or sim.viewer_camera is not None:
            raise ValueError('Unused viewer camera was still created')
        record['viewer_camera_absent'] = True


def validate_capture(before, after, images, depths, sensors, resolution_profile='full_v1'):
    import numpy as np
    from shared_camera_config import resolutions
    sizes = resolutions(resolution_profile)
    if (before.shape != after.shape or before.size == 0 or not np.isfinite(before).all() or
            not np.isfinite(after).all() or np.max(np.abs(after-before)) > 1e-5):
        raise ValueError('Robot changed during render-only RGB-D capture')
    views = {'head', 'left_wrist', 'right_wrist'}
    if set(sensors) != views or set(depths) != views or set(images) != {v+'_rgb' for v in views}:
        raise ValueError('Exact three onboard cameras required')
    for view in sensors:
        rgb = images[view + '_rgb']
        depth, receipt = depths[view], sensors[view]
        expected = (3, sizes[view], sizes[view])
        if rgb.shape != expected or rgb.dtype != np.uint8:
            raise ValueError('Registered RGB resolution/uint8 contract changed')
        if depth.shape != expected[1:] or depth.dtype != np.float32:
            raise ValueError('Registered floating depth resolution changed')
        valid_fraction = float((np.isfinite(depth) & (depth>.025) & (depth<4.)).mean())
        # Preserve invalid pixels as captured. They are not synthesized surfaces;
        # compare the actual finite-depth coverage with the claimed coverage.
        if (type(receipt.get('valid_fraction')) is not float or receipt['valid_fraction'] != valid_fraction or
                valid_fraction == 0. or (view == 'head' and valid_fraction < .1)):
            raise ValueError('Depth coverage receipt differs from actual usable pixels')
        exact = {'shape': list(depth.shape), 'modalities': ['rgb', 'depth_linear'],
            'depth_units': 'metres', 'depth_convention': 'distance_to_image_plane',
            'same_sensor_current_render': True, 'render_barrier_updates': 4,
            'control_steps_in_capture': 0, 'snapshot_id': 1,
            'rgb_sha256': hashlib.sha256(rgb.transpose(1,2,0).tobytes()).hexdigest(),
            'depth_sha256': hashlib.sha256(depth.tobytes()).hexdigest()}
        if any(type(receipt.get(k)) is not type(v) or receipt[k] != v for k,v in exact.items()):
            raise ValueError('RGB-D bytes or render-only receipt mismatch')
        if view == 'head' and float(rgb.std()) < 1.:
            raise ValueError('Head RGB appears blank')


def trace_session_imports(imports, record, *, instance_id=138):
    """Observe the three outer Session API calls, not internal settling steps.

    Wrap only the newly created evaluator instance; installed classes and shared
    modules are untouched. A completed event is recorded only after it returns.
    """
    original_factory = imports.Evaluator
    if type(instance_id) is not int or instance_id < 0:
        raise ValueError('Frozen nonnegative integer instance required')
    sequence = [('reset', None), ('load_task_instance', instance_id), ('reset', None)]
    record['official_api_events'] = []
    record['official_api_resets'] = record['load_frozen_instance_calls'] = 0
    constructed = False

    class EvaluatorTrace:
        def __init__(self, original): self.original = original
        def __getattr__(self, name): return getattr(self.original, name)
        def call(self, name, instance=None):
            events = record['official_api_events']
            if (any(e['status'] != 'completed' for e in events) or len(events) >= len(sequence) or
                    (name, instance) != sequence[len(events)]):
                raise ValueError('Unexpected original Session reset/load order')
            event = {'name': name, 'instance': instance, 'status': 'started'}
            events.append(event); supervisor.write('worker.json', record)
            result = (self.original.reset() if name == 'reset' else self.original.load_task_instance(instance))
            event['status'] = 'completed'
            key = 'official_api_resets' if name == 'reset' else 'load_frozen_instance_calls'
            record[key] += 1; supervisor.write('worker.json', record)
            return result
        def reset(self): return self.call('reset')
        def load_task_instance(self, instance):
            if type(instance) is not int: raise ValueError('Exact frozen integer instance required')
            return self.call('load_task_instance', instance)

    def construct(cfg):
        nonlocal constructed
        if constructed: raise ValueError('Only one evaluator construction permitted')
        constructed = True
        if CAMERA_RESOLUTION_PROFILE != 'full_v1' and not PRECONFIGURE_CAMERAS:
            raise ValueError('Alternate resolutions require preconfigured cameras')
        if PRECONFIGURE_CAMERAS:
            if PATH_TRACING and not CAMERA_PATH_TRACING_ALLOWED:
                raise ValueError('H55 must not combine camera initialization with a renderer change')
            import shared_camera_config as cameras
            if Path(cameras.__file__).resolve() != (supervisor.REPO/'scripts/semantic_robot/shared_camera_config.py').resolve():
                raise ValueError('Preconfigured camera wrapper source was shadowed')
            plain = imports.OmegaConf.to_container(cfg, resolve=True)
            modified, receipt = cameras.prepare_config(plain, CAMERA_RESOLUTION_PROFILE)
            record['camera_configuration'] = receipt
            supervisor.write('worker.json', record)
            cfg = imports.OmegaConf.create(modified)
        original = original_factory(cfg)
        if PRECONFIGURE_CAMERAS:
            record['camera_initialization'] = cameras.validate_wrapper(original.env, receipt['cameras'])
            supervisor.write('worker.json', record)
        return EvaluatorTrace(original)
    return replace(imports, Evaluator=construct)


def close_owned_app(app, record):
    if app is not None:
        try:
            if app.is_running():
                app.close()
        except BaseException as error:
            record.update(phase='close_failed', close_error=repr(error))
            supervisor.write('worker.json', record)
            raise


def scene_worker():
    signal.pthread_sigmask(signal.SIG_UNBLOCK, {signal.SIGTERM, signal.SIGINT})
    commit = supervisor.identity()
    supervisor.claim_stage('worker', commit)
    supervisor.check_resources(supervisor.snapshot())
    cores = {72, 73, 74, 75}
    if not cores <= os.sched_getaffinity(0):
        raise ValueError('Registered CPU affinity unavailable')
    os.sched_setaffinity(0, cores)
    record = {'source_commit': commit, 'pid': os.getpid(), 'phase': 'before_import',
              'session_constructions': 0, 'explicit_final_resets': 0, 'rgbd_captures': 0,
              'onboard_render_calls': 0, 'depth_views': 0,
              'robot_controls': 0, 'expert_prefix_controls': 0, 'old_policy_prefix_controls': 0,
              'model_calls': 0, 'training_steps': 0, 'success_rate': None,
              'task_name': 'turning_on_radio', 'instance': 138, 'split': 'train', 'seed': 0,
              'evaluator': 'existing v3.9.1 development runtime, not official challenge submission',
              'app_config': supervisor.app_configuration()}
    supervisor.write('worker.json', record)
    started = time.monotonic()
    created_app = None
    try:
        if WORKER_PREPARE is not None:
            WORKER_PREPARE(record)
        sys.path.insert(0, str(ADAPTER))
        sys.path.insert(0, str(supervisor.REPO / 'src'))
        import numpy as np
        from PIL import Image
        from native_oracle_low_v1 import official_factory as factory
        if Path(factory.__file__).resolve() != FACTORY.resolve():
            raise ValueError('Unexpected evaluator factory import')
        window = factory.load_official_oracle_window(WINDOW)
        validate_window(window)
        if Path(factory.DEFAULT_EVAL_ROOT).resolve() != EVAL_ROOT.resolve():
            raise ValueError('Unexpected evaluator wrapper root')
        imports = factory._lazy_official_imports(og_root=OG, eval_root=EVAL_ROOT, gpu=3)
        imports = trace_session_imports(imports, record)
        import omnigibson as og
        import omnigibson.simulator as og_startup
        onboard = importlib.import_module('semantic_robot.v2.onboard')
        backend = importlib.import_module('semantic_robot.og_backend')
        for module, relative in ((onboard, 'semantic_robot/v2/onboard.py'), (backend, 'semantic_robot/og_backend.py')):
            if Path(module.__file__).resolve() != (supervisor.REPO/'src'/relative).resolve():
                raise ValueError('Onboard observation import shadowed the frozen Git source')
        OnboardRGBD, array = onboard.OnboardRGBD, backend.array
        if Path(og_startup.__file__).resolve() != OG_SOURCE.resolve():
            raise ValueError('Unexpected installed simulator import')
        configure_viewer_before_launch(imports.gm, og, record)

        def construct():
            nonlocal created_app
            from isaacsim import SimulationApp
            if Path(inspect.getfile(SimulationApp)).resolve() != supervisor.APP_SOURCE.resolve():
                raise ValueError('Unexpected SimulationApp import')
            record['phase'] = 'constructing_app'; supervisor.write('worker.json', record)
            created_app = supervisor.construct_app(SimulationApp)
            import carb.settings
            settings = carb.settings.get_settings()
            if PATH_TRACING:
                # Native SimulationApp overwrites totalSpp with its per-frame
                # spp; fix only this owned empty app before strict readback.
                record['pathtracing_initial_settings'] = pathtracing.apply_settings(settings)
            actual = {key: settings.get(key) for key in supervisor.RUNTIME_SETTINGS}
            supervisor.validate_settings(actual)
            record.update(phase='loading_scene', actual_settings=actual)
            supervisor.write('worker.json', record)
            return created_app

        bindings = {(supervisor.OG_KIT, supervisor.EXPERIENCE): supervisor.DEPENDENCIES[supervisor.OG_KIT],
                    (ICON, INSTALLED_ICON): ICON_SHA}
        renderer_context = nullcontext()
        if PATH_TRACING:
            import shared_pathtracing as pathtracing
            renderer_context = pathtracing.before_scene(og, og_startup,
                source_sha256=EXTRA_DEPENDENCIES[OG_SOURCE], record=record, write=supervisor.write)
        with renderer_context, private_og_startup(og_startup, source_sha256=EXTRA_DEPENDENCIES[OG_SOURCE],
                experience=supervisor.EXPERIENCE, copy_bindings=bindings, construct=construct) as bridge:
            record['startup_bridge'] = bridge
            record['session_constructions'] = 1
            record['phase'] = 'entering_official_session'; supervisor.write('worker.json', record)
            # Save the original exception before official __exit__ can terminate
            # Python through native Kit shutdown.
            with factory.OfficialEvaluatorSession(window, gpu=3, imports=imports) as environment:
                try:
                    record['phase'] = 'final_original_reset'; supervisor.write('worker.json', record)
                    environment.reset()
                    record['explicit_final_resets'] = 1
                    validate_viewer_after_scene(imports.gm, og.sim, record)
                    if PATH_TRACING:
                        pathtracing.validate_after_scene(og, record)
                    env = environment.evaluator.env
                    if PRECONFIGURE_CAMERAS:
                        import shared_camera_config as cameras
                        cameras.validate_wrapper(env, record['camera_configuration']['cameras'])
                    robot = env.robots[0]
                    if robot.action_dim != 23:
                        raise ValueError('Original R1Pro 23-control embodiment required')
                    sensor = (OnboardRGBD(env, resolutions=cameras.resolutions(CAMERA_RESOLUTION_PROFILE))
                              if CAMERA_RESOLUTION_PROFILE != 'full_v1' else OnboardRGBD(env))
                    if CAMERA_RESOLUTION_PROFILE != 'full_v1':
                        intrinsics_before = cameras.camera_intrinsics(sensor.sensors, array, CAMERA_RESOLUTION_PROFILE)
                    metadata = {view: {'width': camera.image_width, 'height': camera.image_height}
                                for view, camera in sensor.sensors.items()}
                    model = SimpleNamespace(spec={'metadata': {'cameras': metadata}})
                    before = array(robot.get_joint_positions()).copy()
                    def render_only():
                        if record['onboard_render_calls'] >= 4:
                            raise ValueError('Exceeded registered observation render barrier')
                        og.sim.render()
                        record['onboard_render_calls'] += 1
                    images, depths, sensors = sensor.read(model, render=render_only)
                    after = array(robot.get_joint_positions()).copy()
                    validate_capture(before, after, images, depths, sensors, CAMERA_RESOLUTION_PROFILE)
                    if PRECONFIGURE_CAMERAS:
                        record['camera_after_capture'] = cameras.validate_wrapper(env, record['camera_configuration']['cameras'])
                        record['preconfigured_cameras_verified'] = True
                    if CAMERA_RESOLUTION_PROFILE != 'full_v1':
                        intrinsics_after = cameras.camera_intrinsics(sensor.sensors, array, CAMERA_RESOLUTION_PROFILE)
                        if intrinsics_before != intrinsics_after:
                            raise ValueError('Camera intrinsics changed during render-only capture')
                        record['camera_intrinsics'] = intrinsics_after
                        record['camera_intrinsics_verified'] = True
                    if PATH_TRACING:
                        pathtracing.validate_after_scene(og, record)
                    if (record['onboard_render_calls'] != 4 or record['official_api_resets'] != 2 or
                            record['load_frozen_instance_calls'] != 1 or
                            any(e['status'] != 'completed' for e in record['official_api_events'])):
                        raise ValueError('Original Session/render events did not complete exactly')
                    for view in sensor.sensors:
                        rgb = images[view + '_rgb'].transpose(1, 2, 0)
                        Image.fromarray(rgb).save(supervisor.OUTPUT / (view + '.png'))
                    np.savez_compressed(supervisor.OUTPUT / 'depth.npz', **depths)
                    supervisor.write('sensors.json', sensors)
                    supervisor.write('robot_proprio.json', {'before': before.tolist(), 'after': after.tolist()})
                    artifacts = ['head.png', 'left_wrist.png', 'right_wrist.png',
                                 'depth.npz', 'sensors.json', 'robot_proprio.json']
                    record.update(phase='scene_observation_complete', rgbd_captures=1, depth_views=len(depths),
                                  seconds=time.monotonic()-started, sensors=sensors,
                                  startup_launch_calls=bridge['launch_calls'],
                                  app_constructions=bridge['app_constructions'],
                                  verified_copy_noops=len(bridge['verified_copy_noops']),
                                  shared_install_writes=bridge['shared_install_writes'],
                                  artifacts_sha256={name: hashlib.sha256(
                                      (supervisor.OUTPUT/name).read_bytes()).hexdigest() for name in artifacts},
                                  input_resolution_changed=CAMERA_RESOLUTION_PROFILE != 'full_v1',
                                  privileged_scene_state_used=False)
                    supervisor.write('worker.json', record)
                except BaseException as error:
                    record.update(phase='failed', error=repr(error), seconds=time.monotonic()-started)
                    supervisor.write('worker.json', record)
                    raise
    except BaseException as error:
        record.update(phase='failed', error=repr(error), seconds=time.monotonic()-started)
        supervisor.write('worker.json', record)
        raise
    finally:
        # Official __exit__ normally owns shutdown; if construction failed
        # before OG could adopt the app, only our explicitly created app closes.
        close_owned_app(created_app, record)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    for mode in ('launch', 'supervise', 'worker'):
        group.add_argument('--' + mode, action='store_true')
    args = parser.parse_args()
    configure_profile()
    (supervisor.launch if args.launch else supervisor.supervise if args.supervise else scene_worker)()
