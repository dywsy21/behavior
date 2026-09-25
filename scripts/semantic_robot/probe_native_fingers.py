"""H67: bounded native R1Pro FK check, not a task rollout or contact test.

One robot on a plain floor, no cameras, no learned policy. Calibration poses
are explicit teleports followed by one native physics step; they are never
exported as training actions or counted as task success. The H52 supervisor's
training identities, memory limits and cleanup remain unchanged.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import inspect
import json
import os
from pathlib import Path
import random
import signal
import sys
import time

import probe_simulator_startup as supervisor
import shared_pathtracing as pathtracing
from shared_og_startup import private_og_startup

OG = Path('/mnt/sdc1/xhz/BEHAVIOR2026/BEHAVIOR-1K/OmniGibson')
OG_SOURCE = OG / 'omnigibson/simulator.py'
ROBOT_CONFIG = OG / 'omnigibson/eval/r1pro.yaml'
ROBOT_ROOT = OG.parent / 'datasets/omnigibson-robot-assets/models/r1pro'
ROBOT_ASSET = ROBOT_ROOT / 'usd/r1pro.usda'
SURFACES = Path('/mnt/nvme_tmp/robodojo_agentic_20260925/h64_finger_collision_asset_v1/result.json')
SURFACE_SHA = '5f2cbf18b989eb22bc98665f48538c6cc306679d3be94ab4de46f2af7b4c6f04'
ICON = OG.parent / 'docs/assets/OmniGibson_logo.png'
INSTALLED_ICON = supervisor.ISAAC / 'apps/OmniGibson_logo.png'
ICON_SHA = '415480a53ed07ee3b909b4a3510f0aa44c4f44aa23296713ebc07af5a461ec23'
EXTRA_DEPENDENCIES = {
    OG_SOURCE: 'd800c2832f24962c440c4781ebfdb4ea78c74aac37d2c74ee7894ae430f1e2a9',
    OG / 'omnigibson/__init__.py': 'c7867c236051fd8994c75973284b5e88b2637c5b8ab1fece9c6e29bdbd9ca03f',
    OG / 'omnigibson/envs/env_base.py': 'ab3e0cefd46a583a8fa9e8ccd42e5cc29590f0b9efddfb0a6f34dab74e4f6c7c',
    OG / 'omnigibson/robots/robot.py': '998826feedcf8c28208681a809cbf4500d3da0cbf1f6d07a7384627e09726271',
    OG / 'omnigibson/controllers/multi_finger_gripper_controller.py': 'dad60798c9df2a2d0f3a2d1489e6925206de1b8c4ab79586296ae9695a0ab4b3',
    OG / 'omnigibson/utils/usd_utils.py': '19a14c5660d923dc73e0b7e8605dd59f31ad8ecdf6c1cc167786e922bde5556a',
    OG / 'omnigibson/scenes/scene_base.py': '16e4a4bda8ef511fa72bb38d86c657f74b2ec0551a2531e814b80e11e59a85ef',
    OG / 'omnigibson/macros.py': '5e8ebb1a00fa3e864c5a4984f126442e91b6653f5869977ecddbf4039ea73228',
    ROBOT_CONFIG: 'a98fa8a81472adfecfb642778d4d7a01e20131be7f70824e0ae095d665cdbb93',
    ROBOT_ASSET: '6029617cdd3aefce981428058a1c82cffe6af20ae61dc5be1da7334342c3bc52',
    ROBOT_ROOT / 'r1pro.yaml': '63f841cffd5c499102416a797a22fa5b4cbaf146539df7dde8a4a1053e2326f1',
    SURFACES: SURFACE_SHA, ICON: ICON_SHA, INSTALLED_ICON: ICON_SHA,
}
# Fractions use each measured joint's own limits, never a guessed mean opening.
# Wrist offsets address the last DOF in each native arm's control index list.
POSES = (
    ('reference_open', (.9, .9, .9, .9), (0., 0.)),
    ('symmetric_mid', (.5, .5, .5, .5), (0., 0.)),
    ('symmetric_narrow', (.1, .1, .1, .1), (0., 0.)),
    ('left_asymmetric', (.2, .8, .5, .5), (0., 0.)),
    ('right_asymmetric', (.5, .5, .2, .8), (0., 0.)),
    ('both_asymmetric', (.2, .8, .8, .2), (0., 0.)),
    ('wrist_rotated_open', (.9, .9, .9, .9), (.12, -.12)),
    ('wrist_rotated_asymmetric', (.2, .8, .8, .2), (.12, -.12)),
    ('return_reference_open', (.9, .9, .9, .9), (0., 0.)),
)
POSITION_TOL_M = .00025
ANGLE_TOL_RAD = .001
# Native smooth-control reset averages the two position targets. Permit modest
# one-step tracking drift while still separating .1/.5/.9 openings and rejecting
# collapse of .2/.8 asymmetric jaws to their mean. This is NOT FK tolerance.
FINGER_POSE_TOL_M = .005
BODY_POSE_TOL = .003  # mixed measured q18 units; coverage gate, not FK accuracy


def configure_profile():
    supervisor.ENTRYPOINT = Path(__file__).resolve()
    supervisor.OUTPUT = Path('/mnt/nvme_tmp/robodojo_agentic_20260925/h67_native_fingers_v1')
    supervisor.RUNTIME = Path('/mnt/nvme_tmp/robodojo_sim_runtime_20260925/h67_native_fingers_v1')
    supervisor.WALL_SECONDS = 600
    supervisor.PROFILE_SETTINGS = dict(pathtracing.SETTINGS)
    supervisor.PROFILE_APP_CONFIG = dict(pathtracing.APP_CONFIG)
    supervisor.RUNTIME_SETTINGS = {**supervisor.SETTINGS, **supervisor.GPU_SETTINGS, **pathtracing.SETTINGS}
    supervisor.DEPENDENCIES = {**supervisor.DEPENDENCIES, **EXTRA_DEPENDENCIES}
    supervisor.FIXED_ENV = {**supervisor.FIXED_ENV, 'OMNIGIBSON_GPU_ID': '3',
                            'OMNIGIBSON_HEADLESS': '1', 'OMNIGIBSON_NO_OMNI_LOGS': '0'}
    supervisor.SUCCESS_FIELDS = {
        'phase': 'native_fingers_complete', 'environment_constructions': 1,
        'calibration_count': 1, 'completed_samples': 9, 'joint_pose_sets': 9,
        'explicit_physics_steps': 9, 'robot_controls': 0, 'model_calls': 0,
        'training_steps': 0, 'task_loads': 0, 'rgbd_captures': 0,
        'startup_launch_calls': 1, 'app_constructions': 1,
        'verified_copy_noops': 2, 'shared_install_writes': 0,
        'native_geometry_passed': True, 'not_success_rate': True,
    }
    supervisor.BUDGET_DETAILS = {
        'environment_constructions': 1, 'joint_pose_sets': 9, 'explicit_physics_steps': 9,
        'calibration_count': 1, 'robot_controls': 0, 'task_loads': 0,
        'model_calls': 0, 'training_steps': 0, 'rgbd_captures': 0,
        'initialization_internal_steps': 'native Environment load/reset; wall bounded',
        'not_success_rate': True,
    }


def robot_only_config(original):
    if original.get('model') != 'r1pro' or original.get('action_normalize') is not False:
        raise ValueError('Original unnormalized R1Pro configuration required')
    robot = copy.deepcopy(original)
    robot.pop('eval')
    robot['obs_modalities'] = []
    robot['include_sensor_names'] = []
    # All physical/controller/reset fields and the authored robot are unchanged.
    return {'scene': {'type': 'Scene', 'use_floor_plane': True, 'use_skybox': False},
            'robots': [robot], 'objects': [], 'task': {'type': 'DummyTask'},
            'env': {'action_frequency': 30, 'rendering_frequency': 30,
                    'physics_frequency': 120, 'device': 'cpu', 'automatic_reset': False}}


def pose_schedule(q, lower, upper, body_indices, finger_indices):
    import numpy as np
    q, lower, upper = (np.asarray(x, dtype=float) for x in (q, lower, upper))
    body, fingers = (np.asarray(x) for x in (body_indices, finger_indices))
    if (q.ndim != 1 or q.shape != lower.shape or q.shape != upper.shape or
            not np.isfinite(q).all() or
            body.shape != (18,) or fingers.shape != (4,) or
            body.dtype.kind not in 'iu' or fingers.dtype.kind not in 'iu'):
        raise ValueError('Finite native vectors and integer embodiment indices required')
    indices = np.r_[body, fingers]
    if (len(set(indices)) != 22 or np.any(indices < 0) or np.any(indices >= len(q)) or
            not np.isfinite(lower[indices]).all() or not np.isfinite(upper[indices]).all() or
            np.any(lower[indices] >= upper[indices])):
        raise ValueError('Distinct native arm/trunk/finger joint indices and limits required')
    schedule = []
    for name, fractions, wrists in POSES:
        target = q.copy()
        target[fingers] = lower[fingers] + np.asarray(fractions)*(upper[fingers]-lower[fingers])
        target[body[[10, 17]]] += wrists
        if np.any(target[indices] < lower[indices]) or np.any(target[indices] > upper[indices]):
            raise ValueError('Registered calibration pose exceeds native limits: ' + name)
        schedule.append((name, target))
    return schedule


def validate_pose_coverage(requested, actual, body_indices, finger_indices):
    import numpy as np
    requested, actual = (np.asarray(v, dtype=float) for v in (requested, actual))
    if (requested.ndim != 1 or requested.shape != actual.shape or
            not np.isfinite(requested).all() or not np.isfinite(actual).all()):
        raise ValueError('Native measured pose is missing or invalid')
    body_error = float(np.max(np.abs(requested[body_indices]-actual[body_indices])))
    finger_error = float(np.max(np.abs(requested[finger_indices]-actual[finger_indices])))
    receipt = {'body_max_abs_error': body_error, 'finger_max_abs_error_m': finger_error,
               'body_tolerance': BODY_POSE_TOL, 'finger_tolerance_m': FINGER_POSE_TOL_M}
    if body_error > BODY_POSE_TOL or finger_error > FINGER_POSE_TOL_M:
        raise ValueError('Actual pose did not cover the registered sample: ' + json.dumps(receipt))
    return receipt


def compare_native(kin, model, surfaces):
    """Compare portable FK with native link reads, including all authored vertices."""
    import numpy as np
    from semantic_robot.og_backend import array
    from semantic_robot.v2.kinematics import transform
    from semantic_robot.v2.finger_kinematics import rigid
    from scipy.spatial.transform import Rotation
    state = kin.state()
    geometry = model.finger_geometry(state.q, state.finger_qpos)
    if geometry.get('valid') is not True:
        raise ValueError('Current named finger geometry unavailable')
    names = {name for links in geometry['arms'].values() for name in links}
    if names != set(surfaces['links']) or len(names) != 4:
        raise ValueError('Asset and native calibrated fingers disagree')
    links, passed = {}, True
    for arm, rows in geometry['arms'].items():
        for name, row in rows.items():
            predicted = rigid(row['T_base_link'])
            native = rigid(transform(*(array(v) for v in
                kin.api.get_link_relative_position_orientation(kin.path, name))))
            vertices = np.concatenate([np.asarray(m['vertices_link_m'], dtype=float)
                                       for m in surfaces['links'][name]['meshes'].values()])
            if vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all():
                raise ValueError('Finite authored finger vertices required')
            p_error = float(np.linalg.norm(predicted[:3, 3]-native[:3, 3]))
            a_error = float(np.linalg.norm(Rotation.from_matrix(
                predicted[:3, :3] @ native[:3, :3].T).as_rotvec()))
            vertex_error = float(np.linalg.norm(
                vertices @ (predicted[:3, :3]-native[:3, :3]).T +
                predicted[:3, 3]-native[:3, 3], axis=1).max())
            valid = p_error <= POSITION_TOL_M and a_error <= ANGLE_TOL_RAD and vertex_error <= POSITION_TOL_M
            passed = passed and valid
            links[name] = {'arm': arm, 'native_T_base_link': native.tolist(),
                           'predicted_T_base_link': predicted.tolist(), 'position_m': p_error,
                           'angle_rad': a_error, 'authored_vertex_max_error_m': vertex_error,
                           'vertices': len(vertices), 'passed': bool(valid)}
    body = kin.compare(model)
    for name in ('left', 'right', 'torso'):
        values = body[name]
        passed = passed and all(np.isfinite(values[k]) and values[k] <= tol
            for k, tol in (('position_m', POSITION_TOL_M), ('angle_rad', ANGLE_TOL_RAD)))
    return {'q18': state.q.tolist(), 'finger_qpos': state.finger_qpos,
            'calibration_sha256': model.sha, 'finger_links': links, 'native_comparison': body,
            'passed': bool(passed), 'position_tolerance_m': POSITION_TOL_M,
            'angle_tolerance_rad': ANGLE_TOL_RAD, 'cooked_contact_verified': False}


def run_samples(robot, sim, surfaces, record, save):
    import numpy as np
    import torch
    from semantic_robot.og_backend import array
    from semantic_robot.v2.og_calibration import CalibratedRobot
    from semantic_robot.v2.kinematics import RobotModel
    kin = CalibratedRobot(robot)  # validates the real 23-control / q18 layout
    fingers = np.r_[array(robot.gripper_control_idx['left']),
                    array(robot.gripper_control_idx['right'])].astype(int)
    schedule = pose_schedule(array(robot.get_joint_positions()).copy(),
                             array(robot.joint_lower_limits), array(robot.joint_upper_limits),
                             kin.indices, fingers)
    record['joint_names'] = list(robot.joints)
    record['kinematics_receipt'] = kin.receipt()
    record['finger_indices'] = fingers.tolist()
    model, samples = None, []
    for name, requested in schedule:
        record.update(phase='measuring_native_pose', active_pose=name)
        save('worker.json', record)
        # Full native state assignment, not a 23-D policy action. Native setter
        # clears articulation caches and resets controller goals before stepping.
        current = robot.get_joint_positions()
        robot.set_joint_positions(torch.as_tensor(requested, dtype=current.dtype, device=current.device),
                                  normalized=False, drive=False)
        record['joint_pose_sets'] += 1
        save('worker.json', record)
        robot.set_joint_velocities(torch.zeros_like(current))
        sim.step_physics()
        record['explicit_physics_steps'] += 1
        save('worker.json', record)
        actual = array(robot.get_joint_positions()).copy()
        sample = {'name': name, 'requested_native_q': requested.tolist(), 'actual_native_q': actual.tolist()}
        samples.append(sample)
        save('samples.json', samples)  # preserve measured evidence even on a failed gate
        sample['coverage'] = validate_pose_coverage(requested, actual, kin.indices, fingers)
        save('samples.json', samples)
        if model is None:
            state = kin.state()
            metadata = {'cameras': {}, 'finger_kinematics': kin.finger_kinematics(state),
                        'source': 'robot_only_no_camera_native_probe', 'scene_truth': False,
                        'joint_indices': kin.indices.tolist(), 'not_deployment_calibration': True}
            model = RobotModel.from_reference(state.q, state.lower, state.upper,
                                              state.poses, state.jacobians, metadata)
            record['calibration_count'] += 1
            save('calibration.json', model.spec)
        sample['geometry'] = compare_native(kin, model, surfaces)
        save('samples.json', samples)
        if not sample['geometry']['passed']:
            raise ValueError('Native finger/body geometry failed at ' + name)
        record['completed_samples'] += 1
        save('worker.json', record)
    if [s['name'] for s in samples] != [p[0] for p in POSES] or record['calibration_count'] != 1:
        raise ValueError('Registered native pose sequence was not completed once')
    record['native_geometry_passed'] = True


def native_worker():
    signal.pthread_sigmask(signal.SIG_UNBLOCK, {signal.SIGTERM, signal.SIGINT})
    commit = supervisor.identity()
    supervisor.claim_stage('worker', commit)
    supervisor.check_resources(supervisor.snapshot())
    cores = {72, 73, 74, 75}
    if not cores <= os.sched_getaffinity(0): raise ValueError('Registered CPU affinity unavailable')
    os.sched_setaffinity(0, cores)
    record = {key: (False if type(value) is bool else 0)
              for key, value in supervisor.SUCCESS_FIELDS.items() if key != 'phase'}
    record.update(source_commit=commit, pid=os.getpid(), phase='before_import', not_success_rate=True,
                  success_rate=None, task_name=None, instance=None, seed=0,
                  cooked_contact_verified=False, training_data_exported=False,
                  dependencies={str(k): v for k, v in supervisor.DEPENDENCIES.items()})
    supervisor.write('worker.json', record)
    started, app = time.monotonic(), None
    try:
        sys.path.insert(0, str(OG))
        sys.path.insert(0, str(supervisor.REPO / 'src'))
        import yaml
        import numpy as np
        import torch
        random.seed(0)
        np.random.seed(0)
        torch.manual_seed(0)
        original = yaml.safe_load(ROBOT_CONFIG.read_text())
        config = robot_only_config(original)
        raw = SURFACES.read_bytes()
        if hashlib.sha256(raw).hexdigest() != SURFACE_SHA:
            raise ValueError('Frozen H64 surfaces changed')
        surfaces = json.loads(raw)['surfaces']
        import omnigibson as og
        import omnigibson.simulator as startup
        from omnigibson.macros import gm
        if Path(startup.__file__).resolve() != OG_SOURCE.resolve() or og.app is not None or og.sim is not None:
            raise ValueError('Fresh pinned OmniGibson import required')
        # CPU PhysX is the installed official evaluator setting, not a backend change.
        if gm.USE_GPU_DYNAMICS is not False:
            raise ValueError('Original evaluator CPU PhysX setting changed')
        with gm.unlocked(): gm.RENDER_VIEWER_CAMERA = False
        record.update(config=config, physics_gpu_dynamics=False, render_viewer_camera=False)

        def construct():
            nonlocal app
            from isaacsim import SimulationApp
            if Path(inspect.getfile(SimulationApp)).resolve() != supervisor.APP_SOURCE.resolve():
                raise ValueError('Unexpected SimulationApp import')
            record['phase'] = 'constructing_app'; supervisor.write('worker.json', record)
            app = supervisor.construct_app(SimulationApp)
            settings = pathtracing.get_settings()
            pathtracing.apply_settings(settings)
            actual = {k: settings.get(k) for k in supervisor.RUNTIME_SETTINGS}
            supervisor.validate_settings(actual)
            record['actual_settings'] = actual
            return app

        bindings = {(supervisor.OG_KIT, supervisor.EXPERIENCE): supervisor.DEPENDENCIES[supervisor.OG_KIT],
                    (ICON, INSTALLED_ICON): ICON_SHA}
        with pathtracing.before_scene(og, startup, source_sha256=EXTRA_DEPENDENCIES[OG_SOURCE],
                record=record, write=supervisor.write), private_og_startup(startup,
                source_sha256=EXTRA_DEPENDENCIES[OG_SOURCE], experience=supervisor.EXPERIENCE,
                copy_bindings=bindings, construct=construct) as bridge:
            record.update(phase='constructing_robot_only_environment', environment_constructions=1)
            supervisor.write('worker.json', record)
            env = og.Environment(configs=config)
            if len(env.robots) != 1 or env.robots[0].sensors or og.sim.viewer_camera is not None:
                raise ValueError('Exactly one robot and no sensor/viewer allowed')
            robot = env.robots[0]
            if Path(robot.usd_path).resolve() != ROBOT_ASSET.resolve():
                raise ValueError('Robot did not load the frozen H64 asset')
            pathtracing.validate_after_scene(og, record)
            run_samples(robot, og.sim, surfaces, record, supervisor.write)
            pathtracing.validate_after_scene(og, record)
            record.update(phase='native_fingers_complete', startup_launch_calls=bridge['launch_calls'],
                          app_constructions=bridge['app_constructions'],
                          verified_copy_noops=len(bridge['verified_copy_noops']),
                          shared_install_writes=bridge['shared_install_writes'],
                          seconds=time.monotonic()-started)
            record['artifacts_sha256'] = {name: hashlib.sha256((supervisor.OUTPUT/name).read_bytes()).hexdigest()
                                          for name in ('samples.json', 'calibration.json')}
            supervisor.write('worker.json', record)
    except BaseException as error:
        record.update(phase='failed', error=repr(error), seconds=time.monotonic()-started)
        supervisor.write('worker.json', record)
        raise
    finally:
        if app is not None:
            try:
                if app.is_running(): app.close()
            except BaseException as error:
                record.update(phase='close_failed', close_error=repr(error))
                supervisor.write('worker.json', record)
                raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    for mode in ('launch', 'supervise', 'worker'): group.add_argument('--'+mode, action='store_true')
    args = parser.parse_args()
    configure_profile()
    (supervisor.launch if args.launch else supervisor.supervise if args.supervise else native_worker)()
