"""H72 read-only native clock/pose audit. Never authorizes an actor gate.

Unchanged runner, actions, four-render barrier and all safety thresholds.
Privileged robot pose is written only to a diagnostic journal, never returned
to the runner, its controller, odometer or model. No extra physics or render.
"""
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.spatial.transform import Rotation

INSTALLED = {
    Path('/mnt/sdc1/xhz/miniconda3/envs/behavior/lib/python3.11/site-packages/isaacsim/'
         'exts/isaacsim.core.api/isaacsim/core/api/simulation_context/simulation_context.py'):
        'ebafc6bcb30a454925fe21b96dcdbd4637c922a3fa9d5a6947308c9796ba5028',
    Path('/mnt/sdc1/xhz/BEHAVIOR2026/BEHAVIOR-1K/OmniGibson/omnigibson/utils/usd_utils.py'):
        '19a14c5660d923dc73e0b7e8605dd59f31ad8ecdf6c1cc167786e922bde5556a',
    Path('/mnt/sdc1/xhz/BEHAVIOR2026/BEHAVIOR-1K/OmniGibson/omnigibson/prims/xform_prim.py'):
        '7a52fe6a45346580e945bf922a9152622370f8b78a1476c0cdd56f27e6220005',
}


def validate_installed():
    for path, expected in INSTALLED.items():
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError('H72 native clock/render implementation changed: '+str(path))


def audit_digest(normal_digest):
    files = ('probe_observation_clock.py', 'launch_h72.py')
    payload = {'normal_digest': normal_digest, 'purpose': 'AUDIT_ONLY_NOT_POLICY_GATE',
               'files': {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                         for name in files}}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def array(value):
    if hasattr(value, 'detach'): value = value.detach().cpu().numpy()
    return np.asarray(value)


def head_hash(sensor):
    sample, _ = sensor.get_obs()  # Read annotator buffers, no step/update.
    rgb = array(sample['rgb'])[..., :3].copy()
    depth = array(sample['depth_linear']).copy()
    return {'rgb_sha256': hashlib.sha256(rgb.tobytes()).hexdigest(),
            'depth_sha256': hashlib.sha256(depth.tobytes()).hexdigest(),
            'rgb_shape': list(rgb.shape), 'depth_shape': list(depth.shape),
            'depth_dtype': str(depth.dtype)}


def native_pose_reader(robot):
    # robot.get_position_orientation() reads Fabric, so is NOT independent
    # physical truth. Read the existing PhysX tensor view directly, without
    # clearing/populating its OG cache or forcing a Fabric/render update.
    from omnigibson.utils.usd_utils import ControllableObjectViewAPI, get_robot_kinematic_tree_pattern
    path = robot.articulation_root_path
    view = ControllableObjectViewAPI._VIEWS_BY_PATTERN[get_robot_kinematic_tree_pattern(path)]
    row = view._idx[path]
    link = view._link_idx[row][robot.base_footprint_link_name]
    def read():
        raw = view._view.get_link_transforms()
        read.last_tensor_backend = {'type': type(raw).__module__+'.'+type(raw).__name__,
                                    'device': str(getattr(raw, 'device', 'cpu'))}
        poses = array(raw).copy()
        if poses.ndim != 3 or poses.shape[-1] != 7:
            raise ValueError('Expected native PhysX articulation link transforms')
        pose = poses[row, link]
        return {'physx': (pose[:3], pose[3:]), 'fabric': robot.get_position_orientation()}
    return read


class Journal:
    def __init__(self, output):
        self.output = Path(output)
        self.stream = None
        self.origin = None
        self.rows = 0
        self.robot = None
        self.read_poses = None
        self.last_state_clock = None

    def record(self, phase, sim, *, head=None, snapshot=None, render_index=None):
        if self.rows >= 8192: raise ValueError('Bounded H72 journal exhausted')
        started = time.monotonic()
        # Buffer FIRST: extra PhysX reads may synchronize GPU work. This remains
        # an instrumented, not timing-identical, execution; record its cost.
        head_value = head_hash(head) if head is not None else None
        after_head = time.monotonic()
        transforms = {}
        for source, pose in self.read_poses().items():
            p, q = (array(x).copy() for x in pose)
            if p.shape != (3,) or q.shape != (4,) or not np.isfinite(np.r_[p, q]).all():
                raise ValueError('Finite native diagnostic robot pose required')
            T = np.eye(4); T[:3, :3] = Rotation.from_quat(q).as_matrix(); T[:3, 3] = p
            transforms[source] = T
        if set(transforms) != {'physx', 'fabric'}: raise ValueError('Both independent pose sources required')
        if self.origin is None: self.origin = transforms['physx'].copy()
        # Persist only the relative trajectory. Neither it nor the origin is
        # ever placed in the policy observation or a control return value.
        after_pose = time.monotonic()
        row = {'event': self.rows, 'phase': phase, 'snapshot_id': snapshot,
               'render_index': render_index, 'simulation_time': float(sim.current_time),
               'physics_index': int(sim.current_time_step_index),
               'physics_dt': float(sim.get_physics_dt()),
               'rendering_dt': float(sim.get_rendering_dt()),
               'sim_step_dt': float(sim.get_sim_step_dt()),
               **{'diagnostic_'+name+'_body_in_initial_body': (np.linalg.inv(self.origin)@T).tolist()
                  for name,T in transforms.items()},
               'wall_monotonic_started': started, 'head_read_hash_seconds': after_head-started,
               'pose_read_seconds': after_pose-after_head,
               'physx_tensor_backend': getattr(self.read_poses, 'last_tensor_backend', None),
               'never_sent_to_actor_or_odometer': True}
        if head_value is not None: row['head'] = head_value
        if self.stream is None: self.stream = (self.output/'PRIVILEGED_clock_pose.jsonl').open('x', buffering=1)
        self.stream.write(json.dumps(row, allow_nan=False)+'\n')
        self.rows += 1

    def finish(self):
        if self.stream is not None: self.stream.flush()
        path = self.output/'PRIVILEGED_clock_pose.jsonl'
        return {'purpose': 'observation_clock_diagnostic_not_policy_gate', 'events': self.rows,
                'journal_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                'extra_controls': 0, 'extra_renders': 0, 'truth_sent_to_actor_or_odometer': False,
                'instrumented_timing_not_identical_to_H71': True}

    def close(self):
        if self.stream is not None: self.stream.close()


def instrument(runner, output, get_sim, make_pose_reader=native_pose_reader):
    journal = Journal(output)
    original_robot, original_camera = runner.CalibratedRobot, runner.OnboardRGBD
    original_write, original_digest = runner.write, runner.implementation_digest

    class AuditRobot(original_robot):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            journal.robot = self.robot
            journal.read_poses = make_pose_reader(self.robot)

        def state(self):
            value = super().state()
            sim = get_sim()
            clock = (float(sim.current_time), int(sim.current_time_step_index))
            if clock != journal.last_state_clock:
                journal.record('new_state_clock', sim)
                journal.last_state_clock = clock
            return value  # Original object; no private fields injected.

    class AuditCamera(original_camera):
        def read(self, model, *, render):
            snapshot = self.snapshot_id + 1
            journal.record('capture_before', get_sim(), head=self.sensors['head'], snapshot=snapshot)
            count = 0
            def observed_render():
                nonlocal count
                render(); count += 1
                journal.record('capture_after_render', get_sim(), head=self.sensors['head'],
                               snapshot=snapshot, render_index=count)
            value = super().read(model, render=observed_render)
            if count != 4: raise ValueError('H72 must preserve original four-render barrier')
            journal.record('capture_return', get_sim(), head=self.sensors['head'], snapshot=snapshot)
            return value

    def audit_write(path, value):
        if Path(path) == Path(output)/'result.json':
            journal.record('final_after_safe_hold', get_sim())
            original_write(Path(output)/'clock_audit.json', journal.finish())
        elif Path(path) == Path(output)/'failure.json':
            # This callback is inside run_v2's native session, before SDK exit.
            # Seal existing evidence without another pose/read that could mask
            # the original exception. Initialization may precede any samples.
            try:
                summary = journal.finish() if journal.rows else {'events': 0}
                original_write(Path(output)/'clock_audit_failure.json',
                               {**summary, 'complete': False, 'original_failure': value})
            except BaseException as error:
                print('H72 partial audit seal failed: '+repr(error), file=sys.stderr, flush=True)
        return original_write(path, value)

    runner.CalibratedRobot, runner.OnboardRGBD = AuditRobot, AuditCamera
    runner.write = audit_write
    runner.implementation_digest = lambda: audit_digest(original_digest())
    return journal


def main():
    validate_installed()
    import run_v2
    # Only the launcher can choose the original registered gate arguments.
    args = sys.argv[1:]
    for key, value in {'--mode': 'gate', '--task': '0', '--prefix': '0',
                       '--native-profile': 'a100_full_v1', '--gpu': '3'}.items():
        if args.count(key) != 1 or args[args.index(key)+1] != value:
            raise ValueError('H72 is a zero-prefix task0 diagnostic, never an actor')
    out = Path(args[args.index('--output')+1])
    def get_sim():
        import omnigibson as og
        return og.sim
    journal = instrument(run_v2, out, get_sim)
    try: run_v2.main()
    finally: journal.close()


if __name__ == '__main__': main()
