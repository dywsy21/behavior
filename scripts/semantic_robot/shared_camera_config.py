"""Preconfigure the three evaluation cameras; never resize live render products.

This module is CPU-importable. The installed EnvironmentWrapper is imported
only when Hydra wraps an already constructed environment. Its deferred-space
flag deliberately preserves the reviewed chunk runner's post-articulation
observation-space reload, without that runner's destructive camera setters.
"""
from copy import deepcopy
import hashlib
import json
import re


RESOLUTIONS = {'head': 720, 'left_wrist': 480, 'right_wrist': 480}
RESOLUTION_PROFILES = {
    'full_v1': RESOLUTIONS,
    'shared_v1': {'head': 512, 'left_wrist': 320, 'right_wrist': 320},
}
ORIGINAL_WRAPPER = 'omnigibson.eval.wrappers.RGBDFullResWrapper'
PRECONFIGURED_WRAPPER = 'shared_camera_config.wrap_preconfigured'


def resolutions(profile='full_v1'):
    if not isinstance(profile, str) or profile not in RESOLUTION_PROFILES:
        raise ValueError('Unknown preregistered camera resolution profile')
    return RESOLUTION_PROFILES[profile].copy()


def camera_specs(robot_name, eval_config, resolution_profile='full_v1'):
    sizes = resolutions(resolution_profile)
    roles = eval_config.get('camera_sensor_names', {})
    if not isinstance(robot_name, str) or not robot_name or set(roles) != set(RESOLUTIONS):
        raise ValueError('Exact three named evaluation camera roles required')
    specs = {}
    for role, size in sizes.items():
        name = roles[role]
        prefix = robot_name + ':'
        if (not isinstance(name, str) or not name.startswith(prefix) or
                not re.fullmatch(r'[A-Za-z0-9_]+:Camera:0', name[len(prefix):])):
            raise ValueError('Camera must be a named link of this robot')
        specs[role] = {'name': name, 'config_key': name[len(prefix):], 'height': size, 'width': size}
    if len({s['name'] for s in specs.values()}) != 3:
        raise ValueError('Evaluation camera roles cannot alias each other')
    return specs


def _digest(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True, allow_nan=False).encode()).hexdigest()


def prepare_config(config, resolution_profile='full_v1'):
    """Copy a resolved evaluator config; change only three sensor overrides and wrapper.

    A per-link sensor entry replaces, rather than merges with, the class entry
    in this frozen OG version. Copy the entire class entry before overriding
    resolution/modalities so noise, enabled state and other camera kwargs are
    not accidentally reset to create_sensor defaults.
    """
    if not isinstance(config, dict) or config.get('env_wrapper') != {'_target_': ORIGINAL_WRAPPER}:
        raise ValueError('Expected original RGB-D wrapper config')
    robot = config.get('robot', {})
    if robot.get('model') != 'r1pro' or robot.get('obs_modalities') != ['proprio', 'rgb']:
        raise ValueError('Frozen R1Pro RGB/proprio configuration required')
    specs = camera_specs(robot.get('name'), robot.get('eval', {}), resolution_profile)
    sensors = robot.get('sensor_config', {})
    if set(sensors) != {'VisionSensor'}:
        raise ValueError('Unexpected existing per-link or other sensor override')
    original = sensors['VisionSensor']
    kwargs = original.get('sensor_kwargs', {})
    if (original.get('enabled', True) is not True or 'modalities' in original or
            any(type(kwargs.get(k)) is not int or kwargs[k] != 1080 for k in ('image_height', 'image_width'))):
        raise ValueError('Expected original enabled 1080-square camera defaults')
    modified = deepcopy(config)
    for spec in specs.values():
        entry = deepcopy(original)
        entry['modalities'] = ['rgb', 'depth_linear']
        entry['sensor_kwargs']['image_height'] = spec['height']
        entry['sensor_kwargs']['image_width'] = spec['width']
        modified['robot']['sensor_config'][spec['config_key']] = entry
    modified['env_wrapper'] = {'_target_': PRECONFIGURED_WRAPPER}
    if resolution_profile != 'full_v1':
        modified['env_wrapper']['resolution_profile'] = resolution_profile
    # Assert the complete config delta, not only selected physics fields.
    restored = deepcopy(modified)
    for spec in specs.values():
        del restored['robot']['sensor_config'][spec['config_key']]
    restored['env_wrapper'] = deepcopy(config['env_wrapper'])
    if restored != config:
        raise ValueError('Non-camera evaluator configuration was changed')
    receipt = {'original_config_sha256': _digest(config), 'effective_config_sha256': _digest(modified),
               'cameras': specs, 'resolution_profile': resolution_profile,
               'original_camera_pixels': 3 * 1080 * 1080,
               'configured_camera_pixels': sum(s['height'] * s['width'] for s in specs.values()),
               'changed_paths': ['env_wrapper'] + ['robot.sensor_config.' + s['config_key'] for s in specs.values()]}
    return modified, receipt


def inspect_cameras(env, resolution_profile='full_v1'):
    """Read only robot camera identities and configuration, never object/scene state."""
    if len(env.robots) != 1:
        raise ValueError('One original evaluation robot required')
    robot = env.robots[0]
    specs = camera_specs(robot.name, getattr(env, '_eval_robot_config', {}), resolution_profile)
    visual = {name for name, sensor in robot.sensors.items()
              if hasattr(sensor, 'image_height') or hasattr(sensor, 'image_width')}
    if visual != {s['name'] for s in specs.values()}:
        raise ValueError('Missing or unexpected visual sensor; no cameras are removed to fit memory')
    rows, references = {}, {}
    for role, spec in specs.items():
        sensor = robot.sensors[spec['name']]
        initial = sensor._load_config
        expected = (spec['height'], spec['width'])
        actual = (sensor.image_height, sensor.image_width)
        loaded = (initial.get('image_height'), initial.get('image_width'))
        if (sensor.name != spec['name'] or any(type(x) is not int for x in (*actual, *loaded)) or
                actual != expected or loaded != expected or set(sensor.modalities) != {'rgb', 'depth_linear'} or
                sensor._render_product is None):
            raise ValueError('Camera was not created at the final RGB-D configuration: ' + role)
        rows[role] = {**spec, 'initial_height': loaded[0], 'initial_width': loaded[1],
                      'modalities': ['rgb', 'depth_linear']}
        references[role] = (sensor, sensor._render_product)
    return rows, references


def wrap_preconfigured(env, resolution_profile='full_v1'):
    from omnigibson.envs.env_wrapper import EnvironmentWrapper
    rows, references = inspect_cameras(env, resolution_profile)
    wrapper = EnvironmentWrapper(env=env)
    wrapper._behavior_camera_initialization = rows
    wrapper._behavior_camera_references = references
    wrapper._behavior_camera_resolution_profile = resolution_profile
    # The already installed chunk patch reloads after original Evaluator.__init__
    # initializes articulation handles. Never reload here or call camera setters.
    wrapper._behavior_deferred_space_reload = True
    return wrapper


def validate_wrapper(wrapper, expected):
    from omnigibson.envs.env_wrapper import EnvironmentWrapper
    if type(wrapper) is not EnvironmentWrapper or vars(wrapper).get('_behavior_deferred_space_reload') is not True:
        raise ValueError('Expected preconfigured wrapper and original deferred-space contract')
    profile = vars(wrapper).get('_behavior_camera_resolution_profile')
    resolutions(profile)  # A missing/unknown binding cannot silently default.
    rows, references = inspect_cameras(wrapper.env, profile)
    if set(expected) != set(rows):
        raise ValueError('Expected camera roles do not match registered configuration')
    if vars(wrapper).get('_behavior_camera_initialization') != rows:
        raise ValueError('Camera initialization receipt missing or changed')
    original = vars(wrapper).get('_behavior_camera_references', {})
    if set(original) != set(references):
        raise ValueError('Camera/render-product identity receipt missing')
    for role, (sensor, product) in references.items():
        if original[role][0] is not sensor or original[role][1] is not product:
            raise ValueError('A live camera or render product was replaced: ' + role)
        if any(rows[role][k] != v for k, v in expected[role].items()):
            raise ValueError('Final cameras differ from the preregistered config')
    return rows


def camera_intrinsics(sensors, as_array, resolution_profile):
    """Actual robot-camera calibration, not guessed rescaling or scene truth."""
    import numpy as np
    sizes = resolutions(resolution_profile)
    if set(sensors) != set(sizes):
        raise ValueError('Three actual onboard cameras required for intrinsics')
    result = {}
    for view, size in sizes.items():
        camera = sensors[view]
        matrix = np.asarray(as_array(camera.intrinsic_matrix), dtype=float)
        if (camera.image_height != size or camera.image_width != size or
                matrix.shape != (3, 3) or not np.isfinite(matrix).all() or
                matrix[0, 0] <= 0 or matrix[1, 1] <= 0 or
                not np.allclose(matrix[2], [0, 0, 1], rtol=0, atol=1e-6) or
                not (0 <= matrix[0, 2] < size and 0 <= matrix[1, 2] < size)):
            raise ValueError('Invalid actual camera intrinsics or resolution: ' + view)
        result[view] = {'K': matrix.tolist(), 'width': size, 'height': size,
                        'source': 'robot_camera_intrinsic_matrix', 'scene_truth': False}
    return result
