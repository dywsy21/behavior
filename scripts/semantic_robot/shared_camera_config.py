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
ORIGINAL_WRAPPER = 'omnigibson.eval.wrappers.RGBDFullResWrapper'
PRECONFIGURED_WRAPPER = 'shared_camera_config.wrap_preconfigured'


def camera_specs(robot_name, eval_config):
    roles = eval_config.get('camera_sensor_names', {})
    if not isinstance(robot_name, str) or not robot_name or set(roles) != set(RESOLUTIONS):
        raise ValueError('Exact three named evaluation camera roles required')
    specs = {}
    for role, size in RESOLUTIONS.items():
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


def prepare_config(config):
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
    specs = camera_specs(robot.get('name'), robot.get('eval', {}))
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
    # Assert the complete config delta, not only selected physics fields.
    restored = deepcopy(modified)
    for spec in specs.values():
        del restored['robot']['sensor_config'][spec['config_key']]
    restored['env_wrapper'] = deepcopy(config['env_wrapper'])
    if restored != config:
        raise ValueError('Non-camera evaluator configuration was changed')
    receipt = {'original_config_sha256': _digest(config), 'effective_config_sha256': _digest(modified),
               'cameras': specs, 'original_camera_pixels': 3 * 1080 * 1080,
               'configured_camera_pixels': sum(s['height'] * s['width'] for s in specs.values()),
               'changed_paths': ['env_wrapper'] + ['robot.sensor_config.' + s['config_key'] for s in specs.values()]}
    return modified, receipt


def inspect_cameras(env):
    """Read only robot camera identities and configuration, never object/scene state."""
    if len(env.robots) != 1:
        raise ValueError('One original evaluation robot required')
    robot = env.robots[0]
    specs = camera_specs(robot.name, getattr(env, '_eval_robot_config', {}))
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


def wrap_preconfigured(env):
    from omnigibson.envs.env_wrapper import EnvironmentWrapper
    rows, references = inspect_cameras(env)
    wrapper = EnvironmentWrapper(env=env)
    wrapper._behavior_camera_initialization = rows
    wrapper._behavior_camera_references = references
    # The already installed chunk patch reloads after original Evaluator.__init__
    # initializes articulation handles. Never reload here or call camera setters.
    wrapper._behavior_deferred_space_reload = True
    return wrapper


def validate_wrapper(wrapper, expected):
    from omnigibson.envs.env_wrapper import EnvironmentWrapper
    if type(wrapper) is not EnvironmentWrapper or vars(wrapper).get('_behavior_deferred_space_reload') is not True:
        raise ValueError('Expected preconfigured wrapper and original deferred-space contract')
    rows, references = inspect_cameras(wrapper.env)
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
