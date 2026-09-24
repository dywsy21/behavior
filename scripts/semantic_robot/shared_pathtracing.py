"""Process-private, pre-scene OptiX rendering profile for the bounded H54 probe.

Keep the official simulator constructor, physics setup, scene and sensors. The
public launch wrapper only selects renderer settings after the original empty
simulator returns, before Environment starts loading its scene or robot.
"""
from contextlib import contextmanager
import hashlib
from pathlib import Path
from threading import Lock


SETTINGS = {
    '/rtx/rendermode': 'PathTracing',
    '/rtx/post/aa/op': 0,
    '/rtx/pathtracing/dlss/enabled': False,
    '/rtx/pathtracing/spp': 4,
    '/rtx/pathtracing/totalSpp': 16,
    '/rtx/pathtracing/maxBounces': 4,
    '/rtx/pathtracing/maxSpecularAndTransmissionBounces': 6,
    '/rtx/pathtracing/optixDenoiser/enabled': True,
    '/rtx/pathtracing/optixDenoiser/temporalMode/enabled': False,
    '/rtx/pathtracing/optixDenoiser/blendFactor': 0.0,
}
APP_CONFIG = {'renderer': 'PathTracing', 'anti_aliasing': 0, 'denoiser': True,
              'samples_per_pixel_per_frame': 4, 'max_bounces': 4,
              'max_specular_transmission_bounces': 6}
_GATE = Lock()
_ACTIVE, _CONSUMED = set(), set()


def get_settings():
    # Imported only inside the already constructed, owned simulator process.
    import carb.settings
    return carb.settings.get_settings()


def read_checked(settings):
    actual = {key: settings.get(key) for key in SETTINGS}
    if any(type(actual[key]) is not type(value) or actual[key] != value
           for key, value in SETTINGS.items()):
        raise ValueError('Registered PathTracing/OptiX settings changed')
    return actual


def apply_settings(settings):
    # SimulationApp resets totalSpp to its per-frame samples during construction;
    # the CLI flags alone are not sufficient. Only call at the two reviewed
    # empty-app / empty-simulator boundaries, never after sensors are live.
    for key, value in SETTINGS.items():
        settings.set(key, value)
    return read_checked(settings)


def validate_after_scene(og, record):
    receipt = record.get('pathtracing', {})
    if (receipt.get('launch_calls') != 1 or receipt.get('applied_before_scene') is not True or
            og.sim is None or og.sim.viewer_camera is not None):
        raise ValueError('PathTracing was not applied at the registered empty-simulator boundary')
    record['pathtracing_actual_settings'] = read_checked(get_settings())
    record['pathtracing_profile_verified'] = True


@contextmanager
def before_scene(og, simulator, *, source_sha256, record, write):
    """One public launch attempt, no hot sensor changes, restore only wrapper."""
    source = Path(simulator.__file__).resolve()
    if hashlib.sha256(source.read_bytes()).hexdigest() != source_sha256:
        raise ValueError('Installed simulator source changed')
    original = og.launch
    if (original is not simulator._launch_simulator or og.app is not None or og.sim is not None):
        raise ValueError('Fresh original public OG launch route required')
    with _GATE:
        if source in _ACTIVE or source in _CONSUMED:
            raise RuntimeError('Rendering profile launch already reserved or attempted')
        _ACTIVE.add(source)
    active = True
    receipt = {'launch_calls': 0, 'applied_before_scene': False,
               'original_simulator_sha256': source_sha256, 'renderer': 'PathTracing',
               'image_distribution_changed': True, 'shared_install_writes': 0}
    record['pathtracing'] = receipt

    def launch(*args, **kwargs):
        with _GATE:
            if not active or source in _CONSUMED:
                raise RuntimeError('Only one active renderer-profile launch attempt allowed')
            _CONSUMED.add(source)
            receipt['launch_calls'] += 1
        write('worker.json', record)
        result = original(*args, **kwargs)
        if (result is None or result is not og.sim or og.app is None or
                len(result.scenes) != 0 or result.viewer_camera is not None):
            raise ValueError('Rendering profile cannot modify an occupied simulator or live camera')
        settings = get_settings()
        receipt['original_render_mode'] = settings.get('/rtx/rendermode')
        # H54 observed the native stack returning the known legacy mode after
        # OG's RT2 assignment and play/stop updates. That mutable setting is not
        # an identity proof. Source/alias/empty-scene checks above stay strict;
        # the resulting PT settings below and after capture must still match.
        if receipt['original_render_mode'] not in ('RealTimePathTracing', 'RaytracedLighting'):
            raise ValueError('Unexpected pre-scene native renderer mode')
        receipt['actual_settings_before_scene'] = apply_settings(settings)
        receipt['applied_before_scene'] = True
        write('worker.json', record)
        return result

    og.launch = launch
    try:
        yield receipt
    finally:
        with _GATE:
            active = False
            _ACTIVE.remove(source)
        og.launch = original
        # Do not restore renderer settings after live cameras exist. This owned
        # process closes its app; no installed file or another process changed.
