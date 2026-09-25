"""Process-private, pre-scene OptiX rendering profile for the bounded H54 probe.

Keep the official simulator constructor, physics setup, scene and sensors. The
public launch wrapper only selects renderer settings after the original empty
simulator returns, before Environment starts loading its scene or robot.
"""
from contextlib import contextmanager, ExitStack
import hashlib
import json
from pathlib import Path
from threading import Lock, RLock
import time
import traceback

INSTALLED_DEPENDENCIES = {
    Path('/mnt/sdc1/xhz/miniconda3/envs/behavior/lib/python3.11/site-packages/isaacsim/extscache/'
         'omni.replicator.core-1.12.27+107.3.3.lx64.r.cp311/omni/replicator/core/scripts/settings.py'):
        'e4901268a16048062fe5c33bb3507752a907cb17fa7f4b6797835d8687816006',
}

SETTINGS = {
    '/rtx/rendermode': 'PathTracing',
    # Replicator's own non-DLSS AA selection clears this restriction first.
    # H71 tests this omitted prerequisite against H70's first-camera aa/op=3.
    # Configure before any sensor exists; never repair settings per frame.
    '/rtx-transient/post/aa/limitedOps': False,
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


def read_checked(settings, record=None):
    actual = {key: settings.get(key) for key in SETTINGS}
    differences = {key: {'expected': value, 'actual': actual[key],
                        'actual_type': type(actual[key]).__name__}
                   for key, value in SETTINGS.items()
                   if type(actual[key]) is not type(value) or actual[key] != value}
    if record is not None:
        # Keep the failing values too. H69 previously only saved the earlier
        # passing snapshot, so its exception could not identify the conflict.
        record['pathtracing_actual_settings'] = actual
        record['pathtracing_differences'] = differences
        record['pathtracing_profile_verified'] = not differences
    if differences:
        raise ValueError('Registered PathTracing/OptiX settings changed: ' +
                         json.dumps(differences, sort_keys=True))
    return actual


@contextmanager
def observe_changes(settings, record, write):
    """Read-only Carb notifications after the empty-simulator boundary.

    No corrective setters, per-frame polling or installed-source modification.
    A bounded trace is persisted immediately, before native shutdown can exit
    Python. Python stacks identify Python callers, not a native C++ backtrace.
    """
    receipt = {'initial': read_checked(settings), 'events': [], 'notifications': 0,
               'dropped': 0, 'errors': [], 'settings_writes': 0, 'max_events': 64}
    record['pathtracing_change_trace'] = receipt
    lock, subscriptions = RLock(), []
    started = time.monotonic()

    def changed(key):
        def callback(_item, event_type):
            with lock:
                receipt['notifications'] += 1
                try:
                    if len(receipt['events']) >= receipt['max_events']:
                        receipt['dropped'] += 1
                        return
                    receipt['events'].append({
                        'key': key, 'event_type': str(event_type),
                        'elapsed_seconds': time.monotonic() - started,
                        'actual': {k: settings.get(k) for k in SETTINGS},
                        'python_stack': traceback.format_stack(limit=20)[:-1],
                    })
                    write('pathtracing_changes.json', receipt)
                except BaseException as error:
                    # Carb may swallow callback exceptions; retain them and
                    # fail at the validation boundary, never silently pass.
                    if len(receipt['errors']) < 8:
                        receipt['errors'].append(repr(error))
        return callback

    try:
        for key in SETTINGS:
            subscriptions.append(settings.subscribe_to_node_change_events(key, changed(key)))
        write('pathtracing_changes.json', receipt)
        yield receipt
    finally:
        for subscription in subscriptions:
            settings.unsubscribe_to_change_events(subscription)
        write('pathtracing_changes.json', receipt)


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
    read_checked(get_settings(), record)
    trace = record.get('pathtracing_change_trace', {})
    if trace.get('errors') or trace.get('dropped', 0):
        record['pathtracing_profile_verified'] = False
        raise ValueError('Renderer change trace incomplete')
    if any(type(event['actual'].get(key)) is not type(value) or event['actual'][key] != value
           for event in trace.get('events', []) for key, value in SETTINGS.items()):
        record['pathtracing_profile_verified'] = False
        raise ValueError('Renderer settings changed during the observed live session')


@contextmanager
def before_scene(og, simulator, *, source_sha256, record, write, trace_write=None):
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
    observers = ExitStack()
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
        receipt['original_aa_limited_ops'] = settings.get('/rtx-transient/post/aa/limitedOps')
        # H54 observed the native stack returning the known legacy mode after
        # OG's RT2 assignment and play/stop updates. That mutable setting is not
        # an identity proof. Source/alias/empty-scene checks above stay strict;
        # the resulting PT settings below and after capture must still match.
        if receipt['original_render_mode'] not in ('RealTimePathTracing', 'RaytracedLighting'):
            raise ValueError('Unexpected pre-scene native renderer mode')
        receipt['actual_settings_before_scene'] = apply_settings(settings)
        receipt['applied_before_scene'] = True
        write('worker.json', record)
        if trace_write is not None:
            observers.enter_context(observe_changes(settings, record, trace_write))
        return result

    og.launch = launch
    try:
        yield receipt
    finally:
        try:
            observers.close()
        finally:
            with _GATE:
                active = False
                _ACTIVE.remove(source)
            og.launch = original
        # Do not restore renderer settings after live cameras exist. This owned
        # process closes its app; no installed file or another process changed.
