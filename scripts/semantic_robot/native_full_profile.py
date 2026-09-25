"""Opt-in full-resolution A100 profile, using the reviewed H56 adapters.

Only this process owns the application and changes the in-memory launch route.
No shared SDK file, physics configuration, task instance or sensor is hot edited.
"""
from contextlib import contextmanager
import inspect
import importlib
import importlib.util
import os
from pathlib import Path

NAME = 'a100_full_v1'
DEPENDENCY_FILES = (
    'native_full_profile.py', 'launch_h69.py', 'probe_scene_compatible_cameras.py',
    'probe_scene_preconfigured_cameras.py', 'probe_scene_without_viewer.py',
    'probe_scene_startup.py', 'probe_simulator_startup.py',
    'shared_og_startup.py', 'shared_pathtracing.py', 'shared_camera_config.py',
)


def checked_helpers():
    folder = Path(__file__).resolve().parent
    modules = {}
    for filename in DEPENDENCY_FILES:
        if filename in ('native_full_profile.py', 'launch_h69.py'):
            continue
        name = Path(filename).stem
        spec = importlib.util.find_spec(name)
        if spec is None or spec.origin is None or Path(spec.origin).resolve() != folder/filename:
            raise ValueError('Native profile helper import shadowed: ' + name)
        module = importlib.import_module(name)
        if Path(module.__file__).resolve() != folder/filename:
            raise ValueError('Loaded native helper is not the frozen source: ' + name)
        modules[name] = module
    return modules


# Bind the complete local helper graph while the runner script directory is
# still first, before its legacy ADAPTER path is inserted. No GPU imports or
# profile mutation occur. Also recheck at use, not merely hash unused files.
_LOCAL_HELPERS = checked_helpers()


def configure(output, runtime):
    helpers = checked_helpers()
    profile = helpers['probe_scene_compatible_cameras']
    scene = helpers['probe_scene_startup']
    profile.configure_profile()
    base = scene.supervisor
    base.OUTPUT, base.RUNTIME = Path(output).resolve(), Path(runtime).resolve()
    if not base.OUTPUT.is_dir() or not base.RUNTIME.is_dir():
        raise ValueError('Separately reserved output/runtime required')
    if not base.RUNTIME.is_relative_to('/mnt/nvme_tmp/robodojo_sim_runtime_20260925'):
        raise ValueError('Registered private runtime root required')
    if any(os.environ.get(k) != str(base.RUNTIME / v) for k, v in base.ROUTES.items()):
        raise ValueError('Private cache environment does not match runtime')
    base.identity()  # All installed H56 dependencies, interpreter and clean source.
    return scene, base


def prepare_imports(imports, record, *, instance_id):
    import probe_scene_startup as scene
    # Same audited camera preparation plus completed reset/load event tracing.
    return scene.trace_session_imports(imports, record, instance_id=instance_id)


@contextmanager
def session(factory, window, *, gpu, output, runtime):
    if type(gpu) is not int or gpu != 3 or os.environ.get('CUDA_VISIBLE_DEVICES'):
        raise ValueError('A100 profile uses unremapped physical GPU3')
    scene, base = configure(output, runtime)
    imports = factory._lazy_official_imports(og_root=scene.OG, eval_root=scene.EVAL_ROOT, gpu=3)
    checked_helpers()
    import omnigibson as og
    import omnigibson.simulator as startup
    import shared_pathtracing as renderer
    import shared_camera_config as cameras
    from shared_og_startup import private_og_startup
    if Path(startup.__file__).resolve() != scene.OG_SOURCE.resolve():
        raise ValueError('Unexpected native simulator source')
    record = {'profile': NAME, 'phase': 'before_application', 'image_distribution_changed': True,
              'renderer': 'PathTracing', 'resolution_profile': 'full_v1',
              'source_commit': base.identity(), 'shared_install_writes': 0}
    # The borrowed adapters write phase receipts; keep them in a distinct file.
    def save(_name, value):
        base.write('native_profile.json', value)
    scene.configure_viewer_before_launch(imports.gm, og, record)
    imports = prepare_imports(imports, record, instance_id=window.instance_id)
    owned_app = None

    def construct_app():
        nonlocal owned_app
        from isaacsim import SimulationApp
        if Path(inspect.getfile(SimulationApp)).resolve() != base.APP_SOURCE.resolve():
            raise ValueError('Unexpected SimulationApp import')
        record['phase'] = 'constructing_application'; save('', record)
        owned_app = base.construct_app(SimulationApp)
        settings = renderer.get_settings()
        renderer.apply_settings(settings)
        actual = {key: settings.get(key) for key in base.RUNTIME_SETTINGS}
        base.validate_settings(actual)
        record.update(phase='loading_scene', actual_settings=actual); save('', record)
        return owned_app

    def validate(environment, resets):
        scene.validate_viewer_after_scene(imports.gm, og.sim, record)
        renderer.validate_after_scene(og, record)
        cameras.validate_wrapper(environment.evaluator.env, record['camera_configuration']['cameras'])
        if (record['official_api_resets'] != resets or record['load_frozen_instance_calls'] != 1 or
                any(e['status'] != 'completed' for e in record['official_api_events'])):
            raise ValueError('Original session reset/load calls did not complete exactly')
        record['phase'] = 'native_profile_validated'; save('', record)

    class CheckedEnvironment:
        def __init__(self, original): self.original = original
        def __getattr__(self, key): return getattr(self.original, key)
        def reset(self):
            result = self.original.reset()
            validate(self.original, 2)
            return result

    bindings = {(base.OG_KIT, base.EXPERIENCE): base.DEPENDENCIES[base.OG_KIT],
                (scene.ICON, scene.INSTALLED_ICON): scene.ICON_SHA}
    try:
        with renderer.before_scene(og, startup, source_sha256=scene.EXTRA_DEPENDENCIES[scene.OG_SOURCE],
                record=record, write=save), private_og_startup(startup,
                source_sha256=scene.EXTRA_DEPENDENCIES[scene.OG_SOURCE], experience=base.EXPERIENCE,
                copy_bindings=bindings, construct=construct_app) as bridge:
            record['startup_bridge'] = bridge
            with factory.OfficialEvaluatorSession(window, gpu=3, imports=imports) as environment:
                validate(environment, 1)
                yield CheckedEnvironment(environment)
    finally:
        # The external watchdog also bounds native shutdown, which may exit
        # Python directly. Never close a different process/application.
        if owned_app is not None and owned_app.is_running():
            owned_app.close()
