"""Explicit RTX4090 profile: same physics/cameras/renderer, relocated dependencies.

Only new probe processes opt in. Existing A100 profiles and running sources
are untouched; every relocated runtime dependency still needs its pinned SHA.
"""
from contextlib import contextmanager
import inspect
import json
import os
from pathlib import Path
import sys

import native_full_profile as original_profile
from native_rl_profile import traced_imports
from rtx_paths import ROOT, SDK, ENV, ADAPTER, TEMPLATE, TASKS, RUNTIME, host_guard
from common import sha


def relocate(path, mappings, special):
    path = Path(path)
    if path in special:
        return special[path]
    for old, new in mappings:
        if path.is_relative_to(old):
            return new/path.relative_to(old)
    raise ValueError('Unregistered runtime dependency path: '+str(path))


def bind_dependencies():
    host_guard()
    helpers = original_profile.checked_helpers()
    helpers['probe_scene_compatible_cameras'].configure_profile()
    scene = helpers['probe_scene_startup']; base = scene.supervisor
    old_env, old_sdk = base.PYTHON.parent.parent, scene.OG.parent
    old_adapter, old_eval, old_window = scene.ADAPTER, scene.EVAL_ROOT, scene.WINDOW
    mappings = [(old_env, ENV), (old_sdk, SDK), (old_adapter, ADAPTER), (old_eval, ADAPTER)]
    special = {old_window: TEMPLATE}
    fixtures = json.loads((ROOT/'fixtures/manifest.json').read_text())
    if sha(TEMPLATE) != fixtures['window_sha256']:
        raise ValueError('Relocated robot/window fixture changed')
    def remap(deps):
        return {relocate(p, mappings, special): fixtures['window_sha256'] if p == old_window else digest
                for p, digest in deps.items()}
    base.DEPENDENCIES = remap(base.DEPENDENCIES)
    scene.EXTRA_DEPENDENCIES = remap(scene.EXTRA_DEPENDENCIES)
    for name in ('PYTHON', 'ISAAC', 'EXPERIENCE', 'OG_KIT', 'APP_SOURCE'):
        setattr(base, name, relocate(getattr(base, name), mappings, special))
    for name in ('OG', 'OG_SOURCE', 'ADAPTER', 'FACTORY', 'EVAL_ROOT', 'ROBOT_CONFIG',
                 'WINDOW', 'ICON', 'INSTALLED_ICON'):
        setattr(scene, name, relocate(getattr(scene, name), mappings, special))
    return scene, base


def configure(output, runtime):
    host_guard()
    output, runtime = Path(output).resolve(), Path(runtime).resolve()
    if (not output.is_dir() or not runtime.is_dir()
            or runtime.parent != RUNTIME or runtime.name not in ('worker_0', 'worker_1')):
        raise ValueError('Fresh registered per-worker RTX paths required')
    scene, base = bind_dependencies()
    base.OUTPUT, base.RUNTIME = output, runtime
    base.RUNTIME_SETTINGS = dict(base.RUNTIME_SETTINGS,
        **{'/renderer/activeGpu': 0, '/physics/cudaDevice': 0})
    if any(os.environ.get(k) != str(runtime/v) for k, v in base.ROUTES.items()):
        raise ValueError('RTX private runtime routing mismatch')
    if os.environ.get('OMNIGIBSON_DATA_PATH') != str(ROOT/'datasets'):
        raise ValueError('RTX assets must be isolated from other environments')
    base.identity()
    return scene, base


@contextmanager
def session(factory, window, *, gpu, output, runtime, evaluation_only=False):
    host_guard()
    if type(gpu) is not int or gpu != 0 or evaluation_only or window.official_mode != 'train':
        raise ValueError('Single physical GPU0 / fixed TRAIN speed probe only')
    expected_mode = 'train'
    scene, base = configure(output, runtime)
    # Process-local registration: do not change existing evaluation defaults.
    base.RUNTIME_SETTINGS = dict(base.RUNTIME_SETTINGS,
        **{'/renderer/activeGpu': gpu, '/physics/cudaDevice': gpu})
    imports = factory._lazy_official_imports(og_root=scene.OG, eval_root=scene.EVAL_ROOT, gpu=gpu)
    import omnigibson as og
    import omnigibson.simulator as startup
    import shared_pathtracing as renderer
    import shared_camera_config as cameras
    from shared_og_startup import private_og_startup
    if Path(startup.__file__).resolve() != scene.OG_SOURCE.resolve():
        raise ValueError('Unexpected native simulator source')
    record = dict(profile='rtx4090_full_speed_v1', gpu=gpu, instance=window.instance_id,
                  split=expected_mode, evaluation_only=evaluation_only,
                  source_commit=base.identity(), shared_install_writes=0,
                  renderer='PathTracing', resolution_profile='full_v1')
    def write(row): base.write('native_rl_profile.json', row)
    scene.configure_viewer_before_launch(imports.gm, og, record)
    imports = traced_imports(imports, record, instance_id=window.instance_id, write=write)
    owned_app = None

    def construct():
        nonlocal owned_app
        from isaacsim import SimulationApp
        if Path(inspect.getfile(SimulationApp)).resolve() != base.APP_SOURCE.resolve():
            raise ValueError('Unexpected SimulationApp source')
        cfg = base.app_configuration()
        cfg.update(active_gpu=gpu, physics_gpu=gpu)
        argv = sys.argv
        try:
            sys.argv = [argv[0], '--portable-root', str(base.RUNTIME/'portable')]
            owned_app = SimulationApp(cfg, experience=str(base.EXPERIENCE))
        finally: sys.argv = argv
        settings = renderer.get_settings(); renderer.apply_settings(settings)
        actual = {key: settings.get(key) for key in base.RUNTIME_SETTINGS}
        base.validate_settings(actual)
        record['actual_settings'] = actual; write(record)
        return owned_app

    def validate(env):
        scene.validate_viewer_after_scene(imports.gm, og.sim, record)
        renderer.validate_after_scene(og, record)
        cameras.validate_wrapper(env.evaluator.env, record['camera_configuration']['cameras'])
        if (record['load_count'] != 1 or record['reset_count'] < 1 or
                any(e['status'] != 'completed' for e in record['events'])):
            raise ValueError('Incomplete original reset/load')
        record['phase'] = 'validated'; write(record)

    class Checked:
        def __init__(self, env): self.original = env
        def __getattr__(self, key): return getattr(self.original, key)
        def reset(self):
            result = self.original.reset(); validate(self.original); return result

    bindings = {(base.OG_KIT, base.EXPERIENCE): base.DEPENDENCIES[base.OG_KIT],
                (scene.ICON, scene.INSTALLED_ICON): scene.ICON_SHA}
    try:
        with renderer.before_scene(og, startup, source_sha256=scene.EXTRA_DEPENDENCIES[scene.OG_SOURCE],
                record=record, write=lambda _,row: write(row), trace_write=base.write), private_og_startup(
                startup, source_sha256=scene.EXTRA_DEPENDENCIES[scene.OG_SOURCE], experience=base.EXPERIENCE,
                copy_bindings=bindings, construct=construct, gpu=gpu):
            with factory.OfficialEvaluatorSession(window, gpu=gpu, imports=imports, tasks_path=TASKS) as env:
                validate(env)
                yield Checked(env)
                validate(env)
    finally:
        if owned_app is not None and owned_app.is_running(): owned_app.close()
