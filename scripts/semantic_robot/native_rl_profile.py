"""Opt-in persistent TRAIN sessions on explicitly assigned physical A100s.

Default one-shot evaluation remains unchanged. Reset replays the official
initial scene; intermediate starts are real controls, not pose teleportation.
"""
from contextlib import contextmanager
from dataclasses import replace
import inspect
import os
from pathlib import Path
import sys

import native_full_profile as original_profile


def traced_imports(imports, record, *, instance_id, write):
    import shared_camera_config as cameras
    original = imports.Evaluator
    record.update(events=[], reset_count=0, load_count=0)
    constructed = False

    class Evaluator:
        def __init__(self, env): self.original = env
        def __getattr__(self, key): return getattr(self.original, key)
        def call(self, name, instance=None):
            events = record['events']
            if events and events[-1]['status'] != 'completed':
                raise RuntimeError('Previous reset/load did not complete')
            expected = [('reset', None), ('load', instance_id)]
            if len(events) < 2:
                if (name, instance) != expected[len(events)]:
                    raise ValueError('Official initial reset/load order changed')
            elif name != 'reset' or instance is not None:
                raise ValueError('Persistent session permits only reset of its fixed TRAIN instance')
            event = dict(name=name, instance=instance, status='started')
            events.append(event); write(record)
            result = self.original.reset() if name == 'reset' else self.original.load_task_instance(instance)
            event['status'] = 'completed'
            record[name+'_count'] += 1; write(record)
            return result
        def reset(self): return self.call('reset')
        def load_task_instance(self, instance): return self.call('load', instance)

    def construct(cfg):
        nonlocal constructed
        if constructed: raise ValueError('Only one evaluator per process')
        constructed = True
        plain = imports.OmegaConf.to_container(cfg, resolve=True)
        modified, receipt = cameras.prepare_config(plain, 'full_v1')
        record['camera_configuration'] = receipt; write(record)
        env = original(imports.OmegaConf.create(modified))
        cameras.validate_wrapper(env.env, receipt['cameras'])
        return Evaluator(env)
    return replace(imports, Evaluator=construct)


@contextmanager
def session(factory, window, *, gpu, output, runtime, evaluation_only=False):
    if type(gpu) is not int or gpu not in (2,3) or os.environ.get('CUDA_VISIBLE_DEVICES'):
        raise ValueError('RL simulators require unremapped physical GPU2 or GPU3')
    expected_mode='public_test' if evaluation_only else 'train'
    if window.official_mode != expected_mode:
        raise ValueError('Persistent session split differs from explicit role')
    scene, base = original_profile.configure(output, runtime)
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
    record = dict(profile='a100_full_rl_v1', gpu=gpu, instance=window.instance_id,
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
            with factory.OfficialEvaluatorSession(window, gpu=gpu, imports=imports) as env:
                validate(env)
                yield Checked(env)
                validate(env)
    finally:
        if owned_app is not None and owned_app.is_running(): owned_app.close()
