"""CPU preflight/registration only; requires completed isolated preparation."""
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import numpy as np

from common import commit, sha, save
from rtx_paths import ROOT, REPO, SDK, ENV, OUT, RUNTIME, TEMPLATE, host_guard, validate_worker


def dead(pid):
    p = Path(f'/proc/{pid}/stat')
    return not p.exists() or p.read_text().rsplit(')', 1)[1].split()[0] == 'Z'


def main():
    host_guard()
    if OUT.exists() or RUNTIME.exists(): raise ValueError('Do not overwrite a previous speed probe')
    bootstrap = json.loads((ROOT/'runs/bootstrap_v1/status.json').read_text())
    assets = json.loads((ROOT/'runs/assets_v2/status.json').read_text())
    if any(r['status'] != 'completed' or not dead(r['pid']) for r in (bootstrap, assets)):
        raise ValueError('Environment and assets must finish and release before GPU use')
    if shutil.disk_usage(ROOT).free < 200*1024**3: raise ValueError('Disk reserve')
    expected = {'omnigibson':'3.9.1','bddl':'3.7.0','isaacsim':'5.1.0.0','torch':'2.7.0+cu128','numpy':'1.26.0'}
    actual = {p:importlib.metadata.version(p) for p in expected}
    if actual != expected: raise ValueError('Simulator dependency version drift')
    if subprocess.check_output(['git','-C',str(SDK),'rev-parse','HEAD'], text=True).strip() != bootstrap['sdk_commit']:
        raise ValueError('Official SDK source changed')
    inputs = json.loads((REPO/'configs/rl/rtx_speed_inputs.json').read_text())
    for name, metadata in inputs['files'].items():
        p = ROOT/'fixtures'/name
        if p.stat().st_size != metadata['bytes'] or sha(p) != metadata['sha256']:
            raise ValueError('Transferred fixture differs from robo: '+name)
    for name, version in [('behavior-1k-assets','3.9.0'), ('omnigibson-robot-assets','3.8.2')]:
        if (ROOT/'datasets'/name/'VERSION').read_text().strip() != version:
            raise ValueError('Unpacked asset version changed')
    if not (ROOT/'datasets/omnigibson.key').is_file():
        raise ValueError('Existing authorized dataset access configuration missing')
    robot = SDK/'OmniGibson/omnigibson/eval/r1pro.yaml'
    if sha(robot) != 'a98fa8a81472adfecfb642778d4d7a01e20131be7f70824e0ae095d665cdbb93':
        raise ValueError('Official robot config changed')
    window = json.loads((ROOT/'fixtures/window-original.json').read_text())
    window.update(robot_config_path=str(robot), prefix_actions_path=str(ROOT/'fixtures/demo_121.npy'),
                  prefix_actions_sha256=inputs['files']['demo_121.npy']['sha256'])
    def immutable(path, obj):
        if path.exists():
            if json.loads(path.read_text()) != obj: raise ValueError('Prepared fixture changed')
        else: save(path, obj)
    immutable(TEMPLATE, window)
    immutable(ROOT/'fixtures/manifest.json', dict(window_sha256=sha(TEMPLATE), source_inputs=inputs,
        template_purpose='robot/official session construction only; no oracle inputs or policy in speed probe'))
    isaac = ENV/'lib/python3.11/site-packages/isaacsim'
    # Install immutable OG launch resources only into the NEW stopped private env.
    for src, dst in [(SDK/'OmniGibson/omnigibson/omnigibson_5_1_0.kit', isaac/'apps/omnigibson_5_1_0.kit'),
                     (SDK/'docs/assets/OmniGibson_logo.png', isaac/'apps/OmniGibson_logo.png')]:
        if dst.exists():
            if sha(src) != sha(dst): raise ValueError('Refuse to overwrite differing installed resource')
        else:
            with src.open('rb') as reader, dst.open('xb') as writer: shutil.copyfileobj(reader, writer)
    sys.path.insert(0, str(REPO/'scripts/semantic_robot'))
    import rtx_profile
    scene, base = rtx_profile.bind_dependencies()
    source = base.identity()  # all original runtime hashes, relocated, no GPU import
    workers = []
    for spec in inputs['workers']:
        spec = dict(spec, gpu=0, evaluation_only=False, actions=str(ROOT/'fixtures'/f'demo_{spec["episode"]}.npy'))
        validate_worker(spec, evaluation=False)
        controls = np.load(spec['actions'], allow_pickle=False)
        if controls.ndim != 2 or controls.shape[1] != 23 or len(controls) < 128 or not np.isfinite(controls).all():
            raise ValueError('Invalid original benchmark controls')
        workers.append(spec)
    OUT.mkdir(parents=True, exist_ok=False)
    save(OUT/'manifest.json', dict(source_commit=source, entry='rtx_sim_speed', workers=workers,
        max_controls=544, max_active_wall_seconds=3600, cleanup_seconds=40,
        max_human_gate_seconds=300, physical_gpu_count=1, resident_simulators=2,
        includes_model_inference=False, training_updates=0, renderer='PathTracing', camera_profile='full_v1',
        versions=actual, dependencies={str(p):digest for p,digest in base.DEPENDENCIES.items()},
        reference_resets={str(w['instance']):str(ROOT/'fixtures'/f'reference_reset_{w["instance"]}.json') for w in workers},
        task_inputs=inputs, reset_tolerance=1e-5,
        original_a100_reference=dict(serial_seconds=53.312854667, parallel_seconds=28.040516529,
            controls_per_measurement=256, physical_gpu_count=2, cpu_machine_different=True),
        hypothesis='matched full-resolution simulator collection throughput may improve on RTX4090',
        comparison_scope='whole-machine simulator only, not pure GPU or complete RL speed'))
    print(json.dumps(dict(run=str(OUT), source=source, versions=actual, gpu_started=False)))


if __name__ == '__main__': main()
