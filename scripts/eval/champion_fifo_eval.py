"""Compose FIFO admission with the original registered champion evaluator.

The original HERE/REPO, worker code, model service, simulator entry, config and
preflights stay unchanged. Only the loader dependency and supervisor's child
entry are redirected. The original module uses __file__ after import solely to
spawn its worker dispatcher; it computes all actor paths before this binding.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from loader_admission import loader_lock

REPO = Path(__file__).resolve().parents[2]
RUNTIME_ENV = 'BEHAVIOR_CHAMPION_FIFO_RUNTIME'
COMMIT_ENV = 'BEHAVIOR_CHAMPION_FIFO_COMMIT'


def frozen(path, commit):
    if (subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=path, text=True).strip() != commit
            or subprocess.check_output(['git', 'status', '--porcelain'], cwd=path, text=True).strip()):
        raise ValueError('Exact clean frozen coordinator/runtime required')


def bind_admission(module, source):
    source = Path(source).resolve()
    entry = source / 'scripts/eval/champion_full_eval.py'
    if (module.REPO != source or module.HERE != entry.parent or Path(module.__file__).resolve() != entry):
        raise ValueError('Original registered actor paths changed')
    before = dict(actor_repo=str(module.REPO), actor_eval_dir=str(module.HERE), original_dispatcher=str(entry))
    module.loader_lock = loader_lock
    module.__file__ = str(Path(__file__).resolve())
    return before


def load_runtime(job, source, commit):
    source, job = Path(source).resolve(), Path(job).resolve()
    frozen(REPO, commit)
    manifest = json.loads((job / 'manifest.json').read_text())
    frozen(source, manifest['source_commit'])
    entry = source / 'scripts/eval/champion_full_eval.py'
    spec = importlib.util.spec_from_file_location('champion_registered_runtime', entry)
    module = importlib.util.module_from_spec(spec); sys.modules[spec.name] = module; spec.loader.exec_module(module)
    module.registration(job)  # original source/config/export/terminal gates, not new model registration
    paths = bind_admission(module, source)
    receipt = dict(schema='champion_fifo_orchestration_binding_v1', **paths, coordinator_source=str(REPO),
        coordinator_commit=commit, runtime_commit=manifest['source_commit'], manifest_sha256=hashlib.sha256((job/'manifest.json').read_bytes()).hexdigest(),
        wrapper=str(Path(__file__).resolve()), original_dispatcher_sha256=hashlib.sha256(entry.read_bytes()).hexdigest(),
        loader_sha256=hashlib.sha256((REPO/'scripts/eval/loader_admission.py').read_bytes()).hexdigest(),
        actor_changed=False, physics_changed=False, preflights_reused=True,
        admission='FIFO two slots; queue waiting is not active loading or official rollout time')
    return module, receipt


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=('inspect', 'worker', 'supervise', 'report'))
    p.add_argument('--job', type=Path, required=True); p.add_argument('--gpu', type=int)
    p.add_argument('--num-envs', type=int, choices=(2,4)); p.add_argument('--preflight', action='store_true')
    p.add_argument('--final', action='store_true')
    p.add_argument('--runtime-source', type=Path, default=os.environ.get(RUNTIME_ENV))
    p.add_argument('--coordinator-commit', default=os.environ.get(COMMIT_ENV))
    args = p.parse_args()
    if not args.runtime_source or not args.coordinator_commit or args.preflight:
        raise ValueError('Explicit frozen source/commit and existing preflights required; no new TRAIN reroll')
    module, receipt = load_runtime(args.job, args.runtime_source, args.coordinator_commit)
    if args.command == 'inspect':
        print(json.dumps(receipt)); return
    # The original supervisor's worker argument list stays identical; children
    # inherit these explicit, non-secret frozen source bindings.
    os.environ[RUNTIME_ENV] = str(args.runtime_source.resolve()); os.environ[COMMIT_ENV] = args.coordinator_commit
    if args.command in ('worker','supervise'):
        path = args.job / 'coordinator-receipts'; path.mkdir(exist_ok=True)
        name = f'worker-gpu{args.gpu}' if args.command == 'worker' else 'supervisor'
        with (path / (name + '.json')).open('x') as stream:
            json.dump(receipt | dict(pid=os.getpid(), started=time.time(), command=args.command), stream, indent=2)
    if args.command == 'worker': module.worker(args.job,args.gpu,args.num_envs,False)
    elif args.command == 'supervise': module.supervise(args.job)
    else: print(json.dumps(module.report(args.job,final=args.final)))


if __name__ == '__main__': main()
