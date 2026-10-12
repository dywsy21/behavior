"""Guarded champion100 training entry; default Git recipe is NOT approved.

Formal execution requires the user-confirmed recipe to be recorded in Git.
--engineering runs only three updates and labels all outputs non-deployable.
No automatic relaunch, retention deletion, VPN changes, or other-node work.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import shutil
import time

from behavior_champion.runner import StopRequest, run_updates, validate_launch, write_json
from behavior_champion.sampling import digest_json


def check_environment():
    required = dict(CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7', NCCL_P2P_DISABLE='1',
                    NCCL_IB_DISABLE='1', NCCL_SOCKET_IFNAME='bond0')
    if any(os.environ.get(key) != value for key, value in required.items()):
        raise ValueError('Use the explicit lc1 eight-A800 launch environment')
    flags = os.environ.get('XLA_FLAGS', '').split()
    if any(flag not in flags for flag in ('--xla_gpu_use_memcpy_local_p2p=false',
                                         '--xla_gpu_exclude_nondeterministic_ops=true')):
        raise ValueError('Checked HOST transport and deterministic GPU ops are required')
    if subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip():
        raise RuntimeError('Eight GPUs must be idle; do not interrupt any teammate process')
    for key, value in dict(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
            XLA_PYTHON_CLIENT_PREALLOCATE='false', XLA_PYTHON_CLIENT_ALLOCATOR='default',
            OPENBLAS_NUM_THREADS='4', OMP_NUM_THREADS='4').items():
        os.environ[key] = value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('/data/workspace/wsy/behavior2026'))
    parser.add_argument('--recipe', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--credential-file', type=Path, required=True)
    parser.add_argument('--resume-latest', action='store_true')
    parser.add_argument('--engineering', action='store_true')
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[2]
    if args.recipe.resolve() != repo / 'configs/champion2026/shared100_v1.json':
        raise ValueError('Use the registered Git recipe, not an untracked external approval toggle')
    recipe = json.loads(args.recipe.read_text())
    # This check is intentionally BEFORE GPU discovery, W&B, output mutation,
    # source imports, or parent loading. Default invocation cannot start a job.
    validate_launch(recipe, engineering=args.engineering)
    check_environment()
    if args.resume_latest:
        if not args.output.is_dir():
            raise ValueError('Resume requires the original run directory')
        result_file = args.output / 'result.json'
        if result_file.exists() and json.loads(result_file.read_text())['status'] != 'saved_on_signal':
            raise ValueError('Only an interrupted/signal-paused run may resume; completed/time-limited runs cannot extend')
        if (args.output / 'failure.json').exists():
            raise ValueError('A failed run needs diagnosis and an explicit reviewed continuation, not this normal resume')
    else:
        args.output.mkdir(parents=True, exist_ok=False)
    with (args.output / '.run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        started = time.monotonic()
        from behavior_champion.training_engine import TrainingEngine
        from behavior_champion.tracking import OnlineRun
        receipt, identity, prior = None, None, 0.
        if args.resume_latest:
            receipt = json.loads((args.output / 'latest-checkpoint.json').read_text())
            identity = json.loads((args.output / 'wandb-identity.json').read_text())
            prior_contract = json.loads((args.output / 'contract.json').read_text())
            # Include work after the most recent checkpoint: crash/replay must
            # never buy a fresh 24h. Truncated journals fail closed for review.
            events = [json.loads(line) for line in (args.output / 'events.jsonl').read_text().splitlines()]
            prior = max((event['elapsed_seconds'] for event in events), default=0.)
            # Counting downtime conservatively makes process death unable to
            # reset or undercount the approved wall-clock window.
            timing = json.loads((args.output / 'timing.json').read_text())
            prior = max(prior, time.time() - timing['started_unix'])
            if prior >= (1800 if args.engineering else recipe['max_hours'] * 3600):
                raise ValueError('Original wall-clock window exhausted; no implicit extension')
            if (args.output / 'result.json').exists():
                history = args.output / 'resume-history'; history.mkdir(exist_ok=True)
                shutil.copy2(args.output / 'result.json', history / f'result-{time.time_ns()}.json')
        else:
            write_json(args.output / 'timing.json', dict(started_unix=time.time(), downtime_counts=True))
        engine = TrainingEngine(args.root, repo, args.output,
                                recipe, engineering=args.engineering)
        if receipt is not None and engine.contract != prior_contract:
            raise ValueError('Frozen code/recipe/data/runtime contract differs on resume')
        if receipt is None:
            write_json(args.output / 'contract.json', engine.contract)
        tracker = OnlineRun(args.output, credential_file=args.credential_file,
            entity=recipe['wandb_entity'], project=recipe['wandb_project'], name=args.output.name,
            config=engine.contract, resume_id=None if identity is None else identity['run_id'],
            resume_update=None if receipt is None else receipt['completed_updates'])
        write_json(args.output / 'wandb-identity.json', tracker.identity)
        try:
            with StopRequest() as stop:
                engine.initialize(receipt)
                run_updates(engine, tracker, args.output,
                    updates=3 if args.engineering else recipe['updates'],
                    max_seconds=1800 if args.engineering else recipe['max_hours'] * 3600,
                    checkpoint_every=2 if args.engineering else recipe['checkpoint_every'],
                    eval_every=3 if args.engineering else recipe['eval_every'], stop=stop,
                    prior_seconds=prior, started=started)
        except Exception:
            # run_updates records its own failed state. Initialization errors
            # are also terminal; no automated retry or partial checkpoint claim.
            if not (args.output / 'failure.json').exists():
                write_json(args.output / 'failure.json', dict(status='initialization_failed_no_retry',
                    contract_sha256=digest_json(engine.contract)))
                tracker.finish('initialization_failed_no_retry', completed=0 if receipt is None else receipt['completed_updates'], error=True)
            raise


if __name__ == '__main__': main()
