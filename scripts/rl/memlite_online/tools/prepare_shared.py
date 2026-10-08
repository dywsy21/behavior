"""Seal the existing TRAIN-only partition and a fresh shared-policy run."""
# ruff: noqa: E402 -- pin dependency paths before importing runtime modules.
import argparse
import hashlib
import json
import random
from pathlib import Path
import subprocess
import time

from bootstrap import bootstrap
SOURCE = bootstrap()
from checkpoint_io import atomic_json
from synchronous import task_weights

ROOT = Path('/run/ti/rl_memlite_stage1_20261006')


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--job', type=Path, required=True)
    parser.add_argument('--hours', type=float, default=24.)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--world-size', type=int, default=8)
    parser.add_argument('--port-base', type=int, default=19800)
    parser.add_argument('--distributed-port', type=int, default=29880)
    args = parser.parse_args()
    if args.job.exists() or not 0 < args.hours <= 24 or args.world_size != 8:
        raise ValueError('Fresh 8-rank run required, within the declared 24h cap')
    old_path = ROOT / 'runs/large_scale73h_trainonly_20261007/manifest.json'
    old = json.loads(old_path.read_text())
    groups = sorted(old['groups'], key=lambda group: group['gpu'])
    if len(groups) != 8:
        raise ValueError('Expected existing 8-way TRAIN split')
    tasks = [task for group in groups for task in group['tasks']]
    if len(tasks) != 100 or {t['task_index'] for t in tasks} != set(range(100)):
        raise ValueError('Expected 100 unique official tasks')
    for task in tasks:
        train = set(task['train_instances'])
        excluded = set(task['sft_heldout_excluded']) | set(task['development_instances'])
        if len(train) < 2 or train & excluded or not all(1 <= i <= 300 for i in train):
            raise ValueError('TRAIN/heldout/public partition violation: ' + task['task'])
    for group in groups:
        # No all-short-tasks-first bias. Fixed shuffled round robin gives every
        # task one episode pair per sweep; horizons enter the loss weighting.
        random.Random(20261008 + group['gpu']).shuffle(group['tasks'])
    model = old['model']
    for relative, info in model['checkpoints'].items():
        path = ROOT / 'models/stage1' / relative
        if path.stat().st_size != info['bytes'] or sha256(path) != info['sha256']:
            raise ValueError('SFT checkpoint identity changed')
    repo = SOURCE.parents[2]
    commit = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
    if subprocess.check_output(['git', '-C', str(repo), 'status', '--porcelain'], text=True).strip():
        raise ValueError('Source must be a clean frozen checkout')
    if args.resume and not args.resume.is_file():
        raise ValueError('Missing shared resume checkpoint')
    started = time.time()
    manifest = dict(kind='memlite_shared_rl100_v1', created=started, source_commit=commit,
        source=str(SOURCE), source_partition=str(old_path), source_partition_sha256=sha256(old_path),
        model=model, groups=groups, world_size=8, task_weights=task_weights(groups),
        weighting='inverse_nominal_chunk_occupancy; report actual occupancy/early success separately',
        training=dict(chunks_per_rank=64, num_envs=2, actor_lr=1e-7, critic_lr=1e-4,
            target_kl=0.02, max_clip_fraction=0.25, update_epochs=1, microbatch_size=1,
            transition_std=0.02, checkpoint_interval=10, acceptance_pause_updates=2,
            high_batch=True, max_updates=3000, control_gamma=0.99999,
            high_frozen=True, low_vlm_frozen=True, reward_protocol='shared_terminal_q_v1'),
        hours=args.hours, absolute_deadline=None, port_base=args.port_base,
        distributed_endpoint=f'tcp://127.0.0.1:{args.distributed_port}',
        resume_checkpoint=str(args.resume.resolve()) if args.resume else None,
        resume_sha256=sha256(args.resume) if args.resume else None,
        evaluation_enabled=False, recovery_bc_eligible=False,
        resume_environment_rule='reset_all_envs_discard_uncommitted_rollouts',
        startup='24h clock starts at supervisor start, not preparation',
        disk_reserve_gib=150)
    if args.resume:
        import torch
        payload = torch.load(args.resume, map_location='cpu', mmap=True, weights_only=False)
        shared = payload.get('distributed_state')
        if not shared or shared['world_size'] != 8 or shared['task_weight_map'] != manifest['task_weights']:
            raise ValueError('Not a compatible shared-policy checkpoint')
        manifest['resume_cursors'] = {}
        for rank, saved in enumerate(shared['rank_states']):
            episodes = saved['episodes']
            if not episodes or any(e['cycle'] != episodes[0]['cycle'] or e['gpu'] != rank for e in episodes):
                raise ValueError('Unbound checkpoint collector cursor')
            manifest['resume_cursors'][str(rank)] = episodes[0]['cycle']
        # Explicit fresh reset, not a claim of restoring simulator state.
        manifest['resume_seed_offset'] = 10000000 + int(payload['updates']) * 1000
        manifest['resume_update'] = int(payload['updates'])
        manifest['training']['acceptance_pause_updates'] = int(payload['updates']) + 2
    args.job.mkdir(parents=True)
    atomic_json(args.job / 'manifest.prepared.json', manifest)
    print(json.dumps(dict(prepared=str(args.job), commit=commit, tasks=len(tasks),
        task_weight_min=min(manifest['task_weights'].values()),
        task_weight_max=max(manifest['task_weights'].values()))))


if __name__ == '__main__':
    main()
