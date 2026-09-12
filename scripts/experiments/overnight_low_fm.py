"""One bounded A4 weight-warm-start run using the immutable, validated A3 trainer.

This is an experiment recipe, not integration of the A3 model into main. No
model, data, running service, or shared Python installation is edited. Start
from a clean pinned Git worktree on robo. Failed stages are never retried.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from copy import deepcopy
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
from types import SimpleNamespace

WORK = Path('/mnt/sdc1/robodojo/behavior_dev/direct_execution_L6rqZ6_20260910')
SOURCE = WORK / 'a2_lora_history_candidate_v7_samplercoverage'
SOURCE_SHA = '356c717fe4f44d40517a1879e6742f12c9300daa5de13e3a89c71d1fa2fc9281'
PARENT = WORK / 'formal_a3_episodecoverage_5000_v1/checkpoints/step_5000.pt'
PARENT_SHA = '865193f1c8a257d72ea159bd7a46cebf945b0fd24905110b831f0e8a3438e940'
INPUT = WORK / 'a3_samplercoverage_real_processor_v2_microbatch2'
POOLS = Path('/mnt/sdc1/robodojo/behavior_dev/GalaxeaVLA_memlite_coordination_dev_20260908/integration_receipts/formal_a_5000_inputs_v3_20260910')
ROOT = Path('/mnt/sdc1/robodojo/behavior_dev/overnight_a4_20260912')
PYTHON = '/mnt/sdc1/robodojo/GalaxeaVLA/.venv/bin/python3.10'
MAX_STEPS = 2500
# The user removed the overnight wall-clock cutoff. The finite step budget
# remains; disk/numerical failures still stop safely without automatic retries.
WALL_SECONDS = None
MIN_FREE_DISK = 120 * 1024**3


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def publish(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    with temp.open('w') as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    temp.replace(path)


def bootstrap():
    sys.path[:0] = [str(SOURCE / 'src'), str(SOURCE / 'scripts'), str(SOURCE)]
    os.chdir(SOURCE)
    import g05
    from g05.utils.training.coordination_runtime import source_tree_sha256
    if SOURCE / 'src' not in Path(g05.__file__).resolve().parents:
        raise RuntimeError('Wrong editable/import source')
    if source_tree_sha256(SOURCE) != SOURCE_SHA:
        raise RuntimeError('Immutable A3 source changed; do not launch')
    return source_tree_sha256


def environment():
    env = dict(os.environ)
    env.update(PYTHONPATH=os.pathsep.join(map(str, (SOURCE / 'src', SOURCE / 'scripts', SOURCE))),
               CUDA_VISIBLE_DEVICES='0,1,2,3', OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4',
               WANDB_MODE='offline', TOKENIZERS_PARALLELISM='false', PYTHONUNBUFFERED='1')
    for key in ('DRY_RUN', 'MAX_EMBODIMENTS', 'MAX_DATASETS', 'OVERRIDE_DATASET',
                'MEMLITE_COORDINATION_CONFIG_ONLY', 'MEMLITE_COORDINATION_CONFIG_RESOURCE_RECEIPT'):
        env.pop(key, None)
    return env


def gpu_budget(indices):
    raw = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.free,memory.used,utilization.gpu',
                                   '--format=csv,noheader,nounits'], text=True)
    rows = [list(map(int, line.split(','))) for line in raw.strip().splitlines()]
    by_index = {row[0]: row for row in rows}
    if any(index not in by_index or by_index[index][1] < 34000 for index in indices):
        raise RuntimeError('Insufficient measured GPU headroom; no other jobs are stopped: ' + raw)
    if shutil.disk_usage(ROOT).free < MIN_FREE_DISK:
        raise RuntimeError('Disk safety reserve would be violated')
    return rows


def overrides(phase):
    if phase not in {'smoke', 'formal'}:
        raise ValueError(phase)
    run = ROOT / phase
    return ['model.batch_size=2', 'model.grad_accumulation_steps=2',
            'model.enable_bf16_training=true', 'model.model_weights_to_bf16=false',
            'model.num_workers=4', 'model.prefetch_factor=2',
            f'model.max_steps={5 if phase == "smoke" else MAX_STEPS}', 'model.max_epochs=null',
            'model.learning_rate=0.00001', 'model.warmup_steps=100', 'model.lr_min_ratio=0.1',
            'resume_ckpt=null', 'batch_size_val=1', 'checkpointing_steps=500', 'eval_steps=100',
            'logger.type=wandb', 'logger.mode=offline', 'logger.workspace=null', 'logger.log_steps=10',
            'logger.task=r1pro_memlite_skill_fm_a3_episodecoverage',
            f'logger.experiment_name=a4-overnight-{phase}-20260912', f'hydra.run.dir={run}']


def arguments(phase):
    return SimpleNamespace(stage='low_fm_lora_history',
        task_config=str(SOURCE / 'configs/task/r1pro_memlite_skill_fm_a3_episodecoverage.yaml'),
        source_root=str(SOURCE), sampling_index=str(POOLS / 'sampling_index.json'),
        eval_manifest=str(POOLS / 'eval_manifest.json'), run_dir=str(ROOT / phase),
        low_initial_checkpoint=str(PARENT), high_initial_checkpoint=None,
        official_asset_receipt=str(ROOT / 'gate/asset_receipt.json'),
        resume_ckpt=None, resume_receipt=None, num_obs_steps=6, nproc_per_node=4,
        seed=29, override=overrides(phase))


def verify_parent(model, parent):
    import torch
    old, actual = parent['model_state_dict'], model.state_dict()
    if parent['step'] != 5000 or set(old) != set(actual) or len(actual) != 1138:
        raise RuntimeError('Incomplete A3 checkpoint restoration')
    if sum('lora_' in name for name in actual) != 192:
        raise RuntimeError('Missing A3 LoRA entries')
    for name, tensor in actual.items():
        a, b = tensor.detach().cpu(), old[name]
        if a.shape != b.shape or a.dtype != b.dtype or not torch.isfinite(a).all():
            raise RuntimeError('Bad parent tensor: ' + name)
        if not torch.equal(a.contiguous().reshape(-1).view(torch.uint8),
                           b.contiguous().reshape(-1).view(torch.uint8)):
            raise RuntimeError('Parent bytes not restored: ' + name)


def gate():
    bootstrap()
    import torch
    from torch import distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    from omegaconf import OmegaConf
    from g05.utils.common.pytorch_utils import dict_apply
    from g05.utils.training.coordination_grad_audit import CoordinationGradientAudit
    from preflight_memlite_skillfm_gpu import _load_and_configure_gpu_model, _stage_contract

    output = ROOT / 'gate'
    output.mkdir(exist_ok=False)
    receipt = read(INPUT / 'result.json')
    if (receipt['status'] != 'complete' or not receipt['train_only'] or not receipt['all_five_tasks']
            or receipt['actual_batch_file_sha256'] != sha(INPUT / 'actual_cpu_batches.pt')
            or receipt['processor_config_sha256'] != sha(INPUT / 'diagnostic_processor_config.yaml')
            or receipt['sampler_sha256'] != sha(SOURCE / 'src/g05/utils/common/coordination_sampler.py')):
        raise RuntimeError('Cached, previously reviewed original-train processor inputs changed')
    if sha(PARENT) != PARENT_SHA:
        raise RuntimeError('A3 checkpoint digest changed')
    cfg = OmegaConf.load(INPUT / 'diagnostic_processor_config.yaml')
    # Only the weight initialization changes. Past/present inputs and labels
    # remain the exact already-reviewed original train batches.
    cfg.model.pretrained_ckpt = str(PARENT)
    cfg.resume_ckpt = None
    cfg.model.learning_rate = 1e-5
    torch.set_num_threads(4)
    torch.manual_seed(29)
    torch.cuda.set_device(0)
    device = torch.device('cuda:0')
    model, parent, _, _, _ = _load_and_configure_gpu_model(cfg, PARENT, device, _stage_contract(cfg, 'b'))
    verify_parent(model, parent)
    del parent
    gc.collect()
    model.apply_fp32_params()
    audit = CoordinationGradientAudit(model, expected_groups=['action_expert', 'vlm_lora'],
        frozen_prefixes=list(cfg.model.model_arch.coordination_train.frozen_group_prefixes),
        output_dir=output, rank=0)
    dist.init_process_group('nccl')
    ddp = DDP(model, device_ids=[0], find_unused_parameters=True, gradient_as_bucket_view=True, bucket_cap_mb=128)
    audit.after_distributed_initialization()
    groups = model.get_optim_param_groups(lr=1e-5, weight_decay=float(cfg.model.weight_decay),
        apply_decay_on_norm_and_bias=False,
        backbone_lr_multiplier=float(cfg.model.get('backbone_lr_multiplier', 1.0)),
        vision_lr_multiplier=float(cfg.model.get('vision_lr_multiplier', 1.0)))
    optimizer = torch.optim.AdamW(groups, lr=1e-5, betas=tuple(cfg.model.betas))
    if {id(p) for group in optimizer.param_groups for p in group['params']} != {id(p) for p in model.parameters() if p.requires_grad}:
        raise RuntimeError('Optimizer coverage differs from declared training parameters')
    batches = torch.load(INPUT / 'actual_cpu_batches.pt', map_location='cpu', weights_only=False)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.reset_peak_memory_stats()
    losses = []
    for i, original in enumerate(batches[:4]):
        batch = dict_apply(deepcopy(original), lambda x: x.to(device) if isinstance(x, torch.Tensor) else x)
        update = (i + 1) % 2 == 0
        with (nullcontext() if update else ddp.no_sync()):
            with torch.autocast('cuda', dtype=torch.bfloat16):
                loss, metrics = ddp(batch)
            if not torch.isfinite(loss) or not torch.allclose(loss.detach(), metrics['fm_loss'].detach()):
                raise RuntimeError('Nonfinite loss or unexpected non-FM training objective')
            (loss / 2).backward()
        if any(not torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):
            raise RuntimeError('Nonfinite gradient')
        facts = audit.after_backward() if i == 1 else None
        if update:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.model.max_grad_norm)
            rates = [float(group['lr']) for group in optimizer.param_groups]
            optimizer.step()
            if i == 1:
                audit.after_optimizer_step(facts, step=1, effective_lrs=rates)
                bindings = dict(hf_processor_path=str(cfg.model.model_arch.hf_processor_path),
                                action_tokenizer_ckpt=str(cfg.model.model_arch.AT_CONFIG.ckpt_dir))
                publish(output / 'asset_receipt.json', dict(passed=True, mode='gpu_one_step',
                    formal_training=False, checkpoint_saved=False, checkpoint=str(PARENT),
                    checkpoint_contract={'checkpoint_sha256': PARENT_SHA}, official_asset_bindings=bindings,
                    actual_optimizer_steps_so_far=1, actual_microbatches_so_far=2,
                    source_root_sha256=SOURCE_SHA, full_parent_state_entries_bitwise_equal=1138,
                    lora_entries_bitwise_equal=192, original_processor_receipt_sha256=sha(INPUT / 'result.json')))
            optimizer.zero_grad(set_to_none=True)
        losses.append(float(loss.detach()))
        del batch, loss, metrics
    audit._assert_frozen_unchanged()
    audit.require_verified_update()
    counters = {int(s['step'].item()) for s in optimizer.state.values()}
    if len(losses) != 4 or counters != {2} or any(not torch.isfinite(p).all() for p in model.parameters()):
        raise RuntimeError('Actual two-update gate failed')
    publish(output / 'result.json', dict(passed=True, actual_updates=2, microbatches=4,
        losses=losses, peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        source_sha256=SOURCE_SHA, parent_sha256=PARENT_SHA, checkpoint_saved=False,
        frozen_bitwise_unchanged=True, model_groups_updated=['action_expert', 'vlm_lora'],
        probe_world_size=1, formal_world_size=4, success_rate_claim=False))
    dist.destroy_process_group()


def stop_owned_child(child):
    # Child was created by this supervisor in its own session and not reaped.
    # No PID discovered from someone else's receipt is ever signalled.
    if child.poll() is None:
        os.killpg(child.pid, signal.SIGTERM)
        try:
            child.wait(timeout=45)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait(timeout=15)


def execute(command, env, log_path, seconds, heartbeat=None):
    started = time.monotonic()
    with Path(log_path).open('x') as log:
        child = subprocess.Popen(command, cwd=SOURCE, env=env, stdin=subprocess.DEVNULL,
                                 stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            while child.poll() is None:
                if seconds is not None and time.monotonic() - started > seconds:
                    raise TimeoutError(f'Owned job reached wall-clock cap of {seconds} seconds')
                if shutil.disk_usage(ROOT).free < MIN_FREE_DISK:
                    raise RuntimeError('Disk reserve reached; preserve existing checkpoints and stop this run')
                if heartbeat:
                    heartbeat(child)
                time.sleep(10)
            if child.returncode != 0:
                raise RuntimeError(f'Child exited {child.returncode}; inspect {log_path}')
        finally:
            stop_owned_child(child)


def inspect_checkpoint(phase, expected_step):
    import torch
    torch.set_num_threads(4)
    run = ROOT / phase
    path = run / f'checkpoints/step_{expected_step}.pt'
    saved = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
    parent = torch.load(PARENT, map_location='cpu', weights_only=False, mmap=True)
    expected_micro = expected_step * 2
    if (saved['step'] != expected_step or saved['scheduler_state_dict']['last_epoch'] != expected_step
            or saved['coordination_training_identity'] != read(run / 'coordination_run_receipt.json')['identity']
            or saved['coordination_stage_contract']['trainability_profile'] != 'low_ae_lora_history'):
        raise RuntimeError('Checkpoint step/scheduler/identity/profile mismatch')
    rng = saved['coordination_rng_state_by_rank']
    if rng['world_size'] != 4 or sorted(r['rank'] for r in rng['states_by_rank']) != [0, 1, 2, 3]:
        raise RuntimeError('Missing per-rank resume RNG')
    states = saved['optimizer_state_dict']['state']
    if len(states) != 504 or {int(s['step']) for s in states.values()} != {expected_step}:
        raise RuntimeError('Adam state coverage/counters mismatch')
    for state in states.values():
        if any(not torch.isfinite(state[k]).all() for k in ('exp_avg', 'exp_avg_sq')):
            raise RuntimeError('Nonfinite Adam moments')
    names = set(read(run / 'coordination_trainability_receipt.json')['model_trainability']['trainable_parameter_names'])
    old, new = parent['model_state_dict'], saved['model_state_dict']
    if set(old) != set(new) or len(new) != 1138 or len(names) != 514:
        raise RuntimeError('Model checkpoint schema changed')
    changed = []
    for name, tensor in new.items():
        baseline = old[name]
        if tensor.shape != baseline.shape or tensor.dtype != baseline.dtype or not torch.isfinite(tensor).all():
            raise RuntimeError('Model shape/dtype/numeric error: ' + name)
        equal = torch.equal(tensor.contiguous().reshape(-1).view(torch.uint8),
                            baseline.contiguous().reshape(-1).view(torch.uint8))
        if not equal:
            if name not in names:
                raise RuntimeError('Frozen checkpoint tensor changed: ' + name)
            changed.append(name)
    if not any('lora_' in n for n in changed) or not any('action_expert' in n for n in changed):
        raise RuntimeError('Expected both low groups to update')
    if read(run / 'coordination_train_source_spec.json')['sampler_schedule']['leaf_ordering'] != 'episode_round_robin_v1':
        raise RuntimeError('Wrong sampler used')
    counts, tasks = [], set()
    for rank in range(4):
        rows = [json.loads(line) for line in (run / f'coordination_sample_receipts_rank{rank}.jsonl').open()]
        if len(rows) != expected_micro:
            raise RuntimeError('Missing actual microbatch receipts')
        for i, row in enumerate(rows):
            if row['step'] != i // 2 + 1 or len(row['samples']) != 2:
                raise RuntimeError('Sample/optimizer clock mismatch')
            for sample in row['samples']:
                if sample['requested_branch'] != 'low' or sample['actual_branch'] != 'low':
                    raise RuntimeError('Unexpected high sample')
                tasks.add(int(sample['episode_index']) // 200)
        counts.append(sum(len(row['samples']) for row in rows))
    if tasks != set(range(5)) or sha(run / 'dataset_stats.json') != sha(PARENT.parent.parent / 'dataset_stats.json'):
        raise RuntimeError('Task coverage or train-only normalization mismatch')
    result = dict(passed=True, checkpoint=str(path), checkpoint_sha256=sha(path),
        step=expected_step, actual_adam_states=len(states), changed_trainable_entries=len(changed),
        frozen_bitwise_unchanged=True, all_model_and_adam_finite=True,
        all_rank_rng_saved=True, per_rank_sample_counts=counts, source_sha256=SOURCE_SHA,
        parent_sha256=PARENT_SHA, success_rate_claim=False)
    publish(ROOT / f'{phase}_checkpoint_inspection.json', result)


def train_phase(phase):
    import coordination_launcher as launcher
    from g05.utils.training.coordination_runtime import (CoordinationLock, mark_running,
        refuse_if_live_run, update_run_receipt)
    args = arguments(phase)
    plan = launcher.build_plan(args)
    plan['source_root'] = str(SOURCE)
    command = launcher.training_command(args, plan)
    if phase == 'formal' and not read(ROOT / 'smoke_checkpoint_inspection.json')['passed']:
        raise RuntimeError('Actual four-GPU saved-checkpoint smoke required')
    gpu = gpu_budget(range(4))
    publish(ROOT / f'{phase}_plan.json', dict(plan=plan, command=command, gpu_before=gpu, time=now()))
    preflight_env = launcher.trainer_config_preflight_environment(
        plan, ROOT / f'{phase}_config_preflight.json', command)
    execute(command, preflight_env, ROOT / f'{phase}_config_preflight.log', 900)
    run = ROOT / phase
    refuse_if_live_run(run)
    launcher.prepare_run_receipt(args, plan=plan, command=command)
    with CoordinationLock(purpose='training') as lock:
        def heartbeat(child):
            lock.heartbeat(metadata=dict(run_dir=str(run), pid=child.pid, source_root=str(SOURCE)))
            mark_running(run, pid=child.pid, pgrp=child.pid, config_sha256=plan['identity']['config_sha256'])
            publish(ROOT / 'status.json', dict(state='running', phase=phase, time=now(),
                supervisor_pid=os.getpid(), trainer_pid=child.pid, run=str(run),
                max_steps=5 if phase == 'smoke' else MAX_STEPS, wall_seconds=WALL_SECONDS,
                source_sha256=SOURCE_SHA, parent_sha256=PARENT_SHA))
        try:
            execute(command, launcher.child_environment(plan), ROOT / f'{phase}.log',
                    1800 if phase == 'smoke' else WALL_SECONDS, heartbeat)
            update_run_receipt(run, state='complete', returncode=0, source_root=str(SOURCE))
        except BaseException as error:
            update_run_receipt(run, state='failed', error=str(error), source_root=str(SOURCE))
            raise
    inspect_checkpoint(phase, 5 if phase == 'smoke' else MAX_STEPS)


def supervise():
    try:
        bootstrap()
        publish(ROOT / 'status.json', dict(state='preflight', time=now(), supervisor_pid=os.getpid()))
        gpu_budget([2])
        env = environment()
        env['CUDA_VISIBLE_DEVICES'] = '2'
        execute([PYTHON, '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=1',
                 str(Path(__file__).resolve()), 'gate'], env, ROOT / 'gate.log', 1800)
        if not read(ROOT / 'gate/result.json')['passed']:
            raise RuntimeError('Actual parent/gradient gate failed')
        train_phase('smoke')
        train_phase('formal')
        publish(ROOT / 'status.json', dict(state='complete', time=now(),
            verified_optimizer_steps=MAX_STEPS, checkpoint=str(ROOT / f'formal/checkpoints/step_{MAX_STEPS}.pt'),
            success_rate_claim=False, next='Compare paired action diagnostics; do not auto-retrain or deploy'))
    except BaseException as error:
        publish(ROOT / 'status.json', dict(state='failed', time=now(), error=f'{type(error).__name__}: {error}',
            automatic_retry=False, prior_checkpoints_preserved=True))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('start', 'supervise', 'gate'))
    args = parser.parse_args()
    if args.mode == 'gate':
        gate()
    elif args.mode == 'supervise':
        supervise()
    else:
        repo = Path(__file__).resolve().parents[2]
        if subprocess.check_output(['git', '-C', str(repo), 'status', '--porcelain'], text=True).strip():
            raise RuntimeError('Start only from a clean pinned recipe worktree')
        commit = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
        ROOT.mkdir(exist_ok=False)
        with (ROOT / 'supervisor.log').open('x') as stream:
            child = subprocess.Popen([PYTHON, str(Path(__file__).resolve()), 'supervise'],
                env=environment(), stdin=subprocess.DEVNULL, stdout=stream,
                stderr=subprocess.STDOUT, start_new_session=True)
        receipt = dict(time=now(), supervisor_pid=child.pid, recipe_commit=commit,
            recipe=str(Path(__file__).resolve()), recipe_sha256=sha(__file__), run_root=str(ROOT),
            source=str(SOURCE), source_sha256=SOURCE_SHA, parent=str(PARENT), parent_sha256=PARENT_SHA,
            seed=29, maximum_new_updates=MAX_STEPS, formal_wall_seconds=WALL_SECONDS,
            fresh_optimizer_and_scheduler=True, equivalent_resume=False,
            high_unchanged=True, datasets_unchanged=True, other_processes_stopped=False)
        publish(ROOT / 'launch.json', receipt)
        print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
