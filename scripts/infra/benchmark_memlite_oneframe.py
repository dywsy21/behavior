"""Bounded A4/SkillFM compute benchmark, not a formal 100-task trainer.

Only audited original TRAIN examples are repeated. Video I/O is deliberately
not included in this compute probe; no trained checkpoint is saved.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone, timedelta
import gc
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT = Path('/data/workspace/wsy/behavior2026')
WEIGHT = ROOT / 'models/memlite-a4-20260912/step_2500.pt'
WEIGHT_SHA = '6186704788c27c9fae3502c884df0e259de5242ee8690fe578dcbc1f2632f269'
INPUT = ROOT / 'runs/memlite_oneframe_benchmark_20260930/inputs/actual_cpu_batches.pt'
INPUT_SHA = '237acf01b29bd0d6806ed1a11d9033a747640b3ea9e0bf7246b7a62b4e92be81'
CAMERAS = ('exterior', 'wrist_left', 'wrist_right')
PAD = (7, 8, 17, 18)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def publish(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def single_frame_pool(batches):
    """Keep the current camera/state, but do NOT offset future actions by 5."""
    import torch
    from g05.data_processor.processor.memlite_v6_projection import validate_embedded_model_projection
    rows = []
    old_markers = ''.join(f'<image{i}_image_!>' for i in range(18))
    new_markers = ''.join(f'<image{i}_image_!>' for i in range(3))
    for batch in batches:
        if tuple(batch['pixel_values']) != CAMERAS:
            raise ValueError('Unexpected camera-major order')
        for j, original in enumerate(batch['samples']):
            sample = deepcopy(original)
            validate_embedded_model_projection(sample)
            if sample['template'].count(old_markers) != 1:
                raise ValueError('Expected the original eighteen-image template')
            sample['template'] = sample['template'].replace(old_markers, new_markers)
            sizes = [sample[f'image{i}'] for i in (5, 11, 17)]
            for key in list(sample):
                if re.fullmatch(r'image\d+', key):
                    del sample[key]
            sample.update({f'image{i}': size for i, size in enumerate(sizes)})
            value = sample['proprio']['value']
            if value.shape != (6, 27):
                raise ValueError('Expected six original proprio observations')
            sample['proprio']['value'] = value[-1:].clone()
            pixels = {}
            for key in CAMERAS:
                value = batch['pixel_values'][key][j]
                if value.shape != (6, 3, 256, 256):
                    raise ValueError('Expected six original RGB frames')
                pixels[key] = value[-1:].clone()
            pad = batch['action_dim_is_pad'][j]
            if tuple(torch.nonzero(pad).flatten().tolist()) != PAD:
                raise ValueError('Real base/trunk controls must remain unmasked')
            actions = batch['action'][j].clone()
            if actions.shape != (32, 27) or not torch.isfinite(actions).all():
                raise ValueError('Invalid future-action target')
            # Outer sampler locators/audit payloads intentionally do not enter
            # this new model-visible batch. The original file is immutable.
            rows.append(dict(samples=sample, pixel_values=pixels, action=actions,
                             action_is_pad=batch['action_is_pad'][j].clone(),
                             action_dim_is_pad=pad.clone()))
    if len(rows) != 10:
        raise ValueError('Expected exactly ten previously audited TRAIN rows')
    return rows


def collate(rows, indices, device):
    import torch
    selected = [rows[i % len(rows)] for i in indices]
    return dict(samples=[deepcopy(row['samples']) for row in selected],
                pixel_values={key: torch.stack([row['pixel_values'][key] for row in selected]).to(device)
                              for key in CAMERAS},
                **{key: torch.stack([row[key] for row in selected]).to(device)
                   for key in ('action', 'action_is_pad', 'action_dim_is_pad')})


def configuration():
    from omegaconf import OmegaConf
    from g05.utils.config.config_resolvers import register_default_resolvers
    register_default_resolvers()
    cfg = OmegaConf.load(ROOT / 'models/memlite-a4-20260912/A4-training-config.yaml')
    arch = cfg.model.model_arch
    arch.num_obs_steps = arch.cond_steps = 1
    arch.num_input_images = 3
    arch.coordination_train.low_observation_steps = 1
    cfg.model.processor.num_obs_steps = cfg.data.obs_size = 1
    arch.hf_processor_path = str(ROOT / 'models/qwen3_5_2b_base_processor')
    cfg.tokenizer.vq_config.ckpt_dir = str(ROOT / 'models/action_tokenizer.pt')
    arch.pretrained_model_path = None
    # Keep the original temporal module/PE shapes so no checkpoint tensor is
    # discarded. A single-frame forward does not execute temporal attention.
    return OmegaConf.to_container(arch, resolve=True)


def same_tensor_bytes(value, reference):
    """Compare across constructor-selected devices, including signed zeros."""
    import torch
    if value.dtype != reference.dtype or value.shape != reference.shape:
        return False
    left = value.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
    right = reference.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
    return torch.equal(left, right)


def restored_policy(arch):
    import torch
    from hydra.utils import instantiate
    saved = torch.load(WEIGHT, map_location='cpu', mmap=True, weights_only=False)
    state = saved['model_state_dict']
    if saved['step'] != 2500 or len(state) != 1138:
        raise ValueError('Not the registered complete A4 checkpoint')
    model = instantiate(arch)
    mapped = model.remap_checkpoint_state_dict(state)
    # No partial/shape-mismatched base restore and no discarded LoRA weights.
    model.load_state_dict({k: v for k, v in mapped.items() if 'lora_' not in k}, strict=True)
    lora = model.post_checkpoint_load(state)
    if lora.get('adapter_load_mode') != 'resume' or lora.get('restored') != 192:
        raise ValueError('The trained A4 adapter was not fully restored')
    actual = model.state_dict()
    if set(actual) != set(state):
        raise ValueError('Full model-state coverage changed')
    for name, value in actual.items():
        if not same_tensor_bytes(value, state[name]):
            raise ValueError('Checkpoint tensor is not bitwise restored: ' + name)
    contract = model.configure_coordination_trainability()
    groups = model.coordination_trainable_parameter_groups()
    if {key: len(value) for key, value in groups.items()} != {'action_expert': 322, 'vlm_lora': 192}:
        raise ValueError('A4 trainability groups changed')
    del saved, state, actual, mapped
    gc.collect()
    return model, contract


def batch_plan(global_batch, accumulation=1, world=8):
    """Keep each authorized probe bounded; batch256 has no implicit retry."""
    if world != 8 or global_batch not in (64, 128, 256):
        raise ValueError('Only registered eight-GPU batch sizes are allowed')
    if accumulation not in (1, 2) or (accumulation == 2 and global_batch != 128):
        raise ValueError('Only the preregistered batch128 fallback may accumulate')
    warmup, measured = (5, 20) if accumulation == 2 else (6, 24)
    return global_batch // world // accumulation, warmup, measured


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--global-batch', type=int, choices=(64, 128, 256), default=64)
    ap.add_argument('--accumulation', type=int, choices=(1, 2), default=1)
    ap.add_argument('--cpu-preflight', action='store_true')
    args = ap.parse_args()
    repo = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo / 'src'))
    if subprocess.check_output(['git', '-C', str(repo), 'status', '--porcelain'], text=True).strip():
        raise RuntimeError('Use a clean frozen source worktree')
    import torch
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    commit = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
    if digest(INPUT) != INPUT_SHA:
        raise RuntimeError('Original approved TRAIN input changed')
    rows = single_frame_pool(torch.load(INPUT, map_location='cpu', weights_only=False))
    arch = configuration()
    if args.cpu_preflight:
        args.output.mkdir(parents=True, exist_ok=False)
        from g05.models.g05.g05_policy_memlite_skill_fm import G05PolicyMEMLiteSkillFM
        from types import SimpleNamespace
        probe = SimpleNamespace(interface_schema_version=6, model_config=SimpleNamespace(action_dim=27),
                                _memlite_branch_masks=G05PolicyMEMLiteSkillFM._memlite_branch_masks)
        batch = collate(rows, range(10), 'cpu')
        G05PolicyMEMLiteSkillFM._validate_skill_batch(probe, batch['samples'], batch['action'], batch['action_is_pad'])
        publish(args.output / 'result.json', dict(passed=True, commit=commit, rows=len(rows),
                input_sha256=INPUT_SHA, frames_per_camera=1, horizon=32, action_dim=27,
                pad_dims=PAD, cuda_initialized=torch.cuda.is_initialized(), model_instantiated=False,
                arch=arch))
        return
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    rank, world, local = (int(os.environ[key]) for key in ('RANK', 'WORLD_SIZE', 'LOCAL_RANK'))
    if world != 8 or rank != local:
        raise ValueError('Exactly one eight-GPU node is required')
    micro, warmup, measured = batch_plan(args.global_batch, args.accumulation, world)
    torch.manual_seed(73 + rank)
    torch.cuda.set_device(local)
    dist.init_process_group('nccl', timeout=timedelta(seconds=180))
    if rank == 0:
        args.output.mkdir(parents=True, exist_ok=False)
        if WEIGHT.stat().st_size != 16581363550 or digest(WEIGHT) != WEIGHT_SHA:
            raise RuntimeError('A4 checkpoint final content identity failed')
    dist.barrier()
    model, receipt = restored_policy(arch)
    publish(args.output / f'restored_rank{rank}.json', dict(commit=commit, exact_model_tensors=1138,
            exact_lora_tensors=192, profile=receipt['trainability_profile'], num_obs_steps=1,
            pid=os.getpid(), gpu=local, checkpoint_sha256=WEIGHT_SHA))
    model.to(f'cuda:{local}').train()
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=1e-5, betas=(.9, .95), weight_decay=.03, fused=False)
    ddp = DDP(model, device_ids=[local], find_unused_parameters=True,
              gradient_as_bucket_view=True, bucket_cap_mb=128)
    own_pids = [None] * world
    dist.all_gather_object(own_pids, os.getpid())
    torch.cuda.reset_peak_memory_stats()
    durations, losses = [], []
    gradient_receipt = None
    for step in range(warmup + measured):
        if rank == 0:
            active = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid',
                                             '--format=csv,noheader,nounits'], text=True)
            foreign = {int(x.strip()) for x in active.splitlines() if x.strip().isdigit()} - set(own_pids)
            if foreign:
                raise RuntimeError('External GPU process appeared; stop our benchmark')
        dist.barrier()
        torch.cuda.synchronize()
        started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.
        for accum in range(args.accumulation):
            start = (step * args.global_batch + rank * micro + accum * micro * world) % len(rows)
            batch = collate(rows, range(start, start + micro), f'cuda:{local}')
            # Standard synchronized backward is retained in the one allowed
            # accumulation fallback; do not misreport it as a micro16 run.
            with torch.autocast('cuda', dtype=torch.bfloat16):
                loss, metrics = ddp(batch)
            if not torch.isfinite(loss) or not torch.equal(loss, metrics['fm_loss']):
                raise RuntimeError('Expected finite pure FM objective')
            (loss / args.accumulation).backward()
            total_loss += float(loss.detach()) / args.accumulation
        if step == 0:
            groups = model.coordination_trainable_parameter_groups()
            gradient_receipt = {key: sum(p.grad is not None for _, p in entries)
                                for key, entries in groups.items()}
            if gradient_receipt != {'action_expert': 322, 'vlm_lora': 182}:
                raise RuntimeError('Expected A4 FM-to-AE/LoRA gradient coverage')
            if any(p.grad is not None for p in model.parameters() if not p.requires_grad):
                raise RuntimeError('A frozen parameter received gradient')
            if not any(p.grad is not None and bool(p.grad.abs().sum() > 0)
                       for _, p in groups['vlm_lora']):
                raise RuntimeError('No nonzero FM gradient reached VLM LoRA')
        norm = torch.nn.utils.clip_grad_norm_(params, 1.)
        if not torch.isfinite(norm):
            raise RuntimeError('Nonfinite gradient norm')
        optimizer.step()
        torch.cuda.synchronize()
        dist.barrier()
        elapsed = time.perf_counter() - started
        if step >= warmup:
            durations.append(elapsed)
            losses.append(total_loss)
        if rank == 0:
            print(json.dumps(dict(step=step + 1, seconds=elapsed, fm_loss=total_loss,
                                  global_batch=args.global_batch, microbatch=micro)), flush=True)
    evidence = dict(rank=rank, seconds=durations, fm_loss=losses, gradients=gradient_receipt,
                    peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                    peak_reserved_bytes=torch.cuda.max_memory_reserved())
    gathered = [None] * world
    dist.all_gather_object(gathered, evidence)
    if rank == 0:
        slowest = [max(row['seconds'][i] for row in gathered) for i in range(measured)]
        total = sum(slowest)
        publish(args.output / 'result.json', dict(status='complete', commit=commit,
                completed_utc=datetime.now(timezone.utc).isoformat(), global_batch=args.global_batch,
                microbatch=micro, accumulation=args.accumulation, warmup_updates=warmup,
                measured_updates=measured, measured_samples=args.global_batch * measured,
                measured_seconds=total, samples_per_second=args.global_batch * measured / total,
                checkpoint_sha256=WEIGHT_SHA, input_sha256=INPUT_SHA, ranks=gathered,
                scope='cached approved 5-task TRAIN inputs; no raw video decoding',
                full_100task_end_to_end_training=False, deployed_checkpoint_saved=False))
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
