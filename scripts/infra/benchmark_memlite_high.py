"""Bounded real B-final planner-only training throughput, never a full trainer.

Uses audited original TRAIN observations with unchanged targets. No rollout,
outcome supervision, formal 100-task label release, or deployed checkpoint.
"""
from __future__ import annotations
import argparse
from contextlib import nullcontext
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path('/data/workspace/wsy/behavior2026')
ASSETS = ROOT / 'models/memlite-b-final-20260910'
WEIGHT = ASSETS / 'B-final-model.pt'
WEIGHT_SHA = 'e7cd7bf738eb46901565088f829aa82c5c6ea95f610d4df1499e634959d29b13'
PARENT_SHA = 'd4580d80cdc91a707c233a6c1e625f8fbdeca8896dd38a3d570c6fad340184ef'
INPUTS = ROOT / 'runs/memlite_high_benchmark_20260930/inputs'


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


def batch_plan(micro, capacity=False):
    if isinstance(micro, bool) or micro not in (4, 8, 16, 32):
        raise ValueError('Registered single-node microbatch must be 4/8/16/32')
    return dict(global_batch=256, world=8, micro=micro, accumulation=32 // micro,
                warmup=1 if capacity else 6, measured=1 if capacity else 12)


def profile_indices(rows, token_records, profile):
    if profile == 'mixed':
        return list(range(len(rows)))
    if profile == 'long':
        return sorted(range(len(rows)), key=lambda i: token_records[i]['sequence_tokens'])[-8:]
    raise ValueError('Only actual mixed and longest TRAIN profiles are admitted')


def configuration():
    from omegaconf import OmegaConf
    from g05.utils.config.config_resolvers import register_default_resolvers
    register_default_resolvers()
    cfg = OmegaConf.load(ASSETS / 'B-training-config.yaml')
    arch = cfg.model.model_arch
    arch.num_obs_steps = arch.cond_steps = 1
    arch.num_input_images = 3
    arch.coordination_train.high_observation_steps = 1
    arch.hf_processor_path = str(ROOT / 'models/qwen3_5_2b_base_processor')
    cfg.tokenizer.vq_config.ckpt_dir = str(ROOT / 'models/action_tokenizer.pt')
    arch.pretrained_model_path = None
    if (arch.continuous_action or not arch.discrete_action or not arch.predict_cot
            or arch.coordination_train.trainability_profile != 'high_planner_only'
            or arch.planner_outcome.memory_update_ce_weight != .25
            or arch.planner_outcome.outcome_loss_weight != 0):
        raise ValueError('The real B-final planner-only objective changed')
    return OmegaConf.to_container(arch, resolve=True)


def load_inputs():
    import torch
    receipt = json.loads((INPUTS / 'receipt.json').read_text())
    target = INPUTS / 'B-train-singleframe-samples.pt'
    if (receipt.get('status') != 'complete' or receipt.get('split') != 'original_train_only'
            or receipt.get('labels_modified') is not False or receipt.get('observation_frames') != 1
            or digest(target) != receipt['sha256']):
        raise ValueError('Frozen original TRAIN fixture identity failed')
    rows = torch.load(target, map_location='cpu', weights_only=False)
    if len(rows) != receipt['rows'] or len(rows) != 30:
        raise ValueError('Expected thirty length-stratified original TRAIN rows')
    return rows, receipt


def preflight(arch, rows):
    """Real tokenizer/field masks, without allocating a policy or CUDA state."""
    import torch
    from omegaconf import OmegaConf
    from g05.models.g05.io.input_preprocessor import InputPreprocessor
    from g05.models.g05.g05_policy_memlite_planner_outcome import G05PolicyMEMLitePlannerOutcome as Policy
    cfg = OmegaConf.create(arch)
    pp = cfg.input_preprocessor
    processor = InputPreprocessor(hf_processor_path=cfg.hf_processor_path,
        hf_processor_class=cfg.hf_processor_class, action_tokenizer_class=cfg.action_tokenizer,
        at_config=cfg.AT_CONFIG, base_vocab_size=cfg.base_vocab_size,
        padded_vocab_size=cfg.padded_vocab_size, pad_token_id=cfg.pad_token_id,
        image_token_index=cfg.image_token_index, num_image_tokens=cfg.vision.num_image_tokens,
        input_action_corruption=pp.input_action_corruption, pred_eov=pp.pred_eov,
        batchify_action=pp.batchify_action, pi05_ft_mode=pp.pi05_ft_mode,
        proprio_encoder=cfg.proprio_encoder, model_type='qwen35', model_cfg=cfg)
    probe = Policy.__new__(Policy)
    torch.nn.Module.__init__(probe)
    probe.processor = processor
    probe.planner_only = True
    probe.interface_schema_version = 6
    probe.memory_update_ce_weight = .25
    probe.supervise_task_parent_format = True
    probe.task_parent_formats = dict(cfg.planner_outcome.task_parent_format_by_task)
    result = []
    for i, row in enumerate(rows):
        sample = row['samples']
        Policy._validate_high_batch(probe, [sample])
        if len(row['pixel_values']) != 3 or any(v.shape != (1, 3, 256, 256) for v in row['pixel_values'].values()):
            raise ValueError('Only one real frame per camera is admitted')
        ids, labels, attn, _ = processor.encode_train([sample], device=torch.device('cpu'),
            training=False, max_chunk_token_length=4096, max_pad_token_length=None)
        context, context_mask = processor.encode_inference([sample], device=torch.device('cpu'),
                                                         mode='ar', training=False)
        positions = Policy._context_positions(ids, attn, context, context_mask)
        Policy._assert_tokenized_prefix_contract(probe, context, context_mask, [sample])
        spans = Policy._mask_unsupervised_ar_fields(probe, ids, labels, positions, [sample])
        weights, _ = Policy._planner_only_loss_token_weights(probe, labels, spans, samples=[sample])
        for field in ('outcome_target', 'task_complete'):
            if not bool((labels[0, spans[0][field]] == -100).all()):
                raise ValueError('Missing physical truth acquired CE supervision')
        valid = labels[:, 1:] != -100
        count = float(weights[:, 1:][valid].sum())
        if count <= 0:
            raise ValueError('No actual planning CE target')
        result.append(dict(index=i, sequence_tokens=int((attn != 0).sum()),
                           context_tokens=int((context_mask != 0).sum()),
                           supervised_tokens=int(valid.sum()), ce_weight_sum=count,
                           outcome_and_terminal_masked=True))
    mixed_checks = []
    # Actual variable-length batches exercise left-padding and field offsets,
    # not just the historically easier batch=1 prefix contract.
    for training in (False, True):
        probe.train(training)
        for micro in (4, 8, 16, 32):
            indices = [(i * 11) % len(rows) for i in range(micro)]
            samples = [deepcopy(rows[i]['samples']) for i in indices]
            ids, labels, attn, _ = processor.encode_train(samples, device=torch.device('cpu'),
                training=training, max_chunk_token_length=4096, max_pad_token_length=None)
            context, context_mask = processor.encode_inference(samples, device=torch.device('cpu'),
                                                               mode='ar', training=training)
            positions = Policy._context_positions(ids, attn, context, context_mask)
            Policy._assert_tokenized_prefix_contract(probe, context, context_mask, samples)
            spans = Policy._mask_unsupervised_ar_fields(probe, ids, labels, positions, samples)
            weights, _ = Policy._planner_only_loss_token_weights(probe, labels, spans, samples=samples)
            valid = labels[:, 1:] != -100
            expected = sum(result[i]['ce_weight_sum'] for i in indices)
            if float(weights[:, 1:][valid].sum()) != expected:
                raise ValueError('Variable-length batch changed the single-row objective mask/weights')
            mixed_checks.append(dict(training=training, micro=micro,
                                     ce_weight_sum=expected, prefix_and_weight_identity=True))
    if torch.cuda.is_initialized():
        raise RuntimeError('CPU token preflight unexpectedly initialized CUDA')
    return result, mixed_checks


def collate(rows, indices, device):
    import torch
    selected = [rows[i] for i in indices]
    keys = tuple(selected[0]['pixel_values'])
    if any(tuple(row['pixel_values']) != keys for row in selected):
        raise ValueError('Camera order varies within original fixture')
    return dict(samples=[deepcopy(row['samples']) for row in selected],
                pixel_values={key: torch.stack([row['pixel_values'][key] for row in selected]).to(device)
                              for key in keys})


def restored_model(arch):
    import torch
    from hydra.utils import instantiate
    from benchmark_memlite_oneframe import same_tensor_bytes
    saved = torch.load(WEIGHT, map_location='cpu', mmap=True, weights_only=False)
    if saved['step'] != 1500 or saved['source_sha256'] != PARENT_SHA:
        raise ValueError('Not the registered B-final model-only export')
    state = saved['model_state_dict']
    if len(state) != 950:
        raise ValueError('B-final full state count changed')
    policy = instantiate(arch)
    policy.load_state_dict(state, strict=True)
    actual = policy.state_dict()
    if set(actual) != set(state) or any(not same_tensor_bytes(actual[k], state[k]) for k in state):
        raise ValueError('B-final tensorwise restoration failed')
    contract = policy.configure_coordination_trainability()
    groups = policy.coordination_trainable_parameter_groups()
    if set(groups) != {'planner_vlm'} or len(groups['planner_vlm']) != 326:
        raise ValueError('Must train the actual full planner group, not low LoRA or outcome head')
    del saved, state, actual
    gc.collect()
    return policy, contract


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--cpu-preflight', action='store_true')
    ap.add_argument('--token-receipt', type=Path)
    ap.add_argument('--profile', choices=('mixed', 'long'), default='mixed')
    ap.add_argument('--micro', type=int, choices=(4, 8, 16, 32), default=8)
    ap.add_argument('--capacity-only', action='store_true')
    args = ap.parse_args()
    repo = Path(__file__).resolve().parents[2]
    os.chdir(repo)
    sys.path.insert(0, str(repo / 'src'))
    if subprocess.check_output(['git', 'status', '--porcelain'], text=True).strip():
        raise RuntimeError('Only a clean frozen source may run')
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    import torch
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    rows, receipt = load_inputs()
    arch = configuration()
    if args.cpu_preflight:
        args.output.mkdir(parents=True, exist_ok=False)
        records, mixed_checks = preflight(arch, rows)
        publish(args.output / 'result.json', dict(status='complete', commit=commit, rows=records,
                mixed_batch_checks=mixed_checks, input_sha256=receipt['sha256'],
                arch=arch, cuda_initialized=False))
        return
    if not args.token_receipt:
        raise ValueError('GPU measurement requires the successful CPU token receipt')
    token = json.loads(args.token_receipt.read_text())
    if token.get('status') != 'complete' or token['input_sha256'] != receipt['sha256'] or token['arch'] != arch:
        raise ValueError('CPU/GPU token/config identities disagree')
    if len(token.get('mixed_batch_checks', [])) != 8 or not all(
            check['prefix_and_weight_identity'] for check in token['mixed_batch_checks']):
        raise ValueError('Missing variable-length CPU batch contract checks')
    plan = batch_plan(args.micro, args.capacity_only)
    selected = profile_indices(rows, token['rows'], args.profile)
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    rank, world, local = (int(os.environ[k]) for k in ('RANK', 'WORLD_SIZE', 'LOCAL_RANK'))
    if world != 8 or rank != local:
        raise ValueError('Exactly one eight-GPU node is authorized')
    torch.manual_seed(17 + rank)
    torch.cuda.set_device(local)
    dist.init_process_group('nccl', timeout=timedelta(seconds=240))
    if rank == 0:
        args.output.mkdir(parents=True, exist_ok=False)
        if digest(WEIGHT) != WEIGHT_SHA:
            raise ValueError('Published B-final asset hash changed')
    dist.barrier()
    policy, contract = restored_model(arch)
    publish(args.output / f'restored_rank{rank}.json', dict(commit=commit, exact_model_tensors=950,
        trainable_tensors=326, parent_sha256=PARENT_SHA, checkpoint_sha256=WEIGHT_SHA,
        profile=contract['trainability_profile'], num_obs_steps=1, pid=os.getpid(), gpu=local))
    policy.to(f'cuda:{local}').train()
    groups = policy.get_optim_param_groups(lr=1e-5, weight_decay=.03,
                                           apply_decay_on_norm_and_bias=False)
    params = [p for p in policy.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(groups, lr=1e-5, betas=(.9, .95), fused=False)
    ddp = DDP(policy, device_ids=[local], find_unused_parameters=False,
              bucket_cap_mb=128, gradient_as_bucket_view=True)
    own = [None] * world
    dist.all_gather_object(own, os.getpid())
    durations, losses = [], []
    torch.cuda.reset_peak_memory_stats()
    for step in range(plan['warmup'] + plan['measured']):
        if rank == 0:
            active = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid',
                                             '--format=csv,noheader,nounits'], text=True)
            foreign = {int(x.strip()) for x in active.splitlines() if x.strip().isdigit()} - set(own)
            if foreign:
                raise RuntimeError('External GPU process appeared; stop our own benchmark')
        global_indices = [selected[(step * 256 + i) % len(selected)] for i in range(256)]
        total_weight = sum(token['rows'][i]['ce_weight_sum'] for i in global_indices)
        dist.barrier()
        torch.cuda.synchronize()
        start = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.
        for accum in range(plan['accumulation']):
            start_index = (accum * world + rank) * plan['micro']
            indices = global_indices[start_index:start_index+plan['micro']]
            micro_weight = sum(token['rows'][i]['ce_weight_sum'] for i in indices)
            context = ddp.no_sync() if accum + 1 < plan['accumulation'] else nullcontext()
            with context:
                batch = collate(rows, indices, f'cuda:{local}')
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    loss, metrics = ddp(batch)
                actual_weight = float(policy.model.ar_helper._last_ce_cache['token_weights'].sum())
                if actual_weight != micro_weight or not torch.isfinite(loss):
                    raise RuntimeError('Loss mask/weight identity or finite objective failed')
                # Exact global weighted-token mean under DDP averaging, including accumulation.
                scaled = loss * (micro_weight * world / total_weight)
                scaled.backward()
                total_loss += float(loss.detach()) * micro_weight * world / total_weight
        if step == 0:
            if sum(p.grad is not None for p in params) != 326:
                raise RuntimeError('Full planner gradient coverage failed')
            if any(p.grad is not None for p in policy.parameters() if not p.requires_grad):
                raise RuntimeError('A frozen parameter received a gradient')
            publish(args.output / f'gradients_rank{rank}.json', dict(planner_gradient_tensors=326,
                    frozen_with_grad=0, outcome_trained=False,
                    trainable_parameters=sum(p.numel() for p in params)))
        norm = torch.nn.utils.clip_grad_norm_(params, 1.)
        if not torch.isfinite(norm):
            raise RuntimeError('Nonfinite gradient norm')
        optimizer.step()
        torch.cuda.synchronize()
        dist.barrier()
        seconds = time.perf_counter() - start
        if step >= plan['warmup']:
            durations.append(seconds)
            losses.append(total_loss)
        if rank == 0:
            print(json.dumps(dict(step=step+1, seconds=seconds, local_weighted_ce=total_loss,
                                  profile=args.profile, **plan)), flush=True)
    evidence = dict(rank=rank, seconds=durations, local_weighted_ce=losses,
                    peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                    peak_reserved_bytes=torch.cuda.max_memory_reserved())
    gathered = [None] * world
    dist.all_gather_object(gathered, evidence)
    if rank == 0:
        slowest = [max(r['seconds'][i] for r in gathered) for i in range(plan['measured'])]
        seconds = sum(slowest)
        publish(args.output / 'result.json', dict(status='complete', commit=commit,
            completed_utc=datetime.now(timezone.utc).isoformat(), profile=args.profile, **plan,
            measured_samples=256 * plan['measured'], measured_seconds=seconds,
            samples_per_second=256 * plan['measured']/seconds, ranks=gathered,
            token_records=[token['rows'][i] for i in selected], checkpoint_sha256=WEIGHT_SHA,
            input_sha256=receipt['sha256'], accumulation_uses_no_sync=True,
            loss_normalization='global weighted-token mean, not mean of microbatch means',
            scope='Real B-final planner-only graph; 30 old TRAIN rows, three current RGB views; cached I/O',
            full_100task_end_to_end_training=False, deployed_checkpoint_saved=False))
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
