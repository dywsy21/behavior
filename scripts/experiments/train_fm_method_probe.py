"""One bounded, identity-bound FM ablation on the real five-task train loader.

Reuse the frozen A3 trainer and the tested A4 process/checkpoint guards, not a
cache-only imitation of training. Each invocation runs one named arm: actual
GPU gate -> five-update four-GPU save/reload gate -> at most 500 new updates.
All arms weight-warm-start A4 with fresh Adam/scheduler. No automatic retry,
deployment, baseline replacement, data mutation or success-rate claim.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from copy import deepcopy
import gc
import importlib.util
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
from types import SimpleNamespace

HERE = Path(__file__).resolve()
REPO = HERE.parents[2]
METHODS = REPO / 'src/g05/utils/training/fm_training_methods.py'
DEPENDENCY_HELPER = HERE.with_name('train_action_method_probe.py')
BASE = Path('/mnt/sdc1/robodojo/behavior_dev/dual_track_fm_ar_20260913')
LEGACY = Path('/mnt/sdc1/robodojo/behavior_dev/git_worktrees/a4_overnight_20260912/scripts/experiments/overnight_low_fm.py')
LEGACY_SHA = '3e21038efc97e68ccfb79b87f584025690ee035f43fd246b1c34b2e9a0ff5342'
PARENT = Path('/mnt/sdc1/robodojo/behavior_dev/overnight_a4_20260912/formal/checkpoints/step_2500.pt')
PARENT_SHA = '6186704788c27c9fae3502c884df0e259de5242ee8690fe578dcbc1f2632f269'
TRIALS = {
    'control': dict(action_lr=1e-5, lora_lr=1e-5, time_sampling='iid', execution_weight=1.),
    'ae_lr2x': dict(action_lr=2e-5, lora_lr=1e-5, time_sampling='iid', execution_weight=1.),
    'beta_stratified': dict(action_lr=1e-5, lora_lr=1e-5, time_sampling='beta_stratified', execution_weight=1.),
    'exec_weight2': dict(action_lr=1e-5, lora_lr=1e-5, time_sampling='iid', execution_weight=2.),
}


def sha(path):
    import hashlib
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    loaded = importlib.util.module_from_spec(spec)
    sys.modules[name] = loaded
    spec.loader.exec_module(loaded)
    return loaded


def legacy(spec=None):
    if sha(LEGACY) != LEGACY_SHA:
        raise RuntimeError('Frozen A4 process/checkpoint guard code changed')
    old = module(LEGACY, 'fm_method_legacy_guard')
    old.PARENT, old.PARENT_SHA = PARENT, PARENT_SHA
    old.MAX_STEPS = 500
    if spec is not None:
        old.ROOT = Path(spec['output'])
    return old


def validate_spec(spec):
    if (spec['trial'] not in TRIALS or spec['settings'] != TRIALS[spec['trial']]
            or spec['max_updates'] != 500 or spec['smoke_updates'] != 5
            or spec['seed'] != 41 or spec['parent_sha256'] != PARENT_SHA
            or spec['methods_sha256'] != sha(METHODS) or spec['entry_sha256'] != sha(HERE)
            or Path(spec['output']).parent != BASE):
        raise RuntimeError('Method recipe/input identity or budget mismatch')
    commit = subprocess.check_output(['git', '-C', str(REPO), 'rev-parse', 'HEAD'], text=True).strip()
    if commit != spec['commit']:
        raise RuntimeError('Do not switch a running recipe worktree')
    if subprocess.check_output(['git', '-C', str(REPO), 'status', '--porcelain'], text=True).strip():
        raise RuntimeError('Recipe worktree is dirty')
    if spec.get('after_fm') and spec.get('after_action'):
        raise RuntimeError('Use one serial predecessor, not two conflicting queues')
    if spec.get('after_fm') or spec.get('after_action'):
        if (spec.get('dependency_helper_sha256') != sha(DEPENDENCY_HELPER)
                or spec.get('initialization') != 'a4' or spec.get('parent_path') != str(PARENT)):
            raise RuntimeError('Serial helper or independent A4 initialization changed')


def dependency_helpers():
    # Import at the lifecycle boundary, not while this module is imported by
    # the action trainer. This avoids a circular top-level import.
    import train_action_method_probe
    return train_action_method_probe


def settings_for(methods, spec):
    selected = spec['settings']
    return methods.FMTrainingMethods(time_sampling=selected['time_sampling'],
        execution_horizon=16, execution_weight=selected['execution_weight'])


def overrides(spec, phase, spec_sha):
    if phase not in {'smoke', 'formal'}:
        raise ValueError(phase)
    selected = spec['settings']
    output = Path(spec['output']) / phase
    return ['model.batch_size=2', 'model.grad_accumulation_steps=2',
        'model.enable_bf16_training=true', 'model.model_weights_to_bf16=false',
        'model.num_workers=4', 'model.prefetch_factor=2', 'model.max_epochs=null',
        f'model.max_steps={5 if phase == "smoke" else 500}',
        f'model.learning_rate={selected["action_lr"]}',
        f'++model.backbone_lr_multiplier={selected["lora_lr"] / selected["action_lr"]}',
        'model.warmup_steps=50', 'model.lr_min_ratio=0.1', 'resume_ckpt=null',
        'batch_size_val=1', 'checkpointing_steps=500', 'eval_steps=100',
        'logger.type=wandb', 'logger.mode=offline', 'logger.workspace=null', 'logger.log_steps=10',
        'logger.task=r1pro_memlite_skill_fm_a3_episodecoverage',
        f'logger.experiment_name=fm-method-{spec["trial"]}-{phase}-20260913',
        f'hydra.run.dir={output}', f'+fm_method_recipe_sha256={spec_sha}']


def arguments(old, spec, phase, spec_path):
    return SimpleNamespace(stage='low_fm_lora_history',
        task_config=str(old.SOURCE / 'configs/task/r1pro_memlite_skill_fm_a3_episodecoverage.yaml'),
        source_root=str(old.SOURCE), sampling_index=str(old.POOLS / 'sampling_index.json'),
        eval_manifest=str(old.POOLS / 'eval_manifest.json'), run_dir=str(old.ROOT / phase),
        low_initial_checkpoint=str(PARENT), high_initial_checkpoint=None,
        official_asset_receipt=str(old.ROOT / 'gate/asset_receipt.json'),
        resume_ckpt=None, resume_receipt=None, num_obs_steps=6, nproc_per_node=4,
        seed=41, override=overrides(spec, phase, sha(spec_path)))


def verify_parent(model, parent):
    import torch
    actual, expected = model.state_dict(), parent['model_state_dict']
    if parent['step'] != 2500 or set(actual) != set(expected) or len(actual) != 1138:
        raise RuntimeError('Incomplete A4 parent restoration')
    if sum('lora_' in name for name in actual) != 192:
        raise RuntimeError('Missing A4 LoRA entries')
    for name, tensor in actual.items():
        left, right = tensor.detach().cpu(), expected[name]
        if left.shape != right.shape or left.dtype != right.dtype or not torch.isfinite(left).all():
            raise RuntimeError('Invalid A4 state: ' + name)
        if not torch.equal(left.contiguous().reshape(-1).view(torch.uint8),
                           right.contiguous().reshape(-1).view(torch.uint8)):
            raise RuntimeError('A4 state bytes not restored: ' + name)


def gpu_gate(old, spec):
    import torch
    from torch import distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    from omegaconf import OmegaConf
    from g05.utils.common.pytorch_utils import dict_apply
    from g05.utils.config.config_resolvers import register_default_resolvers
    from g05.utils.training.coordination_grad_audit import CoordinationGradientAudit
    from preflight_memlite_skillfm_gpu import _load_and_configure_gpu_model, _stage_contract

    output = old.ROOT / 'gate'
    output.mkdir(exist_ok=False)
    register_default_resolvers()
    receipt = old.read(old.INPUT / 'result.json')
    if (receipt['status'] != 'complete' or not receipt['train_only'] or not receipt['all_five_tasks']
            or receipt['actual_batch_file_sha256'] != sha(old.INPUT / 'actual_cpu_batches.pt')
            or receipt['processor_config_sha256'] != sha(old.INPUT / 'diagnostic_processor_config.yaml')
            or sha(PARENT) != PARENT_SHA):
        raise RuntimeError('Audited original training inputs or A4 parent changed')
    cfg = OmegaConf.load(old.INPUT / 'diagnostic_processor_config.yaml')
    cfg.model.pretrained_ckpt, cfg.resume_ckpt = str(PARENT), None
    cfg.model.learning_rate = spec['settings']['action_lr']
    cfg.model.backbone_lr_multiplier = spec['settings']['lora_lr'] / cfg.model.learning_rate
    torch.set_num_threads(4)
    torch.manual_seed(41)
    torch.cuda.set_device(0)
    device = torch.device('cuda:0')
    model, parent, _, _, _ = _load_and_configure_gpu_model(cfg, PARENT, device, _stage_contract(cfg, 'b'))
    verify_parent(model, parent)
    del parent
    gc.collect()
    model.apply_fp32_params()
    methods = module(METHODS, 'fm_gpu_gate_methods')
    batches = torch.load(old.INPUT / 'actual_cpu_batches.pt', map_location='cpu', weights_only=False)
    batch = dict_apply(deepcopy(batches[0]), lambda x: x.to(device) if isinstance(x, torch.Tensor) else x)
    # Compare actual, unmodified evaluation computation before any optimizer
    # update. The outer hook must not enter its training-only inner scope.
    model.eval()
    reference = None
    for option in (None, methods.FMTrainingMethods(), settings_for(methods, spec)):
        with torch.random.fork_rng(devices=[0]):
            torch.manual_seed(771)
            torch.cuda.manual_seed_all(771)
            scope = nullcontext() if option is None else methods.fm_policy_training_methods(type(model), option)
            with scope, torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                value, metrics = model(batch)
            if not torch.isfinite(value) or not torch.equal(value, metrics['fm_loss']):
                raise RuntimeError('Reference FM metric invalid')
            rng = (torch.random.get_rng_state(), torch.cuda.get_rng_state())
            if reference is None:
                reference = (value.detach().clone(), rng)
            elif not torch.equal(value, reference[0]) or not all(torch.equal(x, y) for x, y in zip(rng, reference[1])):
                raise RuntimeError('Method hook changed actual reference eval loss/RNG')
    old.publish(output / 'reference_metric_gate.json', dict(passed=True, actual_model_forwards=3,
        reference_fm_loss=float(reference[0]), eval_loss_and_rng_bitwise_equal=True,
        selected_methods=settings_for(methods, spec).as_dict(), no_optimizer_updates_yet=True))
    print('Actual A4 restoration and unchanged reference evaluation passed', flush=True)
    del batch, reference, value, metrics
    audit = CoordinationGradientAudit(model, expected_groups=['action_expert', 'vlm_lora'],
        frozen_prefixes=list(cfg.model.model_arch.coordination_train.frozen_group_prefixes),
        output_dir=output, rank=0)
    dist.init_process_group('nccl')
    ddp = DDP(model, device_ids=[0], find_unused_parameters=True, gradient_as_bucket_view=True, bucket_cap_mb=128)
    audit.after_distributed_initialization()
    groups = model.get_optim_param_groups(lr=cfg.model.learning_rate,
        weight_decay=float(cfg.model.weight_decay), apply_decay_on_norm_and_bias=False,
        backbone_lr_multiplier=cfg.model.backbone_lr_multiplier, vision_lr_multiplier=1.)
    optimizer = torch.optim.AdamW(groups, lr=cfg.model.learning_rate, betas=tuple(cfg.model.betas))
    expected_params = {id(p) for p in model.parameters() if p.requires_grad}
    if {id(p) for g in optimizer.param_groups for p in g['params']} != expected_params:
        raise RuntimeError('Optimizer does not cover exactly the declared trainable parameters')
    for group in optimizer.param_groups:
        if group['params']:
            expected_rate = (spec['settings']['action_lr'] if group['name'].startswith('action_')
                             else spec['settings']['lora_lr'])
            if group['lr'] != expected_rate:
                raise RuntimeError('Incorrect effective parameter group learning rate')
    model.train()
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.reset_peak_memory_stats()
    losses = []
    with methods.fm_policy_training_methods(type(model), settings_for(methods, spec)):
        for i, original in enumerate(batches[:4]):
            batch = dict_apply(deepcopy(original), lambda x: x.to(device) if isinstance(x, torch.Tensor) else x)
            update = (i + 1) % 2 == 0
            with (nullcontext() if update else ddp.no_sync()):
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    loss, metrics = ddp(batch)
                if not torch.isfinite(loss) or not torch.equal(loss, metrics['fm_loss']):
                    raise RuntimeError('Training objective is not finite selected-method FM')
                (loss / 2).backward()
            if any(not torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):
                raise RuntimeError('Nonfinite gradient')
            facts = audit.after_backward() if i == 1 else None
            if update:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.model.max_grad_norm)
                optimizer.step()
                if i == 1:
                    audit.after_optimizer_step(facts, step=1,
                        effective_lrs=[float(g['lr']) for g in optimizer.param_groups])
                optimizer.zero_grad(set_to_none=True)
            losses.append(float(loss.detach()))
            print(json.dumps(dict(microbatches=i + 1, actual_updates=(i + 1) // 2,
                                  training_objective=losses[-1])), flush=True)
            del batch, loss, metrics
    audit._assert_frozen_unchanged()
    audit.require_verified_update()
    if (len(optimizer.state) != 504 or {int(s['step']) for s in optimizer.state.values()} != {2}
            or any(not torch.isfinite(p).all() for p in model.parameters())):
        raise RuntimeError('Actual two-update gate failed')
    old.publish(output / 'asset_receipt.json', dict(passed=True, mode='gpu_one_step',
        formal_training=False, checkpoint_saved=False, checkpoint=str(PARENT),
        checkpoint_contract={'checkpoint_sha256': PARENT_SHA},
        official_asset_bindings=dict(hf_processor_path=str(cfg.model.model_arch.hf_processor_path),
                                    action_tokenizer_ckpt=str(cfg.model.model_arch.AT_CONFIG.ckpt_dir)),
        actual_optimizer_steps_so_far=2, actual_microbatches_so_far=4,
        source_root_sha256=old.SOURCE_SHA, methods_sha256=sha(METHODS),
        full_parent_state_entries_bitwise_equal=1138, lora_entries_bitwise_equal=192))
    old.publish(output / 'result.json', dict(passed=True, actual_updates=2, microbatches=4,
        training_objective_losses=losses, methods=settings_for(methods, spec).as_dict(),
        peak_reserved_bytes=torch.cuda.max_memory_reserved(), frozen_bitwise_unchanged=True,
        reference_eval_unchanged=True, optimizer_groups=[dict(name=g['name'], lr=g['lr'],
        n_parameters=len(g['params'])) for g in optimizer.param_groups], checkpoint_saved=False,
        no_method_efficacy_or_success_rate_claim=True))
    dist.destroy_process_group()


def train_phase(old, spec, spec_path, phase):
    import coordination_launcher as launcher
    from g05.utils.training.coordination_runtime import (CoordinationLock, mark_running,
        refuse_if_live_run, update_run_receipt)
    args = arguments(old, spec, phase, spec_path)
    plan = launcher.build_plan(args)
    plan['source_root'] = str(old.SOURCE)
    command = launcher.training_command(args, plan)
    index = command.index(str(old.SOURCE / 'scripts/finetune.py'))
    command[index:index + 1] = [str(HERE), 'entry', '--spec', str(spec_path)]
    if phase == 'formal' and not old.read(old.ROOT / 'smoke_checkpoint_inspection.json')['passed']:
        raise RuntimeError('Actual four-GPU checkpoint smoke is required')
    gpu = old.gpu_budget(range(4))
    old.publish(old.ROOT / f'{phase}_plan.json', dict(plan=plan, command=command,
        gpu_before=gpu, time=old.now(), method_spec_sha256=sha(spec_path)))
    env = launcher.trainer_config_preflight_environment(plan, old.ROOT / f'{phase}_config_preflight.json', command)
    old.execute(command, env, old.ROOT / f'{phase}_config_preflight.log', 900)
    run = old.ROOT / phase
    refuse_if_live_run(run)
    launcher.prepare_run_receipt(args, plan=plan, command=command)
    with CoordinationLock(purpose='training') as lock:
        def heartbeat(child):
            lock.heartbeat(metadata=dict(run_dir=str(run), pid=child.pid, source_root=str(old.SOURCE)))
            mark_running(run, pid=child.pid, pgrp=child.pid, config_sha256=plan['identity']['config_sha256'])
            old.publish(old.ROOT / 'status.json', dict(state='running', phase=phase, time=old.now(),
                supervisor_pid=os.getpid(), trainer_pid=child.pid, run=str(run),
                max_steps=5 if phase == 'smoke' else 500, trial=spec['trial']))
        try:
            old.execute(command, launcher.child_environment(plan), old.ROOT / f'{phase}.log',
                        1800 if phase == 'smoke' else None, heartbeat)
            update_run_receipt(run, state='complete', returncode=0, source_root=str(old.SOURCE))
        except BaseException as error:
            update_run_receipt(run, state='failed', error=str(error), source_root=str(old.SOURCE))
            raise
    old.inspect_checkpoint(phase, 5 if phase == 'smoke' else 500)


def supervise(old, spec, spec_path):
    try:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        old.bootstrap()
        if spec.get('after_fm') or spec.get('after_action'):
            dependency_helpers().wait_for_dependency(spec, old)
        old.publish(old.ROOT / 'status.json', dict(state='preflight', time=old.now()))
        old.gpu_budget([1])
        env = old.environment()
        env['CUDA_VISIBLE_DEVICES'] = '1'
        old.execute([old.PYTHON, '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=1',
            str(HERE), 'gate', '--spec', str(spec_path)], env, old.ROOT / 'gate.log', 1800)
        if not old.read(old.ROOT / 'gate/result.json')['passed']:
            raise RuntimeError('Actual GPU method gate failed')
        train_phase(old, spec, spec_path, 'smoke')
        train_phase(old, spec, spec_path, 'formal')
        old.publish(old.ROOT / 'status.json', dict(state='complete', time=old.now(),
            trial=spec['trial'], verified_optimizer_steps=500,
            checkpoint=str(old.ROOT / 'formal/checkpoints/step_500.pt'), success_rate_claim=False,
            next='Paired reference/action and closed-loop comparison; no automatic extra training'))
    except BaseException as error:
        old.publish(old.ROOT / 'status.json', dict(state='failed', time=old.now(),
            error=f'{type(error).__name__}: {error}', automatic_retry=False, prior_evidence_preserved=True))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('start', 'supervise', 'gate', 'entry'))
    parser.add_argument('--trial', choices=tuple(TRIALS))
    parser.add_argument('--output', type=Path)
    parser.add_argument('--spec', type=Path)
    parser.add_argument('--after-fm-run', type=Path)
    parser.add_argument('--after-action-run', type=Path)
    args, remainder = parser.parse_known_args()
    if args.mode == 'start':
        if remainder or not args.trial or args.output is None or args.output.parent != BASE:
            raise ValueError('Start needs one named trial and a new direct child of the experiment root')
        old = legacy()
        if args.after_fm_run and args.after_action_run:
            raise ValueError('Select only one explicitly declared predecessor')
        dependency = dependency_helpers() if args.after_fm_run or args.after_action_run else None
        spec = dict(commit=subprocess.check_output(['git', '-C', str(REPO), 'rev-parse', 'HEAD'], text=True).strip(),
            trial=args.trial, settings=TRIALS[args.trial], output=str(args.output),
            entry_sha256=sha(HERE), methods_sha256=sha(METHODS), parent_sha256=PARENT_SHA,
            max_updates=500, smoke_updates=5, seed=41, high_unchanged=True, train_only=True,
            fresh_optimizer_and_scheduler=True, equivalent_resume=False,
            initialization='a4', parent_path=str(PARENT),
            dependency_helper_sha256=sha(DEPENDENCY_HELPER) if dependency else None,
            after_fm=dependency.dependency_identity(args.after_fm_run, 'fm') if args.after_fm_run else None,
            after_action=dependency.dependency_identity(args.after_action_run, 'action') if args.after_action_run else None)
        validate_spec(spec)
        args.output.mkdir(parents=True, exist_ok=False)
        spec_path = args.output / 'method_spec.json'
        old.publish(spec_path, spec)
        with (args.output / 'supervisor.log').open('x') as stream:
            child = subprocess.Popen([old.PYTHON, str(HERE), 'supervise', '--spec', str(spec_path)],
                env=old.environment(), stdin=subprocess.DEVNULL, stdout=stream,
                stderr=subprocess.STDOUT, start_new_session=True)
        receipt = dict(supervisor_pid=child.pid, time=old.now(), spec=str(spec_path),
                       method_spec_sha256=sha(spec_path), commit=spec['commit'], trial=args.trial)
        old.publish(args.output / 'launch.json', receipt)
        print(json.dumps(receipt), flush=True)
        return
    if args.spec is None or not args.spec.is_absolute() or (remainder and args.mode != 'entry'):
        raise ValueError('A pinned absolute method spec is required')
    spec = json.loads(args.spec.read_text())
    validate_spec(spec)
    old = legacy(spec)
    if args.mode == 'supervise':
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    old.bootstrap()
    if args.mode == 'supervise':
        supervise(old, spec, args.spec)
    elif args.mode == 'gate':
        gpu_gate(old, spec)
    else:
        # This hash is part of the original trainer's config identity and thus
        # its saved checkpoint identity. Do not run an unbound method hook.
        if f'+fm_method_recipe_sha256={sha(args.spec)}' not in remainder:
            raise RuntimeError('Trainer overrides must bind the complete method recipe')
        from g05.models.g05.g05_policy_memlite_skill_fm import G05PolicyMEMLiteSkillFM
        methods = module(METHODS, 'fm_method_training_entry')
        sys.argv = [str(old.SOURCE / 'scripts/finetune.py'), *remainder]
        with methods.fm_policy_training_methods(G05PolicyMEMLiteSkillFM, settings_for(methods, spec)):
            runpy.run_path(sys.argv[0], run_name='__main__')


if __name__ == '__main__':
    main()
