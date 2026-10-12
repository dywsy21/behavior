"""Pinned author's model/math with native2026 IO and explicit HOST transport.

Only the orchestration is new. FM/FAST/stage training, augmentation, AdamW,
EMA and stage construction match the registered eight-A800 engineering probe.
"""
import dataclasses
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from .checkpoint import materialize_host, restore_full, save_full
from .dataset import NativeV3Dataset, collate_native, worker_init
from .evaluation import FixedEvalPlan
from .inference import checked_sources
from .release import RELEASE_SHA256, sha256
from .sampling import BalancedPlan, PlanChunks, digest_json
from .training_io import EvalAccumulator, NativeBatchTransform, combine_chunks, unpack_device_batch
from .runner import write_json


class TrainingEngine:
    def __init__(self, root, repo, directory, recipe, *, engineering=False):
        self.root, self.repo, self.directory = map(lambda p: Path(p).resolve(), (root, repo, directory))
        self.recipe, self.engineering = recipe, engineering
        self.reference = self.root / 'src/champion2026-native-v1'
        self.openpi = self.root / 'src/openpi2026-native-v1'
        self.parent = self.root / 'models/champion2025/shared50-11490241'
        self.assets = self.parent / 'assets/IliaLarchenko/behavior_224_rgb'
        receipts = self.root / 'runs/champion_assets_20261011'
        checked_sources(self.reference, self.openpi, self.repo)
        for path in (self.openpi / 'src', self.reference / 'src'):
            sys.path.insert(0, str(path))
        self.integration_commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=self.repo, text=True).strip()
        if subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'], cwd=self.repo, text=True):
            raise ValueError('Use an immutable clean Git worktree')
        self.verified_file = receipts / 'shared50-verified-v1.json'
        if sha256(self.verified_file) != 'd0ca3f072c11967f3581827151c7242f53e53897650a80f2263f0803fd5b1e8a':
            raise ValueError('Unexpected shared50 parent receipt')
        verified = json.loads(self.verified_file.read_text())
        for row in verified['files']:
            path = self.parent / row['path']
            if path.stat().st_size != row['size'] or sha256(path) != row['sha256']:
                raise ValueError('Parent asset bytes changed: ' + row['path'])
        layout_file = receipts / 'task-layout-admitted-v2.json'
        if sha256(layout_file) != 'd2257aec60b4653808c7b214b33795408c39b69f7515b064ed6314154ae72041':
            raise ValueError('Changed admitted task/stage mapping')
        self.layout = json.loads(layout_file.read_text())['layout']
        self.train_data = NativeV3Dataset(self.root / 'datasets/memlite-stage1-20260930-v4', 'train')
        self.eval_data = NativeV3Dataset(self.root / 'datasets/memlite-stage1-20260930-v4', 'eval')
        self.plan = BalancedPlan(self.train_data, updates=8000, batch_size=256, seed=17)
        self.eval_plan = FixedEvalPlan(self.eval_data, self.layout['dataset_to_model'], per_task=16)
        if self.plan.fingerprint != 'fa1dda371465225506eb662282cad7204c72a8ae4504d973dde86995d654ab0e':
            raise ValueError('Changed exhaustively verified sampling plan')
        if self.eval_plan.fingerprint != '3a10106e2869897be9914ece6d0b9ed7244b61b439292c9859d54752cc2665d7':
            raise ValueError('Changed fixed original-heldout plan')
        self._setup_graph()
        self.contract = json.loads(json.dumps(dict(
            purpose='ENGINEERING_ONLY_NOT_FOR_DEPLOYMENT' if engineering else 'CHAMPION100_FORMAL_SFT',
            integration_commit=self.integration_commit, recipe_sha256=digest_json(recipe),
            parent_receipt_sha256=sha256(self.verified_file), release_sha256=RELEASE_SHA256,
            layout_sha256=digest_json(self.layout), model=dataclasses.asdict(self.config.model),
            optimizer=dataclasses.asdict(self.config.optimizer), lr=dataclasses.asdict(self.config.lr_schedule),
            ema_decay=self.config.ema_decay, flow_samples=self.config.num_flow_samples,
            sampler_plan_sha256=self.plan.fingerprint, eval_plan_sha256=self.eval_plan.fingerprint,
            environment={key: os.environ.get(key) for key in ('XLA_FLAGS', 'NCCL_P2P_DISABLE',
                'NCCL_IB_DISABLE', 'NCCL_SOCKET_IFNAME', 'OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS')},
            jax_version=self.jax.__version__, numpy_version=np.__version__)))
        self.last_saved, self.iterator = None, None

    def _setup_graph(self):
        import jax
        import jax.numpy as jnp
        import flax.nnx as nnx
        from openpi.training import sharding
        from openpi.training.optimizer import CosineDecaySchedule
        from b1k.shared import normalize
        from b1k.training import config as config_lib
        from b1k.transforms_normalize import NormalizeWithPerTimestamp
        from b1k.transforms import TokenizeFASTActions
        from .jax_transport import place_from_host, check_collective_and_gradient
        from .optimizer import RowScaledAdamW
        self.jax, self.nnx, self.sharding = jax, nnx, sharding
        if len(jax.devices()) != 8 or any(device.platform != 'gpu' for device in jax.devices()):
            raise ValueError('Exactly eight CUDA devices required')
        self.mesh = sharding.make_mesh(8)
        transport = check_collective_and_gradient(self.mesh)
        write_json(self.directory / 'transport.json', transport)
        jax.config.update('jax_compilation_cache_dir', str(self.directory / 'jax-cache'))
        self.replicated = jax.sharding.NamedSharding(self.mesh, jax.sharding.PartitionSpec())
        self.data_sharding = jax.sharding.NamedSharding(self.mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
        self.init_key = place_from_host(np.array([0, 17], np.uint32), self.replicated)
        self.train_key = place_from_host(np.array([0, 19], np.uint32), self.replicated)
        # Keep evaluation RNG independent of training cursor and augmentation.
        self.eval_key = place_from_host(np.array([0, 31], np.uint32), self.replicated)
        self.fold_eval_key = jax.jit(lambda key, group, batch: jax.random.fold_in(
            jax.random.fold_in(key, group), batch),
            in_shardings=(self.replicated, self.replicated, self.replicated), out_shardings=self.replicated)
        self.stats = normalize.load(self.assets)
        tokenizer_file = self.assets / 'fast_tokenizer/processing_action_tokenizer.py'
        if sha256(tokenizer_file) != '6f021ca1f4c1b194ab6fa399d80baf3d642eadb17efb8f73301e4ac401522c20':
            raise ValueError('Unreviewed tokenizer code')
        self.transform = NativeBatchTransform(self.layout,
            NormalizeWithPerTimestamp(self.stats, use_per_timestamp=True),
            TokenizeFASTActions(str(self.assets / 'fast_tokenizer'), [(0, 6), (7, 23)], 200,
                                norm_stats=self.stats, use_per_timestamp=True))
        spec = importlib.util.spec_from_file_location('champion_formal_core', self.reference / 'scripts/train.py')
        self.core = importlib.util.module_from_spec(spec); spec.loader.exec_module(self.core)
        base = config_lib.get_config('pi_behavior_b1k_fast')
        self.expected_hashes = {}
        engine = self
        class ParentLoader:
            def load(self, expected):
                return engine._load_expanded(expected)
        self.config = dataclasses.replace(base,
            model=dataclasses.replace(base.model, num_tasks=100, task_num_stages=tuple(self.layout['num_stages'])),
            weight_loader=ParentLoader(), batch_size=256, num_flow_samples=4, fsdp_devices=8,
            optimizer=RowScaledAdamW(), freeze_filter=nnx.Nothing, wandb_enabled=False, ema_decay=.99,
            lr_schedule=CosineDecaySchedule(warmup_steps=300, peak_lr=2.5e-5, decay_steps=8000, decay_lr=2.5e-6))
        if (self.config.model.fast_loss_weight != .05 or self.config.model.subtask_loss_weight != .1 or
                not self.config.model.use_fast_auxiliary or self.config.model.use_knowledge_insulation):
            raise ValueError('Author loss configuration differs from approved mathematical recipe')

    def _load_expanded(self, expected):
        from flax.traverse_util import flatten_dict, unflatten_dict
        from openpi.models.model import restore_params
        from .contracts import append_embeddings
        loaded = flatten_dict(restore_params(self.parent / 'params', restore_type=np.ndarray))
        shapes = flatten_dict(expected)
        if loaded.keys() != shapes.keys():
            raise ValueError('Parent/model graph key coverage changed')
        for key, old_rows in ((('task_embeddings', 'embedding'), 50), (('task_stage_embeddings', 'embedding'), 596)):
            shape = shapes[key].shape
            initializer = self.nnx.Embed(num_embeddings=shape[0], features=shape[1], rngs=self.nnx.Rngs(17 + old_rows))
            fresh = np.asarray(initializer.embedding.value).astype(loaded[key].dtype)
            loaded[key] = append_embeddings(loaded[key], fresh, old_rows)
        ignored = ('action_correlation_cholesky', 'L_spatial', 'L_temporal', 'cached_num_inpaint_actions',
                   'cached_input_action_dim', 'cached_Sigma_uo_Sigma_oo_inv', 'cached_L_cond_free',
                   'cached_Sigma_ou_Sigma_uu_inv', 'cached_L_cond_inp')
        for key, value in loaded.items():
            if value.shape != shapes[key].shape:
                raise ValueError('Parent/model parameter shape changed: ' + str(key))
            if not any(name in key for name in ignored):
                self.expected_hashes[key] = hashlib.sha256(value.tobytes()).hexdigest()
        return unflatten_dict(loaded)

    def initialize(self, receipt=None, *, engineering_source_contract=None):
        jax = self.jax
        with self.sharding.set_mesh(self.mesh):
            self.state, self.state_sharding = self.core.init_train_state(
                self.config, self.init_key, self.mesh, resume=receipt is not None, norm_stats=self.stats)
            if receipt is not None:
                # An explicit engineering cold replay may read a prior probe's
                # full state. A formal run can ONLY restore its identical contract.
                if engineering_source_contract is not None and not self.engineering:
                    raise ValueError('Engineering checkpoint cannot seed a formal run')
                restored = restore_full(receipt['path'],
                    dict(state=self.state, train_rng=jax.ShapeDtypeStruct((2,), np.uint32)),
                    dict(state=self.state_sharding, train_rng=self.replicated),
                    contract=engineering_source_contract or self.contract, plan=self.plan,
                    manifest_sha256=receipt['manifest_sha256'])
                self.state, self.train_key = restored['state'], restored['train_rng']
                self.last_saved = receipt if engineering_source_contract is None else None
            else:
                from flax.traverse_util import flatten_dict
                actual = flatten_dict(self.state.params.to_pure_dict())
                for key, expected in self.expected_hashes.items():
                    if hashlib.sha256(materialize_host(actual[key]).tobytes()).hexdigest() != expected:
                        raise ValueError('Model parameter transport failed: ' + str(key))
                if len(self.expected_hashes) != 74:
                    raise ValueError('Unexpected parent parameter verification coverage')
            jax.block_until_ready(self.state)
            def step(key, state, batch):
                return self.core.train_step(self.config, key, state, unpack_device_batch(batch))
            self.compiled_step = jax.jit(step,
                in_shardings=(self.replicated, self.state_sharding, self.data_sharding),
                out_shardings=(self.state_sharding, self.replicated), donate_argnums=(1,))
            def evaluate(key, params, batch):
                from .losses import compute_eval_losses
                model = self.nnx.merge(self.state.model_def, params)
                observation, actions = unpack_device_batch(batch)
                return compute_eval_losses(model, key, observation, actions, num_flow_samples=4)
            self.compiled_eval = jax.jit(evaluate,
                in_shardings=(self.replicated, self.state_sharding.ema_params, self.data_sharding),
                out_shardings=self.data_sharding)

    @property
    def completed(self):
        return int(self.state.step)

    def _loader(self, dataset, batches, workers=32):
        from torch.utils.data import DataLoader
        return DataLoader(dataset, batch_sampler=batches, num_workers=workers,
            collate_fn=collate_native, worker_init_fn=worker_init, multiprocessing_context='spawn', prefetch_factor=1)

    def step(self, update):
        from .jax_transport import place_from_host
        if update != self.completed:
            raise ValueError('Stale optimizer cursor')
        if self.iterator is None:
            self.iterator = iter(self._loader(self.train_data, PlanChunks(self.plan, start=update)))
        started = time.monotonic()
        chunks = [next(self.iterator) for _ in range(8)]
        source, audit = combine_chunks(chunks, self.plan.batch(update), split='train')
        counts = np.asarray(audit['task_counts'])
        if not np.all((counts == 2) | (counts == 3)) or sum(counts == 3) != 56:
            raise ValueError('Mixed-100-task batch balance failed')
        batch = place_from_host(self.transform(source), self.data_sharding)
        prepared = time.monotonic()
        with self.sharding.set_mesh(self.mesh):
            self.state, metrics = self.compiled_step(self.train_key, self.state, batch)
            self.jax.block_until_ready((self.state, metrics))
        values = {key: float(value) for key, value in metrics.items()}
        values.update(seconds=time.monotonic() - started, input_seconds=prepared - started,
                      distinct_tasks=100, learning_rate=float(self.config.lr_schedule.create()(update)))
        with (self.directory / 'sampling-journal.jsonl').open('a') as stream:
            stream.write(json.dumps(dict(completed_updates=self.completed, **audit)) + '\n')
            stream.flush(); os.fsync(stream.fileno())
        return values

    def evaluate(self):
        from .jax_transport import place_from_host
        groups = list(self.eval_plan.indices)
        batches = [indices.tolist() for group in groups for indices in self.eval_plan.batches(group)]
        accumulator = EvalAccumulator(self.layout['dataset_to_model'])
        # Keep only small per-example metrics, never all 1600 RGB observations.
        for number, chunk in enumerate(self._loader(self.eval_data, batches, workers=8)):
            source, _ = combine_chunks([chunk], batches[number], split='eval')
            batch = place_from_host(self.transform(source), self.data_sharding)
            # All replicas start from checked HOST keys; no GPU0-key resharding.
            group_number, batch_number = divmod(number, 25)
            key = self.fold_eval_key(self.eval_key,
                place_from_host(np.asarray(group_number, np.uint32), self.replicated),
                place_from_host(np.asarray(batch_number, np.uint32), self.replicated))
            with self.sharding.set_mesh(self.mesh):
                metrics = self.compiled_eval(key, self.state.ema_params, batch)
                self.jax.block_until_ready(metrics)
            accumulator.add(batches[number], source['task_id'], self.jax.tree.map(materialize_host, metrics))
        result = dict(completed_updates=self.completed, weights='EMA',
            eval_plan_sha256=self.eval_plan.fingerprint, rows=accumulator.rows, metrics=accumulator.finish(),
            scope='fixed_original_heldout_loss_not_physical_Q_or_SR')
        path = self.directory / f'eval-{self.completed:08d}-{time.time_ns()}.json'
        write_json(path, result)
        return {f'eval/{group}/{key}': value for group, row in result['metrics'].items()
                for key, value in row.items()}

    def save(self):
        if self.last_saved and self.last_saved['completed_updates'] == self.completed:
            return self.last_saved
        self.last_saved = save_full(self.directory / 'checkpoints',
            dict(state=self.state, train_rng=self.train_key), completed=self.completed,
            contract=self.contract, cursor=self.plan.cursor(self.completed))
        return self.last_saved

    def export(self, receipt):
        from .export import export_ema
        return export_ema(self.directory / 'exports', self.state, receipt=receipt,
                          assets=self.assets, layout=self.layout, contract=self.contract)
