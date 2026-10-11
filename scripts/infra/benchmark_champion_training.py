"""Short, disposable 8-A800 full-gradient timing from verified champion weights.

Uses the pinned author's actual init_train_state/train_step, FM+FAST+stage
losses and AdamW/EMA. Default timings save no checkpoint. The explicitly
selected resume probe saves a NOT-FOR-DEPLOYMENT engineering full-state
checkpoint after two updates and checks the third update after restoration.
"""
import argparse
import dataclasses
import functools
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    def checkpoint_progress(row):
        if row['bytes'] >= 64 << 20:
            print(json.dumps(dict(checkpoint_progress=row)),flush=True)
    ap = argparse.ArgumentParser(description=__doc__)
    for key in ('reference','layout','patch','openpi-reference','openpi-patch','checkpoint','verified','probe','output'):
        ap.add_argument('--'+key, type=Path, required=True)
    ap.add_argument('--batch-size', type=int, choices=(32, 64, 128, 256), required=True)
    ap.add_argument('--flow-samples', type=int, choices=(1, 4, 15), default=4)
    ap.add_argument('--release', type=Path, help='Optional actual V3 stream; otherwise cached compute only')
    ap.add_argument('--workers', type=int, choices=(8,16,32), default=32)
    ap.add_argument('--input-only', action='store_true', help='Profile the same real input path without loading/updating the model')
    ap.add_argument('--compact-rgb', action='store_true', help='Keep native uint8 through checked placement; resize/normalize on each assigned GPU')
    ap.add_argument('--resume-probe', action='store_true', help='Three-update balanced/row-Adam full-state save/restore correctness probe; not formal training')
    ap.add_argument('--resume-probe-from', type=Path, help='Cold process: restore the passed engineering checkpoint and replay ONLY update3')
    ap.add_argument('--resume-reference-sha256', help='SHA256 of the original uninterrupted-third-update.json')
    ap.add_argument('--freeze-vision', action='store_true')
    ap.add_argument('--warmup', type=int, default=2)
    ap.add_argument('--measure', type=int, default=5)
    args = ap.parse_args()
    cold_resume = args.resume_probe_from is not None
    if cold_resume and (not args.resume_probe or not args.resume_reference_sha256):
        raise ValueError('Cold replay requires the explicit resume probe and bound original reference')
    if args.resume_probe and (args.release is None or not args.compact_rgb or args.input_only or
                             args.freeze_vision or args.batch_size != 256 or args.flow_samples != 4 or
                             args.warmup != 2 or args.measure != 1):
        raise ValueError('Resume probe is exactly three full-gradient bs256/flow4 native balanced updates')
    if args.input_only and args.release is None:
        raise ValueError('Input-only profiling requires the actual protected V3 release')
    if args.compact_rgb and args.release is None:
        raise ValueError('Compact RGB is a real-stream transport mode')
    if args.output.exists() or args.warmup < 1 or not 1 <= args.measure <= 20:
        raise ValueError('Fresh output and an explicitly short timing window required')
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '0,1,2,3,4,5,6,7':
        raise ValueError('This timing is explicitly single-node eight-GPU')
    communication = {key: os.environ.get(key) for key in
                     ('NCCL_P2P_DISABLE', 'NCCL_IB_DISABLE', 'NCCL_SOCKET_IFNAME')}
    if communication != dict(NCCL_P2P_DISABLE='1', NCCL_IB_DISABLE='1', NCCL_SOCKET_IFNAME='bond0'):
        raise ValueError('Use the measured lc1 PCIe communication workaround from A800_CLUSTER.md')
    if '--xla_gpu_use_memcpy_local_p2p=false' not in os.environ.get('XLA_FLAGS','').split():
        raise ValueError('Explicitly disable XLA local memcpy collectives on this cluster')
    if subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'], text=True).strip():
        raise RuntimeError('All eight GPUs must be verified idle; never interrupt a teammate')
    args.output.mkdir(parents=True)
    os.environ['WANDB_MODE'] = 'disabled'
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
    os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'default'
    os.environ['OPENBLAS_NUM_THREADS'] = '4'
    os.environ['OMP_NUM_THREADS'] = '4'
    source_commit = subprocess.check_output(['git','rev-parse','HEAD'], cwd=args.reference, text=True).strip()
    delta = subprocess.check_output(['git','diff'], cwd=args.reference)
    if source_commit != 'ca556f74a455cef7987a2be4537b5ac85cc56dd7' or delta != args.patch.read_bytes():
        raise ValueError('Unregistered champion source/patch')
    openpi_source = subprocess.check_output(['git','rev-parse','HEAD'], cwd=args.openpi_reference, text=True).strip()
    openpi_delta = subprocess.check_output(['git','diff'], cwd=args.openpi_reference)
    if openpi_source != '01177e0242a1c7e8fad2547caa0e987def614cda' or openpi_delta != args.openpi_patch.read_bytes():
        raise ValueError('Unregistered openpi source/patch')
    verified = json.loads(args.verified.read_text())
    if verified['status'] != 'all_selected_assets_verified' or verified['revision'] != '114902410a6439ad1a0b58a112b84159d3ab3f65':
        raise ValueError('Use the verified shared50 parent, not a final specialist silently')
    if args.resume_probe and hashlib.sha256(args.verified.read_bytes()).hexdigest() != 'd0ca3f072c11967f3581827151c7242f53e53897650a80f2263f0803fd5b1e8a':
        raise ValueError('Use the registered original shared50 verification receipt')
    # Each immutable verified file must still have the same size. The separate
    # verifier hashes ALL files on this explicit root immediately before use.
    for row in verified['files']:
        path = args.checkpoint/row['path']
        if not path.is_file() or path.stat().st_size != row['size']:
            raise ValueError('Changed verified asset: '+str(path))
        if args.resume_probe:
            from behavior_champion.release import sha256
            if sha256(path) != row['sha256']:
                raise ValueError('Changed shared50 asset bytes: '+str(path))
    sys.path.insert(0, str(args.openpi_reference/'src'))
    sys.path.insert(0, str(args.reference/'src'))
    import numpy as np
    import jax
    import jax.numpy as jnp
    import flax.nnx as nnx
    from flax.traverse_util import flatten_dict, unflatten_dict
    import openpi.models.model as model_lib
    import openpi.shared.nnx_utils as nnx_utils
    import openpi.shared.image_tools as image_tools
    import openpi.training.sharding as sharding
    from b1k.training import config as config_lib
    from b1k.shared import normalize
    from b1k.transforms_normalize import NormalizeWithPerTimestamp
    from b1k.transforms import TokenizeFASTActions
    from b1k.models.observation import Observation
    from behavior_champion.contracts import state23_from_proprio61, delta_actions, pad32, training_stage, append_embeddings
    from behavior_champion.jax_transport import place_from_host, check_collective_and_gradient
    if len(jax.devices()) != 8 or any(device.platform != 'gpu' for device in jax.devices()):
        raise ValueError('Expected eight actual CUDA devices')
    mesh = sharding.make_mesh(8)
    transport_receipt = check_collective_and_gradient(mesh)
    (args.output/'transport.json').write_text(json.dumps(transport_receipt,indent=2)+'\n')
    print(json.dumps(transport_receipt),flush=True)
    jax.config.update('jax_compilation_cache_dir', str(args.output/'jax-cache'))
    layout = json.loads(args.layout.read_text())['layout']
    assets = args.checkpoint/'assets/IliaLarchenko/behavior_224_rgb'
    tokenizer_file = assets/'fast_tokenizer/processing_action_tokenizer.py'
    if hashlib.sha256(tokenizer_file.read_bytes()).hexdigest() != '6f021ca1f4c1b194ab6fa399d80baf3d642eadb17efb8f73301e4ac401522c20':
        raise ValueError('Unreviewed downloaded tokenizer code')
    stats = normalize.load(assets)
    normalizer = NormalizeWithPerTimestamp(stats, use_per_timestamp=True)
    tokenizer = TokenizeFASTActions(str(assets/'fast_tokenizer'), [(0,6),(7,23)], 200,
                                   norm_stats=stats, use_per_timestamp=True)
    native = np.load(args.probe/'native.npz', allow_pickle=False)
    probe_receipt = json.loads((args.probe/'result.json').read_text())
    with (args.probe/'native.npz').open('rb') as stream:
        h = hashlib.sha256()
        while block := stream.read(8*1024*1024): h.update(block)
    if h.hexdigest() != probe_receipt['native_npz_sha256'] or len(native['task_id']) != 100:
        raise ValueError('Use the exact real 100-TRAIN-task probe')
    def prepare_batch(source, indices, *, resize_cached):
        examples = []
        for i in indices:
            task = layout['dataset_to_model'][str(int(source['task_id'][i]))]
            state = state23_from_proprio61(source['state61'][i])
            row = normalizer(dict(state=state, actions=delta_actions(source['actions23'][i],state)))
            row = tokenizer(row)
            row['state'], row['actions'] = pad32(row['state']), pad32(row['actions'])
            row['image'] = {key: np.asarray(image_tools.resize_with_pad(source[key][i],224,224)) if resize_cached else source[key][i]
                            for key in ('base_0_rgb','left_wrist_0_rgb','right_wrist_0_rgb')}
            row['image_mask'] = {key: np.bool_(True) for key in row['image']}
            row['tokenized_prompt'] = np.array([task, training_stage(int(source['frame'][i]),int(source['length'][i]),layout['num_stages'][task])],dtype=np.int32)
            row['tokenized_prompt_mask'] = np.ones(2,dtype=bool)
            examples.append(row)
        batch = jax.tree.map(lambda *xs: np.stack(xs), *examples)
        if args.compact_rgb:
            return batch
        return Observation.from_dict({k:v for k,v in batch.items() if k!='actions'}), batch['actions']

    data_iterator = None
    stream_receipt = None
    balanced_plan = None
    if args.release is not None:
        from torch.utils.data import DataLoader, Subset
        from behavior_champion.dataset import NativeV3Dataset, collate_native, worker_init
        stream_dataset = NativeV3Dataset(args.release,'train')
        sample_count = (1 if cold_resume else args.warmup+args.measure)*args.batch_size
        if args.resume_probe:
            from behavior_champion.sampling import BalancedPlan
            balanced_plan = BalancedPlan(stream_dataset, updates=8000, batch_size=256, seed=17)
            selection = np.concatenate([balanced_plan.batch(i) for i in ([2] if cold_resume else range(3))])
            (args.output/'sampling-plan.json').write_text(json.dumps(balanced_plan.contract,indent=2)+'\n')
        else:
            selection = np.random.default_rng(17).permutation(len(stream_dataset))[:sample_count]
        if len(selection) != sample_count: raise ValueError('Insufficient real stream samples')
        loader = DataLoader(Subset(stream_dataset,selection),batch_size=32,num_workers=args.workers,
                            shuffle=False,collate_fn=collate_native,worker_init_fn=worker_init,
                            multiprocessing_context='spawn',prefetch_factor=1)
        data_iterator = iter(loader)
        stream_receipt = dict(release=str(args.release),workers=args.workers,
                              compact_rgb=args.compact_rgb,
                              train_episodes=len(stream_dataset.episodes),stride16_train_samples=len(stream_dataset),
                              sampled_indices_sha256=hashlib.sha256(selection.tobytes()).hexdigest(),
                              real_samples=sample_count,repeat_cached_batch=False,identities=[])
        batch = None
    else:
        batch = prepare_batch(native,[i%100 for i in range(args.batch_size)],resize_cached=True)
    data_sharding = jax.sharding.NamedSharding(mesh,jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    if args.compact_rgb:
        from behavior_champion.frontend import resize_uint8_rgb, resize_normalized_rgb
        rng = np.random.default_rng(17)
        for height,width in ((720,720),(480,480),(480,640)):
            pixels = rng.integers(0,256,(8,height,width,3),dtype=np.uint8)
            compact = jax.jit(resize_uint8_rgb,in_shardings=data_sharding,out_shardings=data_sharding)(
                place_from_host(pixels,data_sharding))
            previous = jax.jit(resize_normalized_rgb,in_shardings=data_sharding,out_shardings=data_sharding)(
                place_from_host(pixels.astype(np.float32)/255.*2.-1.,data_sharding))
            np.testing.assert_array_equal(np.asarray(compact),np.asarray(previous))
        print(json.dumps(dict(compact_frontend_exact_all8=True,image_shapes=3)),flush=True)

    def next_stream_batch():
        started = time.monotonic()
        chunks = [next(data_iterator) for _ in range(args.batch_size//32)]
        read_end = time.monotonic()
        native_source = {key:np.concatenate([chunk[key] for chunk in chunks])
                         for key in chunks[0] if key!='identity'}
        identities = [item for chunk in chunks for item in chunk['identity']]
        batch_tasks = len({item['task'] for item in identities})
        if batch_tasks<2 or any(item['split']!='train' or item['frame']%16!=item['phase'] for item in identities):
            raise ValueError('Mixed-task/stride/protected split contract failed')
        stream_receipt['identities'].extend(identities)
        native_source.update({key:np.array([item[field] for item in identities],dtype=np.int32)
                              for key,field in (('task_id','task'),('frame','frame'),('length','length'))})
        concatenate_end = time.monotonic()
        prepared = prepare_batch(native_source,range(args.batch_size),resize_cached=False)
        prepare_end = time.monotonic()
        placed = place_from_host(prepared,data_sharding)
        place_end = time.monotonic()
        return placed,batch_tasks,dict(read_wait=read_end-started,
            concatenate=concatenate_end-read_end,transforms=prepare_end-concatenate_end,
            placement_and_exact_roundtrip=place_end-prepare_end,
            host_prepared_bytes=sum(np.asarray(x).nbytes for x in jax.tree.leaves(prepared)))

    def finish_stream():
        try: next(data_iterator)
        except StopIteration: pass
        else: raise ValueError('Unexpected extra stream data')
        if len({item['index'] for item in stream_receipt['identities']})!=stream_receipt['real_samples']:
            raise ValueError('Duplicate real training observations')
        (args.output/'stream.json').write_text(json.dumps(stream_receipt,indent=2)+'\n')

    if args.input_only:
        import cProfile
        profiler = cProfile.Profile()
        rows = []
        profiler.enable()
        for index in range(args.warmup+args.measure):
            started = time.monotonic()
            placed,tasks,phases = next_stream_batch()
            jax.block_until_ready(placed)
            row = dict(step=index+1,warmup=index<args.warmup,seconds=time.monotonic()-started,
                       distinct_tasks=tasks,phases=phases)
            rows.append(row)
            with (args.output/'steps.jsonl').open('a') as stream: stream.write(json.dumps(row)+'\n')
            print(json.dumps(row),flush=True)
            del placed
        profiler.disable()
        profiler.dump_stats(str(args.output/'input-profile.pstats'))
        finish_stream()
        result = dict(status='input_only_profile_completed',rows=rows,optimizer_updates=0,
                      trained_checkpoint_saved=False,source_commit=source_commit,
                      integration_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                      compact_frontend_exact_all8=args.compact_rgb,
                      transport_validation=transport_receipt,real_stream={k:v for k,v in stream_receipt.items() if k!='identities'})
        (args.output/'result.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(result),flush=True)
        return
    import_spec = importlib.util.spec_from_file_location('champion_train_core',args.reference/'scripts/train.py')
    core = importlib.util.module_from_spec(import_spec); import_spec.loader.exec_module(core)
    loader_receipt = {}
    expected_parameter_hashes = {}

    class ExpandedLoader:
        def load(self, expected):
            loaded = flatten_dict(model_lib.restore_params(args.checkpoint/'params', restore_type=np.ndarray))
            shapes = flatten_dict(expected)
            if loaded.keys() != shapes.keys():
                raise ValueError('Checkpoint key coverage differs from actual constructed graph')
            grown = {}
            for key, old_rows in ((('task_embeddings','embedding'),50),(('task_stage_embeddings','embedding'),596)):
                old = loaded[key]
                shape = shapes[key].shape
                initializer = nnx.Embed(num_embeddings=shape[0],features=shape[1],rngs=nnx.Rngs(17+old_rows))
                fresh = np.asarray(initializer.embedding.value).astype(old.dtype)
                new = append_embeddings(old, fresh, old_rows)
                if new[:old_rows].tobytes() != old.tobytes():
                    raise ValueError('Changed pretrained embedding rows')
                grown['/'.join(key)] = dict(old=list(old.shape),new=list(new.shape),old_rows_byte_identical=True)
                loaded[key] = new
            for key, value in loaded.items():
                if value.shape != shapes[key].shape:
                    raise ValueError('Non-embedding shape mismatch: '+str(key))
                if not any(name in key for name in ('action_correlation_cholesky','L_spatial','L_temporal',
                           'cached_num_inpaint_actions','cached_input_action_dim','cached_Sigma_uo_Sigma_oo_inv',
                           'cached_L_cond_free','cached_Sigma_ou_Sigma_uu_inv','cached_L_cond_inp')):
                    expected_value = value.astype(jnp.bfloat16) if args.freeze_vision and key[:2]==('PaliGemma','img') else value
                    expected_parameter_hashes[key] = hashlib.sha256(expected_value.tobytes()).hexdigest()
            loader_receipt.update(dict(keys=len(loaded),expanded=grown,other_arrays_unchanged=True))
            return unflatten_dict(loaded)

    original = config_lib.get_config('pi_behavior_b1k_fast')
    config = dataclasses.replace(original,
        model=dataclasses.replace(original.model,num_tasks=100,task_num_stages=tuple(layout['num_stages'])),
        weight_loader=ExpandedLoader(),batch_size=args.batch_size,num_flow_samples=args.flow_samples,
        fsdp_devices=8,wandb_enabled=False,
        freeze_filter=nnx_utils.PathRegex('PaliGemma/img/.*') if args.freeze_vision else nnx.Nothing)
    if args.resume_probe:
        from openpi.training.optimizer import CosineDecaySchedule
        from behavior_champion.optimizer import RowScaledAdamW
        config = dataclasses.replace(config,optimizer=RowScaledAdamW(),
            lr_schedule=CosineDecaySchedule(warmup_steps=300,peak_lr=2.5e-5,decay_steps=8000,decay_lr=2.5e-6),
            ema_decay=.99)
    replicated = jax.sharding.NamedSharding(mesh,jax.sharding.PartitionSpec())
    if batch is not None: batch = place_from_host(batch,data_sharding)
    # Legacy uint32 PRNG keys preserve the same seed while avoiding implicit
    # GPU0->peer copies of jax.random.key(...). Every replica is read back.
    init_key = place_from_host(np.array([0,17],dtype=np.uint32),replicated)
    train_key = place_from_host(np.array([0,19],dtype=np.uint32),replicated)
    if args.resume_probe:
        from behavior_champion.release import RELEASE_SHA256
        from behavior_champion.sampling import digest_json
        resume_contract = dict(purpose='ENGINEERING_ONLY_NOT_FOR_DEPLOYMENT',
            integration_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
            parent_revision=verified['revision'],parent_receipt_sha256=hashlib.sha256(args.verified.read_bytes()).hexdigest(),
            reference_commit=source_commit,reference_patch_sha256=hashlib.sha256(delta).hexdigest(),
            openpi_commit=openpi_source,openpi_patch_sha256=hashlib.sha256(openpi_delta).hexdigest(),
            release_sha256=RELEASE_SHA256,layout_sha256=digest_json(layout),
            model=dataclasses.asdict(config.model),optimizer=dataclasses.asdict(config.optimizer),
            lr=dataclasses.asdict(config.lr_schedule),ema_decay=config.ema_decay,flow_samples=4,
            sampler_plan_sha256=balanced_plan.fingerprint,
            environment={key:os.environ.get(key) for key in ('XLA_FLAGS','NCCL_P2P_DISABLE','NCCL_IB_DISABLE',
                'NCCL_SOCKET_IFNAME','OPENBLAS_NUM_THREADS','OMP_NUM_THREADS')},
            jax_version=jax.__version__,numpy_version=np.__version__)
        # Normalize tuple/list representation before comparison with JSON.
        resume_contract = json.loads(json.dumps(resume_contract))
    if cold_resume:
        from behavior_champion.release import sha256
        reference_run = args.resume_probe_from
        reference_file = reference_run/'uninterrupted-third-update.json'
        if sha256(reference_file) != args.resume_reference_sha256:
            raise ValueError('Cold replay reference digest mismatch')
        # The uninterrupted branch is an independent reference even when the
        # producer's later restore fails. In particular, c22d1b72's tuple/list
        # metadata comparison was too strict before any restore. Do not relabel
        # that failure as a passed same-process test or regenerate its weights.
        reference_result = (json.loads((reference_run/'resume-result.json').read_text())
                            if (reference_run/'resume-result.json').exists() else None)
        if reference_result is not None and reference_result.get('cross_process_restore_tested'):
            raise ValueError('Use the original uninterrupted producer, not another replay')
        saved = json.loads((reference_run/'checkpoint-receipt.json').read_text())
        original_manifest = json.loads((Path(saved['path'])/'manifest.json').read_text())
        original_contract = original_manifest['contract']
        producing_commit = original_contract['integration_commit']
        # This new consumer may add diagnostic code, but cannot silently change
        # any computation/data/optimizer/restore dependency of the producer.
        subprocess.run(['git','diff','--exit-code',producing_commit,'HEAD','--',
            *[f'src/behavior_champion/{name}.py' for name in
              ('contracts','dataset','frontend','jax_transport','optimizer','release','sampling')]],
            stdout=subprocess.DEVNULL,check=True)
        comparable = dict(resume_contract,integration_commit=producing_commit)
        if comparable != original_contract:
            raise ValueError('Cold resume graph/config/data/numeric environment differs from producer')
        resume_contract = original_contract
    with sharding.set_mesh(mesh):
        state, state_sharding = core.init_train_state(config,init_key,mesh,resume=cold_resume,norm_stats=stats)
        if cold_resume:
            from behavior_champion.checkpoint import restore_full
            restored = restore_full(saved['path'],dict(state=state,train_rng=jax.ShapeDtypeStruct((2,),jnp.uint32)),
                dict(state=state_sharding,train_rng=replicated),contract=resume_contract,
                plan=balanced_plan,manifest_sha256=saved['manifest_sha256'],allow_legacy_engineering_weak_types=True)
            state,train_key = restored['state'],restored['train_rng']
            print(json.dumps(dict(cold_checkpoint_restored=True,completed_updates=int(state.step))),flush=True)
        jax.block_until_ready(state)
        # Init may reshard large tensors. Round-trip all loaded parameter bytes
        # before allowing a single update; synthetic collectives alone are insufficient.
        initialized = flatten_dict(state.params.to_pure_dict())
        if not expected_parameter_hashes.keys() <= initialized.keys():
            raise ValueError('Initialized graph lost expected parameter keys')
        for key,value in initialized.items():
            if key not in expected_parameter_hashes: continue
            actual = np.asarray(jax.device_get(value))
            if hashlib.sha256(actual.tobytes()).hexdigest() != expected_parameter_hashes[key]:
                raise ValueError('Initialized parameter bytes changed in transport: '+str(key))
        loader_receipt['initialized_parameter_roundtrip_hashes_passed'] = len(expected_parameter_hashes)
        print(json.dumps(dict(initialized_parameter_roundtrip_hashes_passed=len(expected_parameter_hashes))),flush=True)
        def core_step(key,state,batch):
            if args.compact_rgb:
                from behavior_champion.frontend import resize_uint8_rgb
                fields = {k:v for k,v in batch.items() if k!='actions'}
                fields['image'] = {k:resize_uint8_rgb(v) for k,v in fields['image'].items()}
                batch = Observation.from_dict(fields), batch['actions']
            elif args.release is not None:
                from behavior_champion.frontend import resize_normalized_rgb
                observation,actions = batch
                observation = observation.replace(images={k:resize_normalized_rgb(v) for k,v in observation.images.items()})
                batch = observation,actions
            return core.train_step(config,key,state,batch)
        step = jax.jit(core_step,
                       in_shardings=(replicated,state_sharding,data_sharding),
                       out_shardings=(state_sharding,replicated),donate_argnums=(1,))
        times = []
        resume_receipt = None
        for index in ([2] if cold_resume else range(args.warmup+args.measure)):
            started=time.monotonic()
            data_seconds = 0.
            batch_tasks = None
            phases = None
            if data_iterator is not None:
                # Preserve raw RGB through host placement. The author's existing
                # preprocess_observation resizes on EACH assigned GPU, inside
                # jit, so no unsafe GPU0->peer image copies are introduced.
                batch,batch_tasks,phases = next_stream_batch()
                data_seconds=time.monotonic()-started
            compute_started=time.monotonic()
            state, metrics=step(train_key,state,batch)
            jax.block_until_ready((state,metrics))
            seconds=time.monotonic()-started
            if not all(np.isfinite(np.asarray(x)).all() for x in jax.tree.leaves(metrics)):
                raise ValueError('Nonfinite training metrics')
            row=dict(step=index+1,seconds=seconds,warmup=index<args.warmup,
                     host_data_prepare_and_placement_seconds=data_seconds,
                     input_phases=phases,
                     accelerator_step_seconds=time.monotonic()-compute_started,
                     distinct_tasks=batch_tasks,
                     loss=float(metrics['loss']),grad_norm=float(metrics['grad_norm']))
            with (args.output/'steps.jsonl').open('a') as stream: stream.write(json.dumps(row)+'\n')
            print(json.dumps(row),flush=True)
            if index>=args.warmup:times.append(seconds)
            if args.resume_probe and index == 1:
                from behavior_champion.checkpoint import save_full
                saved = save_full(args.output/'engineering-checkpoints',dict(state=state,train_rng=train_key),
                    completed=2,contract=resume_contract,cursor=balanced_plan.cursor(2),progress=checkpoint_progress)
                (args.output/'checkpoint-receipt.json').write_text(json.dumps(saved,indent=2)+'\n')
                print(json.dumps(dict(full_state_saved=saved)),flush=True)
        if args.resume_probe and not cold_resume:
            import gc
            from behavior_champion.checkpoint import restore_full, tree_fingerprints
            baseline_metrics = jax.tree.map(np.asarray,metrics)
            baseline = tree_fingerprints(dict(state=state,train_rng=train_key),progress=checkpoint_progress)
            (args.output/'uninterrupted-third-update.json').write_text(json.dumps(baseline,indent=2)+'\n')
            template = jax.tree.map(lambda x:jax.ShapeDtypeStruct(x.shape,x.dtype,weak_type=x.weak_type),
                                    dict(state=state,train_rng=train_key))
            del state; gc.collect()
            restored = restore_full(saved['path'],template,dict(state=state_sharding,train_rng=replicated),
                contract=resume_contract,plan=balanced_plan,manifest_sha256=saved['manifest_sha256'])
            state,metrics = step(restored['train_rng'],restored['state'],batch)
            jax.block_until_ready((state,metrics))
            after = tree_fingerprints(dict(state=state,train_rng=restored['train_rng']),progress=checkpoint_progress)
            (args.output/'restored-third-update.json').write_text(json.dumps(after,indent=2)+'\n')
            exact = baseline == after
            metric_exact = all(np.array_equal(a,np.asarray(b)) for a,b in
                zip(jax.tree.leaves(baseline_metrics),jax.tree.leaves(metrics),strict=True))
            resume_receipt = dict(status='passed' if exact and metric_exact else 'failed_no_retry',
                uninterrupted_vs_restored_all_dynamic_leaves_exact=exact,metrics_exact=metric_exact,
                checkpoint=saved,optimizer_updates_from_parent=3,extra_replay_updates=1,
                checkpoint_role='engineering_validation_only_not_a_formal_or_deployable_model',
                cross_process_restore_tested=False,full_state_leaves=len(after['leaves']),
                sampler_plan_sha256=balanced_plan.fingerprint)
            (args.output/'resume-result.json').write_text(json.dumps(resume_receipt,indent=2)+'\n')
            print(json.dumps(resume_receipt),flush=True)
            if not exact or not metric_exact:
                raise ValueError('Full-state restored third update differs; preserve both fingerprints and do not deploy')
        elif cold_resume:
            from behavior_champion.checkpoint import tree_fingerprints
            baseline = json.loads(reference_file.read_text())
            after = tree_fingerprints(dict(state=state,train_rng=train_key),progress=checkpoint_progress)
            (args.output/'cold-restored-third-update.json').write_text(json.dumps(after,indent=2)+'\n')
            original_steps = [json.loads(line) for line in (reference_run/'steps.jsonl').read_text().splitlines()]
            last = original_steps[-1]
            metric_exact = last['step'] == 3 and all(float(metrics[k]) == last[k] for k in ('loss','grad_norm'))
            exact = baseline == after
            resume_receipt = dict(status='passed' if exact and metric_exact else 'failed_no_retry',
                cross_process_restore_tested=True,all_dynamic_leaves_exact=exact,
                loss_and_grad_norm_exact=metric_exact,reference_sha256=args.resume_reference_sha256,
                checkpoint=saved,optimizer_updates_this_process=1,physical_controls=0,
                full_state_leaves=len(after['leaves']),producing_commit=producing_commit,
                consuming_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                producer_same_process_restore_status=None if reference_result is None else reference_result['status'],
                checkpoint_reader_sha256=hashlib.sha256((Path(__file__).resolve().parents[2]/'src/behavior_champion/checkpoint.py').read_bytes()).hexdigest(),
                new_checkpoint_saved=False)
            (args.output/'resume-result.json').write_text(json.dumps(resume_receipt,indent=2)+'\n')
            print(json.dumps(resume_receipt),flush=True)
            if not exact or not metric_exact:
                raise ValueError('Cold-process continuation differs; preserve evidence and investigate')
    if data_iterator is not None:
        finish_stream()
    result=dict(status='engineering_resume_probe_completed' if args.resume_probe else
                ('short_disposable_stream_timing_completed' if args.release else 'short_disposable_compute_timing_completed'),batch_size=args.batch_size,
                flow_samples=args.flow_samples,true_vision_freeze=args.freeze_vision,
                seconds_per_step_median=float(np.median(times)),examples_per_second=args.batch_size/float(np.median(times)),
                warmup_steps=args.warmup,timed_steps=args.measure,optimizer_updates=int(state.step),
                seconds_per_step_aggregate=sum(times)/len(times),
                examples_per_second_aggregate=args.batch_size*len(times)/sum(times),
                trained_checkpoint_saved=False,engineering_full_state_checkpoint_saved=args.resume_probe and not cold_resume,
                resume_validation=resume_receipt,source_commit=source_commit,patch_sha256=hashlib.sha256(delta).hexdigest(),
                openpi_source_commit=openpi_source,openpi_patch_sha256=hashlib.sha256(openpi_delta).hexdigest(),
                communication_environment=communication,
                compact_frontend_exact_all8=args.compact_rgb,
                transport_validation=transport_receipt,
                parent_revision=verified['revision'],probe_sha256=h.hexdigest(),checkpoint_migration=loader_receipt,
                real_stream=None if stream_receipt is None else {k:v for k,v in stream_receipt.items() if k!='identities'},
                warning='Real short stream includes loader, transforms, checked placement and compute but excludes checkpoint/eval and long-run variance.' if args.release else
                        'Cached real batch; no epoch ETA until shuffled full-loader/checkpoint/eval overhead is measured.')
    (args.output/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__ == '__main__':
    main()
