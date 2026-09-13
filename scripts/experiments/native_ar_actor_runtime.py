"""Pinned native-task AR500 actor and the existing observed-history/raw bridge.

No planner, dataset iteration, future action, simulator truth or fake FM
snapshot is required. The optional syntax constraint stays an explicit variant.
"""
from __future__ import annotations

import gc
import importlib
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

from action_training_runtime import bootstrap, INPUT, REPO, sha
from action_training_data import STATS, STATS_SHA
from native_action_initialization import NATIVE_PARENT, NATIVE_PARENT_SHA, verify_native_vocabulary
from probe_ar_decode_consistency import verify_probe_checkpoint
from probe_action_training_gpu import architecture_for, prepare_action_updates
from train_action_method_probe import read
from train_fm_method_probe import BASE

WORK = Path('/mnt/sdc1/robodojo/behavior_dev/direct_execution_L6rqZ6_20260910')
HISTORY_ROOT = WORK / 'native_a3_aligned_prefix_runtime_v2'
BRIDGE = Path('/mnt/sdc1/robodojo/behavior_dev/GalaxeaVLA_memlite_coordination_dev_20260908/sim_runtime/production_native_oracle_low_v1/native_oracle_low_v1/runtime.py')
DEPENDENCIES = {
    HISTORY_ROOT / 'native_a2_history.py': 'a75b21e927452b9a866eec57d89dce4c39cf898a68b2429100166d5e50df5a95',
    HISTORY_ROOT / 'native_b_session.py': 'b811e1ac2c0fdafbc8e15b7dc4d6ffd06fb514545794540267d6ea556525fecb',
    BRIDGE: '61beb65dbe0dc5a6e7bb952a60a9d790c9b2eb51bd546e2ecc53a61fb662c447',
}
NATIVE_RUN = BASE / 'ar_native_task_fulltrain_v1'
CHECKPOINT = NATIVE_RUN / 'formal/checkpoints/step_500.pt'
CHECKPOINT_SHA = '639e64aeeb251113b807751f234077595165e65e9dd9e3b66cd7c4661f9df963'
CONFIG_SHA = 'bc6674b6d89d0eaca38ca9e215c1ff75c8e5d20be499c2e209d01a70d7c36c2f'


def runtime_dependencies():
    for path, expected in DEPENDENCIES.items():
        if sha(path) != expected:
            raise RuntimeError('Reviewed history/official bridge dependency changed: ' + str(path))
    sys.path.insert(0, str(HISTORY_ROOT))
    for name in ('native_b_session', 'native_a2_history'):
        module = importlib.import_module(name)
        if Path(module.__file__).resolve() != HISTORY_ROOT / (name + '.py'):
            raise RuntimeError('A different observation history module is already imported')
    history = sys.modules['native_a2_history']
    spec = importlib.util.spec_from_file_location('native_ar_original_raw_bridge', BRIDGE)
    bridge = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = bridge
    spec.loader.exec_module(bridge)
    return SimpleNamespace(history=history, bridge=bridge, identity={str(p): v for p, v in DEPENDENCIES.items()})


def raw_actions_to_wire(raw, bridge):
    """Validate every predicted raw control before selecting 0:16 execution."""
    import numpy as np
    import torch
    from native_action_observations import ACTION_WIDTHS
    if (not isinstance(raw, dict) or {key for key in raw if not key.startswith('_')} != set(ACTION_WIDTHS)
            or raw.get('_absent_keys')):
        raise ValueError('Need all six inverse-processed raw control groups')
    arrays = {}
    for name, width in ACTION_WIDTHS.items():
        value = raw[name]
        if (not isinstance(value, torch.Tensor) or value.shape != (1, 32, width)
                or not torch.isfinite(value).all()):
            raise ValueError('Invalid raw control shape or nonfinite value: ' + name)
        arrays[name] = value.detach().cpu().float().numpy()[0]
    result = np.stack([bridge.grouped_raw_to_official23({name: value[step] for name, value in arrays.items()})
                       for step in range(32)])
    # Independent explicit assembly only checks the original bridge, never
    # supplies a fallback output or slices a normalized model tensor.
    expected = np.concatenate([arrays[name] for name in
        ('base_qvel', 'trunk_qpos', 'left_arm', 'left_gripper', 'right_arm', 'right_gripper')], axis=1)
    if (result.shape != (32, 23) or result.dtype != np.float32 or not np.isfinite(result).all()
            or not np.array_equal(result, expected)):
        raise ValueError('Original official23 bridge altered controls or their order')
    return result


def load_native_actor():
    """Exact completed native500; no original dataset is instantiated."""
    identity = bootstrap()
    import torch
    from omegaconf import OmegaConf
    from g05.utils.config.config_resolvers import register_default_resolvers
    from g05.utils.checkpoint.checkpoint_utils import load_model_from_checkpoint
    from g05.utils.training.ar_training_methods import ActionTrainingSettings
    from g05.utils.data.processor_utils import build_processors
    from g05.utils.data.normalizer import load_dataset_stats_from_json
    from native_action_observations import NativeTaskObservationProcessor

    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.manual_seed(41)
    torch.cuda.set_device(0)
    if torch.cuda.mem_get_info()[0] < 40 * 1024**3:
        raise RuntimeError('Require 40 GiB GPU headroom; never evict other jobs')
    torch.cuda.set_per_process_memory_fraction(.4)
    register_default_resolvers()
    config = INPUT / 'diagnostic_processor_config.yaml'
    inspection, method = read(NATIVE_RUN / 'formal/checkpoint_inspection.json'), read(NATIVE_RUN / 'method_spec.json')
    if (read(NATIVE_RUN / 'status.json').get('state') != 'complete' or not inspection.get('passed')
            or inspection['actual_updates'] != 500 or inspection['checkpoint'] != str(CHECKPOINT)
            or inspection['checkpoint_sha256'] != CHECKPOINT_SHA or sha(CHECKPOINT) != CHECKPOINT_SHA
            or sha(config) != CONFIG_SHA or sha(STATS) != STATS_SHA
            or (method['route'], method['initialization'], method['conditioning']) != ('ar', 'native', 'native_task')
            or method['parent_sha256'] != NATIVE_PARENT_SHA):
        raise RuntimeError('Completed native500/config/normalizer identity changed')
    cfg = OmegaConf.load(config)
    settings = ActionTrainingSettings(route='ar', conditioning='native_task')
    # Vocabulary follows the native parent even though trained weights now
    # come from CHECKPOINT; passing CHECKPOINT here would register HL_END.
    arch = architecture_for(cfg, settings, parent=NATIVE_PARENT)
    processors = build_processors(cfg)
    processors.set_normalizer_from_stats(load_dataset_stats_from_json(STATS))
    processors.eval()
    adapter = NativeTaskObservationProcessor(processors.processors['galaxea_r1pro'])
    print('Loading exact completed native task AR500', flush=True)
    model, payload = load_model_from_checkpoint(arch, str(CHECKPOINT), device='cuda:0',
                                               eval_mode=False, return_full_checkpoint=True)
    if payload['action_experiment_spec'] != method:
        raise RuntimeError('Saved native training recipe differs from its verified source receipt')
    restored = verify_probe_checkpoint(model, payload, 'ar500')
    vocabulary = verify_native_vocabulary(model.processor)
    del payload
    gc.collect()
    prepare_action_updates(model, 'cuda:0')
    model.eval()
    identity.update(checkpoint=str(CHECKPOINT), checkpoint_sha256=CHECKPOINT_SHA,
        restoration=restored, vocabulary=vocabulary, config_sha256=CONFIG_SHA,
        normalizer_path=str(STATS), normalizer_sha256=STATS_SHA, settings=settings.as_dict(),
        adapter_sha256=sha(REPO / 'scripts/experiments/native_action_observations.py'),
        runtime_sha256=sha(Path(__file__)), dataset_instantiated=False, planner_loaded=False)
    return model, adapter, identity


def saved_radio_prefix():
    """One existing real 448-control replay state, no future actions loaded."""
    import numpy as np
    from native_action_observations import CAMERAS, STATE_WIDTHS
    original = WORK / 'demo_radio_e121_alignment_v2_persist'
    manifest_path = original / 'manifest.json'
    if sha(manifest_path) != 'c0138e093765e634374d762cd9b5de5fe17628baa14419121b9d8aed0a946c8d':
        raise RuntimeError('Original real observation source manifest changed')
    manifest = read(manifest_path)
    context_ref = manifest['context']
    if sha(context_ref['path']) != context_ref['sha256']:
        raise RuntimeError('Original task instruction context changed')
    task = read(context_ref['path'])['task_name']  # Never evaluated_bundle/physical labels.
    frames, captures, refs = list(range(368, 449, 16)), [], []
    for frame in frames:
        path = original / 'actual_replay/observations' / f'f{frame:08d}.npz'
        with np.load(path, allow_pickle=False) as values:
            if values['source_frame'].tolist() != [frame]:
                raise RuntimeError('Stored actual observation clock changed')
            captures.append({key: values[key].copy() for key in
                (*CAMERAS, *('state_' + name for name in STATE_WIDTHS))})
        refs.append(dict(frame=frame, path=str(path), sha256=sha(path)))
    observation = dict(task=task, frequency=30., embodiment_type='galaxea_r1pro',
        history_action_counts=frames, history_is_pad=[False] * 6,
        images={key: np.stack([row[key] for row in captures]) for key in CAMERAS},
        state={key: np.stack([row['state_' + key] for row in captures]) for key in STATE_WIDTHS})
    return observation, dict(source_manifest=str(manifest_path), source_manifest_sha256=sha(manifest_path),
        observation_files=refs, prefix_actions=448, task_id=0, episode=121, instance=138,
        split='train', stored_observations_from_original_action_replay=True, future_actions_loaded=False)
