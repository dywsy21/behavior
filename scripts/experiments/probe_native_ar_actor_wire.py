"""One real observed-only native500 generation plus msgpack, not physics/SR."""
import argparse
import json
import os
from pathlib import Path
import subprocess

from action_training_runtime import REPO, sha
from probe_ar_execution_codec import publish
from train_fm_method_probe import BASE
from train_action_method_probe import now


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpu', type=int, choices=range(4), required=True)
    args = parser.parse_args()
    if args.output.parent != BASE or subprocess.check_output(
            ['git', '-C', str(REPO), 'status', '--porcelain'], text=True).strip():
        raise ValueError('A clean immutable source and new declared experiment child are required')
    args.output.mkdir(exist_ok=False)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    from native_ar_actor_runtime import runtime_dependencies, load_native_actor, saved_radio_prefix, raw_actions_to_wire
    model, adapter, identity = load_native_actor()
    dependencies = runtime_dependencies()
    observation, source = saved_radio_prefix()
    import numpy as np
    import torch
    from g05.utils.websocket import packb, unpackb
    from native_action_observations import infer_native_task_observation

    identity.update(start_time=now(), commit=subprocess.check_output(
        ['git', '-C', str(REPO), 'rev-parse', 'HEAD'], text=True).strip(),
        entry_sha256=sha(Path(__file__)), dependencies=dependencies.identity, source=source,
        gpu=args.gpu, policy_seed=17, static_format_forced=True, max_generations=1,
        optimizer_updates=0, simulator_controls=0, network_socket_used=False)
    publish(args.output / 'manifest.json', identity)
    ingress = dependencies.history.LowHistoryIngress(initial_action_count=448)
    observed, admission = ingress.prepare(observation, execute_steps=16)
    torch.manual_seed(17)
    torch.cuda.manual_seed_all(17)
    result = infer_native_task_observation(model, adapter, observed, device='cuda:0', constrain_format=True)
    raw = result['grouped_raw_action']
    # Save the actual neural output before any wire-only assertion can fail.
    publish(args.output / 'generated.json', dict(action=result['generated']['action'].detach().cpu().tolist(),
        raw_groups={key: value.detach().cpu().tolist() for key, value in raw.items() if not key.startswith('_')},
        schema=result['schema'], trace=result['schema_trace'], timing=result['timing']))
    wire32 = raw_actions_to_wire(raw, dependencies.bridge)
    if not np.array_equal(wire32[:, 6], np.zeros(32, dtype=np.float32)):
        raise RuntimeError('Generated raw trunk channel3 violates the official-reset constant support; never clamp it')
    packet = dict(actions=wire32[:16].copy(), actor_route=result['actor_route'],
        history_admission=dependencies.history.history_admission_to_wire(admission),
        execution_start=0, execution_steps=16, teacher_or_planner_in_actor=False)
    received = unpackb(packb(packet))
    if (not np.array_equal(received['actions'], packet['actions']) or received['actions'].dtype != np.float32
            or received['history_admission'] != packet['history_admission']):
        raise RuntimeError('Actual msgpack roundtrip changed official23 or its history identity')
    ingress.commit(admission)
    publish(args.output / 'result.json', dict(complete=True, actual_generations=1,
        raw23_all32=wire32.tolist(), executed0_16_shape=list(received['actions'].shape),
        msgpack_roundtrip_exact=True, history_committed_count=ingress.last_count,
        static_format_forced=True, schema=result['schema'], actor_route=result['actor_route'],
        official_reset_trunk3_constant_preserved=True, checkpoint_sha256=identity['checkpoint_sha256'],
        optimizer_updates=0, simulator_controls=0, network_socket_used=False, success_rate_claim=False,
        peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        limitations=['Stored original-prefix observations, not newly generated learner history.',
                     'Static syntax is imposed; this is not learned completeness or action-quality/SR proof.',
                     'Only serialization roundtrip; real websocket/controller/physics still need evaluation.']))
    print(json.dumps(dict(complete=True, result_sha256=sha(args.output / 'result.json'))), flush=True)


if __name__ == '__main__':
    main()
