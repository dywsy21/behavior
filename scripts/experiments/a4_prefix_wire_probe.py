"""Seven real-observation calls through the corrected isolated low service."""
import asyncio
import json
from pathlib import Path

from a4_prefix_pilot import WORK, OUTPUT, PORT, RUNTIME, CHECKPOINT_SHA, sha, write_new, validate


async def main():
    manifest, _, context = validate()
    import numpy as np
    import websockets
    from native_a2_history import LowHistoryIngress, WIDTHS, history_admission_to_wire
    from native_b_session import CAMERAS
    from g05.utils.websocket import packb, unpackb
    destination = OUTPUT/'wire_probe'
    if destination.exists():
        raise FileExistsError('Never overwrite or append an actual wire probe')
    destination.mkdir()
    original = WORK/'demo_radio_e121_alignment_v2_persist'
    source_manifest = json.loads((original/'manifest.json').read_text())
    context_ref = source_manifest['context']
    if sha(context_ref['path']) != context_ref['sha256']:
        raise ValueError('Original actual observation instruction changed')
    source_context = json.loads(Path(context_ref['path']).read_text())
    goal = {k:source_context['evaluated_bundle'][k] for k in ('parent_goal', 'active_skills_semantic_json')}
    rows = []
    ingress = LowHistoryIngress(initial_action_count=448)
    async with websockets.connect(f'ws://127.0.0.1:{PORT}', max_size=64<<20, ping_timeout=None) as ws:
        hello = unpackb(await ws.recv())
        identity = hello['identity']
        if (hello['mode'] != 'native_skill_fm_history6' or identity['checkpoint_sha256'] != CHECKPOINT_SHA
                or identity['inference_alignment_fix']['receipt_sha256'] != manifest['inference_alignment_receipt_sha256']):
            raise ValueError('Wrong actual model/patch endpoint')
        async def call(request):
            await ws.send(packb(request))
            reply = unpackb(await asyncio.wait_for(ws.recv(), timeout=600))
            if reply.get('ok') is not True:
                raise ValueError('Actual service rejected the request: '+str(reply))
            return reply['response']
        await call(dict(kind='begin', seed=17))
        for anchor in range(448, 560, 16):
            frames = list(range(anchor-80, anchor+1, 16))
            captured, refs = [], []
            for frame in frames:
                path = original/'actual_replay/observations'/f'f{frame:08d}.npz'
                with np.load(path, allow_pickle=False) as values:
                    if values['source_frame'].tolist() != [frame]:
                        raise ValueError('Wrong actual observation time')
                    captured.append({k:values[k].copy() for k in values.files if k != 'source_frame'})
                refs.append(dict(frame=frame, path=str(path), sha256=sha(path)))
            observation = dict(task=source_context['task_name'], frequency=30., embodiment_type='galaxea_r1pro',
                history_action_counts=frames, history_is_pad=[False]*6,
                images={k:np.stack([v[k] for v in captured]) for k in CAMERAS},
                state={k:np.stack([v['state_'+k] for v in captured]) for k in WIDTHS})
            _, admission = ingress.prepare(observation, execute_steps=16)
            response = await call(dict(kind='native_low_chunk', observation=observation,
                installed_subgoal=goal, execute_steps=16))
            actions = np.asarray(response['actions'])
            if (actions.shape != (16, 23) or actions.dtype != np.float32 or not np.isfinite(actions).all()
                    or response['history_admission'] != history_admission_to_wire(admission)
                    or response['low_component_identity']['checkpoint_sha256'] != CHECKPOINT_SHA):
                raise ValueError('Actual action/history/checkpoint contract failed')
            aligned = response['inference_alignment_admission']
            if (aligned['padded_dimensions'] != [7, 8, 17, 18] or aligned['real_control_dimensions'] != 23
                    or aligned['image_history_frames'] != 6 or aligned['action_history_steps'] != 0
                    or aligned['action_execution_start_index'] != 0 or aligned['expert_actions_used']):
                raise ValueError('Actual native preparation did not apply both corrections')
            ingress.commit(admission)
            path = destination/f'f{anchor:08d}.npy'
            with path.open('xb') as stream: np.save(stream, actions, allow_pickle=False)
            rows.append(dict(frame=anchor, observations=refs, prediction_sha256=sha(path),
                history_admission=response['history_admission'], inference_alignment_admission=aligned,
                timing=response['timing']))
    result = dict(passed=True, model_calls=len(rows), rows=rows, all_alignment_admissions_correct=True,
        source_sha256=sha(__file__), manifest_sha256=sha(OUTPUT/'manifest.json'), checkpoint_sha256=CHECKPOINT_SHA,
        optimizer_steps=0, physics_actions=0, training_admissible=False, success_rate_claim=False,
        saved_observations_are_original_action_replay_not_new_learner_rollout=True)
    write_new(destination/'result.json', result)
    print(json.dumps(dict(passed=True, model_calls=len(rows))), flush=True)


if __name__ == '__main__':
    asyncio.run(main())
