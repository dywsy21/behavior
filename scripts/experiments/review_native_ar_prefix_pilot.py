"""Verify all recorded AR controls/history/physics references and emit review aids.

This is read-only analysis of one completed run. New output is review material,
not a training release, neural inference, simulator run, or success label.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

REMOTE = Path('/mnt/sdc1/robodojo/behavior_dev/dual_track_fm_ar_20260913/ar_native_prefix_pilot_v1')
CAMERAS = ('head_rgb', 'left_wrist_rgb', 'right_wrist_rgb')
STATE_WIDTHS = dict(base_qvel=3, trunk_qpos=4, left_arm=7, left_gripper=2, right_arm=7, right_gripper=2)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def local_ref(root, path, expected):
    local = root / Path(path).relative_to(REMOTE)
    if '..' in local.relative_to(root).parts or sha(local) != expected:
        raise ValueError('Copied reference identity changed: ' + str(path))
    return local


def frame(root, count):
    path = root / 'actual_rollout/observable' / f'f{count:08d}.npz'
    with np.load(path, allow_pickle=False) as values:
        if values['source_frame'].tolist() != [count]:
            raise ValueError('Observation source clock differs')
        return {key: values[key].copy() for key in
            (*CAMERAS, *('state_' + name for name in STATE_WIDTHS))}


def anchor_hash(raw):
    h = hashlib.sha256()
    for group, keys in (('images', CAMERAS), ('state', tuple(STATE_WIDTHS))):
        for name in sorted(keys):
            value = raw[name if group == 'images' else 'state_' + name]
            h.update(f'{group}/{name}/{value.dtype}/{value.shape}'.encode())
            h.update(value.tobytes())
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve(strict=True)
    output = root / 'review'
    if output.exists():
        raise FileExistsError('Do not overwrite a completed review')
    result = read(root / 'actual_rollout/collection_result.json')
    if (sha(root / 'actual_rollout/collection_result.json') != '5a200786ea42317c97327431eed3cbf794355be00eac53b7bf4fa38fb426757d'
            or result['status'] != 'complete' or result['prefix_actions_executed'] != 448
            or result['native_policy_actions_consumed'] != 256 or len(result['actual_history_requests']) != 16
            or sha(root / 'actual_rollout/rollout.mp4') != 'e845f688b7d7a45a1805261ff0af0d707f38cb59334e4b39752efd47c768e5d0'):
        raise ValueError('Only the exact completed pilot/video may be reviewed here')
    events = [json.loads(line) for line in (root / 'actual_rollout/events.jsonl').read_text().splitlines()]
    if len(events) != 18 or [row['phase'] for row in events] != ['initial', 'before_policy'] + ['post_chunk'] * 16:
        raise ValueError('Missing/reordered physical review events')
    all_actions, physical, chunks = [], [], []
    for index, event in enumerate(events):
        action_path = local_ref(root, event['action_ref']['path'], event['action_ref']['sha256'])
        actions = np.load(action_path, allow_pickle=False)
        private = read(local_ref(root, event['physical_evidence_ref']['path'], event['physical_evidence_ref']['sha256']))
        receipt = read(local_ref(root, private['physical_receipt_path'], private['physical_receipt_sha256']))
        if not np.array_equal(np.asarray(receipt['actual_actions_23d'], dtype=np.float32).reshape(-1, 23), actions):
            raise ValueError('Physical recorder controls differ from the executor action reference')
        held = receipt['outcome']['evidence'][0]['fields']['held']
        physical.append(dict(frame=receipt['observable_inputs']['anchor_frame'], held=held,
            outcome=receipt['outcome']['outcome'], causal_completed=receipt['outcome']['causal_completed']))
        if index < 2:
            continue
        number, count = index - 2, 448 + 16 * (index - 2)
        chunk = read(root / 'service' / f'chunk_{number:02d}.json')
        generated = np.asarray(chunk['raw23_all32'], dtype=np.float32)
        if (generated.shape != (32, 23) or actions.shape != (16, 23) or actions.dtype != np.float32
                or not np.array_equal(generated[:16], actions) or not np.isfinite(actions).all()
                or not np.all(actions[:, 6] == 0) or chunk['history']['consumed_actions'] != count
                or chunk['schema']['action_tokens'] != 60 or chunk['schema']['rule_safe_clamp']
                or event['physical_outcome_label']['outcome_evidence_end_frame'] != count + 16):
            raise ValueError('Actual complete AR 0:16/23D and physical clock no longer agree')
        expected = {str(n): anchor_hash(frame(root, n)) for n in range(count - 80, count + 1, 16)}
        if (chunk['history']['anchor_hashes'] != expected
                or result['actual_history_requests'][number]['service_admission']['anchor_hashes'] != expected
                or result['actual_history_requests'][number]['evaluator_subgoal_sent']):
            raise ValueError('Actual model history differs from captured physics observations')
        current = frame(root, count)
        deltas = {}
        for name, section in [('left_arm', slice(7, 14)), ('right_arm', slice(15, 22))]:
            difference = actions[:, section] - current['state_' + name][None]
            deltas[name] = dict(rms=float(np.sqrt(np.mean(difference ** 2))), maximum=float(np.abs(difference).max()))
        chunks.append(dict(index=number, frame=count, mean_yaw_command=float(actions[:, 2].mean()),
            arm_target_minus_current_state=deltas, forced_tokens=chunk['schema']['raw_argmax_overruled'],
            infer_ms=chunk['timing']['infer_ms']))
        all_actions.append(actions)
    actions = np.concatenate(all_actions)
    observed = {name: np.stack([frame(root, count)['state_' + name] for count in range(448, 705, 16)])
                for name in STATE_WIDTHS}
    summary = dict(verified=True, actual_controls_verified=256, actual_chunks_verified=16,
        physical_events_verified=18, actual_history_anchors_verified=96, source_result_sha256=sha(root / 'actual_rollout/collection_result.json'),
        physical_boundary_observations=physical, chunks=chunks,
        command_yaw=dict(minimum=float(actions[:, 2].min()), maximum=float(actions[:, 2].max()),
            mean_abs=float(np.abs(actions[:, 2]).mean()), integral_radians_not_measured_pose=float(actions[:, 2].sum() / 30)),
        gripper_commands={side: np.unique(actions[:, index]).tolist() for side, index in [('left', 14), ('right', 22)]},
        observed_state_ranges={name: dict(minimum=values.min(0).tolist(), maximum=values.max(0).tolist())
                               for name, values in observed.items()},
        training_admissible=False, new_annotation_release=False, optimizer_updates=0, model_calls=0, simulator_actions=0,
        manual_visual_review_pending=True,
        limitations=['Physical receipts are 18 sampled boundaries, not a saved every-frame held trace.',
                     'Integrated yaw command is not measured robot heading.',
                     'Inherited C1 eligible flags and active_bundle are evaluator metadata, not released labels or actor intent.',
                     'Short prefix-assisted diagnostic, not full-task SR or an equal-budget A4 causal comparison.'])
    output.mkdir()
    (output / 'machine_checks.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    selected = [448, 480, 528, 576, 640, 704]
    sheet = Image.new('RGB', (720, 384 * len(selected)), 'white')
    draw = ImageDraw.Draw(sheet)
    for row, count in enumerate(selected):
        raw = frame(root, count)
        for col, key in enumerate(('left_wrist_rgb', 'right_wrist_rgb')):
            img = Image.fromarray(np.moveaxis(raw[key], 0, -1))
            img.thumbnail((360, 360))
            sheet.paste(img, (360 * col, row * 384 + 24))
            draw.text((360 * col + 8, row * 384 + 4), f'frame {count} / {key}', fill='black')
    sheet.save(output / 'wrists-contact-sheet.png')
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == '__main__':
    main()
