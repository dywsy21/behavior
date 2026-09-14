"""Read-only, all-event review of one completed FM512 arm and local media copy.

Verify copied hashes, physical controls, causal outcomes and every real history
anchor. Emit contact sheets for manual inspection; never manufacture success
labels or treat a machine pass as the manual review itself.
"""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from run_fm_screen_prefix import SCREEN, OUTPUT as REMOTE, sha, read, write
from review_native_ar_prefix_pilot import frame, anchor_hash, CAMERAS, STATE_WIDTHS


def local_ref(root, remote_root, path, digest):
    relative = Path(path).relative_to(remote_root)
    if '..' in relative.parts:
        raise ValueError('Invalid copied evidence path')
    actual = root / relative
    if sha(actual) != digest:
        raise ValueError('Copied event evidence SHA differs: ' + str(relative))
    return actual


def make_sheet(root, destination, counts, cameras, columns):
    cells = [(count, camera) for count in counts for camera in cameras]
    sheet = Image.new('RGB', (columns * 288, ((len(cells) + columns - 1) // columns) * 316), 'white')
    draw = ImageDraw.Draw(sheet)
    for index, (count, camera) in enumerate(cells):
        x, y = (index % columns) * 288, (index // columns) * 316
        values = frame(root, count)
        image = Image.fromarray(np.moveaxis(values[camera], 0, -1))
        image.thumbnail((288, 288))
        sheet.paste(image, (x, y + 24))
        draw.text((x + 4, y + 4), f'f{count} / {camera}', fill='black')
    sheet.save(destination)


def review(root, name):
    if name not in SCREEN or root.name != name:
        raise ValueError('Review one of the three declared matching arm directories')
    remote_root = REMOTE / name
    manifest, completion = read(root.parent / 'manifest.json'), read(root / 'completion.json')
    result_path = root / 'actual_rollout/collection_result.json'
    result = read(result_path)
    controls = result['native_policy_actions_consumed']
    if (manifest['screen'] != SCREEN or manifest['recipe']['max_model_controls'] != 512
            or not completion['complete'] or completion['result_sha256'] != sha(result_path)
            or result['status'] != 'complete' or result['prefix_actions_executed'] != 448
            or type(controls) is not int or not 0 < controls <= 512 or controls % 16
            or result['training_admissible'] or result['full_task_success_rate_claim']
            or result['correction_teacher'] or result['high_policy_calls'] != 0
            or result['low_service_identity']['checkpoint_sha256'] != SCREEN[name]):
        raise ValueError('Not the exact completed, bounded FM500 physical arm')
    endpoint = read(root / 'socket_gate.json')
    if not endpoint['passed'] or endpoint['model_calls'] != 0 or endpoint['identity'] != result['low_service_identity']:
        raise ValueError('Actual simulation endpoint differs from the verified private service')
    events_path = root / 'actual_rollout/events.jsonl'
    events = [json.loads(line) for line in events_path.read_text().splitlines()]
    requests = result['actual_history_requests']
    if (len(events) != controls // 16 + 2 or len(requests) != controls // 16
            or [event['phase'] for event in events] != ['initial', 'before_policy'] + ['post_chunk'] * len(requests)):
        raise ValueError('Missing, extra or reordered actual physics events')
    context = read(root / 'event_context.json')
    goal = {key: context['evaluated_bundle'][key] for key in ('parent_goal', 'active_skills_semantic_json')}
    physical, details, consumed = [], [], []
    for index, event in enumerate(events):
        count = 0 if index == 0 else 448 + 16 * (index - 1)
        if (event['event_index'] != index or event['observable_refs_audit']['anchor_frame'] != count
                or event['source']['split'] != 'train' or event['source']['instance_id'] != 138
                or event['observable_prefix_descriptor']['active_bundle'] != goal
                or event['admission']['correction_teacher_eligible'] or event['admission']['reinject']):
            raise ValueError('Physical event clock/source/skill/admission changed')
        ref = event['action_ref']
        actions = np.load(local_ref(root, remote_root, ref['path'], ref['sha256']), allow_pickle=False)
        private = event['physical_evidence_ref']
        envelope = read(local_ref(root, remote_root, private['path'], private['sha256']))
        receipt = read(local_ref(root, remote_root, envelope['physical_receipt_path'], envelope['physical_receipt_sha256']))
        original = np.asarray(receipt['actual_actions_23d'], dtype=np.float32).reshape(-1, 23)
        expected_count = 0 if index < 2 else 16
        label = event['physical_outcome_label']
        if (actions.shape != (expected_count, 23) or actions.dtype != np.float32
                or not np.isfinite(actions).all() or not np.array_equal(actions, original)
                or ref['actual_consumed_actions'] != expected_count
                or receipt['observable_inputs']['anchor_frame'] != count
                or receipt['outcome']['outcome'] != label['outcome_target']
                or receipt['source_emitted_bundle_outcome_audit'] != event['source_bundle_audit']
                or receipt['privileged_teacher_fields']['teacher_eligible']):
            raise ValueError('Actual consumed23D controls and independent physical evidence differ')
        physical.append(dict(frame=count, outcome=label['outcome_target'],
            policy_causal_success=label['policy_causal_success'], initial_satisfied=label['initial_satisfied'],
            held=receipt['outcome']['evidence'][0]['fields']['held'], per_skill=label['per_skill']))
        if index < 2:
            continue
        before, number = count - 16, index - 2
        anchor_counts = list(range(before - 80, before + 1, 16))
        expected_hashes = {str(n): anchor_hash(frame(root, n)) for n in anchor_counts}
        query = requests[number]
        if (query['actual_consumed_actions_before_query'] != before or query['actual_anchor_frames'] != anchor_counts
                or query['padded_frames'] != 0 or query['privileged_truth_sent']
                or query['service_admission']['anchor_hashes'] != expected_hashes
                or not np.all(actions[:, 6] == 0)
                or label['outcome_evidence_end_frame'] != count):
            raise ValueError('Saved observations, actual FM history or official23 clock disagree')
        current, final = frame(root, before), frame(root, count)
        motion = {}
        for side, start in [('left', 7), ('right', 15)]:
            state = current['state_' + side + '_arm']
            requested = actions[:, start:start + 7] - state[None]
            moved = final['state_' + side + '_arm'] - state
            motion[side] = dict(requested_delta_rms=float(np.sqrt(np.mean(requested**2))),
                               observed_delta_rms=float(np.sqrt(np.mean(moved**2))))
        details.append(dict(chunk=number, start=before, end=count, arm_motion=motion,
                            yaw_mean=float(actions[:, 2].mean()), outcome=label['outcome_target']))
        consumed.append(actions)
    joined = np.concatenate(consumed)
    if len(joined) != controls or result['physical_observation_captures'] != 448 + controls + 1:
        raise ValueError('Actual execution and observation counts disagree')
    # Physical replay with equal source/seed is not assumed bitwise identical.
    control = root.parent / 'fm_control_v1'
    before = [frame(root, count) for count in range(368, 449, 16)]
    reference = [frame(control, count) for count in range(368, 449, 16)]
    comparison = dict(reference_arm='fm_control_v1',
        rgb_history_bitwise_equal={key: bool(np.array_equal(np.stack([r[key] for r in before]),
            np.stack([r[key] for r in reference]))) for key in CAMERAS},
        state_max_abs_difference={key: float(np.max(np.abs(np.stack([r['state_' + key] for r in before]) -
            np.stack([r['state_' + key] for r in reference])))) for key in STATE_WIDTHS})
    counts = sorted(set(list(range(448, 449 + controls, 32)) + [448 + controls]))
    wrists = sorted(set(448 + 16 * int(round(x)) for x in np.linspace(0, controls // 16, 6)))
    summary = dict(verified=True, arm=name, checkpoint_sha256=SCREEN[name], result_sha256=sha(result_path),
        events_sha256=sha(events_path), video_sha256=sha(root / 'actual_rollout/rollout.mp4'),
        actual_model_controls=controls, actual_chunks=len(requests), history_anchors_verified=6 * len(requests),
        actual_controls_match_physical_receipts=True, physical_boundaries=physical, chunks=details,
        post_chunk_outcomes=dict(Counter(row['outcome'] for row in physical[2:])),
        causal_success_boundaries=[row for row in physical[2:] if row['outcome'] == 'SUCCEEDED' and row['policy_causal_success']],
        paired_prefix_comparison=comparison,
        yaw_command_integral_not_measured_pose=float(joined[:, 2].sum() / 30.),
        head_review_frames=counts, wrist_review_frames=wrists, manual_review_pending=True,
        full_task_success_rate_claim=False, training_admissible=False, optimizer_updates=0,
        model_calls=0, new_simulator_controls=0,
        limitations=['Only one development training-instance GRASP/seed per policy, not task success rate.',
            'Physical receipts cover saved boundaries, not every between-chunk transient hold.',
            'Original collector does not save all32 model outputs; audit covers every consumed16 and service alignment.',
            'Equal replay commands/seed do not imply identical simulator rendering/physics; differences reported.',
            'Inherited C1 eligibility flags are not an approved annotation release.'])
    destination = root / 'review'
    destination.mkdir(exist_ok=False)
    write(destination / 'machine_checks.json', summary)
    make_sheet(root, destination / 'head-contact-sheet.png', counts, ['head_rgb'], 4)
    make_sheet(root, destination / 'wrists-contact-sheet.png', wrists, ['left_wrist_rgb', 'right_wrist_rgb'], 2)
    print(json.dumps({key: summary[key] for key in ('arm', 'verified', 'actual_model_controls', 'post_chunk_outcomes')}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--arm', choices=tuple(SCREEN), required=True)
    args = parser.parse_args()
    review(args.root.resolve(strict=True), args.arm)


if __name__ == '__main__':
    main()
