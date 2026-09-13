"""Verify every A4 L1 action/history/physical receipt, paired to aligned A3."""
from collections import Counter
import json
from pathlib import Path
import sys

from paired_a3_a4_actions import WORK, CHECKPOINTS, sha, write_new


def main():
    import numpy as np
    root = Path('/mnt/sdc1/robodojo/behavior_dev/a4_radio_e121_l1_20260913_v1')
    runtime = WORK/'native_a3_aligned_prefix_runtime_v2'
    output = root/'actual_analysis.json'
    if output.exists():
        raise FileExistsError('Keep previous actual analysis')
    sys.path[:0] = [str(runtime), str(WORK/'c1_v2'),
        str(WORK/'a3_history_serving_candidate_v1/src')]
    from native_a2_prefix_history import actual_anchor_hashes
    from c1_feedback.feedback_c1 import hydrate_observable_prefix
    manifest_path = root/'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    for path, expected in manifest['files'].items():
        if sha(path) != expected:
            raise ValueError('Frozen diagnostic source changed: '+path)
    result_path, events_path = root/'actual_rollout/collection_result.json', root/'actual_rollout/events.jsonl'
    result = json.loads(result_path.read_text())
    launch = json.loads((root/'rollout.launch.json').read_text())
    if (launch['manifest_sha256'] != sha(manifest_path) or result['status'] != 'complete'
            or result['prefix_actions_executed'] != manifest['prefix_actions']
            or result['high_policy_calls'] != 0 or result['training_admissible']
            or result['full_task_success_rate_claim'] or result['correction_teacher']
            or result['low_service_identity']['checkpoint_sha256'] != manifest['checkpoint_sha256']
            or not result['low_service_identity']['checkpoint_load']['full_base_and_adapter_bitwise_equal']):
        raise ValueError('Not the completed independent exact A3 low-only diagnostic')
    fix = result['low_service_identity']['inference_alignment_fix']
    wire_path = root/'wire_probe/result.json'
    wire = json.loads(wire_path.read_text())
    if (fix['receipt_sha256'] != sha(runtime/'inference_alignment_receipt.json')
            or fix['receipt_sha256'] != manifest['inference_alignment_receipt_sha256']
            or fix['padded_dimensions'] != [7,8,17,18] or fix['action_execution_start_index'] != 0
            or fix['expert_actions_used'] or not wire['passed'] or wire['model_calls'] != 7
            or not wire['all_alignment_admissions_correct']):
        raise ValueError('Real corrected model service identity/admission differs')
    rows = [json.loads(line) for line in events_path.open()]
    requests = result['actual_history_requests']
    if len(rows) != result['events_written'] or len(rows) != len(requests)+2:
        raise ValueError('Actual events and serving queries have different counts')
    context = json.loads(Path(manifest['context_path']).read_text())
    goal = {key: context['evaluated_bundle'][key] for key in ('parent_goal', 'active_skills_semantic_json')}
    commands, outcomes, details = [], Counter(), []
    for index, event in enumerate(rows):
        count = 0 if index == 0 else manifest['prefix_actions']+16*(index-1)
        if (event['event_index'] != index or event['observable_refs_audit']['anchor_frame'] != count
                or event['observable_prefix_descriptor']['active_bundle'] != goal
                or event['source']['split'] != 'train' or event['source']['instance_id'] != 138
                or event['admission']['correction_teacher_eligible'] or event['admission']['reinject']):
            raise ValueError('Event clock/skill/source/privilege binding differs')
        label = event['physical_outcome_label']
        envelope_ref = event['physical_evidence_ref']
        if sha(envelope_ref['path']) != envelope_ref['sha256']:
            raise ValueError('Physical evidence envelope changed')
        envelope = json.loads(Path(envelope_ref['path']).read_text())
        if sha(envelope['physical_receipt_path']) != envelope['physical_receipt_sha256']:
            raise ValueError('Original physical observation receipt changed')
        physical = json.loads(Path(envelope['physical_receipt_path']).read_text())
        if (physical['observable_inputs']['anchor_frame'] != count
                or physical['outcome']['outcome'] != label['outcome_target']
                or physical['source_emitted_bundle_outcome_audit'] != event['source_bundle_audit']
                or physical['privileged_teacher_fields']['teacher_eligible']):
            raise ValueError('Physical evidence is not the same actual event')
        ref = event['action_ref']
        if sha(ref['path']) != ref['sha256']:
            raise ValueError('Actually consumed action file changed')
        actions = np.load(ref['path'], allow_pickle=False)
        original = np.asarray(physical['actual_actions_23d'], dtype=np.float32).reshape(-1, 23)
        expected = 0 if index < 2 else 16
        if (actions.shape != (expected, 23) or ref['actual_consumed_actions'] != expected
                or not np.array_equal(actions, original) or not np.isfinite(actions).all()):
            raise ValueError('Saved actual controls and physics evidence differ')
        if index < 2:
            continue
        query = requests[index-2]
        before = hydrate_observable_prefix(rows[index-1])
        after = hydrate_observable_prefix(event)
        actual_counts = list(range(count-16-80, count-16+1, 16))
        wire = dict(history_action_counts=actual_counts,
            images={key: np.stack(value) for key, value in before['rgb_history'].items()},
            state={key: np.stack(value) for key, value in before['proprio_history'].items()})
        if (query['actual_consumed_actions_before_query'] != count-16
                or query['actual_anchor_frames'] != actual_counts or query['padded_frames'] != 0
                or query['privileged_truth_sent']
                or query['service_admission']['anchor_hashes'] != actual_anchor_hashes(wire)):
            raise ValueError('Model did not receive the exact independently saved pre-query anchors')
        outcomes[label['outcome_target']] += 1
        row = dict(chunk=index-2, start=count-16, end=count, actual_input_anchor_frames=actual_counts,
            actual_input_tensor_hashes_match=True, outcome=label['outcome_target'],
            causal_success=label['policy_causal_success'], per_skill=label['per_skill'],
            mean_abs_base_command=np.mean(np.abs(actions[:, :3]), axis=0).tolist(),
            gripper_controls={side: dict(min=float(actions[:, dim].min()), max=float(actions[:, dim].max()),
                mean=float(actions[:, dim].mean())) for side, dim in (('left', 14), ('right', 22))},
            arm_motion={})
        for side, start in (('left', 7), ('right', 15)):
            initial = np.asarray(before['proprio_history'][side+'_arm'][-1])
            final = np.asarray(after['proprio_history'][side+'_arm'][-1])
            requested = actions[:, start:start+7]-initial
            row['arm_motion'][side] = dict(requested_joint_change_rms=float(np.sqrt(np.mean(requested**2))),
                actual_chunk_joint_change_rms=float(np.sqrt(np.mean((final-initial)**2))),
                initial=initial.tolist(), final=final.tolist())
        details.append(row)
        commands.append(actions)
    joined = np.concatenate(commands)
    if (len(joined) != result['native_policy_actions_consumed']
            or result['physical_observation_captures'] != manifest['prefix_actions']+len(joined)+1):
        raise ValueError('Actual executed controls/observation capture count differ')
    old_path = WORK/'a3_aligned_radio_e121_l1_v2/actual_analysis_v2.json'
    old = json.loads(old_path.read_text())
    if (not old['passed'] or old['checkpoint_sha256'] != CHECKPOINTS['A3'][1]
            or old['prefix_actions'] != manifest['prefix_actions']
            or old['original_semantic_condition'] != goal):
        raise ValueError('The paired aligned A3 differs in checkpoint/initial prefix/skill')
    if manifest['checkpoint_sha256'] != CHECKPOINTS['A4'][1]:
        raise ValueError('Not the completed A4 checkpoint')
    old_events_path = WORK/'a3_aligned_radio_e121_l1_v2/actual_rollout/events.jsonl'
    if sha(old_events_path) != old['events_sha256']:
        raise ValueError('Prior aligned A3 physical event file changed')
    old_rows = [json.loads(line) for line in old_events_path.open()]
    a, b = hydrate_observable_prefix(old_rows[1]), hydrate_observable_prefix(rows[1])
    prefix_comparison = dict(
        state_max_abs_difference={key: float(np.max(np.abs(
            np.asarray(a['proprio_history'][key])-np.asarray(b['proprio_history'][key]))))
            for key in a['proprio_history']},
        rgb_history_bitwise_equal={key: bool(np.array_equal(
            np.asarray(a['rgb_history'][key]), np.asarray(b['rgb_history'][key])))
            for key in a['rgb_history']},
        state_replay_bitwise_equality_not_assumed=True)
    success_events = [dict(event_index=r['event_index'], anchor=r['observable_refs_audit']['anchor_frame'],
        per_skill=r['physical_outcome_label']['per_skill'],
        initial_satisfied=r['physical_outcome_label']['initial_satisfied'])
        for r in rows[2:] if r['physical_outcome_label']['outcome_target']=='SUCCEEDED'
        and r['physical_outcome_label']['policy_causal_success']]
    report = dict(kind='actual_A4_aligned_real_prefix_L1_audit_v1', passed=True,
        manifest_sha256=sha(manifest_path), result_sha256=sha(result_path), events_sha256=sha(events_path),
        video_sha256=sha(root/'actual_rollout/rollout.mp4'), checkpoint_sha256=manifest['checkpoint_sha256'],
        prefix_actions=448, actual_model_chunks=len(requests), actual_model_actions=len(joined),
        distinct_actual_physical_observation_captures=result['physical_observation_captures'],
        high_model_calls=0, post_chunk_outcomes=dict(outcomes), actual_history_all_queries_verified=True,
        original_semantic_condition=goal, actual_controls_equal_saved_physical_receipts=True,
        transient_between_chunk_holds_cannot_be_excluded=True, original_source_code_sha256=sha(__file__),
        training_admissible=False, full_task_success_rate_claim=False, correction_teacher=False,
        initial_physical_outcome=rows[0]['physical_outcome_label'],
        before_policy_physical_outcome=rows[1]['physical_outcome_label'], chunks=details,
        personal_video_review='separate manual report pending')
    report.update(prefix_comparison=prefix_comparison, actual_causal_success_events=success_events,
        inference_alignment_fix=fix, actual_wire_probe_sha256=sha(wire_path),
        paired_old_audit_sha256=sha(old_path), paired_old_post_chunk_outcomes=old['post_chunk_outcomes'],
        paired_old_model_actions=old['actual_model_actions'],
        unchanged_prefix_and_semantic_condition=True, checkpoint_changed_to_A4=True,
        chunk_level_alignment_metadata_not_recorded_by_unchanged_collector=True)
    write_new(output, report)
    print(json.dumps({key: report[key] for key in ('passed', 'actual_model_chunks', 'actual_model_actions',
        'post_chunk_outcomes', 'actual_history_all_queries_verified', 'training_admissible')}), flush=True)


if __name__ == '__main__':
    main()
