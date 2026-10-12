"""Read-only original-TRAIN observations and real periodic observer feedback.

This is annotation-derived command history, not a replay of actor actions.
Targets are retained for later human review, never used in observer prefixes.
"""
import hashlib
import json

from recovery_corpus import digest
from recovery_handoff_feedback import requests_for_candidate
from recovery_periodic_teacher import original_label_module, periodic_annotation_trace

PREFIX_FIELDS = ('task_name', 'parent_goal', 'issued_bundle', 'member_index', 'memory', 'served_controls')


def tree_sha(value):
    """Device-independent hash including dtype/shape; never omit image bytes."""
    h = hashlib.sha256()
    def visit(v):
        if isinstance(v, dict):
            h.update(b'dict')
            for k in sorted(v): visit(k); visit(v[k])
        elif isinstance(v, (list, tuple)):
            h.update(b'list'); visit(len(v))
            for x in v: visit(x)
        elif hasattr(v, 'detach'):
            a = v.detach().cpu().contiguous()
            visit(str(a.dtype)); visit(list(a.shape))
            # Viewing uint8 also supports bfloat16, which numpy cannot export.
            import torch
            h.update(a.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif hasattr(v, 'dtype') and hasattr(v, 'tobytes'):
            visit(str(v.dtype)); visit(list(v.shape)); h.update(v.tobytes())
        else:
            h.update(json.dumps(v, sort_keys=True, allow_nan=False).encode()); h.update(b'\0')
    visit(value)
    return h.hexdigest()


def unique_checks(row):
    """Deduplicate overlap only within one decision; never share task memory."""
    found = {}
    for request in row['requests']:
        for check in request['checks']:
            key = digest(check)
            if key in found and found[key] != check: raise ValueError('Check hash collision')
            found[key] = check
    return list(found.values())


def reconstruct_request(row, original, data, excluded):
    from recovery_periodic_teacher import validate_candidate_projection
    i = row['original_identity']; index = i['original_candidate_index']
    episode, frame, *_ = data.locate(index)
    validate_candidate_projection(original, episode, original_index=index, original_pair=data.candidates[index])
    if original['identity'] != i: raise ValueError('Changed original target identity')
    frame_map = {}
    for check in unique_checks(row):
        number = check['original_candidate_index']; ep, at, *_ = data.locate(number)
        if (int(data.candidates[number][0]) != i['episode_serial'] or ep['row'] != episode['row']
                or int(data.task_ids[number]) != i['task'] or at != check['frame']):
            raise ValueError('Cross-task/episode/frame original observation')
        if at in frame_map and frame_map[at] != number: raise ValueError('Duplicate original frame')
        frame_map[at] = number
    rebuilt, reason = requests_for_candidate(original, frame_map, excluded_groups=excluded)
    if reason is not None or rebuilt != row: raise ValueError('Altered or protected periodic request')
    return episode


def observation_and_prefix(data, processor, check, token_processor):
    """Exercise exact serving RGB/proprio path; poison target-side fields too."""
    import torch
    from recovery_causal_inference import observable_raw
    from g05.utils.memlite_causal_feedback import single_frame_member_prefix
    from check_memlite_periodic_processor import validate_processed_cameras, valid_token_count
    index = check['original_candidate_index']
    raw, identity = data.raw(index)
    episode, frame, *_ = data.locate(index)
    if frame != check['frame'] or raw['task'] != check['task_name']:
        raise ValueError('Observation no longer matches command clock/task')
    observed = dict(images={k:v[0].numpy().copy() for k,v in raw['images'].items()},
        proprio=data._raw_arrays(episode['row'])['observation.state'][frame].copy())
    original_images = {k:v.clone() for k,v in raw['images'].items()}
    # Action, annotation, current target and index never enter this raw input.
    serving_raw = observable_raw(observed, check['task_name'], data.config)
    prepared = processor._process_tensors(serving_raw)
    reference = processor._process_tensors(raw)
    fields = ('proprio', 'proprio_dim_is_pad', 'pixel_values', '_instructions')
    if tree_sha({k:prepared[k] for k in fields}) != tree_sha({k:reference[k] for k in fields}):
        raise ValueError('Original-reader and serving normalization/camera mismatch')
    validate_processed_cameras(processor, original_images, prepared)
    args = {k:check[k] for k in PREFIX_FIELDS}
    prefix = single_frame_member_prefix(processor.samples_builder, prepared, **args)
    poisoned = dict(prepared, model_projection={'active_skills_text':'FUTURE_LABEL_POISON'},
        action=torch.full((32,27), float('nan')), outcome_target='SUCCEEDED', reward=123456)
    other = single_frame_member_prefix(processor.samples_builder, poisoned, **args)
    if tree_sha(prefix) != tree_sha(other): raise ValueError('Answer/action/physics entered prefix')
    ids, attention = token_processor.encode_inference([prefix], device=torch.device('cpu'), mode='ar', training=False)
    receipt = dict(check_sha256=digest(check), observation_sha256=tree_sha(observed),
        prepared_sha256=tree_sha({k:prepared[k] for k in fields}), prefix_sha256=tree_sha(prefix),
        encoded_sha256=tree_sha([ids, attention]), tokens=valid_token_count(attention), source_identity=identity)
    return prefix, prepared['pixel_values'], receipt, observed


def feedback_from_trace(row, episode, trace, calibration):
    """Replay original commands and exact serving two-fresh-check gate.

    Never infer outcome from a target change. The unchanged teacher command is
    retained even if it disagrees with the actual frozen observer's estimate.
    """
    from g05.utils.memlite_causal_feedback import CausalFeedbackLedger, FeedbackIdentity
    from recovery_calibrated_observer import calibrated_prediction
    wanted = {r['request_id']:r for r in row['requests']}
    actual = {r['request_id']:r for r in trace}
    if len(actual) != len(trace) or set(actual) != set(wanted): raise ValueError('Missing/duplicate/foreign prediction')
    for key, r in actual.items():
        if set(r) != set(wanted[key]) | {'logits'} or any(r[k] != v for k,v in wanted[key].items()):
            raise ValueError('Prediction uses foreign observation window or prior command')
    i = row['original_identity']; labels = original_label_module()
    records, _ = periodic_annotation_trace(labels.anchor_records(episode['segments'], episode['phase']), episode['segments'])
    first = records[0]['frame']; end = i['frame']-first
    identity = FeedbackIdentity('periodic-observer:'+row['sample_id'], episode['task_name'], i['instance'],
        str(i['episode']), calibration['high_sha256'], calibration['observer_sha256'])
    ledger = CausalFeedbackLedger(identity, uncertainty_protocol='unready_zero_confidence_v1')
    for r in records:
        if r['frame'] >= i['frame']: break
        segment = episode['segments'][r['segment']]
        ledger.issued(identity, r['frame']-first, segment['semantic'], segment['parent'])
    if json.loads(ledger.projection(identity, end-16))['same_intent_controls'] + 16 != json.loads(
            row['original_projection']['execution_feedback'])['same_intent_controls']:
        raise ValueError('Reconstructed original intent clock drift')
    # All previous proposals are irrelevant to the production deque(maxlen=2).
    members = json.loads(ledger.bundle)
    for step in (end-16, end):
        ledger.observe_clock(identity, step)
        predictions = []
        for member, skill in enumerate(members):
            matches = [r for r in trace if r['member_index'] == member and r['control_step'] == step]
            if len(matches) != 1: raise ValueError('Missing aligned member prediction')
            predictions.append(calibrated_prediction(matches[0]['logits'],
                temperature=calibration['temperature'], verb=skill['verb']))
        ledger.estimated(identity, step, predictions, calibrated=True)
    feedback = ledger.projection(identity, end)
    before = json.loads(row['original_projection']['execution_feedback']); after = json.loads(feedback)
    for key in set(before) - {'estimated_member_outcomes','estimated_bundle_outcome'}:
        if before[key] != after[key]: raise ValueError('Prediction changed original counter/source: '+key)
    return feedback
