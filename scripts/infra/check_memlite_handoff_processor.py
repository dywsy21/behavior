"""CPU acceptance of the actual handoff training adapter and serving prefix.

Reconstruct every original periodic target and all383 frozen predictions from
their real logits. Check current original observations against the generation
receipts, then exercise production mixed-batch token/CE masking. No model load,
CUDA, outcome labels, new calibration or fitting.
"""
import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO/'src'), str(REPO/'scripts'), str(REPO/'scripts/eval'),
               str(REPO/'scripts/infra'), str(REPO/'scripts/rl/memlite_online/code')]
from recovery_corpus import digest, file_sha
from recovery_handoff_inference import tree_sha, unique_checks, feedback_from_trace
from recovery_periodic_data import PeriodicTeacherPool, with_periodic_teacher
from recovery_periodic_teacher import validate_candidate_projection
from check_memlite_periodic_processor import make_token_processor, validate_token_batch, validate_processed_cameras


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--recipe', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    a = ap.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or a.output.exists():
        raise ValueError('Fresh CPU-only actual adapter audit required')
    if subprocess.check_output(['git', 'status', '--porcelain'], cwd=REPO, text=True).strip():
        raise ValueError('Audit must use a clean frozen source')
    import torch
    torch.set_num_threads(2)
    from g05.data.memlite_stage1_dataset import Stage1Dataset
    from g05.utils.memlite_causal_feedback import single_frame_planner_prefix
    from g05.utils.memlite_skill_protocol import planner_input_projection
    from recovery_causal_inference import observable_raw
    from generate_memlite_handoff_feedback import source_inputs
    recipe = json.loads(a.recipe.read_text()); root = Path(recipe['root'])
    cfg_path = REPO/'configs/recovery_sft/a800_handoff_observer_v1.json'
    cfg = json.loads(cfg_path.read_text())
    if Path(cfg['root']).resolve() != root.resolve(): raise ValueError('Foreign handoff recipe root')
    context = source_inputs(cfg)
    pool = PeriodicTeacherPool(root, recipe['H1']['periodic_teacher'], repo=REPO)
    if pool.handoff is None: raise ValueError('Missing actual learned-feedback adapter')
    handoff = pool.handoff
    if handoff.result['config_sha256'] != file_sha(cfg_path): raise ValueError('Changed generation source recipe')
    data = with_periodic_teacher(Stage1Dataset, pool)(context.data.release, context.data.config, 'high', 'train')
    processor = context.processor; tokens = make_token_processor(data.config)
    reconstructed = []
    for row in pool.rows:
        index = row['identity']['original_candidate_index']; pair = data.candidates[index]
        projection = validate_candidate_projection(row, data.episode(int(pair[0])),
            original_index=index, original_pair=pair)
        actual = handoff.projection(index, projection)
        if any(actual[k] != projection[k] for k in projection if k != 'execution_feedback'):
            raise ValueError('An expert target changed')
        reconstructed.append(dict(index=index, original_sha256=digest(projection), actual_sha256=digest(actual)))
    trace = defaultdict(list)
    for line in handoff.files['handoff_trace'].read_text().splitlines():
        row = json.loads(line); sample_id = row.pop('decision_sample_id'); trace[sample_id].append(row)
    checks = {}
    for line in handoff.files['handoff_checks'].read_text().splitlines():
        row = json.loads(line); key = (row['sample_id'], row['check_sha256'])
        if key in checks: raise ValueError('Duplicated original observation receipt')
        checks[key] = row
    review = json.loads(handoff.files['handoff_review_manifest'].read_text())
    reviewed = {r['sample_id']: r for r in review['rows']}
    if len(reviewed) != 160: raise ValueError('Missing personal review rows')
    a.output.mkdir(parents=True); started = time.monotonic()
    accepted, batch, token_checks, reviewed_matched = [], [], [], set()
    for request in sorted(context.rows, key=lambda r: r['sample_id']):
        index = request['original_identity']['original_candidate_index']; sid = request['sample_id']
        ep, frame, *_ = data.locate(index)
        expected_feedback = feedback_from_trace(request, ep, trace[sid], context.calibration)
        raw, identity = data.raw(index); projection = raw['model_projection']
        if projection['execution_feedback'] != expected_feedback:
            raise ValueError('Actual adapter differs from real gated observer logits')
        observed = dict(images={k:v[0].numpy().copy() for k,v in raw['images'].items()},
            proprio=data._raw_arrays(ep['row'])['observation.state'][frame].copy())
        last = unique_checks(request)[-1]; expected = checks[sid, digest(last)]
        # The wrapper adds an audit-only history protocol, not an actor feature.
        source_identity = {k:v for k,v in identity.items() if k != 'planner_history_protocol'}
        if tree_sha(observed) != expected['observation_sha256'] or source_identity != expected['source_identity']:
            raise ValueError('Actual SFT decoded a different current observation')
        if sid in reviewed:
            rr = reviewed[sid]
            if rr['identity'] != request['original_identity'] or rr['projection'] != request['original_projection']:
                raise ValueError('Human reviewed a different source/target')
            for key, rgb in observed['images'].items():
                if tree_sha(rgb) != rr['original_rgb_sha256'][f'{frame}:{key}']:
                    raise ValueError('SFT pixels differ from personally reviewed original panels')
            reviewed_matched.add(sid)
        originals = {k:v.clone() for k,v in raw['images'].items()}
        prepared = processor.preprocess(raw); sample = prepared['samples']
        validate_processed_cameras(processor, originals, prepared)
        if (not prepared['action_is_pad'].all() or sample['low_action_supervision_mask']
                or sample['outcome_supervision_mask'] or sample['known_previous_outcome'] != 'UNKNOWN'
                or tuple(torch.nonzero(prepared['action_dim_is_pad']).flatten().tolist()) != (7,8,17,18)):
            raise ValueError('New outcome/BC label or changed real23/model27 mapping')
        for key in ('execution_feedback', 'memory', 'previous_intent', 'active_skills_semantic_json', 'memory_update'):
            if sample[key] != projection[key]: raise ValueError('Processor changed planner field: '+key)
        causal = planner_input_projection(projection)
        serving_prepared = processor._process_tensors(observable_raw(observed, projection['task_name'], data.config))
        normalized_fields = ('pixel_values', 'proprio', 'proprio_dim_is_pad')
        if tree_sha({k:prepared[k] for k in normalized_fields}) != tree_sha({k:serving_prepared[k] for k in normalized_fields}):
            raise ValueError('Actual serving normalization differs from training')
        serving = single_frame_planner_prefix(processor.samples_builder, serving_prepared, causal)
        train_prefix = tokens.encode_inference([sample], device=torch.device('cpu'), mode='ar', training=False)
        serve_prefix = tokens.encode_inference([serving], device=torch.device('cpu'), mode='ar', training=False)
        if tree_sha(train_prefix) != tree_sha(serve_prefix): raise ValueError('Training/serving prefix mismatch')
        poisoned = dict(serving_prepared, model_projection={'active_skills_text':'FUTURE_TARGET'}, reward=1234,
            action=torch.full((32,27), float('nan')), outcome_target='SUCCEEDED')
        if tree_sha(serving) != tree_sha(single_frame_planner_prefix(processor.samples_builder, poisoned, causal)):
            raise ValueError('Target/physics entered serving prefix')
        batch.append(sample)
        if len(batch) == 8:
            token_checks.extend(validate_token_batch(tokens, data.config, batch)); batch.clear()
        accepted.append(dict(sample_id=sid, identity=identity, observation_sha256=tree_sha(observed),
            prefix_sha256=tree_sha(train_prefix), estimated_outcome=json.loads(expected_feedback)['estimated_bundle_outcome']))
        if len(accepted) % 20 == 0: print(json.dumps(dict(processed=len(accepted))), flush=True)
    if batch: token_checks.extend(validate_token_batch(tokens, data.config, batch))
    if len(accepted) != 383 or len(token_checks) != 383 or len(reconstructed) != 1800 or reviewed_matched != set(reviewed):
        raise ValueError('Incomplete actual adapter audit')
    # Reconstruct the exact expert sequence with the old control's recovery
    # rows/rank slots. No independent reshuffling or feedback-based sampling.
    from recovery_periodic_data import remap_control_schedule
    original_control = json.loads(pool.control_path.read_text())
    periodic_control = json.loads(handoff.files['handoff_control_schedule'].read_text())
    # Sorted new_events do not encode rank order: load the real admitted pool.
    from recovery_sft_data import require_training_pool
    original_recovery = REPO/'configs/recovery_sft/a800_h1_periodic_final_transition_v1.json'
    recovery_cfg = json.loads(original_recovery.read_text())
    admission, rows = require_training_pool(root/recovery_cfg['union']/'admission', 'planner', recovery_cfg['admission_sha256'])
    train_rows = [r for r in rows if r['candidate']['split'] == 'train']
    binding = dict(component='H1',parent_sha256=recipe['parents']['high']['sha256'],
        admission_sha256=recovery_cfg['admission_sha256'], world_size=8,micro_batch=1)
    schedule = remap_control_schedule(original_control,pool.by_task,lambda i:int(data.task_ids[i]),
        seed=recipe['seed'],recovery_rows=train_rows,binding=binding)
    if schedule != periodic_control['schedule'] or len(schedule) != recipe['maximum_updates_per_line']:
        raise ValueError('Learned-feedback fitting changed the same-parent periodic control sequence')
    result = dict(schema='handoff_planner_processor_audit_v1', status='passed',
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
        recipe_sha256=file_sha(a.recipe), handoff_admission_sha256=file_sha(handoff.admission_path),
        feedback_sha256=file_sha(handoff.files['handoff_feedback']), candidate_sha256=pool.spec['candidate_sha256'],
        original_targets_reconstructed=1800, learned_feedback_rows=383, tokenized_rows=383,
        reviewed_current_images_matched=len(reviewed_matched)*3, original_observations_match=True,
        estimates_recomputed_from_logits=True, training_serving_prefix_equal=True,
        same_periodic_control_schedule=True, schedule_sha256=digest(schedule),
        outcome_tokens_supervised=0, low_action_supervision=False, physical_truth_added=0,
        model_forwards=0, optimizer_updates=0, seconds=time.monotonic()-started,
        reconstructed=reconstructed, processed=accepted, token_checks=token_checks)
    (a.output/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('reconstructed','processed','token_checks')}),flush=True)


if __name__ == '__main__': main()
