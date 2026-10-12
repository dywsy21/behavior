"""CPU reconstruction of all candidates and real processor check on reviewed RGB.

No fitting or automatic admission. Every current target remains same-frame
expert annotation; resampled command history is NOT an executed actor trace.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'scripts/rl/memlite_online/code'))
from recovery_periodic_teacher import validate_candidate_projection
from behavior_champion.release import AcceptedRelease, sha256


def validate_processed_cameras(processor, original_images, sample):
    """Verify raw-key -> configured camera-type renaming AND actual pixels.

    BaseProcessor.process_images keys tensors by camera_type, not raw RGB key.
    Test independent per-camera transforms so two equal-size wrist tensors
    cannot be silently exchanged while preserving keys/shapes.
    """
    import torch
    expected_mapping = (('head_rgb', 'exterior'), ('left_wrist_rgb', 'wrist_left'),
                        ('right_wrist_rgb', 'wrist_right'))
    mapping = tuple((meta['key'], meta['camera_type']) for meta in processor.shape_meta['images'])
    raw_keys, output_keys = zip(*expected_mapping)
    if (mapping != expected_mapping or tuple(original_images) != raw_keys
            or tuple(processor.samples_builder._image_keys) != raw_keys
            or tuple(sample['pixel_values']) != output_keys):
        raise ValueError('Raw/template/processed camera mapping or order changed')
    for raw_key, camera_type in mapping:
        expected = original_images[raw_key].clone()
        for transform in processor.val_transforms[raw_key]:
            expected = transform(expected)
        actual = sample['pixel_values'][camera_type]
        if (actual.dtype != expected.dtype or actual.shape != expected.shape
                or not torch.isfinite(actual).all() or not torch.equal(actual, expected)):
            raise ValueError('Processed RGB differs from its own source camera: ' + raw_key)
    return dict(mapping)


def make_token_processor(config):
    """Build the real token stack without constructing a VLM or using CUDA."""
    from omegaconf import OmegaConf
    from g05.models.g05.io.input_preprocessor import InputPreprocessor
    from g05.models.g05.g05_policy_qwen35 import G05PolicyQwen35
    cfg = OmegaConf.create(config['arch'])
    pp = cfg.input_preprocessor
    return InputPreprocessor(
        hf_processor_path=cfg.hf_processor_path,
        hf_processor_class=cfg.get('hf_processor_class', G05PolicyQwen35.default_hf_processor_class),
        action_tokenizer_class=cfg.action_tokenizer, at_config=cfg.AT_CONFIG, model=None,
        base_vocab_size=cfg.get('base_vocab_size', G05PolicyQwen35.default_base_vocab_size),
        padded_vocab_size=cfg.get('padded_vocab_size', G05PolicyQwen35.default_padded_vocab_size),
        pad_token_id=cfg.pad_token_id, image_token_index=cfg.image_token_index,
        num_image_tokens=cfg.vision.num_image_tokens, input_action_corruption=pp.input_action_corruption,
        pred_eov=pp.pred_eov, batchify_action=pp.batchify_action, pi05_ft_mode=pp.pi05_ft_mode,
        proprio_encoder=cfg.proprio_encoder, model_type=G05PolicyQwen35.default_model_type,
        model_cfg=cfg, add_loc_tokens=bool(cfg.get('add_loc_tokens', False)))


def validate_token_batch(token_processor, config, samples):
    """Exercise production token boundaries and CE weights; no model forward."""
    from types import SimpleNamespace
    import torch
    from g05.models.g05.g05_policy_memlite_planner_outcome import G05PolicyMEMLitePlannerOutcome as Policy
    if not samples or len(samples) > 8:
        raise ValueError('Expected 1..8 aligned CPU token samples')
    cfg = config['arch']; head = cfg['planner_outcome']
    shell = SimpleNamespace(processor=token_processor, training=True, planner_only=True,
        supervise_task_parent_format=head.get('supervise_task_parent_format', False),
        task_parent_formats=head.get('task_parent_format_by_task', {}),
        memory_update_ce_weight=head['memory_update_ce_weight'])
    ids, labels, attention, _ = token_processor.encode_train(samples, device=torch.device('cpu'),
        training=True, max_chunk_token_length=cfg['max_chunk_token_length'],
        max_pad_token_length=cfg.get('max_pad_token_length'))
    context, context_attention = token_processor.encode_inference(samples, device=torch.device('cpu'),
        mode='ar', training=True)
    Policy._assert_tokenized_prefix_contract(shell, context, context_attention, samples)
    positions = Policy._context_positions(ids, attention, context, context_attention)
    spans = Policy._mask_unsupervised_ar_fields(shell, ids, labels, positions, samples)
    weights, codes = Policy._planner_only_loss_token_weights(shell, labels, spans, samples=samples)
    checks = []
    for number, (sample, fields, position) in enumerate(zip(samples, spans, positions)):
        Policy._validate_planner_only_outcome_contract(sample, index=number)
        if weights is None or not torch.all(labels[number, :position + 1] == -100):
            raise ValueError('Planner context received supervision or memory weighting disappeared')
        for field in ('outcome_target', 'task_complete'):
            if (not torch.all(labels[number, fields[field]] == -100)
                    or not torch.all(weights[number, fields[field]] == 0)):
                raise ValueError('Unevidenced outcome/terminal tokens received CE: ' + field)
        for field, wanted_weight in (('next_decision', 1.), ('active_skills_semantic_json', 1.),
                                     ('memory_update', .25)):
            valid = labels[number, fields[field]] != -100
            if not valid.all() or not valid.numel() or not torch.all(weights[number, fields[field]] == wanted_weight):
                raise ValueError('Lost or incorrectly weighted planner supervision: ' + field)
        checks.append(dict(task_name=sample['task_name'],
            input_tokens=int(attention[number].sum()), context_tokens=int(context_attention[number].sum()),
            supervised_tokens=int((labels[number] != -100).sum()),
            supervised_field_tokens={field:int((labels[number, span] != -100).sum()) for field,span in fields.items()},
            field_codes=sorted(set(codes[number][labels[number] != -100].tolist())),
            memory_weight=.25, outcome_tokens=0, terminal_tokens=0))
    return checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'candidates', 'review', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--candidate-sha256', required=True)
    parser.add_argument('--review-sha256', required=True)
    parser.add_argument('--check-tokens', action='store_true', help='Also run actual mixed-task token/CE-mask stack on CPU')
    args = parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or args.output.exists():
        raise ValueError('CPU-only and fresh output required')
    if sha256(args.candidates) != args.candidate_sha256 or sha256(args.review) != args.review_sha256:
        raise ValueError('Changed candidate/review source')
    import torch
    from g05.data.memlite_stage1_dataset import Stage1Dataset
    from g05.utils.training.stage1_model import configuration, make_processor
    release = args.root / 'datasets/memlite-stage1-20260930-v4'
    accepted = AcceptedRelease(release)
    for name in ('train_candidates.npy', 'train_tasks.npy', 'episode_offsets.npy'): accepted.verify_file(name)
    config = configuration(args.root, 'high', accepted.manifest['task_names'])
    data = Stage1Dataset(release, config, 'high', 'train')
    rows = [json.loads(line) for line in args.candidates.read_text().splitlines()]
    review = json.loads(args.review.read_text())
    if review['candidate_sha256'] != args.candidate_sha256: raise ValueError('Foreign review')
    reviewed = {row['identity']['original_candidate_index']: row for row in review['rows']}
    processor = make_processor(config, False)
    token_processor = make_token_processor(config) if args.check_tokens else None
    args.output.mkdir(parents=True)
    verified, processed, seen, token_batch, token_checks = [], [], set(), [], []
    started = time.monotonic()
    for row in rows:
        index = row['identity']['original_candidate_index']
        if index in seen: raise ValueError('Duplicate candidate')
        seen.add(index)
        pair = data.candidates[index]
        ep = data.episode(int(pair[0]))
        projection = validate_candidate_projection(row, ep, original_index=index, original_pair=pair)
        if int(data.task_ids[index]) != row['identity']['task']: raise ValueError('Cross-task original index')
        verified.append(dict(index=index, projection_sha256=hashlib.sha256(json.dumps(projection, sort_keys=True).encode()).hexdigest()))
        if index not in reviewed: continue
        if reviewed[index]['identity'] != row['identity'] or reviewed[index]['projection'] != projection:
            raise ValueError('Reviewed image/target identity differs')
        raw, identity = data.raw(index)
        if identity['frame'] != row['identity']['frame'] or identity['episode'] != row['identity']['episode']:
            raise ValueError('Processor decoded a different source observation')
        camera_names = {'head_rgb':'base_0_rgb', 'left_wrist_rgb':'left_wrist_0_rgb', 'right_wrist_rgb':'right_wrist_0_rgb'}
        for name, pixels in raw['images'].items():
            if pixels.dtype != torch.uint8 or pixels.shape[0] != 1:
                raise ValueError('Expected original single-frame uint8 RGB')
            digest = hashlib.sha256(pixels[0].permute(1,2,0).contiguous().numpy().tobytes()).hexdigest()
            if digest != reviewed[index]['original_rgb_sha256'][str(identity['frame']) + ':' + camera_names[name]]:
                raise ValueError('Training reader RGB differs from the personally reviewed original frame')
        original_images = {name: pixels.clone() for name, pixels in raw['images'].items()}
        raw.update(model_projection=projection, action_is_pad=torch.ones(32, dtype=torch.bool),
            action={meta['key']: torch.zeros(32, meta['raw_shape']) for meta in config['raw_shape']['action']})
        sample = processor.preprocess(raw); content = sample['samples']
        for key in ('memory', 'previous_intent', 'execution_feedback', 'memory_update',
                    'active_skills_semantic_json', 'outcome_supervision_mask', 'low_action_supervision_mask'):
            if content[key] != projection[key]: raise ValueError('Processor changed planner field: ' + key)
        if content['known_previous_outcome'] != 'UNKNOWN' or not sample['action_is_pad'].all():
            raise ValueError('Unexpected outcome or low-action supervision')
        if tuple(torch.nonzero(sample['action_dim_is_pad']).flatten().tolist()) != (7, 8, 17, 18):
            raise ValueError('R1Pro padding mapping changed')
        camera_mapping = validate_processed_cameras(processor, original_images, sample)
        template = content['template'] if 'template' in content else processor.samples_builder.template
        prefix, separator, suffix = template.partition('<EOC>')
        if (not separator or any(token in prefix for token in ('active_skills_semantic_json',
                 'current_parent_goal', 'memory_update', 'outcome_target'))):
            raise ValueError('Current target leaked into autoregressive prefix')
        processed.append(dict(identity=identity, original_reviewed_rgb_exact=True,
                              camera_mapping=camera_mapping, processed_rgb_exact_per_camera=True,
                              pixel_shapes={key:list(value.shape) for key,value in sample['pixel_values'].items()},
                              outcome_supervision=False, low_supervision=False))
        if token_processor is not None:
            token_batch.append(content)
            if len(token_batch) == 8:
                token_checks.extend(validate_token_batch(token_processor, config, token_batch))
                token_batch.clear()
        if len(processed) % 20 == 0: print(json.dumps(dict(processed_review_cases=len(processed))), flush=True)
    if len(verified) != 1800 or len(processed) != len(reviewed) or len(reviewed) != 120:
        raise ValueError('Incomplete fixed candidate/review coverage')
    if token_batch: token_checks.extend(validate_token_batch(token_processor, config, token_batch))
    if args.check_tokens and len(token_checks) != len(reviewed):
        raise ValueError('Incomplete token/CE-mask coverage')
    result = dict(status='passed_cpu_causal_source_and_processor_contract', candidate_sha256=args.candidate_sha256,
        review_sha256=args.review_sha256, reconstructed_candidates=len(verified), processed_review_cases=len(processed),
        candidates=verified, processed=processed, seconds=time.monotonic()-started,
        optimizer_updates=0, decoded_current_rgb_images=len(processed)*3,
        token_checks=token_checks, tokenized_review_cases=len(token_checks), model_forwards=0,
        training_admitted=False, limit='Engineering contract only; no physical outcome/action truth or model effect')
    (args.output / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({key:value for key,value in result.items() if key not in ('candidates', 'processed', 'token_checks')}), flush=True)


if __name__ == '__main__': main()
