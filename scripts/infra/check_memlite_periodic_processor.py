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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'candidates', 'review', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--candidate-sha256', required=True)
    parser.add_argument('--review-sha256', required=True)
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
    args.output.mkdir(parents=True)
    verified, processed, seen = [], [], set()
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
        if len(processed) % 20 == 0: print(json.dumps(dict(processed_review_cases=len(processed))), flush=True)
    if len(verified) != 1800 or len(processed) != len(reviewed) or len(reviewed) != 120:
        raise ValueError('Incomplete fixed candidate/review coverage')
    result = dict(status='passed_cpu_causal_source_and_processor_contract', candidate_sha256=args.candidate_sha256,
        review_sha256=args.review_sha256, reconstructed_candidates=len(verified), processed_review_cases=len(processed),
        candidates=verified, processed=processed, seconds=time.monotonic()-started,
        optimizer_updates=0, decoded_current_rgb_images=len(processed)*3,
        training_admitted=False, limit='Engineering contract only; no physical outcome/action truth or model effect')
    (args.output / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({key:value for key,value in result.items() if key not in ('candidates', 'processed')}), flush=True)


if __name__ == '__main__': main()
