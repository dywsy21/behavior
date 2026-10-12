"""Fixed TRAIN fit diagnosis: field CE, first-choice margin, free generation.

Never optimize, change labels, relax the runtime parser, or use EVAL rows.
First-choice probabilities are teacher-forced conditional diagnostics, NOT
free-generation probabilities or a physical outcome estimate.
"""
import argparse
import gc
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO/'src'), str(REPO/'scripts'), str(REPO/'scripts/infra'),
               str(REPO/'scripts/eval'), str(REPO/'scripts/eval/memlite_sft100'),
               str(REPO/'scripts/rl/memlite_online/code')]
from audit_memlite_handoff_training import load_audit
from recovery_calibration_launch import bound, read, write_new
from recovery_corpus import digest, file_sha
from recovery_generation_metrics import score_event, summarize
from recovery_generation_shards import generation_shard
from recovery_handoff_exposure import choice_cache_index, preprocess_planner_example
from recovery_handoff_inference import tree_sha
from recovery_process_cache import isolated_cache_env


def choice_layout(model, prepared):
    """CPU-testable exact field/mask mapping, independent of model logits."""
    import torch
    from g05.utils.memlite_planner_fields import planner_field_text
    from g05.utils.memlite_skill_protocol import canonical_json, parse_active_skills_text
    sample = prepared['samples']; samples = [sample]
    ids, labels, attention, _ = model.processor.encode_train(samples, device=torch.device('cpu'),
        training=False, max_chunk_token_length=model.max_chunk_token_length,
        max_pad_token_length=model.max_pad_token_length)
    context, context_attention = model.processor.encode_inference(samples, device=torch.device('cpu'),
        mode='ar', training=False)
    positions = model._context_positions(ids, attention, context, context_attention)
    spans = model._mask_unsupervised_ar_fields(ids, labels, positions, samples)[0]
    mask = (labels[0,1:] != -100).tolist()
    previous = canonical_json(parse_active_skills_text(sample['previous_intent'], allow_empty=False))
    alternatives = {'active_skills_semantic_json': 'Active skills: ' + previous}
    if sample['current_parent_goal_supervision_mask'] and sample['previous_parent_goal'] != 'None':
        alternatives['current_parent_goal'] = 'Parent goal: ' + sample['previous_parent_goal']
    text_processor = model.processor.modality_processors['text']
    def encode(text):
        return list(text_processor.process(text, is_masked=False, training=False,
            tokenizer=model.processor.tokenizer, sample_dict=sample)[0])
    result = {}
    for field, repeated in alternatives.items():
        target = encode(planner_field_text(sample, field)); span = spans[field]
        if ids[0,span].tolist() != target:
            raise ValueError('Diagnostic field tokenization differs from actual training')
        where = choice_cache_index(target, encode(repeated), span.start, labels[0].tolist(), mask)
        if where is None:
            result[field] = dict(available=False, reason='identical_or_strict_prefix_tokenization')
            continue
        result[field] = dict(where, available=True)
    return dict(choices=result, mask=mask, seq_len=labels.shape[1]-1)


def choice_statistics(model, prepared, expected_layout_sha256):
    """Reuse the actual CE cache; no second model forward or new objective."""
    import torch
    layout = choice_layout(model,prepared)
    if digest(layout) != expected_layout_sha256:
        raise ValueError('Model token layout differs from CPU acceptance')
    cache = model.model.ar_helper._last_ce_cache
    if (cache is None or cache['batch_size'] != 1 or cache['shift_logits'] is None
            or cache['seq_len'] != layout['seq_len'] or cache['mask'].cpu().tolist() != layout['mask']):
        raise ValueError('Missing or misaligned exact single-row eager CE cache')
    result = {}
    for field, where in layout['choices'].items():
        if not where['available']:
            result[field] = where
            continue
        index = where['ce_cache_index']; wanted = where['target_token']; old = where['repeated_token']
        if int(cache['shift_labels_masked'][index]) != wanted:
            raise ValueError('Wrong shifted CE target at first choice')
        logits = cache['shift_logits'][index].float()
        if not torch.isfinite(logits).all(): raise ValueError('Nonfinite decision logits')
        normalizer = torch.logsumexp(logits, 0)
        result[field] = dict(where, teacher_forced=True,
            target_log_probability=float(logits[wanted]-normalizer),
            repeated_log_probability=float(logits[old]-normalizer),
            target_minus_repeated_logit=float(logits[wanted]-logits[old]),
            greedy_token=int(logits.argmax()), target_is_greedy=int(logits.argmax()) == wanted,
            preceding_fields_are_teacher_forced=True)
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', type=Path, required=True)
    ap.add_argument('--mode', choices=('cpu','model'), required=True)
    ap.add_argument('--model', choices=('control','candidate'), required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--num-shards', type=int, default=1)
    ap.add_argument('--shard-index', type=int, default=0)
    ap.add_argument('--export-registration', type=Path,
                    help='CPU-only, exact-byte portable diagnostic bundle; no new training data')
    args = ap.parse_args(); cfg = read(args.config); root = Path(cfg['root'])
    if (args.output.exists() or subprocess.check_output(['git','status','--porcelain'],cwd=REPO,text=True).strip()
            or cfg['node'] not in subprocess.check_output(['hostname','-I'],text=True).split()):
        raise ValueError('Require fresh output, clean frozen source and the registered node')
    gpu = os.environ.get('CUDA_VISIBLE_DEVICES')
    if args.export_registration and (args.mode != 'cpu' or args.num_shards != 1 or args.shard_index != 0):
        raise ValueError('Portable export requires the entire CPU-only TRAIN universe')
    def idle():
        if gpu not in tuple(map(str,range(1,8))) or subprocess.check_output(
                ['nvidia-smi','-i',gpu,'--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
            raise ValueError('Select one actually idle lc3 GPU1..7; preserve all other jobs')
    if args.mode == 'cpu':
        if gpu != '': raise ValueError('CPU mode must explicitly hide CUDA')
    else: idle()
    audit_data, exposure, exposure_rows = load_audit(cfg)
    by_index = {r['index']:r for r in exposure_rows}
    recipe = read(bound(REPO,cfg['recipe']))
    generation = read(bound(REPO,cfg['observer_generation']))
    if generation.get('schema') != 'periodic_handoff_observer_generation_v1':
        raise ValueError('This entry cannot consume heldout observer requests')
    # Only checkpoint identities are reused from that existing pair registration;
    # none of its EVAL inputs or predictions are opened.
    spec = read(bound(REPO,cfg['model_specs_only']))['models'][args.model]
    from recovery_handoff_heldout import validate_fixed_fit
    validate_fixed_fit(read(bound(root,spec['fit_result'])),read(bound(root,spec['latest'])),spec['checkpoint'])
    args.output.mkdir(parents=True); os.environ.update(isolated_cache_env(args.output))
    import torch
    torch.set_num_threads(2); torch.manual_seed(cfg['seed']); started = time.monotonic()
    from generate_memlite_handoff_feedback import source_inputs
    from g05.data.memlite_stage1_dataset import Stage1Dataset, collate_stage1, to_device
    from g05.utils.memlite_causal_feedback import single_frame_planner_prefix
    from g05.utils.memlite_skill_protocol import planner_input_projection
    from recovery_periodic_data import PeriodicTeacherPool, with_periodic_teacher
    from recovery_causal_inference import observable_raw
    from check_memlite_periodic_processor import make_token_processor
    context = source_inputs(generation)
    pool = PeriodicTeacherPool(root,recipe['H1']['periodic_teacher'],repo=REPO)
    if pool.handoff is None: raise ValueError('Missing admitted learned-feedback overlay')
    data = with_periodic_teacher(Stage1Dataset,pool)(context.data.release,context.data.config,'high','train')
    requests = sorted(context.rows,key=lambda r:r['sample_id'])
    if len(requests) != cfg['rows'] or cfg['rows'] != 383:
        raise ValueError('Missing or filtered original TRAIN universe')
    schedule = [('train_'+by_index[r['original_identity']['original_candidate_index']]['kind'],i)
                for i,r in enumerate(requests)]
    local, sharding = generation_shard(schedule,args.shard_index,args.num_shards)
    accepted = {r['sample_id']:r for r in audit_data['processor_audit']['processed']}
    tokenizer = make_token_processor(data.config); prepared_rows = []; cpu_rows = []
    from types import SimpleNamespace
    from g05.models.g05.g05_policy_memlite_planner_outcome import G05PolicyMEMLitePlannerOutcome as Policy
    arch = data.config['arch']; head = arch['planner_outcome']
    shell = SimpleNamespace(processor=tokenizer,training=False,planner_only=True,
        max_chunk_token_length=arch['max_chunk_token_length'],max_pad_token_length=arch.get('max_pad_token_length'),
        supervise_task_parent_format=head.get('supervise_task_parent_format',False),
        task_parent_formats=head.get('task_parent_format_by_task',{}))
    shell._context_positions = Policy._context_positions
    shell._mask_unsupervised_ar_fields = lambda *args: Policy._mask_unsupervised_ar_fields(shell,*args)
    for position, split, index in local:
        request = requests[index]; sid = request['sample_id']; candidate = request['original_identity']['original_candidate_index']
        raw, identity = data.raw(candidate); ep, frame, *_ = data.locate(candidate)
        observed = dict(images={k:v[0].numpy().copy() for k,v in raw['images'].items()},
                        proprio=data._raw_arrays(ep['row'])['observation.state'][frame].copy())
        if identity != accepted[sid]['identity'] or tree_sha(observed) != accepted[sid]['observation_sha256']:
            raise ValueError('Current training observation differs from the accepted original')
        target, prepared = preprocess_planner_example(context.processor,raw,identity)
        causal = planner_input_projection(target)
        serving_prepared = context.processor._process_tensors(observable_raw(observed,target['task_name'],data.config))
        normalized = ('pixel_values','proprio','proprio_dim_is_pad')
        if tree_sha({k:prepared[k] for k in normalized}) != tree_sha({k:serving_prepared[k] for k in normalized}):
            raise ValueError('Serving and teacher-forced normalization differ')
        prefix = single_frame_planner_prefix(context.processor.samples_builder,serving_prepared,causal)
        encoded = tokenizer.encode_inference([prefix],device=torch.device('cpu'),mode='ar',training=False)
        training_prefix = tokenizer.encode_inference([prepared['samples']],device=torch.device('cpu'),mode='ar',training=False)
        if tree_sha(encoded) != tree_sha(training_prefix) or tree_sha(encoded) != accepted[sid]['prefix_sha256']:
            raise ValueError('TRAIN diagnostic did not use the accepted target-free prefix')
        record = dict(sample_id=sid,identity=identity,encoded_sha256=tree_sha(encoded),
                      pixels_sha256=tree_sha(prepared['pixel_values']),causal_input_sha256=digest(causal),
                      optimizer_exposures=by_index[candidate]['exposures'],
                      choice_layout_sha256=digest(choice_layout(shell,prepared)))
        cpu_rows.append(record); prepared_rows.append((position,split,index,request,target,prefix,prepared,record))
        if len(cpu_rows)%20 == 0: print(json.dumps(dict(cpu_rows=len(cpu_rows),total=len(local))),flush=True)
    common = dict(config_sha256=file_sha(args.config),source_commit=subprocess.check_output(
        ['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),checkpoint_sha256=spec['checkpoint']['sha256'],
        fit_result_sha256=spec['fit_result']['sha256'],schedule_sha256=digest(schedule),sharding=sharding,
        data_split='train',optimizer_updates=0,physical_success_measured=False,target_free=True,
        teacher_forced_diagnostics_are_not_generation_accuracy=True)
    write_new(args.output/'cpu-prefix-audit.json',dict(common,status='passed',rows=cpu_rows))
    del tokenizer; gc.collect()
    if args.mode == 'cpu':
        portable = None
        if args.export_registration:
            from recovery_handoff_portable import export_pack
            portable = export_pack(args.output/'portable', registration=read(args.export_registration),
                registration_sha256=file_sha(args.export_registration), diagnostic=cfg,
                diagnostic_sha256=file_sha(args.config), model_config=data.config,
                prepared_rows=prepared_rows,cpu_rows=cpu_rows,common=common,repo=REPO)
        write_new(args.output/'result.json',dict(common,status='completed_cpu_only',rows=len(cpu_rows),
                  model_forwards=0,seconds=time.monotonic()-started,exposure=exposure,portable=portable))
        return
    checkpoint = bound(root,spec['checkpoint']); idle()
    run_diagnostic(args,checkpoint,data.config,prepared_rows,common,started)


def run_diagnostic(args,checkpoint,model_config,prepared_rows,common,started):
    """Identical model calls for native and SHA-verified portable inputs."""
    import torch
    from g05.data.memlite_stage1_dataset import collate_stage1, to_device
    from recovery_cuda_startup import initialize_cuda_backend
    from g05.models.g05.qwen35 import vision
    from g05.utils.training.stage1_model import restore_model
    from sparse_cache import indexed_cache_freeze
    from g05.models.kv_cache import SparseKVCache
    from train_memlite_recovery import parameter_digest
    initialize_cuda_backend(0); vision._flash_attn_varlen = None; vision._flash_attn_backend = None
    def forbidden(*args,**kwargs): raise RuntimeError('Optimizer forbidden in TRAIN diagnosis')
    torch.optim.Adam.step = torch.optim.AdamW.step = forbidden
    saved = torch.load(checkpoint,map_location='cpu',mmap=True,weights_only=False)
    if saved['state']['step'] != 360: raise ValueError('Not the fixed final360 checkpoint')
    model, restoration = restore_model(model_config,'high',state=saved['model_state_dict'])
    del saved; gc.collect(); model.requires_grad_(False).eval().cuda()
    before = parameter_digest(model,frozen=True)
    model.model.ar_helper._need_full_logits = True
    common.update(restoration=restoration,model_before_sha256=before)
    write_new(args.output/'manifest.json',common); rows = []
    for position,split,index,request,target,prefix,prepared,record in prepared_rows:
        tick = time.monotonic()
        actual = model.processor.encode_inference([prefix],device=torch.device('cpu'),mode='ar',training=False)
        if tree_sha(actual) != record['encoded_sha256']: raise ValueError('Loaded model tokenizer differs')
        event = error = text = None
        try:
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16),indexed_cache_freeze(model.model.ar_helper,SparseKVCache):
                generated = model.generate_high_level([prefix],{k:v[None].cuda() for k,v in prepared['pixel_values'].items()},temperature=0.)
            event = generated['planner_events'][0]; text = generated.get('high_level_text')
        except (ValueError,RuntimeError) as exc:
            if isinstance(exc,torch.cuda.OutOfMemoryError) or not hasattr(exc,'planner_texts'): raise
            error = str(exc); text = exc.planner_texts
        # Obtain the target-free result before presenting this row's targets
        # to the separate teacher-forced diagnostic forward.
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
            loss, metrics = model(to_device(collate_stage1([prepared]),torch.device('cuda')))
            choices = choice_statistics(model,prepared,record['choice_layout_sha256'])
        scalars = {key:float(value) for key,value in metrics.items()
                   if key.startswith('ce_field_') or key in ('raw_unweighted_ce','weighted_training_ce','memory_update_ce_weight')}
        if not torch.isfinite(loss) or not all(math.isfinite(v) for v in scalars.values()):
            raise ValueError('Nonfinite teacher-forced diagnostic')
        model.model.ar_helper._last_ce_cache = None
        row = dict(global_schedule_index=position,scheduled_sample_index=index,split=split,
            sample_id=request['sample_id'],identity=request['original_identity'],input_audit=record,
            teacher_forced_fields=scalars,first_choice=choices,score=score_event(event,target),
            event=event,error=error,text=text,seconds=time.monotonic()-tick,
            expected={k:target[k] for k in ('next_decision','target_parent_goal','active_skills_semantic_json','memory_update')})
        rows.append(row)
        with (args.output/'rows.jsonl').open('a') as stream: stream.write(json.dumps(row,sort_keys=True,allow_nan=False)+'\n')
        print(json.dumps(dict(completed=len(rows),total=len(prepared_rows),score=row['score'])),flush=True)
    after = parameter_digest(model,frozen=True)
    if after != before: raise ValueError('Diagnostic mutated model weights')
    write_new(args.output/'result.json',dict(common,status='completed',rows=len(rows),
        model_after_sha256=after,summary=summarize(rows),seconds=time.monotonic()-started,
        cpu_prefix_audit_sha256=file_sha(args.output/'cpu-prefix-audit.json')))


if __name__=='__main__': main()
