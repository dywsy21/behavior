"""CPU-gated frozen observer generation; neither fitting nor data admission."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO/'src'), str(REPO/'scripts'), str(REPO/'scripts/infra'),
               str(REPO/'scripts/rl/memlite_online/code')]
from recovery_calibration_launch import bound, read, write_new
from recovery_corpus import file_sha, digest, group_key
from recovery_handoff_inference import (reconstruct_request, unique_checks, observation_and_prefix,
                                       feedback_from_trace)
from recovery_process_cache import isolated_cache_env
from recovery_handoff_heldout import generation_split, SELECTION


def source_inputs(cfg):
    """All lineage/exposure checks occur before CUDA or target generation."""
    from behavior_champion.release import AcceptedRelease
    from recovery_postfit_feedback import validate_exposure
    from recovery_sft_data import require_training_pool
    from recovery_postfit import calibration_gate
    from g05.utils.training.stage1_model import configuration, make_processor
    from g05.data.memlite_stage1_dataset import Stage1Dataset
    split = generation_split(cfg)
    root = Path(cfg['root']); request_path = bound(root, cfg['requests'])
    prep = read(bound(root, cfg['preparation'])); recipe = read(REPO/cfg['original_recipe'])
    h1 = recipe['H1']; periodic = h1['periodic_teacher']
    if (prep['requests_sha256'] != cfg['requests']['sha256'] or prep['selected'] != cfg['rows']
            or prep['observer_checks'] != cfg['head_checks'] or prep['model_calls'] != 0
            or prep['optimizer_updates'] != 0 or prep['human_admitted'] is not False
            or file_sha(REPO/cfg['original_recipe']) != prep['recipe_sha256']):
        raise ValueError('Wrong CPU request lineage')
    if split == 'train':
        candidates = bound(root, dict(path=periodic['candidates'], sha256=periodic['candidate_sha256']))
    else:
        candidates = bound(root, cfg['candidates'])
        registration = bound(REPO, cfg['registration'])
        if (prep['status'] != 'CPU_HELDOUT_EVAL_REQUESTS_ONLY_TRAINING_FORBIDDEN'
                or prep['purpose'] != 'heldout_eval_only' or prep['training_admission_forbidden'] is not True
                or prep['predictions_used_in_selection'] is not False or prep['selection'] != SELECTION
                or prep['registration_sha256'] != file_sha(registration)
                or prep['candidates_sha256'] != cfg['candidates']['sha256']):
            raise ValueError('Unregistered, mixed-split or model-selected heldout candidates')
    exposure = read(bound(root, h1['exposure'])); excluded = validate_exposure(exposure)
    paths = {}
    for key, binding in exposure['bindings'].items():
        path = Path(binding['path']).resolve()
        if not path.is_relative_to(root.resolve()) or file_sha(path) != binding['sha256']:
            raise ValueError('Changed fixed-observer dependency: '+key)
        paths[key] = path
    fit, fit_cfg, calibration = (read(paths[k]) for k in ('fit_result','fit_config','calibration'))
    calibration_gate(calibration)
    if (fit['status'] != 'observer_adapter_fit_complete_not_deployed'
            or fit['selected_checkpoint_sha256'] != h1['observer_sha256']
            or exposure['observer_sha256'] != h1['observer_sha256']
            or fit['config_sha256'] != file_sha(paths['fit_config'])
            or fit['frozen_before_sha256'] != fit['frozen_after_sha256']
            or fit_cfg['high']['sha256'] != h1['observer_backbone_sha256']
            or exposure['high_sha256'] != h1['observer_backbone_sha256']
            or calibration['observer_sha256'] != h1['observer_sha256']
            or calibration['high_sha256'] != h1['observer_backbone_sha256']
            or file_sha(paths['calibration']) != h1['calibration']['sha256']
            or sorted(calibration['source_groups']) != sorted(exposure['calibration_groups'])
            or fit_cfg.get('history_protocol','cadence16_v1') != 'cadence16_v1'
            or fit_cfg.get('include_served_controls',False)):
        raise ValueError('Observer/calibration/history lineage changed')
    bound(root, fit_cfg['high'])
    eval_cfg = read(REPO/'configs/recovery_sft/a800_h1_periodic_final_transition_v1.json')
    _, recovery = require_training_pool(root/eval_cfg['union']/'admission','planner',eval_cfg['admission_sha256'])
    excluded |= {r['candidate']['source_group'] for r in recovery if r['candidate']['split'] == 'dev'}
    release = AcceptedRelease(root/recipe['expert_release'])
    for name in (split+'_candidates.npy',split+'_tasks.npy','episode_offsets.npy'):
        release.verify_file(name)
    if split == 'eval':
        train_groups = {group_key(ep['task_name'],int(ep['row']['task_instance_id']))
                        for _,ep in release.episodes('train')}
        if len(train_groups) != prep['train_groups_excluded']:
            raise ValueError('Original source exclusions changed')
        excluded |= train_groups
    config = configuration(root,'high',release.manifest['task_names'])
    if file_sha(config['stats_path']) != fit_cfg['stats_sha256']: raise ValueError('Changed normalization')
    data = Stage1Dataset(release.root,config,'high',split)
    rows = [json.loads(x) for x in request_path.read_text().splitlines()]
    originals = {r['identity']['original_candidate_index']:r
        for r in (json.loads(x) for x in candidates.read_text().splitlines())}
    if len(rows) != cfg['rows'] or len({r['sample_id'] for r in rows}) != len(rows):
        raise ValueError('Missing or duplicated decision samples')
    for row in rows:
        reconstruct_request(row, originals[row['original_identity']['original_candidate_index']], data, excluded,
                            expected_split=split)
    if split == 'eval' and (len(originals) != len(rows) or
            set(originals) != {r['original_identity']['original_candidate_index'] for r in rows}):
        raise ValueError('Missing or filtered heldout targets')
    models = dict(root=str(root), observer_backbone=fit_cfg['high'],
        observer_adapter=dict(path=str(paths['selected_observer'].relative_to(root)),sha256=h1['observer_sha256']))
    accepted = dict(fit_config=fit_cfg,fit=fit,paths={'observer_fit_config':paths['fit_config']})
    identities = SimpleNamespace(observer_backbone=h1['observer_backbone_sha256'],
        observer_adapter=h1['observer_sha256'],observer_normalization=fit_cfg['stats_sha256'])
    return SimpleNamespace(rows=rows, data=data, processor=make_processor(config,False),
        calibration=calibration, exposure=exposure, models=models, accepted=accepted,
        identities=identities, names=release.manifest['task_names'])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--mode',choices=['audit','predict'],required=True)
    ap.add_argument('--audit',type=Path)
    ap.add_argument('--audit-sha256')
    a = ap.parse_args(); cfg = read(a.config); split = generation_split(cfg)
    if (cfg['optimizer_updates'] != 0 or a.output.exists()
            or subprocess.check_output(['git','status','--porcelain'],cwd=REPO,text=True).strip()):
        raise ValueError('Explicit read-only recipe, clean frozen source and fresh output required')
    gpu = os.environ.get('CUDA_VISIBLE_DEVICES')
    if a.mode == 'audit':
        if gpu != '' or a.audit or a.audit_sha256: raise ValueError('CPU audit has no CUDA or prior certificate')
    elif (gpu != str(cfg['gpu']) or not a.audit or not a.audit_sha256
            or subprocess.check_output(['nvidia-smi','-i',gpu,'--query-compute-apps=pid',
                '--format=csv,noheader'],text=True).strip()):
        raise ValueError('Require the designated actually idle GPU and a pinned full CPU audit')
    a.output.mkdir(parents=True)
    os.environ.update(isolated_cache_env(a.output))
    started = time.monotonic()
    status = dict(status='validating',mode=a.mode,source_commit=subprocess.check_output(
        ['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),config_sha256=file_sha(a.config),
        pid=os.getpid(),hostname=socket.gethostname(),optimizer_updates=0,started=time.time(),
        requests_sha256=cfg['requests']['sha256'],human_admitted=False,physical_outcome_asserted=False,
        data_split=split,purpose='heldout_eval_only' if split=='eval' else 'train_input_generation',
        training_admission_forbidden=split=='eval')
    write_new(a.output/'status.json',status)
    import torch
    torch.set_num_threads(2); torch.manual_seed(cfg['seed'])
    data = source_inputs(cfg)
    from check_memlite_periodic_processor import make_token_processor
    cpu_tokens = make_token_processor(data.data.config) if a.mode == 'audit' else None
    observer = None; audited = {}
    if a.mode == 'predict':
        if file_sha(a.audit) != a.audit_sha256: raise ValueError('Changed CPU audit')
        audit = read(a.audit)
        if (audit['status'] != 'cpu_observable_prefix_audit_completed' or audit['optimizer_updates'] != 0
                or audit['source_commit'] != status['source_commit'] or audit['config_sha256'] != status['config_sha256']
                or audit['requests_sha256'] != cfg['requests']['sha256'] or audit['rows'] != cfg['rows']
                or file_sha(a.audit.parent/'checks.jsonl') != audit['checks_sha256']):
            raise ValueError('Full exact-source/recipe/RGB/prefix audit required')
        for row in (json.loads(x) for x in (a.audit.parent/'checks.jsonl').read_text().splitlines()):
            key = (row['sample_id'],row['check_sha256'])
            if key in audited: raise ValueError('Duplicate CPU audit check')
            audited[key] = row
        if len(audited) != audit['observations']: raise ValueError('Missing audited observations')
        from recovery_cuda_startup import initialize_cuda_backend
        from recovery_observer_pilot_runtime import LoadedPilotObserver
        from train_memlite_recovery import parameter_digest
        from g05.models.g05.qwen35 import vision
        initialize_cuda_backend(0); vision._flash_attn_varlen=None; vision._flash_attn_backend=None
        def forbidden(*args,**kwargs): raise RuntimeError('Optimizer forbidden in observer generation')
        torch.optim.Adam.step = torch.optim.AdamW.step = forbidden
        observer = LoadedPilotObserver(dict(models=data.models,calibrated=data.accepted),
            data.identities,data.names,parameter_digest=parameter_digest)
        if tree_config(observer.config) != tree_config(data.data.config): raise ValueError('Loaded observer preprocessing drift')
        cpu_tokens = observer.policy.processor
        data.processor = observer.processor
    checks_written=0; predictions=0; feedback_counts=Counter(); cross=Counter()
    with (a.output/'checks.jsonl').open('x') as check_stream, (a.output/'trace.jsonl').open('x') as trace_stream, \
            (a.output/'feedback.jsonl').open('x') as feedback_stream:
        for number,row in enumerate(data.rows):
            memo = {}; trace = []  # Strictly per decision; no cross-task/history memo.
            for check in unique_checks(row):
                prefix,pixels,receipt,_ = observation_and_prefix(data.data,data.processor,check,cpu_tokens)
                receipt = dict(receipt,sample_id=row['sample_id'])
                if observer is not None:
                    if audited.pop((row['sample_id'],receipt['check_sha256']),None) != receipt:
                        raise ValueError('GPU model token prefix/RGB differs from audited CPU source')
                    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
                        context = observer.policy.outcome_context_from_prefix([prefix],
                            {k:v[None].cuda() for k,v in pixels.items()})
                    if context.ndim != 2 or context.shape[0] != 1 or not torch.isfinite(context).all():
                        raise ValueError('Nonfinite or cross-batch observer context')
                    memo[digest(check)] = (context[0].detach(),prefix['proprio']['value'].reshape(27).cuda())
                check_stream.write(json.dumps(receipt,sort_keys=True,allow_nan=False)+'\n'); check_stream.flush()
                checks_written += 1
            if observer is not None:
                for request in row['requests']:
                    checks = request['checks']; contexts,states = zip(*(memo[digest(c)] for c in checks))
                    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
                        logits = observer.head(context=torch.stack(contexts)[None],proprio=torch.stack(states)[None],
                            steps=torch.tensor([[c['control_step'] for c in checks]],device='cuda'),
                            valid=torch.ones(1,len(checks),dtype=torch.bool,device='cuda')).float().cpu()
                    if logits.shape != (1,4) or not torch.isfinite(logits).all(): raise ValueError('Invalid temporal logits')
                    item = dict(request,logits=logits[0].tolist()); trace.append(item); predictions += 1
                    trace_stream.write(json.dumps(dict(item,decision_sample_id=row['sample_id']),sort_keys=True)+'\n')
                ep,*_ = data.data.locate(row['original_identity']['original_candidate_index'])
                feedback = feedback_from_trace(row,ep,trace,data.calibration)
                outcome = json.loads(feedback)['estimated_bundle_outcome']; feedback_counts[outcome] += 1
                cross[row['target_change_class']+' / '+outcome] += 1
                feedback_stream.write(json.dumps(dict(sample_id=row['sample_id'],source_group=row['source_group'],
                    original_identity=row['original_identity'],execution_feedback=feedback,
                    original_execution_feedback=row['original_projection']['execution_feedback'],
                    target_projection_sha256=digest(row['original_projection']),
                    known_previous_outcome='UNKNOWN',physical_outcome_asserted=False,new_human_admission=False),
                    sort_keys=True,allow_nan=False)+'\n')
                trace_stream.flush(); feedback_stream.flush()
            if (number+1)%10 == 0 or number+1 == len(data.rows):
                print(json.dumps(dict(rows=number+1,observations=checks_written,head_checks=predictions,
                    feedback=dict(feedback_counts),seconds=time.monotonic()-started)),flush=True)
    final = dict(status,status='cpu_observable_prefix_audit_completed' if observer is None else 'frozen_handoff_predictions_completed_not_admitted',
        rows=len(data.rows),observations=checks_written,head_checks=predictions,seconds=time.monotonic()-started,
        checks_sha256=file_sha(a.output/'checks.jsonl'),trace_sha256=file_sha(a.output/'trace.jsonl'),
        feedback_sha256=file_sha(a.output/'feedback.jsonl'),feedback_counts=dict(feedback_counts),
        by_target_change=dict(cross),observer_backbone_sha256=data.identities.observer_backbone,
        observer_adapter_sha256=data.identities.observer_adapter,observer_normalization_sha256=data.identities.observer_normalization,
        calibration_sha256=data.exposure['bindings']['calibration']['sha256'],calibration_refit=False,
        all_rows_retained=True,known_truth_not_added=True,
        limitation=('Original heldout annotation-command histories, not executed actor rollouts or physical outcome truth. All original TRAIN and observer adapter exposure sources excluded; never eligible for SFT.' if split=='eval' else
        'Annotation-command history over original TRAIN, not an executed actor rollout; source exclusions apply to observer adapter fit/selection/CAL/test and recovery DEV, not the pretrained backbone original TRAIN. Human review required before any SFT.'))
    if observer is not None:
        if audited or predictions != cfg['head_checks']: raise ValueError('Incomplete actual generation')
        final.update(observer.finish(),cpu_audit_sha256=a.audit_sha256)
    write_new(a.output/'result.json',final); print(json.dumps(final,indent=2),flush=True)


def tree_config(config):
    return json.dumps(config,sort_keys=True,allow_nan=False)


if __name__ == '__main__': main()
