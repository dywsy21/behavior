"""Fit a small result head to convergence on signed data; no runtime release.

Separate from immutable 5-event-pass pilot tickets. The current user goal
authorizes a proper capacity/optimization diagnostic. Selection DEV is not a
blind test or calibration set; only a new independent cohort can certify it.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import uuid

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO/'scripts/rl/memlite_online/code'))
from recovery_corpus import file_sha, digest
from recovery_sft_data import require_training_pool
from recovery_observer_training import OUTCOMES, request_key, temporal_batch, outcome_metrics
from recovery_convergence import event_weights, check_splits, selection_key, causal_suffix_training_items


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args(); cfg = json.loads(args.config.read_text()); root = Path(cfg['root'])
    if cfg.get('schema') != 'recovery_observer_convergence_diagnostic_v1' or not cfg.get('user_goal_authorized'):
        raise ValueError('New diagnostic requires explicit recipe/authority')
    if args.output.exists():
        raise FileExistsError(args.output)
    if subprocess.check_output(['git','status','--porcelain'], cwd=REPO, text=True).strip():
        raise ValueError('Clean frozen source required')
    _, rows = require_training_pool(root/cfg['admission'], 'outcome', cfg['admission_sha256'])
    receipt = json.loads((root/cfg['feature_receipt']).read_text())
    path = root/cfg['features']
    if (file_sha(path) != cfg['features_sha256'] or receipt['features_sha256'] != cfg['features_sha256']
            or receipt['admission_sha256'] != cfg['admission_sha256'] or receipt['diagnostic_only']
            or receipt['high_sha256'] != cfg['high_sha256']):
        raise ValueError('Changed frozen feature/parent/admission binding')
    gpu = os.environ.get('CUDA_VISIBLE_DEVICES')
    if gpu not in tuple(map(str, range(8))) or subprocess.check_output(
            ['nvidia-smi','-i',gpu,'--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
        raise RuntimeError('Need one verified idle A800')
    import torch
    import torch.nn.functional as F
    from g05.models.g05.helpers.temporal_outcome import TemporalOutcomeObserver
    from g05.utils.training.stage1_runtime import atomic_json, capture_rng, save_checkpoint, init_wandb
    torch.set_num_threads(2); torch.manual_seed(cfg['seed'])
    train = [r for r in rows if r['candidate']['split'] == 'train']
    dev = [r for r in rows if r['candidate']['split'] == 'dev']; check_splits(train, dev)
    cache = torch.load(path, map_location='cpu', weights_only=False)
    if (cache['schema'] != 'recovery_member_feature_cache_v1'
            or digest(cache['requests']) != receipt['requests_sha256']):
        raise ValueError('Invalid feature request provenance')
    features = cache['features']
    if any(request_key(r) not in features for r in rows):
        raise ValueError('Approved labels lack a causal feature')
    device = torch.device('cuda',0)
    observer_kwargs=cfg.get('observer_kwargs',{})
    if set(observer_kwargs)-{'include_absolute_proprio'}:raise ValueError('Unregistered observer architecture')
    head = TemporalOutcomeObserver(next(iter(features.values()))['context'].shape[-1],**observer_kwargs).to(device)
    head.head.load_state_dict(cache['initial_head'], strict=True)
    optimizer = torch.optim.AdamW(head.parameters(), lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'])
    train_items=[features[request_key(r)] for r in train]
    train_labels=[OUTCOMES.index(r['approval']['label']['value']) for r in train]
    weighting = cfg.get('outcome_weighting', 'physical_event')
    if weighting not in ('physical_event', 'mechanism_then_physical_event'):
        raise ValueError('Unregistered outcome weighting')
    mechanisms = None
    if weighting == 'mechanism_then_physical_event':
        requests_by_id = {r['request_id']:r for r in cache['requests']}
        mechanisms = []
        for row in train:
            request = requests_by_id[request_key(row)]
            # The prefix is bound to the actual old attempt for predecision
            # rows. Never infer its mechanism from a later planner target.
            bundle = json.loads(request['checks'][-1]['issued_bundle'])
            mechanisms.append(bundle[request['member_index']]['verb'])
    row_weights=event_weights(train, mechanisms)
    augmentation=cfg.get('history_augmentation','none')
    if augmentation not in ('none','all_causal_suffixes_equal_row_mass'):
        raise ValueError('Unregistered causal training augmentation')
    if augmentation=='all_causal_suffixes_equal_row_mass':
        train_items,row_weights,source_indices=causal_suffix_training_items(train_items,row_weights)
        train_labels=[train_labels[i] for i in source_indices]
    values=temporal_batch(train_items,device)
    y=torch.tensor(train_labels,device=device)
    weights = torch.tensor(row_weights, device=device)
    if any(v.requires_grad for v in values.values()):
        raise ValueError('Frozen cache must not carry backbone gradients')
    args.output.mkdir(parents=True)
    commit = subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()
    run_id = uuid.uuid4().hex[:12]
    fingerprint = digest(dict(config=file_sha(args.config), commit=commit))
    wb = init_wandb(dict(cfg['wandb'],name=args.output.name),args.output,run_id=run_id,resume=False,
        metadata=dict(component='H0-convergence', high_sha256=cfg['high_sha256'], source_commit=commit,
            config_sha256=file_sha(args.config), calibration_or_runtime_ready=False))
    def measure(step):
        metrics = {split:outcome_metrics(head, items, features, device) for split,items in [('train',train),('dev',dev)]}
        with (args.output/'evaluations.jsonl').open('a') as stream:
            stream.write(json.dumps(dict(step=step, evaluation=metrics))+'\n')
        wb.log({'train/update':step, **{f'eval/{s}/{k}':m[k] for s,m in metrics.items()
            for k in ('event_weighted_ce','event_weighted_accuracy','balanced_accuracy')}})
        atomic_json(args.output/'progress.json',dict(step=step,metrics=metrics,runtime_ready=False))
        return metrics
    started = time.monotonic(); initial = measure(0); best = selection_key(initial['dev'])
    best_step = 0; best_state = {k:v.detach().cpu().clone() for k,v in head.state_dict().items()}
    stopping = [False]
    for sig in (signal.SIGTERM,signal.SIGINT):
        signal.signal(sig, lambda *_:stopping.__setitem__(0,True))
    step = 0; last = initial; stop_reason = 'configured_convergence_window'
    while step < cfg['maximum_updates']:
        if stopping[0]:
            stop_reason = 'signal_saved'; break
        head.train(); optimizer.zero_grad(set_to_none=True)
        logits = head(**values); loss = (F.cross_entropy(logits,y,reduction='none')*weights).sum()
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite observer loss')
        loss.backward(); torch.nn.utils.clip_grad_norm_(head.parameters(),1.,error_if_nonfinite=True); optimizer.step()
        step += 1
        wb.log({'train/update':step, 'train/outcome_ce':float(loss.detach())})
        if step % cfg['evaluate_every'] == 0 or step == cfg['maximum_updates']:
            last = measure(step)
            if selection_key(last['dev']) > best:
                best = selection_key(last['dev']); best_step = step
                best_state = {k:v.detach().cpu().clone() for k,v in head.state_dict().items()}
            if step >= cfg['minimum_updates'] and step-best_step >= cfg['patience_updates']:
                stop_reason = 'selection_dev_plateau'; break
    state = dict(step=step, save_sequence=1, fingerprint=fingerprint, run_id=run_id,
                 consumed_seconds=time.monotonic()-started, best_step=best_step)
    save_checkpoint(args.output/'checkpoints',model=head,optimizer=optimizer,state=state,rng_by_rank=[capture_rng()])
    head.load_state_dict(best_state,strict=True)
    torch.save(best_state,args.output/'selected-observer.pt')
    selected = measure(step)
    head.eval()
    with torch.no_grad():
        for split,items in [('train',train),('dev',dev)]:
            probabilities = head(**temporal_batch([features[request_key(r)] for r in items],device)).softmax(-1).cpu().tolist()
            with (args.output/(split+'-predictions.jsonl')).open('x') as stream:
                for r,p in zip(items, probabilities):
                    stream.write(json.dumps(dict(sample_id=r['candidate']['sample_id'],source_group=r['candidate']['source_group'],
                        label=r['approval']['label']['value'],probabilities=dict(zip(OUTCOMES,p))))+'\n')
    atomic_json(args.output/'result.json',dict(status='fit_diagnostic_complete', stop_reason=stop_reason,
        steps=step,selected_step=best_step,initial=initial,last=last,selected=selected,seconds=time.monotonic()-started,
        high_sha256=cfg['high_sha256'],selected_observer_sha256=file_sha(args.output/'selected-observer.pt'),
        runtime_ready=False,dev_used_for_model_selection=True,requires_new_independent_calibration=True,
        observer_kwargs=observer_kwargs,
        history_augmentation=augmentation,training_views=len(train_items),reviewed_training_rows=len(train),
        outcome_weighting=weighting,
        wandb_url=wb.url,source_commit=commit,config_sha256=file_sha(args.config)))
    wb.finish()


if __name__ == '__main__':
    main()
