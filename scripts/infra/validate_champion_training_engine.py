"""Cold full-state replay + real heldout/EMA export, not formal training.

Consumes the original deterministic three-update engineering reference. The
single replayed update is NOT a new deployable model. No metrics/byte gates are
relaxed to accommodate an inconsistent checkpoint or numerical environment.
"""
import argparse
import json
import os
from pathlib import Path
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--source-run', type=Path, required=True)
    parser.add_argument('--reference-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    # Reuse the exact entry guard, not a weaker validation-only environment.
    import importlib.util
    repo = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location('champion_entry', repo / 'scripts/train/train_champion2026.py')
    entry = importlib.util.module_from_spec(spec); spec.loader.exec_module(entry)
    entry.check_environment()
    from behavior_champion.checkpoint import tree_fingerprints
    from behavior_champion.release import sha256
    from behavior_champion.runner import write_json
    from behavior_champion.training_engine import TrainingEngine
    reference = args.source_run / 'uninterrupted-third-update.json'
    if sha256(reference) != args.reference_sha256:
        raise ValueError('Uninterrupted reference hash differs')
    source_result = json.loads((args.source_run / 'resume-result.json').read_text())
    if source_result['status'] != 'passed':
        raise ValueError('Only the same-process-passed deterministic engineering checkpoint is eligible')
    receipt = json.loads((args.source_run / 'checkpoint-receipt.json').read_text())
    contract = json.loads((Path(receipt['path']) / 'manifest.json').read_text())['contract']
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    engine = TrainingEngine(args.root, repo, args.output,
        json.loads((repo / 'configs/champion2026/shared100_v1.json').read_text()), engineering=True)
    for key in ('model', 'optimizer', 'lr', 'ema_decay', 'flow_samples', 'sampler_plan_sha256',
                'environment', 'jax_version', 'numpy_version', 'parent_receipt_sha256', 'release_sha256', 'layout_sha256'):
        if engine.contract[key] != contract[key]:
            raise ValueError('Cold engine/reference contract differs: ' + key)
    if contract['purpose'] != 'ENGINEERING_ONLY_NOT_FOR_DEPLOYMENT':
        raise ValueError('Cannot run this validation on a formal checkpoint')
    engine.initialize(receipt, engineering_source_contract=contract)
    print(json.dumps(dict(cold_checkpoint_restored=True, completed_updates=engine.completed)), flush=True)
    # Export the exactly restored step2 before any mutation; no second 54GB
    # full-state checkpoint is needed just to check the inference format.
    from behavior_champion.export import export_ema
    exported = export_ema(args.output / 'exports', engine.state, receipt=receipt, assets=engine.assets,
                          layout=engine.layout, contract=contract)
    write_json(args.output / 'export-result.json', exported)
    print(json.dumps(dict(ema_export_verified=exported)), flush=True)
    metrics = engine.step(2)
    after = tree_fingerprints(dict(state=engine.state, train_rng=engine.train_key))
    write_json(args.output / 'cold-third-update.json', after)
    original = [json.loads(line) for line in (args.source_run / 'steps.jsonl').read_text().splitlines()][-1]
    arrays_exact = after == json.loads(reference.read_text())
    metrics_exact = all(metrics[key] == original[key] for key in ('loss', 'grad_norm'))
    write_json(args.output / 'cold-replay.json', dict(all_302_dynamic_arrays_exact=arrays_exact,
        loss_and_grad_norm_exact=metrics_exact, optimizer_updates_this_process=1,
        original_reference_sha256=args.reference_sha256, step_metrics=metrics))
    if not arrays_exact or not metrics_exact:
        raise ValueError('New-engine cold continuation differs; do not launch formal training')
    print(json.dumps(dict(cold_next_update_exact=True)), flush=True)
    heldout = engine.evaluate()
    result = dict(status='cold_resume_real_eval_and_ema_export_verified', formal_optimizer_updates=0,
        engineering_replay_updates=1, physical_controls=0, source_commit=engine.integration_commit,
        seconds=time.monotonic()-started, eval_observations=1600, metrics=heldout, export=exported,
        effect_claim='None: three engineering updates and supervised loss are not Q/SR evidence')
    write_json(args.output / 'result.json', result)
    print(json.dumps({key: value for key, value in result.items() if key != 'metrics'}), flush=True)


if __name__ == '__main__': main()
