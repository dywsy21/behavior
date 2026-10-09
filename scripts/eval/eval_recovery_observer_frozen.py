"""Once-off frozen observer diagnostic on predeclared independent groups.

No optimizer, checkpoint selection, threshold fitting, or runtime release.
The selected head and its backbone must both match immutable provenance.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'scripts/rl/memlite_online/code'))
from recovery_corpus import file_sha, digest
from recovery_observer_training import OUTCOMES, outcome_metrics, request_key, temporal_batch
from recovery_sft_data import require_training_pool


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    cfg = json.loads(a.config.read_text()); root = Path(cfg['root'])
    if cfg.get('schema') != 'recovery_frozen_observer_evaluation_v1':
        raise ValueError('Explicit frozen evaluation recipe required')
    if a.output.exists():
        raise FileExistsError(a.output)
    if subprocess.check_output(['git', 'status', '--porcelain'], cwd=REPO, text=True).strip():
        raise ValueError('Clean frozen source required')
    fit_result_path = root / cfg['fit_result']
    head_path = root / cfg['selected_observer']
    if (file_sha(fit_result_path) != cfg['fit_result_sha256']
            or file_sha(head_path) != cfg['selected_observer_sha256']):
        raise ValueError('Changed selected fit or checkpoint')
    fit = json.loads(fit_result_path.read_text())
    if (fit['high_sha256'] != cfg['high_sha256']
            or fit['selected_observer_sha256'] != cfg['selected_observer_sha256']):
        raise ValueError('Observer/backbone identity mismatch')
    fit_config_path = root / cfg['fit_config']
    if file_sha(fit_config_path) != fit['config_sha256']:
        raise ValueError('Original fit data specification changed')
    fit_cfg = json.loads(fit_config_path.read_text())
    _, old = require_training_pool(root / fit_cfg['admission'], 'outcome',
                                   fit_cfg['admission_sha256'], purpose='feature_extraction')
    receipt, rows = require_training_pool(root / cfg['admission'], 'outcome',
                                          cfg['admission_sha256'], purpose='frozen_evaluation')
    groups = {r['candidate']['source_group'] for r in rows}
    exposed = {r['candidate']['source_group'] for r in old}
    if not groups or groups & exposed or groups != set(cfg['expected_groups']):
        raise ValueError('Frozen diagnostic groups changed or overlap model selection/training')
    cache_receipt = json.loads((root / cfg['feature_receipt']).read_text())
    if (cache_receipt['admission_sha256'] != cfg['admission_sha256']
            or cache_receipt['features_sha256'] != file_sha(root / cfg['features'])
            or cache_receipt['features_sha256'] != cfg['features_sha256']
            or cache_receipt['high_sha256'] != cfg['high_sha256']
            or cache_receipt['diagnostic_only']):
        raise ValueError('Feature provenance mismatch')
    import torch
    from g05.models.g05.helpers.temporal_outcome import TemporalOutcomeObserver
    torch.set_num_threads(2)
    cache = torch.load(root / cfg['features'], map_location='cpu', weights_only=False)
    if cache['schema'] != 'recovery_member_feature_cache_v1' or digest(cache['requests']) != cache_receipt['requests_sha256']:
        raise ValueError('Feature request provenance mismatch')
    features = cache['features']; device = torch.device('cpu')
    head = TemporalOutcomeObserver(next(iter(features.values()))['context'].shape[-1])
    head.load_state_dict(torch.load(head_path, map_location='cpu', weights_only=False), strict=True)
    head.requires_grad_(False).eval()
    metrics = outcome_metrics(head, rows, features, device)
    with torch.inference_mode():
        probs = head(**temporal_batch([features[request_key(r)] for r in rows], device)).softmax(-1).tolist()
    predictions = [dict(sample_id=r['candidate']['sample_id'], source_group=r['candidate']['source_group'],
                        control_step=r['candidate']['control_step'], label=r['approval']['label']['value'],
                        prediction=OUTCOMES[max(range(4), key=lambda i:prob[i])], probabilities=dict(zip(OUTCOMES, prob)))
                   for r, prob in zip(rows, probs)]
    confusion = {name: {other: 0 for other in OUTCOMES} for name in OUTCOMES}
    for row in predictions:
        confusion[row['label']][row['prediction']] += 1
    if file_sha(head_path) != cfg['selected_observer_sha256']:
        raise ValueError('Frozen head unexpectedly changed')
    a.output.mkdir(parents=True)
    (a.output / 'predictions.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in predictions))
    result = dict(status='completed_frozen_cross_task_diagnostic', metrics=metrics, confusion=confusion,
        groups=sorted(groups), optimizer_steps=0, calibration_fitted=False, runtime_ready=False,
        no_physical_policy_success_measurement=True, selected_observer_sha256=cfg['selected_observer_sha256'],
        high_sha256=cfg['high_sha256'], feature_sha256=cache_receipt['features_sha256'],
        source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip(),
        config_sha256=file_sha(a.config), admission_sha256=cfg['admission_sha256'],
        partition_sha256=receipt['files'][receipt['evaluation_partition_file']])
    (a.output / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
