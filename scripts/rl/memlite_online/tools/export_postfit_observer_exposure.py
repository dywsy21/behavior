"""Export metadata-only source exclusions for fixed-observer H1 preparation."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

REPO = Path(__file__).resolve().parents[4]
sys.path[:0] = [str(REPO/'src'), str(REPO/'scripts/rl/memlite_online/code')]
from recovery_calibration_launch import bound, read, write_new
from recovery_corpus import file_sha
from recovery_sft_data import require_training_pool
from recovery_postfit import exposure_manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--declaration', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists() or subprocess.check_output(['git', 'status', '--porcelain'], cwd=REPO, text=True).strip():
        raise ValueError('Clean frozen source and new output required')
    declaration = read(a.declaration); root = Path(declaration['root'])
    fit = read(bound(root, declaration['fit_result']))
    cfg = read(bound(root, declaration['fit_config']))
    bound(root, declaration['selected_observer'])
    _, rows = require_training_pool(root/cfg['admission'], 'outcome', cfg['admission_sha256'])
    run = root/declaration['run_output']
    prepared = read(run/'result.json')
    config = read(bound(root, dict(path=str((run/'calibration-config.json').relative_to(root)),
        sha256=prepared['config_sha256'])))
    selection = read(bound(root, dict(path=config['anchor_selection'], sha256=config['anchor_selection_sha256'])))
    result = read(run/'calibration/result.json')
    calibration = read(bound(root, dict(path=str((run/'calibration/calibration.json').relative_to(root)),
        sha256=result['calibration_sha256'])))
    if (fit['status'] != 'observer_adapter_fit_complete_not_deployed'
            or result['status'] != 'completed_prospective_adapter_calibration_not_deployed'
            or result['selected_observer_sha256'] != fit['selected_checkpoint_sha256']
            or result['selected_observer_sha256'] != declaration['selected_observer']['sha256']
            or fit['config_sha256'] != declaration['fit_config']['sha256']
            or result['config_sha256'] != prepared['config_sha256']
            or result['high_sha256'] != cfg['high']['sha256']
            or result['anchor_selection_sha256'] != config['anchor_selection_sha256']
            or result['frozen_before_sha256'] != result['frozen_after_sha256']
            or result['optimizer_updates'] != 0 or not result['frozen_test_not_read']
            or result['calibration'] != calibration):
        raise ValueError('Completed fit/calibration binding changed')
    historical = REPO/'configs/recovery_sft/a800_independent_observer_calibration_v1.json'
    union = bound(REPO, declaration['union_spec'])
    paths = dict(declaration=a.declaration.resolve(), fit_result=bound(root, declaration['fit_result']),
        fit_config=bound(root, declaration['fit_config']), fit_admission=root/cfg['admission']/'admission.json',
        fit_outcome_rows=root/cfg['admission']/'outcome.jsonl', selected_observer=bound(root, declaration['selected_observer']),
        selection=root/config['anchor_selection'], calibration_result=run/'calibration/result.json',
        calibration=run/'calibration/calibration.json', historical_reserved=historical, union_spec=union)
    bindings = {k:dict(path=str(v), sha256=file_sha(v)) for k, v in paths.items()}
    output = exposure_manifest(rows, selection, calibration, read(historical)['expected_groups'],
        read(union)['evaluation_partition']['frozen_test_groups'], bindings)
    output['source_commit'] = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip()
    write_new(a.output, output)
    print(json.dumps(dict(status='postfit_exposure_pinned_no_predictions', output=str(a.output),
        sha256=file_sha(a.output), excluded_groups=len(output['excluded_groups']),
        observer_train_groups=len(output['observer_train_groups']), observer_selection_groups=len(output['observer_selection_groups']))))


if __name__ == '__main__':
    main()
