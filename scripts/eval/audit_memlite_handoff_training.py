"""Read-only accounting of the actual, completed handoff TRAIN schedule."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO/'scripts/rl/memlite_online/code')]
from recovery_calibration_launch import bound, read, write_new
from recovery_corpus import file_sha
from recovery_handoff_exposure import training_exposure


def load_audit(config, local_cache=None):
    if (config['schema'] != 'handoff_train_diagnostic_v1' or config['data_split'] != 'train'
            or config['optimizer_updates'] != 0):
        raise ValueError('Require a read-only original TRAIN diagnosis')
    data = {}
    for key, ref in config['inputs'].items():
        if local_cache is None:
            path = bound(Path(config['root']), ref)
        else:
            # Cache aliases carry no trust; original content SHA remains mandatory.
            path = bound(local_cache, dict(path=key+Path(ref['path']).suffix, sha256=ref['sha256']))
        data[key] = ([json.loads(s) for s in path.read_text().splitlines()]
                     if path.suffix == '.jsonl' else read(path))
    summary, rows = training_exposure(**data)
    return data, summary, rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', type=Path, required=True)
    ap.add_argument('--local-cache', type=Path)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    if args.output.exists(): raise ValueError('Refusing to replace an audit')
    data, summary, rows = load_audit(read(args.config), args.local_cache)
    args.output.mkdir(parents=True)
    result = dict(schema='actual_handoff_train_exposure_v1', status='completed_cpu_audit',
                  config_sha256=file_sha(args.config), source_commit=subprocess.check_output(
                      ['git','rev-parse','HEAD'], cwd=REPO,text=True).strip(),
                  source_dirty=bool(subprocess.check_output(['git','status','--porcelain'],cwd=REPO,text=True).strip()),
                  inputs=read(args.config)['inputs'], summary=summary, rows=rows,
                  task_success_claim=False, training_effect_claim=False)
    write_new(args.output/'result.json', result)
    print(json.dumps(summary,sort_keys=True))


if __name__=='__main__': main()
