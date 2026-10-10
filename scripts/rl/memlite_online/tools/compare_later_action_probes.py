"""Bind two machine-audited physical arms; no training, selection or promotion."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'code'))
from recovery_corpus import file_sha
from recovery_probe_pair import compare_probe_pair


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('configs', 'services', 'audits'):
        p.add_argument('--'+name, nargs=2, type=Path, required=True,
                       help='Explicit order: control, later-actions')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    groups = [a.configs, a.services, a.audits]
    result = compare_probe_pair(*[[json.loads(path.read_text()) for path in group] for group in groups],
                                [file_sha(path) for path in a.configs])
    result['inputs'] = {name: [dict(path=str(path.resolve()), sha256=file_sha(path)) for path in group]
                       for name, group in zip(('configs', 'services', 'audits'), groups)}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
