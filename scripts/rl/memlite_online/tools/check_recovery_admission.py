"""Freeze preparation evidence and block unreviewed pools before any GPU use."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'code'))
from recovery_admission import audit_admission  # noqa: E402
from recovery_corpus import canonical, file_sha  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('audit', 'protected-groups', 'approvals', 'evidence-root', 'output'):
        parser.add_argument('--' + key, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Use a new immutable admission version')
    admitted, spans, receipt = audit_admission(args.audit, args.protected_groups, args.approvals, args.evidence_root)
    args.output.mkdir(parents=True)
    files = {}
    for pool, rows in admitted.items():
        target = args.output / (pool + '.jsonl')
        target.write_text(''.join(canonical(r) + '\n' for r in rows))
        files[target.name] = file_sha(target)
    (args.output / 'candidate-event-queue.json').write_text(json.dumps(spans, indent=2) + '\n')
    receipt['files'] = files
    (args.output / 'admission.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
