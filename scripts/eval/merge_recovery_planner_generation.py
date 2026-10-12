"""Merge fully finished, identically bound planner shards, never partial scores."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO/'scripts/rl/memlite_online/code'))
from recovery_corpus import file_sha
from recovery_generation_metrics import summarize
from recovery_generation_shards import merge_generation_shards


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--shards', type=Path, nargs='+', required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    parts = []
    evidence = []
    for path in args.shards:
        manifest = json.loads((path/'manifest.json').read_text())
        result = json.loads((path/'result.json').read_text())
        rows = [json.loads(line) for line in (path/'rows.jsonl').read_text().splitlines()]
        parts.append((manifest, result, rows))
        evidence.append(dict(path=str(path), **{name+'_sha256': file_sha(path/name)
            for name in ('manifest.json', 'result.json', 'rows.jsonl')}))
    receipt, rows, seconds = merge_generation_shards(parts)
    result = dict(receipt, status='completed', summary=summarize(rows), rows=len(rows),
                  shard_seconds=seconds, wall_clock_not_measured_by_merge=True,
                  probe_kind_counts=dict(Counter(row['probe_kind'] for row in rows)),
                  shard_evidence=evidence)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/'rows.jsonl').write_text(''.join(json.dumps(row, sort_keys=True)+'\n' for row in rows))
    (args.output/'manifest.json').write_text(json.dumps(receipt, indent=2)+'\n')
    (args.output/'result.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, sort_keys=True))


if __name__ == '__main__':
    main()
