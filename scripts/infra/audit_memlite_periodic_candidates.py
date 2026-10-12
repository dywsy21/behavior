"""Audit existing candidate classification without replacing labels or split."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'scripts/rl/memlite_online/code'))
from recovery_periodic_teacher import command_change_class
from behavior_champion.release import sha256
from g05.utils.memlite_skill_protocol import parse_active_skills_text, parse_active_skills_semantic_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidates', type=Path, required=True)
    parser.add_argument('--candidate-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    if sha256(args.candidates) != args.candidate_sha256: raise ValueError('Changed candidate input')
    rows = [json.loads(line) for line in args.candidates.read_text().splitlines()]
    seen, counts, changes = set(), Counter(), []
    for row in rows:
        identity, projection = row['identity'], row['projection']
        index = identity['original_candidate_index']
        if identity['split'] != 'train' or index in seen or row['human_admission'] is not False:
            raise ValueError('Candidate split/identity/admission drift')
        seen.add(index)
        if projection['outcome_supervision_mask'] or projection['low_action_supervision_mask']:
            raise ValueError('Periodic augmentation cannot supervise outcomes or fixed-skill actions')
        if parse_active_skills_text(projection['active_skills_text']) != parse_active_skills_semantic_json(projection['active_skills_semantic_json']):
            raise ValueError('Text target and semantic target mismatch')
        kind = command_change_class(projection['previous_intent'], projection['active_skills_text'])
        counts[kind] += 1
        changes.append(dict(identity=identity, label_change_class=kind,
                            physical_progress_verified=False, training_admitted=False))
    result = dict(status='label_change_audit_only_not_training_admission', candidate_sha256=args.candidate_sha256,
        candidates=len(rows), tasks=len({r['identity']['task'] for r in rows}), counts=dict(counts), rows=changes,
        original_labels_modified=False, training_admitted=False,
        limitation='Explicit command changes are not physical success, outcome or executed actor history.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream: json.dump(result, stream, indent=2); stream.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k != 'rows'}), flush=True)


if __name__ == '__main__': main()
