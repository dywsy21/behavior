"""Audit reviewed GRASP labels against their actual issued arm semantics.

Read-only with respect to corpora/approvals; writes a separate evidence report.
This does not select labels from model predictions or grant new approval.
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'code'))
from recovery_admission import local_file
from recovery_corpus import file_sha
from recovery_teacher_corpus import validate_branch, validate_grasp_review_semantics


def audit(root, decisions_paths):
    root = Path(root).resolve()
    records, conflicts, files = [], [], []
    cache = {}
    for path in decisions_paths:
        path = Path(path)
        choices = json.loads(path.read_text())
        if choices['schema'] != 'owner_local_recovery_review_v1':
            raise ValueError('Require exact owner decisions, not inferred candidate approval')
        if choices.get('role') == 'calibration_only_never_training_or_selection':
            raise ValueError('This training-data audit does not read independent calibration/test cohorts')
        files.append(dict(path=str(path), sha256=file_sha(path)))
        for choice in choices['branches']:
            material_path = local_file(root, Path(choice['review_directory']) / 'review.json')
            material = json.loads(material_path.read_text())
            branch = local_file(root, Path(material['source']) / 'manifest.json').parent
            if branch not in cache:
                manifest = json.loads((branch / 'manifest.json').read_text())
                rows = [json.loads(s) for s in (branch / 'transitions.jsonl').read_text().splitlines()]
                plans = json.loads((branch / 'plans.json').read_text())
                if (file_sha(branch / 'transitions.jsonl') != manifest['transitions_sha256']
                        or file_sha(branch / 'plans.json') != manifest['plans_sha256']):
                    raise ValueError('Changed actual control/plan evidence')
                validate_branch(rows, manifest, plans)
                cache[branch] = (manifest, rows, plans)
            manifest, rows, plans = cache[branch]
            if (file_sha(branch / 'manifest.json') != choice['manifest_sha256']
                    or material['manifest_sha256'] != choice['manifest_sha256']
                    or material['transitions_sha256'] != manifest['transitions_sha256']):
                raise ValueError('Review and measured branch differ')
            for sheet in material['sheets']:
                if file_sha(material_path.parent / sheet['path']) != sheet['sha256']:
                    raise ValueError('Reviewed original media changed')
            tail = 0
            for row in reversed(rows):
                if row['physical_audit']['grasp'][manifest['arm']] != 'TRUE':
                    break
                tail += 1
            checks = [(o['control_step'], 'outcome', o['value']) for o in choice['outcomes']]
            checks += [(t, 'planner', None) for t in choice['planner_steps']]
            for t, pool, label in checks:
                if t == len(rows):
                    # Terminal-observation approvals have a separate verifier.
                    records.append(dict(case=choice['case'], branch=choice['branch'], pool=pool,
                                        control_step=t, status='separate_terminal_verifier'))
                    continue
                plan = next(p for p in reversed(plans) if p['control_step'] <= t)
                record = dict(case=choice['case'], branch=choice['branch'], pool=pool,
                              control_step=t, label=label, issued_skills=json.loads(plan['active_skills_semantic_json']),
                              current_grasp=rows[t]['physical_before']['grasp'],
                              servo_arm=manifest['arm'], stable_tail_controls=tail,
                              decisions=str(path), source_manifest_sha256=choice['manifest_sha256'])
                try:
                    validate_grasp_review_semantics(rows, t, manifest['arm'], plan,
                                                    outcome=label, planner_retry=pool == 'planner')
                    record['status'] = 'passed'
                except ValueError as error:
                    record.update(status='semantic_conflict', error=str(error))
                    conflicts.append(record)
                records.append(record)
    return dict(schema='reviewed_grasp_semantic_audit_v1', status='passed' if not conflicts else 'conflicts_found',
                decisions=files, branches=len(cache), checked=len(records), conflicts=conflicts,
                records=records, new_approvals=0, optimizer_steps=0)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--decisions', type=Path, nargs='+', required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError('Use a new audit receipt; preserve prior evidence')
    result = audit(args.root, args.decisions)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ('records', 'decisions')}, indent=2))
    if result['conflicts']:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
