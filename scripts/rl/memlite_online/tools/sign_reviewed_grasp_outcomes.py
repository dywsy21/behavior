"""Bind explicit owner review decisions to exact causal evidence, never infer approval.

This deliberately narrow signer only handles manually reviewed GRASP success.
It cannot release BC/planner targets, fabricate missing result classes, or mark
an admission pool ready. The separate admission audit still applies all gates.
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'code'))
from recovery_admission import local_file
from recovery_corpus import file_sha


def verify_causal_grasp(evidence, item, review):
    if (review['decision'] != 'approve_outcome_only' or review['label'] != 'SUCCEEDED'
            or not review['notes'].strip() or type(review['member_index']) is not int
            or not 0 <= review['member_index'] < len(item['members'])):
        raise ValueError('Requires an explicit narrow owner decision')
    member = item['members'][review['member_index']]
    if (member['verb'] != 'GRASP' or member['target'] != review['target']
            or review['arm'] not in member['evidence_arms']
            or member['value'] != 'SUCCEEDED' or evidence['members'] != item['members']):
        raise ValueError('Owner target/arm differs from reviewed member')
    if (evidence['sample_id'] != item['sample_id'] or evidence['label_anchor'] != item['step']
            or evidence['source_group'] != item['source_group']
            or evidence['independent_event_conservative'] != item['event_id']
            or evidence['future_physics_used'] is not False):
        raise ValueError('Different sample, grouping or future evidence')
    rows = evidence['causal_physics']
    if [r['control_step'] for r in rows] != list(range(item['step'] - 6, item['step'])):
        raise ValueError('Six consecutive causal applied controls required')
    if not all(r['physical_audit']['grasp_states'].get(member['entity'], {}).get(review['arm']) == 'TRUE'
               for r in rows):
        raise ValueError('Exact target/arm grasp absent in causal evidence')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('index', 'review', 'evidence-root', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError('Preserve old approvals; choose a new version')
    review = json.loads(a.review.read_text())
    if (review['schema'] != 'recovery_exact_grasp_review_v1'
            or file_sha(a.index) != review['index_sha256'] or not review['reviewer'].strip()
            or review['action_samples_approved'] or review['planner_samples_approved']):
        raise ValueError('Unbound or overbroad owner review')
    index = json.loads(a.index.read_text())
    lookup = {i['ordinal']: i for i in index['items']}
    approvals, seen = [], set()
    for decision in review['items']:
        ordinal = decision['ordinal']
        if ordinal in seen:
            raise ValueError('Repeated review decision')
        seen.add(ordinal)
        item = lookup[ordinal]
        evidence_refs = []
        for key, kind in (('image', 'original_media_review'), ('physics', 'physical_semantic_review')):
            ref = item[key]
            path = local_file(a.index.parent, ref['path'])
            if file_sha(path) != ref['sha256']:
                raise ValueError('Changed review material')
            evidence_refs.append(dict(path=str(path.relative_to(a.evidence_root.resolve())),
                                      sha256=ref['sha256'], kind=kind))
        evidence = json.loads(local_file(a.index.parent, item['physics']['path']).read_text())
        verify_causal_grasp(evidence, item, decision)
        approvals.append(dict(sample_id=item['sample_id'], pool='outcome', reviewer=review['reviewer'],
            event_id=item['event_id'], source_group=item['source_group'],
            reviewed_start=item['reviewed_start'], reviewed_end=item['reviewed_end'], evidence=evidence_refs,
            label=dict(value='SUCCEEDED', member_index=decision['member_index'],
                       available_control_step=item['step'], evidence_end_control_step=item['step'])))
    result = dict(schema='recovery_sample_approvals_v1', inventory_sha256=index['inventory_sha256'],
                  anchors_sha256=index['anchors_sha256'], owner_review_sha256=file_sha(a.review),
                  approvals=approvals)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(output=str(a.output), sha256=file_sha(a.output), outcome_samples=len(approvals),
        independent_events=len({a['event_id'] for a in approvals}), action_samples=0, planner_samples=0,
        training_readiness='must run separate admission audit; no automatic release')))


if __name__ == '__main__':
    main()
