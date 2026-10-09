"""Separate candidate QA from per-sample, per-objective training admission.

No torch or GPU dependency. A contact sheet cannot sign a training release.
The reviewer supplies real evidence references; this code validates their
binding, not their semantic truth. Missing evidence keeps each pool blocked.
"""
from collections import Counter, defaultdict
import json
from pathlib import Path

from recovery_corpus import canonical, digest, file_sha, group_key

POOLS = ("outcome", "planner", "action")
KNOWN_RESULTS = {"IN_PROGRESS", "SUCCEEDED", "FAILED", "UNKNOWN"}


def local_file(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if path == root or root not in path.parents or not path.is_file():
        raise ValueError("Evidence reference escapes or is not a real file")
    return path


def independent_candidates(anchors):
    """Consecutive anchor runs, NOT an assertion of independent real recovery.

    Gaps larger than the 16-control image stride break an evidence span. Several
    spans may still be the same physical event, so this remains an upper bound.
    """
    groups = defaultdict(list)
    for row in anchors:
        if row['label_audit']['outcome']['value'] is None or 'binding_quarantine' in row['label_audit']:
            continue
        groups[canonical([row['source_episode'], row['actor_input']['issued_skills_semantic_json'],
                          row['actor_input']['parent_goal'], row['label_audit']['outcome']['value']])].append(row)
    spans = []
    for key, rows in sorted(groups.items()):
        current = []
        for row in sorted(rows, key=lambda r: r['control_step']):
            if current and not 0 < row['control_step'] - current[-1]['control_step'] <= 16:
                spans.append(current)
                current = []
            current.append(row)
        if current:
            spans.append(current)
    return [dict(event_candidate_id=digest([r[0]['source_episode'], r[0]['control_step'],
                         r[0]['actor_input']['issued_skills_semantic_json']]),
                 source_episode=r[0]['source_episode'], source_group=r[0]['source_group'],
                 task=r[0]['task'], split=r[0]['split'], start=r[0]['control_step'], end=r[-1]['control_step'],
                 anchors=len(r), proposed_outcome=r[0]['label_audit']['outcome']['value'],
                 sample_ids=[a['sample_id'] for a in r], verified_recovery=False)
            for r in spans]


def verify_approval(approval, row, evidence_root):
    """Explicit per-objective approval; no whole-clip / whole-episode blanket."""
    required = {'sample_id', 'pool', 'reviewer', 'evidence', 'event_id', 'reviewed_start',
                'reviewed_end', 'source_group', 'label'}
    usage = {'usage_role', 'cohort_sha256'}
    if set(approval) not in (required, required | usage) or approval['pool'] not in POOLS:
        raise ValueError('Invalid per-objective approval schema')
    if 'usage_role' in approval:
        sha = approval['cohort_sha256']
        if (approval['usage_role'] not in ('calibration', 'frozen_test')
                or row['split'] != 'dev' or approval['pool'] != 'outcome'
                or not isinstance(sha, str) or len(sha) != 64
                or any(c not in '0123456789abcdef' for c in sha)):
            raise ValueError('Independent outcome approvals cannot become TRAIN or BC targets')
    if (row['split'] not in {'train', 'dev'} or row['label_status'].startswith('quarantined')
            or 'binding_quarantine' in row['label_audit']):
        raise ValueError('Protected or quarantined row cannot be approved')
    if approval['source_group'] != row['source_group']:
        raise ValueError('Reviewer changed source instance grouping')
    if not all(isinstance(approval[k], str) and approval[k].strip() for k in ('reviewer', 'event_id')):
        raise ValueError('Missing reviewer or independent event identity')
    step = row['control_step']
    if not (type(approval['reviewed_start']) is int and type(approval['reviewed_end']) is int
            and 0 <= approval['reviewed_start'] <= step <= approval['reviewed_end']):
        raise ValueError('Media review does not cover this observation')
    evidence = approval['evidence']
    if not isinstance(evidence, list) or not evidence:
        raise ValueError('No exact media/physical review evidence')
    kinds = set()
    for ref in evidence:
        if set(ref) != {'path', 'sha256', 'kind'}:
            raise ValueError('Unpinned review evidence')
        if file_sha(local_file(evidence_root, ref['path'])) != ref['sha256']:
            raise ValueError('Review evidence hash mismatch')
        kinds.add(ref['kind'])
    if not {'original_media_review', 'physical_semantic_review'} <= kinds:
        raise ValueError('Require visual AND physical/semantic review, not self-report')
    label = approval['label']
    if approval['pool'] == 'outcome':
        fields = {'value', 'member_index', 'available_control_step', 'evidence_end_control_step'}
        if set(label) not in (fields, fields | {'history_role'}):
            raise ValueError('Unrecognized outcome target')
        role = label.get('history_role','observable')
        if role not in ('observable','predecision'):
            raise ValueError('Unknown result attempt context')
        if role == 'predecision':
            prior = row['label_audit'].get('predecision_outcome')
            if (not prior or prior['value'] != label['value'] or not prior['context_id']
                    or prior['evidence_end_control_step'] != step):
                raise ValueError('No bound old-attempt result at this decision observation')
        members = json.loads(row['actor_input']['issued_skills_semantic_json'])
        if (label['value'] not in KNOWN_RESULTS or type(label['member_index']) is not int
                or not 0 <= label['member_index'] < len(members)
                or any(type(label[k]) is not int or not 0 <= label[k] <= step
                       for k in ('available_control_step', 'evidence_end_control_step'))):
            raise ValueError('Outcome uses future evidence or an invalid skill member')
    elif approval['pool'] == 'action':
        if label != {'quality': 'verified_correct_execution', 'executed_controls': 32}:
            raise ValueError('Only verified real actions may receive positive FM supervision')
        if (not row['label_audit']['full_executed_32_step_target_available']
                or approval['reviewed_end'] < step + 32):
            raise ValueError('BC review must include entire real 32-control target and endpoint')
    else:
        if set(label) != {'verified_plan_path', 'verified_plan_sha256', 'kind'} or label['kind'] not in {
                'verified_correct_handoff', 'verified_recovery_continuation'}:
            raise ValueError('A failed issued plan is not a positive planner target')
        if file_sha(local_file(evidence_root, label['verified_plan_path'])) != label['verified_plan_sha256']:
            raise ValueError('Unbound planner target; no invented recovery')
    return dict(candidate=row, approval=approval)


def audit_admission(audit, protected_path, approvals_path, evidence_root):
    audit = Path(audit)
    summary = json.loads((audit / 'summary.json').read_text())
    inventory = json.loads((audit / 'inventory.json').read_text())
    anchors = [json.loads(line) for line in (audit / 'anchors.jsonl').read_text().splitlines()]
    protected = json.loads(Path(protected_path).read_text())
    approvals = json.loads(Path(approvals_path).read_text())
    if (not summary['source_manifest_supplied'] or not summary['protected_groups_supplied']
            or not summary['bindings_sha256'] or summary['conflicting_evidence'] or summary['quarantined']
            or digest(inventory) != summary['inventory_sha256']
            or file_sha(protected_path) != summary['protected_groups_sha256']):
        raise ValueError('Unverified source, split, bindings or structurally conflicted evidence')
    if (approvals.get('schema') != 'recovery_sample_approvals_v1'
            or approvals.get('inventory_sha256') != summary['inventory_sha256']
            or approvals.get('anchors_sha256') != file_sha(audit / 'anchors.jsonl')):
        raise ValueError('Approvals do not bind this exact corpus')
    lookup = {row['sample_id']: row for row in anchors}
    if len(lookup) != len(anchors):
        raise ValueError('Duplicate candidate sample identity')
    protected_groups = set(protected['groups'])
    for row in anchors:
        if row['source_group'] != group_key(row['task'], row['instance_id']):
            raise ValueError('Changed source group')
        if row['source_group'] in protected_groups and row['split'] != 'protected':
            raise ValueError('Original held-out instance leaked into candidates')
    admitted, seen, events = {pool: [] for pool in POOLS}, set(), {}
    for approval in approvals['approvals']:
        key = (approval['sample_id'], approval['pool'])
        # Multiple members at one time are separate H0 samples, but cannot be
        # counted as multiple independent events by changing member index.
        if approval['pool'] == 'outcome':
            key += (approval['label'].get('member_index'),)
        if key in seen:
            raise ValueError('Duplicate objective/sample approval')
        seen.add(key)
        row = lookup[approval['sample_id']]
        if row['source_group'] in protected_groups:
            raise ValueError('Protected sample approval')
        item = verify_approval(approval, row, evidence_root)
        event = approval['event_id']
        if event in events and events[event] != (row['source_group'], row['split']):
            raise ValueError('Event identity crosses source groups or splits')
        events[event] = (row['source_group'], row['split'])
        admitted[approval['pool']].append(item)
    gates = {}
    for pool, rows in admitted.items():
        coverage = {split: dict(samples=sum(r['candidate']['split'] == split for r in rows),
            independent_events=len({r['approval']['event_id'] for r in rows if r['candidate']['split'] == split}),
            source_groups=len({r['candidate']['source_group'] for r in rows if r['candidate']['split'] == split}),
            tasks=len({r['candidate']['task'] for r in rows if r['candidate']['split'] == split})) for split in ('train', 'dev')}
        missing = []
        for split, minimum_events in (('train', 8), ('dev', 4)):
            if coverage[split]['independent_events'] < minimum_events or coverage[split]['source_groups'] < 2:
                missing.append(f'{split}: require >= {minimum_events} independently reviewed events and >=2 source groups')
        if pool == 'outcome':
            classes = {split: dict(Counter(r['approval']['label']['value'] for r in rows
                                          if r['candidate']['split'] == split)) for split in ('train', 'dev')}
            coverage['classes'] = classes
            for split, counts in classes.items():
                if not {'IN_PROGRESS', 'SUCCEEDED', 'FAILED'} <= counts.keys():
                    missing.append(split + ': missing known outcome classes; do not fabricate labels')
        if pool == 'planner' and not any(r['approval']['label']['kind'] == 'verified_recovery_continuation' for r in rows):
            missing.append('No verified recovery continuation')
        gates[pool] = dict(training_ready=not missing, coverage=coverage, blockers=missing)
    spans = independent_candidates(anchors)
    result = dict(schema='recovery_admission_v1', inventory_sha256=summary['inventory_sha256'],
        anchors_sha256=file_sha(audit / 'anchors.jsonl'), protected_groups_sha256=file_sha(protected_path),
        approvals_sha256=file_sha(approvals_path), source_manifest_sha256=summary['source_manifest_sha256'],
        bindings_sha256=summary['bindings_sha256'], pools=gates, training_ready=any(v['training_ready'] for v in gates.values()),
        candidate_labeled_spans=len(spans),
        candidate_success_spans=sum(s['proposed_outcome']=='SUCCEEDED' for s in spans),
        candidate_success_episodes=len({canonical(s['source_episode']) for s in spans if s['proposed_outcome']=='SUCCEEDED'}),
        note='Candidate spans are NOT confirmed independent recovery events; individual pool gates are mandatory.')
    return admitted, spans, result
