"""Build an immutable union from individually signed, revalidated corpora.

This is relocation of existing approvals, never new semantic approval. Each
input binds its original inventory, anchors, protected groups and media. Only
episodes containing approved rows are copied; all their causal observations
remain available, but unapproved anchors never enter an objective pool.
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'code'))
from recovery_admission import audit_admission, local_file
from recovery_corpus import canonical, digest, file_sha
from recovery_sft_data import CandidateArchiveReader,require_training_pool
from recovery_evaluation_partition import build_partition


def relocate_reference(reference, old_root, shared_root):
    path = local_file(old_root, reference)
    return str(path.relative_to(shared_root.resolve()))


def build_union(spec, output):
    if spec.get('superseded_by'):
        raise ValueError('Superseded data spec must not be built: ' + str(spec['superseded_by']))
    root = Path(spec['evidence_root']).resolve()
    protected = Path(spec['protected_groups'])
    output = Path(output)
    if output.exists():
        raise FileExistsError('Use a new immutable union directory')
    inventories, anchors, histories, approvals, sources = [], [], [], [], []
    quarantine=spec.get('exposure_quarantine')
    exposed=set();removed=[];declared_exclusions=set()
    if quarantine is not None:
        if (set(quarantine)!={'fit_admission','fit_admission_sha256','groups','reason'}
                or quarantine['reason']!='previous_training_or_model_selection_source'):
            raise ValueError('Exposure quarantine requires immutable actual-fit admission')
        _,fit_rows=require_training_pool(root/quarantine['fit_admission'],'outcome',quarantine['fit_admission_sha256'])
        exposed={r['candidate']['source_group'] for r in fit_rows}
        declared_exclusions=set(quarantine['groups'])
        if len(declared_exclusions)!=len(quarantine['groups']) or not declared_exclusions<=exposed:
            raise ValueError('Only actually exposed groups may be quarantined')
    episode_owners, sample_ids, objective_ids = {}, set(), set()
    copies, approved_units = [], []
    for unit_index, unit in enumerate(spec['units']):
        corpus, evidence = Path(unit['corpus']), Path(unit['evidence_root'])
        audit = corpus / 'audit'
        original_inventory = json.loads((audit / 'inventory.json').read_text())
        original_anchors = [json.loads(s) for s in (audit / 'anchors.jsonl').read_text().splitlines()]
        original_history = [json.loads(s) for s in (corpus / 'history/contexts.jsonl').read_text().splitlines()]
        history_receipt = json.loads((corpus / 'history/receipt.json').read_text())
        if (history_receipt['inventory_sha256'] != digest(original_inventory)
                or history_receipt['anchors_sha256'] != file_sha(audit / 'anchors.jsonl')
                or history_receipt['contexts_sha256'] != file_sha(corpus / 'history/contexts.jsonl')):
            raise ValueError('Changed source causal history')
        unit_rows = []
        for approval_path in unit['approvals']:
            admitted, _, receipt = audit_admission(audit, protected, approval_path, evidence)
            unit_rows.extend(row for rows in admitted.values() for row in rows)
            sources.append(dict(corpus=str(corpus.resolve()), approvals=str(Path(approval_path).resolve()),
                approvals_sha256=file_sha(approval_path), history_sha256=history_receipt['contexts_sha256'],
                admission_audit=receipt))
        if not unit_rows:
            raise ValueError('An empty unit is not a reviewed training source')
        if quarantine is not None:
            for row in unit_rows:
                if row['candidate']['source_group'] in exposed:
                    if (row['approval'].get('usage_role')!='calibration' or row['approval']['pool']!='outcome'
                            or row['candidate']['source_group'] not in declared_exclusions):
                        raise ValueError('Cannot silently remove TRAIN, action, planner or undeclared exposure')
                    removed.append(dict(source_group=row['candidate']['source_group'],
                        sample_id=row['candidate']['sample_id'],approval=row['approval']))
            unit_rows=[r for r in unit_rows if r['candidate']['source_group'] not in exposed]
            if not unit_rows:raise ValueError('No eligible independent observations in source unit')
        approved_units.append(unit_rows)
        selected_episodes = {canonical(row['candidate']['source_episode']) for row in unit_rows}
        for identity in selected_episodes:
            if identity in episode_owners:
                raise ValueError('Duplicate episode across snapshots; choose one exact signed version')
            episode_owners[identity] = unit_index
        path_map = {}
        for item in original_inventory:
            ep = canonical([item['episode']['run'], item['episode']['episode_id']])
            if ep not in selected_episodes:
                continue
            source = local_file(corpus / 'raw', item['path'])
            if file_sha(source) != item['sha256']:
                raise ValueError('Changed source archive')
            target = f'unit_{unit_index:02d}/' + item['path']
            path_map[item['path']] = target
            clone = deepcopy(item); clone['path'] = target
            inventories.append(clone); copies.append((source, target, item['sha256']))
        selected_ids = set()
        for row in original_anchors:
            if canonical(row['source_episode']) not in selected_episodes:
                continue
            sid = row['sample_id']
            if sid in sample_ids:
                raise ValueError('Duplicate observation identity across corpus versions')
            sample_ids.add(sid); selected_ids.add(sid)
            row = deepcopy(row)
            row['actor_input']['rgb']['archive'] = path_map[row['actor_input']['rgb']['archive']]
            anchors.append(row)
        unit_history = [r for r in original_history if r['sample_id'] in selected_ids]
        if len(unit_history) != len(selected_ids) or len({r['sample_id'] for r in unit_history}) != len(selected_ids):
            raise ValueError('Missing/duplicated causal observation history')
        histories.extend(unit_history)
        for row in unit_rows:
            app = deepcopy(row['approval'])
            key = (app['sample_id'], app['pool'], app['label'].get('member_index'))
            if key in objective_ids:
                raise ValueError('Duplicate objective approval')
            objective_ids.add(key)
            for ref in app['evidence']:
                ref['path'] = relocate_reference(ref['path'], evidence, root)
            if app['pool'] == 'planner':
                app['label']['verified_plan_path'] = relocate_reference(app['label']['verified_plan_path'], evidence, root)
            approvals.append(app)
    if quarantine is not None and {r['source_group'] for r in removed}!=declared_exclusions:
        raise ValueError('Exposure quarantine differs from actual approved source intersection')
    partition = build_partition(spec.get('evaluation_partition'), approved_units)
    output.mkdir(parents=True, exist_ok=False)
    for directory in ('raw', 'audit', 'history', 'admission'):
        (output / directory).mkdir()
    for source, relative, sha in copies:
        destination = output / 'raw' / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        if file_sha(destination) != sha:
            raise ValueError('Archive relocation failed exact content check')
    def write_json(path, value):
        path.write_text(json.dumps(value, indent=2) + '\n')
    def write_rows(path, rows):
        path.write_text(''.join(canonical(row) + '\n' for row in rows))
    write_json(output / 'source-provenance.json', sources)
    if quarantine is not None:
        write_json(output/'exposure-quarantine.json',dict(quarantine,removed_approvals=removed,
            original_approvals_unchanged=True,quarantined_source_count=len(declared_exclusions)))
    write_json(output / 'audit/inventory.json', inventories)
    write_rows(output / 'audit/anchors.jsonl', anchors)
    write_rows(output / 'history/contexts.jsonl', histories)
    bindings = digest([s['admission_audit']['bindings_sha256'] for s in sources])
    manifest = digest([s['admission_audit']['source_manifest_sha256'] for s in sources])
    summary = dict(schema='reviewed_recovery_union_audit_v1', source_manifest_supplied=True,
        protected_groups_supplied=True, bindings_sha256=bindings, source_manifest_sha256=manifest,
        conflicting_evidence=0, quarantined=0, inventory_sha256=digest(inventories),
        protected_groups_sha256=file_sha(protected), source_provenance_sha256=file_sha(output/'source-provenance.json'),
        derived_identity_note='Manifest/binding digests identify ordered original validated sources, not new semantic labels.')
    write_json(output / 'audit/summary.json', summary)
    write_json(output / 'history/receipt.json', dict(inventory_sha256=digest(inventories),
        contexts_sha256=file_sha(output/'history/contexts.jsonl'), anchors_sha256=file_sha(output/'audit/anchors.jsonl')))
    write_json(output / 'approvals.json', dict(schema='recovery_sample_approvals_v1',
        inventory_sha256=digest(inventories), anchors_sha256=file_sha(output/'audit/anchors.jsonl'), approvals=approvals))
    admitted, _, receipt = audit_admission(output/'audit', protected, output/'approvals.json', root)
    files = {}
    for pool, rows in admitted.items():
        path = output / 'admission' / (pool + '.jsonl'); write_rows(path, rows); files[path.name] = file_sha(path)
    if partition is not None:
        path = output / 'admission/evaluation_partition.json'
        write_json(path, partition)
        files[path.name] = file_sha(path)
    # Read every original observation including unlabelled causal history;
    # separately read every positive action target, never terminal-only rows.
    reader = CandidateArchiveReader(output/'raw', inventories)
    for row in anchors:
        reader.observation(row)
    for item in admitted['action']:
        reader.observation_and_actions(item['candidate'])
    receipt.update(files=files, source_provenance_sha256=file_sha(output/'source-provenance.json'))
    if partition is not None:
        receipt['evaluation_partition_file'] = 'evaluation_partition.json'
    write_json(output / 'admission/admission.json', receipt)
    result = dict(schema='reviewed_recovery_union_receipt_v1', status='passed',
        admission_sha256=file_sha(output/'admission/admission.json'), copied_archives=len(copies),
        observations_read=len(anchors), approved_rows={k:len(v) for k,v in admitted.items()},
        pools=receipt['pools'], optimizer_steps=0, new_semantic_approvals=0,
        evidence_root=str(root), raw_root=str(output/'raw'))
    write_json(output / 'receipt.json', result)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--spec',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();print(json.dumps(build_union(json.loads(a.spec.read_text()),a.output),indent=2))


if __name__=='__main__': main()
