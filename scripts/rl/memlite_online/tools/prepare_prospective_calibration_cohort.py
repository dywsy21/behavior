"""Close a predeclared source cohort without selecting on model predictions.

Every attempted source remains in the denominator. Failed reference restores
carry their original receipt, never a fabricated robot FAILED/UNKNOWN label.
This produces review metadata only, not calibration or deployment approval.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys

REPO=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
from recovery_corpus import file_sha
from recovery_independent_cohort import cohort_sources


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('config','source-audit','external-cohort','collection','sources','corpus','output'):
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    if a.output.exists() or subprocess.check_output(['git','status','--porcelain'],cwd=REPO,text=True).strip():
        raise ValueError('New cohort and clean frozen review source required')
    cfg=json.loads(a.config.read_text());audit=json.loads(a.source_audit.read_text())
    if (cfg['schema']!='prospective_observer_calibration_collection_v1'
            or not cfg['declared_before_physical_collection_and_model_predictions']
            or audit['config_sha256']!=file_sha(a.config)
            or audit['source_inventory_sha256']!=file_sha(a.sources/'manifest.json')
            or audit['selected_observer_sha256']!=cfg['selected_observer']['sha256']
            or audit['status']!='source_closure_and_group_isolation_passed_not_labels'):
        raise ValueError('Changed prospective cohort, observer or source closure')
    closed=json.loads((a.collection/'result.json').read_text())
    if closed['status']!='inventory_attempts_finished':
        raise ValueError('Finish all declared attempts; do not review only early favorable cases')
    attempts={r['case']:r for r in closed['results']}
    originals=json.loads((a.sources/'manifest.json').read_text())['cases']
    if (len(attempts)!=len(closed['results']) or set(attempts)!={r['directory'] for r in originals}
            or len(attempts)!=cfg['selection']['metadata_fresh_sources']):
        raise ValueError('Lost, duplicated or skipped prospective attempts')
    old=json.loads(a.external_cohort.read_text())
    protected=[r['source_group'] for r in old['frozen_test_groups']]
    if old['schema']!='recovery_independent_cohort_v1' or len(protected)!=len(set(protected)):
        raise ValueError('Changed external frozen-test metadata')
    queue_path=a.corpus/'review-queue.json';queue=json.loads(queue_path.read_text())
    by_case={}
    for row in queue:by_case.setdefault(row['case'],[]).append(row)
    groups=[];unavailable=[]
    for original in originals:
        case=original['directory'];path=a.sources/case/'manifest.json'
        if file_sha(path)!=original['manifest_sha256']:raise ValueError('Changed source manifest')
        source=json.loads(path.read_text());group=source['source_group'];attempt=attempts[case]
        if source['recovery_split']!='dev' or group in protected:
            raise ValueError('TRAIN or frozen-test source entered calibration')
        record=a.collection/case/'result.json'
        if not record.exists():record=a.collection/case/'status.json'
        state=json.loads(record.read_text())
        if record.name=='result.json' and file_sha(record)!=attempt['closed_result_sha256']:
            raise ValueError('Per-source terminal receipt differs from the closed collector ledger')
        if (state.get('source_group')!=group or state.get('pid')!=attempt['pid']
                or state.get('proposal_sha256')!=original['manifest_sha256']
                or state.get('status') in ('loading','collecting','replaying_prefix')):
            raise ValueError('Missing/unbound terminal source attempt')
        rows=by_case.get(case,[])
        if rows:
            arms={r['arm'] for r in rows}
            if (len(arms)!=1 or any(r['source_group']!=group or r['split']!='dev' for r in rows)
                    or state['status']!='completed_candidates_only' or attempt['returncode']!=0):
                raise ValueError('Candidate identity or physical closure mismatch')
            entry=dict(source_group=group,case=case,arm=next(iter(arms)))
        else:
            if state['status']=='completed_candidates_only':
                raise ValueError('Completed physical data omitted by the candidate corpus; investigate first')
            entry=dict(source_group=group,case=case,arm='not_collected',
                unavailable_collection=dict(reason=state['status'],receipt=str(record.resolve()),
                    receipt_sha256=file_sha(record)))
            unavailable.append(group)
        groups.append(entry)
    if {r['source_group'] for r in groups}!={r['source_group'] for r in audit['groups']}:
        raise ValueError('Prospective source identities changed')
    spec=dict(schema='recovery_independent_calibration_only_cohort_v1',
        declared_before_any_model_predictions=True,training_forbidden=True,model_selection_forbidden=True,
        calibration_must_not_use_frozen_test=True,source_corpus=str(a.corpus.resolve()),
        source_queue_sha256=file_sha(queue_path),calibration_groups=groups,
        external_frozen_test_cohort=str(a.external_cohort.resolve()),
        external_frozen_test_cohort_sha256=file_sha(a.external_cohort),external_frozen_test_source_groups=protected,
        prospective_config=str(a.config.resolve()),prospective_config_sha256=file_sha(a.config),
        prospective_source_audit=str(a.source_audit.resolve()),prospective_source_audit_sha256=file_sha(a.source_audit),
        selected_observer_sha256=cfg['selected_observer']['sha256'],
        collection_result_sha256=file_sha(a.collection/'result.json'),
        declared_sources=len(groups),unavailable_source_groups=unavailable,
        all_sources_have_reviewable_candidates=not unavailable,calibration_ready=False,
        model_predictions_read=False,source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip())
    cohort_sources(queue,spec,'calibration')
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x') as stream:stream.write(json.dumps(spec,indent=2)+'\n')
    print(json.dumps(dict(declared_sources=len(groups),unavailable_sources=len(unavailable),
        calibration_ready=False,cohort_sha256=file_sha(a.output))))


if __name__=='__main__':main()
