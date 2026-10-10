"""Original-media materials for an already frozen calibration/test cohort.

No predictions, training, thresholds, new semantic approvals or action targets.
Keep every declared source, including unsuccessful corrective attempts.
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'code'))
from recovery_corpus import file_sha
from recovery_independent_cohort import cohort_sources, proposed_phase_points
from recovery_teacher_corpus import validate_branch, physical_proposal, PhysicalProposalIndex


def load_branch(source, root):
    branch = root/source['source_path']
    manifest = json.loads((branch/'manifest.json').read_text())
    if (file_sha(branch/'transitions.jsonl') != manifest['transitions_sha256']
            or file_sha(branch/'plans.json') != manifest['plans_sha256']):
        raise ValueError('Changed original closed branch')
    rows = [json.loads(s) for s in (branch/'transitions.jsonl').read_text().splitlines()]
    plans = json.loads((branch/'plans.json').read_text())
    validate_branch(rows, manifest, plans)
    index = PhysicalProposalIndex(rows, manifest['arm'])
    labels = {}
    for t in sorted(int(k) for k in manifest['anchors'] if int(k) < len(rows)):
        attempt = max(p['control_step'] for p in plans if p['control_step'] <= t)
        labels[t] = physical_proposal(rows, t, manifest['arm'], attempt_start=attempt, index=index)
    return branch, manifest, rows, plans, labels


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ('cohort', 'root', 'output'):
        ap.add_argument('--'+name, type=Path, required=True)
    ap.add_argument('--role', choices=('calibration','frozen_test'), required=True)
    a = ap.parse_args()
    spec = json.loads(a.cohort.read_text()); corpus = Path(spec['source_corpus'])
    if spec.get('schema')=='recovery_independent_calibration_only_cohort_v1':
        external=a.root/spec['external_frozen_test_cohort']
        prospective=a.root/spec['prospective_config']
        audit_path=a.root/spec['prospective_source_audit']
        for path,key in ((external,'external_frozen_test_cohort_sha256'),
                         (prospective,'prospective_config_sha256'),(audit_path,'prospective_source_audit_sha256')):
            if file_sha(path)!=spec[key]:raise ValueError('Changed predeclared external-test/model/source binding')
        old=json.loads(external.read_text());audit=json.loads(audit_path.read_text())
        config=json.loads(prospective.read_text())
        if (sorted(spec['external_frozen_test_source_groups'])!=sorted(r['source_group'] for r in old['frozen_test_groups'])
                or config['selected_observer']['sha256']!=spec['selected_observer_sha256']
                or audit['selected_observer_sha256']!=spec['selected_observer_sha256']
                or audit['config_sha256']!=spec['prospective_config_sha256']
                or audit['status']!='source_closure_and_group_isolation_passed_not_labels'
                or {r['source_group'] for r in audit['groups']}!={r['source_group'] for r in spec['calibration_groups']}):
            raise ValueError('Calibrating a different model or silently selecting/dropping new source groups')
    queue_path = corpus/'review-queue.json'
    if file_sha(queue_path) != spec['source_queue_sha256']:
        raise ValueError('Independent cohort source queue changed')
    selected = cohort_sources(json.loads(queue_path.read_text()), spec, a.role)
    if a.output.exists():
        raise FileExistsError(a.output)
    from PIL import Image, ImageDraw
    a.output.mkdir(parents=True); proposals = []; sheets = []; unavailable = []
    for ordinal, (entry, branches) in enumerate(selected):
        candidates = [branches[name] for name in ('open_gripper_joint_jitter','open_gripper') if name in branches]
        if not candidates or 'clean' not in branches:
            raise ValueError('Missing registered corrective or clean branch')
        # Only physical class availability, never a learned prediction or
        # successful-attempt filter. Failed corrective traces remain eligible.
        source = max(candidates, key=lambda q:sum(q['proposed_outcomes'].get(k,0)>0
                     for k in ('FAILED','IN_PROGRESS','SUCCEEDED')))
        branch, manifest, rows, plans, labels = load_branch(source, a.root)
        retry = plans[1]['control_step'] if len(plans)>1 else len(rows)
        chosen, absent = proposed_phase_points(labels, retry)
        for label in absent:
            unavailable.append(dict(source_group=entry['source_group'],label=label,
                                    reason='No physically supported anchor in this causal phase; not invented'))
        clean, cm, cr, cp, cl = load_branch(branches['clean'], a.root)
        clean_times = [t for t,v in cl.items() if v=='SUCCEEDED']
        if not clean_times:
            raise ValueError('Independent clean source needs manual investigation; do not silently drop it')
        reference = chosen[0][0]
        clean_t = min(clean_times, key=lambda t:(abs(t-reference),t))
        samples = [(source,branch,manifest,rows,chosen),
                   (branches['clean'],clean,cm,cr,[(clean_t,'SUCCEEDED')])]
        n = sum(len(item[-1]) for item in samples)
        canvas = Image.new('RGB',(672,n*267),'white'); draw = ImageDraw.Draw(canvas); line = 0
        reviews = []
        for src, folder, m, controls, points in samples:
            review = a.output/(entry['case']+'--'+src['branch']); review.mkdir()
            evidence = []
            for t,label in points:
                draw.text((4,line*267+3),f'{ordinal+1:02d} {entry["case"]} {src["branch"]}',fill='black')
                draw.text((4,line*267+19),f'{a.role} t={t} {entry["arm"]} / CANDIDATE {label}',fill='black')
                sha = m['anchors'][str(t)]['sha256']
                for col,camera in enumerate(('head_rgb','left_wrist_rgb','right_wrist_rgb')):
                    path = folder/'rgb'/f'{t:08d}'/(camera+'.jpg')
                    if file_sha(path)!=sha[camera]:raise ValueError('Changed original RGB')
                    with Image.open(path) as im:
                        if im.size!=(224,224):raise ValueError('Unexpected original image shape')
                        canvas.paste(im.convert('RGB'),(224*col,line*267+40))
                evidence.append(dict(control_step=t,proposed_outcome=label,actual_physics=controls[t]['physical_before'],
                    preceding_six=[dict(control_step=r['control_step'],physical_audit=r['physical_audit'])
                                   for r in controls[max(0,t-6):t]],image_sha256=sha))
                line += 1
            proposals.append(dict(case=entry['case'],branch=src['branch'],manifest_sha256=file_sha(folder/'manifest.json'),
                review_directory=str(review.relative_to(a.root)),
                outcomes=[dict(control_step=t,value=v) for t,v in points if v!='UNLABELLED'],
                action_steps=[],planner_steps=[],corrective_execution_visually_verified=False,
                notes='PENDING owner original-media review; independent '+a.role+' only, NEVER TRAIN or model selection'))
            reviews.append((review,dict(schema='offline_local_recovery_review_materials_v1',source=str(folder),
                manifest_sha256=file_sha(folder/'manifest.json'),transitions_sha256=m['transitions_sha256'],
                frames=[t for t,_ in points],panels=3*len(points),evidence=evidence,human_approved=False)))
        sheet = a.output/f'source-{ordinal:02d}.jpg';canvas.save(sheet,quality=95,subsampling=0)
        sha = file_sha(sheet);sheets.append(dict(path=sheet.name,sha256=sha,**entry,panels=3*n))
        for review,material in reviews:
            material['sheets']=[dict(path='../'+sheet.name,sha256=sha)]
            (review/'review.json').write_text(json.dumps(material,indent=2)+'\n')
    (a.output/'proposed-decisions.json').write_text(json.dumps(dict(schema='pending_outcome_review_not_approval_v1',
        reviewer='',branches=proposals,role=a.role,cohort_sha256=file_sha(a.cohort)),indent=2)+'\n')
    result=dict(sheets=sheets,sources=len(selected),original_rgb_panels=sum(s['panels'] for s in sheets),
        outcomes=sum(len(p['outcomes']) for p in proposals),unavailable_classes=unavailable,
        role=a.role,cohort_sha256=file_sha(a.cohort),training_ready=False,model_predictions_read=False)
    (a.output/'review-index.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('sheets','unavailable_classes')}))


if __name__=='__main__':main()
