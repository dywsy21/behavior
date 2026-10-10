"""Render zero-control RETRY observations for new human UNKNOWN review.

Only use the explicitly pre-reviewed TRAIN planner cohort. This tool never
approves a row, looks at model predictions, or modifies the old corpus.
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'code'))
from recovery_admission import local_file
from recovery_corpus import file_sha
from recovery_teacher_corpus import validate_branch, new_grasp_attempt_unknown


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('root', 'corpus', 'prior-decisions', 'output'):
        parser.add_argument('--' + key, type=Path, required=True)
    args = parser.parse_args(); root = args.root.resolve()
    prior = json.loads(args.prior_decisions.read_text())
    if prior['schema'] != 'owner_local_recovery_review_v1' or not prior['reviewer']:
        raise ValueError('Explicit prior reviewed cohort required')
    if args.output.exists():
        raise FileExistsError(args.output)
    queue = json.loads((args.corpus / 'review-queue.json').read_text())
    choices = [c for c in prior['branches'] if c['planner_steps']]
    if not choices or len({c['case'] for c in choices}) != len(choices):
        raise ValueError('Empty or repeated predeclared source cohort')
    from PIL import Image, ImageDraw
    args.output.mkdir(parents=True)
    decisions, index = [], []
    for choice in choices:
        if len(choice['planner_steps']) != 1:
            raise ValueError('One actually reviewed RETRY required per source')
        t = choice['planner_steps'][0]
        matches = [q for q in queue if (q['case'], q['branch']) == (choice['case'], choice['branch'])]
        if len(matches) != 1 or matches[0]['split'] != 'train':
            raise ValueError('Unchanged TRAIN source only; never selection/calibration/test')
        q = matches[0]
        branch = local_file(root, Path(q['source_path']) / 'manifest.json').parent
        manifest = json.loads((branch / 'manifest.json').read_text())
        if (file_sha(branch/'manifest.json') != choice['manifest_sha256']
                or file_sha(branch/'transitions.jsonl') != manifest['transitions_sha256']
                or file_sha(branch/'plans.json') != manifest['plans_sha256']):
            raise ValueError('Changed prior source')
        old_review = local_file(root, Path(choice['review_directory'])/'review.json')
        if file_sha(old_review) != choice['review_sha256']:
            raise ValueError('Changed prior decision media binding')
        rows = [json.loads(line) for line in (branch/'transitions.jsonl').read_text().splitlines()]
        plans = json.loads((branch/'plans.json').read_text())
        validate_branch(rows, manifest, plans)
        value = new_grasp_attempt_unknown(rows, t, manifest['arm'], plans)
        event = next(p for p in plans if p['control_step'] == t)
        frames = [t-4, t]
        directory = args.output/(choice['case']+'--'+choice['branch']); directory.mkdir()
        canvas = Image.new('RGB', (672, 540), 'white'); draw = ImageDraw.Draw(canvas)
        evidence = []
        for i, step in enumerate(frames):
            label = 'OLD context (no new label)' if step < t else 'NEW RETRY: 0 controls / UNKNOWN candidate'
            draw.text((3, i*270+3), choice['case']+' / '+manifest['arm']+' / t='+str(step), fill='black')
            draw.text((3, i*270+19), label, fill='black')
            hashes = manifest['anchors'][str(step)]['sha256']
            for col, camera in enumerate(('head_rgb','left_wrist_rgb','right_wrist_rgb')):
                path = branch/'rgb'/f'{step:08d}'/(camera+'.jpg')
                if file_sha(path) != hashes[camera]:
                    raise ValueError('Changed original RGB')
                with Image.open(path) as image:
                    if image.size != (224,224): raise ValueError('Unexpected original image shape')
                    canvas.paste(image.convert('RGB'), (224*col, i*270+40))
            evidence.append(dict(control_step=step, actual_physics=rows[step]['physical_before'],
                image_sha256=hashes, proposed_outcome=value if step==t else None,
                preceding_six=[dict(control_step=r['control_step'],physical_audit=r['physical_audit'])
                               for r in rows[step-6:step]]))
        sheet = directory/'review-00.jpg'; canvas.save(sheet,quality=95,subsampling=0)
        materials = dict(schema='offline_local_recovery_review_materials_v1',source=str(branch),
            manifest_sha256=choice['manifest_sha256'],transitions_sha256=manifest['transitions_sha256'],
            frames=frames,panels=6,sheets=[dict(path=sheet.name,sha256=file_sha(sheet))],evidence=evidence,
            actual_issued_retry=event,plan_source_sha256=manifest['plans_sha256'],human_approved=False,
            annotation_contract='new_attempt_zero_applied_controls_unknown_v1',
            old_attempt_causal_failure_verified=True,new_attempt_applied_controls=0,
            action_review_complete=False,review_scope='one UNKNOWN outcome only; no BC/plan promotion')
        review = directory/'review.json'; review.write_text(json.dumps(materials,indent=2)+'\n')
        decisions.append(dict(case=choice['case'],branch=choice['branch'],manifest_sha256=choice['manifest_sha256'],
            review_directory=str(directory.resolve().relative_to(root)),review_sha256=file_sha(review),
            outcomes=[dict(control_step=t,value=value)],action_steps=[],planner_steps=[],
            corrective_execution_visually_verified=False,notes='PENDING new-outcome owner review'))
        index.append(dict(case=choice['case'],control_step=t,source_group=q['source_group'],
            path=str(sheet.relative_to(args.output)),sha256=file_sha(sheet)))
    (args.output/'proposed-decisions.json').write_text(json.dumps(dict(
        schema='pending_retry_unknown_review_not_approval_v1',reviewer='',branches=decisions),indent=2)+'\n')
    receipt = dict(sources=len(choices),original_rgb_panels=6*len(choices),sheets=index,
        prior_decisions_sha256=file_sha(args.prior_decisions),training_ready=False,
        annotation_contract='new_attempt_zero_applied_controls_unknown_v1')
    (args.output/'review-index.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({k:v for k,v in receipt.items() if k!='sheets'}))


if __name__=='__main__':
    main()
