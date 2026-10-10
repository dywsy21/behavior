"""Review all existing source-disjoint corrective plans; no automatic approval.

Original train/dev assignments and unsuccessful source groups remain explicit.
Only actual teacher RETRY commands with a physically verified continuation are
positive plan candidates. Raw outcome labels/actions are not granted here.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import subprocess
import sys

REPO = Path(__file__).resolve().parents[4]
sys.path[:0] = [str(REPO/'src'), str(REPO/'scripts/rl/memlite_online/code')]
from recovery_admission import local_file
from recovery_calibration_launch import write_new
from recovery_corpus import file_sha
from recovery_independent_cohort import semantic_phase_candidates
from recovery_postfit import select_postfit_sources
from recovery_teacher_corpus import validate_branch, validate_grasp_review_semantics


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'corpus', 'exposure', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--exposure-sha256', required=True)
    a = p.parse_args(); root = a.root.resolve()
    if (a.output.exists() or file_sha(a.exposure) != a.exposure_sha256
            or subprocess.check_output(['git', 'status', '--porcelain'], cwd=REPO, text=True).strip()):
        raise ValueError('New output, exact exposure identity and clean frozen source required')
    from PIL import Image, ImageDraw
    queue_path = a.corpus/'review-queue.json'
    exposure = json.loads(a.exposure.read_text())
    selected, ledger = select_postfit_sources(json.loads(queue_path.read_text()), exposure)
    a.output.mkdir(parents=True)
    write_new(a.output/'preselection.json', dict(exposure_sha256=a.exposure_sha256,
        source_queue_sha256=file_sha(queue_path), selected=selected, all_source_ledger=ledger,
        model_predictions_read=False, approvals=0))
    outcomes = {r['source_group']:r for r in ledger}
    sheets, proposals, rejected = [], [], []
    for ordinal, source in enumerate(selected):
        branch = local_file(root, Path(source['source_path'])/'manifest.json').parent
        manifest = json.loads((branch/'manifest.json').read_text())
        if (file_sha(branch/'transitions.jsonl') != manifest['transitions_sha256']
                or file_sha(branch/'plans.json') != manifest['plans_sha256']):
            raise ValueError('Changed closed source; stop instead of relabeling corruption')
        rows = [json.loads(x) for x in (branch/'transitions.jsonl').read_text().splitlines()]
        plans = json.loads((branch/'plans.json').read_text())
        validate_branch(rows, manifest, plans)
        available = sorted(int(t) for t in manifest['anchors'] if int(t) < len(rows))
        retries = [r for r in plans if r['decision'] == 'RETRY']
        try:
            if (len(retries) != 1 or not manifest['physical_recovery_candidate']
                    or manifest['failure'] is not None):
                raise ValueError('Require one actually issued verified corrective RETRY')
            event = retries[0]; retry = event['control_step']
            if retry not in available or any(r['label_kind'] != 'same_state_local_teacher_candidate' for r in rows[retry:]):
                raise ValueError('Missing decision RGB or injected actions in claimed correction')
            validate_grasp_review_semantics(rows, retry, manifest['arm'], event, planner_retry=True)
            labels, semantic_rejections = semantic_phase_candidates(rows, manifest['arm'], plans, available)
            success = next((t for t in available if t > retry and labels[t] == 'SUCCEEDED'), None)
            if success is None or labels[available[-1]] != 'SUCCEEDED':
                raise ValueError('No semantically valid achieved and stable corrective endpoint')
        except ValueError as error:
            rejection = dict(source_group=source['source_group'], case=source['case'], branch=source['branch'],
                split=source['split'], status='semantic_or_continuation_rejected_not_training', reason=str(error),
                manifest_sha256=file_sha(branch/'manifest.json'))
            outcomes[source['source_group']].update(rejection)
            rejected.append(rejection)
            continue
        frames = sorted({max(0, retry-16), retry-4, retry, retry+4, retry+16,
                         retry+32, success, available[-1]} & set(available))
        directory = a.output/(source['case']+'--'+source['branch']); directory.mkdir()
        evidence, local_sheets = [], []
        for offset in range(0, len(frames), 4):
            page_frames = frames[offset:offset+4]
            canvas = Image.new('RGB', (672, 280*len(page_frames)), 'white'); draw = ImageDraw.Draw(canvas)
            for line, t in enumerate(page_frames):
                draw.text((3, line*280+3), f'{ordinal:02d} {source["split"]} {source["case"]} / {manifest["arm"]} t={t}', fill='black')
                draw.text((3, line*280+18), 'ACTUAL RETRY TARGET' if t == retry else
                    'Context only / physical proposal: '+str(labels[t]), fill='black')
                draw.text((3, line*280+31), 'Measured exact target: '+rows[t]['physical_before']['target_name'], fill='black')
                hashes = manifest['anchors'][str(t)]['sha256']
                for col, camera in enumerate(('head_rgb', 'left_wrist_rgb', 'right_wrist_rgb')):
                    path = branch/'rgb'/f'{t:08d}'/(camera+'.jpg')
                    if file_sha(path) != hashes[camera]:
                        raise ValueError('Original image changed')
                    with Image.open(path) as im:
                        if im.size != (224, 224):
                            raise ValueError('Unexpected original camera shape')
                        canvas.paste(im.convert('RGB'), (col*224, line*280+52))
                evidence.append(dict(control_step=t, actual_physics=rows[t]['physical_before'],
                    context_only_outcome_proposal=labels[t], label_kind=rows[t]['label_kind'], image_sha256=hashes,
                    preceding_six=[dict(control_step=r['control_step'], physical_audit=r['physical_audit'])
                        for r in rows[max(0,t-6):t]]))
            image_path = directory/f'review-{offset//4:02d}.jpg'
            canvas.save(image_path, quality=95, subsampling=0)
            local_sheets.append(dict(path=image_path.name, sha256=file_sha(image_path)))
            sheets.append(dict(path=str(image_path.relative_to(a.output)), sha256=file_sha(image_path),
                source_group=source['source_group'], case=source['case'], split=source['split'],
                arm=manifest['arm'], frames=page_frames))
        material = dict(schema='offline_local_recovery_review_materials_v1', source=str(branch),
            manifest_sha256=file_sha(branch/'manifest.json'), transitions_sha256=manifest['transitions_sha256'],
            sheets=local_sheets, frames=frames, panels=3*len(frames), evidence=evidence, human_approved=False,
            actual_issued_retry=event, plan_source_sha256=manifest['plans_sha256'], semantic_rejections=semantic_rejections,
            exposure_sha256=a.exposure_sha256, action_review_complete=False,
            review_scope='Actual corrective planner command only; no action/outcome approval or model predictions')
        write_new(directory/'review.json', material)
        proposals.append(dict(case=source['case'], branch=source['branch'],
            manifest_sha256=file_sha(branch/'manifest.json'), review_directory=str(directory.resolve().relative_to(root)),
            review_sha256=file_sha(directory/'review.json'), outcomes=[], action_steps=[], planner_steps=[retry],
            corrective_execution_visually_verified=False, notes='PENDING owner review of actual plan and original imagery'))
        outcomes[source['source_group']]['status'] = 'original_media_prepared_not_approved'
    write_new(a.output/'proposed-decisions.json', dict(schema='pending_planner_review_not_approval_v1',
        reviewer='', exposure_sha256=a.exposure_sha256, branches=proposals))
    result = dict(schema='postfit_planner_review_materials_v1', source_commit=subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip(), exposure_sha256=a.exposure_sha256,
        source_queue_sha256=file_sha(queue_path), all_source_ledger=ledger, semantic_rejections=rejected,
        prepared_by_split=dict(Counter(r['split'] for r in ledger if r['status']=='original_media_prepared_not_approved')),
        sheets=sheets, original_rgb_panels=sum(3*len(s['frames']) for s in sheets),
        planner_candidates=len(proposals), outcome_candidates=0, action_candidates=0,
        model_predictions_read=False, training_ready=False)
    write_new(a.output/'review-index.json', result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('sheets', 'all_source_ledger', 'semantic_rejections')}, indent=2))


if __name__ == '__main__':
    main()
