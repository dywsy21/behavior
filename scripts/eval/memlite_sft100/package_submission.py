"""Build a draft portal bundle after a complete, verified evaluation.

Run from a separate checkout; never change the active evaluator. Keep internal
provenance JSON outside the ZIP so every *.json inside it is an original rollout.
The existing results.zip remains an internal reproducibility bundle.
"""
import argparse
import json
import os
from pathlib import Path
import time
import zipfile

from common import aggregate, atomic_json, sha256


def wait_for_completion(job):
    """One-shot finalization dependent on the registered evaluator, not retries."""
    job = Path(job).resolve()
    while True:
        state = json.loads((job / 'status.json').read_text())
        if state['status'] == 'completed':
            return
        if state['status'] != 'running':
            raise RuntimeError('Evaluator needs diagnosis; no submission package created')
        pid = int(state['pid'])
        os.kill(pid, 0)
        command = Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0')
        if os.fsencode(str(job)) not in command or not any(x.endswith(b'/supervise.py') for x in command):
            raise RuntimeError('Evaluator process identity changed')
        time.sleep(30)


def package(job):
    job = Path(job)
    state = json.loads((job / 'status.json').read_text())
    summary = json.loads((job / 'summary.json').read_text())
    if state['status'] != 'completed' or summary['status'] != 'complete':
        raise ValueError('Only a fully completed evaluation may be packaged')
    manifest = json.loads((job / 'manifest.json').read_text())
    inventory = json.loads((job / 'submission/rollout_inventory.json').read_text())
    checklist = json.loads((job / 'submission/submission_checklist.json').read_text())
    target = job / 'submission/challenge_results_draft.zip'
    if target.exists() or target.with_suffix('.zip.partial').exists():
        raise ValueError('Never overwrite a previous submission bundle')
    records, originals, names = [], [], set()
    for row in inventory:
        path, video = Path(row['metrics']), Path(row['video'])
        # Inputs must remain inside this run, not arbitrary external paths.
        path.resolve().relative_to((job / 'tasks').resolve())
        video.resolve().relative_to((job / 'tasks').resolve())
        if sha256(path) != row['metrics_sha256']:
            raise ValueError('Original metrics changed: ' + str(path))
        if video.stat().st_size != row['video_bytes'] or sha256(video) != row['video_sha256']:
            raise ValueError('Original video changed: ' + str(video))
        record = json.loads(path.read_text())
        name = f"{record['task']}_{record['instance_id']}_{record['rollout_id']}.json"
        if path.name != name or video.stem != path.stem or name in names:
            raise ValueError('Mismatched or duplicate rollout filename')
        names.add(name)
        records.append(record)
        originals.append((path, name))
    report = aggregate(manifest['tasks'], records)
    if report['completed'] != 1000:
        raise ValueError('The requested full 1000-case inventory is incomplete')
    for key in ('official_q_score', 'official_sr'):
        if abs(report[key] - summary[key]) > 1e-12:
            raise ValueError('Final score disagrees with original outputs: ' + key)
    for key in ('wrapper', 'robot'):
        info = checklist[key]
        if sha256(job / 'submission' / info['path']) != info['sha256']:
            raise ValueError('Submitted configuration changed: ' + key)
    commands = [json.loads(p.read_text()) for p in sorted((job / 'tasks').glob('*/command.json'))]
    if len(commands) != 100:
        raise ValueError('Expected one exact evaluator command per task')
    readme = '\n'.join([
        '# MEM-Lite Stage-1 SFT — draft submission materials', '',
        'Not uploaded or submitted. Deployment information below is still pending.',
        'Every JSON in this ZIP is an untouched official rollout result.',
        'Original MP4 files are separate; provide their hosted link via the portal.',
        f"Evaluation source: {manifest['source_commit']}",
        f"Official simulator: {manifest['official_tag']} / {manifest['official_commit']}",
        '100 tasks × public indices 0–9 (IDs301–310) × rollout0; default official timeout.',
        'RGBOnlyFullResWrapper in rgb_wrapper.py; unchanged official r1pro.yaml.',
        'Native SFT FM10, predict32/execute16 from index0, raw23 controls, no optimizer.',
        'Single-frame three-camera planner, greedy every128 controls; policy seed17 / sim seed0.',
        'public_test301 has prior development exposure: this is not a blind test.',
        '', '## Still required before a real submission', '',
        '- Docker image with launch command and measured single24GB acceptance,',
        '  or public policy IP with at least50 evaluation ports.',
        '- Portable websocket serving adapter: the current local server expects',
        '  current_batch.json from our scheduler. Validate equivalent policy outputs.',
        '- Hosted unmodified-video URL and confirmed portal identity/disclosure fields.',
        '', '## Exact original evaluation commands', '',
        'Paths are those used on the original evaluation host. No claims of a portable image.',
        '```json', json.dumps(commands, indent=2), '```',
        '', '## Model, configuration and normalization provenance', '',
        '```json', json.dumps(manifest['checkpoints'], indent=2), '```', '',
    ])
    partial = target.with_suffix('.zip.partial')
    with zipfile.ZipFile(partial, 'x', compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr('README.md', readme)
        for name in ('rgb_wrapper.py', 'r1pro.yaml'):
            archive.write(job / 'submission' / name, name)
        for path, name in originals:
            archive.write(path, name)
    with zipfile.ZipFile(partial) as archive:
        if archive.testzip() is not None:
            raise ValueError('ZIP integrity check failed')
        if {n for n in archive.namelist() if n.endswith('.json')} != names:
            raise ValueError('Non-rollout JSON found in challenge bundle')
    partial.rename(target)
    receipt = dict(created=time.time(),status='draft_materials_not_ready_for_submission',
        zip=str(target),sha256=sha256(target),rollout_count=1000,
        original_metrics_and_videos_hash_verified=True,
        every_zip_json_is_original_rollout=True,submission_authorized=False,
        source_commit=manifest['source_commit'],
        official_q_score=report['official_q_score'],official_sr=report['official_sr'])
    atomic_json(job / 'submission/challenge_bundle_receipt.json', receipt)
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--job', type=Path, required=True)
    parser.add_argument('--wait', action='store_true')
    args = parser.parse_args()
    if args.wait:
        wait_for_completion(args.job)
    print(json.dumps(package(args.job), indent=2))
