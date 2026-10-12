"""CPU-only source reconstruction for next MEM feedback experiment; no fitting."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO/'src'), str(REPO/'scripts/rl/memlite_online/code')]
from behavior_champion.release import AcceptedRelease, sha256
from recovery_handoff_feedback import requests_for_candidate, SCHEMA
from recovery_periodic_teacher import validate_candidate_projection, original_label_module
from recovery_postfit_feedback import validate_exposure
from recovery_sft_data import require_training_pool


def pinned(root, spec):
    path = (root/spec['path']).resolve()
    if not path.is_relative_to(root.resolve()) or sha256(path) != spec['sha256']:
        raise ValueError('Unpinned or foreign input')
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    a = ap.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or a.output.exists():
        raise ValueError('CPU-only preparation and a fresh output directory required')
    if subprocess.check_output(['git', 'status', '--porcelain'], cwd=REPO, text=True).strip():
        raise ValueError('Use clean frozen source')
    started = time.monotonic()
    recipe_path = REPO/'configs/recovery_sft/a800_h1_periodic_teacher_fit_lc1_v1.json'
    recipe = json.loads(recipe_path.read_text()); spec = recipe['H1']['periodic_teacher']
    candidates = pinned(a.root, dict(path=spec['candidates'], sha256=spec['candidate_sha256']))
    exposure_path = pinned(a.root, recipe['H1']['exposure'])
    exposure = json.loads(exposure_path.read_text())
    excluded = validate_exposure(exposure)
    # The former recovery DEV may not become a new planner TRAIN positive.
    eval_config = json.loads((REPO/'configs/recovery_sft/a800_h1_periodic_final_transition_v1.json').read_text())
    admission, recovery_rows = require_training_pool(a.root/eval_config['union']/'admission',
                                                     'planner', eval_config['admission_sha256'])
    recovery_dev = {r['candidate']['source_group'] for r in recovery_rows if r['candidate']['split'] == 'dev'}
    excluded |= recovery_dev
    accepted = AcceptedRelease(a.root/recipe['expert_release'])
    import numpy as np
    for name in ('train_candidates.npy', 'train_tasks.npy'): accepted.verify_file(name)
    pairs = np.load(accepted.root/'train_candidates.npy', mmap_mode='r')
    tasks = np.load(accepted.root/'train_tasks.npy', mmap_mode='r')
    rows = [json.loads(line) for line in candidates.read_text().splitlines()]
    if len(rows) != 1800: raise ValueError('Original reviewed candidate universe changed')
    wanted = {r['identity']['episode_serial'] for r in rows}
    episodes = {serial:ep for serial,ep in accepted.episodes('train') if serial in wanted}
    if set(episodes) != wanted: raise ValueError('Candidate crossed source quarantine/split')
    labels = original_label_module(); prepared = []; rejected = Counter(); frames_by_episode = {}
    for row in rows:
        i = row['identity']; index = i['original_candidate_index']; serial = i['episode_serial']
        if int(tasks[index]) != i['task']: raise ValueError('Cross-task original source')
        episode = episodes[serial]
        validate_candidate_projection(row, episode, original_index=index, original_pair=pairs[index])
        if serial not in frames_by_episode:
            anchors = labels.anchor_records(episode['segments'], episode['phase'])
            offset = index - int(pairs[index][1])
            frame_map = {r[0]:offset+n for n,r in enumerate(anchors)}
            for n in range(len(anchors)):
                if tuple(map(int, pairs[offset+n])) != (serial, n) or int(tasks[offset+n]) != i['task']:
                    raise ValueError('Original per-episode anchor identity changed')
            frames_by_episode[serial] = frame_map
        value, reason = requests_for_candidate(row, frames_by_episode[serial], excluded_groups=excluded)
        if reason: rejected[reason] += 1
        else: prepared.append(value)
    a.output.mkdir(parents=True)
    with (a.output/'requests.jsonl').open('x') as stream:
        for row in prepared: stream.write(json.dumps(row, sort_keys=True) + '\n')
    result = dict(schema=SCHEMA, status='CPU_REQUESTS_ONLY_NOT_GENERATED_OR_TRAINING_ADMITTED',
        source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip(),
        source_candidates_sha256=sha256(candidates), recipe_sha256=sha256(recipe_path),
        exposure_sha256=sha256(exposure_path), recovery_admission_sha256=eval_config['admission_sha256'],
        requests_sha256=sha256(a.output/'requests.jsonl'), original_rows_reconstructed=len(rows),
        selected=len(prepared), tasks=len({r['original_identity']['task'] for r in prepared}),
        source_groups=len({r['source_group'] for r in prepared}),
        change_classes=dict(Counter(r['target_change_class'] for r in prepared)), rejected=dict(rejected),
        observer_checks=sum(len(r['requests']) for r in prepared),
        protected_groups=len(excluded), recovery_dev_groups=len(recovery_dev),
        images_decoded=0, model_calls=0, optimizer_updates=0, human_admitted=False,
        seconds=time.monotonic()-started,
        limitation='Annotation-derived command history, not on-policy data or physical success truth. Must validate RGB/prefix equivalence, run the actual frozen observer and human-review pairs before training.')
    (a.output/'result.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__': main()
