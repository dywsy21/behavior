"""Prepare original-EVAL feedback tests; never produces SFT-admitted data."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO/'src'), str(REPO/'scripts/infra'), str(REPO/'scripts/rl/memlite_online/code')]
from behavior_champion.release import AcceptedRelease, sha256
from recovery_corpus import group_key
from recovery_handoff_feedback import requests_for_eval_candidate
from recovery_handoff_heldout import select_heldout_requests, SELECTION
from recovery_periodic_teacher import (original_label_module, periodic_annotation_trace,
    command_kind, unknown_feedback, validate_candidate_projection)
from recovery_postfit_feedback import validate_exposure
from recovery_sft_data import require_training_pool
from prepare_memlite_handoff_feedback import pinned


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    a = ap.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or a.output.exists():
        raise ValueError('Fresh CPU-only heldout preparation required')
    if subprocess.check_output(['git','status','--porcelain'], cwd=REPO, text=True).strip():
        raise ValueError('Use clean frozen source')
    tick = time.monotonic()
    registration_path = REPO/'configs/recovery_sft/handoff_heldout_registration_v1.json'
    registration = json.loads(registration_path.read_text())
    if (registration['selection'] != SELECTION or registration['purpose'] != 'heldout_eval_only'
            or registration['training_admission_forbidden'] is not True):
        raise ValueError('Wrong preregistered selection')
    recipe_path = REPO/registration['original_recipe']; recipe = json.loads(recipe_path.read_text())
    exposure_path = pinned(a.root, recipe['H1']['exposure'])
    excluded = validate_exposure(json.loads(exposure_path.read_text()))
    recovery_cfg = json.loads((REPO/'configs/recovery_sft/a800_h1_periodic_final_transition_v1.json').read_text())
    _, recovery = require_training_pool(a.root/recovery_cfg['union']/'admission', 'planner', recovery_cfg['admission_sha256'])
    excluded |= {r['candidate']['source_group'] for r in recovery if r['candidate']['split'] == 'dev'}
    release = AcceptedRelease(a.root/recipe['expert_release'], manifest_sha256=registration['release_manifest_sha256'])
    import numpy as np
    for name in ('eval_candidates.npy','eval_tasks.npy'): release.verify_file(name)
    pairs = np.load(release.root/'eval_candidates.npy', mmap_mode='r')
    tasks = np.load(release.root/'eval_tasks.npy', mmap_mode='r')
    episodes = []; train_groups = set()
    for serial, ep in release.episodes():
        group = group_key(ep['task_name'], int(ep['row']['task_instance_id']))
        if ep['split'] == 'train': train_groups.add(group)
        else: episodes.append((serial, ep))
    excluded |= train_groups
    labels = original_label_module(); cursor = 0; rejected = Counter(); coverage = Counter()
    eligible = []; episode_by_serial = {}
    for serial, ep in episodes:
        episode_by_serial[serial] = ep
        anchors = labels.anchor_records(ep['segments'], ep['phase']); task = int(ep['row']['task_index'])
        part = pairs[cursor:cursor+len(anchors)]
        if (len(part) != len(anchors) or not np.all(part[:,0] == serial)
                or not np.array_equal(part[:,1], np.arange(len(anchors)))
                or not np.all(tasks[cursor:cursor+len(anchors)] == task)):
            raise ValueError('Original EVAL index identity/order changed')
        records, audit = periodic_annotation_trace(anchors, ep['segments'])
        frames = {r[0]:cursor+n for n,r in enumerate(anchors)}
        coverage[task] += len(records)
        for number, rec in enumerate(records):
            if rec['previous_intent'] == 'None':
                rejected['initial_no_previous_intent'] += 1; continue
            segment = ep['segments'][rec['segment']]
            identity = dict(episode_serial=serial, original_candidate_index=cursor+rec['original_anchor'],
                episode=int(ep['row']['episode_index']), task=task, instance=int(ep['row']['task_instance_id']),
                raw_episode=int(ep['row']['raw_episode_id']), frame=rec['frame'], phase=int(ep['phase']),
                split='eval', kind=command_kind(rec, segment))
            target = labels.projection(segment, branch='high', task_name=ep['task_name'],
                previous_intent=rec['previous_intent'], previous_parent=rec['previous_parent'], history=rec['history'])
            target.update(execution_feedback=unknown_feedback(number, records, ep['segments']),
                          low_action_supervision_mask=False)
            original = dict(identity=identity, projection=target, annotation_trace=audit,
                original_stride16_previous_intent=anchors[rec['original_anchor']][2], human_admission=False,
                teacher_kind='annotation_periodic_command_trace_not_actor_rollout')
            request, reason = requests_for_eval_candidate(original, frames, excluded_groups=excluded)
            if reason: rejected[reason] += 1
            else: eligible.append((original, request))
        cursor += len(anchors)
    if cursor != len(pairs) or set(coverage) != set(range(100)):
        raise ValueError('Incomplete original accepted EVAL universe')
    selected, eligible_changes = select_heldout_requests(eligible, excluded_groups=excluded)
    if not selected: raise ValueError('No eligible heldout comparisons')
    for original, request in selected:
        i = original['identity']; index = i['original_candidate_index']
        validate_candidate_projection(original, episode_by_serial[i['episode_serial']],
            original_index=index, original_pair=pairs[index], expected_split='eval')
    a.output.mkdir(parents=True)
    for name, values in [('candidates.jsonl',(p[0] for p in selected)), ('requests.jsonl',(p[1] for p in selected))]:
        with (a.output/name).open('x') as stream:
            for row in values: stream.write(json.dumps(row,sort_keys=True)+'\n')
    requests = [p[1] for p in selected]
    result = dict(status='CPU_HELDOUT_EVAL_REQUESTS_ONLY_TRAINING_FORBIDDEN',purpose='heldout_eval_only',
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
        registration_sha256=sha256(registration_path), recipe_sha256=sha256(recipe_path),
        release_manifest_sha256=sha256(release.root/'manifest.json'),exposure_sha256=sha256(exposure_path),
        recovery_admission_sha256=recovery_cfg['admission_sha256'],selection=SELECTION,
        candidates_sha256=sha256(a.output/'candidates.jsonl'),requests_sha256=sha256(a.output/'requests.jsonl'),
        selected=len(selected),tasks=len({p[0]['identity']['task'] for p in selected}),
        source_groups=len({r['source_group'] for r in requests}),
        original_eval_anchors=cursor,original_eval_episodes=len(episodes),train_groups_excluded=len(train_groups),
        protected_groups=len(excluded),eligible_counts=eligible_changes,rejected=dict(rejected),
        change_classes=dict(Counter(r['target_change_class'] for r in requests)),
        task_selection={str(t):dict(Counter(p[0]['identity']['kind'] for p in selected if p[0]['identity']['task']==t)) for t in range(100)},
        observer_checks=sum(len(r['requests']) for r in requests),model_calls=0,images_decoded=0,
        optimizer_updates=0,human_admitted=False,training_admission_forbidden=True,
        predictions_used_in_selection=False,seconds=time.monotonic()-tick,
        limitation=registration['limitations'])
    (a.output/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='task_selection'},indent=2),flush=True)


if __name__ == '__main__': main()
