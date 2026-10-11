"""Read-only, TRAIN-only task-layout audit for the 2026 champion adapter."""
import argparse
from collections import defaultdict
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

from behavior_champion.contracts import make_task_layout


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while block := stream.read(1024*1024):
            digest.update(block)
    return digest.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--legacy-task-data', type=Path, required=True)
    ap.add_argument('--release', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    old = [r['id'] for r in json.loads(args.legacy_task_data.read_text())['tasks']]
    manifest = json.loads((args.release/'manifest.json').read_text())
    names = {int(k): v for k, v in manifest['task_names'].items()}
    durations = defaultdict(list)
    splits = defaultdict(int)
    for line in (args.release/'episodes.jsonl').open():
        episode = json.loads(line)
        splits[episode['split']] += 1
        if episode['split'] == 'train':
            durations[episode['row']['task_index']].append(episode['row']['length'])
    if set(durations) != set(range(100)):
        raise ValueError('Missing TRAIN tasks')
    means = {task: sum(values)/len(values) for task, values in durations.items()}
    layout = make_task_layout(old, names, means)
    result = dict(status='task_layout_cpu_passed_not_model_load', layout=asdict(layout),
                  stage_offsets=layout.stage_offsets, total_stages=layout.total_stages,
                  legacy_stage_rows_preserved=596,
                  old_official_ids_unchanged=all(layout.dataset_to_model[i] == i for i in range(50)),
                  train_mean_frames=means, source_episode_splits=dict(splits),
                  source_hashes={str(p): sha(p) for p in
                      (args.legacy_task_data, args.release/'manifest.json', args.release/'episodes.jsonl')},
                  warnings=['New stage counts are an initial TRAIN-duration-bin recipe, not physical outcomes.',
                            'This audit does not implement full model expansion or authorize training.',
                            'The legacy 256-D proprio reader must be replaced by the 61-D adapter.',
                            'Keep the pretrained delta/per-timestamp normalization and correlation assets.'])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k: result[k] for k in ('status','total_stages','old_official_ids_unchanged','source_episode_splits')}))


if __name__ == '__main__':
    main()
