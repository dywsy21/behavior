"""Export original TRAIN sources for physical recovery preparation.

Selection precedes simulation. Task-instance grouping and original holdouts are
immutable. Exported actions are references, NOT a corrective teacher or labels.
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'code'))
from recovery_corpus import digest, file_sha, group_key, split_group


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--recipe', type=Path, required=True)
    p.add_argument('--protected', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--tasks', nargs='+', default=['preparing lunch box', 'make pizza',
                                                'turning on radio', 'set up a coffee station in your kitchen'])
    p.add_argument('--train-groups-per-task', type=int, default=2)
    p.add_argument('--dev-groups-per-task', type=int, default=1)
    a = p.parse_args()
    if a.output.exists(): raise FileExistsError(a.output)
    import numpy as np
    from PIL import Image
    from g05.data.memlite_stage1_dataset import Stage1Dataset
    from g05.utils.training.stage1_model import configuration
    from g05.data.lerobot.datasets.video_utils import decode_video_frames_torchcodec
    recipe = json.loads(a.recipe.read_text()); root = Path(recipe['root'])
    release = root/recipe['expert_release']
    accepted = json.loads((release/'acceptance.json').read_text())
    if (accepted['status'] != 'ACCEPTED' or not all(accepted['gates'].values())
            or file_sha(release/'manifest.json') != accepted['manifest_sha256']):
        raise ValueError('Original release is not accepted')
    protected = set(json.loads(a.protected.read_text())['groups'])
    episodes = [json.loads(x) for x in (release/'episodes.jsonl').read_text().splitlines()]
    config = configuration(root, 'low', {x['row']['task_index']:x['task_name'] for x in episodes})
    dataset = Stage1Dataset(release, config, 'low', 'train')
    selected = []
    for task in a.tasks:
        candidates = []
        for ep in episodes:
            r = ep['row']; group = group_key(task, r['task_instance_id'])
            if ep['task_name'] != task or ep['split'] != 'train' or group in protected: continue
            segment = next((s for s in ep['segments'] if len(json.loads(s['semantic'])) == 1
                and json.loads(s['semantic'])[0]['verb'] == 'GRASP'
                and json.loads(s['semantic'])[0]['target']
                and not json.loads(s['semantic'])[0]['unbound_relation']), None)
            if segment is None: continue
            candidates.append((ep, segment, split_group(task, r['task_instance_id'])))
        for split, count in (('train', a.train_groups_per_task), ('dev', a.dev_groups_per_task)):
            # Short original annotated prefixes first; never use rollout results.
            rows = sorted([x for x in candidates if x[2] == split],
                          key=lambda x:(x[1]['end'], digest([task, x[0]['row']['task_instance_id']])))
            if len(rows) < count: raise ValueError(f'Insufficient original groups: {task}/{split}')
            selected.extend(rows[:count])
    a.output.mkdir(parents=True)
    result = []
    for ep, segment, split in selected:
        r = ep['row']; task = ep['task_name']; controls = min(segment['end']+64, r['length']-1)
        out = a.output/f"{task.replace(' ', '_')}_{r['task_instance_id']}"; out.mkdir()
        arrays = dataset._raw_arrays(r)
        np.savez(out/'prefix.npz', action=arrays['action'][:controls],
                 state=arrays['observation.state'][:controls+1])
        files = {'prefix.npz':file_sha(out/'prefix.npz')}
        frames = sorted({0,segment['start'],(segment['start']+segment['end'])//2,segment['end'],controls})
        for m in config['raw_shape']['images']:
            key = 'videos/'+m['lerobot_key']
            path = dataset.root/f"{key}/chunk-{r[key+'/chunk_index']:03d}/file-{r[key+'/file_index']:03d}.mp4"
            rgb = decode_video_frames_torchcodec(path,[r[key+'/from_timestamp']+t/30 for t in frames],
                                                tolerance_s=.4/30,device='cpu')
            for j,t in enumerate(frames):
                name = f'{t:06d}-{m["key"]}.png'
                Image.fromarray(rgb[j].mul(255).round().byte().permute(1,2,0).numpy()).save(out/name)
                files[name] = file_sha(out/name)
        manifest = dict(schema='recovery_expert_grasp_proposal_v2',task=task.replace(' ','_'),
            instance_id=r['task_instance_id'],episode_index=r['episode_index'],
            source_group=group_key(task,r['task_instance_id']),recovery_split=split,
            source_episode=ep,selected_segment=segment,controls=controls,files=files,reference_frames=frames,
            original_release_sha256=file_sha(release/'manifest.json'),protected_groups_sha256=file_sha(a.protected),
            robot_action_dim=23,prefix_is_verified=False,training_approved=False,
            selection='original annotated prefix length then fixed identity hash; no rollout selection')
        (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        result.append(dict(directory=out.name,manifest_sha256=file_sha(out/'manifest.json'),split=split,
                           episode=r['episode_index'],controls=controls))
    (a.output/'manifest.json').write_text(json.dumps(dict(schema='recovery_grasp_source_index_v2',cases=result),indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__ == '__main__': main()
