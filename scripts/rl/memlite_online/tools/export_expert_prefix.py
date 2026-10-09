"""Export small immutable expert prefixes, never label them as recoveries.

CPU only; original accepted TRAIN/protected instance checks precede media IO.
The simulator must independently validate reset alignment and real execution.
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'code'))
from recovery_corpus import file_sha, group_key, split_group


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--recipe', type=Path, required=True)
    p.add_argument('--protected', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists(): raise FileExistsError(a.output)
    recipe = json.loads(a.recipe.read_text()); root = Path(recipe['root'])
    release = root / recipe['expert_release']
    accepted = json.loads((release/'acceptance.json').read_text())
    if (accepted['status'] != 'ACCEPTED' or not all(accepted['gates'].values())
            or file_sha(release/'manifest.json') != accepted['manifest_sha256']):
        raise ValueError('Original release not accepted')
    protected = set(json.loads(a.protected.read_text())['groups'])
    # Chosen from annotation order/short length, never by observed policy loss
    # or simulator outcome. 2 x 352 + 2 x 16 replay controls maximum here.
    cases = {2556: ('preparing lunch box', 241), 9928: ('make pizza', 179)}
    import numpy as np
    from PIL import Image
    from g05.data.memlite_stage1_dataset import Stage1Dataset
    from g05.utils.training.stage1_model import configuration
    from g05.data.lerobot.datasets.video_utils import decode_video_frames_torchcodec
    episodes = [json.loads(x) for x in (release/'episodes.jsonl').read_text().splitlines()]
    names = {x['row']['task_index']: x['task_name'] for x in episodes}
    config = configuration(root, 'low', names)
    dataset = Stage1Dataset(release, config, 'low', 'train')
    result = []
    a.output.mkdir(parents=True)
    for ep in episodes:
        r = ep['row']; i = r['episode_index']
        if i not in cases: continue
        task, instance = cases[i]
        if (ep['split'] != 'train' or ep['task_name'] != task or r['task_instance_id'] != instance
                or group_key(task, instance) in protected):
            raise ValueError('Wrong source or protected instance')
        out = a.output / f'{task.replace(" ", "_")}_{instance}'
        out.mkdir()
        arrays = dataset._raw_arrays(r)
        np.savez(out/'prefix.npz', action=arrays['action'][:352], state=arrays['observation.state'][:353])
        files = {'prefix.npz': file_sha(out/'prefix.npz')}
        frames = [0, 64, 128, 192, 256, 320, 352]
        for m in config['raw_shape']['images']:
            key = 'videos/'+m['lerobot_key']
            path = dataset.root / f"{key}/chunk-{r[key+'/chunk_index']:03d}/file-{r[key+'/file_index']:03d}.mp4"
            rgb = decode_video_frames_torchcodec(path, [r[key+'/from_timestamp']+t/30 for t in frames],
                                                tolerance_s=.4/30, device='cpu')
            for j, t in enumerate(frames):
                name = f'{t:06d}-{m["key"]}.png'
                Image.fromarray(rgb[j].mul(255).round().byte().permute(1, 2, 0).numpy()).save(out/name)
                files[name] = file_sha(out/name)
        row = dict(task=task.replace(' ', '_'), instance_id=instance, episode_index=i,
            source_group=group_key(task, instance), recovery_split=split_group(task, instance),
            source_episode=ep, files=files, controls=352, prefix_is_verified=False,
            original_release_sha256=file_sha(release/'manifest.json'),
            protected_groups_sha256=file_sha(a.protected), robot_action_dim=23,
            scope='expert demonstration proposal only; not a world snapshot or recovery label')
        (out/'manifest.json').write_text(json.dumps(row, indent=2)+'\n')
        result.append(dict(directory=out.name, manifest_sha256=file_sha(out/'manifest.json')))
    if len(result) != len(cases): raise ValueError('Missing selected source episode')
    (a.output/'manifest.json').write_text(json.dumps(dict(schema='expert_prefix_proposals_v1', cases=result), indent=2)+'\n')
    print(json.dumps(dict(output=str(a.output), cases=result)))


if __name__ == '__main__': main()
