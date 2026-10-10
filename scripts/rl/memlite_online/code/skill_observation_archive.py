"""Lossless action-boundary observations for paired RL diagnostics.

No labels/physics enter the arrays. Identity/clocks are audit metadata only,
never policy features. This does not admit a rollout as SFT training data.
"""
from dataclasses import asdict
import json
import os
from pathlib import Path

import numpy as np

from recovery_corpus import file_sha
from skill_aligned_reward import SkillIdentity
from skill_training_protocol import actor_observation,observation_hash


class SkillObservationArchive:
    def __init__(self,root,identity,*,policy_version,policy_sha256,start_control):
        if (not isinstance(identity,SkillIdentity) or type(policy_version) is not int or policy_version<0
                or type(start_control) is not int or start_control<0
                or not isinstance(policy_sha256,str) or len(policy_sha256)!=64
                or any(c not in '0123456789abcdef' for c in policy_sha256)):
            raise ValueError('Bound episode, policy and actual starting clock required')
        self.root=Path(root);self.root.mkdir(parents=False,exist_ok=False)
        self.identity=identity;self.start=start_control;self.last=None
        self.header=dict(schema='skill_observation_archive_v1',identity=asdict(identity),
            policy_version=policy_version,policy_sha256=policy_sha256,start_control=start_control,
            format='lossless_native_rgb_uint8_CHW_proprio61_float32',
            admission_for_training=False,physical_truth_in_policy_input=False)
        (self.root/'manifest.json').write_text(json.dumps(self.header,sort_keys=True,indent=2)+'\n')

    def append(self,identity,control_step,observation):
        if (identity!=self.identity or type(control_step) is not int
                or (self.last is None and control_step!=self.start)
                or (self.last is not None and control_step<=self.last)):
            raise ValueError('Cross-episode, duplicate or noncausal observation archive')
        native=actor_observation(observation)
        if native['proprio'].dtype!=np.float32:raise ValueError('Preserve the native float32 proprio contract')
        name=f'observation-{control_step:08d}.npz'
        path=self.root/name;tmp=self.root/(name+'.pending')
        if path.exists() or tmp.exists():raise FileExistsError(path)
        with tmp.open('xb') as stream:
            np.savez_compressed(stream,proprio=native['proprio'],**native['images'])
            stream.flush();os.fsync(stream.fileno())
        # Hard-link publication is atomic and cannot overwrite existing data.
        os.link(tmp,path);tmp.unlink()
        record=dict(control_step=control_step,file=name,sha256=file_sha(path),
            observation_sha256=observation_hash(native),bytes=path.stat().st_size)
        with (self.root/'observations.jsonl').open('a') as stream:
            stream.write(json.dumps(record,sort_keys=True)+'\n');stream.flush();os.fsync(stream.fileno())
        self.last=control_step
        return record


def load_observation(root,record):
    """Read native arrays without pickle; fail closed on hash/schema drift."""
    path=Path(root)/record['file']
    if (Path(record['file']).name!=record['file'] or file_sha(path)!=record['sha256']):
        raise ValueError('Changed or unsafe observation archive file')
    with np.load(path,allow_pickle=False) as data:
        values={k:data[k] for k in data.files}
    proprio=values.pop('proprio')
    observation=actor_observation(dict(images=values,proprio=proprio))
    if observation_hash(observation)!=record['observation_sha256']:
        raise ValueError('Native observation changed across archive roundtrip')
    return observation
