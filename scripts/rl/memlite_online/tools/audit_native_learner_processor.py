"""All-row real low processor check; no optimizer or launch authorization."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

REPO=Path(__file__).resolve().parents[4]
sys.path[:0]=[str(REPO/'src'),str(REPO/'scripts/rl/memlite_online/code')]
from recovery_corpus import file_sha
from recovery_calibration_launch import read,write_new
from recovery_native_actions import NativeLearnerActionReader,native_low_raw


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('root','bundle','output'):p.add_argument('--'+key,type=Path,required=True)
    for key in ('manifest-sha256','owner-review-sha256','stats-sha256'):p.add_argument('--'+key,required=True)
    a=p.parse_args()
    if a.output.exists() or subprocess.check_output(['git','status','--porcelain'],cwd=REPO,text=True).strip():
        raise ValueError('Frozen source and new audit output required')
    import torch
    from g05.utils.training.stage1_model import configuration,make_processor
    reader=NativeLearnerActionReader(a.bundle,a.manifest_sha256,a.owner_review_sha256)
    names=read(a.root/'datasets/memlite-stage1-20260930-v4/manifest.json')['task_names']
    config=configuration(a.root,'low',names)
    if file_sha(config['stats_path'])!=a.stats_sha256:raise ValueError('Changed inherited low normalization')
    processor=make_processor(config,False);rows=[]
    for i,item in enumerate(reader.rows):
        raw=native_low_raw(reader,i,config);sample=processor.preprocess(raw)
        if (sample['action'].shape!=(32,27) or not torch.isfinite(sample['action']).all()
                or tuple(torch.nonzero(sample['action_dim_is_pad']).flatten().tolist())!=(7,8,17,18)
                or sample['action_is_pad'].any()):
            raise ValueError('Changed native action clock, R1Pro mask, normalization or complete32 target')
        rows.append(dict(sample_id=item['sample_id'],source_group=item['source_group'],
            observation_step=item['observation']['control_step'],actions_sha256=item['actions_sha256'],
            shape=list(sample['action'].shape),padding_indices=[7,8,17,18]))
    result=dict(schema='native_learner_processor_audit_v1',status='passed',
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
        manifest_sha256=a.manifest_sha256,owner_review_sha256=a.owner_review_sha256,stats_sha256=a.stats_sha256,
        samples=rows,optimizer_steps=0,oracle_inputs=False,training_launch_authorized=False,
        scope='Current native RGB/proprio and independently ACKed32 raw23 actions only; not new expert or high/observer labels')
    a.output.parent.mkdir(parents=True,exist_ok=True);write_new(a.output,result)
    print(json.dumps(dict(status='passed',samples=len(rows),audit_sha256=file_sha(a.output))))


if __name__=='__main__':main()
