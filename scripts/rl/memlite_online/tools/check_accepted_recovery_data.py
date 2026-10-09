"""Exercise every admitted new H1/L0 row through the real A800 processor.

No learning or observer-label fabrication: H1's diagnostic feedback is the
explicit UNKNOWN fallback. It is only an in-memory contract probe, never a
feedback file for training. The future H0/OOF gate is unchanged.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
from recovery_corpus import digest,file_sha
from recovery_admission import local_file
from recovery_sft_data import (VerifiedRecoveryActionDataset,CandidateArchiveReader,raw_observation,require_training_pool)
from recovery_planner_data import planner_projection
from recovery_observer_training import feedback_text


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('recipe','corpus','admission','evidence-root','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--admission-sha256',required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    import torch
    from g05.utils.training.stage1_model import configuration,make_processor
    torch.set_num_threads(2)
    recipe=json.loads(a.recipe.read_text());root=Path(recipe['root'])
    names=json.loads((root/recipe['expert_release']/'manifest.json').read_text())['task_names']
    inventory=json.loads((a.corpus/'audit/inventory.json').read_text())
    low=configuration(root,'low',names);high=configuration(root,'high',names)
    checks=[]
    for split in ('train','dev'):
        dataset=VerifiedRecoveryActionDataset(a.admission,a.corpus/'raw',inventory,low,
            split=split,admission_sha256=a.admission_sha256)
        for index in range(len(dataset)):
            sample=dataset[index];s=sample['samples']
            if (not s['low_action_supervision_mask'] or s['memlite_branch']!='low'
                    or s.get('outcome_supervision_mask',False) or sample['action_is_pad'].any()):
                raise ValueError('Low masks or branch mismatch')
            if not all(torch.isfinite(v).all() for v in sample['pixel_values'].values()):raise ValueError('Nonfinite RGB')
            checks.append(dict(component='L0',split=split,sample_id=dataset.rows[index]['candidate']['sample_id'],
                event=dataset.rows[index]['approval']['event_id'],action_shape=list(sample['action'].shape),
                padding=torch.nonzero(sample['action_dim_is_pad']).flatten().tolist()))
    _,planners=require_training_pool(a.admission,'planner',a.admission_sha256)
    history={r['sample_id']:r for r in (json.loads(x) for x in (a.corpus/'history/contexts.jsonl').read_text().splitlines())}
    reader=CandidateArchiveReader(a.corpus/'raw',inventory);processor=make_processor(high,False)
    for item in planners:
        row=item['candidate'];sid=row['sample_id'];h=history[sid];prior=h['predecision'];label=item['approval']['label']
        path=local_file(a.evidence_root,label['verified_plan_path'])
        if file_sha(path)!=label['verified_plan_sha256']:raise ValueError('Changed approved plan')
        # Deliberately synthetic diagnostic identifiers, not an actual fitted
        # observer. No persisted feedback/training rows are emitted here.
        provenance=dict(high_sha256=recipe['parents']['high']['sha256'],observer_sha256='b'*64,
            trained_groups=[],calibration_groups=[],target_groups=[h['source_group']],feature_cache_sha256='c'*64,
            fold=-1,calibrated=False,temperature=1.,calibration_receipt_sha256='d'*64)
        feedback=dict(sample_id=sid,source_group=h['source_group'],control_step=h['control_step'],provenance=provenance,
            execution_feedback=feedback_text(prior,[dict(member=i,estimated_outcome='UNKNOWN',confidence=0.)
                for i,_ in enumerate(json.loads(prior['issued_skills_semantic_json']))]))
        projection=planner_projection(row,h,json.loads(path.read_text()),feedback,recipe['parents']['high']['sha256'])
        raw=raw_observation(row,reader,high)
        raw.update(model_projection=projection,action_is_pad=torch.ones(32,dtype=torch.bool),
            action={m['key']:torch.zeros(32,m['raw_shape']) for m in high['raw_shape']['action']})
        sample=processor.preprocess(raw);s=sample['samples']
        if (s['memory']!=prior['memory'] or s['previous_intent']!=prior['previous_intent']
                or s['known_previous_outcome']!='UNKNOWN' or s['outcome_supervision_mask']
                or s['low_action_supervision_mask'] or not sample['action_is_pad'].all()):
            raise ValueError('Planner leaked target/physics or supervised a masked output')
        checks.append(dict(component='H1',split=row['split'],sample_id=sid,event=item['approval']['event_id'],
            feedback='synthetic_UNKNOWN_contract_only',prior_control_step=prior['control_step'],decision_control_step=row['control_step']))
    result=dict(schema='accepted_recovery_processor_audit_v1',status='passed',admission_sha256=a.admission_sha256,
        inventory_sha256=digest(inventory),checks=checks,counts=dict(Counter(r['component']+'_'+r['split'] for r in checks)),
        optimizer_steps=0,generated_training_feedback=False,oracle_inputs=False)
    a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='checks'},indent=2))


if __name__=='__main__':main()
