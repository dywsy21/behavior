"""CPU-only real RGB/proprio plus synthetic feedback contract check; no labels/optimizer.

One actual decision anchor is selected by clock, not success. Synthetic result
categories exercise the full existing processor; they are never data approval,
learned predictions or training output. This does not bypass a formal pool gate.
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
from recovery_corpus import file_sha
from recovery_sft_data import CandidateArchiveReader,raw_observation
from recovery_observer_training import OUTCOMES,feedback_text
from recovery_planner_data import planner_projection


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('recipe','raw','audit','history','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    import torch
    from g05.utils.training.stage1_model import configuration,make_processor
    recipe=json.loads(a.recipe.read_text());root=Path(recipe['root'])
    manifest=json.loads((root/recipe['expert_release']/'manifest.json').read_text())
    config=configuration(root,'high',manifest['task_names']);processor=make_processor(config,False)
    inventory=json.loads((a.audit/'inventory.json').read_text())
    anchors={r['sample_id']:r for r in (json.loads(x) for x in (a.audit/'anchors.jsonl').read_text().splitlines())}
    histories=[json.loads(x) for x in (a.history/'contexts.jsonl').read_text().splitlines()]
    h=next(r for r in histories if r['predecision'] is not None and anchors[r['sample_id']]['split']=='train'
           and 'binding_quarantine' not in anchors[r['sample_id']]['label_audit'])
    candidate=anchors[h['sample_id']];current=h['observable'];prior=h['predecision']
    target=dict(schema='recovery_verified_plan_v1',sample_id=candidate['sample_id'],source_episode=candidate['source_episode'],
        control_step=candidate['control_step'],parent_goal=current['parent_goal'],
        active_skills_semantic_json=current['issued_skills_semantic_json'],decision=current['issued_decision'],memory_update=current['memory'])
    reader=CandidateArchiveReader(a.raw,inventory);checks=[]
    for value in OUTCOMES:
        members=[dict(member=i,estimated_outcome=value,confidence=.99) for i,_ in enumerate(json.loads(prior['issued_skills_semantic_json']))]
        # Clearly synthetic provenance, only for this non-learning processor
        # diagnostic. No feedback JSONL or training artifact is emitted.
        provenance=dict(high_sha256=recipe['parents']['high']['sha256'],observer_sha256='b'*64,
            trained_groups=['synthetic-train'],calibration_groups=['synthetic-calibration'],target_groups=[h['source_group']],
            feature_cache_sha256='c'*64,fold=0,calibrated=True,temperature=1.,calibration_receipt_sha256='d'*64)
        feedback=dict(sample_id=h['sample_id'],source_group=h['source_group'],control_step=h['control_step'],
                      execution_feedback=feedback_text(prior,members),provenance=provenance)
        projection=planner_projection(candidate,h,target,feedback,recipe['parents']['high']['sha256'])
        raw=raw_observation(candidate,reader,config)
        raw.update(model_projection=projection,action_is_pad=torch.ones(32,dtype=torch.bool),
            action={m['key']:torch.zeros(32,m['raw_shape']) for m in config['raw_shape']['action']})
        processed=processor.preprocess(raw);samples=processed['samples']
        if (samples['memory']!=prior['memory'] or samples['previous_intent']!=prior['previous_intent']
                or samples['known_previous_outcome']!='UNKNOWN'
                or samples['execution_feedback']!=feedback['execution_feedback'] or samples['outcome_supervision_mask']
                or samples['low_action_supervision_mask']):raise ValueError('Processor leaked targets or dropped feedback')
        checks.append(dict(synthetic_feedback=value,pixel_shapes={k:list(v.shape) for k,v in processed['pixel_values'].items()},
                           action_shape=list(processed['action'].shape),outcome_supervision=False,low_supervision=False))
    receipt=dict(status='passed_cpu_processor_contract',scope='synthetic feedback on one real unapproved observation; no training',
        sample_id=candidate['sample_id'],history_sha256=file_sha(a.history/'contexts.jsonl'),
        checks=checks,optimizer_steps=0,generated_training_labels=0)
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt))


if __name__=='__main__':main()
