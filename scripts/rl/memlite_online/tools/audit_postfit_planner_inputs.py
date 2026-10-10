"""Actual high-layer processor audit on every fixed-observer feedback row."""
import argparse
from collections import Counter
import json
from pathlib import Path
import subprocess
import sys

REPO=Path(__file__).resolve().parents[4]
sys.path[:0]=[str(REPO/'src'),str(REPO/'scripts/rl/memlite_online/code')]
from recovery_calibration_launch import bound,read,write_new
from recovery_corpus import file_sha
from recovery_postfit_feedback import validate_training_release
from recovery_planner_data import VerifiedRecoveryPlannerDataset


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('recipe','corpus','feedback','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();recipe=read(a.recipe);root=Path(recipe['root']);corpus=a.corpus
    if a.output.exists() or subprocess.check_output(['git','status','--porcelain'],cwd=REPO,text=True).strip():
        raise ValueError('New audit and clean frozen code required')
    paths=dict(admission=corpus/'admission/admission.json',inventory=corpus/'audit/inventory.json',
        history=corpus/'history/contexts.jsonl',feedback=a.feedback/'feedback.jsonl',
        feedback_receipt=a.feedback/'feedback_receipt.json',observer_prediction_trace=a.feedback/'prediction-trace.json',
        observer_exposure=bound(root,recipe['H1']['exposure']),observer_calibration=bound(root,recipe['H1']['calibration']))
    ticket=dict(admission=str(corpus/'admission'),files={k:dict(path=str(v),sha256=file_sha(v)) for k,v in paths.items()})
    validation=validate_training_release(ticket,recipe)
    import torch
    from g05.utils.training.stage1_model import configuration
    torch.set_num_threads(2)
    config=configuration(root,'high',read(root/recipe['expert_release']/'manifest.json')['task_names'])
    histories=[json.loads(x) for x in paths['history'].read_text().splitlines()]
    feedback=[json.loads(x) for x in paths['feedback'].read_text().splitlines()]
    by_id={r['sample_id']:r for r in feedback};checks=[]
    for split in ('train','dev'):
        ds=VerifiedRecoveryPlannerDataset(corpus/'admission',corpus/'raw',read(paths['inventory']),config,
            split=split,admission_sha256=file_sha(paths['admission']),history=histories,feedback=feedback,
            evidence_root=root,high_sha256=recipe['parents']['high']['sha256'],
            uncertainty_protocol=recipe['H1']['uncertainty_protocol'])
        for index,item in enumerate(ds.rows):
            sid=item['candidate']['sample_id'];sample=ds[index];s=sample['samples'];prior=ds.history[sid]['predecision']
            if (s['execution_feedback']!=by_id[sid]['execution_feedback'] or s['known_previous_outcome']!='UNKNOWN'
                    or s['memory']!=prior['memory'] or s['previous_intent']!=prior['previous_intent']
                    or s['outcome_supervision_mask'] or s['low_action_supervision_mask']
                    or not sample['action_is_pad'].all() or sample['action'].shape!=(32,27)
                    or tuple(torch.nonzero(sample['action_dim_is_pad']).flatten().tolist())!=(7,8,17,18)
                    or not all(torch.isfinite(v).all() for v in sample['pixel_values'].values())):
                raise ValueError('Processor dropped actual feedback, leaked future memory or changed action masking')
            checks.append(dict(component='H1',split=split,sample_id=sid,source_group=item['candidate']['source_group'],
                feedback_sha256=__import__('hashlib').sha256(s['execution_feedback'].encode()).hexdigest(),
                estimated_outcome=json.loads(s['execution_feedback'])['estimated_bundle_outcome'],
                known_previous_outcome='UNKNOWN',supervises_action=False,supervises_outcome=False))
    out=dict(schema='accepted_recovery_processor_audit_v1',status='passed',
        admission_sha256=file_sha(paths['admission']),feedback_sha256=file_sha(paths['feedback']),
        recipe_sha256=file_sha(a.recipe),source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
        validation=validation,counts=dict(Counter(r['component']+'_'+r['split'] for r in checks)),checks=checks,
        optimizer_steps=0,generated_training_feedback=False,oracle_inputs=False)
    write_new(a.output,out);print(json.dumps({k:v for k,v in out.items() if k!='checks'},indent=2))


if __name__=='__main__':main()
