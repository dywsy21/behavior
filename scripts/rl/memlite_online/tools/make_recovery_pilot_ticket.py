"""Pin a user-authorized reviewed-data pilot. Does not launch any job."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
from recovery_corpus import file_sha
from recovery_sft_data import require_training_pool
from recovery_train_contract import validate_launch


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('recipe','corpus','processor-audit','output'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--component',choices=('H0','H1','L0'),required=True)
    p.add_argument('--feature-cache',type=Path);p.add_argument('--feature-audit',type=Path)
    p.add_argument('--feedback-directory',type=Path)
    p.add_argument('--expert-feedback-audit',type=Path)
    p.add_argument('--expert-feedback-human-review',type=Path)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    recipe=json.loads(a.recipe.read_text());root=Path(recipe['root'])
    if recipe.get('training_authorized_by_this_file') is not True or not recipe.get('authorization'):
        raise ValueError('Recipe has no explicit pilot authority')
    corpus=a.corpus.resolve();admission=corpus/'admission';sha=file_sha(admission/'admission.json')
    require_training_pool(admission,recipe[a.component]['pool'],sha)
    check=json.loads(a.processor_audit.read_text())
    if check['status']!='passed' or check['admission_sha256']!=sha or check['optimizer_steps']!=0:
        raise ValueError('Missing actual full processor validation')
    files=dict(recipe=a.recipe.resolve(),admission=admission/'admission.json',
        inventory=corpus/'audit/inventory.json',history=corpus/'history/contexts.jsonl',
        processor_audit=a.processor_audit.resolve(),union_receipt=corpus/'receipt.json',
        expert_index=root/'runs/recovery_sft_preparation_20261009/expert-anchor-index-v1.json',
        high_stats=root/'models/memlite-b-final-20260910/B-dataset-stats.json')
    if a.component=='H0':
        if a.feature_cache is None or a.feature_audit is None:raise ValueError('Need real accepted features and reload audit')
        check=json.loads(a.feature_audit.read_text())
        if check['status']!='passed' or check['admission_sha256']!=sha or check['optimizer_steps']!=0:raise ValueError('Unverified actual features')
        files.update(features=a.feature_cache/'features.pt',feature_receipt=a.feature_cache/'receipt.json',feature_audit=a.feature_audit)
    if a.component=='H1':
        if a.feedback_directory is None:raise ValueError('Need actual completed observer OOF artifacts')
        receipt=json.loads((a.feedback_directory/'feedback_receipt.json').read_text())
        fixed=recipe['H1'].get('feedback')=='frozen_source_disjoint_observer_v1'
        expected=('frozen_source_disjoint_feedback_completed' if fixed else 'trained_calibration_and_oof_completed')
        if (receipt['diagnostic_only'] or receipt['status']!=expected
                or receipt['high_sha256']!=recipe['parents']['high']['sha256']):raise ValueError('Invalid observer feedback')
        files.update(feedback=a.feedback_directory/'feedback.jsonl',feedback_receipt=a.feedback_directory/'feedback_receipt.json')
        if fixed:
            from recovery_calibration_launch import bound
            files.update(observer_exposure=bound(root,recipe['H1']['exposure']),
                observer_calibration=bound(root,recipe['H1']['calibration']),
                observer_prediction_trace=a.feedback_directory/'prediction-trace.json')
        if recipe['H1'].get('original_feedback')=='causal_expert_unknown_v1':
            if a.expert_feedback_audit is None or a.expert_feedback_human_review is None:
                raise ValueError('New original feedback requires processor audit and human review')
            files.update(expert_feedback_audit=a.expert_feedback_audit,
                         expert_feedback_human_review=a.expert_feedback_human_review)
    small=a.component=='H0';pilot=recipe['pilot']
    ticket=dict(schema='recovery_sft_launch_v1',component=a.component,
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        formal_training_authorized=True,engineering_smoke=False,
        authorization=recipe['authorization'],maximum_updates=recipe['maximum_updates_per_line'],
        event_passes=pilot['event_passes'],wall_seconds=recipe['maximum_wall_seconds_per_line'],
        world_size=1 if small else pilot['H1_L0_world_size'],micro_batch=1,
        global_batch=pilot['H0_global_batch'] if small else pilot['H1_L0_global_batch'],
        checkpoint_every_updates=1 if small else pilot['checkpoint_every_updates'],
        admission=str(admission),raw_root=str(corpus/'raw'),evidence_root=str(root),
        files={key:dict(path=str(path.resolve()),sha256=file_sha(path)) for key,path in files.items()},
        scope='Small reviewed-data SFT; no automatic runtime readiness, broad RL, or full-task success claim')
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(ticket,indent=2)+'\n')
    validate_launch(a.output,a.component)
    print(json.dumps(dict(output=str(a.output),sha256=file_sha(a.output),component=a.component,authorized=True)))


if __name__=='__main__':main()
