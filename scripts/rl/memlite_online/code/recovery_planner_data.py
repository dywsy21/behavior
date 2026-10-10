"""H1 inputs from pre-decision observations, targets from verified execution."""
import json

from recovery_admission import local_file
from recovery_corpus import digest, file_sha
from recovery_features import validate_prediction_provenance
from recovery_sft_data import CandidateArchiveReader, raw_observation, require_training_pool


def planner_projection(candidate,history,target,feedback,high_sha256,*,uncertainty_protocol='raw_observer_confidence_v1'):
    from g05.utils.memlite_skill_protocol import (semantic_active_skills_text,parse_active_skills_semantic_json,
        append_b_memory_idempotent,V6_MODEL_PROJECTION_FIELDS)
    sid=candidate['sample_id'];prior=history['predecision'];current=history['observable']
    if (prior is None or history['sample_id']!=sid or history['source_episode']!=candidate['source_episode']
            or history['source_group']!=candidate['source_group'] or history['control_step']!=candidate['control_step']
            or current['control_step']!=candidate['control_step'] or prior['control_step']>=candidate['control_step']):
        raise ValueError('H1 requires a real decision with complete prior context')
    required={'schema','sample_id','source_episode','control_step','parent_goal','active_skills_semantic_json','decision','memory_update'}
    if (set(target)!=required or target['schema']!='recovery_verified_plan_v1' or target['sample_id']!=sid
            or target['source_episode']!=candidate['source_episode'] or target['control_step']!=candidate['control_step']
            or target['parent_goal']!=current['parent_goal']
            or target['active_skills_semantic_json']!=candidate['actor_input']['issued_skills_semantic_json']
            or target['memory_update']!=current['memory'] or target['decision']!=current['issued_decision']):
        raise ValueError('Planner target is not the actually verified issued command')
    if (feedback['sample_id']!=sid or feedback['source_group']!=candidate['source_group']
            or feedback['control_step']!=candidate['control_step']): raise ValueError('Shifted predicted feedback')
    validate_prediction_provenance(feedback['provenance'],group=candidate['source_group'],high_sha256=high_sha256)
    value=json.loads(feedback['execution_feedback'])
    fields={'schema','same_intent_controls','same_intent_planner_refreshes','attempt_index','estimated_member_outcomes',
            'attempt_count_scope','estimated_bundle_outcome','source','stalled_is_not_failed'}
    if (set(value)!=fields or value['schema']!='causal_execution_feedback_v1'
            or value['source']!='observable_counter_and_learned_observer' or value['stalled_is_not_failed'] is not True
            or value['same_intent_controls']!=prior['served_controls']
            or value['same_intent_planner_refreshes']!=prior['repeated_planning_count']
            or value['attempt_index']!=prior['attempt_number']): raise ValueError('Noncausal/unwhitelisted feedback')
    members=parse_active_skills_semantic_json(prior['issued_skills_semantic_json'])
    predictions=value['estimated_member_outcomes']
    if len(predictions)!=len(members): raise ValueError('Wrong assessed bundle')
    from recovery_observer_training import OUTCOMES
    for i,r in enumerate(predictions):
        if (set(r)!={'member','estimated_outcome','confidence'} or r['member']!=i
                or r['estimated_outcome'] not in OUTCOMES or not 0<=r['confidence']<=1
                or (not feedback['provenance']['calibrated'] and r['estimated_outcome']!='UNKNOWN')):
            raise ValueError('Uncalibrated/oracle member feedback')
    from recovery_observer_training import feedback_text
    if json.loads(feedback_text(prior,predictions))!=value: raise ValueError('Inconsistent aggregate feedback')
    if uncertainty_protocol not in ('raw_observer_confidence_v1','unready_zero_confidence_v1'):
        raise ValueError('Unknown planner uncertainty input contract')
    execution_feedback=feedback['execution_feedback']
    if uncertainty_protocol=='unready_zero_confidence_v1' and not feedback['provenance']['calibrated']:
        # Preserve the real OOF provenance and UNKNOWN outcomes. An unready
        # head supplies no usable confidence, just as no observer on original
        # demonstrations supplies none. This is not an oracle replacement.
        execution_feedback=feedback_text(prior,[dict(r,confidence=0.) for r in predictions])
    task=candidate['task'].replace('_',' ')
    if target['memory_update']!=append_b_memory_idempotent(prior['memory'],prior['previous_intent'],task_name=task):
        raise ValueError('Target memory uses unobserved future command')
    semantic=target['active_skills_semantic_json']
    result=dict(schema_version=6,memlite_branch='high',task_name=task,parent_goal=target['parent_goal'],
        target_parent_goal=target['parent_goal'],previous_parent_goal=prior['parent_goal'],
        previous_intent=prior['previous_intent'],memory=prior['memory'],
        # An observer's calibrated estimate is still not privileged known
        # truth. Keep it in the explicitly estimated feedback channel. This
        # also preserves the frozen high_planner_only UNKNOWN input contract.
        known_previous_outcome='UNKNOWN',execution_feedback=execution_feedback,
        active_skills_semantic_json=semantic,active_skills_text=semantic_active_skills_text(parse_active_skills_semantic_json(semantic)),
        next_decision=target['decision'],memory_update=target['memory_update'],task_complete=False,
        outcome_target='UNKNOWN',outcome_supervision_mask=False,parent_goal_supervision_mask=True,low_action_supervision_mask=False)
    if set(result)!=set(V6_MODEL_PROJECTION_FIELDS): raise ValueError('Projection schema drift')
    return result


class VerifiedRecoveryPlannerDataset:
    def __init__(self,release,raw_root,inventory,config,*,split,admission_sha256,history,feedback,evidence_root,high_sha256,
                 uncertainty_protocol='raw_observer_confidence_v1'):
        self.receipt,rows=require_training_pool(release,'planner',admission_sha256)
        if digest(inventory)!=self.receipt['inventory_sha256']: raise ValueError('Wrong corpus')
        self.rows=[r for r in rows if r['candidate']['split']==split]
        if not self.rows: raise ValueError('No approved planner split')
        self.reader=CandidateArchiveReader(raw_root,inventory);self.config=config;self.processor=None
        self.history={r['sample_id']:r for r in history};self.feedback={r['sample_id']:r for r in feedback}
        if len(self.history)!=len(history) or len(self.feedback)!=len(feedback): raise ValueError('Duplicate context/feedback')
        self.projections=[]
        for item in self.rows:
            label=item['approval']['label'];path=local_file(evidence_root,label['verified_plan_path'])
            if file_sha(path)!=label['verified_plan_sha256']: raise ValueError('Modified verified target')
            row=item['candidate'];sid=row['sample_id']
            self.projections.append(planner_projection(row,self.history[sid],json.loads(path.read_text()),self.feedback[sid],high_sha256,
                uncertainty_protocol=uncertainty_protocol))

    def __len__(self): return len(self.rows)

    def __getitem__(self,index):
        import torch
        from g05.utils.training.stage1_model import make_processor
        if self.processor is None: self.processor=make_processor(self.config,False)
        row=self.rows[index]['candidate'];raw=raw_observation(row,self.reader,self.config)
        # High CE has no continuous target. Zero placeholders keep the shared
        # collator shape but are fully masked and never supervise the low FM.
        raw.update(model_projection=self.projections[index],action_is_pad=torch.ones(32,dtype=torch.bool),
            action={m['key']:torch.zeros(32,m['raw_shape']) for m in self.config['raw_shape']['action']})
        sample=self.processor.preprocess(raw)
        sample['source_identity']=dict(pool='verified_planner',sample=row['sample_id'],event=self.rows[index]['approval']['event_id'],
                                       source_group=row['source_group'],task=row['task'])
        return sample
