"""Target-free, strictly isolated low-policy diagnostics; no planner or reward inputs."""
import json

from recovery_corpus import digest


def low_prefix(builder, prepared, *, task, parent_goal, semantic_bundle):
    from g05.utils.memlite_skill_protocol import (
        parse_active_skills_semantic_json, semantic_active_skills_text, validate_semantic_parent_goal)
    members = parse_active_skills_semantic_json(semantic_bundle)
    if not members or any(x['verb'] == 'SKILL_UNKNOWN' for x in members):
        raise ValueError('An issued, known skill bundle is required')
    validate_semantic_parent_goal(parent_goal, field='parent_goal')
    if (prepared['_instructions'] != task or builder.num_input_images != 3
            or tuple(builder._image_sizes) != ('head_rgb','left_wrist_rgb','right_wrist_rgb')
            or builder.hardcode_instruction is not None or builder.hardcode_proprio_pad_zeros):
        raise ValueError('Task, single-frame camera order or proprio contract drift')
    if builder.template.count('<EOC>') != 1 or '<action_action' in builder.template:
        raise ValueError('Not a target-free FM template')
    result = dict(template=builder.template.split('<EOC>',1)[0]+'<EOC>', command=task,
        parent_goal=parent_goal, active_skills_semantic_json=semantic_bundle,
        active_skills_text=semantic_active_skills_text(members), memlite_branch='low',
        schema_version=6, memlite_schema_version=6, next_decision='EXECUTE',
        task_complete=False, low_action_supervision_mask=False,
        embodiment=builder.embodiment_type,
        proprio=dict(value=prepared['proprio'],proprio_dim_is_pad=prepared['proprio_dim_is_pad']))
    for i, camera in enumerate(builder._image_sizes): result[f'image{i}'] = builder._image_sizes[camera]
    return result


class FixedSkillSession:
    """One connection, one pinned attempt; metadata never enters model tokens."""
    def __init__(self, cases, models):
        self.cases=cases; self.models=models; self.bound=None; self.next_control=None

    def begin(self, request):
        if self.bound is not None or set(request)!={'op','case','model','episode','seed','manifest_sha256'}:
            raise ValueError('Duplicate reset or malformed session')
        if request['op']!='begin' or request['case'] not in self.cases or request['model'] not in self.models:
            raise ValueError('Unregistered case or model')
        case=self.cases[request['case']]
        if (request['manifest_sha256']!=case['manifest_sha256'] or type(request['seed']) is not int
                or request['seed'] not in case['seeds'] or not isinstance(request['episode'],str)
                or not request['episode']):
            raise ValueError('Unbound seed, source branch or episode')
        self.bound=dict(request,context_id=case['context_id'],bundle_sha256=digest(json.loads(case['semantic_bundle'])))
        self.next_control=case['start_control']
        return dict(model_sha256=self.models[request['model']]['sha256'],case=request['case'],
            episode=request['episode'],context_id=case['context_id'],control_step=self.next_control)

    def check_action(self, request):
        if self.bound is None or set(request)!={'op','episode','control_step','context_id','images','proprio'}:
            raise ValueError('Action requires exact observable whitelist after begin')
        case=self.cases[self.bound['case']]
        if (request['op']!='action' or request['episode']!=self.bound['episode']
                or request['context_id']!=self.bound['context_id'] or type(request['control_step']) is not int
                or request['control_step']!=self.next_control
                or request['control_step']>=case['end_control']):
            raise ValueError('Cross-context, stale, skipped or post-horizon request')
        return case

    def acknowledge_chunk(self):
        self.next_control+=16


def validate_rgb_proprio(images, proprio):
    import numpy as np
    if set(images)!={'head_rgb','left_wrist_rgb','right_wrist_rgb'}:
        raise ValueError('Exactly three official RGB streams required')
    for im in images.values():
        if not isinstance(im,np.ndarray) or im.dtype!=np.uint8 or im.ndim!=3 or im.shape[0]!=3:
            raise ValueError('Expected one CHW uint8 RGB, no history or extra modalities')
        if not 16<=im.shape[1]<=4096 or not 16<=im.shape[2]<=4096:
            raise ValueError('Invalid RGB extent')
    if not isinstance(proprio,np.ndarray) or proprio.shape!=(61,) or not np.isfinite(proprio).all():
        raise ValueError('Exactly 61 finite observed proprio values required')
