"""Selective learner-action transport, preserving real boundary observations.

The recorder has no dense proprio. This separate format never pretends it is
a recovery_candidate_v1 expert archive. A preparation bundle is not a launch
ticket and cannot add outcome/planner labels or independent source events.
"""
from dataclasses import asdict
import json
from pathlib import Path
import random
import shutil

import numpy as np

from recovery_calibration_launch import bound, child, read, write_new
from recovery_corpus import digest, file_sha
from skill_aligned_reward import SkillIdentity, SkillReward, skill_measurement
from skill_observation_archive import load_observation


SCHEMA = 'reviewed_native_learner_actions_v1'


def checked_action_window(controls, *, at, identity, policy_version):
    """s[t] predicts applied controls t+1..t+32; no fabricated tail padding."""
    indexed = {r['control_step']:r for r in controls}
    if len(indexed) != len(controls) or type(at) is not int or any(type(r['control_step']) is not int for r in controls):
        raise ValueError('Unique actual control clock required')
    future = [indexed.get(t) for t in range(at+1, at+33)]
    if any(r is None for r in future):
        raise ValueError('No complete real future32 target')
    for i, row in enumerate(future):
        if (row['simulator_apply_ack'] is not True or row['policy_version'] != policy_version
                or row['action_source'] != 'train' or row['skill_reward']['identity'] != identity
                or (i < 31 and (row['skill_reward']['terminated'] or row['skill_reward']['truncated']))):
            raise ValueError('Unacknowledged/cross-policy/cross-intent/terminal learner chunk')
    raw = [r['action_executed_raw23'] for r in future]
    action = np.asarray(raw, dtype=np.float32)
    if action.shape != (32,23) or not np.isfinite(action).all() or action.tolist() != raw:
        raise ValueError('Preserve exact actually applied float32 raw23 controls')
    return action, digest(raw)


def export_native_bundle(config_path, review_path, review_root, output, *, source_commit):
    config_path, review_path, review_root, output = map(Path,(config_path,review_path,review_root,output))
    cfg, review = read(config_path), read(review_path)
    if (output.exists() or review.get('schema') != 'native_rl_action_window_owner_review_v1'
            or review['source_config_sha256'] != file_sha(config_path)
            or review['source_original_split'] != 'train' or review['source_recovery_split'] != 'train'
            or review['approved_action_type'] != 'observed_learner_late_corrective_progress_not_expert'
            or review['actor_input_privileged_state_allowed'] is not False
            or review['outcome_or_planner_label_permission'] is not False):
        raise ValueError('New output and exact action-only owner approval required')
    output.mkdir(parents=True);samples=[];files={}
    def copy(source, relative):
        target=child(output,relative);target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists():raise FileExistsError(target)
        shutil.copyfile(source,target)
        if file_sha(source)!=file_sha(target):raise ValueError('Transport copy changed bytes')
        files[relative]=dict(path=relative,sha256=file_sha(target),bytes=target.stat().st_size)
    copy(config_path,'source-config.json');copy(review_path,'owner-review.json')
    seen=set()
    for approved in review['trajectory_reviews']:
        index_path=child(review_root,approved['index_relative_path'])
        # Locations are selected by the owner declaration, not by a heuristic
        # that discovers whichever trajectories happen to have succeeded.
        if file_sha(index_path)!=approved['index_sha256']:raise ValueError('Changed human-reviewed index')
        index=read(index_path);proof=next(r for r in index['episodes'] if r['job_id']==approved['job_id'])
        audit_path=Path(index['audit_path'])
        if (file_sha(audit_path)!=approved['physical_audit_sha256']
                or index['audit_sha256']!=approved['physical_audit_sha256']
                or index['config_sha256']!=file_sha(config_path)
                or proof['source_policy_sha256']!=approved['source_policy_sha256']):
            raise ValueError('Owner/physical/rollout-policy evidence mismatch')
        directory=Path(proof['original_run']);case=cfg['cases'][proof['case']]
        if case['original_split']!='train' or case['recovery_split']!='train':
            raise ValueError('Protected source cannot supply learner action targets')
        prefix=f"episodes/{approved['job_id']}"
        for name,key in [('start.json','source_start_sha256'),('controls.jsonl','source_controls_sha256'),
                         ('result.json','source_result_sha256')]:
            if file_sha(directory/name)!=proof[key]:raise ValueError('Changed reviewed episode: '+name)
            copy(directory/name,f'{prefix}/{name}')
        copy(index_path,f'{prefix}/review-index.json');copy(audit_path,f'{prefix}/physical-audit.json')
        result=read(directory/'result.json');start=read(directory/'start.json')
        controls=[json.loads(x) for x in (directory/'controls.jsonl').read_text().splitlines()]
        if len(controls)!=result['scored_controls'] or len(controls)!=approved['controls']:
            raise ValueError('Incomplete reviewed trajectory')
        records=[json.loads(x) for x in (directory/'observations/observations.jsonl').read_text().splitlines()]
        for name,key in [('manifest.json','manifest_sha256'),('observations.jsonl','observations_sha256')]:
            if file_sha(directory/'observations'/name)!=result['observation_archive'][key]:
                raise ValueError('Original observation clocks or policy identity changed')
            copy(directory/'observations'/name,f'{prefix}/observations/{name}')
        identity=SkillIdentity(**result['identity']);skill=json.loads(case['semantic_bundle'])[0]
        if (result['job']['id']!=approved['job_id'] or result['job']['seed']!=approved['seed']
                or result['job']['phase']!='train' or result['policy_sha256']!=approved['source_policy_sha256']
                or start['identity']!=asdict(identity) or identity.bundle_sha256!=digest([skill])
                or identity.context_id!=case['context_id'] or identity.task!=case['task']
                or identity.instance!=case['instance_id'] or result['skill_success'] is not True
                or result['whole_task_sr'] is not False):
            raise ValueError('Not the exact same-skill successful TRAIN learner episode')
        reward=SkillReward(identity,skill_measurement(skill,start['initial_physical_evidence']),control_step=case['start_control'])
        for t,row in enumerate(controls,case['start_control']+1):
            if row['control_step']!=t or row['simulator_apply_ack'] is not True:raise ValueError('Missing actual ACK')
            actual=reward.advance(identity,t,skill_measurement(skill,row['physical_evidence']),protected_values={},
                official_terminal=row['official_terminal'],time_limit=row['official_truncated'] or t==result['policy_end_control'])
            if dict(actual,identity=asdict(identity))!=row['skill_reward']:raise ValueError('Reward/physical audit differs')
        if actual['skill_success'] is not True or not actual['terminated']:
            raise ValueError('Physical terminal success does not recompute')
        steps=approved['approved_observation_control_steps'];shas=approved['approved_action_sha256']
        if not steps or len(steps)!=len(set(steps)) or len(steps)!=len(shas):raise ValueError('Explicit unique approved windows required')
        windows={r['observation_control_step']:r for r in proof['windows']}
        native={r['control_step']:r for r in records}
        group=f"{case['task'].replace('_',' ')}:{case['instance_id']}"
        for t,expected in zip(steps,shas):
            if (approved['job_id'],t) in seen:raise ValueError('Repeated approved learner sample')
            seen.add((approved['job_id'],t))
            action,sha=checked_action_window(controls,at=t,identity=asdict(identity),policy_version=result['policy_version'])
            window=windows[t];record=native[t]
            if (sha!=expected or sha!=window['action_sha256'] or window['complete_32_applied_actions'] is not True
                    or record['observation_sha256']!=window['observation_sha256']):
                raise ValueError('Different reviewed observation or future-action alignment')
            load_observation(directory/'observations',record)
            copy(directory/'observations'/record['file'],f"{prefix}/observations/{record['file']}")
            sid=digest([SCHEMA,approved['job_id'],t,record['observation_sha256'],sha])
            samples.append(dict(sample_id=sid,source_group=group,task=case['task'],instance=case['instance_id'],
                source_episode=approved['job_id'],original_split='train',recovery_split='train',
                source_policy_sha256=approved['source_policy_sha256'],source_policy_version=result['policy_version'],
                observation=record,actions_sha256=sha,episode_directory=prefix,
                parent_goal=case['parent_goal'],semantic_bundle=case['semantic_bundle'],
                action_source='reviewed_learner_late_correction_not_expert'))
    groups=sorted({r['source_group'] for r in samples})
    if groups!=sorted(review['original_source_groups']) or len(groups)!=review['independent_source_groups']:
        raise ValueError('Source grouping changed or correlated windows inflated to independent events')
    manifest=dict(schema=SCHEMA,status='prepared_exact_native_action_windows_not_training_admission',
        source_commit=source_commit,owner_review_sha256=file_sha(review_path),source_config_sha256=file_sha(config_path),
        samples=samples,source_groups=groups,files=list(files.values()),
        dense_proprio_fabricated=False,training_admission=False,outcome_or_planner_labels=False,
        scope='TRAIN-only learner supplement to existing same-source event; actual processor and explicit launch gates required')
    write_new(output/'manifest.json',manifest)
    return manifest


class NativeLearnerActionReader:
    """Read-only preparation reader; does not certify an optimizer launch."""
    def __init__(self, root, manifest_sha256, owner_review_sha256):
        self.root=Path(root)
        path=bound(self.root,dict(path='manifest.json',sha256=manifest_sha256));self.manifest=read(path)
        m=self.manifest
        if (m.get('schema')!=SCHEMA or m.get('status')!='prepared_exact_native_action_windows_not_training_admission'
                or m.get('owner_review_sha256')!=owner_review_sha256 or m.get('dense_proprio_fabricated') is not False
                or m.get('training_admission') is not False or m.get('outcome_or_planner_labels') is not False):
            raise ValueError('Unpinned native action preparation bundle')
        bound(self.root,dict(path='owner-review.json',sha256=owner_review_sha256))
        for f in m['files']:
            path=bound(self.root,{k:f[k] for k in ('path','sha256')})
            if path.stat().st_size!=f['bytes']:raise ValueError('Changed transport size')
        self.rows=m['samples'];self.controls={}
        if len({r['sample_id'] for r in self.rows})!=len(self.rows):raise ValueError('Duplicate native sample')

    def read(self,index):
        row=self.rows[index];directory=child(self.root,row['episode_directory'])
        if row['source_episode'] not in self.controls:
            self.controls[row['source_episode']]=[json.loads(x) for x in (directory/'controls.jsonl').read_text().splitlines()]
        result=read(directory/'result.json')
        action,sha=checked_action_window(self.controls[row['source_episode']],at=row['observation']['control_step'],
            identity=result['identity'],policy_version=row['source_policy_version'])
        if sha!=row['actions_sha256']:raise ValueError('Native target bytes changed')
        observation=load_observation(directory/'observations',row['observation'])
        return observation,action,dict(task=row['task'].replace('_',' '),parent_goal=row['parent_goal'],semantic_bundle=row['semantic_bundle'])


def native_low_raw(reader,index,config):
    """Use the inherited low processor; only current state/goal and raw23."""
    import torch
    from recovery_causal_inference import observable_raw
    from g05.data.memlite_stage1_labels import projection
    from g05.utils.memlite_skill_protocol import parse_active_skills_semantic_json,semantic_active_skills_text
    observation,action,goal=reader.read(index)
    raw=observable_raw(observation,goal['task'],config)
    semantic=goal['semantic_bundle']
    segment=dict(parent=goal['parent_goal'],semantic=semantic,parent_supervised=False,
        text=semantic_active_skills_text(parse_active_skills_semantic_json(semantic)))
    raw.update(idx=index,action_is_pad=torch.zeros(32,dtype=torch.bool),
        model_projection=projection(segment,branch='low',task_name=goal['task'],
            previous_intent='None',previous_parent='none',history=[]))
    raw['action']={m['key']:torch.from_numpy(action[:,m['start_index']:m['start_index']+m['raw_shape']].copy())
        for m in config['raw_shape']['action']}
    return raw


def same_event_supplement_rows(base_rows,native_rows):
    """No new event weight: each learner anchor must match one existing event.

    TRAIN source, actual issued semantic skill and parent are all required.
    Several windows or stochastic policies never become independent events.
    """
    if not native_rows:raise ValueError('An explicit supplement cannot be empty')
    output=[]
    for native in native_rows:
        if (native['original_split']!='train' or native['recovery_split']!='train'
                or native.get('action_source')!='reviewed_learner_late_correction_not_expert'):
            raise ValueError('Only original TRAIN learner samples may supplement TRAIN')
        matches=[]
        for item in base_rows:
            row=item['candidate'];actor=row['actor_input']
            if (row['source_group']==native['source_group'] and row['task']==native['task']
                    and row['split']=='train' and item['approval']['pool']=='action'
                    and actor['parent_goal']==native['parent_goal']
                    and json.loads(actor['issued_skills_semantic_json'])==json.loads(native['semantic_bundle'])):
                matches.append(item['approval']['event_id'])
        if len(set(matches))!=1:
            raise ValueError('Require one already approved same-source/skill/parent event; no new event weight')
        output.append(dict(candidate=dict(sample_id=native['sample_id'],source_group=native['source_group'],
            task=native['task'],split='train',native_learner=True),
            approval=dict(pool='action',event_id=matches[0],label=dict(quality='reviewed_learner_late_correction_not_expert'))))
    if len({r['candidate']['sample_id'] for r in base_rows+output})!=len(base_rows)+len(output):
        raise ValueError('Duplicate supplement/base sample identity')
    return output


def validate_native_training_supplement(ticket,recipe,base_rows):
    """Explicit same-source extension; old admission and processor gates stay."""
    spec=recipe.get('L0',{}).get('native_learner_supplement')
    if spec is None:
        if any(k.startswith('native_') for k in ticket['files']):raise ValueError('Undeclared learner supplement')
        return None
    if (ticket['component']!='L0' or set(spec)!={'protocol','manifest','owner_review'}
            or spec['protocol']!='same_existing_source_event_v1'):
        raise ValueError('Explicit action-only same-event supplement required')
    files=ticket['files'];root=Path(recipe['root'])
    for name,key in [('manifest','native_manifest'),('owner_review','native_owner_review')]:
        path=bound(root,spec[name])
        if files.get(key)!=dict(path=str(path),sha256=spec[name]['sha256']):
            raise ValueError('Ticket does not bind the exact learner supplement')
    audit_ref=files.get('native_processor_audit')
    if audit_ref is None or file_sha(audit_ref['path'])!=audit_ref['sha256']:
        raise ValueError('Actual native input/action processor validation required')
    reader=NativeLearnerActionReader(Path(files['native_manifest']['path']).parent,
        files['native_manifest']['sha256'],files['native_owner_review']['sha256'])
    same_event_supplement_rows(base_rows,reader.rows)
    audit=read(audit_ref['path'])
    if (audit.get('schema')!='native_learner_processor_audit_v1' or audit.get('status')!='passed'
            or audit.get('source_commit')!=ticket['source_commit'] or audit.get('optimizer_steps')!=0
            or audit.get('oracle_inputs') is not False or audit.get('stats_sha256')!=recipe['stats_sha256']
            or audit.get('training_processor_checked') is not True
            or audit.get('recipe_sha256')!=ticket['files']['recipe']['sha256']
            or audit.get('admission_sha256')!=ticket['files']['admission']['sha256']
            or audit.get('same_event_schedule_check',{}).get('unchanged_expert_event_rank_order') is not True
            or audit.get('manifest_sha256')!=files['native_manifest']['sha256']
            or audit.get('owner_review_sha256')!=files['native_owner_review']['sha256']
            or [(s['sample_id'],s['actions_sha256']) for s in audit['samples']]!=[
                (s['sample_id'],s['actions_sha256']) for s in reader.rows]
            or any(s['shape']!=[32,27] or s['padding_indices']!=[7,8,17,18]
                   or s.get('train_masks_verified') is not True for s in audit['samples'])):
        raise ValueError('Native processor receipt/model/source/clock drift')
    return reader


class SameEventLearnerSupplement:
    """TRAIN only; DEV stays the untouched original recovery/normal examples."""
    def __init__(self,base,reader):
        if base.split!='train':raise ValueError('Learner actions never enter the fixed DEV set')
        self.base,self.reader,self.config=base,reader,base.config
        self.rows=base.rows+same_event_supplement_rows(base.rows,reader.rows)
        self.processor=None

    def __len__(self):return len(self.rows)

    def __getitem__(self,index):
        if index<len(self.base):return self.base[index]
        import torch
        from g05.utils.training.stage1_model import make_processor
        if self.processor is None:self.processor=make_processor(self.config,True)
        local=index-len(self.base);row=self.rows[index]
        old_random,old_np=random.getstate(),np.random.get_state()
        try:
            with torch.random.fork_rng(devices=[]):
                seed=int(row['candidate']['sample_id'][:8],16)
                torch.manual_seed(seed);random.seed(seed);np.random.seed(seed)
                sample=self.processor.preprocess(native_low_raw(self.reader,local,self.config))
        finally:
            random.setstate(old_random);np.random.set_state(old_np)
        if (sample['action'].shape!=(32,27) or not torch.isfinite(sample['action']).all()
                or tuple(torch.nonzero(sample['action_dim_is_pad']).flatten().tolist())!=(7,8,17,18)
                or sample['action_is_pad'].any()):
            raise ValueError('Invalid exact native action sample')
        sample['source_identity']=dict(pool='reviewed_learner_action',sample=row['candidate']['sample_id'],
            event=row['approval']['event_id'],task=row['candidate']['task'],source_group=row['candidate']['source_group'])
        return sample
