"""Bounded, aligned recovery CANDIDATES, never automatically accepted BC data.

Privileged audits are used only by this recorder/detector. No fields from this
module are added to actor or planner inputs. RGB anchors are real observations
at action-chunk boundaries; controls/proprio/rewards are recorded every step.
"""
from collections import deque
from copy import deepcopy
import hashlib
import io
import json
import math
import os
from pathlib import Path
import shutil
import time
import zipfile

import numpy as np
from PIL import Image

CAMERAS = ('head_rgb', 'left_wrist_rgb', 'right_wrist_rgb')


def finite_vector(value, width):
    if hasattr(value, 'detach'): value = value.detach().cpu().numpy()
    result = np.asarray(value, dtype=np.float32).reshape(-1)
    if result.shape != (width,) or not np.isfinite(result).all():
        raise ValueError('Invalid aligned vector, expected width ' + str(width))
    return result.tolist()


def proprio61(observation):
    found = [v for k, v in observation.items() if str(k).endswith('::proprio')]
    if len(found) != 1: raise ValueError('No unique observable proprioception')
    return finite_vector(found[0], 61)


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, path)


class RecoveryDetector:
    """Debounced physical events plus explicitly uncertain stall candidates."""
    def __init__(self, stall_controls=256, *, capture_normal=False):
        self.stall_controls = stall_controls
        self.holds = {}; self.releases = {}; self.stable = set(); self.lost = {}
        self.last_step = 0; self.context = None; self.last_progress = 0
        self.q_peak = 0.0; self.skill_peak = 0.0
        self.q_regression_count = 0; self.regression_q = None
        self.stall_emitted = False; self.last_q = 0.0
        self.entity_bindings = {}
        self.capture_normal = bool(capture_normal)

    def step(self, step, context, audit):
        if step != self.last_step + 1: raise ValueError('Detector control clock is discontinuous')
        self.last_step = step
        q = float(audit['official_q']); skill = float(audit['skill_potential'])
        if not all(math.isfinite(v) and 0 <= v <= 1 for v in (q, skill)):
            raise ValueError('Invalid physical audit score')
        bundle = context['active_skills_semantic_json']
        skills = json.loads(bundle)
        identity = (bundle, context['parent_goal'])
        events = []
        if self.context != identity:
            self.context = identity; self.last_progress = step
            self.skill_peak = skill; self.stall_emitted = False
        if q > self.last_q + 1e-6 or skill > self.skill_peak + 0.01:
            if self.stall_emitted and q > self.last_q + 1e-6:
                events.append(dict(kind='goal_progress_after_stall', step=step,
                                   evidence='official_Q_increased', q_before=self.last_q, q_after=q))
            self.last_progress = step; self.stall_emitted = False
        self.skill_peak = max(self.skill_peak, skill)
        # Audit scope contains task objects AND tools; a broom must not disappear
        # just because the final BDDL predicates mention only the floor.
        scope = audit.get('grasp_states', {})
        def entity(name):
            return name if name in scope else self.entity_bindings.get(name)
        planned_release = {entity(s.get('target')) for s in skills
                           if s['verb'].startswith('PLACE_') or s['verb'] in ('RELEASE','UNGRASP')}
        for obj, arms in audit.get('grasp_states', {}).items():
            if any(v not in ('TRUE','FALSE','UNKNOWN') for v in arms.values()):
                raise ValueError('Invalid tri-state grasp enum')
            definitely_released = bool(arms) and all(v == 'FALSE' for v in arms.values())
            for arm in ('left', 'right'):
                key = (obj, arm)
                self.holds[key] = self.holds.get(key, 0) + 1 if arms.get(arm) == 'TRUE' else 0
            self.releases[obj] = self.releases.get(obj, 0) + 1 if definitely_released else 0
            stable_arms = [arm for arm in ('left', 'right') if self.holds[(obj, arm)] >= 6]
            if stable_arms:
                newly_stable = obj not in self.stable
                self.stable.add(obj)
                if obj in self.lost:
                    lost_step = self.lost.pop(obj)
                    if step - lost_step <= 1024:
                        events.append(dict(kind='regrasp_after_loss', object=obj, step=step,
                                           loss_step=lost_step, evidence='same_object_same_arm_TRUE_6_controls',
                                           arms=stable_arms, recovery_success_label=False))
                elif newly_stable and self.capture_normal:
                    events.append(dict(kind='stable_grasp_observed', object=obj, step=step,
                        arms=stable_arms, evidence='same_object_same_arm_TRUE_6_controls',
                        active_target_matches=any(entity(s.get('target')) == obj for s in skills),
                        action_quality_verified=False))
            if self.releases[obj] >= 6 and obj in self.stable:
                self.stable.remove(obj)
                # Intentional placement/release is not automatically a failure.
                if obj not in planned_release and q <= self.last_q + 1e-6:
                    self.lost[obj] = step
                    events.append(dict(kind='grasp_loss_candidate', object=obj, step=step,
                                       evidence='previous_stable_hold_then_both_arms_FALSE_6_controls',
                                       failure_truth=False))
        if q < self.q_peak - 1e-6:
            self.q_regression_count += 1
            if self.q_regression_count == 3 and self.regression_q is None:
                self.regression_q = self.q_peak
                events.append(dict(kind='goal_regression_candidate', step=step,
                                   q_peak=self.q_peak, q_after=q, failure_truth=False))
        else:
            self.q_regression_count = 0
            if self.regression_q is not None and q >= self.regression_q - 1e-6:
                events.append(dict(kind='goal_progress_restored', step=step,
                                   restored_q=q, evidence='official_Q_restored', task_success=bool(audit['official_success'])))
                self.regression_q = None
        if step - self.last_progress >= self.stall_controls and not self.stall_emitted:
            events.append(dict(kind='stalled_uncertain', step=step,
                               unchanged_context_controls=step-self.last_progress,
                               evidence='unchanged_skill_and_no_significant_reward_potential_progress', failure_truth=False))
            self.stall_emitted = True
        self.q_peak = max(self.q_peak, q); self.last_q = q
        return events


class RecoveryRecorder:
    def __init__(self, root, num_envs, settings, provenance):
        self.root = Path(root); self.root.mkdir(parents=True, exist_ok=True)
        self.settings = settings; self.num_envs = num_envs
        self.provenance = deepcopy(provenance)
        self.quota = int(settings['recorder_quota_gib'] * 2**30)
        self.reserve = int(settings['disk_reserve_gib'] * 2**30)
        self.used = sum(p.stat().st_size for p in self.root.glob('*.zip'))
        self.counts = dict(saved=0, quota_rejected=0, episode_cap_rejected=0)
        self.slots = []; self.started = False
        self.publish_status()

    def publish_status(self):
        atomic_json(self.root/'recorder_status.json', dict(updated=time.time(), used_bytes=self.used,
                    quota_bytes=self.quota, counts=self.counts, bc_eligible=False,
                    owner_run=self.provenance.get('run'), schema='recovery_candidate_v1'))

    def begin(self, metadata):
        if self.started: self.close('reset_boundary')
        if len(metadata) != self.num_envs: raise ValueError('Episode metadata mismatch')
        self.slots = []
        for episode in metadata:
            required = {'episode_id','task','instance_id','split','policy_seed','source_commit','resume_checkpoint_sha256'}
            if not required <= episode.keys() or episode['split'] != 'train':
                raise ValueError('Unbound or non-TRAIN candidate episode')
            self.slots.append(dict(metadata=deepcopy(episode),
                ring=deque(maxlen=self.settings['recovery_pre_controls']), active=None,
                detector=RecoveryDetector(self.settings['recovery_stall_controls'], capture_normal=True),
                label_bindings=None, normal_clips=0,
                anchor=None, context=None, chunk=None, pending=None, step=0, clips=0, stalled_clips=0))
        self.started = True

    def bind_scope(self, env, named_scope):
        """Exact live object names, recorder-only; never sent through actor RPC."""
        slot = self.slots[env]
        if slot['step'] != 0 or slot['label_bindings'] is not None:
            raise ValueError('Bind live scope once, before executing any control')
        if not isinstance(named_scope, dict) or not named_scope:
            raise ValueError('Missing live task scope')
        available = {key: name for key, name in named_scope.items() if name is not None}
        if (any(not isinstance(k, str) or not isinstance(v, str) or not v for k, v in available.items())
                or len(set(available.values())) != len(available)):
            raise ValueError('Ambiguous live task object binding')
        mapping = {name: key for key, name in available.items()}
        slot['label_bindings'] = dict(schema='live_task_entity_bindings_v1', by_asset_name=mapping,
            unavailable_entities=sorted(set(named_scope)-set(available)),
            episode_id=slot['metadata']['episode_id'], observed_control_step=0, actor_input_permitted=False)
        slot['detector'].entity_bindings = mapping

    def on_chunk(self, env, observation, context, experience_id, policy_update):
        if not self.started: return
        slot = self.slots[env]
        encoded = {}; dimensions = {}
        if set(observation['images']) != set(CAMERAS): raise ValueError('Three RGB cameras required')
        for camera in CAMERAS:
            value = np.asarray(observation['images'][camera])
            if value.ndim != 3 or value.shape[0] != 3 or value.dtype != np.uint8:
                raise ValueError('Recorder expects original uint8 CHW RGB')
            buffer = io.BytesIO()
            Image.fromarray(value.transpose(1,2,0)).save(buffer, format='JPEG', quality=90, subsampling=0)
            encoded[camera] = buffer.getvalue(); dimensions[camera] = list(value.shape)
        # The anchor is at s_t, before control t -> t+1; never claim it was
        # captured at every later control sharing this image reference.
        slot['anchor'] = dict(control_step=slot['step'], images=encoded, dimensions=dimensions,
                              state={k:np.asarray(v).tolist() for k,v in observation['state'].items()})
        slot['context'] = deepcopy(context)
        slot['chunk'] = dict(experience_id=int(experience_id), policy_update=int(policy_update),
                             chunk_start_control_step=slot['step'])

    def before_action(self, env, observation, raw23):
        if not self.started: return
        slot = self.slots[env]
        if slot['pending'] is not None or slot['anchor'] is None:
            raise ValueError('Misaligned action or missing RGB anchor')
        slot['pending'] = dict(control_step=slot['step'], proprio_before=proprio61(observation),
            action_executed_raw23=finite_vector(raw23,23), context=deepcopy(slot['context']),
            **slot['chunk'], _anchor=slot['anchor'])

    def confirm_applied(self,env,raw23):
        """Called only after the real evaluator _apply_actions returned."""
        if not self.started:return
        record=self.slots[env]['pending']
        if record is None or finite_vector(raw23,23)!=record['action_executed_raw23']:
            raise ValueError('Recorded action differs from actual simulator command')
        record['simulator_apply_ack']=True

    def observe(self, env, post_observation, reward, audit, terminated, truncated):
        if not self.started: return
        slot = self.slots[env]; record = slot['pending']
        if record is None: raise ValueError('No action before reward observation')
        if not record.get('simulator_apply_ack'):raise ValueError('Simulator did not acknowledge recorded control')
        slot['step'] += 1
        if record['control_step'] != slot['step']-1: raise ValueError('Control clock mismatch')
        record.update(proprio_after=proprio61(post_observation), reward=float(reward),
                      physical_audit=deepcopy(audit), terminated=bool(terminated), truncated=bool(truncated))
        slot['pending'] = None
        events = slot['detector'].step(slot['step'], slot['context'], audit)
        terminal = bool(terminated or truncated)
        if terminal:
            events.append(dict(kind='official_success' if audit['official_success'] else 'episode_ended_without_success',
                               step=slot['step'], terminated=bool(terminated), truncated=bool(truncated),
                               official_success=bool(audit['official_success']), official_q=float(audit['official_q']),
                               recovery_success_label=False))
        active = slot['active']
        if active is not None: active['records'].append(record)
        eligible = [e for e in events
                    if (e['kind'] != 'stalled_uncertain' or slot['stalled_clips'] < 2)
                    and (e['kind'] != 'stable_grasp_observed' or slot['normal_clips'] < 2)]
        if eligible and active is None:
            if slot['clips'] >= self.settings['recovery_max_clips_per_episode']:
                self.counts['episode_cap_rejected'] += 1
            else:
                active = dict(records=list(slot['ring'])+[record], events=[],
                              trigger_step=slot['step'], end_step=slot['step']+self.settings['recovery_post_controls'])
                slot['active'] = active
                slot['clips'] += 1
                if any(e['kind']=='stalled_uncertain' for e in eligible): slot['stalled_clips'] += 1
                if any(e['kind']=='stable_grasp_observed' for e in eligible): slot['normal_clips'] += 1
        if active is not None:
            active['events'].extend(events)
            # Related follow-up evidence may extend a clip, but boundedly.
            if any(e['kind'] in ('regrasp_after_loss','goal_progress_restored','goal_progress_after_stall') for e in events):
                active['end_step'] = max(active['end_step'],slot['step']+64)
            if (terminal or slot['step'] >= active['end_step'] or
                    len(active['records']) >= self.settings['recovery_max_controls']):
                self.publish(env,'episode_terminal' if terminal else 'window_complete')
        slot['ring'].append(record)

    def publish(self, env, reason):
        slot = self.slots[env]; active = slot['active']
        if active is None: return
        records = active['records']; anchors = {}
        for row in records: anchors[row['_anchor']['control_step']] = row['_anchor']
        identity = slot['metadata']['episode_id'] + ':' + str(active['trigger_step'])
        clip_id = hashlib.sha256(identity.encode()).hexdigest()[:24]
        target = self.root/(clip_id+'.zip'); temporary = self.root/(clip_id+'.pending.zip')
        if target.exists() or temporary.exists(): raise RuntimeError('Refuse to overwrite recovery evidence')
        rows = [{k:v for k,v in row.items() if k!='_anchor'} |
                {'rgb_anchor_control_step':row['_anchor']['control_step']} for row in records]
        transitions = ''.join(json.dumps(r,allow_nan=False)+'\n' for r in rows).encode()
        estimated = len(transitions)+sum(len(b) for a in anchors.values() for b in a['images'].values())+65536
        if self.used+estimated>self.quota or shutil.disk_usage(self.root).free-estimated<self.reserve:
            self.counts['quota_rejected'] += 1; slot['active'] = None; self.publish_status(); return
        manifest = dict(schema='recovery_candidate_v1', clip_id=clip_id, label_kind='on_policy_candidate',
            bc_eligible=False, human_review='pending', synthetic_expert=False,
            episode=slot['metadata'], provenance=self.provenance, events=active['events'],
            label_only_entity_bindings=deepcopy(slot['label_bindings']),
            start_control_step=rows[0]['control_step'], end_control_step=rows[-1]['control_step']+1,
            stop_reason=reason, dense_controls=True, control_hz=30,
            rgb_sampling='real pre-action observations at 16-control chunk boundaries',
            image_codec='JPEG quality90 subsampling0 original resolution',
            executed_action_schema=['base_qvel:3','trunk_qpos:4','left_arm:7','left_gripper:1','right_arm:7','right_gripper:1'],
            rgb_anchors={str(t):dict(dimensions=a['dimensions'],state=a['state'],
                        sha256={k:hashlib.sha256(v).hexdigest() for k,v in a['images'].items()}) for t,a in anchors.items()},
            transitions_sha256=hashlib.sha256(transitions).hexdigest(),
            limitation='Physical recovery events are candidate evidence, not a human-approved corrective action target or full task success.')
        with zipfile.ZipFile(temporary,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=3) as output:
            output.writestr('manifest.json',json.dumps(manifest,indent=2,allow_nan=False))
            output.writestr('transitions.jsonl',transitions)
            for step,anchor in anchors.items():
                for camera,data in anchor['images'].items():
                    output.writestr(f'rgb/{step:08d}/{camera}.jpg',data,compress_type=zipfile.ZIP_STORED)
        with temporary.open('rb') as stream: os.fsync(stream.fileno())
        digest=hashlib.sha256(temporary.read_bytes()).hexdigest()
        os.replace(temporary,target)
        descriptor=os.open(self.root,os.O_DIRECTORY)
        try: os.fsync(descriptor)
        finally: os.close(descriptor)
        size=target.stat().st_size;self.used+=size;self.counts['saved']+=1
        with (self.root/'index.jsonl').open('a') as stream:
            stream.write(json.dumps(dict(path=str(target),sha256=digest,bytes=size,episode=slot['metadata'],
                events=active['events'],bc_eligible=False,human_review='pending'))+'\n')
            stream.flush();os.fsync(stream.fileno())
        slot['active']=None;self.publish_status()

    def close(self, reason='collector_exit'):
        if not self.started:return
        for index,slot in enumerate(self.slots):
            if slot['pending'] is not None:
                # The simulator did not acknowledge this control. Never save
                # it as an executed transition or fabricate an after-state.
                slot['pending']=None
            self.publish(index,reason)
        self.started=False
