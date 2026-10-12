"""Frozen causal MEM-Lite inference; shared weights, never shared live memory."""
from copy import deepcopy
from dataclasses import asdict
import time

import numpy as np

from lifecycle import validate_episode_metadata
from full_episode_session import FullEpisodeSession
from g05.utils.memlite_causal_session import CausalSessionIdentity
from recovery_causal_inference import observable_raw
from g05.utils.memlite_causal_feedback import single_frame_planner_prefix
from skill_training_protocol import actor_observation, observation_hash


def plan_many(planner, wrappers, observations, adapter):
    """One actual VLM generation, staged for ALL rows before any commit.

    Pixel/token rows and return rows retain the same order. KV cache slicing is
    owned by the existing sparse-cache context, not a shared per-episode cache.
    """
    import torch
    from g05.models.g05.inferencer import PolicyInferencer
    requests, prepared_rows = [], []
    try:
        for wrapper, observation in zip(wrappers, observations, strict=True):
            session = wrapper.session
            if session.identity.models != planner.models:
                raise ValueError('Foreign planner/low/observer binding')
            token, causal = session.begin_planning(session.identity, session.feedback.control_step)
            requests.append((session, token, causal))
            prepared = planner.processor._process_tensors(
                observable_raw(observation, session.identity.task, planner.config))
            prepared['samples'] = single_frame_planner_prefix(planner.processor.samples_builder, prepared, causal)
            prepared_rows.append(prepared)
        batch = PolicyInferencer._collate(prepared_rows, padding_input_id=planner.processor.pad_token_id)
        pixels = {key:value.to(planner.device) for key,value in batch['pixel_values'].items()}
        with torch.inference_mode(), torch.autocast('cuda',dtype=torch.bfloat16), planner.cache_context():
            result = planner.policy.generate_high_level(batch['samples'], pixels, temperature=0.)
        events = result['planner_events']
        if len(events) != len(requests):
            raise ValueError('Planner returned the wrong batch size')
        for (session, token, _causal), event, observation in zip(requests, events, observations, strict=True):
            goal = session.stage(session.identity, token, event)
            adapter.prepare(observation, goal)
        output = []
        for (session, token, causal), event in zip(requests, events, strict=True):
            goal = session.commit(session.identity, token)
            output.append(dict(goal=goal, event=event, causal_input=causal, reused=False,
                control_step=session.feedback.control_step, revision=session.revision,
                physical_success_asserted=False, observer_feedback_mode=session.observer_feedback_mode))
        return output
    except BaseException:
        for session, token, _causal in requests:
            if session.request is not None and session.request['token'] == token:
                session.discard(session.identity, token)
        raise


def sample_many(policy, adapter, observations, goals, steps, *, seed=17):
    """Same FP32 Euler FM10; one prefill/velocity batch, per-row RNG clocks.

    Ending another environment cannot alter this row's initial noise. Preserve
    the previous single-episode seed+1000003*chunk clock, including seed17.
    """
    import torch
    from types import SimpleNamespace
    from direct_a4_flow import A4DirectPPO
    from g05.models.g05.inferencer import PolicyInferencer
    rows = [adapter.prepare(obs, goal) for obs,goal in zip(observations,goals,strict=True)]
    batch = adapter.collate([r[0] for r in rows])
    fm = policy.model.fm_helper
    if (fm.num_inference_steps, fm.horizon_steps, fm.action_dim) != (10,32,27):
        raise ValueError('Fixed low FM10/32/27 contract changed')
    pad = torch.zeros(27,dtype=torch.bool,device='cuda'); pad[[7,8,17,18]] = True
    owner = SimpleNamespace(policy=policy, fm=fm)
    with torch.inference_mode(), adapter._branch_context():
        state = policy.prefill(batch['samples'],batch['pixel_values'])
        noise = []
        for step, sample in zip(steps,batch['samples'],strict=True):
            with torch.random.fork_rng(devices=[0]):
                torch.manual_seed(seed + 1000003*(step//16))
                dummy = torch.zeros(1,32,27,device='cuda',dtype=torch.float32)
                noise.append(fm._sample_noise(dummy,torch.float32,[sample.get('embodiment')]))
        action = torch.cat(noise); action[...,pad] = 0
        velocity = A4DirectPPO._velocity_fn(owner,state)
        pi = str(fm.time_convention) == 'pi_convention'
        for i in range(10):
            times = torch.full((len(rows),),(10-i)/10 if pi else i/10,device='cuda',dtype=torch.float32)
            action = action + (-1. if pi else 1.)/10*velocity(action,times)
            action[...,pad] = 0
    cpu = {k:v.detach().cpu() for k,v in batch.items() if isinstance(v,torch.Tensor)}
    cpu.update(action=action.detach().cpu(),selected_action_source='fm')
    results = []
    order = [('base_qvel',3),('trunk_qpos',4),('left_arm',7),('left_gripper',1),('right_arm',7),('right_gripper',1)]
    for i, (_prepared, anchor) in enumerate(rows):
        groups = PolicyInferencer._postprocess_single(cpu,i,adapter.processor,raw_state_anchor=anchor)
        results.append(torch.cat([torch.as_tensor(groups[k]).reshape(32,w) for k,w in order],-1)[:16].float().numpy())
    actions = np.stack(results)
    if actions.shape != (len(rows),16,23) or not np.isfinite(actions).all():
        raise ValueError('Nonfinite or malformed native low action batch')
    return actions


class CausalBatchRuntime:
    def __init__(self, *, models, calibration, planner, adapter, observer_factory, sample, emit, run_id):
        self.models, self.calibration = models, calibration
        self.planner, self.adapter = planner, adapter
        self.observer_factory, self.sample, self.emit = observer_factory, sample, emit
        self.run_id, self.sessions, self.seen = run_id, {}, set()
        self.metadata = []
        self.requests = 0

    def begin(self, metadata, mode):
        metadata = validate_episode_metadata(metadata,allowed_mode=mode)
        if any(not w.session.closed for w in self.sessions.values()):
            raise ValueError('Reset cannot discard live/unacknowledged episodes')
        if any(r['episode'] in self.seen for r in metadata):
            raise ValueError('Implicit episode retry forbidden')
        fresh = {}
        for slot,row in enumerate(metadata):
            identity = CausalSessionIdentity(self.run_id+'-slot'+str(slot),row['task'].replace('_',' '),row['instance_id'],row['episode'],self.models)
            def emit(kind, value, *, identity=identity, slot=slot):
                self.emit(kind,dict(value,slot=slot,identity=asdict(identity)))
            fresh[slot] = FullEpisodeSession(identity,calibration=self.calibration,planner=self.planner,
                low=self.adapter,observer_factory=self.observer_factory,emit=emit)
        if (len({id(w.session) for w in fresh.values()}) != len(fresh)
                or len({id(w.observer) for w in fresh.values()}) != len(fresh)):
            raise ValueError('Live planner or observer sessions are aliased')
        self.sessions, self.metadata = fresh, metadata
        self.seen.update(r['episode'] for r in metadata)

    def acknowledge(self, acknowledgements):
        # Every row has a complete identity-bound token and the actual float32
        # actions returned by the successful official stepping call.
        for slot,row in acknowledgements.items():
            if type(slot) is not int or slot not in self.sessions or set(row) != {'token','controls'}:
                raise ValueError('Foreign ACK slot or fields')
            w = self.sessions[slot]
            w.acknowledge(w.session.identity,row['token'],row['controls'])

    def close(self, slots, observations):
        for slot in slots:
            w = self.sessions[slot]
            if w.session.closed:
                raise ValueError('Duplicate episode close')
            w.close(w.session.identity,observations[slot])

    def infer(self, observations, active):
        if (len(observations) != len(self.sessions) or len(set(active)) != len(active)
                or not active or any(type(i) is not int or i not in self.sessions for i in active)):
            raise ValueError('Misaligned observation/active batch')
        started = time.monotonic()
        wrappers = [self.sessions[i] for i in active]
        for slot,w in zip(active,wrappers,strict=True):
            actor_observation(observations[slot])
            w.session._check(w.session.identity,w.session.feedback.control_step)
            if w.clock.pending is not None:
                raise ValueError('Never infer from unacknowledged actions')
            w.observe(observations[slot])
        due = [j for j,w in enumerate(wrappers) if w.session.planning_due(w.session.identity,interval_controls=128)]
        planned = plan_many(self.planner,[wrappers[j] for j in due],[observations[active[j]] for j in due],self.adapter) if due else []
        for j,plan in zip(due,planned,strict=True):
            w=wrappers[j]; w.last_plan=deepcopy(plan)
            w.emit('plan',dict(result=plan,observation_sha256=observation_hash(observations[active[j]])))
            w.observe(observations[active[j]])
        goals = []
        for w in wrappers:
            if w.last_observer is None or w.last_observer['attempt_generation'] != w.session.attempt_generation:
                raise ValueError('Observer output belongs to another intent/attempt')
            goals.append(w.session.low_goal(w.session.identity))
        actions = self.sample([observations[i] for i in active],goals,[w.session.feedback.control_step for w in wrappers])
        if actions.shape != (len(active),16,23) or not np.isfinite(actions).all():
            raise ValueError('Action rows differ from live environments')
        result = {}
        for row,slot,w,goal in zip(actions,active,wrappers,goals,strict=True):
            token = w.clock.offer(w.session.identity,row)
            display = dict(control_step=w.session.feedback.control_step,revision=w.session.revision,
                attempt_generation=w.session.attempt_generation,goal=goal,observer=deepcopy(w.last_observer))
            w.emit('action',dict(token=token,display=display,raw_actions=row.tolist(),
                observation_sha256=observation_hash(observations[slot])))
            result[slot] = dict(token=token,action_chunk=row)
        self.requests += 1
        return result, dict(seconds=time.monotonic()-started,active_slots=list(active),high_batch=len(due),low_batch=len(active))
