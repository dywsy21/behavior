"""Per-episode champion stage voting, rolling inpainting and real control ACKs.

Execution equations follow champion ca556f74 eval_b1k_wrapper.py. No simulator
privileged state enters this module. Model state/RNG is not shared by sessions.
"""
from collections import deque
import hashlib
import json

import numpy as np

from .contracts import state23_from_proprio61


class ChampionSession:
    def __init__(self, *, episode, task, model_task, num_stages, seed, predictor,
                 correct_actions, gripper_variation, apply_legacy_rules=True,
                 clock_kind='physical_ack'):
        if not episode or not 0 <= task < 100 or not 0 <= model_task < 100 or not 5 <= num_stages <= 15:
            raise ValueError('Explicit episode/task/stage contract required')
        if not 0 <= seed < 2**32:
            raise ValueError('Explicit uint32 policy seed required')
        if apply_legacy_rules and model_task >= 50:
            raise ValueError('Never apply old-task correction tables to new tasks')
        if clock_kind not in ('physical_ack', 'policy_delivery'):
            raise ValueError('An explicit physical or delivery clock is required')
        self.clock_kind = clock_kind
        self.delivered_controls = 0
        self.episode, self.task, self.model_task = episode, task, model_task
        self.num_stages, self.seed = num_stages, seed
        self.predictor, self.correct_actions = predictor, correct_actions
        self.gripper_variation, self.apply_legacy_rules = gripper_variation, apply_legacy_rules
        self.stage = 0
        self.history = deque(maxlen=3)
        self.initial_actions = None
        self.rng = np.array([0,seed],dtype=np.uint32)
        self.control_step = self.predictions = 0
        self.pending = None
        self.token = None
        self.used = self.closed = False

    def begin(self):
        if self.used or self.closed:
            raise ValueError('A session cannot reset/reuse another episode')
        self.used = True
        return dict(episode=self.episode,task=self.task,**self.clock())

    def clock(self):
        if self.clock_kind == 'physical_ack':
            return dict(control_step=self.control_step)
        # The standard official websocket has no execution acknowledgement.
        # A sent chunk may be only partly executed when the episode terminates.
        return dict(clock_kind='policy_delivery',delivered_controls=self.delivered_controls,
                    physically_confirmed_controls=None)

    def _vote(self, predicted):
        self.history.append(predicted)
        if len(self.history) != 3:
            return
        next_stage = self.stage + 1
        # Preserve original policy, including no ordinary voting out of the
        # final stage; radio recovery is an explicit legacy correction rule.
        if next_stage <= self.num_stages - 1:
            if (sum(v==next_stage for v in self.history) >= 2 or
                    sum(v==next_stage+1 for v in self.history) == 3):
                self.stage = next_stage
                self.history.clear()
            elif self.stage > 0 and sum(v==self.stage-1 for v in self.history) == 3:
                self.stage -= 1
                self.history.clear()

    def infer(self, observation):
        if not self.used or self.closed or self.pending is not None:
            raise ValueError('BEGIN and complete prior control ACKs required')
        if set(observation) != {'images','proprio'}:
            raise ValueError('Only deployed RGB and proprioception are accepted')
        raw = np.asarray(observation['proprio'])
        if raw.shape != (61,):
            raise ValueError('One real R1Pro61 observation required')
        current = state23_from_proprio61(raw)
        input_stage = self.stage
        previous = None if self.initial_actions is None else self.initial_actions.copy()
        output,new_rng = self.predictor(observation,self.model_task,input_stage,previous,self.rng.copy())
        actions = np.asarray(output['actions'])
        logits = np.asarray(output['subtask_logits'])
        if actions.shape != (30,23) or not np.isfinite(actions).all():
            raise ValueError('Champion must return exactly30 finite native23 controls')
        if logits.shape != (15,) or not np.isfinite(logits[:self.num_stages]).all():
            raise ValueError('Expected finite logits for every legal stage')
        if not np.isneginf(logits[self.num_stages:]).all():
            raise ValueError('Invalid stages must be masked, not silently clipped')
        new_rng = np.asarray(new_rng)
        if new_rng.shape!=(2,) or new_rng.dtype!=np.uint32:
            raise ValueError('Predictor must return its own next PRNG state')
        corrected_stage = self.stage
        corrected = actions.copy()
        if self.apply_legacy_rules:
            corrected,corrected_stage = self.correct_actions(self.model_task,self.stage,current,corrected)
        if (np.asarray(corrected).shape!=(30,23) or not np.isfinite(corrected).all()
                or not 0 <= corrected_stage < self.num_stages):
            raise ValueError('Invalid legacy correction output')
        if corrected_stage != self.stage:
            self.stage = int(corrected_stage)
            self.history.clear()
        # Upstream executes20 controls in both branches. With low gripper
        # variation it compresses original26 ->20 and retains26:30. With high
        # variation it uses original0:20 and retains20:24 (not26:30).
        has_variation,_,_ = self.gripper_variation(corrected,26)
        compress = not has_variation
        consumed = 26 if compress else 20
        self.initial_actions = np.array(corrected[consumed:consumed+4],copy=True)
        chunk = np.array(corrected[:consumed],copy=True)
        if compress:
            from scipy.interpolate import interp1d
            x = np.linspace(0,consumed-1,consumed)
            y = np.linspace(0,consumed-1,20)
            chunk = np.stack([interp1d(x,chunk[:,dim],kind='cubic')(y) for dim in range(23)],axis=1)
            chunk[:,:3] *= consumed/20
        chunk = chunk.astype(np.float32)
        if chunk.shape!=(20,23) or not np.isfinite(chunk).all():
            raise ValueError('Nonfinite interpolated controls')
        predicted = int(np.argmax(logits))
        legal = logits[:self.num_stages].astype(np.float64)
        prob = np.exp(legal-legal.max());prob /= prob.sum()
        self._vote(predicted)
        self.rng = new_rng.copy()
        self.predictions += 1
        identity = dict(episode=self.episode,task=self.task,prediction=self.predictions,
                        start=self.control_step if self.clock_kind=='physical_ack' else self.delivered_controls,
                        actions_sha256=hashlib.sha256(chunk.tobytes()).hexdigest())
        if self.clock_kind != 'physical_ack':
            identity['clock_kind'] = self.clock_kind
        self.token = hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
        self.pending = chunk.copy()
        return dict(token=self.token,action_chunk=chunk,display=dict(
            episode=self.episode,task=self.task,**self.clock(),
            input_stage=input_stage,corrected_stage=int(corrected_stage),next_stage=self.stage,
            predicted_stage=predicted,stage_probability=float(prob[predicted]),
            stage_probabilities=prob.tolist(),stage_votes=list(self.history),compressed=compress,
            inpainting_input_steps=0 if previous is None else len(previous),
            inpainting_next_start=consumed,legacy_corrections=self.apply_legacy_rules,
            meaning='duration-stage prediction, NOT MEM-Lite intent or physical outcome'))

    def ack(self, *, token, controls):
        if self.clock_kind != 'physical_ack':
            raise ValueError('A delivery-only session cannot accept fabricated physical ACKs')
        if self.closed or self.pending is None or token!=self.token or not controls:
            raise ValueError('Stale, foreign or empty control ACK')
        if len(controls)>len(self.pending):
            raise ValueError('More controls acknowledged than offered')
        for offset,row in enumerate(controls):
            if set(row)!= {'control_step','simulator_apply_ack','action_executed_raw23'}:
                raise ValueError('Malformed control ACK')
            actual=np.asarray(row['action_executed_raw23'],dtype=np.float32)
            if (row['simulator_apply_ack'] is not True or row['control_step']!=self.control_step+offset
                    or actual.shape!=(23,) or not np.array_equal(actual,self.pending[offset])):
                raise ValueError('Simulator ACK does not match offered native23/clock')
        self.control_step += len(controls)
        self.pending = self.pending[len(controls):].copy()
        if not len(self.pending):
            self.pending = None
            self.token = None
        return dict(control_step=self.control_step)

    def record_delivery(self, *, token, actions):
        """Called only after transport send; deliberately NOT a physical ACK."""
        if self.clock_kind != 'policy_delivery':
            raise ValueError('Physical sessions require real simulator ACKs')
        sent = np.asarray(actions)
        if (self.closed or self.pending is None or token != self.token
                or sent.dtype != np.float32 or sent.ndim != 2 or sent.shape[1] != 23
                or not 0 < len(sent) <= len(self.pending)
                or not np.array_equal(sent,self.pending[:len(sent)])):
            raise ValueError('Foreign, stale or changed policy delivery')
        self.delivered_controls += len(sent)
        self.pending = self.pending[len(sent):].copy()
        if not len(self.pending):
            self.pending = None
            self.token = None
        return self.clock()

    def finish(self):
        if not self.used or self.closed:
            raise ValueError('Cannot finish an unused/closed episode')
        # A true official terminal may leave a partial offered chunk unused.
        unused=0 if self.pending is None else len(self.pending)
        self.closed=True
        self.pending=None
        self.initial_actions=None
        self.history.clear()
        return dict(episode=self.episode,task=self.task,**self.clock(),
                    predictions=self.predictions,unused_offered_controls=unused,optimizer_updates=0)
