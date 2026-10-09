"""Collect the shared 100-task reward with consistent finite-horizon targets.

Official episode timeouts end the task. An optimizer batch boundary does not
end it: its last state is bootstrapped and the live reward state is retained.
The remaining time is a critic-only feature, never an actor/planner input.
"""
from collections import deque
import json
import os

from dense_reward import chunk_targets
from pilot_policy import PilotPolicy
from shared_reward_math import CONTROL_GAMMA

PROTOCOL = 'shared_terminal_q_v1'


class SharedPilotPolicy(PilotPolicy):
    def __init__(self, num_envs, task, run):
        self.recorder=None
        self.horizon = int(os.environ['RL_EPISODE_STEPS'])
        if self.horizon <= 0:
            raise ValueError('Official task horizon must be positive')
        super().__init__(num_envs, task, run)
        from recovery_recorder import RecoveryRecorder
        from training_config import load_training_config
        self.recorder=RecoveryRecorder(os.environ['RL_RECOVERY_ROOT'],num_envs,load_training_config(),
            dict(run=str(self.run),source_commit=os.environ['RL_SOURCE_COMMIT'],
                 policy_source='on_policy_stochastic_flow_PPO',teacher=False))

    def begin_recording(self,evaluator):
        for index,metadata in enumerate(self.episode_metadata):
            actual=int(evaluator.instance_eval_states[index].instance_id)
            if actual!=metadata['instance_id']:raise ValueError('Actual instance differs from recorder metadata')
        self.recorder.begin(self.episode_metadata)
        for index, state in enumerate(evaluator.instance_eval_states):
            named_scope = {}
            for entity, wrapped in state.env_accessor.object_scope.items():
                obj = getattr(wrapped, 'wrapped_obj', wrapped)
                named_scope[entity] = getattr(obj, 'name', None) if obj is not None else None
            self.recorder.bind_scope(index, named_scope)

    def close_recording(self,reason):
        if self.recorder is not None:self.recorder.close(reason)

    def reset(self):
        if hasattr(self, 'rows') and any(self.rows):
            raise RuntimeError('Unconsumed trajectories at reset')
        self.rows = [[] for _ in range(self.num_envs)]
        self.actions = [deque() for _ in range(self.num_envs)]
        self.elapsed = [0 for _ in range(self.num_envs)]
        seed=int(os.environ.get('RL_SEED_BASE','17'))+self.reset_count
        supplied=json.loads(os.environ['RL_EPISODE_METADATA'])
        if len(supplied)!=self.num_envs:raise ValueError('Missing episode provenance')
        self.episode_metadata=[dict(row,policy_seed=seed,
            episode_id=f"{self.run.name}:env{index}:instance{row['instance_id']}:seed{seed}")
            for index,row in enumerate(supplied)]
        self.rpc('begin', task=self.task, num_envs=self.num_envs,
                 seed=seed,episode_metadata=self.episode_metadata,
                 reward_protocol=PROTOCOL, control_gamma=CONTROL_GAMMA)
        self.reset_count += 1

    def observations(self, obs, indices):
        observations = super().observations(obs, indices)
        for observation, index in zip(observations, indices):
            # The installed official timeout fires after max_steps, giving
            # max_steps+1 executed controls. Keep this explicit and measurable.
            observation['critic_remaining_fraction'] = max(
                0.0, 1.0 - self.elapsed[index] / (self.horizon + 1))
            observation['control_step']=self.elapsed[index]
        return observations

    def observe(self, evaluator, active, terminated, truncated):
        for index in active:
            task_terminal = bool(terminated[index] or truncated[index])
            result, audit = evaluator.reward_adapters[index].step(task_terminal)
            row = self.rows[index][-1]
            row['reward'] += CONTROL_GAMMA ** row['controls'] * result.total
            row['controls'] += 1
            self.elapsed[index] += 1
            row['terminated'] = task_terminal
            row['truncated'] = bool(truncated[index])
            if task_terminal:
                row['next_value'] = 0.0
                self.actions[index].clear()
            self.total_steps += 1
            self.recorder.observe(index,evaluator.instance_eval_states[index].obs,result.total,audit,
                                  terminated[index],truncated[index])
            record = dict(task=self.task, env=index, step=self.total_steps,
                          episode=self.episode_metadata[index],experience_id=row['experience_id'],
                          episode_control_step=self.elapsed[index],
                          reward=result.total, terminal_q_reward=result.terminal_q_reward,
                          shaping=result.shaping, task_terminal=task_terminal,
                          reward_protocol=PROTOCOL, control_gamma=CONTROL_GAMMA, **audit)
            with (self.run / 'reward_steps.jsonl').open('a') as output:
                output.write(json.dumps(record) + '\n')
        if sum(map(len, self.rows)) >= int(os.environ.get('RL_CHUNKS_PER_UPDATE', '64')) and all(not self.actions[i] for i in active):
            self.flush(evaluator)

    def flush(self, evaluator):
        indices = [i for i, rows in enumerate(self.rows)
                   if rows and rows[-1]['next_value'] is None]
        if indices:
            values = self.rpc('value', observations=self.observations(evaluator._batch_obs(), indices),
                              indices=indices)['values']
            for index, value in zip(indices, values):
                self.rows[index][-1]['next_value'] = float(value)
        identifiers, advantages, returns = [], [], []
        for rows in self.rows:
            if not rows:
                continue
            assert all(row['controls'] > 0 and row['next_value'] is not None for row in rows)
            local_advantages, local_returns = chunk_targets(rows, gamma=CONTROL_GAMMA, gae_lambda=0.95)
            identifiers.extend(row['experience_id'] for row in rows)
            advantages.extend(local_advantages)
            returns.extend(local_returns)
        if identifiers:
            result = self.rpc('update', experience_ids=identifiers, advantages=advantages, returns=returns)
            result.update(reward_protocol=PROTOCOL, control_gamma=CONTROL_GAMMA,
                          gae_lambda_per_chunk=0.95, collected_chunks=len(identifiers))
            with (self.run / 'update_receipts.jsonl').open('a') as output:
                output.write(json.dumps(result) + '\n')
            self.rows = [[] for _ in range(self.num_envs)]
