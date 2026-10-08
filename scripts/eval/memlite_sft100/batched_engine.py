"""True high/low batching over the frozen G0.5 native FM implementation.

No weights, ODE steps, planner frequency, decoding grammar or physics changes.
The legacy scalar-shaped RNG calls are retained: batching torch.randn itself
would change the CUDA random stream even for the same seed and environment pair.
"""
from contextlib import contextmanager
from copy import deepcopy
import json
import time
import numpy as np
import torch
from native_engine import NativeSFT
from g05.models.g05.inferencer import PolicyInferencer
from batch_core import validate_indices, needs_plan, stage_plans, commit_plans
from sparse_cache import indexed_cache_freeze
from g05.models.kv_cache import SparseKVCache
from chunk_trace import AlignmentTrace, scalar_prompt_fields


@contextmanager
def row_noise(helper, fixed=None, capture=None):
    """Scope only this helper instance; leave torch/random and the ODE intact."""
    had_attribute = '_sample_noise' in helper.__dict__
    prior_attribute = helper.__dict__.get('_sample_noise')
    original = helper._sample_noise
    calls = 0

    def sample(actions, dtype, embodiment_types=None):
        nonlocal calls
        calls += 1
        if calls != 1:
            raise RuntimeError('Native FM must draw initial noise exactly once')
        if fixed is None:
            rows = [original(actions[i:i+1], dtype,
                             None if embodiment_types is None else embodiment_types[i:i+1])
                    for i in range(len(actions))]
            result = torch.cat(rows, dim=0)
        else:
            if tuple(fixed.shape) != tuple(actions.shape):
                raise ValueError('Fixed-noise audit shape mismatch')
            result = fixed.to(device=actions.device, dtype=dtype).clone()
        if capture is not None:
            capture.append(result.detach().cpu().clone())
        return result

    helper._sample_noise = sample
    try:
        yield
        if calls != 1:
            raise RuntimeError('Native FM did not request initial noise')
    finally:
        if had_attribute:
            helper._sample_noise = prior_attribute
        else:
            del helper._sample_noise


class BatchedSFT(NativeSFT):
    def __init__(self, output, *, mode='batch', capture=False, trace_dir=None):
        if mode not in ('batch', 'serial'):
            raise ValueError('Unknown inference mode')
        super().__init__(output)
        self.mode = mode
        self.capture = capture
        self.last_timing = {}
        self.audit_inputs = []
        self.session_generation = 0
        self.captured_tasks = set()
        self.alignment_trace = AlignmentTrace(trace_dir) if trace_dir is not None else None
        self.last_low_samples = []

    def begin(self, task, num_envs, seed, episode_metadata=None):
        if episode_metadata is None or len(episode_metadata) != num_envs:
            raise ValueError('Every environment needs an explicit episode identity')
        identities = [(m.get('task'), m.get('split'), m.get('instance_id')) for m in episode_metadata]
        if (len(set(identities)) != num_envs or any(t != task or s not in ('train_smoke','public_test')
                or type(i) is not int for t,s,i in identities)):
            raise ValueError('Cross-task, duplicate, or missing episode identity')
        super().begin(task, num_envs, seed, episode_metadata)
        self.session_generation += 1
        for slot in self.slots:
            memory = json.loads(slot['ledger'].memory)
            if (memory['task_name'] != self.task or memory['issued_command_history'] or
                    memory['verified_world_facts'] or slot['ledger'].revision != 0 or
                    slot['projection'] is not None or slot['chunks'] != 0 or
                    slot['context_id'] is not None or slot['planned_chunk'] is not None):
                raise RuntimeError('Task reset retained another episode\'s memory/intent')
        if len({id(slot['ledger']) for slot in self.slots}) != num_envs:
            raise RuntimeError('Environment ledgers alias one another')
        with (self.output.parent/'session_begin.jsonl').open('a') as stream:
            stream.write(json.dumps(dict(generation=self.session_generation, task=task,
                identities=identities, all_memories_empty=True, independent_ledgers=True,
                contexts_cleared=True, requests_before=self.requests))+'\n')
        if getattr(self, 'alignment_trace', None) is not None:
            self.alignment_trace.begin(task=task, episode_metadata=episode_metadata,
                                       session_generation=self.session_generation)

    def prepare(self, side, observations, projections):
        if side not in ('high','low') or len(observations) != len(projections):
            raise ValueError('Invalid high/low preparation request')
        for obs, projection in zip(observations, projections, strict=True):
            if (obs['task'] != self.task or projection['task_name'] != self.task or
                    projection['memlite_branch'] != side):
                raise ValueError('Cross-task or cross-branch conditioning rejected')
        return super().prepare(side, observations, projections)

    @torch.no_grad()
    def plan_batch(self, observations, indices):
        projections = [self.slots[i]['ledger'].projection() for i in indices]
        batch = self.prepare('high', observations, projections)
        with self._branch_context(), indexed_cache_freeze(self.models['high'].model.ar_helper, SparseKVCache):
            result = self.models['high'].generate_high_level(
                batch['samples'], batch['pixel_values'], temperature=0.)
        staged = stage_plans(self.slots, indices, result['planner_events'])
        # All rows parsed and validated before any live ledger is committed.
        with (self.output.parent/'planner_events.jsonl').open('a') as stream:
            for row, observation in zip(staged, observations, strict=True):
                index = row['index']
                stream.write(json.dumps(dict(task=self.task, env=index,
                    chunk=self.slots[index]['chunks'], event=row['event'],
                    context_id=row['context_id'], episode=self.episode_metadata[index],
                    control_step=observation.get('control_step'), policy_update=0,
                    inference_mode=self.mode, high_batch_size=len(indices)))+'\n')
        commit_plans(self.slots, staged)
        if getattr(self, 'alignment_trace', None) is not None:
            texts = result.get('high_level_text', [])
            if len(texts) != len(indices):
                raise RuntimeError('Alignment trace requires one raw planner text per row')
            for row, observation, projection, sample, raw_text in zip(
                    staged, observations, projections, batch['samples'], texts, strict=True):
                self.alignment_trace.register_high(env=row['index'],
                    chunk_id=self.slots[row['index']]['chunks'], observation=observation,
                    projection=projection, sample=sample, raw_text=raw_text,
                    event=row['event'], memory_after=row['ledger'].memory,
                    context_id=row['context_id'])
        return result

    @torch.no_grad()
    def low_batch(self, observations, projections, *, fixed_noise=None, capture_noise=None):
        batch = self.prepare('low', observations, projections)
        self.last_low_samples = ([scalar_prompt_fields(sample) for sample in batch['samples']]
                                 if getattr(self, 'alignment_trace', None) is not None else [])
        policy = self.models['low']
        if not policy.continuous_action or policy.discrete_action or policy.predict_cot:
            raise ValueError('Not the original SkillFM inference route')
        helper = policy.model.fm_helper
        if helper.num_inference_steps != 10 or helper.horizon_steps != 32 or helper.action_dim != 27:
            raise ValueError('FM10/32-step/27D model contract changed')
        with self._branch_context(), row_noise(helper, fixed_noise, capture_noise):
            result = policy.forward_inference(samples=batch['samples'],
                pixel_values=batch['pixel_values'], action_dim_is_pad=batch['action_dim_is_pad'])
        if result.get('selected_action_source') != 'fm':
            raise ValueError('Expected native FM output')
        cpu = {k:v.cpu() for k,v in batch.items() if isinstance(v, torch.Tensor)}
        cpu['action'] = result['action'].cpu()
        cpu['selected_action_source'] = 'fm'
        decoded = []
        for row, observation in enumerate(observations):
            raw = self.raw(observation)
            anchor = {k:raw['state'][k][-1:].clone().unsqueeze(0)
                      for k in ('left_arm', 'right_arm', 'trunk_qpos')}
            groups = PolicyInferencer._postprocess_single(cpu, row, self.processors['low'],
                                                          raw_state_anchor=anchor)
            order = [('base_qvel',3), ('trunk_qpos',4), ('left_arm',7), ('left_gripper',1),
                     ('right_arm',7), ('right_gripper',1)]
            action = torch.cat([torch.as_tensor(groups[k]).reshape(32,w) for k,w in order], -1)[:16]
            decoded.append(action.float().numpy())
        actions = np.stack(decoded)
        if actions.shape != (len(observations),16,23) or not np.isfinite(actions).all():
            raise ValueError('Invalid batched raw 23D action')
        return actions, cpu['action'].float().numpy()

    @torch.no_grad()
    def infer_native(self, observations, indices):
        validate_indices(observations, indices, len(self.slots))
        capture = self.capture and (self.requests in (0, 8) or self.task not in self.captured_tasks)
        if capture:
            # TRAIN-only captures for offline fixed-input and fixed-noise QA.
            snapshot = dict(observations=deepcopy(observations), indices=list(indices),
                slots=deepcopy(self.slots), episode_metadata=deepcopy(self.episode_metadata),
                task=self.task, requests=self.requests, cuda_rng=torch.cuda.get_rng_state(),
                inference_mode=self.mode, session_generation=self.session_generation)
        started = time.monotonic()
        if self.mode == 'serial':
            actions, contexts = super().infer_native(observations, indices)
            self.last_timing = dict(mode='serial', total_seconds=time.monotonic()-started,
                                    batch_size=len(indices))
        else:
            for observation in observations:
                observation['task'] = self.task
            due = [row for row, index in enumerate(indices) if needs_plan(self.slots[index])]
            high_started = time.monotonic()
            if due:
                self.plan_batch([observations[row] for row in due], [indices[row] for row in due])
            high_seconds = time.monotonic()-high_started
            projections = [self.slots[i]['projection'] for i in indices]
            low_started = time.monotonic()
            actions, _ = self.low_batch(observations, projections)
            low_seconds = time.monotonic()-low_started
            if getattr(self, 'alignment_trace', None) is not None:
                if len(self.last_low_samples) != len(indices):
                    raise RuntimeError('Low prompt trace cardinality mismatch')
                timing = dict(mode='batch', high_seconds=high_seconds, low_seconds=low_seconds,
                    total_seconds=time.monotonic()-started, high_batch_size=len(due),
                    batch_size=len(indices))
                for row, index in enumerate(indices):
                    self.alignment_trace.record_chunk(env=index,
                        chunk_id=self.slots[index]['chunks'], observation=observations[row],
                        low_projection=projections[row], low_sample=self.last_low_samples[row],
                        action_chunk=actions[row], inference_request=self.requests,
                        timing=timing, context_id=self.slots[index]['context_id'])
            for index in indices:
                self.slots[index]['chunks'] += 1
            contexts = [dict(context_id=self.slots[i]['context_id'], parent_goal=p['parent_goal'],
                             active_skills_semantic_json=p['active_skills_semantic_json'])
                        for i,p in zip(indices, projections, strict=True)]
            self.requests += 1
            self.last_timing = dict(mode='batch', high_seconds=high_seconds, low_seconds=low_seconds,
                total_seconds=time.monotonic()-started, high_batch_size=len(due), batch_size=len(indices))
        if capture:
            snapshot.update(actions=actions, contexts=contexts)
            path = self.output.parent/f'audit_input_{snapshot["requests"]:04d}.pt'
            torch.save(snapshot, path)
            self.audit_inputs.append(str(path))
            self.captured_tasks.add(self.task)
        assert self.trainer is None
        return actions, contexts

    def verify(self):
        return super().verify() | dict(inference_mode=self.mode,
            noise_protocol='legacy scalar-shaped draws in row order', audit_inputs=self.audit_inputs)
