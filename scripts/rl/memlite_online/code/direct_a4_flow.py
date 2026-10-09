"""Direct PPO adapter for A4's existing Flow Matching action expert.

The visual/language prefix is computed by the frozen A4 stack.  PPO gradients
enter the original ``model.action_expert`` through stochastic Euler
transitions; no post-hoc residual action network is used.
"""
from __future__ import annotations
import time

from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time
from typing import Any, Sequence

import torch
from torch import Tensor, nn

from direct_ppo_core import (
    ppo_clipped_policy_loss,
    recompute_flow_log_prob,
    recompute_flow_step_log_probs,
    sample_stochastic_flow,
)


class ValueHead(nn.Module):
    def __init__(self, feature_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, 1),
        )

    def forward(self, features: Tensor) -> Tensor:
        return self.network(features.float()).squeeze(-1)


def _detach_cpu(value: Any) -> Any:
    if isinstance(value, Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _detach_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_detach_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_detach_cpu(item) for item in value)
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


class A4DirectPPO:
    """Own optimizers, replay handles and direct-A4 PPO updates."""

    def __init__(
        self,
        policy: nn.Module,
        action_dim_is_pad: Tensor,
        output_dir: Path,
        *,
        transition_std: float = 0.02,
        actor_lr: float = 1e-8,
        critic_lr: float = 1e-4,
        clip_ratio: float = 0.2,
        target_kl: float = 0.05,
        max_clip_fraction: float = 0.25,
        update_epochs: int = 1,
        microbatch_size: int = 1,
        reward_protocol: str | None = None,
    ) -> None:
        if not 0 < transition_std < 1:
            raise ValueError("transition_std must be in (0, 1)")
        self.policy = policy
        self.output_dir = Path(output_dir)
        self.transition_std = float(transition_std)
        self.clip_ratio = float(clip_ratio)
        self.target_kl = float(target_kl)
        self.max_clip_fraction = float(max_clip_fraction)
        if not 0 < self.target_kl <= 0.05 or not 0 < self.max_clip_fraction <= 0.5:
            raise ValueError('Invalid trust-region safeguard')
        self.nominal_actor_lr = float(actor_lr)
        self.update_epochs = int(update_epochs)
        self.microbatch_size = int(microbatch_size)
        self.training_mode = False
        self.reward_protocol = reward_protocol or os.environ.get('RL_REWARD_PROTOCOL', 'legacy_two_task_v1')
        if self.reward_protocol not in ('legacy_two_task_v1', 'shared_terminal_q_v1', 'skill_aligned_v1'):
            raise ValueError('Unknown reward protocol')
        self.update_count = 0
        self.actor_update_count = 0
        self.next_experience_id = 0
        self.experiences: dict[int, dict[str, Any]] = {}

        pad = torch.as_tensor(action_dim_is_pad, dtype=torch.bool)
        if pad.ndim == 2:
            if not torch.equal(pad, pad[:1].expand_as(pad)):
                raise ValueError("A4 direct PPO requires one shared padding schema")
            pad = pad[0]
        if pad.ndim != 1:
            raise ValueError("action_dim_is_pad must be [D] or identical [B,D]")
        self.active_action_dims = (~pad).cuda()

        self.actor_parameters = []
        self.actor_names = []
        for name, parameter in policy.named_parameters():
            enabled = name.startswith("model.action_expert.")
            parameter.requires_grad_(enabled)
            if enabled:
                self.actor_names.append(name)
                self.actor_parameters.append(parameter)
        if not self.actor_parameters:
            raise RuntimeError("A4 exposes no action_expert parameters")
        if {id(p) for p in self.actor_parameters} != {
            id(p) for p in policy.model.action_expert.parameters()
        }:
            raise RuntimeError("action expert parameter identity is not exact")
        self.policy.eval()
        # A flow trajectory contains thousands of Gaussian terms.  Even a tiny
        # actor change can therefore move its joint probability substantially.
        # Disable weight decay and use a deliberately small PPO step.
        self.actor_optimizer = torch.optim.AdamW(
            self.actor_parameters, lr=actor_lr, weight_decay=0.0
        )
        self.critic_lr = float(critic_lr)
        self.critic: ValueHead | None = None
        self.critic_optimizer: torch.optim.Optimizer | None = None
        self._actor_reference = self.actor_parameters[0].detach().float().cpu().clone()

    def load_checkpoint(self, path: Path) -> dict[str, Any]:
        """Restore the direct action expert, value head, optimizers and RNG."""
        payload = torch.load(path, map_location="cuda", weights_only=False)
        if payload.get("kind") != "a4_direct_flow_ppo_trainable_state_v1":
            raise ValueError("wrong direct PPO checkpoint kind")
        if payload.get('reward_protocol', 'legacy_two_task_v1') != self.reward_protocol:
            raise ValueError('Checkpoint reward/critic protocol differs; do not silently reuse its critic')
        if float(payload["transition_std"]) != self.transition_std:
            raise ValueError("resume transition_std differs from service configuration")
        if float(payload["clip_ratio"]) != self.clip_ratio:
            raise ValueError("resume clip_ratio differs from service configuration")
        self.policy.model.action_expert.load_state_dict(payload["action_expert"], strict=True)
        self.actor_optimizer.load_state_dict(payload["actor_optimizer"])
        feature_dim = int(payload["critic"]["network.0.weight"].shape[1])
        self.critic = ValueHead(feature_dim).cuda()
        self.critic.load_state_dict(payload["critic"], strict=True)
        self.critic_optimizer = torch.optim.AdamW(
            self.critic.parameters(), lr=self.critic_lr, weight_decay=0.0
        )
        self.critic_optimizer.load_state_dict(payload["critic_optimizer"])
        actor_equal = all(torch.equal(value, payload['action_expert'][name])
                          for name, value in self.policy.model.action_expert.state_dict().items())
        critic_equal = all(torch.equal(value, payload['critic'][name])
                           for name, value in self.critic.state_dict().items())
        if not actor_equal or not critic_equal:
            raise RuntimeError('Restored weights differ from checkpoint')
        self.update_count = int(payload["updates"])
        self.actor_update_count = int(payload.get('actor_updates', self.update_count))
        if 'python_rng' in payload:
            random.setstate(payload['python_rng'])
        if 'numpy_rng' in payload:
            import numpy as np
            np.random.set_state(payload['numpy_rng'])
        torch.set_rng_state(payload["torch_rng_cpu"].cpu())
        torch.cuda.set_rng_state(payload["torch_rng_cuda"].cpu())
        return {
            "path": str(path),
            "sha256": _sha256(path),
            "updates": self.update_count,
            "actor_matches_checkpoint": actor_equal,
            "critic_matches_checkpoint": critic_equal,
            "actor_optimizer_states": len(self.actor_optimizer.state),
            "critic_optimizer_states": len(self.critic_optimizer.state),
            "actor_adam_step_min": min((float(v['step']) for v in self.actor_optimizer.state.values()),default=None),
            "actor_adam_step_max": max((float(v['step']) for v in self.actor_optimizer.state.values()),default=None),
            "actor_lr": self.actor_optimizer.param_groups[0]["lr"],
            "new_nominal_actor_lr": self.nominal_actor_lr,
            "actor_updates": self.actor_update_count,
        }

    @property
    def fm(self):
        return self.policy.model.fm_helper

    def _ensure_critic(self, features: Tensor) -> None:
        if self.critic is None:
            self.critic = ValueHead(int(features.shape[-1])).cuda()
            self.critic_optimizer = torch.optim.AdamW(
                self.critic.parameters(), lr=self.critic_lr, weight_decay=0.0
            )
        elif self.critic.network[0].in_features != features.shape[-1]:
            raise RuntimeError("A4 prefix feature dimension changed")

    def _prefix(self, branch_batch: dict[str, Any]):
        # The Stage1 serving builder already validates the target-free low prefix.
        # Never call the legacy six-frame policy validator on this new model.
        for sample in branch_batch["samples"]:
            if sample.get("memlite_branch") != "low" or not sample["template"].endswith("<EOC>"):
                raise ValueError("Expected validated target-free Stage1 low prefix")
        with torch.no_grad():
            state = self.policy.prefill(branch_batch["samples"], branch_batch["pixel_values"])
        features = torch.nn.functional.layer_norm(
            state.last_hidden.detach().float(), state.last_hidden.shape[-1:]
        )
        if self.reward_protocol == 'shared_terminal_q_v1':
            remaining = branch_batch['critic_remaining_fraction'].float().reshape(-1, 1)
            if (remaining.shape[0] != features.shape[0] or not torch.isfinite(remaining).all()
                    or not ((remaining >= 0) & (remaining <= 1)).all()):
                raise ValueError('Invalid critic-only remaining time')
            features = torch.cat((features, remaining.to(features.device)), dim=-1)
        self._ensure_critic(features)
        return state, features

    def _velocity_fn(self, state):
        model = self.policy.model
        fm = self.fm
        first_image = next(iter(state.pixel_values.values()))
        dtype = torch.float32
        action_mask, action_pos = model.build_action_mask_and_position_ids(
            state.attention_mask,
            action_len=fm.horizon_steps,
            position_ids_prefix=state.position_ids,
            split_index=state.attention_mask.size(1),
            dtype=dtype,
            action_causal=fm.action_causal,
        )
        vlm_kv = state.kv_cache
        if hasattr(model, "_build_prefix_action_kv"):
            prefix = model._build_prefix_action_kv(vlm_kv, vlm_kv.num_items())
            # Only cast the newly built action-conditioning cache. The frozen VLM stays BF16.
            for name in ("key_cache", "value_cache", "recurrent_states", "split_recurrent_states"):
                values = getattr(prefix, name, {})
                for key, value in values.items():
                    if isinstance(value, Tensor) and value.is_floating_point():
                        values[key] = value.float()
            kv_kwargs = {"kv_cache": prefix}
        else:
            kv_kwargs = {"past_key_values": vlm_kv}

        def velocity(actions: Tensor, times: Tensor) -> Tensor:
            with torch.autocast(actions.device.type, enabled=False):
                action_embeds = model.action_expert.embed(actions.float())
                time_cond = model.action_expert.encode_time(times)
                hidden = model.action_expert(
                    inputs_embeds=action_embeds,
                    attention_mask=action_mask,
                    position_ids=action_pos,
                    time_cond=time_cond,
                    attn_implementation=model.attn_implementation,
                    mixture_name="action",
                    **kv_kwargs,
                )
                return model.action_expert.decode(hidden)

        return velocity

    @torch.no_grad()
    def sample_branch(self, branch_batch: dict[str, Any], observations: Sequence[dict[str, Any]],
                      local_subgoals: Sequence[Any]) -> dict[str, Any]:
        state, features = self._prefix(branch_batch)
        batch_size = features.shape[0]
        first_image = next(iter(state.pixel_values.values()))
        dummy = torch.zeros(
            batch_size, self.fm.horizon_steps, self.fm.action_dim,
            device=first_image.device, dtype=torch.float32,
        )
        embodiments = [sample.get("embodiment") for sample in branch_batch["samples"]]
        # FP32 flow states AND time grid must match likelihood recomputation.
        # Never let an image tensor's BF16 dtype choose the Euler time grid.
        initial = self.fm._sample_noise(dummy, torch.float32, embodiments)
        initial[..., ~self.active_action_dims] = 0
        chain = sample_stochastic_flow(
            self._velocity_fn(state),
            initial,
            self.transition_std,
            self.active_action_dims,
            flow_steps=int(self.fm.num_inference_steps),
            pi_convention=str(self.fm.time_convention) == "pi_convention",
        )
        assert self.critic is not None
        values = self.critic(features)
        ids = []
        for index in range(batch_size):
            experience_id = self.next_experience_id
            self.next_experience_id += 1
            self.experiences[experience_id] = {
                "observation": deepcopy(observations[index]),
                "local_subgoal": deepcopy(local_subgoals[index]),
                "chain": chain.states[index].detach().cpu(),
                "old_log_prob": float(chain.log_prob[index].item()),
                "old_step_log_prob": chain.step_log_prob[index].detach().cpu(),
                "old_value": float(values[index].item()),
            }
            ids.append(experience_id)
        return {
            "action": chain.states[:, -1],
            "fm_action": chain.states[:, -1],
            "selected_action_source": "direct_ppo_stochastic_fm",
            "experience_ids": ids,
            "old_log_prob": chain.log_prob,
            "old_value": values,
        }

    def evaluate_prepared(self, branch_batch: dict[str, Any], chains: Tensor,
                          old_log_probs: Tensor, advantages: Tensor,
                          returns: Tensor) -> tuple[Tensor, dict[str, float]]:
        state, features = self._prefix(branch_batch)
        assert self.critic is not None
        new_step_log_prob = recompute_flow_step_log_probs(
            self._velocity_fn(state), chains.cuda(), self.transition_std,
            self.active_action_dims,
            pi_convention=str(self.fm.time_convention) == "pi_convention",
        )
        # One PPO ratio per stochastic flow transition.  Multiplying all ten
        # transition ratios creates an unnecessarily ill-conditioned joint
        # ratio for this 32x23-dimensional action trajectory.
        old_steps = old_log_probs.cuda()
        expanded_advantages = advantages.cuda().unsqueeze(1).expand_as(old_steps)
        policy_loss, ppo_metrics = ppo_clipped_policy_loss(
            new_step_log_prob.reshape(-1), old_steps.reshape(-1),
            expanded_advantages.reshape(-1), clip_ratio=self.clip_ratio
        )
        values = self.critic(features)
        value_loss = 0.5 * (values - returns.cuda()).square().mean()
        total = policy_loss + 0.5 * value_loss
        metrics = {
            "policy_loss": float(policy_loss.detach()),
            "value_loss": float(value_loss.detach()),
            "ratio_mean": float(ppo_metrics["ratio_mean"]),
            "approx_kl": float(ppo_metrics["approx_kl"]),
            "clip_fraction": float(ppo_metrics["clip_fraction"]),
        }
        return total, metrics

    def _save_checkpoint(self, metrics: dict[str, Any]) -> dict[str, Any]:
        if self.critic is None or self.critic_optimizer is None:
            raise RuntimeError("critic is not initialized")
        payload = {
            "kind": "a4_direct_flow_ppo_trainable_state_v1",
            "reward_protocol": self.reward_protocol,
            "updates": self.update_count,
            "actor_updates": self.actor_update_count,
            "training_config": {"nominal_actor_lr": self.nominal_actor_lr,
                                "target_kl": self.target_kl,
                                "max_clip_fraction": self.max_clip_fraction},
            "action_expert": self.policy.model.action_expert.state_dict(),
            "critic": self.critic.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "transition_std": self.transition_std,
            "clip_ratio": self.clip_ratio,
            "torch_rng_cpu": torch.get_rng_state(),
            "torch_rng_cuda": torch.cuda.get_rng_state(),
            "metrics": metrics,
        }
        from checkpoint_io import publish_checkpoint
        import numpy as np
        payload['python_rng'] = random.getstate()
        payload['numpy_rng'] = np.random.get_state()
        payload['source_snapshot'] = str(Path(__file__).resolve().parent)
        payload.update(getattr(self, 'checkpoint_extra', {}))
        return publish_checkpoint(self.output_dir, payload)

    def update(self, inferencer: Any, experience_ids: Sequence[int],
               advantages: Sequence[float], returns: Sequence[float]) -> dict[str, Any]:
        torch.cuda.synchronize()
        update_started = time.perf_counter()
        if not experience_ids or len(experience_ids) != len(advantages) or len(experience_ids) != len(returns):
            raise ValueError("update arrays must be nonempty and aligned")
        if len(set(experience_ids)) != len(experience_ids):
            raise ValueError("duplicate experience id")
        records = []
        for experience_id, advantage, expected_return in zip(experience_ids, advantages, returns):
            if experience_id not in self.experiences:
                raise KeyError(f"unknown or already-consumed experience {experience_id}")
            if not math.isfinite(float(advantage)) or not math.isfinite(float(expected_return)):
                raise ValueError("non-finite PPO target")
            records.append((experience_id, self.experiences[experience_id],
                            float(advantage), float(expected_return)))
        advantage_tensor = torch.tensor([row[2] for row in records], dtype=torch.float32)
        raw_advantages = advantage_tensor.clone()
        return_tensor = torch.tensor([row[3] for row in records], dtype=torch.float32)
        old_values = torch.tensor([row[1]['old_value'] for row in records], dtype=torch.float32)
        actor_update_requested = bool(torch.count_nonzero(raw_advantages))
        # Scale by RMS while preserving the sign. Mean-centering a batch made
        # entirely of failed episodes would turn some negative advantages into
        # positive ones and weaken the explicit failure penalty.
        advantage_rms = advantage_tensor.square().mean().sqrt()
        if float(advantage_rms) > 1e-8:
            advantage_tensor = advantage_tensor / (advantage_rms + 1e-8)
        records = [(*row[:2], float(advantage_tensor[index]), row[3])
                   for index, row in enumerate(records)]

        # A difficult batch may require a very small one-off line-search step.
        # Start the next independent PPO batch from the configured learning
        # rate again; the post-update KL check below still backtracks any batch
        # that exceeds the trust region.
        for group in self.actor_optimizer.param_groups:
            group["lr"] = self.nominal_actor_lr

        before = self.actor_parameters[0].detach().float().cpu().clone()
        actor_backup = [parameter.detach().clone() for parameter in self.actor_parameters]
        optimizer_backup = deepcopy(self.actor_optimizer.state_dict())
        epoch_reports = []
        optimizer_steps = 0
        rng = random.Random(20260917 + self.update_count)
        post_update_metrics = None
        accepted_actor_lr = None
        for epoch in range(self.update_epochs):
            order = list(range(len(records)))
            rng.shuffle(order)
            self.actor_optimizer.zero_grad(set_to_none=True)
            if self.critic_optimizer is not None:
                self.critic_optimizer.zero_grad(set_to_none=True)
            epoch_metrics = []
            for begin in range(0, len(order), self.microbatch_size):
                indices = order[begin:begin + self.microbatch_size]
                subset = [records[index] for index in indices]
                branch_batch = inferencer.prepare_training_batch(
                    [row[1]["observation"] for row in subset],
                    [row[1]["local_subgoal"] for row in subset],
                )
                chains = torch.stack([row[1]["chain"] for row in subset])
                old = torch.stack([row[1]["old_step_log_prob"] for row in subset])
                adv = torch.tensor([row[2] for row in subset])
                ret = torch.tensor([row[3] for row in subset])
                with inferencer._branch_context():
                    loss, metrics = self.evaluate_prepared(branch_batch, chains, old, adv, ret)
                # Before the first update, stored and recomputed likelihoods
                # must describe the identical behavior policy even after batching.
                if epoch == 0 and (not math.isfinite(metrics["approx_kl"]) or metrics["approx_kl"] > 0.005):
                    raise RuntimeError(f"Unchanged policy KL mismatch before optimizer step: {metrics}")
                (loss * (len(subset) / len(records))).backward()
                epoch_metrics.append(metrics)
            if self.critic is None or self.critic_optimizer is None:
                raise RuntimeError("critic was not initialized during update")
            frozen_grads = [name for name, parameter in self.policy.named_parameters()
                            if not parameter.requires_grad and parameter.grad is not None]
            if frozen_grads:
                raise RuntimeError(f"frozen parameters received gradients: {frozen_grads[:3]}")
            actor_grad = torch.nn.utils.clip_grad_norm_(self.actor_parameters, 1.0)
            critic_grad = torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
            if not torch.isfinite(actor_grad) or not torch.isfinite(critic_grad):
                raise RuntimeError("non-finite direct PPO gradient")
            mean_kl = sum(item["approx_kl"] for item in epoch_metrics) / len(epoch_metrics)
            epoch_reports.append({
                "epoch": epoch,
                "actor_grad_norm": float(actor_grad),
                "critic_grad_norm": float(critic_grad),
                "mean_approx_kl": mean_kl,
                "ratio_mean": sum(item["ratio_mean"] for item in epoch_metrics) / len(epoch_metrics),
            })
            if epoch > 0 and mean_kl > self.target_kl:
                break

        if self.update_epochs != 1:
            raise RuntimeError("KL-controlled direct PPO currently requires one update epoch")

        # Re-evaluate the stored chains after the optimizer step.  The ratio
        # measured before the first step is always one, so this is the useful
        # trust-region diagnostic for direct flow PPO.
        backtrack_reports = []
        for attempt in range(7 if actor_update_requested else 0):
            self.actor_optimizer.step()
            optimizer_steps += 1
            post_items = []
            with torch.no_grad():
                for begin in range(0, len(records), self.microbatch_size):
                    subset = records[begin:begin + self.microbatch_size]
                    branch_batch = inferencer.prepare_training_batch(
                        [row[1]["observation"] for row in subset],
                        [row[1]["local_subgoal"] for row in subset],
                    )
                    chains = torch.stack([row[1]["chain"] for row in subset])
                    old = torch.stack([row[1]["old_step_log_prob"] for row in subset])
                    adv = torch.tensor([row[2] for row in subset])
                    ret = torch.tensor([row[3] for row in subset])
                    with inferencer._branch_context():
                        _, metrics = self.evaluate_prepared(branch_batch, chains, old, adv, ret)
                    post_items.append(metrics)
            post_update_metrics = {
                "ratio_mean": sum(item["ratio_mean"] for item in post_items) / len(post_items),
                "mean_approx_kl": sum(item["approx_kl"] for item in post_items) / len(post_items),
                "clip_fraction": sum(item["clip_fraction"] for item in post_items) / len(post_items),
            }
            attempted_lr = float(self.actor_optimizer.param_groups[0]["lr"])
            changed_this_attempt = any(not torch.equal(p,saved) for p,saved in zip(self.actor_parameters,actor_backup))
            backtrack_reports.append({"attempt": attempt, "actor_lr": attempted_lr,
                                      "parameters_changed":changed_this_attempt, **post_update_metrics})
            if (math.isfinite(post_update_metrics["mean_approx_kl"])
                    and math.isfinite(post_update_metrics['ratio_mean'])
                    and post_update_metrics["mean_approx_kl"] <= self.target_kl
                    and post_update_metrics['clip_fraction'] <= self.max_clip_fraction
                    and changed_this_attempt):
                accepted_actor_lr = attempted_lr
                break
            # Restore the exact pre-update actor, discard Adam's unsafe moment
            # state, reduce the step, and retry with the already-clipped PPO
            # gradient.  The value head is updated only after actor acceptance.
            with torch.no_grad():
                for parameter, saved in zip(self.actor_parameters, actor_backup, strict=True):
                    parameter.copy_(saved)
            self.actor_optimizer.load_state_dict(deepcopy(optimizer_backup))
            for group in self.actor_optimizer.param_groups:
                group["lr"] = self.nominal_actor_lr * (0.5 ** (attempt + 1))
        else:
            if actor_update_requested:
                self.actor_optimizer.load_state_dict(deepcopy(optimizer_backup))
                raise RuntimeError(f"no safe direct PPO step after backtracking: {backtrack_reports}")
            # A legitimate one-chunk/zero-advantage terminal batch still trains
            # the critic and is consumed. Do not manufacture an actor update.
            accepted_actor_lr = 0.0
            post_update_metrics = dict(ratio_mean=1.0, mean_approx_kl=0.0, clip_fraction=0.0)
        self.critic_optimizer.step()
        critic_after=[]
        if self.reward_protocol=='skill_aligned_v1':
            with torch.no_grad():
                for begin in range(0,len(records),self.microbatch_size):
                    subset=records[begin:begin+self.microbatch_size]
                    batch=inferencer.prepare_training_batch([r[1]['observation'] for r in subset],
                                                             [r[1]['local_subgoal'] for r in subset])
                    with inferencer._branch_context():
                        _,features=self._prefix(batch)
                    critic_after.extend(self.critic(features).float().cpu().tolist())
        actor_changed = any(not torch.equal(p, saved) for p, saved in zip(self.actor_parameters, actor_backup))
        del actor_backup

        delta = float((before - self.actor_parameters[0].detach().float().cpu()).abs().max())
        if actor_update_requested and not actor_changed:
            raise RuntimeError("direct PPO optimizer did not change action_expert")
        if any(not torch.isfinite(parameter).all() for parameter in self.actor_parameters):
            raise RuntimeError("non-finite action expert after update")
        self.update_count += 1
        self.actor_update_count += int(actor_update_requested)
        def distribution(tensor):
            return dict(mean=float(tensor.mean()), std=float(tensor.std(unbiased=False)),
                        min=float(tensor.min()), max=float(tensor.max()),
                        positive_fraction=float((tensor>0).float().mean()))
        return_variance = float(return_tensor.var(unbiased=False))
        report: dict[str, Any] = {
            "update": self.update_count,
            "experiences": len(records),
            "optimizer_steps": optimizer_steps,
            "actor_updates": self.actor_update_count,
            "actor_updated": actor_update_requested,
            "terminal_small_batch_retained": len(records) < 16,
            "raw_advantages": distribution(raw_advantages),
            "returns": distribution(return_tensor),
            "old_values": distribution(old_values),
            "advantage_rms_before_normalization": float(advantage_rms),
            "explained_variance_old": (1.0-float((return_tensor-old_values).var(unbiased=False))/return_variance
                                       if return_variance>1e-12 else None),
            "policy_loss_before": sum(m['policy_loss'] for m in epoch_metrics)/len(epoch_metrics),
            "value_loss_before": sum(m['value_loss'] for m in epoch_metrics)/len(epoch_metrics),
            "max_reference_tensor_change": delta,
            "epochs": epoch_reports,
            "post_update": post_update_metrics,
            "accepted_actor_lr": accepted_actor_lr,
            "nominal_actor_lr": self.nominal_actor_lr,
            "backtracking": backtrack_reports,
            "cuda_allocated_bytes": torch.cuda.memory_allocated(),
            "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        }
        if critic_after:
            after_values=torch.tensor(critic_after,dtype=torch.float32)
            report['value_loss_after']=float(.5*(after_values-return_tensor).square().mean())
            report['explained_variance_after']=(1.-float((return_tensor-after_values).var(unbiased=False))/return_variance
                                                if return_variance>1e-12 else None)
        # Persist the configured starting rate in recovery checkpoints. The
        # accepted_actor_lr above records the actual line-search step.
        for group in self.actor_optimizer.param_groups:
            group["lr"] = self.nominal_actor_lr
        # The runner explicitly saves at exit; periodic saves occur every 5000 updates.
        torch.cuda.synchronize()
        report["compute_seconds"] = time.perf_counter() - update_started
        save_started = time.perf_counter()
        report["checkpoint"] = (
            self._save_checkpoint(report)
            if self.update_count % 5000 == 0
            else None
        )
        report["checkpoint_seconds"] = time.perf_counter() - save_started
        report["microbatch_size"] = self.microbatch_size
        for experience_id, *_ in records:
            del self.experiences[experience_id]
        with (self.output_dir / "updates.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(report) + "\n")
        return report


def make_direct_rl_inferencer(base_class, trainer: A4DirectPPO):
    """Extend the verified A4 preprocess/postprocess path without editing it."""
    class DirectRLInferencer(base_class):
        def prepare_training_batch(self, observations, local_subgoals):
            prepared = [self._prepare_native_low(obs, goal, num_obs_steps=6)
                        for obs, goal in zip(observations, local_subgoals, strict=True)]
            padding_id = prepared[0].prepared.sub_processor.pad_token_id
            batch = self._collate([entry.prepared.sample for entry in prepared],
                                  padding_input_id=padding_id)
            return self._branch_device(batch)

        def infer_native_low_with_timing(self, observations, local_subgoals, *, num_obs_steps):
            if not trainer.training_mode:
                return super().infer_native_low_with_timing(
                    observations, local_subgoals, num_obs_steps=num_obs_steps
                )
            started = time.monotonic()
            prepared = [self._prepare_native_low(obs, goal, num_obs_steps=num_obs_steps)
                        for obs, goal in zip(observations, local_subgoals, strict=True)]
            padding_id = prepared[0].prepared.sub_processor.pad_token_id
            batch = self._collate([entry.prepared.sample for entry in prepared],
                                  padding_input_id=padding_id)
            branch_batch = self._branch_device(batch)
            prepare_done = time.monotonic()
            with self._branch_context():
                generated = trainer.sample_branch(branch_batch, observations, local_subgoals)
            infer_done = time.monotonic()
            result_batch = _detach_cpu(generated)
            result_batch.setdefault("proprio", branch_batch["proprio"].detach().cpu())
            for key in ("action_dim_is_pad", "proprio_dim_is_pad"):
                if key in branch_batch and key not in result_batch:
                    result_batch[key] = _detach_cpu(branch_batch[key])
            grouped = []
            for index, entry in enumerate(prepared):
                kwargs = {}
                if entry.prepared.raw_state_anchor is not None:
                    kwargs["raw_state_anchor"] = entry.prepared.raw_state_anchor
                grouped.append(self._postprocess_single(
                    result_batch, index=index,
                    sub_processor=entry.prepared.sub_processor, **kwargs,
                ))
            self.last_direct_rl = {
                "experience_ids": list(generated["experience_ids"]),
                "old_log_prob": _detach_cpu(generated["old_log_prob"]).tolist(),
                "old_value": _detach_cpu(generated["old_value"]).tolist(),
            }
            finished = time.monotonic()
            return grouped, {
                "native_fm_requests": len(grouped),
                "selected_action_source": "direct_ppo_stochastic_fm",
                "preprocess_ms": (prepare_done - started) * 1000,
                "infer_ms": (infer_done - prepare_done) * 1000,
                "postprocess_ms": (finished - infer_done) * 1000,
                "model_horizon": int(result_batch["action"].shape[1]),
                "model_action_dim": int(result_batch["action"].shape[2]),
            }

    return DirectRLInferencer
