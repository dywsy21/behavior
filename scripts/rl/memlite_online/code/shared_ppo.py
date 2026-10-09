"""One low-level policy and critic, trained by all synchronous rollout ranks."""
from copy import deepcopy
import json
import math
import os
import random
import time

import numpy as np
import torch

from checkpoint_io import atomic_json
from direct_a4_flow import A4DirectPPO, ValueHead
from synchronous import global_accept, state_digest


class SharedFlowPPO(A4DirectPPO):
    def __init__(self, *args, collective, task_weight_map, feature_dim,
                 checkpoint_interval=10, **kwargs):
        super().__init__(*args, **kwargs)
        self.collective = collective
        self.task_weight_map = dict(task_weight_map)
        self.checkpoint_interval = checkpoint_interval
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.critic = ValueHead(feature_dim).to(collective.device)
        self.critic_optimizer = torch.optim.AdamW(self.critic.parameters(),
                                                 lr=self.critic_lr, weight_decay=0.)
        self.shared_checkpoint_state = None
        self.last_shared_receipt = None
        self.initial_update = 0
        collective.same(dict(actor=self.actor_names, shapes=[tuple(p.shape) for p in self.actor_parameters],
            weights=self.task_weight_map, actor_lr=self.nominal_actor_lr, critic_lr=self.critic_lr,
            target_kl=self.target_kl, clip=self.clip_ratio, noise=self.transition_std,
            feature_dim=feature_dim), "initial configuration")
        collective.broadcast_parameters(self.actor_parameters)
        collective.broadcast_parameters(list(self.critic.parameters()))
        self.verify_replicas("initial")

    def verify_replicas(self, phase):
        values = dict(actor=state_digest(self.policy.model.action_expert.state_dict()),
            critic=state_digest(self.critic.state_dict()),
            actor_optimizer=state_digest(self.actor_optimizer.state_dict()),
            critic_optimizer=state_digest(self.critic_optimizer.state_dict()),
            updates=self.update_count, actor_updates=self.actor_update_count)
        self.collective.same(values, phase + " parameter/optimizer")
        atomic_json(self.output_dir.parent / "replica_identity.json", dict(
            phase=phase, rank=self.collective.rank, verified=time.time(), **values))
        return values

    def load_checkpoint(self, path):
        payload = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
        shared = payload.get("distributed_state")
        if not shared or shared["world_size"] != self.collective.world_size:
            raise ValueError("Resume requires a shared checkpoint with the same world size")
        if shared["task_weight_map"] != self.task_weight_map:
            raise ValueError("Resume task weighting changed")
        result = super().load_checkpoint(path)
        saved = shared["rank_states"][self.collective.rank]
        random.setstate(saved["python_rng"])
        np.random.set_state(saved["numpy_rng"])
        torch.set_rng_state(saved["torch_rng_cpu"].cpu())
        torch.cuda.set_rng_state(saved["torch_rng_cuda"].cpu())
        self.next_experience_id = saved["next_experience_id"]
        self.shared_checkpoint_state = shared
        self.initial_update = self.update_count
        result["replica_identity"] = self.verify_replicas("resumed")
        result["environment_resume"] = "reset_all_envs_discard_uncommitted_rollouts"
        return result

    def _save_checkpoint(self, metrics):
        # No collectives here: shutdown can arrive independently on each rank.
        # All published checkpoints describe a previously committed shared round.
        if self.collective.rank != 0:
            return self.last_shared_receipt or {"shared_writer_rank": 0, "updates": self.update_count}
        if self.shared_checkpoint_state is None:
            raise RuntimeError("No globally committed round to checkpoint")
        self.checkpoint_extra = {"distributed_state": self.shared_checkpoint_state}
        result = super()._save_checkpoint(metrics)
        result["shared_writer_rank"] = 0
        if self.update_count == self.initial_update + 2 or self.update_count % 100 == 0:
            milestone = self.output_dir / f"shared_update_{self.update_count:06d}.pt"
            if not milestone.exists():
                os.link(result['latest'], milestone)
            result['shared_milestone'] = str(milestone)
        return result

    def update(self, inferencer, experience_ids, advantages, returns):
        c = self.collective
        started = time.monotonic()
        records, error = [], None
        try:
            if len(experience_ids) != len(advantages) or len(experience_ids) != len(returns):
                raise ValueError("Unaligned update arrays")
            if len(set(experience_ids)) != len(experience_ids):
                raise ValueError("Duplicate experience")
            for identity, advantage, target in zip(experience_ids, advantages, returns, strict=True):
                record = self.experiences[identity]
                if record["policy_version"] != self.update_count:
                    raise ValueError("Stale rollout policy version")
                task = record["task_identity"]
                if not math.isfinite(advantage) or not math.isfinite(target):
                    raise ValueError("Nonfinite PPO targets")
                records.append((identity, record, float(advantage), float(target), self.task_weight_map[task]))
            if set(experience_ids) != set(self.experiences):
                raise ValueError("Update would leave stale unconsumed experience")
        except Exception as exception:
            error = exception
        c.check(error, "rollout admission")
        c.same((self.update_count, self.actor_update_count), "policy version")
        weight, square, count, nonzero = c.sum([
            sum(row[4] for row in records), sum(row[4] * row[2] ** 2 for row in records),
            len(records), sum(row[2] != 0 for row in records)])
        if not count or weight <= 0:
            raise ValueError("No global on-policy samples")
        rms = math.sqrt(square / weight)
        actor_requested = bool(nonzero)
        raw_advantages = [row[2] for row in records]
        records = [(i, row, adv / (rms + 1e-8) if rms > 1e-8 else adv, target, w)
                   for i, row, adv, target, w in records]
        for group in self.actor_optimizer.param_groups:
            group["lr"] = self.nominal_actor_lr
        actor_backup = [p.detach().clone() for p in self.actor_parameters]
        optimizer_backup = deepcopy(self.actor_optimizer.state_dict())
        self.actor_optimizer.zero_grad(set_to_none=True)
        self.critic_optimizer.zero_grad(set_to_none=True)

        critic_features = {}

        def compute(record, backward=False):
            _, saved, adv, target, row_weight = record
            batch = inferencer.prepare_training_batch([saved["observation"]], [saved["local_subgoal"]])
            captured = []
            hook = (self.critic.register_forward_pre_hook(
                lambda _module, inputs: captured.append(inputs[0].detach().clone())) if backward else None)
            try:
                with inferencer._branch_context():
                    loss, metrics = self.evaluate_prepared(batch, saved["chain"].unsqueeze(0),
                        saved["old_step_log_prob"].unsqueeze(0), torch.tensor([adv]), torch.tensor([target]))
            finally:
                if hook is not None:
                    hook.remove()
            if backward:
                if len(captured) != 1 or captured[0].shape[0] != 1:
                    raise RuntimeError("Expected one detached critic feature per collected chunk")
                critic_features[record[0]] = captured[0]
                if not all(math.isfinite(value) for value in metrics.values()) or metrics["approx_kl"] > 0.005:
                    raise RuntimeError(f"Unchanged-policy likelihood mismatch: {metrics}")
                (loss * row_weight).backward()
            return metrics

        before_metrics, error = [], None
        try:
            order = list(range(len(records)))
            random.Random(20261008 + self.update_count + c.rank).shuffle(order)
            for index in order:
                before_metrics.append((records[index][4], compute(records[index], True)))
            if any(p.grad is not None for p in self.policy.parameters() if not p.requires_grad):
                raise RuntimeError("Frozen VLM/planner received gradient")
            if any(p.grad is not None and not torch.isfinite(p.grad).all()
                   for p in self.actor_parameters + list(self.critic.parameters())):
                raise RuntimeError("Nonfinite local gradient")
        except Exception as exception:
            error = exception
        c.check(error, "local backward")
        communication = c.gradients(self.actor_parameters, denominator=weight)
        communication += c.gradients(list(self.critic.parameters()), denominator=weight)
        actor_norm = float(torch.nn.utils.clip_grad_norm_(self.actor_parameters, 1.0))
        critic_norm = float(torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0))
        c.check(None if math.isfinite(actor_norm) and math.isfinite(critic_norm)
                else ValueError("Nonfinite global gradient"), "gradient clipping")

        def measured(items):
            local_weight = sum(w for w, _ in items)
            fields = ["approx_kl", "clip_fraction", "ratio_mean", "policy_loss", "value_loss"]
            totals = c.sum([local_weight] + [sum(w * m[key] for w, m in items) for key in fields])
            global_metrics = dict(zip(fields, [v / totals[0] for v in totals[1:]], strict=True))
            local = dict(rank=c.rank, experiences=len(items),
                         tasks=sorted({row[1]["task_identity"] for row in records}))
            for key in fields:
                local[key] = sum(w * m[key] for w, m in items) / local_weight if local_weight else 0.
            local["mean_approx_kl"] = local.pop("approx_kl")
            global_metrics["mean_approx_kl"] = global_metrics.pop("approx_kl")
            return global_metrics, c.objects(local)

        before, _ = measured(before_metrics)
        backtracks = []
        post = dict(mean_approx_kl=0., clip_fraction=0., ratio_mean=1.)
        group_metrics = []
        accepted_lr = 0.
        for attempt in range(7 if actor_requested else 0):
            error, post_items = None, []
            try:
                self.actor_optimizer.step()
                with torch.no_grad():
                    post_items = [(row[4], compute(row)) for row in records]
            except Exception as exception:
                error = exception
            c.check(error, "post-update recomputation")
            post, group_metrics = measured(post_items)
            changed = any(not torch.equal(p, saved) for p, saved in zip(self.actor_parameters, actor_backup))
            changes = c.objects(changed)
            accepted = all(changes) and global_accept(post, group_metrics,
                target_kl=self.target_kl, max_clip_fraction=self.max_clip_fraction)
            attempted_lr = float(self.actor_optimizer.param_groups[0]["lr"])
            backtracks.append(dict(attempt=attempt, actor_lr=attempted_lr, accepted=accepted,
                                   global_metrics=post, groups=group_metrics))
            c.same(accepted, "trust-region decision")
            if accepted:
                accepted_lr = attempted_lr
                break
            with torch.no_grad():
                for parameter, saved in zip(self.actor_parameters, actor_backup, strict=True):
                    parameter.copy_(saved)
            self.actor_optimizer.load_state_dict(deepcopy(optimizer_backup))
            for group in self.actor_optimizer.param_groups:
                group["lr"] = self.nominal_actor_lr * 0.5 ** (attempt + 1)
        else:
            if actor_requested:
                self.actor_optimizer.load_state_dict(optimizer_backup)
                raise RuntimeError(f"All shared line-search attempts rejected: {backtracks}")
        def critic_diagnostics():
            # Cached features come from the frozen observable prefix. Measuring
            # the small critic here does not repeat the VLM / ten FM steps.
            sums = [0.] * 7
            error = None
            try:
                with torch.no_grad():
                    for identity, _, _, target, w in records:
                        value = float(self.critic(critic_features[identity]).item())
                        residual = value - target
                        values = [w, w*target, w*target*target, w*value, w*value*value,
                                  w*residual, w*residual*residual]
                        sums = [a+b for a, b in zip(sums, values)]
            except Exception as exception:
                error = exception
            c.check(error, "local critic diagnostics")
            totals = c.sum(sums)
            weight_sum = totals[0]
            target_mean, target_square, value_mean, value_square, error_mean, error_square = [
                v/weight_sum for v in totals[1:]]
            variance = max(0., target_square - target_mean**2)
            result = dict(return_mean=target_mean, return_std=math.sqrt(variance),
                value_mean=value_mean, value_std=math.sqrt(max(0., value_square-value_mean**2)),
                half_mse=.5*error_square,
                explained_variance=(1.-max(0., error_square-error_mean**2)/variance
                                    if variance > 1e-12 else None),
                weighted=True, features="cached_frozen_observable_prefix")
            c.check(None if all(v is None or not isinstance(v, float) or math.isfinite(v)
                                for v in result.values()) else ValueError("Nonfinite critic diagnostics"),
                    "critic diagnostics")
            return result

        critic_before = critic_diagnostics()
        self.critic_optimizer.step()
        critic_after = critic_diagnostics()
        # Actor trust-region metrics above intentionally precede the critic
        # update. The public post_update/value_loss now really follows it.
        post, group_metrics = dict(post), deepcopy(group_metrics)
        post["value_loss"] = critic_after["half_mse"]
        for item in group_metrics:
            item["value_loss_before_critic_step"] = item.pop("value_loss")
        delta = max(float((p - saved).abs().max()) for p, saved in zip(self.actor_parameters, actor_backup))
        del actor_backup, optimizer_backup
        self.update_count += 1
        self.actor_update_count += int(actor_requested)
        for group in self.actor_optimizer.param_groups:
            group["lr"] = self.nominal_actor_lr
        c.check(None if all(torch.isfinite(p).all() for p in self.actor_parameters + list(self.critic.parameters()))
                else ValueError("Nonfinite updated weights"), "parameter commit")
        for identity in experience_ids:
            del self.experiences[identity]
        c.same((self.update_count, self.actor_update_count), "committed version")
        report = dict(update=self.update_count, actor_updates=self.actor_update_count, actor_updated=actor_requested,
            rank=c.rank, world_size=c.world_size, experiences=len(records), global_experiences=int(count),
            global_weight=weight, advantage_rms_before_normalization=rms, raw_advantages=raw_advantages,
            actor_grad_norm=actor_norm, critic_grad_norm=critic_norm, actor_max_delta=delta,
            accepted_actor_lr=accepted_lr, nominal_actor_lr=self.nominal_actor_lr,
            before=before, post_update=post, groups=group_metrics, backtracking=backtracks,
            critic_before=critic_before, critic_after=critic_after,
            communication_seconds=communication, compute_seconds=time.monotonic() - started,
            microbatch_size=1, collected_policy_version=self.update_count - 1,
            tasks=sorted({row[1]["task_identity"] for row in records}))
        should_save = self.update_count <= self.initial_update + 2 or self.update_count % self.checkpoint_interval == 0
        if should_save:
            report["replica_identity"] = self.verify_replicas("update_" + str(self.update_count))
        rank_state = dict(rank=c.rank, torch_rng_cpu=torch.get_rng_state(),
            torch_rng_cuda=torch.cuda.get_rng_state(), python_rng=random.getstate(),
            numpy_rng=np.random.get_state(), next_experience_id=self.next_experience_id,
            episodes=deepcopy(getattr(inferencer, "episode_metadata", [])))
        self.shared_checkpoint_state = dict(world_size=c.world_size, task_weight_map=self.task_weight_map,
            rank_states=c.objects(rank_state), committed_update=self.update_count,
            resume_contract="reset_all_envs_discard_uncommitted_rollouts")
        if should_save:
            error, receipt = None, None
            try:
                if c.rank == 0:
                    receipt = self._save_checkpoint(report)
            except Exception as exception:
                error = exception
            c.check(error, "checkpoint publication")
            self.last_shared_receipt = c.objects(receipt)[0]
            report["checkpoint"] = self.last_shared_receipt
        report["total_seconds"] = time.monotonic() - started
        atomic_json(self.output_dir.parent / "latest_update.json", report)
        with (self.output_dir / "updates.jsonl").open("a") as stream:
            stream.write(json.dumps(report, allow_nan=False) + "\n")
        return report
