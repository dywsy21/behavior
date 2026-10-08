"""Shared parameters; strictly episode-local planner state and B1 flow likelihoods."""
import json
from pathlib import Path
import sys

import torch

from checkpoint_io import atomic_json
from shared_ppo import SharedFlowPPO
from stage1_engine import Stage1Engine, ROOT
from training_config import load_training_config

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "eval/memlite_sft100"))
from batch_core import validate_indices, needs_plan, stage_plans, commit_plans
from sparse_cache import indexed_cache_freeze
from g05.models.kv_cache import SparseKVCache


class SharedStage1Engine(Stage1Engine):
    def configure_shared(self, collective, manifest):
        self.collective = collective
        self.shared_manifest = manifest
        self.session_generation = 0
        self.requests = 0
        self.high_batch = manifest["training"]["high_batch"]
        settings = load_training_config()
        config = json.loads((ROOT / "configs/low_model.local.json").read_text())
        pad = torch.zeros(27, dtype=torch.bool)
        pad[[7, 8, 17, 18]] = True
        self.trainer = SharedFlowPPO(self.models["low"], pad, self.output,
            collective=collective, task_weight_map=manifest["task_weights"],
            feature_dim=config["arch"]["vlm"]["hidden_size"] + 1,
            checkpoint_interval=manifest["training"]["checkpoint_interval"],
            microbatch_size=1, actor_lr=settings["actor_lr"], critic_lr=settings["critic_lr"],
            target_kl=settings["target_kl"], max_clip_fraction=settings["max_clip_fraction"])
        if manifest.get("resume_checkpoint"):
            receipt = self.trainer.load_checkpoint(Path(manifest["resume_checkpoint"]))
            atomic_json(self.output.parent / "resume_receipt.json", receipt)
        assert all(not p.requires_grad for p in self.models["high"].parameters())

    def begin(self, task, num_envs, seed, episode_metadata=None):
        if not episode_metadata or len(episode_metadata) != num_envs:
            raise ValueError("Missing per-environment episode identity")
        ids = [m["episode_id"] for m in episode_metadata]
        if len(set(ids)) != num_envs or any(m["task"] != task or m["split"] != "train" or
                m["gpu"] != self.collective.rank for m in episode_metadata):
            raise ValueError("Cross-task/rank or nontrain episode registration")
        allowed = {t["task"]: t for t in self.shared_manifest["groups"][self.collective.rank]["tasks"]}
        if task not in allowed or any(m["instance_id"] not in allowed[task]["train_instances"] for m in episode_metadata):
            raise ValueError("Task/instance outside this rank's sealed TRAIN partition")
        super().begin(task, num_envs, seed, episode_metadata)
        self.session_generation += 1
        for slot in self.slots:
            memory = json.loads(slot["ledger"].memory)
            if (memory["task_name"] != self.task or memory["issued_command_history"] or
                    memory["verified_world_facts"] or slot["ledger"].revision != 0 or
                    slot["projection"] is not None or slot["context_id"] is not None):
                raise RuntimeError("Reset retained memory/intent from another episode")
        if len({id(slot["ledger"]) for slot in self.slots}) != num_envs:
            raise RuntimeError("Aliased planner ledgers")
        with (self.output.parent / "session_begin.jsonl").open("a") as stream:
            stream.write(json.dumps(dict(rank=self.collective.rank, generation=self.session_generation,
                task=task, episode_ids=ids, all_memories_empty=True, independent_ledgers=True,
                policy_version=self.trainer.update_count)) + "\n")

    def prepare(self, side, observations, projections):
        for observation, projection in zip(observations, projections, strict=True):
            if (observation.get("task_identity", "").replace("_", " ") != self.task or
                    projection["task_name"] != self.task or projection["memlite_branch"] != side):
                raise ValueError("Cross-task or high/low branch conditioning")
        return super().prepare(side, observations, projections)

    def validate_observations(self, observations, indices):
        validate_indices(observations, indices, len(self.slots))
        for observation, index in zip(observations, indices, strict=True):
            metadata = self.episode_metadata[index]
            if (observation.get("task_identity") != metadata["task"] or
                    observation.get("episode_id") != metadata["episode_id"]):
                raise ValueError("Observation is bound to another environment/task")

    @torch.no_grad()
    def plan_batch(self, observations, indices):
        batch = self.prepare("high", observations, [self.slots[i]["ledger"].projection() for i in indices])
        with self._branch_context(), indexed_cache_freeze(self.models["high"].model.ar_helper, SparseKVCache):
            result = self.models["high"].generate_high_level(batch["samples"], batch["pixel_values"], temperature=0.)
        staged = stage_plans(self.slots, indices, result["planner_events"])
        with (self.output.parent / "planner_events.jsonl").open("a") as stream:
            for row, observation in zip(staged, observations, strict=True):
                index = row["index"]
                stream.write(json.dumps(dict(task=self.task, env=index, rank=self.collective.rank,
                    chunk=self.slots[index]["chunks"], event=row["event"], context_id=row["context_id"],
                    episode=self.episode_metadata[index], control_step=observation.get("control_step"),
                    policy_update=self.trainer.update_count, high_batch_size=len(indices))) + "\n")
        commit_plans(self.slots, staged)

    def ensure_batch(self, observations, indices):
        self.validate_observations(observations, indices)
        for observation in observations:
            observation["task"] = self.task
        due = [row for row, index in enumerate(indices) if needs_plan(self.slots[index])]
        if due and self.high_batch:
            self.plan_batch([observations[row] for row in due], [indices[row] for row in due])

    def infer(self, observations, indices):
        self.ensure_batch(observations, indices)
        result = super().infer(observations, indices)
        for identity, observation, index in zip(result["experience_ids"], observations, indices, strict=True):
            self.trainer.experiences[identity].update(policy_version=self.trainer.update_count,
                task_identity=observation["task_identity"], episode_id=observation["episode_id"],
                env_index=index, worker_id=self.collective.rank)
        if len(observations) == 1:
            self.requests += 1
        return result

    def value(self, observations, indices):
        self.ensure_batch(observations, indices)
        return super().value(observations, indices)
