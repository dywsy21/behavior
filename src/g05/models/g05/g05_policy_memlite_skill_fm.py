"""Skill-conditioned, genuinely trainable continuous-FM MEM-Lite controller.

Unlike the v11 intent-residual experiment, the structured current skill enters
the regular VLM prefix and the FM loss optimizes the actual action expert.
"""
from __future__ import annotations

from collections import defaultdict
import json
from typing import Mapping

import torch

from g05.data_processor.processor.memlite_v6_projection import (
    validate_embedded_model_projection,
)
from .g05_policy_qwen35 import G05PolicyQwen35
from .helpers.vlm_lora import (
    VLMloraConfig,
    inject_vlm_lora,
    normalize_pre_injection_state_keys,
    restore_lora_state,
)
from g05.utils.memlite_skill_protocol import MEMLITE_SKILL_SCHEMA_VERSION


class G05PolicyMEMLiteSkillFM(G05PolicyQwen35):
    """FM policy whose VLM prefix includes a canonical active-skills bundle.

    ``memlite_train_mode`` stays ``off`` intentionally: the inherited ``low``
    mode is an AR/CE-only route.  This class performs its own low-branch checks
    before delegating to the normal continuous-FM training path.
    """

    def __init__(self, **model_cfg):
        # This minimal Git port is admitted for the single-frame benchmark.
        # The frozen original has additional history-aware builder/vision/
        # training-loader changes; importing this class alone does not port
        # that complete pipeline. Refuse six-frame use until P0-02 is done.
        if model_cfg.get("num_obs_steps") != 1 or isinstance(model_cfg.get("num_obs_steps"), bool):
            raise ValueError("The current SkillFM Git port supports only the audited single-frame route")
        super().__init__(**model_cfg)
        if self.discrete_action or not self.continuous_action or self.predict_cot:
            raise ValueError(
                "SkillFM requires continuous_action=true, discrete_action=false, predict_cot=false"
            )
        if self.memlite_train_mode != "off":
            raise ValueError(
                "SkillFM must use memlite_train_mode=off; it validates low rows then uses the real FM route"
            )
        self.coordination_train = dict(model_cfg.get("coordination_train", {}))
        self.interface_schema_version = int(self.coordination_train.get("interface_schema_version", 6))
        if self.interface_schema_version != MEMLITE_SKILL_SCHEMA_VERSION:
            raise ValueError(
                "SkillFM model/data schema mismatch: "
                f"config={self.interface_schema_version} protocol={MEMLITE_SKILL_SCHEMA_VERSION}"
            )
        self.stage = str(self.coordination_train.get("stage", "skill_fm_stage_a"))
        if self.stage not in {"skill_fm_stage_a", "skill_fm_stage_b"}:
            raise ValueError(f"Unsupported SkillFM coordination stage {self.stage!r}")
        self.pipeline_stage = str(self.coordination_train.get("pipeline_stage", ""))
        expected_pipeline_stage = {
            "skill_fm_stage_a": "A",
            # This run-role name is intentionally not the global pipeline
            # stage.  LoRA/history is the second low-level optimization phase
            # of pipeline A; pipeline B is the high planner/outcome phase.
            "skill_fm_stage_b": "A",
        }[self.stage]
        if self.pipeline_stage != expected_pipeline_stage:
            raise ValueError(
                f"SkillFM stage {self.stage} requires pipeline_stage={expected_pipeline_stage!r}"
            )
        self.component = str(self.coordination_train.get("component", ""))
        self.trainability_profile = str(self.coordination_train.get("trainability_profile", ""))
        expected_profile = {
            "skill_fm_stage_a": "low_ae",
            "skill_fm_stage_b": "low_ae_lora_history",
        }[self.stage]
        if self.component != "low" or self.trainability_profile != expected_profile:
            raise ValueError(
                f"SkillFM stage {self.stage} requires component='low' and "
                f"trainability_profile={expected_profile!r}"
            )
        self.low_vlm_lora = VLMloraConfig.from_mapping(model_cfg.get("low_vlm_lora"))
        if self.stage == "skill_fm_stage_a" and self.low_vlm_lora.enabled:
            raise ValueError("Stage A must not enable VLM LoRA")
        if self.stage == "skill_fm_stage_b" and not self.low_vlm_lora.enabled:
            raise ValueError("Stage B requires low_vlm_lora.enabled=true")
        # The generic loader does not retain policy-specific post-load data.
        # Keep the receipt locally so the optimizer gate can prove that LoRA
        # injection/restoration happened first.
        self._coordination_post_load_receipt = None

    def train(self, mode: bool = True):
        super().train(mode)
        # Frozen modules have no useful train-mode behavior in Stage A.  Keeping
        # them in eval prevents accidental dropout/statistics drift while the
        # action expert remains trainable.
        if mode and self.stage == "skill_fm_stage_a":
            self.model.vision_tower.eval()
            self.model.vlm.eval()
            if self.model.proprio_embedder is not None:
                self.model.proprio_embedder.eval()
        return self

    def _validate_skill_batch(self, samples, actions, action_pad_masks) -> None:
        self._memlite_branch_masks(samples, "low")
        if actions is None or action_pad_masks is None or actions.ndim != 3:
            raise ValueError("SkillFM requires normalized continuous actions [batch,horizon,action_dim]")
        if actions.shape[-1] != int(self.model_config.action_dim):
            raise ValueError(
                f"SkillFM action_dim={actions.shape[-1]} != checkpoint config {self.model_config.action_dim}"
            )
        if action_pad_masks.shape != actions.shape[:2] or action_pad_masks.all():
            raise ValueError("SkillFM requires nonempty [batch,horizon] action padding masks")
        for index, sample in enumerate(samples):
            if int(sample.get("memlite_schema_version", sample.get("schema_version", -1))) != self.interface_schema_version:
                raise ValueError(f"SkillFM sample {index} has incompatible schema version")
            try:
                # The builder validates the exact 19-field data projection;
                # this final gate extracts that projection from its generated
                # template/image container and validates it again.
                validate_embedded_model_projection(sample)
            except (TypeError, ValueError) as error:
                raise ValueError(f"SkillFM sample {index} carries an audit payload") from error
            raw_bundle = sample.get("active_skills_semantic_json")
            rendered_bundle = sample.get("active_skills_text")
            try:
                parsed_bundle = json.loads(raw_bundle)
            except (TypeError, json.JSONDecodeError) as error:
                raise ValueError(f"SkillFM sample {index} has invalid semantic active-skills JSON") from error
            if not isinstance(parsed_bundle, list) or not parsed_bundle:
                raise ValueError("SkillFM needs a nonempty semantic active-skills list")
            allowed_fields = {"verb", "target", "source", "destination", "target_part", "arm", "unbound_relation"}
            if any(
                not isinstance(skill, dict)
                or set(skill) != allowed_fields
                for skill in parsed_bundle
            ):
                raise ValueError("SkillFM semantic skills contain an audit field or malformed leaf")
            if not isinstance(rendered_bundle, str) or not rendered_bundle.startswith("Active skills:"):
                raise ValueError(
                    "SkillFM requires semantic active-skills JSON/text in the VLM prefix"
                )
            if any(skill["verb"] == "SKILL_UNKNOWN" for skill in parsed_bundle):
                raise ValueError("SkillFM refuses unknown active skills as BC conditions")
            if bool(sample.get("task_complete", False)) or str(sample.get("next_decision", "")).upper() != "EXECUTE":
                raise ValueError("SkillFM refuses terminal/STOP rows")
            template = str(sample.get("template", ""))
            if (
                "<active_skills_text_text_!>" not in template
                or "<EOC>" not in template
                or template.index("<active_skills_text_text_!>") > template.index("<EOC>")
            ):
                raise ValueError("SkillFM active_skills_text must be a masked EOC-prefix input")
            if "<parent_goal_text_!>" not in template or template.index("<parent_goal_text_!>") > template.index("<EOC>"):
                raise ValueError("SkillFM parent_goal must be a masked EOC-prefix input")
            if "<action_action>" in template:
                raise ValueError("SkillFM template must not construct discrete ActionCodec targets")

    def _assert_stage_freeze_contract(self) -> None:
        expected = {"action_expert"}
        if self.stage == "skill_fm_stage_b":
            expected.add("vlm_lora")
        actual = set(self.coordination_trainable_parameter_groups())
        if actual != expected:
            raise RuntimeError(
                f"SkillFM {self.stage} trainable groups must be {sorted(expected)}, "
                f"found {sorted(actual)}. Call configure_coordination_trainability "
                "after checkpoint/LoRA restoration and before DDP."
            )

    def configure_coordination_trainability(self):
        """Fail-closed stage gate called by the training loader before DDP.

        Stage A updates exactly the pre-existing action expert.  Stage B keeps
        that expert plus PEFT tensors, freezing all vision, VLM base, proprio,
        and projector weights.  This cannot be expressed by a prefix-only
        freezer because LoRA lives below ``model.vlm``.
        """
        if self._coordination_post_load_receipt is None:
            raise RuntimeError(
                "SkillFM configure_coordination_trainability requires post_checkpoint_load first"
            )
        if self.stage == "skill_fm_stage_b" and not any(
            "lora_" in name for name, _ in self.model.vlm.named_parameters()
        ):
            raise RuntimeError("SkillFM Stage B needs post_checkpoint_load LoRA injection before stage gating")
        enabled_names = []
        for name, parameter in self.named_parameters():
            enabled = name.startswith("model.action_expert.")
            if self.stage == "skill_fm_stage_b":
                enabled = enabled or (name.startswith("model.vlm.") and "lora_" in name)
            parameter.requires_grad_(enabled)
            if enabled:
                enabled_names.append(name)
        groups = self.coordination_trainable_parameter_groups()
        expected = {"action_expert"} | ({"vlm_lora"} if self.stage == "skill_fm_stage_b" else set())
        if set(groups) != expected:
            raise RuntimeError(f"SkillFM stage gate failed: expected={sorted(expected)} actual={sorted(groups)}")
        receipt = {
            "stage": self.stage,
            "pipeline_stage": self.pipeline_stage,
            "component": self.component,
            "trainability_profile": self.trainability_profile,
            "interface_schema_version": self.interface_schema_version,
            "num_obs_steps": int(self.model_config.num_obs_steps),
            "expected_trainable_groups": sorted(expected),
            "trainable_parameter_names": enabled_names,
            "frozen_parameter_count": sum(1 for parameter in self.parameters() if not parameter.requires_grad),
        }
        receipt["post_checkpoint_load"] = dict(self._coordination_post_load_receipt)
        receipt["adapter_load_mode"] = str(
            self._coordination_post_load_receipt.get("adapter_load_mode", "missing")
        )
        return receipt

    def forward_train(self, samples, pixel_values, actions=None, action_pad_masks=None,
                      action_dim_is_pad=None, **kwargs):
        self._validate_skill_batch(samples, actions, action_pad_masks)
        self._assert_stage_freeze_contract()
        # G05Policy.forward_train is the audited encode_train -> G05Model ->
        # FMHelper graph.  With memlite_train_mode=off it cannot fall into the
        # CE-only MEM-Lite branch.
        return super().forward_train(
            samples=samples,
            pixel_values=pixel_values,
            actions=actions,
            action_pad_masks=action_pad_masks,
            action_dim_is_pad=action_dim_is_pad,
            **kwargs,
        )

    def coordination_trainable_parameter_groups(self):
        groups = defaultdict(list)
        for name, parameter in self.named_parameters():
            if not parameter.requires_grad:
                continue
            if name.startswith("model.action_expert."):
                key = "action_expert"
            elif "model.vlm." in name and "lora_" in name:
                key = "vlm_lora"
            elif name.startswith("model.vlm."):
                key = "vlm_base"
            elif name.startswith("model.vision_tower."):
                key = "vision"
            else:
                key = "other"
            groups[key].append((name, parameter))
        return dict(groups)

    def remap_checkpoint_state_dict(self, state_dict: Mapping[str, torch.Tensor]):
        """Hook for the loader before base weights are restored."""
        return normalize_pre_injection_state_keys(state_dict)

    def post_checkpoint_load(self, state_dict: Mapping[str, torch.Tensor], checkpoint_info=None):
        """Hook for the loader after base load, before freezing/DDP/optimizer."""
        if not self.low_vlm_lora.enabled:
            receipt = {
                "enabled": False,
                "adapter_load_mode": "disabled",
                "component": self.component,
                "trainability_profile": self.trainability_profile,
            }
            self._coordination_post_load_receipt = dict(receipt)
            return receipt
        self.model.vlm = inject_vlm_lora(self.model.vlm, self.low_vlm_lora)
        receipt = restore_lora_state(self, state_dict)
        receipt.update(
            enabled=True,
            adapter_name=self.low_vlm_lora.adapter_name,
            component=self.component,
            trainability_profile=self.trainability_profile,
        )
        self._coordination_post_load_receipt = dict(receipt)
        return receipt
