"""Experimental G0.5 action-token / CE+FM / knowledge-insulation controller.

Run against the identity-checked A3 runtime plus these explicit Git extensions.
This is not a flag change to SkillFM and does not loosen that policy's guards.
Pure AR, joint CE+FM and KI have distinct parameter/gradient contracts. The
planner stays separate. Task-only AR is an explicit conditioning ablation,
not a claim of reproducing upstream G0.5's complete CoT training recipe.
"""
from __future__ import annotations

from collections import defaultdict
import time

import torch

from .g05_model_qwen35 import G05ModelQwen35
from .g05_policy_qwen35 import G05PolicyQwen35
from .g05_policy_memlite_skill_fm import G05PolicyMEMLiteSkillFM
from .helpers.vlm_lora import VLMloraConfig, inject_vlm_lora, normalize_pre_injection_state_keys, restore_lora_state
from g05.data_processor.processor.samples_builder import validate_embedded_model_projection
from g05.utils.memlite_skill_protocol import MEMLITE_SKILL_SCHEMA_VERSION
from g05.utils.training.ar_training_methods import (
    ActionTrainingSettings, action_prefix_samples, action_training_samples, validate_complete_action_tokens,
    native_task_actor_samples, native_subtask_cot_actor_samples, action_token_loss_metrics,
)


class ActionTrainingModelQwen35(G05ModelQwen35):
    def forward(self, input_ids, *args, **kwargs):
        if args:
            raise TypeError("The audited policy passes model inputs as explicit keyword arguments")
        if getattr(self, "_action_suffix_boundary_scope", False):
            raise RuntimeError("Concurrent/nested action-training model forwards are unsupported")
        required = (kwargs.get("continuous_action", False)
                    and input_ids.shape[1] > kwargs["split_index"])
        self._action_suffix_boundary_scope = True
        self._require_action_suffix_boundary = required
        try:
            return super().forward(input_ids=input_ids, **kwargs)
        finally:
            del self._require_action_suffix_boundary
            del self._action_suffix_boundary_scope

    def _build_prefix_action_kv(self, vlm_kv, split_index):
        # KV slicing alone is insufficient for Qwen's linear-attention layers.
        # During a joint forward the full-sequence recurrent state includes
        # teacher-forced expert action tokens. Never fall back to that state.
        need_boundary = (getattr(self, "_require_action_suffix_boundary", False)
                         or vlm_kv.num_items() > split_index)
        if need_boundary and vlm_kv.recurrent_states:
            if set(vlm_kv.split_recurrent_states) != set(vlm_kv.recurrent_states):
                raise RuntimeError("Missing exact prefix-boundary recurrent states: teacher-action leakage risk")
            if any(not isinstance(state, torch.Tensor) or state.shape != vlm_kv.recurrent_states[index].shape
                   for index, state in vlm_kv.split_recurrent_states.items()):
                raise RuntimeError("Invalid prefix-boundary recurrent-state tensors")
        return super()._build_prefix_action_kv(vlm_kv, split_index)


class G05PolicyMEMLiteAction(G05PolicyQwen35):
    model_cls = ActionTrainingModelQwen35

    def __init__(self, **model_cfg):
        settings = ActionTrainingSettings(**dict(model_cfg.get("action_training", {})))
        if (not model_cfg.get("discrete_action")
                or bool(model_cfg.get("continuous_action")) != settings.uses_fm
                or bool(model_cfg.get("predict_cot")) != settings.predicts_cot
                or model_cfg.get("memlite_train_mode", "off") != "off"):
            raise ValueError("AR/joint/KI flags must match the declared action/CoT route")
        if settings.predicts_cot and (not model_cfg["input_preprocessor"].get("pred_eov")
                or model_cfg.get("register_memlite_hl_end", True)
                or model_cfg["ar"].get("block_wise_autoregressive", False)
                or not 1 <= int(model_cfg.get("native_cot_max_new_tokens", 0)) <= 512):
            raise ValueError("Native Subtask-CoT requires original vocabulary, supervised EOV and a bounded non-BAR decoder")
        if settings.uses_fm and bool(model_cfg["fm"]["joint_training"]) != (settings.route == "joint"):
            raise ValueError("joint passes FM gradients to the VLM; KI must detach them")
        if int(model_cfg.get("num_obs_steps", 0)) != 6 or int(model_cfg.get("action_dim", 0)) != 27:
            raise ValueError("Initial R1Pro action experiments require six observations and 27D representation")
        if int(model_cfg.get("horizon_steps", 0)) != 32:
            raise ValueError("The released action codec requires the declared 32-step representation")
        if bool(model_cfg["AT_CONFIG"].get("dropout_noop_parts", False)):
            raise ValueError("Constant gripper targets must be supervised, not dropped as noop")
        self.action_training = settings
        super().__init__(**model_cfg)
        self.interface_schema_version = MEMLITE_SKILL_SCHEMA_VERSION
        self.low_vlm_lora = VLMloraConfig.from_mapping(model_cfg.get("low_vlm_lora"))
        if not self.low_vlm_lora.enabled:
            raise ValueError("This bounded experiment declares VLM LoRA, not implicit full/frozen VLM training")
        self.component = "low"
        self.pipeline_stage = "A"
        self.stage = "experimental_" + settings.route
        self.trainability_profile = "low_ar_lora" if not settings.uses_fm else "low_ce_fm_lora"
        self._coordination_post_load_receipt = None

    def remap_checkpoint_state_dict(self, state_dict):
        return normalize_pre_injection_state_keys(state_dict)

    def post_checkpoint_load(self, state_dict, checkpoint_info=None):
        self.model.vlm = inject_vlm_lora(self.model.vlm, self.low_vlm_lora)
        receipt = restore_lora_state(self, state_dict)
        receipt.update(enabled=True, adapter_name=self.low_vlm_lora.adapter_name,
                       action_training=self.action_training.as_dict(), component=self.component,
                       trainability_profile=self.trainability_profile)
        self._coordination_post_load_receipt = dict(receipt)
        return receipt

    def coordination_trainable_parameter_groups(self):
        groups = defaultdict(list)
        for name, parameter in self.named_parameters():
            if parameter.requires_grad:
                group = ("action_expert" if name.startswith("model.action_expert.") else
                         "vlm_lora" if name.startswith("model.vlm.") and "lora_" in name else "unexpected")
                groups[group].append((name, parameter))
        return dict(groups)

    def configure_coordination_trainability(self):
        if self._coordination_post_load_receipt is None:
            raise RuntimeError("Restore base and LoRA before configuring the optimizer")
        for name, parameter in self.named_parameters():
            enabled = (name.startswith("model.vlm.") and "lora_" in name)
            enabled |= self.action_training.uses_fm and name.startswith("model.action_expert.")
            parameter.requires_grad_(enabled)
        self._assert_action_training_contract()
        return dict(component=self.component, pipeline_stage=self.pipeline_stage,
            trainability_profile=self.trainability_profile, action_training=self.action_training.as_dict(),
            num_obs_steps=6, expected_trainable_groups=sorted(self.coordination_trainable_parameter_groups()),
            trainable_parameter_names=[n for n, p in self.named_parameters() if p.requires_grad],
            post_checkpoint_load=dict(self._coordination_post_load_receipt))

    def _assert_action_training_contract(self):
        expected = {"vlm_lora"} | ({"action_expert"} if self.action_training.uses_fm else set())
        if set(self.coordination_trainable_parameter_groups()) != expected:
            raise RuntimeError("AR/joint/KI parameter groups do not match the explicit training route")
        if self.action_training.uses_fm and bool(self.model.fm_helper.joint_training) != (self.action_training.route == "joint"):
            raise RuntimeError("Actual FM gradient isolation differs from the declared recipe")

    def forward_train(self, samples, pixel_values, actions=None, action_pad_masks=None,
                      action_dim_is_pad=None, **kwargs):
        # Validate the original model-safe v6 sample with all existing semantic
        # guards before attaching supervised action tokens. The input builder
        # remains target-free SkillFM; no audit/privileged fields are added.
        G05PolicyMEMLiteSkillFM._validate_skill_batch(self, samples, actions, action_pad_masks)
        self._assert_action_training_contract()
        prepared = action_training_samples(samples, actions, action_pad_masks,
                                           action_dim_is_pad, self.action_training)
        loss, metrics = super().forward_train(prepared, pixel_values, actions=actions,
            action_pad_masks=action_pad_masks, action_dim_is_pad=action_dim_is_pad, **kwargs)
        metrics.update(action_token_loss_metrics(self.model.ar_helper._last_ce_cache,
            self.action_tokenizer.action_token_begin_idx, self.action_tokenizer.action_token_end_idx))
        return loss, metrics

    @torch.no_grad()
    def forward_inference(self, samples, pixel_values, **kwargs):
        # Inference must never tokenize a provided expert action by accident.
        # Action GT is not used even for optional metric calculation here.
        if kwargs.get("actions") is not None:
            raise ValueError("Deployment action entry is target-free; evaluate GT outside the actor")
        mask = kwargs.get("action_dim_is_pad")
        if not isinstance(mask, torch.Tensor) or mask.dtype != torch.bool or mask.shape != (len(samples), 27):
            raise ValueError("Target-free deployment still requires explicit 27D static padding metadata")
        expected = torch.zeros_like(mask)
        expected[:, [7, 8, 17, 18]] = True
        if not torch.equal(mask, expected):
            raise ValueError("Missing real base/trunk controls or incorrect padding mask")
        if self.action_training.conditioning in {"native_task", "native_subtask_cot"}:
            # End-to-end task AR must not require a planner/oracle bundle just
            # to pass an interface check. No semantic sidecar enters prefill.
            builder = native_subtask_cot_actor_samples if self.action_training.predicts_cot else native_task_actor_samples
            prepared = builder(samples, num_images=int(self.model_config.num_input_images))
            for sample in prepared:
                proprio = sample["proprio"]
                if not isinstance(proprio, dict):
                    raise ValueError("Native actor requires explicit observed proprioception metadata")
                value, state_mask = proprio.get("value"), proprio.get("proprio_dim_is_pad")
                if (not isinstance(value, torch.Tensor) or value.shape != (6, 27) or not torch.isfinite(value).all()
                        or not isinstance(state_mask, torch.Tensor) or state_mask.dtype != torch.bool
                        or state_mask.shape != (27,) or not torch.equal(state_mask.to(mask.device), expected[0])):
                    raise ValueError("Native actor requires six observed 27D states and the real embodiment mask")
        else:
            for sample in samples:
                projection = validate_embedded_model_projection(sample)
                if (projection["memlite_branch"] != "low" or projection["task_complete"]
                        or projection["next_decision"] != "EXECUTE"
                        or any(skill["verb"] == "SKILL_UNKNOWN" for skill in projection["active_skills"])):
                    raise ValueError("Actor requires a nonterminal, valid low-level semantic condition")
            prepared = action_prefix_samples(samples, self.action_training)
        started = time.monotonic()
        state = self.prefill(prepared, pixel_values)
        cot_result = {}
        if self.action_training.predicts_cot:
            if len(samples) != 1:
                raise ValueError("Initial native CoT actor is single-environment only; mixed-length batch handoff is unverified")
            eov = self.processor.eov_token_id
            cot_prefix_length = state.attention_mask.shape[1]
            state = self.generate_text(state, max_new_tokens=int(self.model_config.native_cot_max_new_tokens),
                stop_token_ids=[eov], trim_token_ids=self._get_trim_token_ids(include_eov=True))
            ids = state.generated_ids
            begin, end = self.action_tokenizer.action_token_begin_idx, self.action_tokenizer.action_token_end_idx
            if (not isinstance(ids, torch.Tensor) or ids.ndim != 2 or ids.shape[0] != 1 or ids.shape[1] < 2
                    or int(ids[0, -1]) != eov or ((ids >= begin) & (ids < end)).any()
                    or state.attention_mask.shape[1] != cot_prefix_length + ids.shape[1] - 1
                    or not state.generated_texts or not state.generated_texts[0].strip().startswith("Subtask:")
                    or not state.generated_texts[0].split("|", 1)[0].removeprefix("Subtask:").strip()):
                raise RuntimeError("Free AR generation failed to produce a bounded text subtask followed by EOV")
            cot_result = dict(cot_text=list(state.generated_texts), cot_generated_ids=ids.detach().clone())
            # The actual non-BAR decoder stops BEFORE forwarding its stop token.
            # Actions must condition on the generated EOV, not reuse the hidden
            # that still predicts EOV. Never fabricate the subtask or skip this
            # boundary merely because the text string looks well formed.
            state = self._commit_generated_cot_eov(state)
        if self.action_training.route == "ar":
            result = self.generate_action(state, prepared, action_dim_is_pad=mask, only_ar=True)
            if result.get("selected_action_source") != "ar":
                raise RuntimeError("Pure AR actor unexpectedly used continuous FM")
            absent = result.get("ar_absent_keys")
            if absent is None or len(absent) != len(samples) or any(absent):
                raise RuntimeError("Free AR generation omitted action groups; no GT or FM fallback is allowed")
            # The legacy decoder returns zeros AND empty absent sets for an
            # empty/unparseable string. An empty absent set alone is not proof
            # of a valid action. Reject its placeholder token [0] explicitly.
            tokens = result.get("decoded_action_tokens")
            begin, end = (self.action_tokenizer.action_token_begin_idx,
                          self.action_tokenizer.action_token_end_idx)
            if (tokens is None or len(tokens) != len(samples)
                    or any(not isinstance(ids, torch.Tensor) or ids.ndim != 1
                           or not ((ids >= begin) & (ids < end)).any() for ids in tokens)):
                raise RuntimeError("Free AR generation produced no valid action tokens")
            try:
                result["ar_complete_block_receipts"] = [
                    validate_complete_action_tokens(ids, self.action_tokenizer) for ids in tokens]
            except ValueError as exc:
                raise RuntimeError("Free AR generation has incomplete or malformed codec blocks: " + str(exc)) from exc
        else:
            # No auxiliary autoregressive action generation in deployed KI.
            result = dict(action=self.model.inference_fm(
                attention_mask=state.attention_mask, pixel_values=state.pixel_values,
                past_key_values=state.kv_cache, action_dim_is_pad=mask,
                position_ids_override=state.position_ids,
                embodiment_types=[s.get("embodiment") for s in prepared]), selected_action_source="fm")
        action = result["action"]
        if action.shape != (len(samples), 32, 27) or not torch.isfinite(action).all():
            raise RuntimeError("Invalid generated action shape or numerics")
        # Padding is static embodiment metadata, not a suppressed real control.
        result["action"] = action.masked_fill(mask.to(action.device)[:, None, :], 0.)
        result["execution_start"] = 0
        result["execution_steps"] = 16
        result["action_training"] = self.action_training.as_dict()
        result.update(cot_result)
        result.setdefault("_timing", {})["action_route_total_ms"] = (time.monotonic() - started) * 1000
        return result

    def _commit_generated_cot_eov(self, state):
        """Consume exactly one genuinely generated EOV into the Qwen cache."""
        from g05.models.kv_cache import SparseKVCache
        state.check_invariants("native cot handoff entry")
        if (state.generated_ids is None or state.generated_ids.shape[0] != 1
                or int(state.generated_ids[0, -1]) != self.processor.eov_token_id
                or getattr(state, "_native_cot_eov_committed", False)
                or not isinstance(state.kv_cache, SparseKVCache)):
            raise RuntimeError("Native CoT handoff requires a real EOV and the declared Qwen cache")
        core = self.model
        token = state.generated_ids[:, -1:]
        kind = core.ar_helper._assign_token_index(token)
        mask = torch.cat([state.attention_mask, kind], dim=1)
        embeddings = core.vlm.embed(token)
        step_mask, step_pos = core.build_causal_mask_and_position_ids(token, mask,
            kv_len=state.kv_cache.num_items(), dtype=embeddings.dtype)
        hidden, cache = core.vlm(inputs_embeds=embeddings, attention_mask=step_mask,
            position_ids=step_pos, kv_cache=state.kv_cache, return_kv_cache=True,
            attn_implementation=core.attn_implementation, mixture_name="vlm")
        # generate_text's generic position metadata is 2-D. Rebuild the actual
        # Qwen MRoPE positions from the unchanged image grid and observed types;
        # verify the final position against the one just used by neural decode.
        positions = core.mask_helper._build_mrope_position_ids(mask, state.device)
        if not torch.equal(positions[..., -1:], step_pos):
            raise RuntimeError("Native CoT EOV cache handoff has inconsistent MRoPE positions")
        state.kv_cache, state.attention_mask, state.position_ids = cache, mask, positions
        state.last_hidden = hidden[:, -1]
        state._native_cot_eov_committed = True
        state.check_invariants("native cot handoff exit")
        return state
