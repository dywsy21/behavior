"""Independent MEM-Lite FM controller with a frozen, coherent G0.5 baseline.

The AR planner is deliberately NOT a child of this module. It is loaded from its
own checkpoint by the separated inferencer. Training here cannot change it.
"""
from __future__ import annotations

from copy import deepcopy

import torch

from .g05_policy_qwen35 import G05PolicyQwen35
from .helpers.fm_intent_adapter import FMIntentAdapter


class G05PolicyMEMLiteFM(G05PolicyQwen35):
    def __init__(self, **model_cfg):
        super().__init__(**model_cfg)
        if self.discrete_action or not self.continuous_action or self.predict_cot:
            raise ValueError("FM low policy requires continuous=true, discrete=false, predict_cot=false")
        if self.memlite_train_mode != "low" or self.memlite_conditioning.enabled:
            raise ValueError("FM adapter requires low-only mode and unchanged baseline proprio conditioning")
        if int(self.model_config.num_obs_steps) != 1:
            raise ValueError("v11 preserves the old FM controller's single current observation")
        cfg = self.model_config.fm_intent_adapter
        self.intent_dropout = float(cfg.get("intent_dropout", 0.1))
        self.residual_weight = float(cfg.get("residual_weight", 0.001))
        if not 0 <= self.intent_dropout < 1 or self.residual_weight < 0:
            raise ValueError("Invalid intent dropout/residual regularization")
        self.intent_enabled = bool(cfg.get("enabled", True))
        # Freeze BEFORE registering the only trainable module. This includes
        # the tokenizer and all shared/tied embeddings, not only the FM head.
        self.requires_grad_(False)
        self.fm_intent_adapter = FMIntentAdapter(
            token_dim=int(self.model_config.vlm.hidden_size),
            feature_dim=int(self.model_config.action_expert.hidden_size),
            action_dim=int(self.model_config.action_dim),
            hidden_dim=int(cfg.get("hidden_dim", 256)),
            num_heads=int(cfg.get("num_heads", 4)), num_layers=int(cfg.get("num_layers", 2)),
            max_tokens=int(cfg.get("max_tokens", 256)),
        )
        self.model.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        # No train-mode dropout or checkpointing drift in the preserved model.
        self.model.eval()
        # VQActionTokenizer is a registry/codec wrapper, not necessarily nn.Module.
        if callable(getattr(self.action_tokenizer, "eval", None)):
            self.action_tokenizer.eval()
        return self

    def get_optim_param_groups(self, lr, weight_decay, apply_decay_on_norm_and_bias=False,
                               backbone_lr_multiplier=1.0, vision_lr_multiplier=1.0):
        # The inherited grouping intentionally sees only self.model; the new
        # independent adapter lives outside it and needs explicit coverage.
        modules = dict(self.fm_intent_adapter.named_modules())
        decay, no_decay = [], []
        for name, parameter in self.fm_intent_adapter.named_parameters():
            owner, _, leaf = name.rpartition(".")
            apply_decay = apply_decay_on_norm_and_bias or self._should_apply_weight_decay(modules[owner], leaf, parameter)
            (decay if apply_decay else no_decay).append(parameter)
        parameters = decay + no_decay
        expected = [p for p in self.parameters() if p.requires_grad]
        if len(parameters) != len(expected) or {id(p) for p in parameters} != {id(p) for p in expected}:
            raise RuntimeError("FM optimizer must cover exactly the intent adapter, no baseline parameters")
        return [{"params": decay, "lr": lr, "weight_decay": weight_decay, "name": "intent_decay"},
                {"params": no_decay, "lr": lr, "weight_decay": 0., "name": "intent_no_decay"}]

    def forward(self, batch, inference_mode=False):
        # Parent inference restores self.model.train(was_training), which is
        # correct for joint fine-tuning but not a preserved frozen controller.
        was_training = self.training
        if inference_mode:
            self.eval()
        self.model.eval()
        try:
            return super().forward(batch, inference_mode=inference_mode)
        finally:
            self.train(was_training)

    @staticmethod
    def baseline_samples(samples):
        """A task-only, current-state prefix. Reject implicit template changes."""
        from g05.data_processor.processor.samples_builder import BaseSamplesBuilder
        output = []
        for sample in samples:
            item = deepcopy(sample)
            images = sorted(int(k[5:]) for k in item if k.startswith("image") and k[5:].isdigit())
            if images != [0, 1, 2]:
                raise ValueError("FM baseline needs exactly three ordered camera placeholders")
            builder = BaseSamplesBuilder(num_input_images=3,
                image_sizes={f"camera{i}": item[f"image{i}"] for i in images})
            # Keep the exact original prefix. Strip all targets BEFORE the
            # preprocessor (which otherwise may call ActionCodec on spare data).
            item["template"] = builder.template.replace("<action_action>|", "")
            for key in ("action", "intent", "memory", "memory_update", "intent_status",
                        "previous_intent", "execution_feedback", "memlite_causal_prompt"):
                item.pop(key, None)
            proprio = item["proprio"]["value"]
            if proprio.ndim != 2 or proprio.shape[0] != 1:
                raise ValueError("FM baseline must receive the single latest proprio row")
            output.append(item)
        return output

    def _intent_context(self, intents, device):
        from g05.utils.memlite_protocol import is_terminal_intent
        texts = [str(x or "").strip() for x in intents]
        if any(is_terminal_intent(x) for x in texts if x):
            raise ValueError("Terminal intent belongs to the runtime hold path, not FM generation")
        active = torch.tensor([bool(x) and self.intent_enabled for x in texts], device=device)
        if self.training and torch.is_grad_enabled() and self.intent_dropout:
            active = active & (torch.rand(len(texts), device=device) >= self.intent_dropout)
        encoded = self.processor.tokenizer([x or "No intent." for x in texts],
            padding=True, truncation=False, add_special_tokens=False, return_tensors="pt")
        ids = encoded["input_ids"].to(device)
        mask = encoded["attention_mask"].to(device).bool()
        if ids.shape[1] > self.fm_intent_adapter.max_tokens:
            raise ValueError(f"Intent has {ids.shape[1]} tokens; increase the audited context limit")
        with torch.no_grad():
            embeddings = self.model.vlm.input_proj(ids)
        return self.fm_intent_adapter.encode(embeddings, mask, active)

    def forward_train(self, samples, pixel_values, actions=None, action_pad_masks=None,
                      action_dim_is_pad=None, **kwargs):
        self._memlite_branch_masks(samples, "low")
        if actions is None or action_pad_masks is None or actions.ndim != 3:
            raise ValueError("FM training requires raw normalized continuous targets and time masks")
        if actions.shape[-1] != int(self.model_config.action_dim) or not torch.isfinite(actions).all():
            raise ValueError("Invalid FM continuous action target")
        if action_pad_masks.shape != actions.shape[:2] or action_pad_masks.all():
            raise ValueError("FM batch must have correctly shaped, nonempty action supervision")
        if any(not str(s.get("intent", "")).strip() for s in samples):
            raise ValueError("FM supervised rows require a causal intent annotation")
        baseline = self.baseline_samples(samples)
        state = self.prefill(baseline, pixel_values)
        context = self._intent_context([s["intent"] for s in samples], actions.device)
        residuals = []

        def adapt(features, velocity):
            delta = self.fm_intent_adapter(features, context)
            residuals.append(delta)
            return velocity + delta.to(velocity.dtype)

        prefix_kv = self.model._build_prefix_action_kv(state.kv_cache, state.attention_mask.shape[1])
        fm = self.model.fm_helper.train_step(self.model, prefix_kv,
            state.attention_mask, state.position_ids, actions, action_pad_masks,
            action_dim_is_pad, next(iter(state.pixel_values.values())).dtype,
            embodiment_types=[s.get("embodiment") for s in samples], velocity_adapter=adapt)
        regularizer = residuals[0].square().mean() * self.residual_weight
        self._fwd_step += 1
        self.train_action_accuracy = self.train_cot_accuracy = 0.0
        metrics = {"fm_loss": fm.detach(), "intent_residual_loss": regularizer.detach(),
                   "intent_active_fraction": context["active"].float().mean().detach(),
                   "intent_residual_rms": residuals[0].square().mean().sqrt().detach()}
        return fm + regularizer, metrics

    @torch.no_grad()
    def generate_low_level_action(self, samples, pixel_values, *, intent_text,
                                  action_dim_is_pad=None, action_gt=None):
        intents = [intent_text] * len(samples) if isinstance(intent_text, str) else list(intent_text)
        if len(intents) != len(samples):
            raise ValueError("Intent batch must match observation batch")
        state = self.prefill(self.baseline_samples(samples), pixel_values)
        # The disabled branch skips the adapter entirely; exact baseline parity
        # must not depend on residual numerical noise or attention kernel choices.
        adapt = None
        if self.intent_enabled and any(str(x or "").strip() for x in intents):
            context = self._intent_context(intents, state.device)
            def adapt(features, velocity):
                return velocity + self.fm_intent_adapter(features, context).to(velocity.dtype)
        action = self.model.inference_fm(attention_mask=state.attention_mask,
            pixel_values=state.pixel_values, past_key_values=state.kv_cache,
            action_dim_is_pad=action_dim_is_pad, position_ids_override=state.position_ids,
            embodiment_types=[s.get("embodiment") for s in samples], velocity_adapter=adapt)
        return {"action": action, "fm_action": action, "selected_action_source": "fm",
                "low_level_intent": intents}

    def forward_inference(self, samples, pixel_values, action_dim_is_pad=None, **kwargs):
        return self.generate_low_level_action(samples, pixel_values,
            intent_text=[s.get("intent", "") for s in samples], action_dim_is_pad=action_dim_is_pad)
