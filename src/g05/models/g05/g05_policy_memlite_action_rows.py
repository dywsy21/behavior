"""Explicit pure-AR marker-row experiment; existing FM/AR defaults unchanged."""
from __future__ import annotations

from collections import defaultdict

import torch

from .g05_policy_memlite_action import G05PolicyMEMLiteAction
from .helpers.action_marker_rows import (
    required_marker_ids, install_marker_rows, restore_marker_rows, assert_marker_bindings,
)


class G05PolicyMEMLiteActionRows(G05PolicyMEMLiteAction):
    def __init__(self, **model_cfg):
        row_cfg = dict(model_cfg.get("marker_row_adaptation", {}))
        if (set(row_cfg) != {"train_rows", "mode"}
                or type(row_cfg["train_rows"]) is not bool or row_cfg["mode"] != "required_codec_markers"
                or model_cfg.get("action_training", {}).get("route") != "ar"
                or model_cfg.get("ar", {}).get("use_fused_ce", True)):
            raise ValueError("Marker-row recipe requires explicit pure AR, eager CE and declared train_rows")
        self.train_marker_rows = row_cfg["train_rows"]
        super().__init__(**model_cfg)

    @property
    def fp32_param_patterns(self):
        return super().fp32_param_patterns + ["action_marker_rows.delta"]

    def _marker_vlm(self):
        return self.model.vlm.get_base_model()

    def post_checkpoint_load(self, state_dict, checkpoint_info=None):
        receipt = super().post_checkpoint_load(state_dict, checkpoint_info)
        self.model.action_marker_rows = install_marker_rows(self._marker_vlm(), required_marker_ids(self.action_tokenizer))
        receipt["marker_rows"] = restore_marker_rows(self.model.action_marker_rows, state_dict)
        if not self.train_marker_rows and torch.count_nonzero(self.model.action_marker_rows.delta):
            raise RuntimeError("The matched LoRA-only control cannot inherit a trained marker adapter")
        receipt["marker_rows"]["train_rows"] = self.train_marker_rows
        self._coordination_post_load_receipt = dict(receipt)
        return receipt

    def coordination_trainable_parameter_groups(self):
        groups = defaultdict(list)
        for name, parameter in self.named_parameters():
            if parameter.requires_grad:
                group = ("vlm_lora" if name.startswith("model.vlm.") and "lora_" in name else
                         "action_marker_rows" if name == "model.action_marker_rows.delta" else "unexpected")
                groups[group].append((name, parameter))
        return dict(groups)

    def configure_coordination_trainability(self):
        if self._coordination_post_load_receipt is None:
            raise RuntimeError("Restore base, LoRA and marker rows before configuring the optimizer")
        for name, parameter in self.named_parameters():
            enabled = (name.startswith("model.vlm.") and "lora_" in name)
            enabled |= self.train_marker_rows and name == "model.action_marker_rows.delta"
            parameter.requires_grad_(enabled)
        self._assert_action_training_contract()
        return dict(action_training=self.action_training.as_dict(), train_marker_rows=self.train_marker_rows,
                    expected_trainable_groups=sorted(self.coordination_trainable_parameter_groups()),
                    trainable_parameter_names=[name for name, p in self.named_parameters() if p.requires_grad],
                    marker_token_ids=self.model.action_marker_rows.token_ids.cpu().tolist(),
                    post_checkpoint_load=dict(self._coordination_post_load_receipt))

    def _assert_action_training_contract(self):
        expected = {"vlm_lora"} | ({"action_marker_rows"} if self.train_marker_rows else set())
        if (self.action_training.route != "ar" or self.model.ar_helper.use_fused_ce
                or set(self.coordination_trainable_parameter_groups()) != expected):
            raise RuntimeError("Marker rows require the declared groups, pure AR and eager CE (no fused bypass)")
        assert_marker_bindings(self._marker_vlm(), self.model.action_marker_rows)

    @torch.no_grad()
    def forward_inference(self, *args, **kwargs):
        assert_marker_bindings(self._marker_vlm(), self.model.action_marker_rows)
        return super().forward_inference(*args, **kwargs)
