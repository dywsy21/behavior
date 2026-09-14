"""Explicit EMA shadows for FP32/FP64 trainable parameters, not a second model.

The caller updates once AFTER each successful optimizer step. Frozen parameters
and buffers stay in the online model. Evaluation copies shadows temporarily and
restores the online tensors (including on exceptions); it never adopts EMA as
the next optimizer iterate. No random sampling or implicit warmup is performed.
"""
from __future__ import annotations

from contextlib import contextmanager
import math

import torch


class TrainableParameterEMA:
    FORMAT_VERSION = 1

    def __init__(self, model, *, beta=0.99):
        if type(beta) not in (float, int) or not math.isfinite(beta) or not 0 <= beta < 1:
            raise ValueError("EMA beta must be finite and in [0, 1)")
        if not isinstance(model, torch.nn.Module):
            raise ValueError("EMA requires the online torch module")
        self._model = model
        self._parameters = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
        if not self._parameters or any(p.layout != torch.strided or p.dtype not in
                (torch.float32, torch.float64) for _, p in self._parameters):
            raise ValueError("This EMA explicitly tracks nonempty dense FP32/FP64 master parameters")
        if len({p.device for _, p in self._parameters}) != 1:
            raise ValueError("One device per EMA/DDP rank is required")
        self._beta = float(beta)
        self._schema = [dict(name=n, shape=list(p.shape), dtype=str(p.dtype)) for n, p in self._parameters]
        self._device = self._parameters[0][1].device
        self._assert_finite([p.detach() for _, p in self._parameters], "online initialization")
        self._shadow = {n: p.detach().clone() for n, p in self._parameters}
        self._num_updates = 0
        self._applied = False

    @staticmethod
    def _assert_finite(tensors, where):
        # One synchronization per device, not one .item() for each parameter.
        by_device = {}
        for tensor in tensors:
            by_device.setdefault(tensor.device, []).append(torch.isfinite(tensor).all())
        if any(not bool(torch.stack(flags).all()) for flags in by_device.values()):
            raise ValueError("Nonfinite EMA tensors in " + where)

    def _validate_online(self):
        current = [(n, p) for n, p in self._model.named_parameters() if p.requires_grad]
        if len(current) != len(self._parameters):
            raise ValueError("EMA trainable parameter membership changed")
        for (name, parameter), (old_name, old_parameter), schema in zip(
                current, self._parameters, self._schema):
            if (name != old_name or parameter is not old_parameter
                    or list(parameter.shape) != schema["shape"] or str(parameter.dtype) != schema["dtype"]
                    or parameter.device != self._device or parameter.layout != torch.strided):
                raise ValueError("EMA parameter identity/shape/dtype/device changed")

    def _require_online(self):
        if self._applied:
            raise RuntimeError("EMA cannot update, load, export or nest while average weights are applied")
        self._validate_online()

    @property
    def num_updates(self):
        return self._num_updates

    @property
    def parameter_names(self):
        return tuple(n for n, _ in self._parameters)

    @property
    def shadow_bytes(self):
        return sum(t.numel() * t.element_size() for t in self._shadow.values())

    @torch.no_grad()
    def update(self, *, optimizer_step):
        self._require_online()
        if type(optimizer_step) is not int or optimizer_step != self._num_updates + 1:
            raise ValueError("EMA requires exactly one consecutive successful optimizer step")
        self._assert_finite([p.detach() for _, p in self._parameters], "online update")
        for name, parameter in self._parameters:
            # Convex combination avoids overflowing the subtraction of two
            # opposite, individually finite values in a lerp implementation.
            self._shadow[name].mul_(self._beta).add_(parameter.detach(), alpha=1.0 - self._beta)
        self._num_updates = optimizer_step

    def _metadata(self):
        return dict(format_version=self.FORMAT_VERSION, kind="trainable_parameter_ema",
            beta=self._beta, update_every=1, initialization="copy_parent_before_update_1",
            num_updates=self._num_updates, parameters=[dict(row, shape=list(row["shape"])) for row in self._schema])

    @torch.no_grad()
    def state_dict(self):
        self._require_online()
        self._assert_finite(self._shadow.values(), "state export")
        return dict(**self._metadata(), shadow={n: t.detach().cpu().clone() for n, t in self._shadow.items()})

    def _validated_state(self, state, expected_updates):
        if not isinstance(state, dict) or set(state) != set(self._metadata()) | {"shadow"}:
            raise ValueError("Incomplete or unknown EMA state schema")
        updates = state["num_updates"]
        if (type(updates) is not int or updates < 0 or expected_updates is not None
                and (type(expected_updates) is not int or updates != expected_updates)):
            raise ValueError("Wrong saved EMA update clock")
        expected = self._metadata()
        expected["num_updates"] = updates
        if {k: state[k] for k in expected} != expected:
            raise ValueError("EMA recipe or tracked parameter schema differs")
        shadows = state["shadow"]
        if not isinstance(shadows, dict) or set(shadows) != set(self._shadow):
            raise ValueError("EMA shadows must cover all and only tracked parameters")
        for name, old in self._shadow.items():
            tensor = shadows[name]
            if (not isinstance(tensor, torch.Tensor) or tensor.layout != torch.strided
                    or tensor.shape != old.shape or tensor.dtype != old.dtype or tensor.requires_grad):
                raise ValueError("Invalid EMA shadow tensor: " + name)
        self._assert_finite(shadows.values(), "state restore")
        return updates, shadows

    @torch.no_grad()
    def load_state_dict(self, state, *, expected_updates=None):
        self._require_online()
        updates, shadows = self._validated_state(state, expected_updates)
        # Validate and allocate all replacements before changing live state.
        replacement = {n: t.detach().to(self._device).clone() for n, t in shadows.items()}
        self._shadow, self._num_updates = replacement, updates

    @torch.no_grad()
    def assert_matches_state(self, state, *, expected_updates):
        self._require_online()
        updates, shadows = self._validated_state(state, expected_updates)
        if updates != self._num_updates or any(not torch.equal(t.detach().cpu(), shadows[n].cpu())
                for n, t in self._shadow.items()):
            raise ValueError("Saved EMA shadow/clock did not roundtrip exactly")

    @contextmanager
    def average_parameters(self):
        self._require_online()
        self._assert_finite(self._shadow.values(), "temporary evaluation")
        # CPU backups keep peak GPU memory below a third policy-sized copy.
        # Parameter objects and their optimizer references are never replaced.
        backup = {n: p.detach().cpu().clone() for n, p in self._parameters}
        self._applied = True
        try:
            with torch.no_grad():
                for name, parameter in self._parameters:
                    parameter.copy_(self._shadow[name])
                yield self._model
        finally:
            with torch.no_grad():
                for name, parameter in self._parameters:
                    parameter.copy_(backup[name])
            self._applied = False
