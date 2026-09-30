"""Causal schema-v6 planner plus an outcome observer head.

The language model predicts the prior-bundle result first (when evidence is
labelled), then the decision/current bundle/memory/terminal fields.  The small
result head is intentionally read from the EOC-*prefix* hidden state, never
from a teacher-forced answer position.
"""
from __future__ import annotations

from collections import defaultdict
import json
import math
from typing import Any, Dict, List, Mapping, Optional, Union

import torch

from g05.data_processor.processor.base_processor import IGNORE_INDEX
from g05.data_processor.processor.memlite_v6_projection import validate_embedded_model_projection
from g05.models.g05.helpers.proprio_helper import build_proprio_batch
from g05.utils.memlite_planner_fields import planner_field_slot, planner_field_text
from g05.utils.memlite_parent_format import (
    is_declared_task_parent_format, validate_task_parent_formats,
)
from g05.utils.memlite_skill_protocol import (
    MEMLITE_SKILL_SCHEMA_VERSION,
    VALID_DECISIONS,
    VALID_OUTCOMES,
    append_b_memory_idempotent,
    canonical_json,
    parse_active_skills_semantic_json,
    semantic_active_skills_text,
    validate_b_memory_text,
    validate_issued_bundle_text,
    validate_semantic_parent_goal,
)

from .g05_policy_qwen35 import G05PolicyQwen35
from .helpers.outcome_head import (
    OutcomePrediction,
    PlannerOutcomeHead,
    outcome_target_tensor,
)


class G05PolicyMEMLitePlannerOutcome(G05PolicyQwen35):
    """High-level AR planner with leak-free four-way outcome supervision."""

    def __init__(self, **model_cfg):
        # Minimal single-frame benchmark port; full historical loader is not migrated.
        if model_cfg.get("num_obs_steps") != 1 or isinstance(model_cfg.get("num_obs_steps"), bool):
            raise ValueError("Planner benchmark port requires the audited single-frame route")
        super().__init__(**model_cfg)
        if not self.discrete_action or self.continuous_action or not self.predict_cot:
            raise ValueError(
                "PlannerOutcome requires discrete_action=true, continuous_action=false, predict_cot=true"
            )
        if self.memlite_train_mode != "off":
            raise ValueError("PlannerOutcome owns its high CE route; set memlite_train_mode=off")
        self.coordination_train = dict(model_cfg.get("coordination_train", {}))
        self.interface_schema_version = int(self.coordination_train.get("interface_schema_version", 6))
        if self.interface_schema_version != MEMLITE_SKILL_SCHEMA_VERSION:
            raise ValueError(
                "PlannerOutcome model/data schema mismatch: "
                f"config={self.interface_schema_version} protocol={MEMLITE_SKILL_SCHEMA_VERSION}"
            )
        self.stage = str(self.coordination_train.get("stage", "planner_outcome"))
        if self.stage not in {"planner_outcome", "planner_only"}:
            raise ValueError(f"Unsupported PlannerOutcome stage {self.stage!r}")
        self.pipeline_stage = str(self.coordination_train.get("pipeline_stage", ""))
        if self.pipeline_stage != "B":
            raise ValueError("PlannerOutcome requires pipeline_stage='B'")
        self.component = str(self.coordination_train.get("component", ""))
        self.trainability_profile = str(self.coordination_train.get("trainability_profile", ""))
        expected_profile = {
            "planner_outcome": "high_planner_outcome",
            "planner_only": "high_planner_only",
        }[self.stage]
        if self.component != "high" or self.trainability_profile != expected_profile:
            raise ValueError(
                f"PlannerOutcome stage {self.stage!r} requires component='high' and "
                f"trainability_profile={expected_profile!r}"
            )
        self.planner_only = self.trainability_profile == "high_planner_only"
        head_cfg = dict(model_cfg.get("planner_outcome", {}))
        self.supervise_task_parent_format = head_cfg.get("supervise_task_parent_format", False)
        self.task_parent_formats = validate_task_parent_formats(
            self.supervise_task_parent_format,
            head_cfg.get("task_parent_format_by_task", {}), planner_only=self.planner_only,
        )
        self.outcome_loss_weight = float(head_cfg.get("outcome_loss_weight", 1.0))
        if self.outcome_loss_weight < 0.0:
            raise ValueError("planner_outcome.outcome_loss_weight must be nonnegative")
        self.outcome_head = PlannerOutcomeHead(
            hidden_size=int(self.model_config.vlm.hidden_size),
            dropout=float(head_cfg.get("dropout", 0.0)),
        )
        self.outcome_ready = bool(head_cfg.get("outcome_ready", False))
        self.outcome_minimum_confidence = float(head_cfg.get("minimum_confidence", 0.85))
        self.outcome_calibration_receipt = str(head_cfg.get("calibration_receipt", "") or "")
        self.outcome_validated = bool(head_cfg.get("outcome_validated", False))
        raw_coverage = head_cfg.get("outcome_coverage", 0)
        if isinstance(raw_coverage, bool) or int(raw_coverage) < 0:
            raise ValueError("planner_outcome.outcome_coverage must be a nonnegative integer")
        self.outcome_coverage = int(raw_coverage)
        self.initialization_source = str(head_cfg.get("initialization_source", "") or "")
        raw_memory_weight = head_cfg.get("memory_update_ce_weight", 1.0)
        if isinstance(raw_memory_weight, bool):
            raise ValueError("planner_outcome.memory_update_ce_weight must be a finite number")
        try:
            self.memory_update_ce_weight = float(raw_memory_weight)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "planner_outcome.memory_update_ce_weight must be a finite number"
            ) from exc
        if not math.isfinite(self.memory_update_ce_weight) or not (0.0 < self.memory_update_ce_weight <= 1.0):
            raise ValueError(
                "planner_outcome.memory_update_ce_weight must lie in (0, 1]"
            )
        # Keep serving's default coupled to the reviewed AR config.  A
        # planner_outcome override is permitted only as an explicit model
        # contract, never by silently inheriting the legacy short CoT cap.
        ar_cfg = dict(model_cfg.get("ar", {}))
        raw_planner_budget = head_cfg.get(
            "planner_default_max_new_tokens", ar_cfg.get("max_new_tokens", 1024)
        )
        if isinstance(raw_planner_budget, bool) or int(raw_planner_budget) < 1:
            raise ValueError(
                "planner_outcome.planner_default_max_new_tokens must be a positive integer"
            )
        self.planner_default_max_new_tokens = int(raw_planner_budget)
        if self.planner_only:
            # There is no reliable physical result label in the original
            # demonstrations.  Planner-only is therefore intentionally not a
            # weak version of outcome training: its head is inert, tensorwise
            # frozen, absent from AdamW, and permanently UNKNOWN-only.
            if self.outcome_loss_weight != 0.0:
                raise ValueError("high_planner_only requires outcome_loss_weight=0")
            if self.outcome_ready or self.outcome_validated or self.outcome_coverage != 0:
                raise ValueError(
                    "high_planner_only must keep outcome_ready=false, "
                    "outcome_validated=false, and outcome_coverage=0"
                )
            if self.outcome_calibration_receipt:
                raise ValueError("high_planner_only may not load an outcome calibration receipt")
            if self.initialization_source not in {"memlite_v10", "memlite_v10_planner_descendant"}:
                raise ValueError(
                    "high_planner_only requires an explicit v10 root or audited planner descendant"
                )
        elif self.memory_update_ce_weight != 1.0:
            raise ValueError(
                "memory_update_ce_weight is a high_planner_only-only diagnostic training option; "
                "high_planner_outcome must retain 1.0"
            )
        if self.outcome_ready and not self.outcome_calibration_receipt:
            raise ValueError(
                "outcome_ready=true requires a nonempty calibration_receipt from held-out physical outcome eval"
            )
        if self.outcome_ready and (not self.outcome_validated or self.outcome_coverage <= 0):
            raise ValueError(
                "outcome_ready=true requires outcome_validated=true and positive physical outcome coverage"
            )

    def outcome_prediction_from_context_hidden(self, context_hidden: torch.Tensor):
        """Return the classifier-gated, single-source runtime outcome proposal.

        Runtime must treat AR's first result field as a diagnostic parse only:
        it may compare it to this proposal and fail closed on conflict, but may
        never trigger a transition from AR text alone.
        """
        if self.planner_only:
            if context_hidden.ndim != 2:
                raise ValueError(
                    "PlannerOutcome planner-only runtime context must have shape [batch, hidden]"
                )
            # Do not even evaluate random/frozen head weights in the initial
            # planner pass.  A parser must see the same fixed UNKNOWN decision
            # for every row until a separately trained/calibrated profile is
            # deliberately selected.
            return [
                OutcomePrediction(
                    previous_outcome="UNKNOWN",
                    confidence=0.0,
                    outcome_ready=False,
                    source="planner_only_outcome_disabled",
                )
                for _ in range(context_hidden.shape[0])
            ]
        return self.outcome_head.decode_for_runtime(
            self.outcome_head(context_hidden),
            outcome_ready=self.outcome_ready,
            minimum_confidence=self.outcome_minimum_confidence,
        )

    def outcome_readiness(self) -> Dict[str, Any]:
        """Return the immutable deployment gate for the learned result head.

        A runtime may expose this in its receipt, but it must not override the
        model-owned decision.  In particular, a freshly trained head remains
        unavailable until a held-out physical-outcome calibration receipt is
        loaded in the model config.
        """
        return {
            "outcome_ready": bool(self.outcome_ready),
            "outcome_validated": bool(self.outcome_validated),
            "outcome_coverage": int(self.outcome_coverage),
            "minimum_confidence": float(self.outcome_minimum_confidence),
            "calibration_receipt": self.outcome_calibration_receipt,
            "profile": self.trainability_profile,
        }

    _TARGET_FREE_HIGH_FORBIDDEN_FIELDS = frozenset({
        "parent_goal", "target_parent_goal", "current_parent_goal",
        "active_skills_semantic_json", "active_skills_text", "active_skills_json",
        "outcome_target", "outcome_target_value", "outcome_supervision_mask",
        "next_decision", "memory_update", "task_complete",
        "parent_goal_supervision_mask", "low_action_supervision_mask", "action",
    })

    @staticmethod
    def _render_generated_active_skills(skills: list[dict[str, str]]) -> tuple[str, str]:
        """Validate and canonically render a generated semantic skill bundle.

        The parser accepts only the seven-field deployment projection.  It
        never calls the annotation-side skill parser, which would reintroduce
        frame/member/relation audit concepts into generated planner text.
        """
        # The B protocol owns the exact semantic seven-field grammar and the
        # deployed renderer.  Do not keep an almost-identical model-local
        # renderer: a generated bundle, previous_intent and K=3 history must
        # all use the one canonical parser/renderer.
        try:
            semantic_json = canonical_json(skills)
            canonical = parse_active_skills_semantic_json(
                semantic_json, allow_empty=False,
            )
            if any(skill["verb"] == "SKILL_UNKNOWN" for skill in canonical):
                raise ValueError("planner semantic skills may not contain SKILL_UNKNOWN")
            return semantic_json, semantic_active_skills_text(canonical, allow_empty=False)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "planner generated semantic bundle has an unknown/audit field "
                "or violates the canonical B grammar"
            ) from exc

    @staticmethod
    def _expected_b_memory_update(memory_text: str, previous_intent: str, task_name: str) -> str:
        """Apply the published target-free B recurrence without new bundle B_i.

        ``previous_intent`` is the last *issued* complete bundle.  The newly
        generated current bundle never enters this update, preventing both a
        same-row target leak and duplicate periodic replans.
        """
        try:
            # These protocol helpers parse/round-trip both JSON and compact
            # issued-bundle text.  They deliberately prove grammar only; the
            # sim adapter's private ledger remains the authority that binds a
            # valid issued string to an actually admitted prior event.
            validate_b_memory_text(memory_text, task_name=task_name)
            validate_issued_bundle_text(previous_intent, allow_none=True)
            return append_b_memory_idempotent(
                memory_text, previous_intent, task_name=task_name,
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("B planner memory/previous_intent violates canonical K=3 grammar") from exc

    def _validate_target_free_high_prefix(self, samples: List[Mapping[str, Any]]) -> None:
        """Reject a teacher-forced high target before VLM prefill."""
        if not samples:
            raise ValueError("target-free planner inference requires at least one sample")
        expected_images = int(self.model_config.num_input_images)
        required = {
            "template", "command", "task_name", "previous_parent_goal", "previous_intent",
            "memory", "known_previous_outcome", "execution_feedback", "planner_prompt",
            "proprio", "embodiment", "schema_version", "memlite_schema_version", "memlite_branch",
        }
        for index, sample in enumerate(samples):
            keys = set(sample)
            missing = required - keys
            if missing:
                raise ValueError(f"target-free planner sample {index} is missing {sorted(missing)}")
            forbidden = self._TARGET_FREE_HIGH_FORBIDDEN_FIELDS & keys
            if forbidden:
                raise ValueError(f"target-free planner sample {index} carries target field(s) {sorted(forbidden)}")
            # A target-free prefix has no legitimate use for annotation/audit
            # transport.  Check the *built* sample as well as the raw v6
            # projection: a future processor change must not smuggle IDs or
            # ground-truth evidence through a harmless-looking extra key.
            audit_fragments = (
                "audit", "annotation", "raw_", "skill_idx", "bundle_id",
                "episode", "frame", "evidence", "oracle", "ground_truth",
                "goal_status", "contact", "sim_state",
            )
            audit_keys = sorted(
                str(key) for key in keys
                if any(fragment in str(key).casefold() for fragment in audit_fragments)
            )
            if audit_keys:
                raise ValueError(
                    "target-free planner sample contains audit/oracle transport field(s) "
                    f"{audit_keys}"
                )
            if str(sample["memlite_branch"]).lower() != "high":
                raise ValueError("target-free planner sample must use high branch")
            if int(sample["schema_version"]) != self.interface_schema_version or int(sample["memlite_schema_version"]) != self.interface_schema_version:
                raise ValueError("target-free planner sample has incompatible schema version")
            if str(sample["command"]).strip() != str(sample["task_name"]).strip() or not str(sample["task_name"]).strip():
                raise ValueError("target-free planner prefix must retain the original task text")
            if self.planner_only and sample["known_previous_outcome"] != "Known previous outcome: UNKNOWN":
                raise ValueError("high_planner_only may only observe UNKNOWN prior outcome")
            # Validate the K=3 state *before* VLM prefill.  The result is
            # intentionally discarded here: it is recomputed after decoding
            # so an AR proposal cannot insert its newly generated B_i into
            # U_i ahead of the runtime admission boundary.
            self._expected_b_memory_update(
                str(sample["memory"]), str(sample["previous_intent"]), str(sample["task_name"]),
            )
            template = str(sample["template"])
            if not template.endswith("<EOC>"):
                raise ValueError("target-free planner template must stop exactly at EOC")
            for placeholder in (
                "<task_name_text_!>", "<previous_parent_goal_text_!>",
                "<previous_intent_text_!>", "<memory_text_!>",
                "<known_previous_outcome_text_!>", "<execution_feedback_text_!>",
            ):
                if placeholder not in template:
                    raise ValueError(f"target-free planner prefix is missing {placeholder}")
            if any(marker in template for marker in (
                "<outcome_target_text>", "<next_decision_text>", "<current_parent_goal_text>",
                "<active_skills_semantic_json_text>", "<memory_update_text>", "<task_complete_text>",
            )):
                raise ValueError("target-free planner template leaked an EOC-after target placeholder")
            image_indexes = sorted(
                int(key[5:]) for key in sample if key.startswith("image") and key[5:].isdigit()
            )
            if image_indexes != list(range(expected_images)):
                raise ValueError("target-free planner image placeholders disagree with configured history")
            proprio = sample["proprio"]
            if not isinstance(proprio, Mapping) or not isinstance(proprio.get("value"), torch.Tensor):
                raise ValueError("target-free planner requires processor-normalized proprio")

    def _parse_target_free_planner_event(self, text: str, *, sample: Mapping[str, Any]) -> dict[str, Any]:
        """Strictly parse an AR planner event; claims remain nonphysical."""
        raw = str(text or "").strip()
        raw = raw.replace("<HL_END>", "").replace("<EOV>", "").strip()
        # The actual training template has one static separator before HL_END.
        # Keep internal pipes (parallel parent commands / JSON strings) intact.
        if raw.endswith("|"):
            raw = raw[:-1]
        from g05.utils.memlite_planner_fields import split_planner_event_fields
        fields = split_planner_event_fields(raw)
        if len(fields) != 6:
            raise ValueError("planner generation must contain exactly six structured fields")
        outcome_text, decision_text, parent_text, skills_text, memory_text, complete_text = fields
        if not outcome_text.startswith("Previous outcome: "):
            raise ValueError("planner generation has no previous-outcome field")
        ar_outcome = outcome_text.removeprefix("Previous outcome: ").strip().upper()
        if ar_outcome not in VALID_OUTCOMES:
            raise ValueError("planner generation has invalid previous-outcome claim")
        if self.planner_only and ar_outcome != "UNKNOWN":
            raise ValueError("high_planner_only rejects AR outcome claims other than UNKNOWN")
        if not decision_text.startswith("Decision: "):
            raise ValueError("planner generation has no decision field")
        decision = decision_text.removeprefix("Decision: ").strip().upper()
        if decision not in VALID_DECISIONS or (self.planner_only and decision == "STOP"):
            raise ValueError("planner generation has unsupported/untrusted terminal decision")
        if not parent_text.startswith("Parent goal: "):
            raise ValueError("planner generation has no current parent goal")
        parent_goal = validate_semantic_parent_goal(
            parent_text.removeprefix("Parent goal: ").strip(), field="generated_parent_goal",
        )
        if not skills_text.startswith("Active skills: "):
            raise ValueError("planner generation has no active semantic bundle")
        raw_semantic = skills_text.removeprefix("Active skills: ").strip()
        try:
            semantic = json.loads(raw_semantic)
        except json.JSONDecodeError as exc:
            raise ValueError("planner generation active bundle is not JSON") from exc
        semantic_json, active_text = self._render_generated_active_skills(semantic)
        if raw_semantic != semantic_json:
            raise ValueError("planner generation active bundle is not canonical semantic JSON")
        if not memory_text.startswith("Memory update: "):
            raise ValueError("planner generation has no memory-update field")
        memory_update = memory_text.removeprefix("Memory update: ").strip()
        expected_memory = self._expected_b_memory_update(
            str(sample["memory"]), str(sample["previous_intent"]), str(sample["task_name"]),
        )
        if memory_update != expected_memory:
            raise ValueError("planner generation violates causal K=3 memory recurrence")
        if not complete_text.startswith("Task complete: "):
            raise ValueError("planner generation has no task-complete field")
        completion = complete_text.removeprefix("Task complete: ").strip().casefold()
        if completion not in {"true", "false"}:
            raise ValueError("planner generation task_complete is not boolean")
        task_complete_claimed = completion == "true"
        if self.planner_only and task_complete_claimed:
            raise ValueError("high_planner_only rejects unverified task-complete claims")
        if (decision == "STOP") != task_complete_claimed:
            raise ValueError("planner STOP/task_complete grammar is inconsistent")
        return {
            # No AR text is physical feedback.  Runtime may retain the raw
            # claims for audit, but its outcome observer/official evaluator is
            # the only authority for a transition or task success.
            "previous_outcome": "UNKNOWN",
            "ar_previous_outcome": ar_outcome,
            "decision": decision,
            "parent_goal": parent_goal,
            "active_skills_semantic_json": semantic_json,
            "active_skills_text": active_text,
            "memory_update": memory_update,
            "task_complete": False,
            "task_complete_claimed": task_complete_claimed,
            "validated": True,
        }

    @torch.no_grad()
    def generate_high_level(
        self,
        samples: List[Dict[str, Any]],
        pixel_values: Union[torch.Tensor, Dict[str, torch.Tensor]],
        *,
        memory_text: Optional[str] = None,
        max_new_tokens: Optional[int] = None,
        **ar_kwargs,
    ) -> Dict[str, Any]:
        """Generate one target-free initial or later high planner event.

        This override retains the established ``PolicyInferencer`` public
        call surface, so serving cannot accidentally keep invoking the old
        legacy MEM-Lite high template.  It does not update memory/state: the
        runtime commits a validated event through the published B K=3 state
        transition only after admission.
        """
        del memory_text  # text is already in each verified causal sample
        self._validate_target_free_high_prefix(samples)
        state = self.prefill(samples, pixel_values)
        hl_end_id = getattr(self.processor, "hl_end_token_id", None)
        if hl_end_id is None:
            raise RuntimeError(
                "target-free planner generation requires the configured <HL_END> token id; "
                "refusing an unbounded/incompletely delimited AR response"
            )
        stop = [int(hl_end_id)]
        # This is deliberately a generous, explicit planner budget rather
        # than the legacy short CoT cap.  The published B-memory fixture is
        # measured in the CPU contract test; a response that does not reach
        # HL_END within this bound is rejected below rather than partially
        # parsed as an executable plan.
        configured_budget = int(getattr(self, "planner_default_max_new_tokens", 1024))
        if configured_budget < 1:
            raise RuntimeError("planner configured max_new_tokens must be positive")
        # The deployment adapter must pin the reviewed config value into its
        # receipt.  An arbitrary per-request cap could silently turn a valid
        # long parallel/K=3 event into a truncation failure, or enlarge the
        # decoding surface without a matching config/manifest review.  The
        # measured 218-token B fixture is a lower-bound regression case, not
        # a universal replacement for this explicit 1024-token contract.
        if max_new_tokens is not None and int(max_new_tokens) != configured_budget:
            raise ValueError(
                "planner max_new_tokens must equal the configured serving budget; "
                "publish a new reviewed config/manifest to change it"
            )
        token_budget = configured_budget
        from g05.utils.memlite_planner_format import planner_only_format_constants
        with planner_only_format_constants(
            self.model.ar_helper, self.processor, batch_size=len(samples),
            hl_end_id=hl_end_id, enabled=self.planner_only,
        ) as format_receipt:
            state = self.generate_text(
                state,
                max_new_tokens=token_budget,
                stop_token_ids=stop,
                trim_token_ids=(stop or []) + (self._get_trim_token_ids() or []),
                **ar_kwargs,
            )
        generated_ids = getattr(state, "generated_ids", None)
        if not isinstance(generated_ids, torch.Tensor) or generated_ids.ndim != 2:
            raise RuntimeError("planner AR must return generated token ids [batch,tokens]")
        if generated_ids.shape[0] != len(samples) or not torch.isfinite(generated_ids).all():
            raise RuntimeError("planner AR emitted non-finite/misaligned token ids")
        if not torch.all((generated_ids == int(hl_end_id)).any(dim=1)):
            error = RuntimeError(
                "planner AR response hit generation bound before <HL_END>; "
                "rejecting incomplete planner event"
            )
            error.planner_texts = list(getattr(state, "generated_texts", None) or [])
            error.planner_token_count = int(generated_ids.shape[1])
            raise error
        texts = list(getattr(state, "generated_texts", None) or [])
        if len(texts) != len(samples):
            raise RuntimeError("planner AR did not return one text per target-free prefix")
        try:
            events = [
                self._parse_target_free_planner_event(text, sample=sample)
                for text, sample in zip(texts, samples)
            ]
        except ValueError as error:
            # Keep the actual failed decode for diagnosis. A rejected proposal
            # is never executed, but losing its text hides train/serve errors.
            error.planner_texts = texts
            error.planner_token_count = int(generated_ids.shape[1])
            raise
        return {
            "planner_events": events,
            # Compatibility fields consumed by the existing inferencer.  They
            # are proposal text/state only, not physical outcome/termination.
            "intent": [event["active_skills_text"] for event in events],
            "memory": [event["memory_update"] for event in events],
            "status": [event["decision"] for event in events],
            "high_level_text": texts,
            "high_level_state": state,
            "high_level_generated_ids": generated_ids,
            "planner_format_receipt": format_receipt,
        }

    # The service owns decoding / normalization of an RGB and proprio request.
    # It then supplies the exact processor output as ``prepared_branch`` below.
    # Keeping this small adapter in the model package makes the *second* half of
    # that hand-off auditable: target-bearing high-level samples cannot be
    # accidentally recycled as an outcome-observer prefix.
    _OBSERVABLE_CAMERA_ORDER = (
        "head_rgb", "left_wrist_rgb", "right_wrist_rgb",
    )
    _OBSERVABLE_FORBIDDEN_KEY_FRAGMENTS = (
        "oracle", "evidence", "ground_truth", "goal_status", "contact",
        "snapshot", "sim_state", "target_parent", "audit", "episode",
        "frame_index", "skill_idx", "outcome_target", "bundle_result",
    )

    @classmethod
    def observable_outcome_template(cls, *, num_input_images: int = 18) -> str:
        """Return the standalone, target-free six-frame outcome prefix.

        This is deliberately not :class:`PlannerOutcomeBuilder`'s template
        with synthetic EOC-after values.  Service code uses it while it runs
        the official image/state processor, then calls
        :meth:`prepare_outcome_prefix` with the resulting image metadata and
        normalized proprio.  The template is fixed to three cameras × six
        chronological frames, in the camera-major order enforced below.
        """
        if int(num_input_images) != 18:
            raise ValueError("observable outcome prefix requires exactly 18 images (3 cameras x 6 frames)")
        images = "".join(f"<image{index}_image_!>" for index in range(18))
        return (
            "<chat_user_prefix>" + images + "<bos>"
            "Embodiment: <embodiment_text_!>; Task: <command_text_!_200>; "
            "Task goal: <task_name_text_!>; Previous parent goal: <previous_parent_goal_text_!>; "
            "Previous bundle: <previous_intent_text_!>; Memory: <memory_text_!>; "
            "Known result: <known_previous_outcome_text_!>; "
            "Execution feedback: <execution_feedback_text_!>; State: <proprio_proprio_!>;"
            "<chat_user_suffix><chat_assistant_prefix>"
            "<planner_prompt_text_!><EOC>"
        )

    @staticmethod
    def _observable_field(observation: Any, name: str) -> Any:
        if isinstance(observation, Mapping):
            if name not in observation:
                raise ValueError(f"observable outcome input is missing {name!r}")
            return observation[name]
        if not hasattr(observation, name):
            raise ValueError(f"observable outcome input is missing {name!r}")
        return getattr(observation, name)

    @classmethod
    def _reject_observable_oracle_payload(cls, value: Any, *, path: str = "observation") -> None:
        """Recursively reject privileged simulation/annotation fields.

        This is deliberately duplicated at the model boundary.  The serving
        contract performs the same check before it calls us; neither boundary
        is trusted as the sole protection against a future wrapper change.
        Semantic object *targets* remain legal -- they are part of a previous
        generated skill bundle -- while target-parent / label / audit fields
        are not.
        """
        if isinstance(value, Mapping):
            for raw_key, child in value.items():
                key = str(raw_key)
                normalized = key.casefold()
                if any(fragment in normalized for fragment in cls._OBSERVABLE_FORBIDDEN_KEY_FRAGMENTS):
                    raise ValueError(f"observable outcome input contains forbidden field {path}.{key}")
                cls._reject_observable_oracle_payload(child, path=f"{path}.{key}")
        elif isinstance(value, (list, tuple)):
            for index, child in enumerate(value):
                cls._reject_observable_oracle_payload(child, path=f"{path}[{index}]")

    @staticmethod
    def _json_observable_summary(value: Any, *, field: str) -> Any:
        """Convert a bounded, deployment-visible action summary to JSON.

        The runtime normally sends Python floats/lists.  A small CPU tensor is
        accepted for its convenience, but arbitrary objects and large tensors
        are rejected instead of being converted through a surprising repr.
        """
        if isinstance(value, torch.Tensor):
            if value.numel() > 512:
                raise ValueError(f"{field} tensor is too large for an execution summary")
            if not torch.isfinite(value).all():
                raise ValueError(f"{field} tensor contains non-finite values")
            return value.detach().cpu().tolist()
        if value is None or isinstance(value, (str, bool, int, float)):
            return value
        if isinstance(value, Mapping):
            return {
                str(key): G05PolicyMEMLitePlannerOutcome._json_observable_summary(
                    child, field=f"{field}.{key}"
                )
                for key, child in value.items()
            }
        if isinstance(value, (list, tuple)):
            if len(value) > 512:
                raise ValueError(f"{field} has too many entries")
            return [
                G05PolicyMEMLitePlannerOutcome._json_observable_summary(
                    child, field=f"{field}[{index}]"
                )
                for index, child in enumerate(value)
            ]
        # NumPy scalar/array is a common serving representation.  ``tolist``
        # is the only implicit conversion allowed, and its result is checked
        # recursively rather than stringified.
        tolist = getattr(value, "tolist", None)
        if callable(tolist):
            return G05PolicyMEMLitePlannerOutcome._json_observable_summary(
                tolist(), field=field
            )
        raise ValueError(f"{field} is not a JSON-safe observable summary")

    @classmethod
    def prepare_outcome_prefix(
        cls,
        observation: Any,
        *,
        prepared_branch: Mapping[str, Any],
    ) -> Dict[str, Any]:
        """Make one processor-normalized, prefix-only observer sample.

        ``prepared_branch`` is the `samples`/`pixel_values` result of the
        *official* inference processor.  Only image metadata, normalized
        proprio and embodiment are copied from it.  All text is rebuilt from
        observable runtime state, so an EOC-after training target can never
        enter the learned outcome head through a reused sample dictionary.

        The public runtime input is intentionally duck-typed to avoid a model
        dependency on the service's dataclass.  Required fields are exactly
        ``task_name, rgb_history, proprio_history, executed_action_summary,
        memory_text, active_bundle, served_action_count``.
        """
        if not isinstance(prepared_branch, Mapping):
            raise ValueError("prepared_branch must be a mapping with samples and pixel_values")
        source_samples = prepared_branch.get("samples")
        if not isinstance(source_samples, (list, tuple)) or len(source_samples) != 1:
            raise ValueError("observable outcome encoding currently requires exactly one prepared sample")
        source = source_samples[0]
        if not isinstance(source, Mapping):
            raise ValueError("prepared_branch.samples[0] must be a mapping")

        required_names = (
            "task_name", "rgb_history", "proprio_history", "executed_action_summary",
            "memory_text", "active_bundle", "served_action_count",
        )
        visible = {name: cls._observable_field(observation, name) for name in required_names}
        cls._reject_observable_oracle_payload(visible)

        task_name = str(visible["task_name"] or "").strip()
        if not task_name:
            raise ValueError("observable outcome input requires a nonempty deployment task_name")
        rgb_history = visible["rgb_history"]
        if not isinstance(rgb_history, Mapping) or tuple(rgb_history.keys()) != cls._OBSERVABLE_CAMERA_ORDER:
            raise ValueError(
                "observable rgb_history must use exact camera order "
                "[head_rgb, left_wrist_rgb, right_wrist_rgb]"
            )
        if any(not isinstance(rgb_history[camera], (list, tuple)) or len(rgb_history[camera]) != 6
               for camera in cls._OBSERVABLE_CAMERA_ORDER):
            raise ValueError("observable outcome encoding requires six chronological frames per official camera")
        if not isinstance(visible["proprio_history"], Mapping) or not visible["proprio_history"]:
            raise ValueError("observable outcome input requires proprio_history")
        if not isinstance(visible["executed_action_summary"], Mapping):
            raise ValueError("observable outcome input requires a mapping execution summary")
        try:
            served_action_count = int(visible["served_action_count"])
        except (TypeError, ValueError) as exc:
            raise ValueError("served_action_count must be an integer") from exc
        if served_action_count < 0:
            raise ValueError("served_action_count must be nonnegative")

        active_bundle = visible["active_bundle"]
        if not isinstance(active_bundle, Mapping):
            raise ValueError("observable outcome input requires an active_bundle mapping")
        # `target_parent_goal` is never an accepted spelling here: by the time
        # a high-level prediction is stored it must have become `parent_goal`.
        if "target_parent_goal" in active_bundle or "active_skills_json" in active_bundle:
            raise ValueError("active_bundle must contain deployment parent_goal and semantic skills only")
        parent_goal = str(active_bundle.get("parent_goal", "") or "").strip()
        if not parent_goal:
            raise ValueError("active_bundle.parent_goal is required for outcome observation")
        semantic_json = active_bundle.get("active_skills_semantic_json")
        if not isinstance(semantic_json, str) or not semantic_json.strip():
            raise ValueError("active_bundle requires nonempty active_skills_semantic_json")
        try:
            semantic_skills = json.loads(semantic_json)
        except json.JSONDecodeError as exc:
            raise ValueError("active_bundle.active_skills_semantic_json is invalid JSON") from exc
        allowed_semantic_fields = {
            "verb", "target", "source", "destination", "target_part", "arm", "unbound_relation",
        }
        if not isinstance(semantic_skills, list) or not semantic_skills:
            raise ValueError("active_bundle must keep a nonempty complete semantic skill bundle")
        if any(not isinstance(skill, Mapping) or set(skill) - allowed_semantic_fields
               or not isinstance(skill.get("verb"), str) or not skill["verb"].strip()
               for skill in semantic_skills):
            raise ValueError("active_bundle semantic skills are malformed or contain audit fields")

        # Extract the real image-marker metadata / normalized proprio from the
        # same processor branch that produced `pixel_values`; do not retain any
        # text, labels or audit keys from the source training sample.
        result = {
            key: value
            for key, value in source.items()
            if key in {"embodiment", "proprio", "proprio_dim_is_pad", "frequency"}
            or (key.startswith("image") and key[5:].isdigit())
        }
        if "embodiment" not in result or "proprio" not in result:
            raise ValueError("prepared branch is missing processor-normalized embodiment/proprio")
        image_indexes = sorted(
            int(key[5:]) for key in result if key.startswith("image") and key[5:].isdigit()
        )
        if image_indexes != list(range(18)):
            raise ValueError("prepared branch must provide image0..image17 in camera-major history order")
        result.update(
            template=cls.observable_outcome_template(num_input_images=len(image_indexes)),
            command=task_name,
            task_name=task_name,
            previous_parent_goal=parent_goal,
            previous_intent="Previous active skills: " + json.dumps(
                semantic_skills, ensure_ascii=False, separators=(",", ":"), sort_keys=True
            ),
            memory=str(visible["memory_text"] or "").strip() or "Memory: none.",
            # The head is evaluating the active bundle itself.  A newly
            # observed result is deliberately not copied into the prefix.
            known_previous_outcome="Known previous outcome: UNKNOWN",
            execution_feedback=(
                f"served_action_count={served_action_count}; executed_action_summary="
                + json.dumps(
                    cls._json_observable_summary(
                        visible["executed_action_summary"], field="executed_action_summary"
                    ), ensure_ascii=False, separators=(",", ":"), sort_keys=True
                )
            ),
            planner_prompt="Assess only the previous active bundle from observable history.",
            schema_version=MEMLITE_SKILL_SCHEMA_VERSION,
            memlite_schema_version=MEMLITE_SKILL_SCHEMA_VERSION,
        )
        cls._validate_observable_outcome_prefix([result])
        return result

    @torch.no_grad()
    def outcome_context_from_observable(
        self,
        observation: Any,
        *,
        prepared_branch: Mapping[str, Any],
    ) -> torch.Tensor:
        """Encode one service ``ObservableOutcomeInput`` through the real VLM.

        It returns ``[1, hidden_size]``.  The caller must pass that tensor to
        :meth:`outcome_prediction_from_context_hidden`; this method neither
        decodes AR text nor turns a result into a transition.
        """
        sample = self.prepare_outcome_prefix(observation, prepared_branch=prepared_branch)
        pixel_values = prepared_branch.get("pixel_values")
        if pixel_values is None:
            raise ValueError("prepared_branch is missing pixel_values")
        return self.outcome_context_from_prefix([sample], pixel_values)

    @staticmethod
    def _validate_observable_outcome_prefix(samples: List[Dict[str, Any]]) -> None:
        """Reject target/oracle fields before encoding deployment observations.

        This is intentionally separate from :meth:`_validate_high_batch`.
        The latter validates a teacher-forced planner row and therefore
        requires EOC-after targets; an outcome observer must never construct
        or receive those targets in the first place.
        """
        required = {
            "template", "command", "task_name", "previous_parent_goal",
            "previous_intent", "memory", "known_previous_outcome",
            "execution_feedback", "proprio", "embodiment",
        }
        forbidden_exact = {
            "target_parent_goal", "current_parent_goal", "outcome_target",
            "outcome_target_value", "outcome_supervision_mask",
            "active_skills_semantic_json", "active_skills_json",
            "memory_update", "next_decision", "task_complete",
        }
        forbidden_fragments = (
            "oracle", "evidence", "ground_truth", "goal_status", "contact",
            "snapshot", "sim_state", "target_parent", "audit", "episode",
            "frame_index", "skill_idx", "bundle_result",
        )
        for index, sample in enumerate(samples):
            if not isinstance(sample, Mapping):
                raise ValueError(f"observable outcome sample {index} must be a mapping")
            keys = {str(key) for key in sample}
            missing = required - keys
            if missing:
                raise ValueError(
                    f"observable outcome sample {index} is missing causal prefix fields {sorted(missing)}"
                )
            bad = forbidden_exact & keys
            bad.update(key for key in keys if any(fragment in key.casefold() for fragment in forbidden_fragments))
            if bad:
                raise ValueError(
                    "observable outcome prefix contains target/audit/oracle field(s): "
                    f"{sorted(bad)}"
                )
            if int(sample.get("memlite_schema_version", sample.get("schema_version", -1))) != MEMLITE_SKILL_SCHEMA_VERSION:
                raise ValueError("observable outcome prefix has incompatible MEM-Lite schema")
            template = str(sample["template"])
            eoc = template.find("<EOC>")
            if eoc < 0:
                raise ValueError("observable outcome prefix template requires EOC")
            for placeholder in (
                "<task_name_text_!>", "<previous_parent_goal_text_!>",
                "<previous_intent_text_!>", "<memory_text_!>",
                "<known_previous_outcome_text_!>", "<execution_feedback_text_!>",
            ):
                if placeholder not in template or template.find(placeholder) > eoc:
                    raise ValueError(
                        f"observable outcome prefix requires causal EOC-prefix placeholder {placeholder}"
                    )
            task_name = str(sample["task_name"]).strip()
            if not task_name or task_name.casefold() == "none":
                raise ValueError("observable outcome prefix needs the original deployment task text")
            if str(sample["command"]).strip() != task_name:
                raise ValueError(
                    "observable outcome prefix command must exactly equal task_name; "
                    "task ids/indexes are not a substitute"
                )
            proprio = sample["proprio"]
            if not isinstance(proprio, Mapping) or not isinstance(proprio.get("value"), torch.Tensor):
                raise ValueError("observable outcome prefix requires processor-normalized proprio tensor")

    @torch.no_grad()
    def outcome_context_from_prefix(
        self,
        samples: List[Dict[str, Any]],
        pixel_values: Union[torch.Tensor, Dict[str, torch.Tensor]],
    ) -> torch.Tensor:
        """Encode an observable-only planner prefix into ``[B, hidden_size]``.

        This is the model side of the serving outcome observer.  The serving
        adapter owns the conversion of its ``ObservableOutcomeInput`` into a
        processor-normalized *prefix-only* sample and calls this method.  The
        adapter may use only its six RGB frames, 27-D normalized proprio
        history, executed-action summary, old bundle/parent goal, old memory,
        served-action count, and original task text.  It must not append
        planner targets, annotation/audit ids, simulator truth or evidence.

        The returned state is drawn from the final valid EOC-prefix token of
        the real VLM prefill, not from an AR-generated or teacher-forced token.
        Call :meth:`outcome_prediction_from_context_hidden` on this tensor for
        the sole learned outcome proposal.
        """
        self._validate_observable_outcome_prefix(samples)
        if isinstance(pixel_values, dict):
            if not pixel_values:
                raise ValueError("observable outcome prefix has empty pixel_values")
            first_image = next(iter(pixel_values.values()))
        else:
            first_image = pixel_values
        if not isinstance(first_image, torch.Tensor) or first_image.ndim < 4:
            raise ValueError("observable outcome prefix requires batched image tensors")
        device, dtype = first_image.device, first_image.dtype
        input_ids, attention_mask = self.processor.encode_inference(
            samples,
            device=device,
            mode="ar",
            training=False,
        )
        self._assert_observable_tokenized_prefix(input_ids, attention_mask, samples)
        pixel_values_processed = self.process_pixel_values(pixel_values)
        proprio_batch = (
            build_proprio_batch(
                samples,
                device=device,
                dtype=torch.float32,
                zero_values=self.model.proprio_encoder == "zeros",
            )
            if self.model.proprio_embedder is not None
            else None
        )
        hidden, _kv, _position_ids = self.model.vlm_prefill(
            input_ids,
            attention_mask,
            pixel_values_processed,
            dtype=dtype,
            proprio=proprio_batch,
        )
        positions = attention_mask.to(dtype=torch.long).sum(dim=-1) - 1
        if (positions < 0).any():
            raise ValueError("observable outcome prefix has an empty token context")
        rows = torch.arange(len(samples), device=device)
        return hidden[rows, positions]

    def _assert_observable_tokenized_prefix(
        self,
        context_ids: torch.LongTensor,
        context_attention: torch.Tensor,
        samples: List[Dict[str, Any]],
    ) -> None:
        """Prove that actual observer tokens contain its task but no future fields."""
        text_processor = self.processor.modality_processors["text"]

        def ids_for(sample, value):
            ids, _labels, _attention = text_processor.process(
                value,
                is_masked=True,
                training=False,
                tokenizer=self.processor.tokenizer,
                sample_dict=sample,
            )
            return list(ids)

        def occurs(needle, haystack):
            return bool(needle) and any(
                haystack[index:index + len(needle)] == needle
                for index in range(len(haystack) - len(needle) + 1)
            )

        for index, sample in enumerate(samples):
            prefix = context_ids[index, context_attention[index] != 0].tolist()
            if not occurs(ids_for(sample, str(sample["task_name"])), prefix):
                raise RuntimeError("observable outcome prefix lost the original task_name")

    def configure_coordination_trainability(self):
        """Enable the profile-owned high-level tensors, nothing else.

        ``high_planner_only`` updates only planner CE tensors.  The outcome
        head is individually frozen (rather than merely excluded by a module
        prefix), which makes accidental optimizer inclusion fail coverage
        checks below.  ``high_planner_outcome`` remains the separate future
        physical-label/calibration profile.
        """
        enabled_names = []
        for name, parameter in self.named_parameters():
            enabled = (
                name.startswith("model.vlm.")
                or name.startswith("model.multi_modal_projector.")
                or name.startswith("model.proprio_embedder.")
            )
            if not self.planner_only:
                enabled = enabled or name.startswith("outcome_head.")
            parameter.requires_grad_(enabled)
            if enabled:
                enabled_names.append(name)
        groups = self.coordination_trainable_parameter_groups()
        expected = {"planner_vlm"}
        if not self.planner_only:
            expected.add("outcome_head")
        if set(groups) != expected:
            raise RuntimeError(f"Planner stage gate failed: expected={sorted(expected)} actual={sorted(groups)}")
        return {
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

    def coordination_trainable_parameter_groups(self):
        groups = defaultdict(list)
        for name, parameter in self.named_parameters():
            if not parameter.requires_grad:
                continue
            if name.startswith("outcome_head."):
                key = "outcome_head"
            elif name.startswith(("model.vlm.", "model.multi_modal_projector.", "model.proprio_embedder.")):
                key = "planner_vlm"
            elif name.startswith("model.vision_tower."):
                key = "vision"
            elif name.startswith("model.action_expert."):
                key = "action_expert"
            else:
                key = "other"
            groups[key].append((name, parameter))
        return dict(groups)

    def _validate_high_batch(self, samples: List[Dict[str, Any]]) -> None:
        self._memlite_branch_masks(samples, "high")
        for index, sample in enumerate(samples):
            try:
                validate_embedded_model_projection(sample)
            except (ValueError, TypeError) as error:
                raise ValueError(
                    f"PlannerOutcome sample {index} carries an audit payload into model inputs"
                ) from error
            if int(sample.get("memlite_schema_version", sample.get("schema_version", -1))) != self.interface_schema_version:
                raise ValueError(f"PlannerOutcome sample {index} has incompatible schema version")
            template = str(sample.get("template", ""))
            eoc = template.find("<EOC>")
            if eoc < 0:
                raise ValueError("PlannerOutcome template requires an EOC boundary")
            known_slot = f"<{planner_field_slot(sample, 'known_previous_outcome')}_text_!>"
            if known_slot not in template or template.find(known_slot) > eoc:
                raise ValueError("known previous outcome must be an EOC-prefix input")
            if "<previous_parent_goal_text_!>" not in template or template.find("<previous_parent_goal_text_!>") > eoc:
                raise ValueError("only previous_parent_goal may be an EOC-prefix parent input")
            if "<task_name_text_!>" not in template or template.find("<task_name_text_!>") > eoc:
                raise ValueError("PlannerOutcome requires the original task_name as an EOC-prefix input")
            if "<current_parent_goal_text>" not in template or template.find("<current_parent_goal_text>") < eoc:
                raise ValueError("current parent_goal must be predicted EOC-after, never read as input")
            outcome_slot = f"<{planner_field_slot(sample, 'outcome_target')}_text>"
            if outcome_slot not in template or template.find(outcome_slot) < eoc:
                raise ValueError("outcome target must be EOC-after only")
            if "<action_action>" in template:
                raise ValueError("PlannerOutcome never builds ActionCodec targets")
            supervised = bool(sample.get("outcome_supervision_mask", False))
            target = str(sample.get("outcome_target_value", "")).upper()
            rendered = planner_field_text(sample, "outcome_target")
            if supervised:
                if target not in {"IN_PROGRESS", "SUCCEEDED", "FAILED", "UNKNOWN"}:
                    raise ValueError(f"PlannerOutcome sample {index} has invalid supervised outcome")
                if target not in rendered:
                    raise ValueError("supervised outcome must be the first EOC-after AR field")
            elif rendered != f"Previous outcome: {target}":
                raise ValueError("unobserved result must retain fixed grammar for parser-safe token masking")
            if self.planner_only:
                self._validate_planner_only_outcome_contract(sample, index=index)

    @staticmethod
    def _validate_planner_only_outcome_contract(sample: Mapping[str, Any], *, index: int) -> None:
        """Reject every attempt to turn missing demo feedback into a label."""
        if bool(sample.get("outcome_supervision_mask", False)):
            raise ValueError(
                f"high_planner_only sample {index} must not unmask outcome supervision"
            )
        if str(sample.get("outcome_target_value", "")).upper() != "UNKNOWN":
            raise ValueError(
                f"high_planner_only sample {index} must retain UNKNOWN outcome grammar"
            )
        if planner_field_text(sample, "outcome_target") != "Previous outcome: UNKNOWN":
            raise ValueError(
                f"high_planner_only sample {index} has a noncanonical unknown outcome target"
            )
        # An annotation ending is not a physical success observation.  The
        # B-memory overlay permits a STOP target only when earlier task-success
        # evidence exists; that belongs to the later high_planner_outcome
        # profile.  Planner-only has no such evidence and must not train an
        # UNKNOWN terminal as if it were a successful completion.
        decision = str(sample.get("next_decision", "")).strip().upper()
        terminal = str(sample.get("task_complete", "")).strip().casefold()
        if decision in {"STOP", "DECISION: STOP"} or terminal in {
            "true", "task complete: true",
        }:
            raise ValueError(
                f"high_planner_only sample {index} cannot train an UNKNOWN terminal row"
            )

    def _ar_target_field_spans(
        self,
        input_ids: torch.LongTensor,
        context_positions: List[int],
        samples: List[Dict[str, Any]],
    ) -> List[Dict[str, slice]]:
        """Return exact post-EOC field token spans, failing closed on drift.

        Each template segment is tokenized independently by InputPreprocessor;
        walking its exact post-EOC field sequence is safer than searching
        decoded strings or using an EOV split.  Both the evidence mask and the
        planner-only memory weight consume these *same* spans, so a template
        or tokenizer drift cannot silently weight a neighbouring field.
        """
        text_processor = self.processor.modality_processors["text"]

        def token_ids(sample, value):
            ids, _labels, _attention = text_processor.process(
                value,
                is_masked=False,
                training=self.training,
                tokenizer=self.processor.tokenizer,
                sample_dict=sample,
            )
            return list(ids)

        separator_ids = token_ids({}, "|")
        ordered_fields = (
            "outcome_target",
            "next_decision",
            "current_parent_goal",
            "active_skills_semantic_json",
            "memory_update",
            "task_complete",
        )
        spans_by_sample: List[Dict[str, slice]] = []
        for batch_index, (sample, context_position) in enumerate(zip(samples, context_positions)):
            cursor = context_position + 1
            spans: Dict[str, slice] = {}
            for field_index, field in enumerate(ordered_fields):
                field_ids = token_ids(sample, planner_field_text(sample, field))
                end = cursor + len(field_ids)
                if not torch.equal(input_ids[batch_index, cursor:end], input_ids.new_tensor(field_ids)):
                    raise RuntimeError(f"PlannerOutcome token boundary mismatch at field {field!r}")
                spans[field] = slice(cursor, end)
                cursor = end
                if field_index < len(ordered_fields) - 1:
                    separator_end = cursor + len(separator_ids)
                    if not torch.equal(
                        input_ids[batch_index, cursor:separator_end], input_ids.new_tensor(separator_ids)
                    ):
                        raise RuntimeError(f"PlannerOutcome separator mismatch after {field!r}")
                    cursor = separator_end
            spans_by_sample.append(spans)
        return spans_by_sample

    def _mask_unsupervised_ar_fields(
        self,
        input_ids: torch.LongTensor,
        labels: torch.LongTensor,
        context_positions: List[int],
        samples: List[Dict[str, Any]],
    ) -> List[Dict[str, slice]]:
        """Mask only missing-evidence targets while keeping a fixed AR grammar.

        Returns the verified spans so planner-only CE weighting can use exactly
        the same tokenizer boundary after masking.  Existing callers can
        ignore the return value without changing their semantics.
        """
        spans_by_sample = G05PolicyMEMLitePlannerOutcome._ar_target_field_spans(
            self, input_ids, context_positions, samples
        )
        masks = {
            "outcome_target": "outcome_supervision_mask",
            "current_parent_goal": "current_parent_goal_supervision_mask",
        }
        # ``high_planner_only`` consumes original demonstrations without a
        # physical terminal observation.  Its nonterminal ``false`` is kept
        # in the fixed generated grammar, but is not a negative task-success
        # label and must not receive CE.  A later outcome-capable profile has
        # a separate evidence/calibration contract and intentionally retains
        # its existing terminal supervision behavior.
        format_only_fields = {"task_complete"} if getattr(self, "planner_only", False) else set()
        for batch_index, (sample, spans) in enumerate(zip(samples, spans_by_sample)):
            for field, span in spans.items():
                # This does not claim an evidenced fine-grained parent. It
                # teaches the exact public-task fallback already present in
                # masked-parent rows, closing a teacher-forcing/decode hole.
                if (field == "current_parent_goal"
                        and getattr(self, "supervise_task_parent_format", False)
                        and is_declared_task_parent_format(sample, self.task_parent_formats)):
                    continue
                if field in format_only_fields or (
                    field in masks and not bool(sample.get(masks[field], False))
                ):
                    labels[batch_index, span] = IGNORE_INDEX
        return spans_by_sample

    def _planner_only_loss_token_weights(
        self,
        labels: torch.LongTensor,
        spans_by_sample: List[Dict[str, slice]],
        *,
        samples: Optional[List[Dict[str, Any]]] = None,
    ) -> tuple[Optional[torch.Tensor], torch.LongTensor]:
        """Build B-only CE weights after the evidence mask has been applied.

        The field index tensor is metrics-only.  The optional weight tensor is
        passed to the existing AR helper, whose shift and valid-label handling
        remains the sole CE implementation.  Outcome, terminal and ambiguous
        parent fields therefore remain IGNORE_INDEX before any weighting is
        considered.
        """
        format_supervision = getattr(self, "supervise_task_parent_format", False)
        if format_supervision and (samples is None or len(samples) != len(spans_by_sample)):
            raise ValueError("Parent-format metrics require the exact aligned model samples")
        field_codes = torch.zeros_like(labels, dtype=torch.long)
        codes = {
            "outcome_target": 1,
            "next_decision": 2,
            "current_parent_goal": 3,
            "active_skills_semantic_json": 4,
            "memory_update": 5,
            "task_complete": 6,
        }
        weights = torch.ones_like(labels, dtype=torch.float32)
        for batch_index, spans in enumerate(spans_by_sample):
            for field, span in spans.items():
                code = codes[field]
                if (format_supervision and field == "current_parent_goal"
                        and is_declared_task_parent_format(samples[batch_index], self.task_parent_formats)):
                    code = 7  # Known fallback format, not parent semantic evidence.
                field_codes[batch_index, span] = code
                if field == "memory_update":
                    weights[batch_index, span] = self.memory_update_ce_weight
        # Keep masked positions visibly inert even if a future helper changes
        # its valid-token gather.  ARHelper still requires labels != IGNORE.
        weights.masked_fill_(labels == IGNORE_INDEX, 0.0)
        if self.memory_update_ce_weight == 1.0:
            # Preserve the old `token_loss.mean()` code path and its numerical
            # behaviour byte-for-byte for the default profile.
            return None, field_codes
        return weights, field_codes

    @staticmethod
    def _planner_only_field_loss_metrics(
        cache: Mapping[str, Any],
        field_codes: torch.LongTensor,
        *,
        device: torch.device,
    ) -> Dict[str, torch.Tensor]:
        """Expose detached field means without creating another loss path."""
        token_loss = cache.get("token_loss")
        valid_mask = cache.get("mask")
        if not isinstance(token_loss, torch.Tensor) or not isinstance(valid_mask, torch.Tensor):
            return {}
        shifted_codes = field_codes[:, 1:].contiguous().view(-1).to(valid_mask.device)
        valid_codes = shifted_codes[valid_mask]
        if valid_codes.numel() != token_loss.numel():
            raise RuntimeError("PlannerOutcome field metrics no longer align with ARHelper valid tokens")
        names = {
            1: "outcome", 2: "decision", 3: "parent", 4: "bundle",
            5: "memory_update", 6: "task_complete", 0: "structural",
        }
        has_parent_format = bool((valid_codes == 7).any())
        if has_parent_format:
            names[7] = "parent_format"
        result: Dict[str, torch.Tensor] = {}
        if has_parent_format:
            weights = cache.get("token_weights")
            if weights is None:
                weights = torch.ones_like(token_loss)
            if not isinstance(weights, torch.Tensor) or weights.shape != token_loss.shape:
                raise RuntimeError("Parent-format metrics lost aligned CE weights")
            legacy = valid_codes != 7
            if not bool(legacy.any()):
                raise RuntimeError("A planner batch cannot consist only of parent-format labels")
            result["ce_without_parent_format_mean"] = (
                (token_loss[legacy] * weights[legacy]).sum() / weights[legacy].sum()
            ).to(device=device)
        for code, name in names.items():
            selected = valid_codes == code
            count = selected.sum()
            result[f"ce_field_{name}_tokens"] = count.to(device=device)
            result[f"ce_field_{name}_mean"] = (
                token_loss[selected].mean().to(device=device)
                if bool(selected.any())
                else torch.zeros((), device=device, dtype=token_loss.dtype)
            )
        return result

    @staticmethod
    def _context_positions(
        full_ids: torch.LongTensor,
        full_attention: torch.Tensor,
        context_ids: torch.LongTensor,
        context_attention: torch.Tensor,
    ) -> List[int]:
        """Assert full-training prefix equals inference context and return h indices."""
        positions: List[int] = []
        for batch_index in range(full_ids.shape[0]):
            # Both encodings may be left-padded. This planner's EOV is after
            # its AR targets, so encode_train right-aligns the whole sequence.
            # Locate the actual first token; batch=1 hid this padding bug.
            context_valid = context_attention[batch_index] != 0
            context_length = int(context_valid.sum().item())
            if context_length < 1:
                raise ValueError("PlannerOutcome has an empty EOC-prefix context")
            context_tokens = context_ids[batch_index, context_valid]
            full_valid_positions = (full_attention[batch_index] != 0).nonzero(as_tuple=True)[0]
            if full_valid_positions.numel() < context_length:
                raise ValueError("PlannerOutcome full sequence is shorter than its context")
            start = int(full_valid_positions[0].item())
            full_prefix = full_ids[batch_index, start:start + context_length]
            if not torch.equal(full_prefix, context_tokens):
                raise RuntimeError(
                    "PlannerOutcome EOC-prefix mismatch: outcome head would not read the deployment context"
                )
            if (full_attention[batch_index, start:start + context_length] == 0).any():
                raise RuntimeError("PlannerOutcome full EOC-prefix unexpectedly contains padding")
            positions.append(start + context_length - 1)
        return positions

    def _assert_tokenized_prefix_contract(
        self,
        context_ids: torch.LongTensor,
        context_attention: torch.Tensor,
        samples: List[Dict[str, Any]],
    ) -> None:
        """Prove the real inference tokenizer sees task text, never target fields.

        Template-string checks catch placeholder placement; this second check
        catches a bad sidecar mapping such as task-name/index loss after the
        builder has populated samples.  The target parent/outcome strings have
        unique field labels, so their appearance would be direct EOC-prefix
        leakage rather than harmless token overlap with the natural task.
        """
        text_processor = self.processor.modality_processors["text"]

        def ids_for(sample, value):
            ids, _labels, _attention = text_processor.process(
                value,
                is_masked=True,
                training=self.training,
                tokenizer=self.processor.tokenizer,
                sample_dict=sample,
            )
            return list(ids)

        def occurs(needle, haystack):
            return bool(needle) and any(
                haystack[index:index + len(needle)] == needle
                for index in range(len(haystack) - len(needle) + 1)
            )

        for batch_index, sample in enumerate(samples):
            task_name = str(sample.get("task_name", "")).strip()
            if not task_name or task_name.casefold() == "none":
                raise RuntimeError("PlannerOutcome requires nonempty natural-language task_name")
            prefix = context_ids[batch_index, context_attention[batch_index] != 0].tolist()
            if not occurs(ids_for(sample, task_name), prefix):
                raise RuntimeError("PlannerOutcome tokenized prefix lost the original task_name")
            for target_field in ("current_parent_goal", "outcome_target"):
                if occurs(ids_for(sample, planner_field_text(sample, target_field)), prefix):
                    raise RuntimeError(
                        f"PlannerOutcome tokenized prefix leaked EOC-after target {target_field!r}"
                    )

    def forward_train(
        self,
        samples: List[Dict[str, Any]],
        pixel_values: Union[torch.Tensor, Dict[str, torch.Tensor]],
        actions: Optional[torch.FloatTensor] = None,
        action_pad_masks: Optional[torch.BoolTensor] = None,
        action_dim_is_pad: Optional[torch.BoolTensor] = None,
        **kwargs,
    ):
        del actions, action_pad_masks, action_dim_is_pad, kwargs
        self._validate_high_batch(samples)
        expected_groups = {"planner_vlm"}
        if not self.planner_only:
            expected_groups.add("outcome_head")
        if set(self.coordination_trainable_parameter_groups()) != expected_groups:
            raise RuntimeError(
                "PlannerOutcome requires configure_coordination_trainability before DDP/forward; "
                f"expected={sorted(expected_groups)}"
            )
        if isinstance(pixel_values, dict):
            first_image = next(iter(pixel_values.values()))
            device, dtype = first_image.device, first_image.dtype
        else:
            device, dtype = pixel_values.device, pixel_values.dtype

        input_ids, labels, attention_mask, _split_index = self.processor.encode_train(
            samples,
            device=device,
            training=self.training,
            max_chunk_token_length=self.max_chunk_token_length,
            max_pad_token_length=self.max_pad_token_length,
        )
        context_ids, context_attention = self.processor.encode_inference(
            samples,
            device=device,
            mode="ar",
            training=self.training,
        )
        self._assert_tokenized_prefix_contract(context_ids, context_attention, samples)
        context_positions = self._context_positions(
            input_ids, attention_mask, context_ids, context_attention
        )
        # This test is intentionally narrow: no target token can be selected
        # by an off-by-one context index even if an EOV/EOC template changes.
        for batch_index, position in enumerate(context_positions):
            if labels[batch_index, position].item() != IGNORE_INDEX:
                raise RuntimeError("PlannerOutcome selected a supervised token instead of EOC-prefix context")
        spans_by_sample = self._mask_unsupervised_ar_fields(
            input_ids, labels, context_positions, samples
        )
        loss_token_weights: Optional[torch.Tensor] = None
        field_codes: Optional[torch.LongTensor] = None
        if self.planner_only and (
            getattr(self, "memory_update_ce_weight", 1.0) != 1.0
            or getattr(self, "supervise_task_parent_format", False)
        ):
            loss_token_weights, field_codes = self._planner_only_loss_token_weights(
                labels, spans_by_sample, samples=samples
            )

        pixel_values_processed = self.process_pixel_values(pixel_values)
        proprio_batch = (
            build_proprio_batch(
                samples,
                device=device,
                dtype=torch.float32,
                zero_values=self.model.proprio_encoder == "zeros",
            )
            if self.model.proprio_embedder is not None
            else None
        )
        vlm_hidden, _vlm_kv, _position_ids = self.model.vlm_prefill(
            input_ids,
            attention_mask,
            pixel_values_processed,
            dtype=dtype,
            proprio=proprio_batch,
        )
        row_index = torch.arange(len(samples), device=device)
        context_index = torch.tensor(context_positions, device=device, dtype=torch.long)
        if loss_token_weights is None:
            ce_loss, overall_accuracy = self.model.ar_helper.train_step(
                self.model, vlm_hidden, labels
            )
        else:
            ce_loss, overall_accuracy = self.model.ar_helper.train_step(
                self.model, vlm_hidden, labels, loss_token_weights=loss_token_weights
            )
        cache = self.model.ar_helper._last_ce_cache or {}
        action_accuracy = cache.get("action_accuracy", 0.0)
        cot_accuracy = cache.get("cot_accuracy", 0.0)
        if self.language_loss_weight != 1.0:
            ce_loss = ce_loss * self.language_loss_weight
        if self.planner_only:
            # No outcome-head forward/loss is constructed in this profile.
            # This preserves the fixed parser grammar while ensuring that
            # ubiquitous unknown/mask=false rows cannot update a random head.
            total = ce_loss
            outcome_metrics = {
                "outcome_training_disabled": torch.ones((), device=device),
                "outcome_supervised_count": torch.zeros((), device=device, dtype=torch.long),
            }
        else:
            outcome_logits = self.outcome_head(vlm_hidden[row_index, context_index])
            outcome_targets = outcome_target_tensor(
                [sample["outcome_target_value"] for sample in samples], device=device
            )
            outcome_mask = torch.tensor(
                [bool(sample.get("outcome_supervision_mask", False)) for sample in samples],
                device=device,
                dtype=torch.bool,
            )
            outcome = self.outcome_head.loss(outcome_logits, outcome_targets, outcome_mask)
            total = ce_loss + self.outcome_loss_weight * outcome.loss
            outcome_metrics = {
                "outcome_loss": outcome.loss.detach(),
                "outcome_supervised_count": outcome.supervised_count.detach(),
                "outcome_accuracy": outcome.accuracy.detach(),
            }
        self.train_action_accuracy = overall_accuracy
        self.train_cot_accuracy = cot_accuracy
        self._train_acc.push("overall", overall_accuracy)
        self._train_acc.push("action_token", action_accuracy)
        self._train_acc.push("cot", cot_accuracy)
        self._fwd_step += 1
        metrics = {
            "ce_loss": ce_loss.detach(),
            "overall_accuracy": torch.as_tensor(overall_accuracy, device=device).detach(),
            "action_accuracy": torch.as_tensor(action_accuracy, device=device).detach(),
            "cot_accuracy": torch.as_tensor(cot_accuracy, device=device).detach(),
        }
        if self.planner_only:
            raw_unweighted = cache.get("unweighted_token_ce")
            weighted_token = cache.get("weighted_token_ce")
            if isinstance(raw_unweighted, torch.Tensor):
                metrics["raw_unweighted_ce"] = raw_unweighted.to(device=device).detach()
            if isinstance(weighted_token, torch.Tensor):
                metrics["weighted_token_ce"] = weighted_token.to(device=device).detach()
            metrics["weighted_training_ce"] = ce_loss.detach()
            metrics["memory_update_ce_weight"] = torch.tensor(
                float(getattr(self, "memory_update_ce_weight", 1.0)),
                device=device,
                dtype=ce_loss.dtype,
            )
            if field_codes is not None:
                metrics.update(G05PolicyMEMLitePlannerOutcome._planner_only_field_loss_metrics(
                    cache, field_codes, device=device
                ))
                if "ce_without_parent_format_mean" in metrics:
                    metrics["ce_without_parent_format_training"] = (
                        metrics["ce_without_parent_format_mean"]
                        * self.model.ar_helper.ce_weight * self.language_loss_weight
                    )
        metrics.update(outcome_metrics)
        if self.planner_only:
            # These statistics describe the exact shifted, evidence-masked,
            # memory-weighted CE used above, not an average of microbatch means.
            mask = cache["mask"]
            token_loss = cache["token_loss"]
            weights = cache["token_weights"]
            owners = torch.arange(len(samples), device=device).unsqueeze(1).expand(
                -1, labels.shape[1] - 1).reshape(-1)[mask]
            denominator = torch.zeros(len(samples), device=device, dtype=torch.float32)
            numerator = torch.zeros_like(denominator)
            denominator.scatter_add_(0, owners, weights.float())
            numerator.scatter_add_(0, owners, (token_loss * weights).float())
            numerator *= self.model.ar_helper.ce_weight * self.language_loss_weight
            metrics.update(loss_denominator=denominator.sum(), row_loss_numerator=numerator,
                           row_loss_denominator=denominator)
        return total, metrics

    def get_optim_param_groups(
        self,
        lr,
        weight_decay,
        apply_decay_on_norm_and_bias=False,
        backbone_lr_multiplier=1.0,
        vision_lr_multiplier=1.0,
    ):
        """Return exactly the profile's trainable optimizer tensors."""
        del vision_lr_multiplier
        modules_by_name = dict(self.named_modules())
        buckets = {name: [] for name in ("planner_decay", "planner_no_decay")}
        if not self.planner_only:
            buckets.update({"outcome_decay": [], "outcome_no_decay": []})
        for name, parameter in self.named_parameters():
            if not parameter.requires_grad:
                continue
            owner_name, _, leaf_name = name.rpartition(".")
            apply_wd = self._should_apply_weight_decay(
                modules_by_name.get(owner_name), leaf_name, parameter
            )
            prefix = "outcome" if name.startswith("outcome_head.") else "planner"
            suffix = "decay" if apply_wd else "no_decay"
            buckets[f"{prefix}_{suffix}"].append(parameter)
        groups = [
            {"params": buckets["planner_decay"], "lr": lr * backbone_lr_multiplier,
             "weight_decay": weight_decay, "name": "planner_decay"},
            {"params": buckets["planner_no_decay"], "lr": lr * backbone_lr_multiplier,
             "weight_decay": 0.0, "name": "planner_no_decay"},
        ]
        if not self.planner_only:
            groups.extend([
                {"params": buckets["outcome_decay"], "lr": lr,
                 "weight_decay": weight_decay, "name": "outcome_decay"},
                {"params": buckets["outcome_no_decay"], "lr": lr,
                 "weight_decay": 0.0, "name": "outcome_no_decay"},
            ])
        total = sum(len(group["params"]) for group in groups)
        expected = sum(1 for parameter in self.parameters() if parameter.requires_grad)
        if total != expected:
            raise RuntimeError(f"PlannerOutcome optimizer coverage mismatch: {total} != {expected}")
        return groups
