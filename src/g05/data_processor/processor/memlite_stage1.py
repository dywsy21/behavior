# SPDX-License-Identifier: LicenseRef-G0.5-Community-1.0
# Copyright (c) 2026 Galaxea
"""Stage-1 builders restored from the audited A4/B-final coordination source.

The independent entry keeps the legacy mixed/AR data routes unchanged.
"""
from typing import Any, Dict
from .samples_builder import BaseSamplesBuilder
from .galaxea_cot_processor import GalaxeaCoTProcessor
from .memlite_v6_projection import _model_safe_v6_label
from g05.utils.memlite_skill_protocol import (
    MEMLITE_SKILL_SCHEMA_VERSION, MemLiteSkillProtocolError, planner_input_projection,
)

class _CameraMajorHistoryBuilder(BaseSamplesBuilder):
    """Match Qwen35's pixel-dict feature order: camera-major, then time.

    ``G05ModelQwen35._forward_vision`` concatenates every time step of the
    first dict camera before proceeding to the next camera.  The template image
    marker order must use exactly the same sequence; the generic builder's
    modulo order is frame-major and is only correct for one observation step.
    """

    def __init__(
        self,
        num_input_images,
        image_sizes,
        embodiment_type=None,
        high_builder=None,
        **kwargs,
    ):
        # The legacy mixed MEM-Lite task contributes this nullable field while
        # Hydra merges the processor dictionaries.  Accepting only the null
        # tombstone is necessary for a low-only builder to be constructible;
        # a real nested high builder is still an invalid route and must fail.
        if high_builder is not None:
            raise ValueError("MEM-Lite coordination builders may not nest a high_builder")
        super().__init__(num_input_images, image_sizes, embodiment_type, **kwargs)
        camera_order = list(self._image_sizes)
        if len(camera_order) != 3:
            raise ValueError(
                "MEM-Lite coordination requires exactly official 3-camera order "
                "[head_rgb, left_wrist_rgb, right_wrist_rgb]"
            )
        expected = ["head_rgb", "left_wrist_rgb", "right_wrist_rgb"]
        if camera_order != expected:
            raise ValueError(f"MEM-Lite camera order must be {expected}, got {camera_order}")
        if num_input_images % len(camera_order):
            raise ValueError("num_input_images must be an integer number of three-camera observations")
        frames = num_input_images // len(camera_order)
        self._memlite_camera_order = tuple(camera_order)
        self._memlite_observation_steps = frames
        # BaseSamplesBuilder.build maps each image marker through _image_keys.
        # Override that map to [head_t..., left_t..., right_t...] to agree with
        # the model's pixel dict concatenate order.
        self._image_keys = [camera for camera in camera_order for _ in range(frames)]


class SkillFMActionBuilder(_CameraMajorHistoryBuilder):
    """Schema-v6 low controller inputs for the real continuous-FM route.

    A data-side validated semantic skill bundle is injected *before* EOC with
    the ``_!`` input mask.  The annotation audit bundle is never accepted by
    this class.  The template deliberately has no action-token placeholder:
    the normalized continuous action tensor remains the sole action
    supervision owned by :class:`FMHelper`.
    """

    required_fields = (
        "schema_version", "memlite_branch", "active_skills_semantic_json",
        "active_skills_text", "parent_goal", "next_decision", "task_complete",
        "low_action_supervision_mask",
    )
    eval_required_fields = required_fields

    @property
    def template(self) -> str:
        return (
            "<chat_user_prefix>" + self._images + "<bos>"
            "Embodiment: <embodiment_text_!>; Task: <command_text_!_200>; "
            # ``active_skills_text`` is the model-sample key.  Template
            # syntax reserves the final ``_text`` for the text processor, so
            # keeping the key's own suffix requires the explicit double form.
            "Parent goal: <parent_goal_text_!>; <active_skills_text_text_!> State: <proprio_proprio_!>;"
            "<chat_user_suffix><chat_assistant_prefix>"
            "Action: <EOV><EOC><eos>"
        )

    @staticmethod
    def _validated_label(data: Dict[str, Any]) -> Dict[str, Any]:
        label = _model_safe_v6_label(data)
        if label["memlite_branch"] != "low":
            raise MemLiteSkillProtocolError("SkillFMActionBuilder accepts low schema-v6 rows only")
        if label["task_complete"] or label["next_decision"] != "EXECUTE":
            raise MemLiteSkillProtocolError("only nonterminal EXECUTE rows may dispatch a low FM bundle")
        if any(skill["verb"] == "SKILL_UNKNOWN" for skill in label["active_skills"]):
            raise MemLiteSkillProtocolError(
                "unknown active skill is not valid low-level BC supervision; reject it explicitly"
            )
        if not label.get("low_action_supervision_mask", False):
            raise MemLiteSkillProtocolError("low_action_supervision_mask=false rows must not reach FM")
        return label

    def can_handle(self, data: Dict[str, Any]) -> bool:
        try:
            self._validated_label(data)
        except (MemLiteSkillProtocolError, TypeError, ValueError):
            return False
        return True

    def can_handle_for_eval(self, data: Dict[str, Any]) -> bool:
        return self.can_handle(data)

    def _populate_extra_samples(self, data: Dict[str, Any], samples: Dict[str, Any]) -> None:
        label = self._validated_label(data)
        samples.update(
            memlite_branch="low",
            schema_version=MEMLITE_SKILL_SCHEMA_VERSION,
            # The old prefix name is retained only as an explicit assertion
            # carrier; v6 source of truth is schema_version.
            memlite_schema_version=MEMLITE_SKILL_SCHEMA_VERSION,
            # The model never receives the audit bundle / IDs.  Its only
            # deployed condition is the shared semantic projection.
            active_skills_semantic_json=label["active_skills_semantic_json"],
            active_skills_text=label["active_skills_text"],
            parent_goal=label["parent_goal"],
            next_decision=label["next_decision"],
            task_complete=bool(label["task_complete"]),
            low_action_supervision_mask=bool(label["low_action_supervision_mask"]),
        )


class PlannerOutcomeBuilder(_CameraMajorHistoryBuilder):
    """Causal schema-v6 planner with an explicitly optional result target.

    A missing physical result has ``outcome_supervision_mask=false``.  The
    output grammar remains fixed, but PlannerOutcomePolicy masks that exact
    field's AR labels (as well as its classifier loss).  Thus neither loss
    learns unavailable evidence as a default ``UNKNOWN`` result, while serving
    can always parse a fixed five-field planner response.
    """

    required_fields = (
        "schema_version", "memlite_branch", "task_name", "parent_goal", "memory",
        "active_skills_semantic_json", "active_skills_text", "target_parent_goal",
        "next_decision", "task_complete", "outcome_target", "outcome_supervision_mask",
    )
    eval_required_fields = required_fields

    @property
    def template(self) -> str:
        return (
            "<chat_user_prefix>" + self._images + "<bos>"
            "Embodiment: <embodiment_text_!>; Task: <command_text_!_200>; "
            "Task goal: <task_name_text_!>; Previous parent goal: <previous_parent_goal_text_!>; Previous bundle: <previous_intent_text_!>; "
            "Memory: <memory_text_!>; Known result: <known_previous_outcome_text_!>; "
            "Execution feedback: <execution_feedback_text_!>; State: <proprio_proprio_!>;"
            "<chat_user_suffix><chat_assistant_prefix>"
            "<planner_prompt_text_!><EOC><outcome_target_text>|<next_decision_text>|"
            "<current_parent_goal_text>|<active_skills_semantic_json_text>|<memory_update_text>|<task_complete_text>|<HL_END><EOV>"
        )

    @staticmethod
    def _validated_label(data: Dict[str, Any]) -> Dict[str, Any]:
        label = _model_safe_v6_label(data)
        if label["memlite_branch"] != "high":
            raise MemLiteSkillProtocolError("PlannerOutcomeBuilder accepts high schema-v6 rows only")
        return label

    def can_handle(self, data: Dict[str, Any]) -> bool:
        try:
            self._validated_label(data)
        except (MemLiteSkillProtocolError, TypeError, ValueError):
            return False
        return self._check_fields(data, self.required_fields)

    def can_handle_for_eval(self, data: Dict[str, Any]) -> bool:
        # The planner must receive every EOC-prefix field at serving time; its
        # EOC-after targets are generated and therefore intentionally omitted.
        return self.can_handle(data)

    def build(self, data: Dict[str, Any], sample: Dict[str, Any]) -> Dict[str, Any]:
        """Reject a sidecar/task-table mismatch before either becomes VLM text."""
        planner_input = planner_input_projection(self._validated_label(data))
        source_task = str(sample.get("_instructions", "")).strip()
        if not source_task or source_task != planner_input["task_name"]:
            raise ValueError(
                "PlannerOutcome requires sample _instructions to exactly equal schema-v6 task_name; "
                "refusing task index/sidecar text substitution"
            )
        return super().build(data, sample)

    def _populate_extra_samples(self, data: Dict[str, Any], samples: Dict[str, Any]) -> None:
        label = self._validated_label(data)
        planner_input = planner_input_projection(label)
        outcome_is_supervised = bool(label["outcome_supervision_mask"])
        outcome_value = str(label["outcome_target"])
        samples.update(
            memlite_branch="high",
            schema_version=MEMLITE_SKILL_SCHEMA_VERSION,
            memlite_schema_version=MEMLITE_SKILL_SCHEMA_VERSION,
            memory=planner_input["memory"],
            task_name=planner_input["task_name"],
            previous_intent=planner_input["previous_intent"],
            # At deployment this is saved output from the *prior* planner turn.
            # The current/target parent goal never appears before EOC.
            previous_parent_goal=planner_input["previous_parent_goal"],
            known_previous_outcome=(
                "Known previous outcome: " + planner_input["known_previous_outcome"]
            ),
            execution_feedback=planner_input["execution_feedback"],
            # The fixed grammar is retained for serving.  The policy masks this
            # field's token labels below when no physical evidence exists.
            outcome_target=f"Previous outcome: {outcome_value}",
            outcome_target_value=outcome_value,
            outcome_supervision_mask=outcome_is_supervised,
            next_decision=f"Decision: {label['next_decision']}",
            current_parent_goal=(
                f"Parent goal: {label['target_parent_goal']}"
            ),
            current_parent_goal_value=str(label["target_parent_goal"]),
            current_parent_goal_supervision_mask=bool(label["parent_goal_supervision_mask"]),
            active_skills_semantic_json=(
                "Active skills: " + label["active_skills_semantic_json"]
            ),
            active_skills_semantic_json_value=label["active_skills_semantic_json"],
            memory_update=f"Memory update: {label['memory_update']}",
            task_complete=f"Task complete: {str(bool(label['task_complete'])).lower()}",
            # InputPreprocessor turns this into a fail-fast length check: planner
            # context/parallel skills/memory may never be silently truncated.
            memlite_causal_prompt=True,
            planner_prompt=(
                "First report the previous outcome only when it is evidenced; then output "
                "the decision, complete active-skills JSON bundle, memory update, and task completion."
            ),
        )

class Stage1Processor(GalaxeaCoTProcessor):
    """A raw tensor record and a model-safe label are separate trust domains."""

    def preprocess(self, data):
        projection = data.pop("model_projection")
        _model_safe_v6_label(projection)
        sample = self._process_tensors(data)
        sample["samples"] = self.samples_builder.build(projection, sample)
        from g05.utils.memlite_planner_fields import bind_planner_rendered_fields
        bind_planner_rendered_fields(sample["samples"], projection)
        sample["samples"].update(projection)
        return sample
