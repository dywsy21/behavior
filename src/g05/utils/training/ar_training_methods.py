"""Explicit action-token targets and target-free prefixes for AR/KI experiments.

This module does not invent skills, outcomes or trajectories. It converts the
already validated, processed FM training sample and its SAME-STATE actions to
a token-supervised view. A separate inference view never contains action GT.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
import re

import torch

from .fm_training_methods import executed_prefix_codec_input


@dataclass(frozen=True)
class ActionTrainingSettings:
    route: str = "ar"
    conditioning: str = "skills"
    codec_mode: str = "original32"

    def __post_init__(self):
        if self.route not in {"ar", "joint", "ki"}:
            raise ValueError("route must be ar, joint or ki")
        if self.conditioning not in {"skills", "task", "native_task", "native_subtask_cot"}:
            raise ValueError("conditioning must be skills, task, native_task or native_subtask_cot")
        if self.codec_mode not in {"original32", "prefix16_holdpad32"}:
            raise ValueError("unsupported codec horizon convention")
        if self.route != "ar" and (self.conditioning != "skills" or self.codec_mode != "original32"):
            raise ValueError("initial joint/KI ablations preserve the skill prefix and original32 targets")

    def as_dict(self):
        return asdict(self)

    @property
    def uses_fm(self):
        return self.route != "ar"

    @property
    def predicts_cot(self):
        return self.conditioning == "native_subtask_cot"


SKILL_INPUT = "Parent goal: <parent_goal_text_!>; <active_skills_text_text_!> "
FM_ENDING = "<EOV><EOC><eos>"
AR_ENDING = "<EOV><EOC><action_action><eos>"
CANONICAL_PARTS = dict(left_control=9, left_gripper=1, right_control=9,
                       right_gripper=1, lower_body=7)


def action_token_loss_metrics(cache, begin, end):
    """Report action vs text/boundary losses without changing the objective.

    The AR helper already computed these detached, valid-target per-token
    losses. Separating them prevents longer/easier CoT text from masquerading
    as lower action loss. Boundary includes EOV, separators and EOS, not only
    the subtask content. No additional teacher or model forward is involved.
    """
    losses, targets = cache["token_loss"], cache["shift_labels_masked"]
    if (losses.ndim != 1 or targets.shape != losses.shape or not losses.numel()
            or not torch.isfinite(losses).all() or (targets < 0).any() or not begin < end):
        raise ValueError("Invalid valid-target CE cache")
    actions = (targets >= begin) & (targets < end)
    result = {}
    for name, selected in (("action_token", actions), ("text_boundary_token", ~actions)):
        count = int(selected.sum())
        if not count:
            raise ValueError("Declared action-token training needs action and boundary targets")
        result[name + "_targets"] = count
        result[name + "_ce"] = float(losses[selected].detach().float().mean())
    return result


def validate_complete_action_tokens(ids, tokenizer):
    """Reject partial levels/blocks that the legacy codec would zero-fill.

    This reads only static codec metadata and generated IDs, never action GT.
    Group order is allowed to vary, but each required residual/rule block must
    appear exactly once with the full number of real codebook tokens.
    """
    if (not isinstance(ids, torch.Tensor) or ids.ndim != 1
            or ids.dtype not in (torch.int16, torch.int32, torch.int64)):
        raise ValueError("expected one integer action-token sequence")
    serializer = tokenizer.serializer
    markers = serializer.group_marker_action_indices
    required = {}
    for level in range(serializer.num_residuals):
        for key in serializer.nn_key_names:
            marker = f"<{key}_{level}>" if serializer.max_residuals > 1 else f"<{key}>"
            required[markers[marker]] = serializer.code_len
    for key in serializer.rule_key_names:
        required[markers[f"<{key}>"]] = serializer.rule_tokens_per_key
    if not required or any(int(count) < 1 for count in required.values()):
        raise ValueError("codec has no valid complete-block layout")
    codebook_size = int(tokenizer._codebook_size)
    raw = (ids.detach().cpu() - int(tokenizer.action_token_begin_idx)).tolist()
    seen = set()
    cursor = 0
    while cursor < len(raw):
        marker = raw[cursor]
        if marker not in required or marker in seen:
            raise ValueError("unknown, duplicate or misplaced action-group marker")
        count = int(required[marker])
        codes = raw[cursor + 1:cursor + count + 1]
        if len(codes) != count or any(code < 0 or code >= codebook_size for code in codes):
            raise ValueError("truncated action block or group marker inside codebook payload")
        seen.add(marker)
        cursor += count + 1
    if seen != set(required):
        raise ValueError("missing required action group or residual level")
    return dict(token_count=len(raw), complete_blocks=len(seen))


def action_prefix_samples(samples, settings: ActionTrainingSettings):
    """Target-free view with an action-output slot; reject accidental GT ingress.

    The caller validates the v6 semantic sample BEFORE this representation
    transform. Here we guard its rendering boundary. Task-only is explicitly
    a new task-conditioned AR ablation, not an unchanged upstream CoT recipe.
    """
    if not samples:
        raise ValueError("empty action batch")
    if settings.predicts_cot:
        image_keys = [key for key in samples[0] if re.fullmatch(r"image\d+", key)]
        return native_subtask_cot_actor_samples(samples, num_images=len(image_keys))
    result = []
    for sample in samples:
        template = sample.get("template")
        if (not isinstance(template, str) or not template.endswith(FM_ENDING)
                or template.count("<EOC>") != 1 or template.count(SKILL_INPUT) != 1
                or "<action_action" in template or "action" in sample):
            raise ValueError("need the validated target-free SkillFM sample template")
        if any(key in sample for key in ("gt_action", "future_state", "teacher_action")):
            raise ValueError("future/teacher fields cannot enter the actor prefix")
        value = deepcopy(sample)
        if settings.conditioning in {"task", "native_task"}:
            template = template.replace(SKILL_INPUT, "", 1)
            for key in ("parent_goal", "active_skills_text", "active_skills_semantic_json"):
                value.pop(key, None)
        ending = AR_ENDING
        if settings.conditioning == "native_task":
            # Match the upstream BaseSamplesBuilder, including punctuation
            # and the action terminator. No fabricated CoT supervision.
            task_slot = "Task: <command_text_!_200>; "
            if template.count(task_slot) != 1:
                raise ValueError("native task view requires the exact source command slot")
            template = template.replace(task_slot, "Task: <command_text_!_200> ", 1)
            ending = "<EOV><EOC><action_action>|<eos>"
        value["template"] = template[:-len(FM_ENDING)] + ending
        result.append(value)
    return result


def action_training_samples(samples, actions, action_is_pad, action_dim_is_pad,
                            settings: ActionTrainingSettings):
    """Attach all real controls; codec tail padding is NOT future supervision."""
    if (not isinstance(actions, torch.Tensor) or actions.ndim != 3
            or actions.shape[1:] != (32, 27) or actions.shape[0] != len(samples)):
        raise ValueError("expected one 32x27 action tensor per original sample")
    if (not isinstance(action_dim_is_pad, torch.Tensor)
            or action_dim_is_pad.dtype != torch.bool
            or action_dim_is_pad.shape != (len(samples), 27)):
        raise ValueError("expected boolean 27D padding masks")
    if (not isinstance(action_is_pad, torch.Tensor) or action_is_pad.dtype != torch.bool
            or action_is_pad.shape != actions.shape[:2]):
        raise ValueError("expected explicit boolean temporal padding masks")
    expected = torch.zeros_like(action_dim_is_pad)
    expected[:, [7, 8, 17, 18]] = True
    if not torch.equal(action_dim_is_pad, expected):
        raise ValueError("R1Pro requires all 23 controls, with only four canonical padding dimensions")
    # Validate temporal masks even for the original codec. Do not silently
    # convert holes/all-padding into expert targets.
    padded = executed_prefix_codec_input(actions, action_is_pad=action_is_pad)
    codec_actions = actions if settings.codec_mode == "original32" else padded
    result = action_prefix_samples(samples, settings)
    for i, sample in enumerate(result):
        if settings.predicts_cot:
            # This is a representation of the already-reviewed same-state
            # skill bundle, not a newly inferred outcome or upstream atomic_task
            # annotation. All actor-visible fields were whitelisted above.
            from g05.data_processor.processor.samples_builder import validate_embedded_model_projection
            projection = validate_embedded_model_projection(samples[i])
            if (projection["memlite_branch"] != "low" or projection["task_complete"]
                    or projection["next_decision"] != "EXECUTE"
                    or any(s["verb"] == "SKILL_UNKNOWN" for s in projection["active_skills"])):
                raise ValueError("Subtask-CoT requires a valid same-state nonterminal skill target")
            sample["atomic_task"] = "Subtask: " + samples[i]["active_skills_text"]
        sample["action"] = dict(value=codec_actions[i].detach().clone(),
            action_dim_is_pad=action_dim_is_pad[i].detach().clone(),
            action_op_mask=(~action_dim_is_pad[i]).detach().clone(),
            parts_meta=dict(CANONICAL_PARTS))
    return result


def native_subtask_cot_actor_samples(samples, *, num_images):
    """Task+observations only; subtask text is generated, never supplied by GT.

    Use the exact upstream SubtaskCoTBuilder rendering. Existing skill labels
    are added only by action_training_samples on the supervised output side.
    """
    from g05.data_processor.processor.samples_builder import BaseSamplesBuilder, SubtaskCoTBuilder
    if type(num_images) is not int or num_images < 1 or not samples:
        raise ValueError("Subtask-CoT requires an explicit observed image layout")
    base = BaseSamplesBuilder(num_images, {"layout_only": (256, 256)}).template
    cot = SubtaskCoTBuilder(num_images, {"layout_only": (256, 256)}).template
    observed = []
    for sample in samples:
        if any(key in sample for key in ("atomic_task", "cot_target", "teacher_subtask")):
            raise ValueError("Subtask-CoT actor cannot receive a teacher subtask")
        row = deepcopy(sample)
        if row.get("template") == cot:
            if row.get("prompt") != "predict subtask":
                raise ValueError("Subtask-CoT actor requires the declared prompt")
            row["template"] = base
        observed.append(row)
    result = native_task_actor_samples(observed, num_images=num_images)
    for row in result:
        row.update(template=cot, prompt="predict subtask")
    return result


def native_task_actor_samples(samples, *, num_images):
    """Native action-only deployment requires no planner or semantic sidecar.

    Accept either the audited source view or an already-native Base template,
    then pass ONLY task/embodiment/current observations to the neural processor.
    Training still validates its same-state source separately before adding GT.
    """
    from g05.data_processor.processor.samples_builder import BaseSamplesBuilder
    if type(num_images) is not int or num_images < 1 or not samples:
        raise ValueError("native actor requires explicit image layout and nonempty samples")
    image_keys = {f"image{i}" for i in range(num_images)}
    expected = BaseSamplesBuilder(num_input_images=num_images, image_sizes={"layout_only": (256, 256)}).template
    allowed = {"template", "command", "embodiment", "proprio"} | image_keys
    result = []
    for sample in samples:
        if any(key in sample for key in ("action", "gt_action", "future_state", "teacher_action")):
            raise ValueError("native actor must not receive teacher or future fields")
        template = sample.get("template", "")
        row = (action_prefix_samples([sample], ActionTrainingSettings(conditioning="native_task"))[0]
               if isinstance(template, str) and SKILL_INPUT in template else deepcopy(sample))
        if (row.get("template") != expected or not allowed.issubset(row)
                or {key for key in row if re.fullmatch(r"image\d+", key)} != image_keys
                or any(not isinstance(row[key], str) or not row[key].strip() for key in ("command", "embodiment"))):
            raise ValueError("native actor requires the exact Base task template and complete observed inputs")
        result.append({key: row[key] for key in sorted(allowed)})
    return result
