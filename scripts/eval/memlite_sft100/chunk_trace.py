"""Read-only chunk alignment traces for high/low evaluation diagnosis.

The trace is deliberately outside the model inputs and action protocol.  It
records the exact server-side observation at each inference boundary together
with the planner transaction and low-level conditioning that produced the
following action chunk.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time

import numpy as np


TRACE_SCHEMA_VERSION = 1


def scalar_prompt_fields(sample):
    """Keep the exact scalar/template fields supplied to a model branch."""
    result = {}
    for key, value in sample.items():
        if value is None or isinstance(value, (str, bool, int, float)):
            result[key] = value
    return result


def high_prompt_text(observation, projection, sample_fields):
    return "\n".join([
        f"Task: {observation['task']}",
        f"Task goal: {projection['task_name']}",
        f"Previous parent goal: {projection['previous_parent_goal']}",
        f"Previous bundle: {projection['previous_intent']}",
        f"Memory: {projection['memory']}",
        f"Known result: {projection['known_previous_outcome']}",
        f"Execution feedback: {projection['execution_feedback']}",
        f"Planner instruction: {sample_fields.get('planner_prompt', '')}",
    ])


def low_prompt_text(observation, projection):
    return "\n".join([
        f"Task: {observation['task']}",
        f"Parent goal: {projection['parent_goal']}",
        projection['active_skills_text'],
    ])


def _safe_component(value):
    text = str(value)
    if not text or text in {".", ".."} or any(ch in text for ch in "/\\\x00"):
        raise ValueError(f"Unsafe trace path component: {text!r}")
    return text


def _image_metadata(images):
    result = {}
    for name, value in images.items():
        array = np.ascontiguousarray(value)
        result[name] = {
            "shape": list(array.shape),
            "dtype": str(array.dtype),
            "sha256": hashlib.sha256(array.tobytes()).hexdigest(),
        }
    return result


def _save_montage(images, path):
    from PIL import Image, ImageDraw, ImageFont

    expected = {"head_rgb", "left_wrist_rgb", "right_wrist_rgb"}
    if set(images) != expected:
        raise ValueError(f"Expected three RGB cameras, got {sorted(images)}")

    def rgb(name):
        array = np.asarray(images[name])
        if array.ndim != 3 or array.shape[0] != 3 or array.dtype != np.uint8:
            raise ValueError(f"{name} is not uint8 CHW RGB")
        return Image.fromarray(array.transpose(1, 2, 0))

    canvas = Image.new("RGB", (1080, 720), "black")
    canvas.paste(rgb("head_rgb").resize((720, 720), Image.Resampling.LANCZOS), (0, 0))
    canvas.paste(rgb("left_wrist_rgb").resize((360, 360), Image.Resampling.LANCZOS), (720, 0))
    canvas.paste(rgb("right_wrist_rgb").resize((360, 360), Image.Resampling.LANCZOS), (720, 360))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    for label, xy in (("head", (8, 8)), ("left wrist", (728, 8)), ("right wrist", (728, 368))):
        box = draw.textbbox(xy, label, font=font)
        draw.rectangle((box[0] - 3, box[1] - 2, box[2] + 3, box[3] + 2), fill="black")
        draw.text(xy, label, fill="white", font=font)
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, format="JPEG", quality=92, subsampling=0)


class AlignmentTrace:
    """One isolated JSONL and observation-keyframe stream per environment."""

    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.identities = []
        self.active_high = {}
        self.session_generation = None

    def begin(self, *, task, episode_metadata, session_generation):
        self.session_generation = int(session_generation)
        self.identities = []
        self.active_high = {}
        for env, metadata in enumerate(episode_metadata):
            canonical_task = _safe_component(metadata["task"])
            if canonical_task != task:
                raise ValueError("Trace task differs from episode identity")
            instance_id = int(metadata["instance_id"])
            folder = self.root / canonical_task / str(instance_id)
            folder.mkdir(parents=True, exist_ok=False)
            (folder / "observations").mkdir()
            identity = {
                "schema_version": TRACE_SCHEMA_VERSION,
                "task": canonical_task,
                "instance_id": instance_id,
                "env": env,
                "session_generation": self.session_generation,
                "split": metadata["split"],
            }
            (folder / "identity.json").write_text(json.dumps(identity, indent=2) + "\n")
            self.identities.append((identity, folder))

    def register_high(self, *, env, chunk_id, observation, projection, sample, raw_text,
                      event, memory_after, context_id):
        fields = scalar_prompt_fields(sample)
        self.active_high[env] = {
            "high_call_id": f"g{self.session_generation}-e{env}-c{chunk_id}",
            "planned_at_chunk": int(chunk_id),
            "context_id": context_id,
            "prompt_text": high_prompt_text(observation, projection, fields),
            "model_sample_fields": fields,
            "input_projection": projection,
            "memory_before": projection["memory"],
            "raw_output_text": raw_text,
            "parsed_event": event,
            "memory_after": memory_after,
        }

    def record_chunk(self, *, env, chunk_id, observation, low_projection, low_sample,
                     action_chunk, inference_request, timing, context_id):
        if env not in self.active_high:
            raise RuntimeError("Chunk trace has no active high-level context")
        identity, folder = self.identities[env]
        observation_path = folder / "observations" / f"chunk_{chunk_id:04d}.jpg"
        _save_montage(observation["images"], observation_path)
        fields = scalar_prompt_fields(low_sample)
        state = {key: np.asarray(value, dtype=np.float32).tolist()
                 for key, value in observation["state"].items()}
        high = self.active_high[env]
        row = {
            **identity,
            "recorded_at": time.time(),
            "chunk_id": int(chunk_id),
            "control_step": int(observation["control_step"]),
            "nominal_frame_start": int(observation["control_step"]),
            "nominal_frame_end_exclusive": int(observation["control_step"]) + len(action_chunk),
            "inference_request": int(inference_request),
            "inference_timing": timing,
            "observation": {
                "montage": str(observation_path.relative_to(self.root)),
                "images": _image_metadata(observation["images"]),
                "proprio": state,
            },
            "high_level_called": high["planned_at_chunk"] == int(chunk_id),
            "high_level": high,
            "low_level": {
                "context_id": context_id,
                "prompt_text": low_prompt_text(observation, low_projection),
                "model_sample_fields": fields,
                "input_projection": low_projection,
            },
            "action_chunk_raw23": np.asarray(action_chunk, dtype=np.float32).tolist(),
        }
        with (folder / "chunks.jsonl").open("a") as stream:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
            stream.flush()
