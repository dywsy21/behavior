import hashlib
import json
from pathlib import Path
import sys

import numpy as np


SOURCE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE))

from chunk_trace import AlignmentTrace


def observation(control_step=0):
    return {
        "task": "turning on radio",
        "control_step": control_step,
        "images": {
            "head_rgb": np.full((3, 12, 12), 10, dtype=np.uint8),
            "left_wrist_rgb": np.full((3, 8, 8), 20, dtype=np.uint8),
            "right_wrist_rgb": np.full((3, 8, 8), 30, dtype=np.uint8),
        },
        "state": {"base_qvel": np.array([0.0, 1.0, 2.0], dtype=np.float32)},
    }


def high_projection():
    return {
        "task_name": "turning on radio",
        "previous_parent_goal": "none",
        "previous_intent": "None",
        "memory": '{"issued_command_history":[]}',
        "known_previous_outcome": "UNKNOWN",
        "execution_feedback": "none",
    }


def low_projection():
    return {
        "parent_goal": "Approach the radio",
        "active_skills_text": "Active skills: [verb=NAVIGATE].",
    }


def test_trace_isolated_per_instance_and_chunk(tmp_path):
    trace = AlignmentTrace(tmp_path / "trace")
    metadata = [
        {"task": "turning_on_radio", "split": "public_test", "instance_id": instance}
        for instance in (301, 302, 303)
    ]
    trace.begin(task="turning_on_radio", episode_metadata=metadata, session_generation=1)
    trace.register_high(
        env=0,
        chunk_id=0,
        observation=observation(),
        projection=high_projection(),
        sample={"template": "HIGH <EOC>", "planner_prompt": "predict planner event"},
        raw_text="UNKNOWN|EXECUTE|Approach the radio|[]|memory|false",
        event={"decision": "EXECUTE"},
        memory_after="memory",
        context_id="context-0",
    )
    actions = np.zeros((16, 23), dtype=np.float32)
    trace.record_chunk(
        env=0,
        chunk_id=0,
        observation=observation(),
        low_projection=low_projection(),
        low_sample={"template": "LOW <EOC>", "parent_goal": "Approach the radio"},
        action_chunk=actions,
        inference_request=0,
        timing={"batch_size": 3},
        context_id="context-0",
    )
    trace.record_chunk(
        env=0,
        chunk_id=1,
        observation=observation(16),
        low_projection=low_projection(),
        low_sample={"template": "LOW <EOC>", "parent_goal": "Approach the radio"},
        action_chunk=actions,
        inference_request=1,
        timing={"batch_size": 3},
        context_id="context-0",
    )

    folder = tmp_path / "trace/turning_on_radio/301"
    rows = [json.loads(line) for line in (folder / "chunks.jsonl").read_text().splitlines()]
    assert [row["chunk_id"] for row in rows] == [0, 1]
    assert [row["high_level_called"] for row in rows] == [True, False]
    assert rows[0]["high_level"]["raw_output_text"].startswith("UNKNOWN|EXECUTE")
    assert rows[0]["low_level"]["prompt_text"].startswith("Task: turning on radio")
    assert np.asarray(rows[0]["action_chunk_raw23"]).shape == (16, 23)
    image = observation()["images"]["head_rgb"]
    assert rows[0]["observation"]["images"]["head_rgb"]["sha256"] == hashlib.sha256(image.tobytes()).hexdigest()
    assert (tmp_path / "trace" / rows[0]["observation"]["montage"]).is_file()
    assert not (tmp_path / "trace/turning_on_radio/302/chunks.jsonl").exists()
