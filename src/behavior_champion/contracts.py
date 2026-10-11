"""CPU-only task, stage and native-action contracts; no model execution.

Legacy source: behavior-1k-solution ca556f74, b1k_policy.py/config.py.
2026 proprio slices match the dataset's 61-D observation.state metadata.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from collections.abc import Mapping, Sequence

import numpy as np


LEGACY_STAGES = (
    5, 6, 15, 15, 14, 12, 9, 15, 10, 15,
    7, 13, 10, 15, 15, 15, 15, 11, 13, 12,
    14, 15, 9, 15, 15, 15, 15, 15, 15, 15,
    11, 10, 10, 13, 5, 5, 14, 6, 8, 10,
    5, 15, 8, 15, 12, 11, 9, 14, 15, 15,
)
# Native23: base3, trunk4, left7+gripper1, right7+gripper1.
# Trunk's fourth joint stays absolute, exactly as in the released policy.
DELTA_MASK = np.array([False]*3 + [True]*3 + [False] + [True]*7
                      + [False] + [True]*7 + [False], dtype=bool)


def canonical_task(name: str) -> str:
    if not isinstance(name, str) or not name.strip():
        raise ValueError('A nonempty public task name is required')
    return ' '.join(name.replace('_', ' ').lower().split())


@dataclass(frozen=True)
class TaskLayout:
    # Model rows retain legacy order, independently of official dataset IDs.
    names: tuple[str, ...]
    dataset_to_model: dict[int, int]
    num_stages: tuple[int, ...]

    @property
    def stage_offsets(self) -> tuple[int, ...]:
        return tuple(sum(self.num_stages[:i]) for i in range(len(self.names)))

    @property
    def total_stages(self) -> int:
        return sum(self.num_stages)


def make_task_layout(legacy_names: Sequence[str], current_names: Mapping[int, str],
                     new_task_mean_train_frames: Mapping[int, float]) -> TaskLayout:
    """Preserve every old row/bin; append new rows using TRAIN-only duration.

    The appended bins are a registered initial recipe, not success labels.
    Existing 50 tasks keep their exact original counts, even if durations differ.
    """
    legacy = tuple(canonical_task(x) for x in legacy_names)
    current = {int(k): canonical_task(v) for k, v in current_names.items()}
    if len(legacy) != 50 or len(set(legacy)) != 50:
        raise ValueError('Expected 50 unique legacy tasks')
    if len(current) != len(current_names) or len(set(current.values())) != len(current):
        raise ValueError('Duplicate official task IDs or canonical names')
    if set(current) != set(range(100)) or not set(legacy) <= set(current.values()):
        raise ValueError('2026 must contain exactly 100 tasks and all legacy names')
    names = list(legacy)
    stages = list(LEGACY_STAGES)
    for task, name in sorted(current.items()):
        if name in legacy:
            continue
        duration = float(new_task_mean_train_frames[task])
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError('Invalid TRAIN-only episode duration')
        names.append(name)
        stages.append(max(5, min(15, int(duration / 900))))
    return TaskLayout(tuple(names), {task: names.index(name) for task, name in current.items()}, tuple(stages))


def append_embeddings(old: np.ndarray, initialized: np.ndarray, legacy_rows: int) -> np.ndarray:
    """Keep pretrained rows bit-for-bit; use caller's explicit new-row init.

    This is only an array migration primitive, not a permissive checkpoint loader.
    The caller must still match every other checkpoint key and shape strictly.
    """
    old, initialized = np.asarray(old), np.asarray(initialized)
    if (old.ndim != 2 or initialized.ndim != 2 or old.shape[0] != legacy_rows
            or initialized.shape[0] <= legacy_rows or old.shape[1] != initialized.shape[1]
            or old.dtype != initialized.dtype):
        raise ValueError('Embedding migration shape/dtype mismatch')
    if not np.isfinite(old).all() or not np.isfinite(initialized).all():
        raise ValueError('Nonfinite embedding')
    result = initialized.copy()
    result[:legacy_rows] = old
    return result


def state23_from_proprio61(proprio: np.ndarray) -> np.ndarray:
    """Allowed 2026 observation -> exact native champion state ordering."""
    x = np.asarray(proprio)
    if x.ndim < 1 or x.shape[-1] != 61 or not np.isfinite(x).all():
        raise ValueError('Expected finite, explicitly contracted 61-D proprioception')
    return np.concatenate((x[..., 0:3], x[..., 53:57], x[..., 3:10],
                           2*x[..., 24:26].sum(-1, keepdims=True)/0.1 - 1,
                           x[..., 28:35],
                           2*x[..., 49:51].sum(-1, keepdims=True)/0.1 - 1), axis=-1)


def delta_actions(actions: np.ndarray, state23: np.ndarray, *, inverse: bool = False) -> np.ndarray:
    x, state = np.asarray(actions), np.asarray(state23)
    if (x.ndim < 2 or x.shape[-1] != 23 or state.shape != x.shape[:-2] + (23,)
            or not np.isfinite(x).all() or not np.isfinite(state).all()):
        raise ValueError('Native23 action chunk and matching current state required')
    anchor = np.expand_dims(np.where(DELTA_MASK, state, 0), -2)
    return x + anchor if inverse else x - anchor


def pad32(value: np.ndarray) -> np.ndarray:
    x = np.asarray(value)
    if x.ndim < 1 or x.shape[-1] != 23 or not np.isfinite(x).all():
        raise ValueError('Only finite native23 can be padded; G0.5 27-D is NOT compatible')
    return np.pad(x, [(0, 0)]*(x.ndim-1) + [(0, 9)])


def training_stage(frame: int, episode_length: int, num_stages: int) -> int:
    """A training-only duration bin. NEVER infer progress from episode length."""
    if not 0 <= frame < episode_length or not 5 <= num_stages <= 15:
        raise ValueError('Invalid episode frame or stage count')
    return min(num_stages - 1, frame*num_stages // episode_length)
