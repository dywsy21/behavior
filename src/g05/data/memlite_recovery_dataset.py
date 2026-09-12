"""Append audited, real simulator recovery windows to the original train view.

Original episode splits, indices, labels and normalization statistics are not
rewritten. A manifest hash is part of the sampling-view identity.
"""
from collections import OrderedDict
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from g05.utils.common.task_event_sampler import canonical_ranges


CAMERA_SUFFIXES = {"head_rgb": "zed_link:Camera:0::rgb",
    "left_wrist_rgb": "left_realsense_link:Camera:0::rgb",
    "right_wrist_rgb": "right_realsense_link:Camera:0::rgb"}


def recovery_raw_sample(entry, root, processor, trajectory, load_frame):
    step = int(entry["step"])
    obs_steps = int(processor.num_obs_steps)
    history = [max(0, step - 16*j) for j in reversed(range(obs_steps))]
    states, actions = trajectory["states"], trajectory["actions"]
    if step % 16 or step < 80 or step + 32 > len(actions):
        raise ValueError("Recovery windows need real 6x16 history and full 32-step targets")
    if obs_steps not in {1, 6} or int(getattr(processor, "action_horizon", 32)) != 32:
        raise ValueError("Recovery data cadence disagrees with model preprocessing")
    images = {}
    for camera, suffix in CAMERA_SUFFIXES.items():
        frames = []
        for t in history:
            raw = load_frame(root / f"obs_{t:05d}.npz")
            matches = [k for k in raw if k.endswith(suffix)]
            if len(matches) != 1:
                raise ValueError(f"Expected exactly one official camera {suffix}")
            value = raw[matches[0]]
            if value.dtype != np.uint8 or value.ndim != 3 or value.shape[-1] not in (3, 4):
                raise ValueError("Expected official uint8 HWC RGB(A)")
            # Match serving: drop alpha, no channel reversal or resize here.
            frames.append(value[..., :3].transpose(2, 0, 1).copy())
        images[camera] = torch.from_numpy(np.stack(frames))
    def slice_fields(values, metas):
        return {m["key"]: torch.from_numpy(values[:, int(m["start_index"]):
            int(m["start_index"]) + int(m["raw_shape"])].copy()).float() for m in metas}
    sample = {"images": images, "state": slice_fields(states[history], processor.shape_meta["state"]),
        "action": slice_fields(actions[step:step+32], processor.shape_meta["action"]),
        "action_is_pad": torch.full((32,), entry["memlite_branch"] == "high", dtype=torch.bool),
        "state_is_pad": torch.zeros(obs_steps, dtype=torch.bool),
        "image_is_pad": torch.zeros(obs_steps, dtype=torch.bool), "idx": int(entry["index"]),
        "task": entry["command"], "embodiment": "galaxea_r1pro", "frequency": 30.,
        "step_is_qualified": True, "memlite_ignore": False, "memlite_causal_prompt": True,
        "memlite_label_missing": False, "memlite_segment_end": len(actions),
        "memlite_action_horizon_end": step+32,
        "memlite_action_valid_steps": 32 if entry["memlite_branch"] == "low" else 0,
        "dataset_locator": f"recovery={root}, step={step}, branch={entry['memlite_branch']}"}
    for key in ("memlite_branch", "memory", "previous_intent", "intent", "memory_update", "intent_status", "execution_feedback"):
        sample[key] = entry[key]
    return sample


class RecoveryAugmentedDataset(torch.utils.data.Dataset):
    def __init__(self, base, manifest, processor, *, split="train"):
        self.base = base
        self.manifest_path = Path(manifest).resolve(strict=True)
        raw = self.manifest_path.read_bytes()
        data = json.loads(raw)
        if data.get("schema_version") != 1 or not data.get("ready_for_training"):
            raise ValueError("Recovery dataset must pass packaging and primary manual review")
        if split not in {"train", "eval"}:
            raise ValueError("Explicit recovery split required")
        if bool(base.is_training_set) != (split == "train"):
            raise ValueError("Recovery and original Dataset split disagree")
        self.entries = deepcopy(data["samples"][split])
        self.base_length = len(base)
        self.is_training_set = base.is_training_set
        self.is_rank_sharded = bool(getattr(base, "is_rank_sharded", False))
        self.use_weight_for_sampling = bool(getattr(base, "use_weight_for_sampling", False))
        if self.is_rank_sharded or self.use_weight_for_sampling:
            raise ValueError("Recovery append requires an unweighted, unsharded source view")
        self.recovery_identity = {"manifest": str(self.manifest_path),
                                  "sha256": hashlib.sha256(raw).hexdigest(), "split": split}
        self.datasets = base.datasets
        if any(child.action_size != 32 or child.obs_size not in {1, 6} or child.obs_stride != 16 for child in self.datasets):
            raise ValueError("Recovery history/action cadence must match the original training Dataset")
        self.processor = processor["galaxea_r1pro"]
        if any(child.obs_size != self.processor.num_obs_steps for child in self.datasets):
            raise ValueError("Recovery and original processor history lengths disagree")
        self._cache = OrderedDict()
        seen = set()
        for i, row in enumerate(self.entries):
            source = (row["trajectory"], int(row["step"]), row["memlite_branch"])
            if source in seen or row["memlite_branch"] not in {"high", "low"}:
                raise ValueError("Duplicate or invalid recovery source/branch")
            seen.add(source)
            if row.get("source_split") != split or row.get("mode") != "train":
                raise ValueError("Recovery split leak or non-training simulator source")
            row["index"] = self.base_length + i

    def __getattr__(self, name):
        if name in {"base", "_cache"}:
            raise AttributeError(name)
        return getattr(self.base, name)

    def __len__(self):
        return self.base_length + len(self.entries)

    def _load(self, path):
        key = str(path)
        if key in self._cache:
            value = self._cache.pop(key)
        else:
            with np.load(path, allow_pickle=False) as source:
                value = {k: source[k] for k in source.files}
        self._cache[key] = value
        while len(self._cache) > 20:
            self._cache.popitem(last=False)
        return value

    def __getitem__(self, index):
        index = int(index)
        if not 0 <= index < len(self):
            raise IndexError(index)
        if index < self.base_length:
            dataset_index, local_index = self.base._resolve_index(index)
            child = self.datasets[dataset_index]
            expected = int(child._map_active_index(local_index))
            sample = self.base[index]
            if int(sample["idx"]) != expected:
                raise RuntimeError("Original Dataset silently resampled a stratified source row; audit the failed source before training")
            sample["memlite_requested_index"] = index
            sample["memlite_source_kind"] = "original"
            sample.setdefault("execution_feedback", "none")
            return sample
        entry = self.entries[index - self.base_length]
        root = Path(entry["trajectory"])
        data = recovery_raw_sample(entry, root, self.processor, self._load(root / "trajectory.npz"), self._load)
        sample = self.processor.preprocess(data)
        sample["memlite_requested_index"] = index
        sample["memlite_source_kind"] = "recovery"
        sample["embodiment"] = sample["embodiment_type"] = "galaxea_r1pro"
        for key in ("action", "proprio"):
            if isinstance(sample["samples"].get(key), dict):
                sample["samples"][key]["embodiment"] = "galaxea_r1pro"
        return sample

    def get_memlite_branch_sampling_spec(self):
        original = deepcopy(self.base.get_memlite_branch_sampling_spec())
        high, low = list(original["high"]), list(original["low_ranges"])
        for row in self.entries:
            index = row["index"]
            if row["memlite_branch"] == "high":
                high.append(index)
            else:
                low.append((index, index+1))
        return {"high": high, "low_ranges": canonical_ranges(low, len(self))}
