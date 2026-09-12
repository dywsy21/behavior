import json
import multiprocessing as mp
from pathlib import Path

import torch
import pytest

from g05.data.memlite_sidecar import MemLiteSidecar
from g05.data.base_lerobot_dataset import BaseLerobotDataset
from g05.data_processor.processor.samples_builder import MEMLiteMixedBuilder
from g05.utils.data.data_utils import custom_collate_fn, collate_fn_pad_sequences


def _spawned_sidecar_lookup(sidecar):
    return sidecar.get(3, 0)["memlite_branch"]


def _write_sidecar(tmp_path: Path) -> Path:
    path = tmp_path / "labels.jsonl"
    rows = [
        {"episode_index": 3, "frame_index": 0, "memlite_branch": "high",
         "memory": "Task=t; Completed=none.", "intent": "pick cup",
         "memory_update": "Before: none; After: none", "intent_status": "CONTINUE",
         "segment_end": 2, "action_horizon_end": 2},
        {"episode_index": 3, "frame_index": 0, "memlite_branch": "low",
         "intent": "pick cup", "segment_end": 2, "action_horizon_end": 2},
        {"episode_index": 3, "frame_index": 1, "memlite_branch": "low",
         "intent": "pick cup", "segment_end": 2, "action_horizon_end": 2},
        {"episode_index": 3, "frame_index": 2, "memlite_branch": "low",
         "intent": "place cup", "segment_end": 4, "action_horizon_end": 4},
        {"episode_index": 3, "frame_index": 3, "memlite_branch": "low",
         "intent": "place cup", "segment_end": 4, "action_horizon_end": 4},
        {"episode_index": 4, "frame_index": 0, "memlite_branch": "low",
         "intent": "open drawer", "segment_end": 1, "action_horizon_end": 1},
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def _raw_sample():
    return {
        "episode_index": torch.tensor(3),
        "frame_index": torch.tensor(0),
    }


def test_sidecar_is_episode_indexed_and_boundary_prefers_high(tmp_path):
    sidecar = MemLiteSidecar(_write_sidecar(tmp_path), cache_episodes=1)
    assert sidecar.episode_count == 2
    row = sidecar.get(torch.tensor(3), torch.tensor(0))
    assert row["memlite_branch"] == "high"
    assert row["intent"] == "pick cup"
    assert sidecar.get(3, 99) is None
    # Accessing another episode evicts the first decoded chunk, while the
    # compact offset index remains available for a subsequent lookup.
    assert sidecar.get(4, 0)["intent"] == "open drawer"
    assert sidecar.get(3, 2)["intent"] == "place cup"


def test_sampler_index_deduplicates_duplicate_high_boundary_rows(tmp_path):
    path = tmp_path / "labels.jsonl"
    rows = [
        {"episode_index": 0, "frame_index": 0, "memlite_branch": "high", "intent": "a", "memory": "m", "memory_update": "m"},
        {"episode_index": 0, "frame_index": 0, "memlite_branch": "high", "intent": "a", "memory": "m", "memory_update": "m"},
        {"episode_index": 0, "frame_index": 0, "memlite_branch": "low", "intent": "a"},
        {"episode_index": 0, "frame_index": 1, "memlite_branch": "low", "intent": "a"},
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    spec = MemLiteSidecar(path).branch_sampling_index()
    assert spec["high"] == [(0, 0)]
    assert spec["low_ranges"] == [(0, 1, 2)]


def test_parquet_sidecar_uses_row_group_index_and_range_sampler_index(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    jsonl = _write_sidecar(tmp_path)
    rows = [json.loads(line) for line in jsonl.read_text(encoding="utf-8").splitlines()]
    parquet = tmp_path / "memlite_annotations.parquet"
    pq.write_table(pa.Table.from_pylist(rows), parquet, row_group_size=2)
    sidecar = MemLiteSidecar(parquet, cache_episodes=1)
    assert sidecar.get(3, 0)["memlite_branch"] == "high"
    # Low frame 0 is removed because its duplicate high row is authoritative.
    assert sidecar.branch_sampling_index()["low_ranges"] == [(3, 1, 4), (4, 0, 1)]


def test_parquet_sidecar_reopens_after_spawn_pickle(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    jsonl = _write_sidecar(tmp_path)
    rows = [json.loads(line) for line in jsonl.read_text(encoding="utf-8").splitlines()]
    parquet = tmp_path / "memlite_annotations.parquet"
    pq.write_table(pa.Table.from_pylist(rows), parquet, row_group_size=2)
    sidecar = MemLiteSidecar(parquet, cache_episodes=1)
    # Constructed parquet readers are deliberately omitted from pickle state;
    # the spawn worker must lazily reopen from the compact episode index.
    with mp.get_context("spawn").Pool(1) as pool:
        assert pool.apply(_spawned_sidecar_lookup, (sidecar,)) == "high"


def test_null_segment_end_does_not_break_overlay_or_pad_everything():
    class _NullBoundarySidecar:
        @staticmethod
        def get(_episode, _frame):
            return {
                "memlite_branch": "low",
                "memory": "",
                "intent": "pick cup",
                "memory_update": "",
                "intent_status": "CONTINUE",
                "ignore": False,
                "segment_end": None,
                "action_horizon_end": None,
            }

    obj = object.__new__(BaseLerobotDataset)
    obj._memlite_sidecar = _NullBoundarySidecar()
    obj.memlite_sidecar_required = True
    sample = {
        "action": {"arm": torch.arange(8, dtype=torch.float32).reshape(4, 2)},
        "action_is_pad": torch.zeros(4, dtype=torch.bool),
    }
    obj._overlay_memlite(sample, _raw_sample())
    assert sample["memlite_segment_end"] == -1
    assert sample["memlite_action_horizon_end"] == -1
    assert sample["memlite_action_valid_steps"] == 4
    assert not sample["action_is_pad"].any()


def test_overlay_marks_missing_and_pads_action_at_skill_boundary(tmp_path):
    obj = object.__new__(BaseLerobotDataset)
    obj._memlite_sidecar = MemLiteSidecar(_write_sidecar(tmp_path))
    obj.memlite_sidecar_required = True
    sample = {
        "action": {"arm": torch.arange(8, dtype=torch.float32).reshape(4, 2)},
        "action_is_pad": torch.zeros(4, dtype=torch.bool),
    }
    obj._overlay_memlite(sample, _raw_sample())
    assert sample["memlite_branch"] == "high"
    assert sample["memlite_action_valid_steps"] == 2
    assert sample["action_is_pad"].tolist() == [False, False, True, True]
    assert torch.equal(sample["action"]["arm"][2], sample["action"]["arm"][1])

    missing = {"action": {}, "action_is_pad": torch.zeros(1, dtype=torch.bool)}
    obj._overlay_memlite(missing, {"episode_index": torch.tensor(3), "frame_index": torch.tensor(99)})
    assert missing["memlite_ignore"] is True
    assert missing["memlite_label_missing"] is True
    assert missing["step_is_qualified"] is False


def test_base_dataset_getitem_overlay_end_to_end(tmp_path):
    class _Multi:
        def __getitem__(self, index):
            return {"episode_index": torch.tensor(3), "frame_index": torch.tensor(0), "task": "demo"}

    obj = object.__new__(BaseLerobotDataset)
    obj._memlite_sidecar = MemLiteSidecar(_write_sidecar(tmp_path))
    obj.memlite_sidecar_required = True
    obj._start_idx, obj._end_idx = 0, 1
    obj.multi_dataset = _Multi()
    obj.is_training_set = False
    obj.model_fps = 30
    obj.state_meta = []
    obj.action_meta = [{"key": "arm"}]
    obj.image_meta = None
    obj._dummy_image_meta = []
    obj.processor = None
    obj._get_action = lambda meta, raw: torch.arange(8, dtype=torch.float32).reshape(4, 2)
    obj._get_pad_mask = lambda meta, raw: torch.zeros(4, dtype=torch.bool)
    obj._map_active_index = lambda idx: idx
    obj._locate_sample = lambda idx: f"fake:{idx}"
    sample = obj[0]
    assert sample["memlite_branch"] == "high"
    assert sample["memlite_action_valid_steps"] == 2
    assert sample["action_is_pad"].tolist() == [False, False, True, True]


def test_branch_index_maps_active_episode_view_and_deduplicates_boundary(tmp_path):
    obj = object.__new__(BaseLerobotDataset)
    obj._memlite_sidecar = MemLiteSidecar(_write_sidecar(tmp_path))
    # Sidecar episode 3 is mapped to physical frames [10, 14); episode 4 is
    # outside this task-stratified active view.  Frame 0 has both labels but
    # the high planner row must win exactly as get() does.
    obj._full_episode_data_index = {
        "from": torch.tensor([0, 4, 7, 10, 14, 18]),
        "to": torch.tensor([4, 7, 10, 14, 18, 20]),
    }
    obj._active_episode_indices = (3,)
    obj.episode_data_index = {"from": torch.tensor([0]), "to": torch.tensor([4])}
    obj._start_idx, obj._end_idx = 0, 4
    spec = obj.get_memlite_branch_sampling_spec()
    assert spec["high"] == [0]
    assert spec["low_ranges"] == [(1, 4)]


def test_overlay_fields_reach_mixed_builder_and_collate(tmp_path):
    obj = object.__new__(BaseLerobotDataset)
    obj._memlite_sidecar = MemLiteSidecar(_write_sidecar(tmp_path))
    obj.memlite_sidecar_required = True
    high = {"action": {}, "action_is_pad": torch.zeros(1, dtype=torch.bool)}
    obj._overlay_memlite(high, _raw_sample())
    low = {"action": {"arm": torch.zeros(2, 2)}, "action_is_pad": torch.zeros(2, dtype=torch.bool)}
    obj._overlay_memlite(low, {"episode_index": torch.tensor(3), "frame_index": torch.tensor(2)})

    builder = MEMLiteMixedBuilder(num_input_images=1, image_sizes={"cam": (32, 32)})

    def build(data):
        sample = {
            "_instructions": "task",
            "_vlm_action": None,
            "proprio": torch.zeros(1, 2),
            "proprio_dim_is_pad": torch.zeros(2, dtype=torch.bool),
            "action": torch.zeros(2, 2),
            "action_dim_is_pad": torch.zeros(2, dtype=torch.bool),
            "action_op_mask": None,
            "action_parts_meta": None,
        }
        return builder.build(dict(data), sample)

    high_out = build(high)
    low_out = build(low)
    batch = custom_collate_fn([high_out, low_out])
    assert high_out["memlite_branch"] == "high"
    assert low_out["memlite_branch"] == "low"
    assert batch["memlite_branch"] == ["high", "low"]

    # This is the production training collate, not the permissive helper
    # above.  Overlay metadata must exist with stable scalar types on both
    # branches so default_collate cannot fail on a boundary-only field.
    wrapped = []
    for idx, (branch_data, builder_out) in enumerate(((high, high_out), (low, low_out))):
        wrapped.append(
            {
                "samples": builder_out,
                "idx": idx,
                "action": torch.zeros(2, 2),
                "action_is_pad": torch.zeros(2, dtype=torch.bool),
                "memlite_branch": branch_data["memlite_branch"],
                "memlite_ignore": branch_data["memlite_ignore"],
                "memlite_label_missing": branch_data["memlite_label_missing"],
                "memlite_segment_end": branch_data["memlite_segment_end"],
                "memlite_action_horizon_end": branch_data["memlite_action_horizon_end"],
                "memlite_action_valid_steps": branch_data["memlite_action_valid_steps"],
                "step_is_qualified": branch_data["step_is_qualified"],
            }
        )
    collated = collate_fn_pad_sequences(wrapped)
    assert collated["memlite_action_valid_steps"].shape == (2,)
    assert [sample["memlite_branch"] for sample in collated["samples"]] == ["high", "low"]


def test_mixed_builder_fails_before_build_when_branch_fields_are_missing():
    builder = MEMLiteMixedBuilder(num_input_images=1, image_sizes={"cam": (32, 32)})
    sample = {
        "_instructions": "task",
        "_vlm_action": None,
        "proprio": torch.zeros(1, 2),
        "proprio_dim_is_pad": torch.zeros(2, dtype=torch.bool),
        "action": torch.zeros(2, 2),
        "action_dim_is_pad": torch.zeros(2, dtype=torch.bool),
    }
    with pytest.raises(ValueError, match="cannot be built"):
        builder.build({"memlite_branch": "low"}, sample)
