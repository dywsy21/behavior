from types import SimpleNamespace

import torch

from g05.data.base_lerobot_dataset import BaseLerobotDataset
from g05.data.base_lerobot_datasetV3 import (
    _build_global_stride_anchors,
    _build_stats_sampling_plan,
    _compute_frame_ep_end,
    _compute_frame_ep_start,
    _frame_indices_in_ranges,
)


class _EpisodeColumns:
    column_names = ["task_index"]

    def __init__(self, task_indices):
        self._task_indices = list(task_indices)

    def __getitem__(self, key):
        if key != "task_index":
            raise KeyError(key)
        return self._task_indices


def _task_meta(task_indices):
    return SimpleNamespace(
        repo_id="behavior",
        total_episodes=len(task_indices),
        episodes=_EpisodeColumns(task_indices),
    )


def _episode_index(num_episodes, frames_per_episode):
    starts = torch.arange(num_episodes, dtype=torch.long) * frames_per_episode
    return {"from": starts, "to": starts + frames_per_episode}


def _v3_selection_dataset(full_index, active_episode_indices=None, start_idx=0, end_idx=None):
    dataset = BaseLerobotDataset.__new__(BaseLerobotDataset)
    dataset.lerobot_ds_version = "3.0"
    dataset._start_idx = start_idx
    dataset._end_idx = (
        int(full_index["to"][-1]) if end_idx is None else int(end_idx)
    )
    dataset.episode_data_index = full_index
    if active_episode_indices is not None:
        dataset._full_episode_data_index = full_index
        dataset._active_episode_indices = tuple(active_episode_indices)
        # Deliberately wrong for physical sampling: task-stratified Dataset views
        # use this compressed index for __getitem__, not normalization rows.
        dataset.episode_data_index = _episode_index(len(active_episode_indices), 1)
    return dataset


def test_task_stratified_stats_selection_uses_global_ranges_and_never_hits_eval_rows():
    meta = _task_meta([task for task in range(5) for _ in range(200)])
    train_ids, _ = BaseLerobotDataset._select_task_stratified_episodes(
        [meta], val_set_proportion=0.05, is_training_set=True
    )
    eval_ids, _ = BaseLerobotDataset._select_task_stratified_episodes(
        [meta], val_set_proportion=0.05, is_training_set=False
    )
    full_index = _episode_index(1000, 100)
    dataset = _v3_selection_dataset(full_index, active_episode_indices=train_ids)

    selected_ids, selected_ranges = dataset._get_stats_episode_selection()
    eval_ranges = tuple(
        (int(full_index["from"][idx]), int(full_index["to"][idx])) for idx in eval_ids
    )

    assert len(selected_ids) == 950
    assert selected_ids == tuple(train_ids)
    assert selected_ranges[190] == (20_000, 20_100)
    # A compressed active view would put episode 200 near 19,000 instead.
    assert selected_ranges[190] != (
        int(dataset.episode_data_index["from"][190]),
        int(dataset.episode_data_index["to"][190]),
    )

    anchors = _build_global_stride_anchors(selected_ranges, downsample_rate=127)
    required, _, _ = _build_stats_sampling_plan(
        anchors,
        action_size=32,
        state_meta=[{"key": "state", "time_offset": -250}],
        action_meta=[{"key": "action", "time_offset": 250}],
        ep_starts=full_index["from"],
        ep_ends=full_index["to"],
    )

    assert _frame_indices_in_ranges(anchors, selected_ranges).all()
    assert _frame_indices_in_ranges(required, selected_ranges).all()
    assert not _frame_indices_in_ranges(anchors, eval_ranges).any()
    assert not _frame_indices_in_ranges(required, eval_ranges).any()


def test_episode_stride_anchors_are_globally_aligned_and_represent_short_episodes():
    anchors = _build_global_stride_anchors(((5, 6), (11, 37), (100, 121)), downsample_rate=10)

    # [5, 6) has no stride-aligned row, so it still contributes its first row.
    assert anchors.tolist() == [5, 20, 30, 100, 110, 120]


def test_v3_contiguous_and_no_val_stats_selection_use_physical_episode_ranges():
    full_index = {"from": torch.tensor([0, 10, 25]), "to": torch.tensor([10, 25, 40])}

    train = _v3_selection_dataset(full_index, start_idx=0, end_idx=25)
    assert train._get_stats_episode_selection() == ((0, 1), ((0, 10), (10, 25)))

    eval_dataset = _v3_selection_dataset(full_index, start_idx=10, end_idx=40)
    assert eval_dataset._get_stats_episode_selection() == ((1, 2), ((10, 25), (25, 40)))

    no_val = _v3_selection_dataset(full_index)
    assert no_val._get_stats_episode_selection() == ((0, 1, 2), ((0, 10), (10, 25), (25, 40)))


def test_v3_episode_index_fallback_plan_remains_inside_selected_ranges():
    # This mirrors the old-format fallback, where only frame-level
    # episode_index is available instead of meta/episodes parquet metadata.
    episode_indices = torch.tensor([0] * 5 + [1] * 6 + [2] * 4 + [3] * 7)
    frame_starts = _compute_frame_ep_start(episode_indices, len(episode_indices))
    frame_ends = _compute_frame_ep_end(episode_indices, len(episode_indices))
    selected_ranges = ((0, 5), (11, 15))
    anchors = _build_global_stride_anchors(selected_ranges, downsample_rate=4)
    required, _, _ = _build_stats_sampling_plan(
        anchors,
        action_size=8,
        state_meta=[{"key": "state", "time_offset": -10}],
        action_meta=[{"key": "action", "time_offset": 10}],
        frame_ep_start=frame_starts,
        frame_ep_end=frame_ends,
    )

    assert _frame_indices_in_ranges(anchors, selected_ranges).all()
    assert _frame_indices_in_ranges(required, selected_ranges).all()
    assert not _frame_indices_in_ranges(required, ((5, 11), (15, 22))).any()


def test_base_fallback_uses_selected_global_v3_episodes_and_v2_local_episodes():
    full_index = _episode_index(4, 10)
    v3_dataset = _v3_selection_dataset(full_index, active_episode_indices=(0, 2))
    v3_dataset.multi_dataset = SimpleNamespace(num_episodes=4)
    v3_dataset.state_meta = [{"key": "state"}]
    v3_dataset.action_meta = [{"key": "action"}]
    v3_calls = []

    def get_v3_episode(episode_idx):
        v3_calls.append(episode_idx)
        values = torch.full((3, 1), float(episode_idx))
        return {
            "state": {"state": values.unsqueeze(1)},
            "action": {"action": values.unsqueeze(1)},
        }

    v3_dataset._get_episode_data = get_v3_episode

    class _IdentityProcessor:
        @staticmethod
        def action_state_transform(batch):
            return batch

    BaseLerobotDataset.get_dataset_stats(v3_dataset, _IdentityProcessor())
    assert set(v3_calls) == {0, 2}

    v2_dataset = BaseLerobotDataset.__new__(BaseLerobotDataset)
    v2_dataset.lerobot_ds_version = "2.1"
    v2_dataset.multi_dataset = SimpleNamespace(num_episodes=2)
    v2_dataset.episode_data_index = {"from": torch.tensor([100, 120]), "to": torch.tensor([120, 150])}
    assert v2_dataset._get_stats_episode_selection() == ((0, 1), ((100, 120), (120, 150)))
