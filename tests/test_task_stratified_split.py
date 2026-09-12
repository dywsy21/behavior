from types import SimpleNamespace

import torch

from g05.data.base_lerobot_dataset import BaseLerobotDataset


class _EpisodeColumns:
    column_names = ["task_index"]

    def __init__(self, task_indices):
        self._task_indices = list(task_indices)

    def __getitem__(self, key):
        if key != "task_index":
            raise KeyError(key)
        return self._task_indices


def _meta(task_indices, repo_id="dataset"):
    return SimpleNamespace(
        repo_id=repo_id,
        total_episodes=len(task_indices),
        episodes=_EpisodeColumns(task_indices),
    )


def test_task_stratified_split_is_balanced_and_disjoint():
    meta = _meta([0] * 200 + [1] * 200 + [2] * 200 + [3] * 200 + [4] * 200)

    train, train_counts = BaseLerobotDataset._select_task_stratified_episodes(
        [meta], val_set_proportion=0.05, is_training_set=True
    )
    val, val_counts = BaseLerobotDataset._select_task_stratified_episodes(
        [meta], val_set_proportion=0.05, is_training_set=False
    )

    assert train_counts == {str(i): 190 for i in range(5)}
    assert val_counts == {str(i): 10 for i in range(5)}
    assert len(train) == 950
    assert len(val) == 50
    assert set(train).isdisjoint(val)
    assert sorted(train + val) == list(range(1000))
    assert val == (
        list(range(190, 200))
        + list(range(390, 400))
        + list(range(590, 600))
        + list(range(790, 800))
        + list(range(990, 1000))
    )


def test_task_stratified_groups_are_scoped_per_dataset():
    metas = [_meta([0] * 20, "a"), _meta([0] * 20, "b")]

    val, counts = BaseLerobotDataset._select_task_stratified_episodes(
        metas, val_set_proportion=0.05, is_training_set=False
    )

    assert val == [19, 39]
    assert sum(counts.values()) == 2


def test_active_episode_view_maps_local_frames_without_crossing_gaps():
    dataset = BaseLerobotDataset.__new__(BaseLerobotDataset)
    dataset.episode_data_index = {
        "from": torch.tensor([0, 3, 8, 10]),
        "to": torch.tensor([3, 8, 10, 14]),
    }

    dataset._set_active_episodes([0, 2, 3])

    assert dataset._active_episode_indices == (0, 2, 3)
    assert dataset.episode_data_index["from"].tolist() == [0, 3, 5]
    assert dataset.episode_data_index["to"].tolist() == [3, 5, 9]
    assert dataset._map_active_index(0) == 0
    assert dataset._map_active_index(2) == 2
    assert dataset._map_active_index(3) == 8
    assert dataset._map_active_index(4) == 9
    assert dataset._map_active_index(5) == 10
    assert dataset._map_active_index(8) == 13


def test_task_stratified_split_rejects_single_episode_group():
    meta = _meta([0, 1, 1])

    try:
        BaseLerobotDataset._select_task_stratified_episodes(
            [meta], val_set_proportion=0.05, is_training_set=False
        )
    except ValueError as exc:
        assert "at least two episodes" in str(exc)
    else:
        raise AssertionError("Expected a ValueError for a one-episode task group")
