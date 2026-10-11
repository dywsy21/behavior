import numpy as np
import unittest

from behavior_champion.contracts import (
    LEGACY_STAGES, append_embeddings, delta_actions, make_task_layout, pad32,
    state23_from_proprio61, training_stage,
)


def test_named_mapping_survives_changed_official_ids_and_preserves_all_596_rows():
    old = [f'task_{i}' for i in range(50)]
    current = {i: f'task {99-i}' for i in range(100)}
    layout = make_task_layout(old, current, {i: 9000 for i in range(50)})
    assert layout.dataset_to_model[99] == 0
    assert layout.dataset_to_model[50] == 49
    assert layout.num_stages[:50] == LEGACY_STAGES
    assert layout.stage_offsets[50] == 596
    assert len(set(layout.dataset_to_model.values())) == 100


def test_mapping_refuses_missing_and_ambiguous_names():
    old = [f'task_{i}' for i in range(50)]
    current = {i: f'task {i}' for i in range(100)}
    current[0] = current[1]
    with unittest.TestCase().assertRaises(ValueError):
        make_task_layout(old, current, {})


def test_proprio_order_gripper_and_no_global_pose():
    x = np.arange(61, dtype=np.float32)
    x[24:26] = 0.025
    x[49:51] = 0.05
    state = state23_from_proprio61(x)
    assert state.shape == (23,)
    np.testing.assert_array_equal(state[:7], [0, 1, 2, 53, 54, 55, 56])
    np.testing.assert_array_equal(state[7:14], np.arange(3, 10))
    assert state[14] == 0 and state[22] == 1
    np.testing.assert_array_equal(state[15:22], np.arange(28, 35))


def test_delta_roundtrip_preserves_base_trunk4_and_grippers():
    state = np.arange(23, dtype=np.float32)
    action = np.ones((30, 23), dtype=np.float32) * 37
    delta = delta_actions(action, state)
    np.testing.assert_array_equal(delta[:, [0, 1, 2, 6, 14, 22]], action[:, [0, 1, 2, 6, 14, 22]])
    assert delta[0, 3] == 34 and delta[0, 7] == 30
    np.testing.assert_array_equal(delta_actions(delta, state, inverse=True), action)
    padded = pad32(delta)
    np.testing.assert_array_equal(padded[:, :23], delta)
    assert np.count_nonzero(padded[:, 23:]) == 0
    with unittest.TestCase().assertRaises(ValueError):
        pad32(np.zeros((30, 27)))


def test_embedding_expansion_copies_old_bytes_only():
    old = np.arange(12, dtype=np.float32).reshape(3, 4)
    new = np.ones((5, 4), dtype=np.float32)
    result = append_embeddings(old, new, 3)
    assert result[:3].tobytes() == old.tobytes()
    np.testing.assert_array_equal(result[3:], new[3:])
    assert np.all(new == 1)
    with unittest.TestCase().assertRaises(ValueError):
        append_embeddings(old, new.astype(np.float64), 3)


def test_stage_is_training_duration_bin_not_physical_outcome():
    assert [training_stage(f, 100, 5) for f in (0, 19, 20, 99)] == [0, 0, 1, 4]
    with unittest.TestCase().assertRaises(ValueError):
        training_stage(100, 100, 5)


def load_tests(loader, tests, pattern):
    return unittest.TestSuite(unittest.FunctionTestCase(value) for key, value in globals().items()
                              if key.startswith('test_') and callable(value))


if __name__ == '__main__':
    unittest.main()
