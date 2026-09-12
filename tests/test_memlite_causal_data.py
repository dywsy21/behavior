from scripts.data.build_memlite_annotations import Episode, annotate_episode


def annotation():
    return {
        "valid_duration": [0, 20],
        "primitive_annotation": [
            {"primitive_idx": 0, "primitive_description": "prepare cup", "skill_idxes": [0, 1], "frame_duration": [0, 10]},
            {"primitive_idx": 1, "primitive_description": "place cup", "skill_idxes": [2], "frame_duration": [10, 20]},
        ],
        "skill_annotation": [
            {"skill_idx": 0, "skill_description": "navigate", "frame_duration": [0, 4]},
            {"skill_idx": 1, "skill_description": "grasp", "frame_duration": [4, 10]},
            {"skill_idx": 2, "skill_description": "release", "frame_duration": [10, 20]},
        ],
    }


def test_current_observation_targets_and_shared_intents():
    labels = annotate_episode(Episode(0, 0, 25), annotation(), [], causal=True, planner_stride=3, memory_corruption_every=0)
    low = {row["frame_index"]: row for row in labels if row["memlite_branch"] == "low"}
    high = sorted([row for row in labels if row["memlite_branch"] == "high"], key=lambda row: row["frame_index"])
    assert len(low) == 20
    assert all(row["intent_status"] == "CONTINUE" for row in low.values())
    assert low[0]["intent"] == low[4]["intent"] == "prepare cup"
    assert low[10]["intent"] == "place cup"
    assert low[3]["action_horizon_end"] == 4
    assert high[-1]["frame_index"] == 20 and high[-1]["intent_status"] == "DONE"
    assert high[0]["previous_intent"] == "None"
    for index, row in enumerate(high):
        assert row["memory_target_frame"] == row["frame_index"]
        assert row["memory_input_frame"] < row["frame_index"]
        if row["intent_status"] == "CONTINUE":
            assert row["intent"] == low[row["frame_index"]]["intent"]
        if index:
            assert row["memory"] == high[index - 1]["memory_update"]
            assert row["previous_intent"] == high[index - 1]["intent"]
        completed = row["memory_update"].split("; Active=", 1)[0]
        if row["frame_index"] < 10:
            assert "prepare cup" not in completed
        if row["frame_index"] < 20:
            assert "place cup" not in completed


def test_terminal_requires_real_post_end_observation():
    warnings = []
    rows = annotate_episode(Episode(0, 0, 20), annotation(), warnings, causal=True)
    assert not any(row["intent_status"] == "DONE" for row in rows)
    assert any(w["type"] == "no_observed_terminal_frame" for w in warnings)


def test_corruption_changes_input_only_not_observed_targets():
    clean = annotate_episode(Episode(0, 0, 25), annotation(), [], causal=True, planner_stride=2, memory_corruption_every=0)
    corrupt = annotate_episode(Episode(0, 0, 25), annotation(), [], causal=True, planner_stride=2, memory_corruption_every=2)
    assert len(clean) == len(corrupt)
    changes = 0
    for left, right in zip(clean, corrupt):
        assert left["memory_update"] == right["memory_update"]
        assert left["intent"] == right["intent"]
        if left["memory"] != right["memory"]:
            assert right["memory_input_corruption"].startswith("synthetic_text_")
            changes += 1
    assert changes > 1
    first_corrupt = next(row for row in corrupt if row["memlite_branch"] == "high" and row["memory_input_corruption"] != "none")
    assert first_corrupt["frame_index"] < 10
    assert "prepare cup" in first_corrupt["memory"]
    assert "place cup" not in first_corrupt["memory"]


def test_v3_episode_metadata_loads_explicit_identity(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from scripts.data.build_memlite_annotations import load_episodes
    folder = tmp_path / "meta/episodes/chunk-000"
    folder.mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist([dict(episode_index=7, task_index=3, length=123,
                                           raw_episode_id=4321, annotation_path="annotations/task-0003/episode_00004321.json")]), folder / "file-000.parquet")
    result = load_episodes(None, tmp_path, input_root=tmp_path)
    assert len(result) == 1
    assert (result[0].episode_index, result[0].task_index, result[0].length, result[0].raw_episode_id) == (7, 3, 123, 4321)


def test_independent_source_timing_agrees_with_current_targets(tmp_path):
    import json
    from scripts.data.audit_memlite_v9 import source_primitive_intervals
    from scripts.data.build_memlite_annotations import make_memory
    source = tmp_path / "source.json"
    source.write_text(json.dumps(annotation()))
    events = source_primitive_intervals(dict(annotation_path=str(source), length=25))
    rows = annotate_episode(Episode(0, 0, 25), annotation(), [], causal=True, planner_stride=3)
    for row in rows:
        if row["memlite_branch"] != "high":
            continue
        frame = row["frame_index"]
        expected = make_memory("0", [e for e in events if e["_end"] <= frame],
                               [e for e in events if e["_start"] < frame < e["_end"]])
        assert row["memory_update"] == expected


def test_nested_primitive_end_is_quarantined_not_flattened_to_guess_labels():
    raw = annotation()
    raw["primitive_annotation"][0]["frame_duration"] = [0, [8, 15]]
    warnings = []
    rows = annotate_episode(Episode(0, 0, 25), raw, warnings, causal=True)
    assert not rows
    assert any(item["type"] == "ambiguous_primitive_interval" and item["dropped_episode"] for item in warnings)
