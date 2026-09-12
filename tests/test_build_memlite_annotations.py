import json
from pathlib import Path

from scripts.data.build_memlite_annotations import build, make_parser, preflight_annotations


def _fixture(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "dataset"
    (root / "meta").mkdir(parents=True)
    rows = [
        {
            "episode_index": 0,
            "length": 10,
            "task_index": 0,
            "tasks": ["demo_task"],
            "annotation_path": "annotations/task-0000/episode_00000000.json",
        },
        {
            "episode_index": 1,
            "length": 100,
            "task_index": 0,
            "tasks": ["demo_task"],
            "annotation_path": "annotations/task-0000/episode_00000001.json",
        },
    ]
    (root / "meta" / "episodes.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
    )
    ann_root = tmp_path / "annotations" / "task-0000"
    ann_root.mkdir(parents=True)
    annotation = {
        "meta_data": {"task_duration": 10, "valid_duration": [0, 10]},
        "skill_annotation": [
            {
                "skill_idx": 0,
                "skill_description": ["move to"],
                "object_id": [["red_cup"]],
                "frame_duration": [0, 4],
                "skill_type": ["navigation"],
            },
            {
                "skill_idx": 1,
                "skill_description": ["pick up"],
                "object_id": [["red_cup"]],
                "frame_duration": [4, 10],
                "skill_type": ["manipulation"],
            },
        ],
    }
    (ann_root / "episode_00000000.json").write_text(json.dumps(annotation), encoding="utf-8")
    # This annotation is intentionally badly aligned and must be recorded as
    # skipped instead of silently stretching labels over the episode.
    bad = {"meta_data": {"task_duration": 1000, "valid_duration": [0, 1000]}, "skill_annotation": []}
    (ann_root / "episode_00000001.json").write_text(json.dumps(bad), encoding="utf-8")
    return root, tmp_path / "annotations"


def test_build_sidecar_is_frame_aligned_and_non_leaky(tmp_path):
    root, ann_root = _fixture(tmp_path)
    output = tmp_path / "out"
    args = make_parser().parse_args(
        ["--input-root", str(root), "--annotations-root", str(ann_root), "--output-root", str(output)]
    )
    summary = build(args)
    # Two primitive planner rows, ten low action rows, and one terminal
    # planner DONE row at the final low-level frame.
    assert summary["rows"] == 13
    assert summary["status_counts"] == {"ok": 1, "skipped": 1}
    assert (output / "meta" / "memlite_manifest.json").exists()

    import pyarrow.parquet as pq

    rows = pq.read_table(output / "meta" / "memlite_annotations.parquet").to_pylist()
    assert rows[0]["memlite_branch"] == "high"
    # At the boundary, high wins over low when a loader overlays the sidecar.
    boundary = [row for row in rows if row["frame_index"] == 4]
    assert {row["memlite_branch"] for row in boundary} == {"high", "low"}
    assert all(row["action_horizon_end"] <= row["skill_end"] for row in rows)
    final_low = [row for row in rows if row["memlite_branch"] == "low" and row["frame_index"] == 9][0]
    assert final_low["intent_status"] == "DONE"
    # The second skill's memory only contains the first skill; no future skill
    # text appears in the input memory for the first skill.
    first_high = [row for row in rows if row["memlite_branch"] == "high" and row["skill_idx"] == 0][0]
    second_high = [row for row in rows if row["memlite_branch"] == "high" and row["skill_idx"] == 1][0]
    assert "pick up" not in first_high["memory"]
    assert "move to" in second_high["memory"]


def test_existing_sidecar_is_protected(tmp_path):
    root, ann_root = _fixture(tmp_path)
    output = tmp_path / "out"
    output.joinpath("meta").mkdir(parents=True)
    output.joinpath("meta/memlite_annotations.jsonl").write_text("old\n", encoding="utf-8")
    args = make_parser().parse_args(
        ["--input-root", str(root), "--annotations-root", str(ann_root), "--output-root", str(output)]
    )
    try:
        build(args)
    except FileExistsError:
        pass
    else:
        raise AssertionError("generator must protect an existing sidecar")


def test_json_outputs_clip_ranges_and_emit_status(tmp_path):
    annotations = tmp_path / "annotations" / "task-0001"
    annotations.mkdir(parents=True)
    (annotations / "episode_00000012.json").write_text(
        json.dumps(
            {
                "meta_data": {"valid_duration": [0, 6]},
                "skill_annotation": [
                    {
                        "skill_idx": 0,
                        "skill_description": "move",
                        "object_id": ["cup"],
                        "frame_duration": [4, -1],
                    },
                    {
                        "skill_idx": 1,
                        "skill_description": "place",
                        "object_id": ["bin"],
                        "frame_duration": [4, 9],
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    episodes = tmp_path / "episodes.jsonl"
    episodes.write_text(
        json.dumps({"episode_index": 7, "task_index": 1, "length": 6, "raw_episode_id": 12}) + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "labels"
    args = make_parser().parse_args(
        [
            "--annotations-root",
            str(tmp_path / "annotations"),
            "--episodes",
            str(episodes),
            "--output-root",
            str(output),
        ]
    )
    build(args)
    for name in ("manifest.json", "labels.jsonl", "stats.json", "warnings.json"):
        assert (output / name).exists()
    labels = [json.loads(line) for line in (output / "labels.jsonl").read_text().splitlines()]
    assert {row["memlite_branch"] for row in labels} == {"high", "low"}
    assert {row["frame_index"] for row in labels if row["memlite_branch"] == "high"} == {4, 5}
    assert labels[-1]["intent_status"] == "DONE"
    assert all(row["label_confidence"] == 0.95 for row in labels)
    assert all(row["label_source"] == "skill_annotation" for row in labels if row["memlite_branch"] == "low")
    assert all(row["label_source"] == "skill_annotation_fallback" for row in labels if row["memlite_branch"] == "high" and row["intent_status"] == "CONTINUE")
    assert all(row["label_source"] == "derived_terminal" for row in labels if row["intent_status"] == "DONE" and row["memlite_branch"] == "high")
    assert "move" not in next(row for row in labels if row["skill_idx"] == 1 and row["memlite_branch"] == "high")["memory"]
    warning_types = {warning["type"] for warning in json.loads((output / "warnings.json").read_text())}
    assert {"reversed_interval", "clipped_interval"} <= warning_types
    assert all(row["intent_status"] in {"CONTINUE", "DONE"} for row in labels)


def test_primitive_high_skill_low_strict_identity_overlap_gap_and_filter(tmp_path):
    ann_root = tmp_path / "annotations" / "task-0000"
    ann_root.mkdir(parents=True)
    annotation = {
        "meta_data": {"valid_duration": [0, 8]},
        "primitive_annotation": [
            {"skill_idx": 0, "primitive_idx": 0, "primitive_description": "prepare", "frame_duration": [0, 3]},
            {"skill_idx": 0, "primitive_idx": 1, "primitive_description": "grasp", "frame_duration": [3, 4]},
            {"skill_idx": 1, "primitive_idx": 0, "primitive_description": "place", "frame_duration": [4, 8]},
            {"skill_idx": 2, "primitive_idx": 0, "primitive_description": "finish", "frame_duration": [6, 8]},
        ],
        "skill_annotation": [
            {"skill_idx": 0, "skill_description": "move cup", "object_id": ["cup"], "frame_duration": [0, 4]},
            # Deliberate overlap at frame 3; frame 3 must be emitted once.
            {"skill_idx": 1, "skill_description": "place cup", "object_id": ["bin"], "frame_duration": [3, 5]},
            # Gap at frame 5 is explicit and should be ignored with warning.
            {"skill_idx": 2, "skill_description": "finish cup", "object_id": ["bin"], "frame_duration": [6, 8]},
        ],
    }
    (ann_root / "episode_00000042.json").write_text(json.dumps(annotation), encoding="utf-8")
    episodes = tmp_path / "episodes.jsonl"
    episodes.write_text(
        "\n".join(
            [
                json.dumps({"episode_index": 10, "task_index": 0, "length": 8, "raw_episode_id": 42}),
                json.dumps({"episode_index": 11, "task_index": 0, "length": 8}),
                json.dumps({"episode_index": 12, "task_index": 1, "length": 8, "raw_episode_id": 42}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "out"
    args = make_parser().parse_args(
        [
            "--annotations-root", str(tmp_path / "annotations"), "--episodes", str(episodes),
            "--output-root", str(output), "--action-horizon", "2", "--task-filter", "0",
        ]
    )
    summary = build(args)
    labels = [json.loads(line) for line in (output / "labels.jsonl").read_text().splitlines()]
    highs = [row for row in labels if row["memlite_branch"] == "high"]
    ordinary_highs = [row for row in highs if row["intent_status"] != "DONE"]
    assert {(row["skill_idx"], row["frame_index"]) for row in ordinary_highs} == {(0, 0), (0, 3), (1, 4), (2, 6)}
    terminal_high = next(row for row in highs if row["intent_status"] == "DONE")
    assert terminal_high["frame_index"] == 7 and terminal_high["skill_idx"] is None
    # High supervision is the primitive-level intent, while low conditioning
    # is the executable skill intent.  Their shared skill identity is the
    # explicit hierarchy link; no positional matching is allowed.
    high0 = next(row for row in ordinary_highs if row["skill_idx"] == 0)
    low0 = next(
        row for row in labels
        if row["memlite_branch"] == "low" and row["frame_index"] == 0
    )
    assert "prepare" in high0["intent"] and "move cup" not in high0["intent"]
    assert "move cup" in low0["intent"]
    assert low0["skill_idx"] in high0["primitive_skill_idxes"]
    assert all(row["segment_end"] >= row["action_horizon_end"] for row in labels)
    low_keys = [(row["frame_index"], row["memlite_branch"]) for row in labels]
    assert len(low_keys) == len(set(low_keys))
    assert any("merged_parallel_skills" == warning["type"] for warning in json.loads((output / "warnings.json").read_text()))
    parallel = next(row for row in labels if row["memlite_branch"] == "low" and row["frame_index"] == 3)
    assert "move cup" in parallel["intent"] and "place cup" in parallel["intent"]
    assert any("gap" == warning["type"] for warning in json.loads((output / "warnings.json").read_text()))
    assert json.loads((output / "stats.json").read_text())["gap_frames"] == 1
    assert summary["episodes"] == 2  # task filter removed task 1
    assert any("neither annotation_path" in warning.get("message", "") for warning in json.loads((output / "warnings.json").read_text()))
    assert all(row["intent_status"] in {"CONTINUE", "DONE"} for row in labels)


def test_real_v2_primitive_intent_and_severe_truncation(tmp_path):
    ann_root = tmp_path / "annotations" / "task-0002"
    ann_root.mkdir(parents=True)
    (ann_root / "episode_00000544.json").write_text(
        json.dumps({
            "meta_data": {"valid_duration": [0, 8790]},
            "primitive_annotation": [
                {"primitive_idx": 9, "primitive_description": "press", "object_id": ["button"], "skill_idxes": [0], "frame_duration": [0, 100]},
            ],
            "skill_annotation": [
                {"skill_idx": 0, "skill_description": "pick-up", "object_id": ["cup"], "frame_duration": [0, 100]},
            ],
        }),
        encoding="utf-8",
    )
    episodes = tmp_path / "episodes.jsonl"
    episodes.write_text(json.dumps({"episode_index": 544, "task_index": 2, "length": 148, "raw_episode_id": 544}) + "\n", encoding="utf-8")
    output = tmp_path / "out"
    args = make_parser().parse_args(["--annotations-root", str(tmp_path / "annotations"), "--episodes", str(episodes), "--output-root", str(output)])
    summary = build(args)
    assert summary["rows"] == 0
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["episodes"][0]["status"] == "invalid_episode"
    warnings = json.loads((output / "warnings.json").read_text())
    assert any(item["type"] == "severe_truncation" for item in warnings)
    assert not any(item.get("intent_status") == "DONE" for item in (json.loads(line) for line in (output / "labels.jsonl").read_text().splitlines()))

    # A non-truncated v2 sample proves high intent comes from primitive text,
    # not from the skill at the same positional index.
    good = json.loads((ann_root / "episode_00000544.json").read_text())
    good["meta_data"]["valid_duration"] = [0, 100]
    (ann_root / "episode_00000544.json").write_text(json.dumps(good), encoding="utf-8")
    good_ep = tmp_path / "good.jsonl"
    good_ep.write_text(json.dumps({"episode_index": 1, "task_index": 2, "length": 100, "raw_episode_id": 544}) + "\n", encoding="utf-8")
    good_out = tmp_path / "good-out"
    build(make_parser().parse_args(["--annotations-root", str(tmp_path / "annotations"), "--episodes", str(good_ep), "--output-root", str(good_out)]))
    good_labels = [json.loads(line) for line in (good_out / "labels.jsonl").read_text().splitlines()]
    assert "press" in next(row for row in good_labels if row["memlite_branch"] == "high")["intent"]
    assert "pick-up" not in next(row for row in good_labels if row["memlite_branch"] == "high")["intent"]


def test_multi_skill_primitive_dedup_and_nested_interval(tmp_path):
    root = tmp_path / "annotations" / "task-0003"
    root.mkdir(parents=True)
    annotation = {
        "meta_data": {"valid_duration": [2466, 8222]},
        "primitive_annotation": [
            {"primitive_idx": 4, "primitive_description": "press", "object_id": ["button"], "skill_idxes": [0, 1], "frame_duration": [2466, [8039, 8222]]}
        ],
        "skill_annotation": [
            {"skill_idx": 0, "skill_description": "press", "frame_duration": [2466, 5000]},
            {"skill_idx": 1, "skill_description": "pick-up", "frame_duration": [5000, 8222]},
        ],
    }
    path = root / "episode_00000001.json"
    path.write_text(json.dumps(annotation), encoding="utf-8")
    episodes = tmp_path / "episodes.jsonl"
    episodes.write_text(json.dumps({"episode_index": 1, "task_index": 3, "length": 5756, "raw_episode_id": 1}) + "\n", encoding="utf-8")
    output = tmp_path / "out"
    build(make_parser().parse_args(["--annotations-root", str(tmp_path / "annotations"), "--episodes", str(episodes), "--output-root", str(output)]))
    labels = [json.loads(line) for line in (output / "labels.jsonl").read_text().splitlines()]
    highs = [row for row in labels if row["memlite_branch"] == "high"]
    ordinary_highs = [row for row in highs if row["intent_status"] != "DONE"]
    assert len(ordinary_highs) == 1 and ordinary_highs[0]["skill_idxes"] == [0, 1]
    assert len([row for row in highs if row["intent_status"] == "DONE"]) == 1
    assert highs[0]["intent"].count("press") == 1
    warnings = json.loads((output / "warnings.json").read_text())
    assert not any(item["type"] == "invalid_interval" for item in warnings)


def test_disjoint_low_intervals_and_memory_closed_loop(tmp_path):
    root = tmp_path / "annotations" / "task-0008"
    root.mkdir(parents=True)
    annotation = {
        "meta_data": {"valid_duration": [0, 8]},
        "primitive_annotation": [
            {"primitive_idx": 0, "primitive_description": "first", "skill_idxes": [0], "frame_duration": [0, 2]},
            {"primitive_idx": 1, "primitive_description": "second", "skill_idxes": [1], "frame_duration": [2, 4]},
            {"primitive_idx": 2, "primitive_description": "third", "skill_idxes": [0], "frame_duration": [4, 8]},
        ],
        "skill_annotation": [
            # ep881-style disjoint subsegments: no labels at frames 2/3 for skill 0.
            {"skill_idx": 0, "skill_description": "move", "frame_duration": [[0, 2], [4, 8]]},
            {"skill_idx": 1, "skill_description": "wait", "frame_duration": [2, 4]},
        ],
    }
    (root / "episode_00000001.json").write_text(json.dumps(annotation), encoding="utf-8")
    episodes = tmp_path / "episodes.jsonl"
    episodes.write_text(json.dumps({"episode_index": 1, "task_index": 8, "length": 8, "raw_episode_id": 1}) + "\n", encoding="utf-8")
    output = tmp_path / "out"
    build(make_parser().parse_args(["--annotations-root", str(tmp_path / "annotations"), "--episodes", str(episodes), "--output-root", str(output), "--action-horizon", "32"]))
    labels = [json.loads(line) for line in (output / "labels.jsonl").read_text().splitlines()]
    low0 = [row for row in labels if row["memlite_branch"] == "low" and row["skill_idx"] == 0]
    assert {row["frame_index"] for row in low0} == {0, 1, 4, 5, 6, 7}
    assert {row["interval_id"] for row in low0 if row["frame_index"] < 2} == {0}
    assert {row["interval_id"] for row in low0 if row["frame_index"] >= 4} == {1}
    assert {row["primitive_idx"] for row in low0 if row["frame_index"] < 2} == {0}
    assert {row["primitive_idx"] for row in low0 if row["frame_index"] >= 4} == {2}
    assert all(row["action_horizon_end"] <= row["subsegment_end"] for row in low0)

    highs = sorted((row for row in labels if row["memlite_branch"] == "high" and row["intent_status"] == "CONTINUE"), key=lambda row: row["frame_index"])
    assert [row["intent"] for row in highs] == ["first", "second", "third"]
    assert highs[0]["memory"] == "Task=8; Completed=none."
    assert highs[0]["memory_update"] == highs[1]["memory"]
    assert highs[1]["memory_update"] == highs[2]["memory"]
    terminal = next(row for row in labels if row["memlite_branch"] == "high" and row["intent_status"] == "DONE")
    assert terminal["memory"] == highs[-1]["memory_update"] == terminal["memory_update"]
    assert "Before:" not in "\n".join(row["memory_update"] for row in labels)
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["schema_name"] == "memlite_annotations" and manifest["schema_version"] == 4


def test_v6_structured_prefix_identity_reversed_drop_and_sources(tmp_path):
    root = tmp_path / "annotations" / "task-0002"
    root.mkdir(parents=True)
    annotation = {
        "meta_data": {"valid_duration": [0, 8]},
        "primitive_annotation": [
            {
                "primitive_idx": 10,
                "primitive_description": ["open", "close"],
                "object_id": [["left", "drawer"], ["right", "drawer"]],
                "spatial_prefix": ["left", "right"],
                "memory_prefix": ["low_level shelf", "high_level shelf"],
                "skill_idxes": [0],
                "frame_duration": [0, 4],
            },
            {
                "primitive_idx": 11,
                "primitive_description": "open",
                "object_id": [["layer_4", ["low_level", "shelf"]]],
                "spatial_prefix": "right",
                "memory_prefix": "layer_5",
                "skill_idxes": [1],
                "frame_duration": [4, 6],
            },
            # Same text/relation but a different primitive identity must not
            # disappear from canonical completed memory.
            {
                "primitive_idx": 12,
                "primitive_description": "open",
                "object_id": [["layer_4", ["low_level", "shelf"]]],
                "spatial_prefix": "right",
                "memory_prefix": "layer_5",
                "skill_idxes": [2],
                "frame_duration": [6, 8],
            },
            {"primitive_idx": 13, "primitive_description": "move-to", "skill_idxes": [3], "frame_duration": [7, 5]},
        ],
        "skill_annotation": [
            {
                "skill_idx": 0,
                "skill_description": ["open", "close"],
                "object_id": [["left", "drawer"], ["right", "drawer"]],
                "spatial_prefix": ["left", "right"],
                "memory_prefix": ["low_level shelf", "high_level shelf"],
                "frame_duration": [0, 4],
            },
            {"skill_idx": 1, "skill_description": "open", "object_id": [["layer_4", ["low_level", "shelf"]]], "spatial_prefix": "right", "memory_prefix": "layer_5", "frame_duration": [4, 6]},
            {"skill_idx": 2, "skill_description": "open", "object_id": [["layer_4", ["low_level", "shelf"]]], "spatial_prefix": "right", "memory_prefix": "layer_5", "frame_duration": [6, 8]},
            {"skill_idx": 3, "skill_description": "move-to", "frame_duration": [7, 5]},
        ],
    }
    (root / "episode_00000944.json").write_text(json.dumps(annotation), encoding="utf-8")
    episodes = tmp_path / "episodes.jsonl"
    episodes.write_text(json.dumps({"episode_index": 944, "task_index": 2, "length": 8, "raw_episode_id": 944}) + "\n", encoding="utf-8")
    output = tmp_path / "out"
    build(make_parser().parse_args(["--annotations-root", str(tmp_path / "annotations"), "--episodes", str(episodes), "--output-root", str(output)]))
    labels = [json.loads(line) for line in (output / "labels.jsonl").read_text().splitlines()]
    high = [row for row in labels if row["memlite_branch"] == "high" and row["intent_status"] == "CONTINUE"]
    low = [row for row in labels if row["memlite_branch"] == "low"]
    assert "left low_level shelf open [left drawer]" in high[0]["intent"]
    assert "right high_level shelf close [right drawer]" in high[0]["intent"]
    assert "right layer_5 open [layer_4 -> low_level shelf]" in high[1]["intent"]
    assert "left low_level shelf open [left drawer]" in low[0]["intent"]
    assert not any("move-to" in row["intent"] for row in labels)
    assert all(row["label_source"] == "primitive_annotation" for row in high)
    assert all(row["label_source"] == "skill_annotation" for row in low)
    terminal = next(row for row in labels if row["intent_status"] == "DONE" and row["memlite_branch"] == "high")
    assert terminal["label_source"] == "derived_terminal"
    # primitive_idx 11 and 12 have identical text but must both survive M_final.
    assert terminal["memory"].count("right layer_5 open [layer_4 -> low_level shelf]") == 2
    warnings = json.loads((output / "warnings.json").read_text())
    reversed_warning = next(item for item in warnings if item["type"] == "reversed_interval")
    assert reversed_warning["dropped"] and reversed_warning["label_confidence"] == 0.0


def test_v7_recursive_binding_overlap_state_reversed_valid_and_skill_fallback(tmp_path):
    root = tmp_path / "annotations"

    # Recursive prefix/object trees must keep modifiers with their own leaves,
    # including the ep543-style `left` modifier on the cabinet slot.
    prefix_dir = root / "task-0009"
    prefix_dir.mkdir(parents=True)
    prefix_annotation = {
        "meta_data": {"valid_duration": [0, 4]},
        "primitive_annotation": [{
            "primitive_idx": 1,
            "primitive_description": "place",
            "object_id": [[["cup_a", "cup_b"], "cabinet"]],
            "spatial_prefix": [[["near", "far"], ["left"]]],
            "skill_idxes": [0],
            "frame_duration": [0, 4],
        }],
        "skill_annotation": [{
            "skill_idx": 0,
            "skill_description": "place",
            "object_id": [[["cup_a", "cup_b"], "cabinet"]],
            "spatial_prefix": [[["near", "far"], ["left"]]],
            "frame_duration": [0, 4],
        }],
    }
    (prefix_dir / "episode_00000001.json").write_text(json.dumps(prefix_annotation), encoding="utf-8")

    # ep20140-style overlap: the long-running primitive must remain Active at
    # the short primitive's start rather than becoming Completed prematurely.
    overlap_dir = root / "task-0010"
    overlap_dir.mkdir()
    overlap_annotation = {
        "meta_data": {"valid_duration": [0, 1600]},
        "primitive_annotation": [
            {"primitive_idx": 0, "primitive_description": "long", "skill_idxes": [0], "frame_duration": [0, 1500]},
            {"primitive_idx": 1, "primitive_description": "short", "skill_idxes": [0], "frame_duration": [1230, 1400]},
            {"primitive_idx": 2, "primitive_description": "last", "skill_idxes": [0], "frame_duration": [1500, 1600]},
        ],
        "skill_annotation": [{"skill_idx": 0, "skill_description": "execute", "frame_duration": [0, 1600]}],
    }
    (overlap_dir / "episode_00020140.json").write_text(json.dumps(overlap_annotation), encoding="utf-8")

    # A reversed valid_duration invalidates the complete episode (no swap).
    reversed_dir = root / "task-0011"
    reversed_dir.mkdir()
    reversed_annotation = {
        "meta_data": {"valid_duration": [4, 0]},
        "primitive_annotation": [{"primitive_idx": 0, "primitive_description": "bad", "skill_idxes": [0], "frame_duration": [0, 4]}],
        "skill_annotation": [{"skill_idx": 0, "skill_description": "bad", "frame_duration": [0, 4]}],
    }
    (reversed_dir / "episode_00000003.json").write_text(json.dumps(reversed_annotation), encoding="utf-8")

    # Missing primitive_annotation remains usable with explicit skill fallback.
    fallback_dir = root / "task-0012"
    fallback_dir.mkdir()
    fallback_annotation = {
        "meta_data": {"valid_duration": [0, 3]},
        "skill_annotation": [{"skill_idx": 7, "skill_description": "fallback-skill", "object_id": ["cup"], "frame_duration": [0, 3]}],
    }
    (fallback_dir / "episode_00000004.json").write_text(json.dumps(fallback_annotation), encoding="utf-8")

    episodes = tmp_path / "episodes.jsonl"
    episodes.write_text("\n".join(json.dumps(row) for row in [
        {"episode_index": 1, "task_index": 9, "length": 4, "raw_episode_id": 1},
        {"episode_index": 20140, "task_index": 10, "length": 1600, "raw_episode_id": 20140},
        {"episode_index": 3, "task_index": 11, "length": 4, "raw_episode_id": 3},
        {"episode_index": 4, "task_index": 12, "length": 3, "raw_episode_id": 4},
    ]) + "\n", encoding="utf-8")
    output = tmp_path / "out"
    summary = build(make_parser().parse_args(["--annotations-root", str(root), "--episodes", str(episodes), "--output-root", str(output)]))
    assert summary["rows"] > 0
    labels = [json.loads(line) for line in (output / "labels.jsonl").read_text().splitlines()]

    prefix_rows = [row for row in labels if row["episode_index"] == 1 and row["intent_status"] == "CONTINUE"]
    assert prefix_rows
    for row in prefix_rows:
        assert "cup_a near" in row["intent"]
        assert "cup_b far" in row["intent"]
        assert "cabinet left" in row["intent"]

    overlap_high = sorted(
        (row for row in labels if row["episode_index"] == 20140 and row["memlite_branch"] == "high"),
        key=lambda row: row["frame_index"],
    )
    regular = [row for row in overlap_high if row["intent_status"] == "CONTINUE"]
    terminal = next(row for row in overlap_high if row["intent_status"] == "DONE")
    assert [row["frame_index"] for row in regular] == [0, 1230, 1500]
    assert "long" not in regular[0]["memory"]
    assert "Completed=none" in regular[1]["memory"]
    assert "Active=long" in regular[1]["memory"]
    assert "short" not in regular[1]["memory"]
    assert "Completed=long; short" in regular[2]["memory"]
    assert "last" not in regular[2]["memory"]
    assert regular[0]["memory_update"] == regular[1]["memory"]
    assert regular[1]["memory_update"] == regular[2]["memory"]
    assert regular[2]["memory_update"] == terminal["memory"] == terminal["memory_update"]

    fallback_high = next(row for row in labels if row["episode_index"] == 4 and row["memlite_branch"] == "high" and row["intent_status"] == "CONTINUE")
    assert fallback_high["label_source"] == "skill_annotation_fallback"
    assert "fallback-skill" in fallback_high["intent"]

    manifest = json.loads((output / "manifest.json").read_text())
    reversed_episode = next(row for row in manifest["episodes"] if row["episode_index"] == 3)
    assert reversed_episode["status"] == "invalid_episode" and reversed_episode["labels"] == 0
    assert not any(row["episode_index"] == 3 for row in labels)
    warnings = json.loads((output / "warnings.json").read_text())
    assert any(item["type"] == "reversed_interval" and item.get("type_context") == "valid_duration" and item["dropped"] for item in warnings)


def test_v8_recursive_tree_renderer_preserves_shapes_without_debug_paths(tmp_path):
    root = tmp_path / "annotations" / "task-0013"
    root.mkdir(parents=True)
    annotation = {
        "meta_data": {"valid_duration": [0, 4]},
        "primitive_annotation": [
            {
                "primitive_idx": 0,
                "primitive_description": "place",
                "object_id": [[["cup_a", "cup_b"], "cabinet"]],
                "spatial_prefix": [[["near", "far"], ["left"]]],
                "skill_idxes": [0],
                "frame_duration": [0, 1],
            },
            {
                "primitive_idx": 1,
                "primitive_description": "open",
                "object_id": [["right_door", "left_door"]],
                "spatial_prefix": [["right", "left"]],
                "skill_idxes": [1],
                "frame_duration": [1, 2],
            },
            {
                "primitive_idx": 2,
                "primitive_description": "inspect",
                "object_id": [["a", "b"]],
                "spatial_prefix": [["near", "far", "left"]],
                "skill_idxes": [2],
                "frame_duration": [2, 4],
            },
        ],
        "skill_annotation": [
            {"skill_idx": 0, "skill_description": "place", "object_id": [[["cup_a", "cup_b"], "cabinet"]], "spatial_prefix": [[["near", "far"], ["left"]]], "frame_duration": [0, 1]},
            {"skill_idx": 1, "skill_description": "open", "object_id": [["right_door", "left_door"]], "spatial_prefix": [["right", "left"]], "frame_duration": [1, 2]},
            {"skill_idx": 2, "skill_description": "inspect", "object_id": [["a", "b"]], "spatial_prefix": [["near", "far", "left"]], "frame_duration": [2, 4]},
        ],
    }
    (root / "episode_00000881.json").write_text(json.dumps(annotation), encoding="utf-8")
    episodes = tmp_path / "episodes.jsonl"
    episodes.write_text(json.dumps({"episode_index": 881, "task_index": 13, "length": 4, "raw_episode_id": 881}) + "\n", encoding="utf-8")
    output = tmp_path / "out"
    build(make_parser().parse_args(["--annotations-root", str(tmp_path / "annotations"), "--episodes", str(episodes), "--output-root", str(output)]))
    labels = [json.loads(line) for line in (output / "labels.jsonl").read_text().splitlines()]
    intents = "\n".join(row["intent"] for row in labels)
    assert "cup_a near" in intents and "cup_b far" in intents and "cabinet left" in intents
    assert "right_door right" in intents and "left_door left" in intents
    assert "object_path" not in intents and "modifier_path" not in intents
    assert 'Objects=[["a","b"]]; Spatial=[["near","far","left"]]; Memory=null' in intents


def test_empty_primitive_descriptions_use_skill_or_canonical_fallback_and_preflight(tmp_path):
    root = tmp_path / "annotations" / "task-0004"
    root.mkdir(parents=True)
    # The first two primitives mirror raw40430#2 and raw42490#6: their
    # primitive text is an empty list, but the linked skills are available.
    annotation = {
        "meta_data": {"valid_duration": [0, 6]},
        "primitive_annotation": [
            {"primitive_idx": 2, "primitive_description": [], "object_id": [], "spatial_prefix": [], "memory_prefix": [], "skill_idxes": [11], "frame_duration": [0, 2]},
            {"primitive_idx": 6, "primitive_description": [], "object_id": [], "spatial_prefix": [], "memory_prefix": [], "skill_idxes": [14], "frame_duration": [2, 4]},
            {"primitive_idx": 7, "primitive_description": [], "object_id": ["cup"], "spatial_prefix": "near", "skill_idxes": [99], "frame_duration": [4, 6]},
        ],
        "skill_annotation": [
            {"skill_idx": 11, "skill_description": ["move to"], "frame_duration": [0, 2]},
            {"skill_idx": 14, "skill_description": ["turn to"], "frame_duration": [2, 4]},
            {"skill_idx": 88, "skill_description": "wait", "frame_duration": [4, 6]},
        ],
    }
    (root / "episode_00040430.json").write_text(json.dumps(annotation), encoding="utf-8")
    # A second real-id-shaped sample verifies the fix is schema based rather
    # than a special case for raw40430.
    second = json.loads(json.dumps(annotation))
    second["primitive_annotation"] = [second["primitive_annotation"][1]]
    second["primitive_annotation"][0]["frame_duration"] = [0, 6]
    second["skill_annotation"] = [{"skill_idx": 14, "skill_description": ["turn to"], "frame_duration": [0, 6]}]
    (root / "episode_00042490.json").write_text(json.dumps(second), encoding="utf-8")
    episodes = tmp_path / "episodes.jsonl"
    episodes.write_text("\n".join(json.dumps(row) for row in [
        {"episode_index": 840, "task_index": 4, "length": 6, "raw_episode_id": 40430},
        {"episode_index": 964, "task_index": 4, "length": 6, "raw_episode_id": 42490},
    ]) + "\n", encoding="utf-8")
    output = tmp_path / "out"
    build(make_parser().parse_args(["--annotations-root", str(tmp_path / "annotations"), "--episodes", str(episodes), "--output-root", str(output)]))
    labels = [json.loads(line) for line in (output / "labels.jsonl").read_text().splitlines()]
    first_high = sorted(
        (row for row in labels if row["episode_index"] == 840 and row["memlite_branch"] == "high" and row["intent_status"] == "CONTINUE"),
        key=lambda row: row["frame_index"],
    )
    assert [row["intent"] for row in first_high] == ["move to", "turn to", "execute primitive 7 [cup near]"]
    assert all(row["label_confidence"] == 0.5 for row in first_high)
    assert [row["intent_provenance"] for row in first_high] == [
        "skill_annotation_description_fallback",
        "skill_annotation_description_fallback",
        "canonical_primitive_fallback",
    ]
    second_high = next(row for row in labels if row["episode_index"] == 964 and row["memlite_branch"] == "high" and row["intent_status"] == "CONTINUE")
    assert second_high["intent"] == "turn to"
    assert second_high["intent_provenance"] == "skill_annotation_description_fallback"
    warnings = json.loads((output / "warnings.json").read_text())
    assert len([row for row in warnings if row["type"] == "primitive_description_fallback"]) == 4
    preflight = preflight_annotations(tmp_path / "annotations", limit=1000)
    assert preflight == {
        "checked": 2,
        "fallback_count": 4,
        "placeholder_count": 0,
        "render_errors": [],
        "errors": [],
    }


def test_non_alignable_nonempty_skill_is_one_lossless_canonical_intent(tmp_path):
    root = tmp_path / "annotations" / "task-0004"
    root.mkdir(parents=True)
    # This matches the real task4 failure shape: one concrete navigation
    # description, one object group, and two memory-prefix leaves.  The old
    # renderer incorrectly invented a second `execute current skill` slot.
    annotation = {
        "meta_data": {"valid_duration": [0, 3]},
        "primitive_annotation": [
            {"primitive_idx": 0, "primitive_description": "pick", "skill_idxes": [0], "frame_duration": [0, 3]},
        ],
        "skill_annotation": [
            {
                "skill_idx": 0,
                "skill_type": ["navigation"],
                "skill_description": ["move to"],
                "object_id": [["bratwurst_230"]],
                "spatial_prefix": [],
                "memory_prefix": ["back", "the other"],
                "manipulating_object_id": [],
                "frame_duration": [0, 3],
            },
        ],
    }
    (root / "episode_00040570.json").write_text(json.dumps(annotation), encoding="utf-8")
    episodes = tmp_path / "episodes.jsonl"
    episodes.write_text(json.dumps({"episode_index": 854, "task_index": 4, "length": 3, "raw_episode_id": 40570}) + "\n", encoding="utf-8")
    output = tmp_path / "out"
    build(make_parser().parse_args(["--annotations-root", str(tmp_path / "annotations"), "--episodes", str(episodes), "--output-root", str(output)]))
    labels = [json.loads(line) for line in (output / "labels.jsonl").read_text().splitlines()]
    low = next(row for row in labels if row["memlite_branch"] == "low" and row["frame_index"] == 0)
    assert low["intent"] == 'Descriptions=["move to"]; Objects=[["bratwurst_230"]]; Spatial=[]; Memory=["back","the other"]; Manipulating=[]'
    assert "execute current skill" not in low["intent"]
    preflight = preflight_annotations(tmp_path / "annotations", limit=1000)
    assert preflight["placeholder_count"] == 0
    assert preflight["render_errors"] == []
