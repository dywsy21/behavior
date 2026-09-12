#!/usr/bin/env python3
"""Build frame-aligned MEM-Lite labels from skill annotation JSON files.

Annotation files live below ``annotations_root/task-000X/*.json``.  Episodes
may be supplied as JSONL or CSV; if omitted, ``--episode-length`` synthesizes
one episode per annotation file.  Outputs use only the Python standard
library: manifest.json, labels.jsonl, stats.json and warnings.json.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

SCHEMA_VERSION = 4

@dataclass
class Episode:
    episode_index: int
    task_index: int
    length: int
    raw_episode_id: int | None = None
    annotation_path: str = ""

def _clean(value: Any) -> str:
    if value is None: return ""
    if isinstance(value, list): return ", ".join(x for x in (_clean(v) for v in value) if x)
    if isinstance(value, dict): return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return re.sub(r"\s+", " ", str(value)).strip()

def _objects(value: Any) -> list[str]:
    out: list[str] = []
    for item in value if isinstance(value, list) else [value]:
        values = item if isinstance(item, list) else [item]
        for nested in values:
            text = _clean(nested)
            if text and text not in out: out.append(text)
    return out

def _relation_text(value: Any) -> str:
    """Render nested object relations without flattening them into a bag."""
    if value is None:
        return ""
    if isinstance(value, dict):
        return " ".join(f"{key}={_relation_text(item)}" for key, item in sorted(value.items()) if _relation_text(item))
    if isinstance(value, (list, tuple)):
        parts = [_relation_text(item) for item in value]
        parts = [part for part in parts if part]
        if not parts:
            return ""
        separator = " -> " if any(isinstance(item, (list, tuple, dict)) for item in value) else " "
        return separator.join(parts)
    return _clean(value)


def _group_value(value: Any, index: int, count: int) -> tuple[Any, bool]:
    """Select a per-description top-level group without broadcasting it."""
    if count <= 1 or value is None:
        return value, True
    if isinstance(value, (list, tuple)) and len(value) == count:
        return value[index], True
    return value, False


def _top_level_group_count(value: Any) -> int:
    """How many explicit top-level annotation groups are present."""
    return len(value) if isinstance(value, (list, tuple)) and value else 1


def _strip_singletons(value: Any) -> Any:
    while isinstance(value, (list, tuple)) and len(value) == 1:
        value = value[0]
    return value


def _scalar(value: Any) -> str | None:
    if value is None or isinstance(value, (list, tuple, dict)):
        return None
    return _clean(value) or None


def _nonempty_leaves(value: Any) -> list[str] | None:
    """Leaf collection is only used for the explicit scalar-object rule."""
    value = _strip_singletons(value)
    scalar = _scalar(value)
    if scalar is not None:
        return [scalar]
    if isinstance(value, dict):
        return None
    if not isinstance(value, (list, tuple)):
        return []
    leaves: list[str] = []
    for item in value:
        child = _nonempty_leaves(item)
        if child is None:
            return None
        leaves.extend(child)
    return leaves


def _object_leaves(value: Any) -> list[str] | None:
    value = _strip_singletons(value)
    if value is None:
        return []
    scalar = _scalar(value)
    if scalar is not None:
        return [scalar]
    if isinstance(value, dict) or not isinstance(value, (list, tuple)):
        return None
    leaves: list[str] = []
    for item in value:
        child = _object_leaves(item)
        if child is None:
            return None
        leaves.extend(child)
    return leaves


def _bind_tree(objects: Any, modifier: Any, *, root: bool = True) -> tuple[dict[tuple[int, ...], list[str]], str | None] | None:
    """Align one modifier tree to objects, with no leaf-order fallback."""
    objects = _strip_singletons(objects)
    modifier = _strip_singletons(modifier)
    modifier_scalar = _scalar(modifier)
    object_scalar = _scalar(objects)
    if modifier is None or modifier == []:
        return {}, None
    if object_scalar is not None:
        if modifier_scalar is not None:
            return {(): [modifier_scalar]}, None
        leaves = _nonempty_leaves(modifier)
        return ({(): leaves}, None) if leaves is not None else None
    if not isinstance(objects, (list, tuple)) or isinstance(objects, dict):
        return None
    if modifier_scalar is not None:
        # A scalar prefix is global context.  Do not duplicate it per object.
        return ({}, modifier_scalar) if root else None
    if not isinstance(modifier, (list, tuple)) or len(objects) != len(modifier):
        return None
    bindings: dict[tuple[int, ...], list[str]] = {}
    for index, (object_child, modifier_child) in enumerate(zip(objects, modifier)):
        child = _bind_tree(object_child, modifier_child, root=False)
        if child is None:
            return None
        child_bindings, child_global = child
        if child_global is not None:
            return None
        for path, values in child_bindings.items():
            bindings[(index, *path)] = values
    return bindings, None


def _canonical_relation(objects: Any, spatial: Any, memory_prefix: Any, manipulating: Any) -> str:
    def render(value: Any) -> str:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    relation = f"Objects={render(objects)}; Spatial={render(spatial)}; Memory={render(memory_prefix)}"
    if manipulating is not None:
        relation += f"; Manipulating={render(manipulating)}"
    return relation


def _canonical_annotation(
    descriptions: Any,
    objects: Any,
    spatial: Any,
    memory_prefix: Any,
    manipulating: Any,
) -> str:
    """Losslessly render a non-alignable annotation as one intent.

    A description list and its modifier/object trees may only be bound
    positionally when *all* top-level group counts agree.  Rendering a second
    synthetic description slot when they disagree is unsafe: it converts a
    real one-action annotation into ``execute current skill`` supervision.
    The canonical form keeps every original tree available without guessing
    which modifier belongs to which description.
    """
    rendered_descriptions = json.dumps(descriptions, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return f"Descriptions={rendered_descriptions}; {_canonical_relation(objects, spatial, memory_prefix, manipulating)}"


def _description_group_count(value: Any) -> int:
    return len(value) if isinstance(value, (list, tuple)) else 1


def _has_nonempty_description(value: Any) -> bool:
    values = value if isinstance(value, (list, tuple)) else [value]
    return any(_clean(item) for item in values)


def _needs_canonical_annotation(
    descriptions: Any,
    objects: Any,
    spatial: Any,
    memory_prefix: Any,
    manipulating: Any,
) -> bool:
    """True only for nonempty descriptions with unsafe top-level grouping."""
    if not _has_nonempty_description(descriptions):
        return False
    count = _description_group_count(descriptions)
    return any(
        value not in (None, []) and _top_level_group_count(value) != count
        for value in (objects, spatial, memory_prefix, manipulating)
    )


def _render_bound_relation(objects: Any, spatial: Any, memory_prefix: Any, manipulating: Any, *, grouped: bool = True) -> tuple[str, str]:
    """Render only shape-preserving modifier/object alignments.

    Lists are zipped recursively only when their child counts agree.  Unlike
    the previous renderer this never zips unrelated leaves or guesses an
    association from traversal order.
    """
    if not grouped:
        return "", _canonical_relation(objects, spatial, memory_prefix, manipulating)
    leaves = _object_leaves(objects)
    if leaves is None:
        return "", _canonical_relation(objects, spatial, memory_prefix, manipulating)
    if not leaves:
        prefix = " ".join(part for part in (_relation_text(spatial), _relation_text(memory_prefix)) if part)
        return prefix, _relation_text(manipulating)

    spatial_bound = _bind_tree(objects, spatial)
    memory_bound = _bind_tree(objects, memory_prefix)
    if spatial_bound is None or memory_bound is None:
        return "", _canonical_relation(objects, spatial, memory_prefix, manipulating)
    spatial_bindings, spatial_global = spatial_bound
    memory_bindings, memory_global = memory_bound
    prefix = " ".join(part for part in (spatial_global, memory_global) if part)
    if not spatial_bindings and not memory_bindings:
        relation = _relation_text(objects)
        manipulating_text = _relation_text(manipulating)
        if manipulating_text:
            relation = " -> ".join(part for part in (relation, manipulating_text) if part)
        return prefix, relation

    # Recurse once more to retain deterministic object order while carrying
    # the modifier values bound at each matching leaf path.
    def render_tree(value: Any, path: tuple[int, ...] = ()) -> list[str] | None:
        value = _strip_singletons(value)
        scalar = _scalar(value)
        if scalar is not None:
            return [" ".join([scalar, *spatial_bindings.get(path, []), *memory_bindings.get(path, [])])]
        if not isinstance(value, (list, tuple)) or isinstance(value, dict):
            return None
        result: list[str] = []
        for index, child in enumerate(value):
            rendered = render_tree(child, path + (index,))
            if rendered is None:
                return None
            result.extend(rendered)
        return result

    parts = render_tree(objects)
    if parts is None:
        return "", _canonical_relation(objects, spatial, memory_prefix, manipulating)
    relation = "; ".join(parts)
    manipulating_text = _relation_text(manipulating)
    if manipulating_text:
        relation = " -> ".join(part for part in (relation, manipulating_text) if part)
    return prefix, relation


def _annotation_render(
    item: dict[str, Any],
    description_keys: tuple[str, ...],
    *,
    fallback: str,
    fallback_descriptions: Iterable[str] | None = None,
) -> str:
    descriptions: Any = None
    for key in description_keys:
        if item.get(key) is not None:
            descriptions = item[key]
            break
    description_entries = list(descriptions) if isinstance(descriptions, (list, tuple)) else [descriptions]
    source_objects = item.get("object_id", item.get("object_ids"))
    source_spatial = item.get("spatial_prefix")
    source_memory_prefix = item.get("memory_prefix")
    source_manipulating = item.get("manipulating_object_id")
    # A nonempty description is the only safe authority for how many intent
    # slots exist.  If any other top-level tree has a different cardinality,
    # preserve the full original structure in one canonical intent instead of
    # inventing a fallback description for an unpaired modifier/object group.
    if _needs_canonical_annotation(descriptions, source_objects, source_spatial, source_memory_prefix, source_manipulating):
        return _canonical_annotation(descriptions, source_objects, source_spatial, source_memory_prefix, source_manipulating)
    count = max(1, len(description_entries))
    fallback_entries = [text for text in (_clean(value) for value in (fallback_descriptions or ())) if text]
    rendered: list[str] = []
    for index in range(count):
        raw_description = description_entries[index] if index < len(description_entries) else None
        description = _clean(raw_description)
        if not description:
            description = fallback_entries[min(index, len(fallback_entries) - 1)] if fallback_entries else fallback
        objects, objects_grouped = _group_value(source_objects, index, count)
        spatial, spatial_grouped = _group_value(source_spatial, index, count)
        memory_prefix, memory_grouped = _group_value(source_memory_prefix, index, count)
        manipulating, manipulating_grouped = _group_value(source_manipulating, index, count)
        prefix, object_relation = _render_bound_relation(
            objects, spatial, memory_prefix, manipulating,
            grouped=objects_grouped and spatial_grouped and memory_grouped and manipulating_grouped,
        )
        text = " ".join(part for part in (prefix, description) if part)
        rendered.append(text + (f" [{object_relation}]" if object_relation else ""))
    return " AND ".join(dict.fromkeys(rendered))


def _skill_description(skill: dict[str, Any]) -> str:
    return _annotation_render(skill, ("skill_description", "description"), fallback="execute current skill")


def skill_intent(skill: dict[str, Any]) -> str:
    return _skill_description(skill)


def primitive_intent(primitive: dict[str, Any], *, fallback_descriptions: Iterable[str] | None = None) -> str:
    """Intent for a high-level primitive, paired with its annotation context."""
    return _annotation_render(
        primitive,
        ("primitive_description", "description", "name"),
        fallback="execute current primitive",
        fallback_descriptions=fallback_descriptions,
    )


def _primitive_summary(primitive: dict[str, Any]) -> str:
    if not any(key in primitive for key in ("primitive_description", "description", "skill_description")) and isinstance(primitive.get("_skill"), dict):
        primitive = primitive["_skill"]
    return _annotation_render(
        primitive,
        ("primitive_description", "description", "skill_description", "primitive", "name"),
        fallback="primitive",
        fallback_descriptions=primitive.get("_intent_fallback_descriptions"),
    )


def _primitive_identity(primitive: dict[str, Any]) -> str:
    source = primitive.get("_skill") if isinstance(primitive.get("_skill"), dict) else primitive
    identity = {
        "primitive_idx": primitive.get("_primitive_idx", source.get("primitive_idx")),
        "modifier": source.get("modifier", source.get("modifiers")),
        "description": source.get("primitive_description", source.get("description", source.get("skill_description", source.get("name")))),
        "object_relation": source.get("object_id", source.get("object_ids")),
        "manipulating_object_id": source.get("manipulating_object_id"),
        "spatial_prefix": source.get("spatial_prefix"),
        "memory_prefix": source.get("memory_prefix"),
    }
    return json.dumps(identity, ensure_ascii=False, sort_keys=True, default=str)


def _summary(items: Iterable[dict[str, Any]]) -> str:
    seen: set[str] = set()
    values: list[str] = []
    for item in items:
        identity = _primitive_identity(item)
        if identity not in seen:
            seen.add(identity); values.append(_primitive_summary(item))
    return "; ".join(values) if values else "none"

def make_memory(task_name: str, completed: Iterable[dict[str, Any]], active: Iterable[dict[str, Any]] = ()) -> str:
    text = f"Task={task_name}; Completed={_summary(completed)}"
    active_summary = _summary(active)
    if active_summary != "none":
        text += f"; Active={active_summary}"
    return text + "."

def _json_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".csv":
        with path.open(newline="", encoding="utf-8") as fh: return [dict(row) for row in csv.DictReader(fh)]
    text = path.read_text(encoding="utf-8").strip()
    if not text: return []
    try: value = json.loads(text)
    except json.JSONDecodeError:
        rows = []
        for line_no, line in enumerate(text.splitlines(), 1):
            if not line.strip(): continue
            try: row = json.loads(line)
            except json.JSONDecodeError as exc: raise ValueError(f"invalid JSONL {path}:{line_no}: {exc}") from exc
            if not isinstance(row, dict): raise ValueError(f"expected object in {path}:{line_no}")
            rows.append(row)
        return rows
    if isinstance(value, dict): return [value]
    if isinstance(value, list) and all(isinstance(row, dict) for row in value): return value
    raise ValueError(f"expected JSON object/list in {path}")

def _episode_from_row(row: dict[str, Any], ordinal: int) -> Episode:
    def integer(name: str, default: int = 0) -> int:
        value = row.get(name, default)
        return int(float(value)) if value not in (None, "") else default
    raw = row.get("raw_episode_id")
    return Episode(integer("episode_index", ordinal), integer("task_index", 0), integer("length", 0), int(float(raw)) if raw not in (None, "") else None, _clean(row.get("annotation_path", "")))

def _annotation_files(root: Path) -> list[Path]:
    return sorted(path for task in root.glob("task-*") if task.is_dir() for path in task.glob("*.json"))

def load_episodes(episodes_path: Path | None, annotations_root: Path, *, input_root: Path | None = None, episode_length: int | None = None) -> list[Episode]:
    if episodes_path is None and input_root is not None:
        candidate = input_root / "meta" / "episodes.jsonl"
        if candidate.exists(): episodes_path = candidate
        else:
            # LeRobot v3 stores episode metadata in sharded parquet files.
            # Use the explicit identity columns, never infer a task from its
            # ordinal or borrow an arbitrary constant episode length.
            shards = sorted((input_root / "meta" / "episodes").glob("chunk-*/file-*.parquet"))
            if shards:
                import pyarrow.parquet as pq
                columns = ["episode_index", "task_index", "length", "raw_episode_id", "annotation_path"]
                rows = [row for shard in shards for row in pq.read_table(shard, columns=columns).to_pylist()]
                result = [_episode_from_row(row, i) for i, row in enumerate(rows)]
                if len({ep.episode_index for ep in result}) != len(result):
                    raise ValueError("Duplicate episode indices in v3 metadata")
                return result
    if episodes_path is not None: return [_episode_from_row(row, i) for i, row in enumerate(_json_rows(episodes_path))]
    if episode_length is None: raise ValueError("episodes file is required unless --episode-length is supplied")
    if episode_length < 0: raise ValueError("--episode-length must be non-negative")
    result = []
    for ordinal, path in enumerate(_annotation_files(annotations_root)):
        task_match = re.search(r"task-(\d+)", path.parent.name)
        id_match = re.search(r"(\d+)(?=\.json$)", path.name)
        result.append(Episode(ordinal, int(task_match.group(1)) if task_match else 0, episode_length, int(id_match.group(1)) if id_match else ordinal, str(path)))
    return result

def _resolve_annotation(ep: Episode, root: Path, dataset_root: Path | None = None) -> Path:
    if ep.annotation_path:
        value = Path(ep.annotation_path)
        choices = [value] if value.is_absolute() else [root / value, value]
        if dataset_root is not None and not value.is_absolute(): choices.append(dataset_root / value)
        if not value.is_absolute(): choices.append(root.parent / value)
        for candidate in choices:
            if candidate.exists(): return candidate
    # Episode index alone is a dataset-local ordinal and is not a stable
    # annotation identity.  Require an explicit path or raw episode id.
    if ep.raw_episode_id is None:
        raise FileNotFoundError(
            f"episode {ep.episode_index} has neither annotation_path nor raw_episode_id"
        )
    task_dir = root / f"task-{ep.task_index:04d}"
    ids = [ep.raw_episode_id]
    for value in ids:
        for candidate in (task_dir / f"episode_{value:08d}.json", task_dir / f"{value}.json"):
            if candidate.exists(): return candidate
    if task_dir.exists():
        for candidate in sorted(task_dir.glob("*.json")):
            numbers = [int(x) for x in re.findall(r"\d+", candidate.stem)]
            if any(number in ids for number in numbers): return candidate
    raise FileNotFoundError(f"annotation not found for episode {ep.episode_index} task {ep.task_index}")

def _numeric_values(value: Any) -> list[float]:
    """Recursively collect interval endpoints from nested v2 structures."""
    if isinstance(value, dict):
        if "start" in value or "end" in value:
            return _numeric_values(value.get("start")) + _numeric_values(value.get("end"))
        return []
    if isinstance(value, (list, tuple)):
        values: list[float] = []
        for item in value: values.extend(_numeric_values(item))
        return values
    try: return [float(value)]
    except (TypeError, ValueError): return []


def _intervals(value: Any) -> list[tuple[int, int]]:
    if isinstance(value, dict):
        if "start" in value or "end" in value: return _intervals([value.get("start"), value.get("end")])
        return []
    if isinstance(value, (list, tuple)):
        if len(value) >= 2 and not isinstance(value[0], (list, tuple, dict)) and not isinstance(value[1], (list, tuple, dict)):
            try: return [(int(float(value[0])), int(float(value[-1])))]
            except (TypeError, ValueError): return []
        # v2 sometimes stores [start, [sub_start, end]]; its envelope is one
        # interval, while lists of pairs are merged after sorting.
        if len(value) == 2 and not isinstance(value[0], (list, tuple, dict)):
            nums = _numeric_values(value)
            return [(int(nums[0]), int(nums[-1]))] if len(nums) >= 2 else []
        result: list[tuple[int, int]] = []
        for item in value: result.extend(_intervals(item))
        return result
    return []


def _range(value: Any) -> tuple[int, int] | None:
    intervals = _intervals(value)
    if intervals:
        ordered = sorted(intervals)
        return min(item[0] for item in ordered), max(item[1] for item in ordered)
    values = _numeric_values(value)
    if len(values) < 2: return None
    return int(values[0]), int(values[-1])

def _clip(value: Any, low: int, high: int, warnings: list[dict[str, Any]], context: dict[str, Any]) -> tuple[int, int] | None:
    original = _range(value)
    if original is None:
        warnings.append({**context, "type": "invalid_interval", "value": value}); return None
    start, end = original
    if start > end:
        warnings.append({**context, "type": "reversed_interval", "original": [start, end], "dropped": True, "label_confidence": 0.0})
        return None
    clipped = [max(low, min(high, start)), max(low, min(high, end))]
    if clipped != [start, end]: warnings.append({**context, "type": "clipped_interval", "original": [start, end], "clipped": clipped})
    if clipped[1] <= clipped[0]:
        warnings.append({**context, "type": "empty_interval", "interval": clipped}); return None
    return clipped[0], clipped[1]


def _clip_intervals(value: Any, low: int, high: int, warnings: list[dict[str, Any]], context: dict[str, Any]) -> list[tuple[int, int]]:
    """Clip, sort and merge interval pieces without filling real gaps."""
    raw_intervals = _intervals(value)
    if not raw_intervals:
        warnings.append({**context, "type": "invalid_interval", "value": value})
        return []
    clipped: list[tuple[int, int]] = []
    for interval_id, interval in enumerate(raw_intervals):
        current = _clip(list(interval), low, high, warnings, {**context, "interval_id": interval_id})
        if current is not None:
            clipped.append(current)
    merged: list[tuple[int, int]] = []
    for start, end in sorted(clipped):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged

def _primitives(skill: dict[str, Any]) -> list[dict[str, Any]]:
    values = skill.get("primitives", skill.get("primitive_annotation", skill.get("primitive_annotations")))
    if values is None or isinstance(values, dict): values = [values] if isinstance(values, dict) else [skill]
    if not isinstance(values, list) or not values: return [skill]
    return [value if isinstance(value, dict) else {"description": value} for value in values]

def _as_skill_list(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, dict):
        value = value.get("skills", value.get("annotations", [value]))
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _description_values(item: dict[str, Any], keys: tuple[str, ...]) -> list[str]:
    value: Any = None
    for key in keys:
        if item.get(key) is not None:
            value = item[key]
            break
    entries = list(value) if isinstance(value, (list, tuple)) else [value]
    return [text for text in (_clean(entry) for entry in entries) if text]


def _primitive_needs_description_fallback(primitive: dict[str, Any]) -> bool:
    descriptions = _description_values(primitive, ("primitive_description", "description", "name"))
    object_groups = _top_level_group_count(primitive.get("object_id", primitive.get("object_ids")))
    return not descriptions or len(descriptions) < object_groups


def _skill_description_fallback(skills_by_idx: dict[int, dict[str, Any]], skill_indices: Iterable[int]) -> list[str]:
    values: list[str] = []
    for skill_idx in skill_indices:
        skill = skills_by_idx.get(skill_idx)
        if skill is None:
            continue
        for value in _description_values(skill, ("skill_description", "description", "name")):
            if value not in values:
                values.append(value)
    return values


def _fallback_metadata(
    primitive: dict[str, Any],
    primitive_idx: int,
    skill_indices: list[int],
    skills_by_idx: dict[int, dict[str, Any]],
) -> tuple[list[str], str] | None:
    """Provide a high-level text fallback without replacing primitive context."""
    if not _primitive_needs_description_fallback(primitive):
        return None
    descriptions = _skill_description_fallback(skills_by_idx, skill_indices)
    if descriptions:
        return descriptions, "skill_annotation_description_fallback"
    return [f"execute primitive {primitive_idx}"], "canonical_primitive_fallback"


def preflight_annotations(annotations_root: Path, *, limit: int | None = None) -> dict[str, Any]:
    """Parse and render annotation schemas without emitting any frame labels."""
    files = _annotation_files(annotations_root)
    if limit is not None:
        files = files[:max(0, limit)]
    errors: list[dict[str, str]] = []
    fallback_count = 0
    placeholder_count = 0
    render_errors: list[dict[str, str]] = []
    for path in files:
        try:
            annotation = json.loads(path.read_text(encoding="utf-8"))
            skills = _as_skill_list(annotation.get("skill_annotation", annotation.get("skills", [])) or [])
            skills_by_idx: dict[int, dict[str, Any]] = {}
            for position, skill in enumerate(skills):
                try:
                    skills_by_idx[int(skill.get("skill_idx", position))] = skill
                except (TypeError, ValueError):
                    continue
                rendered_skill = skill_intent(skill)
                placeholder_count += int("execute current skill" in rendered_skill)
                if _needs_canonical_annotation(
                    skill.get("skill_description", skill.get("description", skill.get("name"))),
                    skill.get("object_id", skill.get("object_ids")),
                    skill.get("spatial_prefix"),
                    skill.get("memory_prefix"),
                    skill.get("manipulating_object_id"),
                ) and not rendered_skill.startswith("Descriptions="):
                    render_errors.append({"path": str(path), "error": "non-alignable skill was not rendered canonically"})
            primitives = _as_skill_list(annotation.get("primitive_annotation"))
            for position, primitive in enumerate(primitives):
                primitive_idx = int(primitive.get("primitive_idx", position))
                raw_indices = primitive.get("skill_idxes", primitive.get("skill_indices", primitive.get("skill_idx", primitive.get("skill_index", []))))
                if not isinstance(raw_indices, (list, tuple, set)):
                    raw_indices = [raw_indices]
                skill_indices: list[int] = []
                for value in raw_indices:
                    try:
                        skill_indices.append(int(float(value)))
                    except (TypeError, ValueError):
                        continue
                metadata = _fallback_metadata(primitive, primitive_idx, skill_indices, skills_by_idx)
                fallback_count += int(metadata is not None)
                primitive_intent(primitive, fallback_descriptions=metadata[0] if metadata else None)
        except (OSError, ValueError, TypeError, KeyError, IndexError, json.JSONDecodeError) as exc:
            errors.append({"path": str(path), "error": f"{type(exc).__name__}: {exc}"})
    return {
        "checked": len(files),
        "fallback_count": fallback_count,
        "placeholder_count": placeholder_count,
        "render_errors": render_errors,
        "errors": errors,
    }


def _interval_from_item(item: dict[str, Any]) -> Any:
    return item.get("frame_duration", item.get("duration"))


def annotate_episode(ep: Episode, annotation: dict[str, Any], warnings: list[dict[str, Any]], *, action_horizon: int = 32, causal: bool = False, planner_stride: int = 128, memory_corruption_every: int = 4) -> list[dict[str, Any]]:
    if planner_stride < 1 or memory_corruption_every < 0:
        raise ValueError("planner_stride must be positive and memory_corruption_every nonnegative")
    metadata = annotation.get("meta_data", {}) or {}
    valid_raw = annotation.get("valid_duration", metadata.get("valid_duration", [0, ep.length]))
    valid_original = _range(valid_raw)
    if valid_original is None:
        warnings.append({"episode_index": ep.episode_index, "type": "invalid_interval", "type_context": "valid_duration", "value": valid_raw})
        return []
    valid_start, valid_end = valid_original
    if valid_start > valid_end:
        warnings.append({"episode_index": ep.episode_index, "type": "reversed_interval", "type_context": "valid_duration", "original": [valid_start, valid_end], "dropped": True, "label_confidence": 0.0})
        return []
    raw_valid_span = valid_end - valid_start
    if raw_valid_span > 0 and ep.length > 0 and ep.length / raw_valid_span < 0.5:
        warnings.append({"episode_index": ep.episode_index, "type": "severe_truncation", "reason": "observed_length_less_than_half_valid_span", "observed_length": ep.length, "annotation_valid_span": raw_valid_span})
        return []
    # A trimmed episode commonly has length == valid_end-valid_start.  Keep
    # annotation coordinates in that case and translate them to local frames;
    # otherwise clip directly to the episode's frame range.
    offset = -valid_start if valid_end - valid_start == ep.length else 0
    if offset:
        mapped = (valid_start + offset, valid_end + offset)
        if mapped[0] < 0 or mapped[1] > ep.length:
            clipped = (max(0, mapped[0]), min(ep.length, mapped[1]))
            warnings.append({"episode_index": ep.episode_index, "type": "clipped_interval", "type_context": "valid_duration", "original": list(mapped), "clipped": list(clipped)})
            mapped = clipped
        valid_start, valid_end = mapped
        # Convert back to raw coordinates for clipping skill intervals below.
        clip_start, clip_end = valid_start - offset, valid_end - offset
    else:
        clipped = (max(0, min(ep.length, valid_start)), max(0, min(ep.length, valid_end)))
        if clipped != (valid_start, valid_end):
            warnings.append({"episode_index": ep.episode_index, "type": "clipped_interval", "type_context": "valid_duration", "original": [valid_start, valid_end], "clipped": list(clipped)})
        valid_start, valid_end = clipped
        clip_start, clip_end = valid_start, valid_end
    if valid_end <= valid_start:
        warnings.append({"episode_index": ep.episode_index, "type": "empty_interval", "type_context": "valid_duration", "interval": [valid_start, valid_end]})
        return []
    skills_raw = _as_skill_list(annotation.get("skill_annotation", annotation.get("skills", [])) or [])
    if not skills_raw and annotation.get("skill_annotation") not in (None, [], {}):
        warnings.append({"episode_index": ep.episode_index, "type": "invalid_skills"}); return []
    skills_by_idx: dict[int, dict[str, Any]] = {}
    for skill_pos, raw_skill in enumerate(skills_raw):
        try:
            skills_by_idx[int(raw_skill.get("skill_idx", skill_pos))] = raw_skill
        except (TypeError, ValueError):
            warnings.append({"episode_index": ep.episode_index, "type": "invalid_skill_idx", "skill_position": skill_pos})
    # Low-level labels come from the skill_annotation collection.  One row
    # segment per skill is intentional; primitive_annotation is reserved for
    # high-level planner labels below.
    low_records: list[dict[str, Any]] = []
    for skill_pos, raw_skill in enumerate(skills_raw):
        skill_idx = int(raw_skill.get("skill_idx", skill_pos))
        interval_value = _interval_from_item(raw_skill)
        skill_intervals = _clip_intervals(interval_value, clip_start, clip_end, warnings, {"episode_index": ep.episode_index, "skill_idx": skill_idx, "type_context": "skill"}) if interval_value is not None else []
        if not skill_intervals:
            primitives = _primitives(raw_skill)
            for primitive in primitives:
                skill_intervals.extend(_clip_intervals(_interval_from_item(primitive), clip_start, clip_end, warnings, {"episode_index": ep.episode_index, "skill_idx": skill_idx, "type_context": "skill_primitive"}))
        if not skill_intervals:
            warnings.append({"episode_index": ep.episode_index, "skill_idx": skill_idx, "type": "missing_skill_interval"}); continue
        merged_intervals: list[tuple[int, int]] = []
        for start, end in sorted(skill_intervals):
            if merged_intervals and start <= merged_intervals[-1][1]:
                merged_intervals[-1] = (merged_intervals[-1][0], max(merged_intervals[-1][1], end))
            else:
                merged_intervals.append((start, end))
        for interval_id, (start, end) in enumerate(merged_intervals):
            low_records.append({"_start": max(0, min(ep.length, start + offset)), "_end": max(0, min(ep.length, end + offset)), "_interval_id": interval_id, "_skill_idx": skill_idx, "_primitive_idx": int(raw_skill.get("primitive_idx", 0)), "_skill": raw_skill})
    low_records = [row for row in low_records if row["_end"] > row["_start"]]
    low_records.sort(key=lambda row: (row["_start"], row["_skill_idx"], row["_primitive_idx"], row.get("_interval_id", 0)))
    if not low_records:
        warnings.append({"episode_index": ep.episode_index, "type": "no_usable_skills"}); return []
    # High-level boundaries are driven by top-level primitive_annotation.  A
    # legacy fallback to skill segments keeps existing datasets readable when
    # no primitive collection was written.
    primitive_raw = annotation.get("primitive_annotation")
    primitive_items = _as_skill_list(primitive_raw) if primitive_raw is not None else []
    if causal:
        for primitive in primitive_items:
            interval = _interval_from_item(primitive)
            if isinstance(interval, (list, tuple)) and any(isinstance(value, (list, tuple, dict)) for value in interval):
                # Three released can_meat annotations attach the second jar's
                # navigation to the first jar and have [start,[substart,end]].
                # An envelope makes a syntactically valid but semantically
                # wrong planner label. Quarantine instead of inventing GT.
                warnings.append({"episode_index": ep.episode_index, "type": "ambiguous_primitive_interval",
                                 "primitive_idx": primitive.get("primitive_idx"), "value": interval,
                                 "reason": "nested primitive interval requires reviewed skill-parent identity",
                                 "dropped_episode": True})
                return []
    high_records: list[dict[str, Any]] = []
    if not primitive_items:
        # Older annotations can contain only skill_annotation.  Keep those
        # samples usable, but make the weaker supervision explicit instead of
        # pretending it originated in primitive_annotation.
        high_records = [{**row, "_fallback": True} for row in low_records]
        warnings.append({"episode_index": ep.episode_index, "type": "primitive_annotation_fallback", "reason": "missing_or_empty_primitive_annotation"})
    else:
        for primitive_pos, primitive in enumerate(primitive_items):
            raw_skill_indices = primitive.get("skill_idxes", primitive.get("skill_indices", primitive.get("skill_idx", primitive.get("skill_index"))))
            if raw_skill_indices in (None, ""):
                warnings.append({"episode_index": ep.episode_index, "primitive_idx": int(primitive.get("primitive_idx", primitive_pos)), "type": "missing_primitive_skill_idxes"})
                continue
            if not isinstance(raw_skill_indices, (list, tuple, set)):
                raw_skill_indices = [raw_skill_indices]
            try:
                skill_indices = [int(float(value)) for value in raw_skill_indices]
            except (TypeError, ValueError):
                warnings.append({"episode_index": ep.episode_index, "primitive_idx": int(primitive.get("primitive_idx", primitive_pos)), "type": "invalid_primitive_skill_idxes", "value": raw_skill_indices})
                continue
            primitive_idx = int(primitive.get("primitive_idx", primitive_pos))
            fallback_metadata = _fallback_metadata(primitive, primitive_idx, skill_indices, skills_by_idx)
            if fallback_metadata is not None:
                fallback_descriptions, intent_provenance = fallback_metadata
                warnings.append({
                    "episode_index": ep.episode_index,
                    "primitive_idx": primitive_idx,
                    "skill_idxes": skill_indices,
                    "type": "primitive_description_fallback",
                    "provenance": intent_provenance,
                    "label_confidence": 0.5,
                })
            interval = _clip(_interval_from_item(primitive), clip_start, clip_end, warnings, {"episode_index": ep.episode_index, "skill_idx": skill_indices[0] if skill_indices else None, "primitive_idx": int(primitive.get("primitive_idx", primitive_pos)), "type_context": "primitive_annotation"})
            if interval is None: continue
            for skill_idx in dict.fromkeys(skill_indices):
                high_records.append({
                    **primitive,
                    "_start": max(0, min(ep.length, interval[0] + offset)),
                    "_end": max(0, min(ep.length, interval[1] + offset)),
                    "_skill_idx": skill_idx,
                    "_primitive_idx": primitive_idx,
                    "_skill": primitive,
                    "_primitive_skill_idxes": skill_indices,
                    "_intent_fallback_descriptions": fallback_metadata[0] if fallback_metadata else None,
                    "_intent_provenance": fallback_metadata[1] if fallback_metadata else "primitive_annotation",
                    "_intent_confidence": 0.5 if fallback_metadata else 0.95,
                })
    high_records = [row for row in high_records if row["_end"] > row["_start"]]
    high_records.sort(key=lambda row: (row["_start"], row["_skill_idx"], row["_primitive_idx"]))
    if not high_records:
        warnings.append({"episode_index": ep.episode_index, "type": "no_usable_primitive_annotation"})
    for low in low_records:
        candidates = [
            primitive for primitive in high_records
            if primitive["_skill_idx"] == low["_skill_idx"]
            and primitive["_start"] < low["_end"] and low["_start"] < primitive["_end"]
        ]
        match = max(candidates, key=lambda primitive: (min(low["_end"], primitive["_end"]) - max(low["_start"], primitive["_start"]), -primitive["_start"])) if candidates else None
        if match is None:
            low["_primitive_idx"] = None
            warnings.append({"episode_index": ep.episode_index, "skill_idx": low["_skill_idx"], "type": "missing_low_primitive_match"})
            low["_primitive_skill_idxes"] = []
        else:
            low["_primitive_idx"] = match["_primitive_idx"]
            low["_primitive_skill_idxes"] = match.get("_primitive_skill_idxes", [low["_skill_idx"]])
            low["_parent_primitive"] = match
    high_segments = list(high_records)
    labels: list[dict[str, Any]] = []
    seen: set[tuple[int, str]] = set()
    def emit(row: dict[str, Any], branch: str, frame: int, *, status: str = "CONTINUE", update: str = "") -> None:
        key = (frame, branch)
        if key in seen:
            warnings.append({"episode_index": ep.episode_index, "frame_index": frame, "memlite_branch": branch, "type": "overlap"}); return
        seen.add(key)
        item = dict(row); item.update(frame_index=frame, memlite_branch=branch, intent_status=status, memory_update=update, segment_end=row["_end"], action_horizon_end=min(row["_end"], frame + max(1, action_horizon)))
        labels.append(item)
    # Memory state is event-time based.  A primitive is completed only after
    # its own end; overlapping records remain in Active at later starts.
    canonical_records: list[dict[str, Any]] = []
    canonical_seen: set[str] = set()
    for record in high_records:
        identity = _primitive_identity(record)
        if identity not in canonical_seen:
            canonical_seen.add(identity)
            canonical_records.append(record)
    canonical_records.sort(key=lambda row: (row["_start"], row["_end"], row["_primitive_idx"], row["_skill_idx"]))

    def state_at(frame: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        completed = [record for record in canonical_records if record["_end"] <= frame]
        active_records = [record for record in canonical_records if record["_start"] < frame < record["_end"]]
        return completed, active_records

    terminal_end = max((row["_end"] for row in low_records), default=valid_start)
    terminal_frame = terminal_end - 1
    starts: dict[int, list[dict[str, Any]]] = {}
    for record in high_segments:
        # A planner DONE row occupies this frame.  Its state represents the
        # final post-action state at terminal_end, not a primitive start.
        if record["_start"] != terminal_frame:
            starts.setdefault(record["_start"], []).append(record)
    event_starts = sorted(starts)
    for event_index, start in enumerate([] if causal else event_starts):
        parallel = starts[start]
        parallel.sort(key=lambda row: (row["_skill_idx"], row["_primitive_idx"]))
        if len(parallel) > 1:
            warnings.append({"episode_index": ep.episode_index, "frame_index": start, "memlite_branch": "high", "type": "merged_parallel_skills", "skill_idxes": [row["_skill_idx"] for row in parallel]})
        record = parallel[0]
        current_records: list[dict[str, Any]] = []
        seen_current: set[str] = set()
        for item in parallel:
            key = _primitive_identity(item)
            if key not in seen_current:
                seen_current.add(key); current_records.append(item)
        fallback = all(item.get("_fallback") for item in parallel)
        intents = list(dict.fromkeys(
            skill_intent(item["_skill"])
            if item.get("_fallback")
            else primitive_intent(item, fallback_descriptions=item.get("_intent_fallback_descriptions"))
            for item in parallel
        ))
        completed, active_records = state_at(start)
        next_frame = event_starts[event_index + 1] if event_index + 1 < len(event_starts) else terminal_end
        next_completed, next_active = state_at(next_frame)
        memory = make_memory(str(ep.task_index), completed, active_records)
        memory_update = make_memory(str(ep.task_index), next_completed, next_active)
        confidence = min(float(item.get("_intent_confidence", 0.95)) for item in parallel)
        provenance = sorted({item.get("_intent_provenance", "skill_annotation_fallback" if item.get("_fallback") else "primitive_annotation") for item in parallel})
        base = {"episode_index": ep.episode_index, "task_index": ep.task_index, "frame_index": start, "memlite_branch": "high", "memory": memory, "intent": " AND ".join(intents), "intent_status": "CONTINUE", "skill_idx": record["_skill_idx"], "skill_idxes": [item["_skill_idx"] for item in parallel], "primitive_idx": record["_primitive_idx"], "primitive_skill_idxes": sorted({idx for item in parallel for idx in item.get("_primitive_skill_idxes", [item["_skill_idx"]])}), "label_source": "skill_annotation_fallback" if fallback else "primitive_annotation", "label_confidence": confidence, "intent_provenance": " AND ".join(provenance), "annotation_path": ep.annotation_path, "skill_start": start, "skill_end": min(item["_end"] for item in parallel), "segment_start": start, "segment_end": min(item["_end"] for item in parallel), "action_horizon_end": min(item["_end"] for item in parallel), "_end": min(item["_end"] for item in parallel), "memory_update": memory_update}
        emit(base, "high", start, update=base["memory_update"])
    # Low labels are emitted once per frame. Parallel skill intervals are
    # merged into a deterministic multi-intent row instead of dropping one
    # object's supervision.
    start_events: dict[int, list[dict[str, Any]]] = {}
    end_events: dict[int, list[dict[str, Any]]] = {}
    for record in low_records:
        start_events.setdefault(record["_start"], []).append(record)
        end_events.setdefault(record["_end"], []).append(record)
    active: list[dict[str, Any]] = []
    skill_text_cache = {id(row): skill_intent(row["_skill"]) for row in low_records}
    parent_text_cache = {id(row): primitive_intent(row, fallback_descriptions=row.get("_intent_fallback_descriptions"))
                         for row in high_records if not row.get("_fallback")}
    previous_prior_ids = None
    previous_low_memory = ""
    covered_frames: set[int] = set()
    frame_lo, frame_hi = min(valid_start, min(start_events)), max(valid_end, max(end_events))
    for frame in range(frame_lo, frame_hi):
        if frame in end_events:
            ended = set(id(item) for item in end_events[frame])
            active = [item for item in active if id(item) not in ended]
        active.extend(start_events.get(frame, []))
        if not active:
            continue
        covered_frames.add(frame)
        parallel = sorted(active, key=lambda row: (row["_skill_idx"], row["_primitive_idx"] if row["_primitive_idx"] is not None else -1))
        if len(parallel) > 1:
            warnings.append({"episode_index": ep.episode_index, "frame_index": frame, "memlite_branch": "low", "type": "merged_parallel_skills", "skill_idxes": [row["_skill_idx"] for row in parallel]})
        record = parallel[0]
        prior = [item for item in high_records if item["_end"] <= frame]
        def low_intent(item):
            parents = [parent for parent in high_records
                       if parent["_skill_idx"] == item["_skill_idx"]
                       and parent["_start"] <= frame < parent["_end"]]
            if causal and parents and not all(parent.get("_fallback") for parent in parents):
                return " AND ".join(dict.fromkeys(
                    parent_text_cache.get(id(parent), skill_text_cache[id(item)])
                    for parent in parents))
            return skill_text_cache[id(item)]
        intents = list(dict.fromkeys(low_intent(item) for item in parallel))
        prior_ids = tuple(id(item) for item in prior)
        if prior_ids != previous_prior_ids:
            previous_low_memory = make_memory(str(ep.task_index), prior)
            previous_prior_ids = prior_ids
        low_memory = previous_low_memory
        subsegment_end = min(item["_end"] for item in parallel)
        if causal:
            active_skills = {item["_skill_idx"] for item in parallel}
            primitive_boundaries = [boundary for parent in high_records if parent["_skill_idx"] in active_skills
                                    for boundary in (parent["_start"], parent["_end"]) if frame < boundary]
            subsegment_end = min([subsegment_end] + primitive_boundaries)
        base = {"episode_index": ep.episode_index, "task_index": ep.task_index, "frame_index": frame, "memlite_branch": "low", "memory": low_memory, "intent": " AND ".join(intents), "memory_update": low_memory, "intent_status": "CONTINUE", "skill_idx": record["_skill_idx"], "skill_idxes": [item["_skill_idx"] for item in parallel], "primitive_idx": record["_primitive_idx"], "primitive_skill_idxes": sorted({idx for item in parallel for idx in item.get("_primitive_skill_idxes", [])}), "interval_id": record.get("_interval_id"), "label_source": "skill_annotation", "label_confidence": 0.95, "intent_provenance": "skill_annotation", "annotation_path": ep.annotation_path, "skill_start": min(item["_start"] for item in parallel), "skill_end": subsegment_end, "segment_start": min(item["_start"] for item in parallel), "segment_end": subsegment_end, "subsegment_end": subsegment_end, "action_horizon_end": subsegment_end, "_end": subsegment_end}
        emit(base, "low", frame, status="CONTINUE", update=base["memory_update"])
    # Explicitly expose gaps between valid_duration and skill coverage.
    covered = covered_frames
    gaps = [frame for frame in range(valid_start, valid_end) if frame not in covered]
    if gaps:
        start = previous = gaps[0]
        for frame in gaps[1:] + [None]:
            if frame is None or frame != previous + 1:
                warnings.append({"episode_index": ep.episode_index, "type": "gap", "start": start, "end": previous + 1, "frames": previous - start + 1})
                if frame is not None: start = frame
            previous = frame if frame is not None else previous
    if terminal_end < valid_end:
        warnings.append({"episode_index": ep.episode_index, "type": "severe_truncation", "reason": "uncovered_terminal_segment", "covered_end": terminal_end, "valid_end": valid_end, "frames": valid_end - terminal_end})
        return []
    if causal and labels:
        # The planner revises the LAST OBSERVED summary at the CURRENT
        # observation. No target refers to the next anchor or end-1 future.
        low_labels = sorted(labels, key=lambda row: row["frame_index"])
        previous_memory = make_memory(str(ep.task_index), [])
        previous_intent = "None"
        previous_frame = -1
        previous_anchor = None
        last_low_intent = None
        high_labels = []
        for row in low_labels:
            frame = row["frame_index"]
            row.update(previous_intent="", memory_input_frame=frame,
                       memory_target_frame=frame, memory_input_corruption="none",
                       schema_version=5, label_source="canonical_primitive_conditioned_action")
            if previous_anchor is not None and row["intent"] == last_low_intent and frame - previous_anchor < planner_stride:
                continue
            completed, active_records = state_at(frame)
            update = make_memory(str(ep.task_index), completed, active_records)
            memory, corruption = previous_memory, "none"
            ordinal = len(high_labels)
            if memory_corruption_every and ordinal and ordinal % memory_corruption_every == 0:
                # Counterfactual TEXT input only. Never fabricate failure
                # observations, successful recoveries, or reward labels.
                if (ordinal // memory_corruption_every) % 2:
                    # Only overclaim a previously KNOWN intent. Using all
                    # future ground-truth primitives even as a negative would
                    # leak their object identities and full task ordering.
                    known_completed = [item for item in canonical_records if item["_end"] <= previous_frame]
                    known_unfinished = [item for item in canonical_records
                                        if item["_start"] <= previous_frame and item["_end"] > frame]
                    if known_unfinished:
                        memory = make_memory(str(ep.task_index), known_completed + known_unfinished)
                        corruption = "synthetic_text_premature_known_intent_completion"
                    else:
                        memory = make_memory(str(ep.task_index), [])
                        corruption = "synthetic_text_missing_history"
                else:
                    memory = make_memory(str(ep.task_index), [])
                    corruption = "synthetic_text_missing_history"
                if memory == previous_memory:
                    corruption = "none"
            high_labels.append({**row, "memlite_branch": "high", "memory": memory,
                                "previous_intent": previous_intent, "memory_update": update,
                                "memory_input_frame": previous_frame, "memory_target_frame": frame,
                                "memory_input_corruption": corruption,
                                "label_source": "causal_observed_primitive_state"})
            previous_memory, previous_intent = update, row["intent"]
            previous_frame = previous_anchor = frame
            last_low_intent = row["intent"]
        if terminal_end < ep.length:
            completed, active_records = state_at(terminal_end)
            high_labels.append({**low_labels[-1], "frame_index": terminal_end,
                                "memlite_branch": "high", "memory": previous_memory,
                                "previous_intent": previous_intent, "intent": "Task complete",
                                "memory_update": make_memory(str(ep.task_index), completed, active_records),
                                "intent_status": "DONE", "label_source": "observed_annotation_terminal",
                                "intent_provenance": "observed_annotation_terminal",
                                "memory_input_frame": previous_frame, "memory_target_frame": terminal_end,
                                "memory_input_corruption": "none", "skill_idx": None,
                                "skill_idxes": [], "primitive_idx": None, "primitive_skill_idxes": [],
                                "skill_start": terminal_end, "skill_end": terminal_end,
                                "segment_start": terminal_end, "segment_end": terminal_end,
                                "action_horizon_end": terminal_end})
        else:
            warnings.append({"episode_index": ep.episode_index, "type": "no_observed_terminal_frame",
                             "terminal_end": terminal_end, "length": ep.length})
        labels.extend(high_labels)
        for row in labels:
            row.pop("_end", None)
        return labels
    if labels:
        low_labels = [row for row in labels if row["memlite_branch"] == "low"]
        if low_labels:
            final = max(low_labels, key=lambda row: row["frame_index"])
            final["intent_status"] = "DONE"
            terminal_frame = final["frame_index"]
            # A terminal planner target is separate from the last primitive's
            # start: completion is supervised only after the final low frame.
            labels[:] = [row for row in labels if not (row["memlite_branch"] == "high" and row["frame_index"] == terminal_frame)]
            terminal_completed, terminal_active = state_at(terminal_end)
            terminal_memory = make_memory(str(ep.task_index), terminal_completed, terminal_active)
            labels.append({
                "episode_index": ep.episode_index,
                "task_index": ep.task_index,
                "frame_index": terminal_frame,
                "memlite_branch": "high",
                "memory": terminal_memory,
                "intent": "Task complete",
                "memory_update": terminal_memory,
                "intent_status": "DONE",
                "skill_idx": None,
                "skill_idxes": [],
                "primitive_idx": None,
                "primitive_skill_idxes": [],
                "label_source": "derived_terminal",
                "label_confidence": 0.95,
                "intent_provenance": "derived_terminal",
                "annotation_path": ep.annotation_path,
                "skill_start": terminal_frame,
                "skill_end": terminal_frame,
                "segment_start": terminal_frame,
                "segment_end": terminal_frame,
                "action_horizon_end": terminal_frame,
            })
        for row in labels: row.pop("_end", None)
    return labels

def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

def _write_sidecar_compat(labels: list[dict[str, Any]] | Path, output_root: Path, overwrite: bool) -> str:
    meta = output_root / "meta"; meta.mkdir(parents=True, exist_ok=True)
    parquet, jsonl = meta / "memlite_annotations.parquet", meta / "memlite_annotations.jsonl"
    if not overwrite and (parquet.exists() or jsonl.exists()): raise FileExistsError(f"sidecar exists below {meta}; pass --overwrite to replace it")
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
        if isinstance(labels, Path):
            batch: list[dict[str, Any]] = []
            writer = None
            keys: list[str] | None = None
            with labels.open(encoding="utf-8") as fh:
                for line in fh:
                    if line.strip(): batch.append(json.loads(line))
                    if len(batch) < 4096: continue
                    if keys is None: keys = sorted({key for row in batch for key in row})
                    table = pa.Table.from_pydict({key: [row.get(key) for row in batch] for key in keys})
                    if writer is None: writer = pq.ParquetWriter(parquet, table.schema, compression="zstd")
                    writer.write_table(table); batch = []
            if batch:
                if keys is None: keys = sorted({key for row in batch for key in row})
                table = pa.Table.from_pydict({key: [row.get(key) for row in batch] for key in keys})
                if writer is None: writer = pq.ParquetWriter(parquet, table.schema, compression="zstd")
                writer.write_table(table)
            if writer is not None: writer.close()
        else:
            keys = sorted({key for row in labels for key in row})
            pq.write_table(pa.Table.from_pydict({key: [row.get(key) for row in labels] for key in keys}), parquet, compression="zstd")
        return str(parquet)
    except ImportError:
        with jsonl.open("w", encoding="utf-8") as fh:
            if isinstance(labels, Path):
                fh.write(labels.read_text(encoding="utf-8"))
            else:
                for row in labels: fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        return str(jsonl)

def build(args: argparse.Namespace) -> dict[str, Any]:
    annotations_root = Path(args.annotations_root).expanduser().resolve()
    episodes_arg = getattr(args, "episodes", None)
    input_root = Path(args.input_root).expanduser().resolve() if getattr(args, "input_root", None) else None
    episodes_path = Path(episodes_arg).expanduser().resolve() if episodes_arg else None
    output_root = Path(args.output_root).expanduser().resolve()
    action_horizon = max(1, int(getattr(args, "action_horizon", 32)))
    episodes = load_episodes(episodes_path, annotations_root, input_root=input_root, episode_length=getattr(args, "episode_length", None))
    task_filter = {str(value) for value in (getattr(args, "task_filter", None) or [])}
    if task_filter:
        episodes = [ep for ep in episodes if str(ep.task_index) in task_filter]
    overwrite = bool(getattr(args, "overwrite", False))
    if output_root.exists() and any(output_root.iterdir()) and not overwrite:
        raise FileExistsError(f"output directory is non-empty: {output_root}; pass --overwrite to replace it")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_root.name}.", dir=str(output_root.parent)))
    labels_path = staging / "labels.jsonl"
    labels_count = 0
    branch_counts: dict[str, int] = {"high": 0, "low": 0}
    terminal_high_rows = 0
    warning_count = overlap_count = gap_count = gap_frames_total = 0
    manifest_episodes: list[dict[str, Any]] = []
    warnings_path = staging / "warnings.json"
    with labels_path.open("w", encoding="utf-8") as labels_fh, warnings_path.open("w", encoding="utf-8") as warnings_fh:
        warnings_fh.write("[\n")
        first_warning = True
        for ep in sorted(episodes, key=lambda row: row.episode_index):
            ep_warnings: list[dict[str, Any]] = []
            try:
                path = _resolve_annotation(ep, annotations_root, input_root); ep.annotation_path = str(path)
                annotation = json.loads(path.read_text(encoding="utf-8"))
                ep_labels = annotate_episode(ep, annotation, ep_warnings, action_horizon=action_horizon,
                                             causal=getattr(args, "causal", False),
                                             planner_stride=getattr(args, "planner_stride", 128),
                                             memory_corruption_every=getattr(args, "memory_corruption_every", 4))
                for row in ep_labels:
                    labels_fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                    labels_count += 1; branch_counts[row["memlite_branch"]] = branch_counts.get(row["memlite_branch"], 0) + 1
                    terminal_high_rows += int(row["memlite_branch"] == "high" and row["intent_status"] == "DONE")
                covered_frames = len({row["frame_index"] for row in ep_labels if row["memlite_branch"] == "low"})
                gap_frames = sum(int(item.get("frames", 0)) for item in ep_warnings if item.get("type") == "gap")
                invalid = any(
                    item.get("type") == "severe_truncation"
                    or item.get("type") == "ambiguous_primitive_interval"
                    or (item.get("type") == "reversed_interval" and item.get("type_context") == "valid_duration")
                    for item in ep_warnings
                )
                manifest_episodes.append({"episode_index": ep.episode_index, "task_index": ep.task_index, "length": ep.length, "annotation_path": str(path), "status": "invalid_episode" if invalid else ("ok" if ep_labels else "skipped"), "labels": len(ep_labels), "covered_frames": covered_frames, "gap_frames": gap_frames, "valid_frames": covered_frames + gap_frames})
                if len(manifest_episodes) % 25 == 0:
                    print(json.dumps({"episodes_processed": len(manifest_episodes), "episodes_total": len(episodes),
                                      "labels_written": labels_count}), flush=True)
            except (OSError, ValueError, json.JSONDecodeError, TypeError) as exc:
                ep_warnings.append({"episode_index": ep.episode_index, "task_index": ep.task_index, "type": "episode_error", "message": f"{type(exc).__name__}: {exc}"})
                manifest_episodes.append({"episode_index": ep.episode_index, "task_index": ep.task_index, "length": ep.length, "status": "skipped", "labels": 0})
            for warning in ep_warnings:
                if not first_warning: warnings_fh.write(",\n")
                warnings_fh.write(json.dumps(warning, ensure_ascii=False, sort_keys=True))
                first_warning = False
                warning_count += 1
                overlap_count += int(warning.get("type") == "merged_parallel_skills")
                gap_count += int(warning.get("type") == "gap")
                gap_frames_total += int(warning.get("frames", 0)) if warning.get("type") == "gap" else 0
        warnings_fh.write("\n]\n")
    stats = {
        "schema_version": SCHEMA_VERSION,
        "episodes": len(episodes),
        "labels": labels_count,
        "branch_counts": branch_counts,
        "warning_count": warning_count,
        "overlap_count": overlap_count,
        "gap_count": gap_count,
        "gap_frames": gap_frames_total,
        "covered_frames": sum(int(item.get("covered_frames", 0)) for item in manifest_episodes),
        "valid_frames": sum(int(item.get("valid_frames", 0)) for item in manifest_episodes),
        "action_horizon": action_horizon,
        "terminal_high_rows": terminal_high_rows,
    }
    manifest = {"schema_name": "memlite_annotations", "schema_version": SCHEMA_VERSION, "memory_schema": "State(t)=Task=<task_index>; Completed=<primitive end<=t>; Active=<primitive start<t<end> when non-empty. High memory_update=State(next high event); terminal=State(final end).", "episodes": manifest_episodes, "action_horizon": action_horizon, "episodes_file": str(episodes_path) if episodes_path else None, "status_vocabulary": ["CONTINUE", "DONE"], "status_supervision": "success demonstrations only; FAILED and REPLAN are not fabricated", "labels_file": str(output_root / "labels.jsonl"), "stats_file": str(output_root / "stats.json"), "warnings_file": str(output_root / "warnings.json")}
    if getattr(args, "causal", False):
        stats["schema_version"] = manifest["schema_version"] = 5
        manifest["memory_schema"] = "Input=previous observed summary (explicitly tagged synthetic text corruption on selected rows); target=State(current frame), never future state. DONE requires a post-end observation."
        manifest["planner_stride"] = getattr(args, "planner_stride", 128)
        manifest["memory_corruption_every"] = getattr(args, "memory_corruption_every", 4)
    _write_json(staging / "stats.json", stats); _write_json(staging / "manifest.json", manifest)
    sidecar_staging = _write_sidecar_compat(labels_path, staging, True)
    summary = {"rows": labels_count, "episodes": len(episodes), "status_counts": {"ok": sum(e["status"] == "ok" for e in manifest_episodes), "skipped": sum(e["status"] != "ok" for e in manifest_episodes)}, "branch_counts": branch_counts, "sidecar": str(output_root / Path(sidecar_staging).relative_to(staging)), "manifest": manifest_episodes}
    _write_json(staging / "meta" / "memlite_manifest.json", summary)
    # Publish files only after all labels and metadata have been successfully
    # written.  A new output directory is renamed atomically; overwrite mode
    # atomically replaces each existing file/directory.
    if not output_root.exists():
        os.replace(staging, output_root)
    else:
        output_root.mkdir(parents=True, exist_ok=True)
        for child in staging.iterdir():
            target = output_root / child.name
            if target.is_dir() and target.exists(): shutil.rmtree(target)
            os.replace(child, target)
        staging.rmdir()
    return summary

def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations-root", required=True)
    parser.add_argument("--episodes", "--episodes-path", "--episodes-file", "--episodes-jsonl", "--episodes-csv", dest="episodes")
    parser.add_argument("--episode-length", type=int)
    parser.add_argument("--action-horizon", type=int, default=32)
    parser.add_argument("--causal", action="store_true")
    parser.add_argument("--planner-stride", type=int, default=128)
    parser.add_argument("--memory-corruption-every", type=int, default=4)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--input-root")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--task-filter", nargs="*")
    return parser

def main(argv: list[str] | None = None) -> int:
    build(make_parser().parse_args(argv)); return 0

if __name__ == "__main__": raise SystemExit(main())
