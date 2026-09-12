"""Lazy reader for frame-aligned MEM-Lite annotation sidecars.

The annotation builder writes either ``meta/memlite_annotations.parquet`` or
``labels.jsonl``. This reader keeps only a compact episode index and a bounded
LRU of decoded episodes in memory.
"""
from __future__ import annotations

from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Any, Dict, Optional
import json
import math

_BRANCH_ALIASES = {"planner": "high", "high_level": "high", "action": "low", "low_level": "low"}
_VALID_BRANCHES = {"high", "low"}
_INVALID_STATUS = {"INVALID", "GAP", "IGNORE", "MISSING"}


def _scalar(value: Any) -> Any:
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "numel") and value.numel() == 1:
        return value.item()
    if isinstance(value, (list, tuple)) and len(value) == 1:
        return _scalar(value[0])
    return value


def _int(value: Any) -> Optional[int]:
    value = _scalar(value)
    if value is None or value == "":
        return None
    try:
        number = float(value)
        return int(number) if math.isfinite(number) else None
    except (TypeError, ValueError):
        return None


def _text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value).strip()


def _bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def normalize_label(row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Normalize v2/v3/v4-compatible sidecar row fields."""
    episode = _int(row.get("episode_index", row.get("episode")))
    frame = _int(row.get("frame_index", row.get("frame")))
    if episode is None or frame is None or frame < 0:
        return None
    branch = _text(row.get("memlite_branch", row.get("branch"))).lower()
    branch = _BRANCH_ALIASES.get(branch, branch)
    status = _text(row.get("intent_status", row.get("status", "CONTINUE"))).upper()
    ignore = _bool(row.get("ignore", row.get("invalid", False)))
    ignore = ignore or _bool(row.get("gap", False)) or status in _INVALID_STATUS
    if branch not in _VALID_BRANCHES:
        ignore, branch = True, "invalid"
    out = dict(row)
    out.update(
        episode_index=episode,
        frame_index=frame,
        memlite_branch=branch,
        memory=_text(row.get("memory", row.get("prev_memory", ""))),
        intent=_text(row.get("intent", row.get("atomic_task", ""))),
        memory_update=_text(row.get("memory_update", row.get("updated_memory", ""))),
        intent_status=status or "CONTINUE",
        ignore=ignore,
    )
    for key in ("segment_end", "action_horizon_end", "skill_end", "primitive_end"):
        parsed = _int(row.get(key))
        if parsed is not None:
            out[key] = parsed
        else:
            out.pop(key, None)
    return out


class MemLiteSidecar:
    """Episode-indexed lazy sidecar lookup with bounded episode cache."""

    def __init__(self, path: str | Path, *, cache_episodes: int = 2):
        candidate = Path(path).expanduser()
        if candidate.is_dir():
            for name in ("meta/memlite_annotations.parquet", "meta/memlite_annotations.jsonl", "labels.jsonl"):
                probe = candidate / name
                if probe.exists():
                    candidate = probe
                    break
        if not candidate.exists():
            raise FileNotFoundError(f"MEM-Lite sidecar not found: {candidate}")
        self.path = candidate.resolve()
        self.cache_episodes = max(1, int(cache_episodes))
        self._cache: OrderedDict[int, Dict[int, list[Dict[str, Any]]]] = OrderedDict()
        self._episode_groups: dict[int, list[int]] = defaultdict(list)
        self._json_offsets: dict[int, list[int]] = defaultdict(list)
        self._branch_sampling_index: Optional[Dict[str, Any]] = None
        self._format = "parquet" if self.path.suffix.lower() == ".parquet" else "jsonl"
        self._parquet = None
        if self._format == "parquet":
            try:
                import pyarrow.parquet  # noqa: F401 - dependency check only
            except ImportError as exc:
                raise RuntimeError("pyarrow is required to read a parquet MEM-Lite sidecar") from exc
            self._index_parquet()
        else:
            self._index_jsonl()

    @property
    def episode_count(self) -> int:
        return len(self._episode_groups if self._format == "parquet" else self._json_offsets)

    def _open_parquet(self):
        """Open the parquet reader lazily in the current process.

        DataLoader's spawn workers pickle the Dataset/sidecar.  PyArrow's
        ParquetFile holds native file state and is not a reliable pickle
        payload, so the compact Python episode index is serialized instead and
        each worker opens its own reader on first access.
        """
        if self._format != "parquet":
            return None
        if self._parquet is None:
            import pyarrow.parquet as pq

            self._parquet = pq.ParquetFile(self.path)
        return self._parquet

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_parquet"] = None
        # Worker caches are only an optimization; do not balloon the spawn
        # payload by serializing decoded annotation text from the parent.
        state["_cache"] = OrderedDict()
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._parquet = None

    def _index_parquet(self) -> None:
        parquet = self._open_parquet()
        names = list(parquet.schema_arrow.names)
        if "episode_index" not in names:
            raise ValueError(f"sidecar {self.path} lacks episode_index")
        column_idx = names.index("episode_index")
        for group_idx in range(parquet.num_row_groups):
            statistics = parquet.metadata.row_group(group_idx).column(column_idx).statistics
            if statistics is not None and statistics.has_min_max and statistics.min == statistics.max:
                values = (statistics.min,)
            else:
                # Avoid a Python object per row if a third-party sidecar uses
                # unusually large row groups.  This is still only one integer
                # column, not decoded annotation text.
                values = parquet.read_row_group(
                    group_idx, columns=["episode_index"]
                ).column(0).to_numpy(zero_copy_only=False)
            for value in set(values):
                episode = _int(value)
                if episode is not None and group_idx not in self._episode_groups[episode]:
                    self._episode_groups[episode].append(group_idx)

    def _index_jsonl(self) -> None:
        offset = 0
        with self.path.open("rb") as fh:
            for raw in fh:
                length = len(raw)
                if raw.strip():
                    try:
                        row = json.loads(raw)
                    except json.JSONDecodeError:
                        row = None
                    if isinstance(row, dict):
                        episode = _int(row.get("episode_index", row.get("episode")))
                        if episode is not None:
                            self._json_offsets[episode].append(offset)
                offset += length

    def _cache_put(self, episode: int, rows: Dict[int, list[Dict[str, Any]]]) -> None:
        self._cache[episode] = rows
        self._cache.move_to_end(episode)
        while len(self._cache) > self.cache_episodes:
            self._cache.popitem(last=False)

    def _load_episode(self, episode: int) -> Dict[int, list[Dict[str, Any]]]:
        cached = self._cache.get(episode)
        if cached is not None:
            self._cache.move_to_end(episode)
            return cached
        rows: Dict[int, list[Dict[str, Any]]] = defaultdict(list)
        if self._format == "parquet":
            parquet = self._open_parquet()
            for group_idx in self._episode_groups.get(episode, []):
                table = parquet.read_row_group(group_idx)
                for row in table.to_pylist():
                    normalized = normalize_label(row)
                    if normalized is not None and normalized["episode_index"] == episode:
                        rows[normalized["frame_index"]].append(normalized)
        else:
            with self.path.open("rb") as fh:
                for offset in self._json_offsets.get(episode, []):
                    fh.seek(offset)
                    try:
                        row = json.loads(fh.readline())
                    except (json.JSONDecodeError, UnicodeDecodeError):
                        continue
                    normalized = normalize_label(row) if isinstance(row, dict) else None
                    if normalized is not None:
                        rows[normalized["frame_index"]].append(normalized)
        result = dict(rows)
        self._cache_put(episode, result)
        return result

    def get(self, episode_index: Any, frame_index: Any) -> Optional[Dict[str, Any]]:
        episode, frame = _int(episode_index), _int(frame_index)
        if episode is None or frame is None:
            return None
        candidates = self._load_episode(episode).get(frame, [])
        if not candidates:
            return None
        valid = [row for row in candidates if not row.get("ignore", False)]
        ordered = valid or candidates
        ordered = sorted(ordered, key=lambda row: (0 if row.get("memlite_branch") == "high" else 1))
        return ordered[0]

    def branch_sampling_index(self) -> Dict[str, Any]:
        """Build a compact high-key + low-range sampling index.

        A Python tuple per low frame is prohibitive for multi-million-frame
        sidecars.  Instead, low rows are run-length encoded as
        ``(episode, start, end)`` ranges, and planner rows remain a small list
        of individual keys.  Input is required to be episode/frame ordered,
        which the annotation writer guarantees; rejecting unordered input is
        safer than silently admitting gap/cross-skill rows to training.
        """
        if self._branch_sampling_index is not None:
            return self._branch_sampling_index
        high_keys: list[tuple[int, int]] = []
        low_ranges: list[tuple[int, int, int]] = []
        current_episode: Optional[int] = None
        low_runs: list[list[int]] = []
        high_frames: set[int] = set()
        previous_low: Optional[int] = None

        def flush_episode() -> None:
            if current_episode is None:
                return
            # A high row is authoritative at a duplicate boundary.  Split the
            # low ranges at those few planner frames, including a terminal
            # high/DONE row appended after all low labels.
            for start, end in low_runs:
                cursor = start
                for frame in sorted(high_frames):
                    if frame < cursor:
                        continue
                    if frame >= end:
                        break
                    if cursor < frame:
                        low_ranges.append((current_episode, cursor, frame))
                    cursor = frame + 1
                if cursor < end:
                    low_ranges.append((current_episode, cursor, end))

        def consider(raw: Dict[str, Any]) -> None:
            nonlocal current_episode, low_runs, high_frames, previous_low
            row = normalize_label(raw)
            if row is None or row.get("ignore", False):
                return
            episode, frame = row["episode_index"], row["frame_index"]
            if current_episode is None:
                current_episode = episode
            elif episode != current_episode:
                if episode < current_episode:
                    raise ValueError("MEM-Lite sidecar must be ordered by episode_index for range sampling")
                flush_episode()
                current_episode, low_runs, high_frames, previous_low = episode, [], set(), None
            if row["memlite_branch"] == "high":
                # Duplicate labels at one boundary can occur in an appended
                # terminal-DONE pass.  They are one physical planner sample,
                # not two DDP-shardable rows.
                if frame not in high_frames:
                    high_frames.add(frame)
                    high_keys.append((episode, frame))
                return
            if row["memlite_branch"] != "low":
                return
            if previous_low is not None and frame < previous_low:
                raise ValueError("MEM-Lite sidecar low rows must be ordered by frame_index")
            if previous_low == frame:
                return
            if previous_low is not None and frame == previous_low + 1:
                low_runs[-1][1] = frame + 1
            else:
                low_runs.append([frame, frame + 1])
            previous_low = frame

        if self._format == "parquet":
            parquet = self._open_parquet()
            names = set(parquet.schema_arrow.names)
            columns = [
                key
                for key in ("episode_index", "frame_index", "memlite_branch", "branch", "ignore", "invalid", "gap", "intent_status", "status")
                if key in names
            ]
            for group_idx in range(parquet.num_row_groups):
                for record_batch in parquet.iter_batches(
                    batch_size=65536, row_groups=[group_idx], columns=columns
                ):
                    for row in record_batch.to_pylist():
                        consider(row)
        else:
            with self.path.open("rb") as fh:
                for raw in fh:
                    if not raw.strip():
                        continue
                    try:
                        row = json.loads(raw)
                    except (json.JSONDecodeError, UnicodeDecodeError):
                        continue
                    if isinstance(row, dict):
                        consider(row)
        flush_episode()
        result = {"high": high_keys, "low_ranges": low_ranges}
        if not high_keys or not low_ranges:
            raise ValueError(
                "MEM-Lite sidecar needs valid high rows and non-empty low ranges; "
                f"got high={len(high_keys)}, low_ranges={len(low_ranges)}"
            )
        self._branch_sampling_index = result
        return result

    def close(self) -> None:
        self._cache.clear()
        self._branch_sampling_index = None
        self._parquet = None


__all__ = ["MemLiteSidecar", "normalize_label"]
