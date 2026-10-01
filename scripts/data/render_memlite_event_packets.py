"""Create P107 camera-native annotation packets from an immutable event index.

The default mode only seals source-video locators.  ``--decode`` is an
explicit, bounded operation that writes one RGB PNG per camera and a PTS
receipt; it never hashes whole videos or accepts QA contact-sheet composites.
Contact sheets, when requested, are separately marked review-only and are
never referenced from actor input.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time
from typing import Any, Iterable, Mapping


REPO = Path(__file__).resolve().parents[2]


def _load_protocol() -> Any:
    """Keep locator-only use stdlib-only despite g05.data ML imports."""
    path = REPO / "src/g05/data/memlite_event_protocol.py"
    spec = importlib.util.spec_from_file_location("p107_memlite_event_protocol", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load P107 protocol: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_protocol = _load_protocol()
PACKET_SCHEMA_VERSION = _protocol.PACKET_SCHEMA_VERSION
canonical_json = _protocol.canonical_json
canonical_sha256 = _protocol.canonical_sha256
event_id = _protocol.event_id
validate_event = _protocol.validate_event
validate_source_ref = _protocol.validate_source_ref


VIEWS = ("head", "left_wrist", "right_wrist")
PACKET_INDEX_SCHEMA = "memlite-event-packet-index-v1"
PACKET_PAYLOAD_FILES = frozenset(("packets.jsonl", "rendered_asset_receipts.jsonl"))
PACKET_MANIFEST_FIELDS = frozenset((
    "schema_version", "status", "training_eligible", "index_manifest_sha256", "index_event_file_sha256",
    "files", "packets", "rendered_asset_receipts", "decoded_camera_native_rgb",
    "temporal_sample_offsets_frames", "temporal_slots", "distinct_temporal_sample_frames",
    "clamped_duplicate_temporal_samples", "review_only_contact_sheets", "full_video_hashing",
    "renderer_sha256", "protocol_sha256", "wall_seconds",
))
TEMPORAL_SAMPLE_OFFSETS = (-90, -60, -30, -15, 0, 15, 30)
MAX_TEMPORAL_OFFSET_FRAMES = 240  # Eight seconds at the source 30 Hz clock.


def _strict_json(data: bytes, *, name: str) -> Any:
    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON key in {name}: {key}")
            result[key] = value
        return result

    def nonfinite(value: str) -> None:
        raise ValueError(f"Non-finite JSON number in {name}: {value}")

    return json.loads(data, object_pairs_hook=unique, parse_constant=nonfinite)


def _read_json(path: Path) -> Any:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Required regular file is missing: {path}")
    return _strict_json(path.read_bytes(), name=str(path))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Required regular file is missing: {path}")
    rows = []
    for number, line in enumerate(path.read_bytes().splitlines(), start=1):
        if line.strip():
            value = _strict_json(line, name=f"{path}:{number}")
            if not isinstance(value, dict):
                raise ValueError(f"Expected object at {path}:{number}")
            rows.append(value)
    return rows


def _select_index_events(path: Path, receipt: Mapping[str, Any], event_ids: set[str],
                         limit: int | None) -> list[dict[str, Any]]:
    """Stream a large sealed event JSONL; never materialize the full index for a bounded packet batch."""
    if path.is_symlink() or not path.is_file():
        raise ValueError("event index payload must be a regular file")
    if not isinstance(receipt, Mapping) or set(receipt) != {"sha256", "rows", "bytes"}:
        raise ValueError("event index manifest has an invalid event-candidate receipt")
    if path.stat().st_size != receipt["bytes"]:
        raise ValueError("event index payload does not match its sealed manifest")
    digest, rows, selected, found_ids = hashlib.sha256(), 0, [], set()
    with path.open("rb") as stream:
        for number, line in enumerate(stream, start=1):
            digest.update(line)
            if not line.strip():
                continue
            rows += 1
            # Once a finite prefix has been retained, only integrity accounting
            # remains; this keeps peak memory independent of full index size.
            if not event_ids and limit is not None and len(selected) >= limit:
                continue
            value = _strict_json(line, name=f"{path}:{number}")
            if not isinstance(value, dict):
                raise ValueError(f"Expected object at {path}:{number}")
            if event_ids:
                event_id_value = value.get("event_id")
                if event_id_value in event_ids:
                    if event_id_value in found_ids:
                        raise ValueError("sealed event index contains duplicate selected event IDs")
                    found_ids.add(event_id_value)
                    selected.append(value)
            else:
                selected.append(value)
    if digest.hexdigest() != receipt["sha256"] or rows != receipt["rows"]:
        raise ValueError("event index payload does not match its sealed manifest")
    if event_ids and found_ids != event_ids:
        raise ValueError("requested event_id absent from sealed event index")
    return selected


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _forbid_source_output(output: Path, *sources: Path) -> None:
    resolved = output.resolve(strict=False)
    for source in sources:
        source = source.resolve(strict=True)
        if resolved == source or source in resolved.parents:
            raise ValueError("packet output must be outside immutable index/raw source directories")


def _read_event_ids(path: Path | None, direct: Iterable[str]) -> set[str]:
    selected = set(direct)
    if path is None:
        return selected
    if path.is_symlink() or not path.is_file():
        raise ValueError("--event-ids must be a regular text/JSONL file")
    for number, line in enumerate(path.read_text().splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        if line.startswith("{"):
            value = _strict_json(line.encode(), name=f"{path}:{number}")
            line = value.get("event_id") if isinstance(value, dict) else None
        if not isinstance(line, str) or len(line) != 64 or any(c not in "0123456789abcdef" for c in line):
            raise ValueError(f"Invalid event_id at {path}:{number}")
        selected.add(line)
    return selected


def _parse_temporal_offsets(value: str) -> tuple[int, ...]:
    try:
        offsets = tuple(int(part) for part in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("--temporal-offset-frames must be seven comma-separated integers") from error
    if not _valid_temporal_offsets(offsets):
        raise argparse.ArgumentTypeError("--temporal-offset-frames requires seven unique values including 0 within +/-240")
    return offsets


def _valid_temporal_offsets(offsets: Any) -> bool:
    """The packet schedule is deliberately fixed-size and bounded before decode."""
    return (isinstance(offsets, (tuple, list)) and len(offsets) == 7 and len(set(offsets)) == 7 and
            0 in offsets and all(type(offset) is int and abs(offset) <= MAX_TEMPORAL_OFFSET_FRAMES
                                 for offset in offsets))


def _questions(path: Path | None) -> dict[str, dict[str, Any]]:
    """Load actor prompt context, not labels; fail on answer-like payload keys."""
    if path is None:
        return {}
    prompts: dict[str, dict[str, Any]] = {}
    for row in _read_jsonl(path):
        if set(row) != {"event_id", "question_context"} or not isinstance(row["event_id"], str):
            raise ValueError("question rows require exactly event_id and question_context")
        context = row["question_context"]
        if not isinstance(context, dict):
            raise ValueError("question_context must be an object")
        if _actor_has_forbidden_key(context):
            raise ValueError("actor question_context must not carry a label/review/evidence field")
        if row["event_id"] in prompts:
            raise ValueError("duplicate actor question context")
        prompts[row["event_id"]] = context
    return prompts


def _source_annotation_context(event: Mapping[str, Any]) -> dict[str, Any]:
    """Expose a source instruction as a question context, never as visual text or truth."""
    return {
        "task_context": event.get("task_name"),
        "source_annotation_context": event.get("skill_bundle"),
        "source_annotation_is_not_truth": True,
    }


def _packet_id(index_sha256: str, event: Mapping[str, Any], context: Mapping[str, Any], decode: bool,
               temporal_offsets_frames: tuple[int, ...]) -> str:
    return canonical_sha256({
        "schema_version": PACKET_SCHEMA_VERSION,
        "index_manifest_sha256": index_sha256,
        "event_id": event["event_id"],
        "observation_frame": event["observation"]["frame"],
        "question_context": context,
        "decoded_camera_native_rgb": decode,
        "temporal_offsets_frames": list(temporal_offsets_frames),
    })


def _locators_by_view(event: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    rows = event["video_locators"]
    by_view = {row.get("view"): row for row in rows if isinstance(row, Mapping)}
    if len(rows) != 3 or set(by_view) != set(VIEWS) or len(by_view) != len(rows):
        raise ValueError("packet rendering requires exactly three unique source camera locators")
    for view, locator in by_view.items():
        if locator.get("locator_status") != "METADATA_ONLY_UNRESOLVED":
            raise ValueError(f"unexpected source locator status for {view}")
        requested = locator.get("requested_timestamp_s")
        if type(requested) not in (int, float) or isinstance(requested, bool):
            raise ValueError("source locator has no finite requested timestamp")
        expected = locator.get("episode_start_timestamp_s") + event["observation"]["frame"] / 30.0
        if abs(float(requested) - expected) > 1e-9 or locator.get("expected_fps") != 30:
            raise ValueError("source video locator clock mismatch")
    return by_view


def _temporal_samples(event: Mapping[str, Any], offsets_frames: tuple[int, ...]) -> list[dict[str, Any]]:
    if not _valid_temporal_offsets(offsets_frames):
        raise ValueError("temporal packet rendering requires exactly seven unique offsets including zero within +/-240 frames")
    anchor = event["observation"]["frame"]
    length = event["source"]["episode_length"]
    samples = []
    for sample_index, offset in enumerate(offsets_frames):
        frame = min(length - 1, max(0, anchor + offset))
        samples.append({"sample_index": sample_index, "offset_frames": offset, "sample_frame": frame,
                        "timestamp_s": frame / 30.0,
                        "temporal_role": "ACTOR_CAUSAL" if frame <= anchor else "OFFLINE_FUTURE_AUDIT"})
    return samples


def _temporal_locator(locator: Mapping[str, Any], sample: Mapping[str, Any]) -> dict[str, Any]:
    return {**locator, "requested_timestamp_s": locator["episode_start_timestamp_s"] + sample["timestamp_s"],
            "sample_index": sample["sample_index"], "offset_frames": sample["offset_frames"],
            "sample_frame": sample["sample_frame"], "temporal_role": sample["temporal_role"]}


def _resolve_video(raw_root: Path, relative_path: str) -> Path:
    relative = Path(relative_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("video locator must be a safe relative source path")
    path = (raw_root / relative).resolve(strict=True)
    if raw_root.resolve(strict=True) not in path.parents or not path.is_file():
        raise ValueError("video locator escapes raw root or is not a regular file")
    return path


def _require_decode_dependencies(*, contact_sheets: bool) -> tuple[Any, Any | None, Any | None]:
    try:
        import av
    except ImportError as error:
        raise RuntimeError("--decode requires PyAV; install it in an isolated data environment, not the shared training env") from error
    if not contact_sheets:
        return av, None, None
    try:
        from PIL import Image, ImageDraw
    except ImportError as error:
        raise RuntimeError("--contact-sheets requires Pillow together with --decode") from error
    return av, Image, ImageDraw


def _decode_rgb(av: Any, video: Path, requested: float, *, episode_start_timestamp_s: float,
                episode_end_timestamp_s: float,
                actor_anchor_timestamp_s: float | None) -> tuple[Any, dict[str, Any]]:
    """Decode a bounded source PTS without crossing an episode in a shared container.

    ``episode_*`` are source-row global PTS bounds, not MP4 duration.  A
    causal actor frame may round up by less than half a frame, but never past
    the actor anchor; ties choose the non-future PTS.
    """
    if not (episode_start_timestamp_s <= requested <= episode_end_timestamp_s):
        raise ValueError("requested frame lies outside the source episode global PTS bounds")
    if actor_anchor_timestamp_s is not None and requested > actor_anchor_timestamp_s:
        raise ValueError("actor camera request cannot be later than the pre-action observation clock")
    with av.open(str(video)) as container:
        if not container.streams.video:
            raise ValueError(f"source has no video stream: {video}")
        stream = container.streams.video[0]
        rate = float(stream.average_rate) if stream.average_rate is not None else float("nan")
        if abs(rate - 30.0) > 1e-6:
            raise ValueError(f"source FPS changed: {video} ({rate})")
        if stream.time_base is None:
            raise ValueError(f"source stream has no time base: {video}")
        tolerance = 1.0 / 60.0 + 1e-6
        container.seek(max(0, int(max(episode_start_timestamp_s, requested - tolerance) /
                                  float(stream.time_base))), stream=stream, backward=True, any_frame=False)
        best = None
        for frame in container.decode(stream):
            if frame.pts is None:
                continue
            actual = float(frame.pts * stream.time_base)
            # A file can contain contiguous demonstrations.  Never use an
            # adjacent episode merely because it is close to the requested
            # PTS, and never expose a rounded future frame at the actor clock.
            if actual < episode_start_timestamp_s - tolerance:
                continue
            if actual > episode_end_timestamp_s + tolerance:
                break
            distance = abs(actual - requested)
            allowed = (actual <= episode_end_timestamp_s and actual >= episode_start_timestamp_s and
                       (actor_anchor_timestamp_s is None or actual <= actor_anchor_timestamp_s))
            candidate_rank = (distance, actual > requested, actual)
            if allowed and distance <= tolerance and (best is None or candidate_rank < best[0]):
                image = frame.to_image().convert("RGB")
                if best is not None:
                    best[3].close()
                best = (candidate_rank, actual, image, image)
            if actual > requested + tolerance:
                break
        if best is None:
            raise ValueError(f"cannot decode requested source PTS within half-frame: {video}")
        stat = video.stat()
        return best[3], {
            "resolved_path": str(video), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns,
            "requested_timestamp_s": requested, "decoded_timestamp_s": best[1],
            "pts_error_s": best[0][0], "fps": rate, "resolution": list(best[3].size),
            "episode_global_pts_bounds_s": [episode_start_timestamp_s, episode_end_timestamp_s],
            "actor_anchor_timestamp_s": actor_anchor_timestamp_s,
            "full_video_sha256": None,
        }


def _review_contact_sheet(samples: list[tuple[Mapping[str, Any], Mapping[str, Any]]], path: Path,
                          Image: Any, ImageDraw: Any) -> None:
    """Timestamped review-only composite; no text is present in actor-native PNGs."""
    sizes = [samples[0][1][view].size for view in VIEWS]
    width = sum(size[0] for size in sizes)
    row_height = max(size[1] for size in sizes) + 30
    height = row_height * len(samples)
    sheet = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(sheet)
    for row_index, (sample, images) in enumerate(samples):
        x, y = 0, row_index * row_height
        for view in VIEWS:
            image = images[view]
            sheet.paste(image, (x, y + 30))
            draw.text((x + 4, y + 6), f"{view} f={sample['sample_frame']} t={sample['timestamp_s']:.3f}s", fill="black")
            x += image.size[0]
    sheet.save(path)


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    count = 0
    with path.open("xb") as stream:
        for row in rows:
            stream.write(canonical_json(row).encode() + b"\n")
            count += 1
    return {"sha256": _sha256(path), "rows": count, "bytes": path.stat().st_size}


def create_packets(index: Path, output: Path, *, event_ids: set[str], limit: int | None,
                   questions: Mapping[str, Mapping[str, Any]], include_source_annotation_context: bool,
                   decode: bool, raw_root: Path | None, contact_sheets: bool,
                   temporal_offsets_frames: tuple[int, ...] = TEMPORAL_SAMPLE_OFFSETS,
                   max_seconds: float = 1800.0) -> dict[str, Any]:
    index, output = Path(index), Path(output)
    if output.exists():
        raise FileExistsError("packet output already exists; use --resume only to verify its sealed receipt")
    if index.is_symlink() or not index.is_dir():
        raise ValueError("index must be an existing regular directory")
    if not event_ids and limit is None:
        raise ValueError("select explicit --event-id/--event-ids or provide a finite --limit")
    if limit is not None and (type(limit) is not int or limit < 1):
        raise ValueError("limit must be a positive integer")
    if contact_sheets and not decode:
        raise ValueError("--contact-sheets requires --decode camera-native images")
    if decode:
        if raw_root is None or raw_root.is_symlink() or not raw_root.is_dir():
            raise ValueError("--decode requires an existing non-symlink --raw-root")
        av, Image, ImageDraw = _require_decode_dependencies(contact_sheets=contact_sheets)
    else:
        av = Image = ImageDraw = None
    _forbid_source_output(output, index, *( [raw_root] if raw_root is not None else [] ))
    index_manifest_path = index / "manifest.json"
    manifest = _read_json(index_manifest_path)
    if manifest.get("schema_version") != "memlite-event-index-v1":
        raise ValueError("input is not a sealed P107 event index")
    index_sha = _sha256(index_manifest_path)
    receipt = manifest.get("files", {}).get("event_candidates.jsonl", {})
    selected = _select_index_events(index / "event_candidates.jsonl", receipt, event_ids, limit)
    if not selected:
        raise ValueError("selection produced no event packets")
    selected_ids = {event["event_id"] for event in selected}
    if len(selected_ids) != len(selected):
        raise ValueError("sealed event index contains duplicate selected event IDs")
    if set(questions) - selected_ids:
        raise ValueError("question context references an event outside the bounded packet selection")
    started = time.monotonic()
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=str(output.parent)))
    packets = []
    asset_receipts = []
    seen_packet_ids = set()
    try:
        if decode:
            (staging / "assets").mkdir()
            if contact_sheets:
                (staging / "review_assets").mkdir()
        for event in selected:
            if time.monotonic() - started > max_seconds:
                raise TimeoutError("packet rendering CPU budget reached; no packet directory was published")
            validate_event(event)
            if event["event_id"] != event_id(event):
                raise ValueError("event ID drift in sealed index")
            context = dict(questions.get(event["event_id"], {}))
            if _actor_has_forbidden_key(context):
                raise ValueError("actor question_context must not carry nested outcome/label/review/evidence material")
            if include_source_annotation_context:
                if context:
                    raise ValueError("use either explicit question context or source annotation context, not both")
                context = _source_annotation_context(event)
            if _actor_has_forbidden_key(context):
                raise ValueError("actor question_context must not carry nested outcome/label/review/evidence material")
            samples = _temporal_samples(event, temporal_offsets_frames)
            packet_id = _packet_id(index_sha, event, context, decode, temporal_offsets_frames)
            if packet_id in seen_packet_ids:
                raise ValueError("duplicate packet_id in selected immutable events")
            seen_packet_ids.add(packet_id)
            by_view = _locators_by_view(event)
            actor_images: dict[str, str] = {}
            actor_temporal_rgb: list[dict[str, Any]] = []
            offline_future_rgb: list[dict[str, Any]] = []
            decoded_receipts: list[dict[str, Any]] = []
            review_asset = None
            if decode:
                review_images: list[tuple[Mapping[str, Any], Mapping[str, Any]]] = []
                try:
                    for sample in samples:
                        images, paths = {}, {}
                        for view in VIEWS:
                            locator = _temporal_locator(by_view[view], sample)
                            video = _resolve_video(raw_root, locator["relative_path"])
                            episode_start = float(locator["episode_start_timestamp_s"])
                            episode_end = episode_start + (event["source"]["episode_length"] - 1) / 30.0
                            actor_anchor = (episode_start + event["observation"]["frame"] / 30.0
                                            if sample["temporal_role"] == "ACTOR_CAUSAL" else None)
                            image, decoded = _decode_rgb(
                                av, video, float(locator["requested_timestamp_s"]),
                                episode_start_timestamp_s=episode_start, episode_end_timestamp_s=episode_end,
                                actor_anchor_timestamp_s=actor_anchor)
                            native_name = f"{packet_id}_t{sample['sample_index']:02d}_{view}.png"
                            asset_path = staging / "assets" / native_name
                            image.save(asset_path)
                            relative_path = f"assets/{native_name}"
                            paths[view] = relative_path
                            decoded_receipts.append({**decoded, "view": view, **sample,
                                                     "camera_native_png_sha256": _sha256(asset_path),
                                                     "camera_native_png_bytes": asset_path.stat().st_size})
                            asset_receipts.append({
                                "schema_version": PACKET_SCHEMA_VERSION, "packet_id": packet_id, "view": view,
                                "asset_kind": "camera_native_rgb_png", "temporal_role": sample["temporal_role"],
                                "sample_index": sample["sample_index"], "sample_frame": sample["sample_frame"],
                                "requested_timestamp_s": locator["requested_timestamp_s"],
                                "decoded_timestamp_s": decoded["decoded_timestamp_s"], "pts_error_s": decoded["pts_error_s"],
                                "relative_path": relative_path, "sha256": _sha256(asset_path),
                                "bytes": asset_path.stat().st_size,
                            })
                            images[view] = image
                        temporal_record = {**sample, "images": paths}
                        if sample["temporal_role"] == "ACTOR_CAUSAL":
                            actor_temporal_rgb.append(temporal_record)
                            if sample["offset_frames"] == 0:
                                actor_images = dict(paths)
                        else:
                            offline_future_rgb.append(temporal_record)
                        if contact_sheets:
                            review_images.append((sample, images))
                        else:
                            for image in images.values():
                                image.close()
                    if contact_sheets:
                        review_name = f"{packet_id}_contact_sheet.png"
                        _review_contact_sheet(review_images, staging / "review_assets" / review_name, Image, ImageDraw)
                        review_asset = f"review_assets/{review_name}"
                        review_path = staging / review_asset
                        asset_receipts.append({
                            "schema_version": PACKET_SCHEMA_VERSION, "packet_id": packet_id, "view": None,
                            "asset_kind": "review_only_contact_sheet_png", "relative_path": review_asset,
                            "sha256": _sha256(review_path), "bytes": review_path.stat().st_size,
                        })
                finally:
                    for _, images in review_images:
                        for image in images.values():
                            image.close()
            actor_packet = {
                "packet_id": packet_id, "event_id": event["event_id"],
                "observation_frame": event["observation"]["frame"],
                "images": actor_images,
                "causal_temporal_rgb": actor_temporal_rgb,
                "question_context": context,
                "actor_instruction": "Use only the camera-native RGB images and question context. Source annotation context is not ground truth.",
            }
            # Actor input has no source outcome, evidence, review, label, or
            # contact-sheet path.  Those remain in the audit section below.
            packets.append({
                "schema_version": PACKET_SCHEMA_VERSION,
                "packet_id": packet_id,
                "status": "DECODED_PENDING_LABELS" if decode else "LOCATOR_READY_NO_RGB",
                "training_eligible": False,
                "actor_packet": actor_packet,
                "audit": {
                    "event_id": event["event_id"], "source": event["source"],
                    "usage_role": event.get("usage_role"), "video_locators": [by_view[view] for view in VIEWS],
                    "temporal_schedule": [{**sample, "video_locators": [
                        _temporal_locator(by_view[view], sample) for view in VIEWS]} for sample in samples],
                    "temporal_slot_count": len(samples),
                    "distinct_sample_frame_count": len({sample["sample_frame"] for sample in samples}),
                    "clamped_duplicate_sample_frame_count": len(samples) - len({sample["sample_frame"] for sample in samples}),
                    "decoded_pts_receipts": decoded_receipts, "offline_future_rgb": offline_future_rgb,
                    "review_only_contact_sheet": review_asset,
                },
            })
        files = {"packets.jsonl": _write_jsonl(staging / "packets.jsonl", packets),
                 "rendered_asset_receipts.jsonl": _write_jsonl(staging / "rendered_asset_receipts.jsonl", asset_receipts)}
        result = {
            "schema_version": PACKET_INDEX_SCHEMA,
            "status": "CPU_READY_LABELS_PENDING" if decode else "LOCATORS_READY_RENDER_PENDING",
            "training_eligible": False,
            "index_manifest_sha256": index_sha,
            "index_event_file_sha256": receipt["sha256"],
            "files": files, "packets": len(packets), "rendered_asset_receipts": len(asset_receipts),
            "decoded_camera_native_rgb": decode,
            "temporal_sample_offsets_frames": list(temporal_offsets_frames),
            "temporal_slots": len(packets) * len(temporal_offsets_frames),
            "distinct_temporal_sample_frames": sum(
                packet["audit"]["distinct_sample_frame_count"] for packet in packets),
            "clamped_duplicate_temporal_samples": sum(
                packet["audit"]["clamped_duplicate_sample_frame_count"] for packet in packets),
            "review_only_contact_sheets": contact_sheets, "full_video_hashing": False,
            "renderer_sha256": _sha256(Path(__file__)),
            "protocol_sha256": _sha256(REPO / "src/g05/data/memlite_event_protocol.py"),
            "wall_seconds": time.monotonic() - started,
        }
        manifest_path = staging / "manifest.json"
        manifest_path.write_bytes(canonical_json(result).encode() + b"\n")
        packet_manifest_sha256 = _sha256(manifest_path)
        os.rename(staging, output)
        # The manifest deliberately does not self-hash.  Resume callers must
        # retain this externally supplied value; recomputing a replacement is
        # not a valid authority action.
        return {**result, "packet_manifest_sha256": packet_manifest_sha256}
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _actor_has_forbidden_key(value: Any) -> bool:
    forbidden = ("answer", "label", "outcome", "ground_truth", "success", "failure", "review", "evidence",
                 "audit", "privileged")
    if isinstance(value, Mapping):
        return any(any(token in str(key).lower() for token in forbidden) or _actor_has_forbidden_key(item)
                   for key, item in value.items())
    if isinstance(value, list):
        return any(_actor_has_forbidden_key(item) for item in value)
    return False


def _safe_asset_path(value: Any, *, directory: str) -> bool:
    if not isinstance(value, str) or not value:
        return False
    relative = Path(value)
    return not relative.is_absolute() and ".." not in relative.parts and relative.parts[:1] == (directory,)


def _validate_packet_semantics(packet: Mapping[str, Any], *, decoded: bool,
                               expected_offsets: tuple[int, ...]) -> None:
    required = {"schema_version", "packet_id", "status", "training_eligible", "actor_packet", "audit"}
    if set(packet) != required or packet["schema_version"] != PACKET_SCHEMA_VERSION:
        raise ValueError("packet row does not use the exact P107 packet schema")
    if packet["training_eligible"] is not False or packet["status"] != (
            "DECODED_PENDING_LABELS" if decoded else "LOCATOR_READY_NO_RGB"):
        raise ValueError("packet rows are candidate-only and cannot self-promote training eligibility")
    actor = packet["actor_packet"]
    actor_required = {"packet_id", "event_id", "observation_frame", "images", "causal_temporal_rgb",
                      "question_context", "actor_instruction"}
    if not isinstance(actor, Mapping) or set(actor) != actor_required or actor["packet_id"] != packet["packet_id"]:
        raise ValueError("packet actor input has an invalid schema")
    if not isinstance(actor["question_context"], Mapping) or not isinstance(actor["actor_instruction"], str):
        raise ValueError("packet actor question/instruction schema is invalid")
    if _actor_has_forbidden_key(actor):
        raise ValueError("actor input contains forbidden outcome/label/review/evidence material")
    if type(actor["observation_frame"]) is not int or actor["observation_frame"] < 0:
        raise ValueError("actor packet observation clock is invalid")
    if not isinstance(actor["images"], Mapping) or set(actor["images"]) - set(VIEWS):
        raise ValueError("actor packet image paths are invalid")
    if not isinstance(actor["causal_temporal_rgb"], list):
        raise ValueError("actor temporal RGB must be a list")
    causal_paths = set()
    causal_samples: dict[int, Mapping[str, Any]] = {}
    for sample in actor["causal_temporal_rgb"]:
        fields = {"sample_index", "offset_frames", "sample_frame", "timestamp_s", "temporal_role", "images"}
        if not isinstance(sample, Mapping) or set(sample) != fields or sample["temporal_role"] != "ACTOR_CAUSAL":
            raise ValueError("actor temporal RGB sample has an invalid causal schema")
        if (type(sample["sample_index"]) is not int or sample["sample_index"] in causal_samples or
                type(sample["sample_frame"]) is not int or sample["sample_frame"] > actor["observation_frame"]):
            raise ValueError("actor temporal RGB contains a future frame")
        if not isinstance(sample["images"], Mapping) or set(sample["images"]) != set(VIEWS):
            raise ValueError("actor temporal RGB must provide all three camera-native views")
        if any(not _safe_asset_path(path, directory="assets") for path in sample["images"].values()):
            raise ValueError("actor temporal RGB must name only camera-native asset paths")
        causal_samples[sample["sample_index"]] = sample
        causal_paths.update(sample["images"].values())
    if decoded and (set(actor["images"]) != set(VIEWS) or not set(actor["images"].values()) <= causal_paths):
        raise ValueError("decoded actor anchor images must be causal temporal camera-native assets")
    if decoded and any(not _safe_asset_path(path, directory="assets") for path in actor["images"].values()):
        raise ValueError("actor anchor images must name only camera-native asset paths")
    if not decoded and (actor["images"] or actor["causal_temporal_rgb"]):
        raise ValueError("locator-only packet cannot claim decoded actor RGB")
    audit = packet["audit"]
    audit_required = {"event_id", "source", "usage_role", "video_locators", "temporal_schedule",
                      "temporal_slot_count", "distinct_sample_frame_count", "clamped_duplicate_sample_frame_count",
                      "decoded_pts_receipts", "offline_future_rgb", "review_only_contact_sheet"}
    if not isinstance(audit, Mapping) or set(audit) != audit_required or audit["event_id"] != actor["event_id"]:
        raise ValueError("packet audit has an invalid schema")
    validate_source_ref(audit["source"])
    if not isinstance(audit["video_locators"], list) or len(audit["video_locators"]) != 3:
        raise ValueError("packet audit must retain exactly three source camera locators")
    by_view = {locator.get("view"): locator for locator in audit["video_locators"] if isinstance(locator, Mapping)}
    if len(by_view) != 3 or set(by_view) != set(VIEWS):
        raise ValueError("packet audit camera locators are malformed")
    for locator in by_view.values():
        expected_time = locator.get("episode_start_timestamp_s") + actor["observation_frame"] / 30.0
        if (locator.get("locator_status") != "METADATA_ONLY_UNRESOLVED" or locator.get("expected_fps") != 30 or
                type(locator.get("requested_timestamp_s")) not in (int, float) or
                abs(locator["requested_timestamp_s"] - expected_time) > 1e-9):
            raise ValueError("packet audit base locator is not a source-clock camera locator")
    schedule = audit["temporal_schedule"]
    if not isinstance(schedule, list) or len(schedule) != 7:
        raise ValueError("packet audit must retain exactly seven bounded temporal samples")
    if (audit["temporal_slot_count"] != 7 or type(audit["distinct_sample_frame_count"]) is not int or
            not 1 <= audit["distinct_sample_frame_count"] <= 7 or
            audit["clamped_duplicate_sample_frame_count"] != 7 - audit["distinct_sample_frame_count"]):
        raise ValueError("packet temporal slot/duplicate accounting is invalid")
    expected_sample_by_index: dict[int, Mapping[str, Any]] = {}
    episode_length = audit["source"]["episode_length"]
    for position, sample in enumerate(schedule):
        fields = {"sample_index", "offset_frames", "sample_frame", "timestamp_s", "temporal_role", "video_locators"}
        if not isinstance(sample, Mapping) or set(sample) != fields or sample["sample_index"] != position:
            raise ValueError("packet temporal schedule is malformed")
        expected_frame = min(episode_length - 1, max(0, actor["observation_frame"] + expected_offsets[position]))
        expected_role = "ACTOR_CAUSAL" if sample["sample_frame"] <= actor["observation_frame"] else "OFFLINE_FUTURE_AUDIT"
        if (sample["offset_frames"] != expected_offsets[position] or sample["sample_frame"] != expected_frame or
                sample["timestamp_s"] != expected_frame / 30.0 or sample["temporal_role"] != expected_role or
                not isinstance(sample["video_locators"], list) or len(sample["video_locators"]) != 3):
            raise ValueError("packet temporal schedule crosses its causal boundary")
        locators = {locator.get("view"): locator for locator in sample["video_locators"] if isinstance(locator, Mapping)}
        if len(locators) != 3 or set(locators) != set(VIEWS):
            raise ValueError("packet temporal schedule must preserve all three camera views")
        bare_sample = {key: sample[key] for key in ("sample_index", "offset_frames", "sample_frame", "timestamp_s", "temporal_role")}
        if any(dict(locators[view]) != _temporal_locator(by_view[view], bare_sample) for view in VIEWS):
            raise ValueError("packet temporal camera locator does not bind the source clock")
        expected_sample_by_index[position] = sample
    if audit["distinct_sample_frame_count"] != len({sample["sample_frame"] for sample in schedule}):
        raise ValueError("packet temporal duplicate accounting does not match the clipped schedule")
    expected_causal_indices = {index for index, sample in expected_sample_by_index.items()
                                if sample["temporal_role"] == "ACTOR_CAUSAL"}
    if decoded and set(causal_samples) != expected_causal_indices:
        raise ValueError("actor temporal RGB does not exactly match causal packet samples")
    anchor_index = expected_offsets.index(0)
    if decoded and dict(actor["images"]) != dict(causal_samples[anchor_index]["images"]):
        raise ValueError("actor anchor images must bind the zero-offset camera sample")
    offline_samples: dict[int, Mapping[str, Any]] = {}
    if not isinstance(audit["offline_future_rgb"], list):
        raise ValueError("offline audit RGB must be a list")
    for sample in audit["offline_future_rgb"]:
        if (not isinstance(sample, Mapping) or sample.get("temporal_role") != "OFFLINE_FUTURE_AUDIT" or
                sample.get("sample_frame", -1) <= actor["observation_frame"] or
                sample.get("sample_index") in offline_samples):
            raise ValueError("offline audit RGB must be strictly future of the actor observation")
        if set(sample) != {"sample_index", "offset_frames", "sample_frame", "timestamp_s", "temporal_role", "images"}:
            raise ValueError("offline audit RGB has an invalid temporal schema")
        if not isinstance(sample["images"], Mapping) or set(sample["images"]) != set(VIEWS):
            raise ValueError("offline audit RGB must preserve all camera-native views")
        if any(not _safe_asset_path(path, directory="assets") for path in sample["images"].values()):
            raise ValueError("offline audit RGB must name only camera-native asset paths")
        offline_samples[sample["sample_index"]] = sample
        if causal_paths & set(sample.get("images", {}).values()):
            raise ValueError("future offline audit RGB leaked into actor assets")
    expected_offline_indices = {index for index, sample in expected_sample_by_index.items()
                                if sample["temporal_role"] == "OFFLINE_FUTURE_AUDIT"}
    if decoded and set(offline_samples) != expected_offline_indices:
        raise ValueError("offline audit RGB does not exactly match future packet samples")
    if not decoded and (causal_samples or offline_samples or audit["review_only_contact_sheet"] is not None):
        raise ValueError("locator-only packet cannot contain decoded temporal or review image paths")
    decoded_fields = {"resolved_path", "bytes", "mtime_ns", "requested_timestamp_s", "decoded_timestamp_s",
                      "pts_error_s", "fps", "resolution", "episode_global_pts_bounds_s",
                      "actor_anchor_timestamp_s", "full_video_sha256", "view", "sample_index", "offset_frames",
                      "sample_frame", "timestamp_s", "temporal_role", "camera_native_png_sha256",
                      "camera_native_png_bytes"}
    receipts = audit["decoded_pts_receipts"]
    if not isinstance(receipts, list) or (decoded and len(receipts) != 21) or (not decoded and receipts):
        raise ValueError("packet decoded PTS receipt count is inconsistent with its camera render mode")
    seen_receipts = set()
    for receipt in receipts:
        if not isinstance(receipt, Mapping) or set(receipt) != decoded_fields:
            raise ValueError("decoded PTS receipt does not use the exact bounded camera schema")
        key = (receipt["sample_index"], receipt["view"])
        sample = expected_sample_by_index.get(receipt["sample_index"])
        locator = (by_view.get(receipt["view"]) if receipt["view"] in VIEWS else None)
        if key in seen_receipts or sample is None or locator is None or any(
                receipt[field] != sample[field] for field in ("sample_index", "offset_frames", "sample_frame", "timestamp_s", "temporal_role")):
            raise ValueError("decoded PTS receipt does not bind a unique scheduled camera sample")
        seen_receipts.add(key)
        expected_requested = locator["episode_start_timestamp_s"] + sample["timestamp_s"]
        start = locator["episode_start_timestamp_s"]
        end = start + (audit["source"]["episode_length"] - 1) / 30.0
        decoded_time, requested, error = receipt["decoded_timestamp_s"], receipt["requested_timestamp_s"], receipt["pts_error_s"]
        if (type(requested) not in (int, float) or type(decoded_time) not in (int, float) or
                type(error) not in (int, float) or abs(requested - expected_requested) > 1e-9 or
                abs(error - abs(decoded_time - requested)) > 1e-9 or error > 1.0 / 60.0 + 1e-6 or
                not start <= decoded_time <= end or receipt["episode_global_pts_bounds_s"] != [start, end] or
                receipt["fps"] != 30 or receipt["full_video_sha256"] is not None or
                not isinstance(receipt["resolved_path"], str) or type(receipt["bytes"]) is not int or
                type(receipt["mtime_ns"]) is not int or not isinstance(receipt["resolution"], list) or
                len(receipt["resolution"]) != 2 or any(type(value) is not int or value < 1 for value in receipt["resolution"]) or
                not isinstance(receipt["camera_native_png_sha256"], str) or type(receipt["camera_native_png_bytes"]) is not int):
            raise ValueError("decoded PTS receipt has invalid source-bound timing or image metadata")
        expected_anchor = start + actor["observation_frame"] / 30.0 if sample["temporal_role"] == "ACTOR_CAUSAL" else None
        if receipt["actor_anchor_timestamp_s"] != expected_anchor or (expected_anchor is not None and decoded_time > expected_anchor):
            raise ValueError("decoded PTS receipt crosses the actor causal observation clock")
    if decoded and seen_receipts != {(sample_index, view) for sample_index in range(7) for view in VIEWS}:
        raise ValueError("decoded PTS receipts do not cover the complete temporal camera panel")
    if audit["review_only_contact_sheet"] is not None and not _safe_asset_path(
            audit["review_only_contact_sheet"], directory="review_assets"):
        raise ValueError("review-only contact sheet must remain outside actor camera assets")


def _validate_packet_index_binding(packet: Mapping[str, Any], event: Mapping[str, Any], *,
                                   index_manifest_sha256: str, decoded: bool,
                                   temporal_offsets: tuple[int, ...]) -> None:
    """A resigned packet cannot swap its source event, camera locators, or ID."""
    validate_event(event)
    actor, audit = packet["actor_packet"], packet["audit"]
    if (actor["event_id"] != event["event_id"] or actor["observation_frame"] != event["observation"]["frame"] or
            canonical_json(audit["source"]) != canonical_json(event["source"]) or
            audit["usage_role"] != event.get("usage_role") or
            canonical_json(audit["video_locators"]) != canonical_json(event["video_locators"])):
        raise ValueError("packet does not bind the exact canonical event/source/camera record from the sealed index")
    expected_packet_id = _packet_id(index_manifest_sha256, event, actor["question_context"], decoded, temporal_offsets)
    if packet["packet_id"] != expected_packet_id:
        raise ValueError("packet_id does not match its canonical sealed event/context/temporal identity")


def _validate_packet_assets(packet: Mapping[str, Any], assets: list[Mapping[str, Any]], *, decoded: bool,
                            review_sheets: bool) -> None:
    """Bind every native PNG receipt to its packet clock; review composites stay audit-only."""
    actor, audit = packet["actor_packet"], packet["audit"]
    camera = [asset for asset in assets if asset["asset_kind"] == "camera_native_rgb_png"]
    contacts = [asset for asset in assets if asset["asset_kind"] == "review_only_contact_sheet_png"]
    if not decoded:
        if assets:
            raise ValueError("locator-only packet cannot contain rendered RGB asset receipts")
        return
    if len(camera) != 21:
        raise ValueError("decoded packet must seal exactly seven timestamps by three native cameras")
    if len(contacts) != (1 if review_sheets else 0):
        raise ValueError("decoded packet has an invalid review-only contact-sheet receipt count")
    schedule = {sample["sample_index"]: sample for sample in audit["temporal_schedule"]}
    decoded_receipts = {(receipt["sample_index"], receipt["view"]): receipt
                        for receipt in audit["decoded_pts_receipts"]}
    paths: dict[int, dict[str, str]] = {}
    for asset in camera:
        sample = schedule.get(asset["sample_index"])
        scheduled_locators = ({locator.get("view"): locator for locator in sample["video_locators"]}
                              if sample is not None else {})
        decoded_receipt = decoded_receipts.get((asset["sample_index"], asset["view"]))
        if (sample is None or asset["view"] not in VIEWS or asset["sample_frame"] != sample["sample_frame"] or
                asset["temporal_role"] != sample["temporal_role"] or
                asset["requested_timestamp_s"] != scheduled_locators[asset["view"]]["requested_timestamp_s"] or
                decoded_receipt is None or asset["requested_timestamp_s"] != decoded_receipt["requested_timestamp_s"] or
                asset["decoded_timestamp_s"] != decoded_receipt["decoded_timestamp_s"] or
                asset["pts_error_s"] != decoded_receipt["pts_error_s"] or
                asset["sha256"] != decoded_receipt["camera_native_png_sha256"] or
                asset["bytes"] != decoded_receipt["camera_native_png_bytes"]):
            raise ValueError("camera-native receipt does not bind its scheduled source timestamp")
        paths.setdefault(asset["sample_index"], {})[asset["view"]] = asset["relative_path"]
    if set(paths) != set(range(7)) or any(set(views) != set(VIEWS) for views in paths.values()):
        raise ValueError("decoded packet does not seal a complete native three-camera temporal panel")
    causal = {sample["sample_index"]: sample for sample in actor["causal_temporal_rgb"]}
    if any(dict(sample["images"]) != paths[index] for index, sample in causal.items()):
        raise ValueError("actor causal image paths do not match sealed native camera receipts")
    offline = {sample["sample_index"]: sample for sample in audit["offline_future_rgb"]}
    if any(dict(sample["images"]) != paths[index] for index, sample in offline.items()):
        raise ValueError("offline audit image paths do not match sealed native camera receipts")
    if contacts:
        if contacts[0]["relative_path"] != audit["review_only_contact_sheet"]:
            raise ValueError("review contact-sheet receipt must remain audit-only")
    elif audit["review_only_contact_sheet"] is not None:
        raise ValueError("packet references a review contact sheet without its sealed receipt")


def resume_packets(index: Path, output: Path, *, expected_packet_manifest_sha256: str | None) -> dict[str, Any]:
    index, output = Path(index), Path(output)
    if (not isinstance(expected_packet_manifest_sha256, str) or len(expected_packet_manifest_sha256) != 64 or
            any(char not in "0123456789abcdef" for char in expected_packet_manifest_sha256)):
        raise ValueError("--resume requires an externally pinned --expected-packet-manifest-sha256")
    manifest_path = output / "manifest.json"
    if _sha256(manifest_path) != expected_packet_manifest_sha256:
        raise ValueError("packet manifest does not match the externally pinned authority SHA-256")
    result = _read_json(manifest_path)
    if not isinstance(result, Mapping) or set(result) != PACKET_MANIFEST_FIELDS or result.get("schema_version") != PACKET_INDEX_SCHEMA:
        raise ValueError("--resume requires a sealed P107 packet directory")
    if result.get("training_eligible") is not False or result.get("status") not in {
            "CPU_READY_LABELS_PENDING", "LOCATORS_READY_RENDER_PENDING"}:
        raise ValueError("packet manifest cannot self-promote candidate packets to training/audited status")
    offsets = result.get("temporal_sample_offsets_frames")
    if not _valid_temporal_offsets(offsets):
        raise ValueError("packet manifest has an invalid bounded temporal sample schedule")
    temporal_offsets = tuple(offsets)
    if type(result.get("review_only_contact_sheets")) is not bool or result.get("full_video_hashing") is not False:
        raise ValueError("packet manifest has invalid review/video-hashing policy fields")
    if result.get("index_manifest_sha256") != _sha256(index / "manifest.json"):
        raise ValueError("source index manifest changed; refusing resume")
    index_manifest = _read_json(index / "manifest.json")
    index_receipt = index_manifest.get("files", {}).get("event_candidates.jsonl", {}) if isinstance(index_manifest, Mapping) else {}
    if result.get("index_event_file_sha256") != index_receipt.get("sha256"):
        raise ValueError("packet manifest does not bind the sealed event-candidate payload")
    receipt = result.get("files", {}).get("packets.jsonl", {})
    if set(result.get("files", {})) != PACKET_PAYLOAD_FILES:
        raise ValueError("packet manifest must seal the exact packet and rendered-asset receipt inventory")
    packets = _read_jsonl(output / "packets.jsonl")
    if _sha256(output / "packets.jsonl") != receipt.get("sha256") or len(packets) != receipt.get("rows"):
        raise ValueError("sealed packet payload changed")
    if (result.get("packets") != len(packets) or not isinstance(result.get("decoded_camera_native_rgb"), bool) or
            type(result.get("temporal_slots")) is not int or type(result.get("distinct_temporal_sample_frames")) is not int or
            type(result.get("clamped_duplicate_temporal_samples")) is not int or
            result["temporal_slots"] != 7 * len(packets) or
            result["clamped_duplicate_temporal_samples"] != result["temporal_slots"] - result["distinct_temporal_sample_frames"]):
        raise ValueError("packet manifest counters or decode mode are invalid")
    decoded = result["decoded_camera_native_rgb"]
    packet_ids, event_ids = set(), set()
    for packet in packets:
        _validate_packet_semantics(packet, decoded=decoded, expected_offsets=temporal_offsets)
        if packet["packet_id"] in packet_ids or packet["actor_packet"]["event_id"] in event_ids:
            raise ValueError("packet receipt contains duplicate packet/event identities")
        packet_ids.add(packet["packet_id"])
        event_ids.add(packet["actor_packet"]["event_id"])
    selected_events = _select_index_events(index / "event_candidates.jsonl", index_receipt, event_ids, limit=None)
    events_by_id = {event["event_id"]: event for event in selected_events}
    if len(events_by_id) != len(packets):
        raise ValueError("sealed event index has duplicate/missing packet source identities")
    for packet in packets:
        _validate_packet_index_binding(packet, events_by_id[packet["actor_packet"]["event_id"]],
                                       index_manifest_sha256=result["index_manifest_sha256"], decoded=decoded,
                                       temporal_offsets=temporal_offsets)
    if (result["distinct_temporal_sample_frames"] != sum(packet["audit"]["distinct_sample_frame_count"] for packet in packets) or
            result["clamped_duplicate_temporal_samples"] != sum(
                packet["audit"]["clamped_duplicate_sample_frame_count"] for packet in packets)):
        raise ValueError("packet manifest temporal-slot accounting does not match its packet rows")
    asset_receipt = result["files"]["rendered_asset_receipts.jsonl"]
    assets = _read_jsonl(output / "rendered_asset_receipts.jsonl")
    if (_sha256(output / "rendered_asset_receipts.jsonl") != asset_receipt.get("sha256") or
            len(assets) != asset_receipt.get("rows")):
        raise ValueError("sealed rendered-asset receipt payload changed")
    if result.get("rendered_asset_receipts") != len(assets):
        raise ValueError("packet manifest rendered-asset count is invalid")
    packet_ids = {packet["packet_id"] for packet in packets}
    assets_by_packet: dict[str, list[Mapping[str, Any]]] = {packet_id_value: [] for packet_id_value in packet_ids}
    seen_asset_keys = set()
    for row in assets:
        camera_fields = {"schema_version", "packet_id", "view", "asset_kind", "temporal_role", "sample_index",
                         "sample_frame", "requested_timestamp_s", "decoded_timestamp_s", "pts_error_s",
                         "relative_path", "sha256", "bytes"}
        contact_fields = {"schema_version", "packet_id", "view", "asset_kind", "relative_path", "sha256", "bytes"}
        if set(row) not in (camera_fields, contact_fields):
            raise ValueError("rendered asset receipt has an unsealed field")
        if row.get("asset_kind") == "camera_native_rgb_png":
            if set(row) != camera_fields or row["view"] not in VIEWS or row["temporal_role"] not in {
                    "ACTOR_CAUSAL", "OFFLINE_FUTURE_AUDIT"} or type(row["sample_index"]) is not int or \
                    type(row["sample_frame"]) is not int or type(row["pts_error_s"]) not in (int, float) or \
                    row["pts_error_s"] < 0 or row["pts_error_s"] > 1.0 / 60.0 + 1e-6:
                raise ValueError("camera-native asset receipt has invalid temporal/PTS semantics")
            key = (row["packet_id"], row["sample_index"], row["view"])
            if key in seen_asset_keys:
                raise ValueError("duplicate packet temporal camera asset receipt")
            seen_asset_keys.add(key)
        elif row.get("asset_kind") != "review_only_contact_sheet_png" or set(row) != contact_fields:
            raise ValueError("rendered asset receipt has an unsupported artifact kind")
        relative = Path(row["relative_path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("rendered asset receipt escapes packet root")
        path = (output / relative).resolve(strict=True)
        if output.resolve() not in path.parents:
            raise ValueError("rendered asset receipt escapes packet root")
        if path.stat().st_size != row["bytes"] or _sha256(path) != row["sha256"]:
            raise ValueError("sealed rendered PNG bytes changed")
        if row["packet_id"] not in assets_by_packet:
            raise ValueError("rendered asset receipt refers to an absent packet")
        assets_by_packet[row["packet_id"]].append(row)
    for packet in packets:
        _validate_packet_assets(packet, assets_by_packet[packet["packet_id"]], decoded=decoded,
                                review_sheets=result["review_only_contact_sheets"])
    return {"status": "RESUME_VALIDATED", "manifest": result}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--event-id", action="append", default=[])
    parser.add_argument("--event-ids", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--questions", type=Path, help="JSONL actor question contexts, never labels")
    parser.add_argument("--include-source-annotation-context", action="store_true")
    parser.add_argument("--decode", action="store_true", help="decode bounded native RGB and PTS receipts")
    parser.add_argument("--raw-root", type=Path)
    parser.add_argument("--temporal-offset-frames", type=_parse_temporal_offsets,
                        default=TEMPORAL_SAMPLE_OFFSETS,
                        help="exact seven source-frame offsets incl. 0, max +/-240; future samples are audit-only")
    parser.add_argument("--contact-sheets", action="store_true", help="review-only composites; excluded from actor input")
    parser.add_argument("--max-seconds", type=float, default=1800.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--expected-packet-manifest-sha256",
                        help="external SHA-256 returned at packet creation; mandatory with --resume")
    args = parser.parse_args()
    if args.resume and args.expected_packet_manifest_sha256 is None:
        parser.error("--resume requires --expected-packet-manifest-sha256 from an external authority record")
    result = (resume_packets(args.index, args.output,
                             expected_packet_manifest_sha256=args.expected_packet_manifest_sha256) if args.resume else create_packets(
        args.index, args.output, event_ids=_read_event_ids(args.event_ids, args.event_id), limit=args.limit,
        questions=_questions(args.questions), include_source_annotation_context=args.include_source_annotation_context,
        decode=args.decode, raw_root=args.raw_root, contact_sheets=args.contact_sheets,
        temporal_offsets_frames=args.temporal_offset_frames,
        max_seconds=args.max_seconds))
    print(canonical_json(result), flush=True)


if __name__ == "__main__":
    main()
