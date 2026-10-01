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


VIEWS = ("head", "left_wrist", "right_wrist")
PACKET_INDEX_SCHEMA = "memlite-event-packet-index-v1"
PACKET_PAYLOAD_FILES = frozenset(("packets.jsonl", "rendered_asset_receipts.jsonl"))


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
        forbidden = ("answer", "label", "outcome", "ground_truth", "success", "failure", "review", "evidence")
        if any(any(token in key.lower() for token in forbidden) for key in context):
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


def _packet_id(index_sha256: str, event: Mapping[str, Any], context: Mapping[str, Any], decode: bool) -> str:
    return canonical_sha256({
        "schema_version": PACKET_SCHEMA_VERSION,
        "index_manifest_sha256": index_sha256,
        "event_id": event["event_id"],
        "observation_frame": event["observation"]["frame"],
        "question_context": context,
        "decoded_camera_native_rgb": decode,
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


def _decode_rgb(av: Any, video: Path, requested: float) -> tuple[Any, dict[str, Any]]:
    """Decode nearest source PTS under half-frame tolerance; never trust frame ordinal alone."""
    with av.open(str(video)) as container:
        if not container.streams.video:
            raise ValueError(f"source has no video stream: {video}")
        stream = container.streams.video[0]
        rate = float(stream.average_rate) if stream.average_rate is not None else float("nan")
        if abs(rate - 30.0) > 1e-6:
            raise ValueError(f"source FPS changed: {video} ({rate})")
        if stream.time_base is None:
            raise ValueError(f"source stream has no time base: {video}")
        container.seek(max(0, int(requested / float(stream.time_base))), stream=stream, backward=True, any_frame=False)
        best = None
        for frame in container.decode(stream):
            if frame.pts is None:
                continue
            actual = float(frame.pts * stream.time_base)
            distance = abs(actual - requested)
            if best is None or distance < best[0]:
                best = (distance, actual, frame.to_image().convert("RGB"))
            if actual >= requested:
                break
        if best is None or best[0] > 1.0 / 60.0 + 1e-6:
            if best is not None:
                best[2].close()
            raise ValueError(f"cannot decode requested source PTS within half-frame: {video}")
        stat = video.stat()
        return best[2], {
            "resolved_path": str(video), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns,
            "requested_timestamp_s": requested, "decoded_timestamp_s": best[1],
            "pts_error_s": best[0], "fps": rate, "resolution": list(best[2].size),
            "full_video_sha256": None,
        }


def _review_contact_sheet(images: Mapping[str, Any], path: Path, Image: Any, ImageDraw: Any) -> None:
    """A review-only composite; it is intentionally excluded from actor_packet.images."""
    sizes = [images[view].size for view in VIEWS]
    width = sum(size[0] for size in sizes)
    height = max(size[1] for size in sizes) + 30
    sheet = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(sheet)
    x = 0
    for view in VIEWS:
        image = images[view]
        sheet.paste(image, (x, 30))
        draw.text((x + 4, 6), view, fill="black")
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
    event_rows = _read_jsonl(index / "event_candidates.jsonl")
    receipt = manifest.get("files", {}).get("event_candidates.jsonl", {})
    if _sha256(index / "event_candidates.jsonl") != receipt.get("sha256") or len(event_rows) != receipt.get("rows"):
        raise ValueError("event index payload does not match its sealed manifest")
    selected = [event for event in event_rows if not event_ids or event["event_id"] in event_ids]
    if event_ids and {event["event_id"] for event in selected} != event_ids:
        raise ValueError("requested event_id absent from sealed event index")
    if limit is not None:
        selected = selected[:limit]
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
            if include_source_annotation_context:
                if context:
                    raise ValueError("use either explicit question context or source annotation context, not both")
                context = _source_annotation_context(event)
            packet_id = _packet_id(index_sha, event, context, decode)
            if packet_id in seen_packet_ids:
                raise ValueError("duplicate packet_id in selected immutable events")
            seen_packet_ids.add(packet_id)
            by_view = _locators_by_view(event)
            actor_images: dict[str, str] = {}
            decoded_receipts: dict[str, Any] = {}
            review_asset = None
            if decode:
                images = {}
                try:
                    for view in VIEWS:
                        locator = by_view[view]
                        video = _resolve_video(raw_root, locator["relative_path"])
                        image, decoded = _decode_rgb(av, video, float(locator["requested_timestamp_s"]))
                        native_name = f"{packet_id}_{view}.png"
                        asset_path = staging / "assets" / native_name
                        image.save(asset_path)
                        actor_images[view] = f"assets/{native_name}"
                        decoded_receipts[view] = {**decoded, "camera_native_png_sha256": _sha256(asset_path),
                                                  "camera_native_png_bytes": asset_path.stat().st_size}
                        asset_receipts.append({
                            "schema_version": PACKET_SCHEMA_VERSION, "packet_id": packet_id, "view": view,
                            "asset_kind": "camera_native_rgb_png", "relative_path": f"assets/{native_name}",
                            "sha256": _sha256(asset_path), "bytes": asset_path.stat().st_size,
                        })
                        images[view] = image
                    if contact_sheets:
                        review_name = f"{packet_id}_contact_sheet.png"
                        _review_contact_sheet(images, staging / "review_assets" / review_name, Image, ImageDraw)
                        review_asset = f"review_assets/{review_name}"
                        review_path = staging / review_asset
                        asset_receipts.append({
                            "schema_version": PACKET_SCHEMA_VERSION, "packet_id": packet_id, "view": None,
                            "asset_kind": "review_only_contact_sheet_png", "relative_path": review_asset,
                            "sha256": _sha256(review_path), "bytes": review_path.stat().st_size,
                        })
                finally:
                    for image in images.values():
                        image.close()
            actor_packet = {
                "packet_id": packet_id, "event_id": event["event_id"],
                "observation_frame": event["observation"]["frame"],
                "images": actor_images,
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
                    "decoded_pts_receipts": decoded_receipts, "review_only_contact_sheet": review_asset,
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


def resume_packets(index: Path, output: Path, *, expected_packet_manifest_sha256: str | None) -> dict[str, Any]:
    index, output = Path(index), Path(output)
    if (not isinstance(expected_packet_manifest_sha256, str) or len(expected_packet_manifest_sha256) != 64 or
            any(char not in "0123456789abcdef" for char in expected_packet_manifest_sha256)):
        raise ValueError("--resume requires an externally pinned --expected-packet-manifest-sha256")
    manifest_path = output / "manifest.json"
    if _sha256(manifest_path) != expected_packet_manifest_sha256:
        raise ValueError("packet manifest does not match the externally pinned authority SHA-256")
    result = _read_json(manifest_path)
    if result.get("schema_version") != PACKET_INDEX_SCHEMA:
        raise ValueError("--resume requires a sealed P107 packet directory")
    if result.get("index_manifest_sha256") != _sha256(index / "manifest.json"):
        raise ValueError("source index manifest changed; refusing resume")
    receipt = result.get("files", {}).get("packets.jsonl", {})
    if set(result.get("files", {})) != PACKET_PAYLOAD_FILES:
        raise ValueError("packet manifest must seal the exact packet and rendered-asset receipt inventory")
    packets = _read_jsonl(output / "packets.jsonl")
    if _sha256(output / "packets.jsonl") != receipt.get("sha256") or len(packets) != receipt.get("rows"):
        raise ValueError("sealed packet payload changed")
    asset_receipt = result["files"]["rendered_asset_receipts.jsonl"]
    assets = _read_jsonl(output / "rendered_asset_receipts.jsonl")
    if (_sha256(output / "rendered_asset_receipts.jsonl") != asset_receipt.get("sha256") or
            len(assets) != asset_receipt.get("rows")):
        raise ValueError("sealed rendered-asset receipt payload changed")
    for row in assets:
        if set(row) != {"schema_version", "packet_id", "view", "asset_kind", "relative_path", "sha256", "bytes"}:
            raise ValueError("rendered asset receipt has an unsealed field")
        relative = Path(row["relative_path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("rendered asset receipt escapes packet root")
        path = (output / relative).resolve(strict=True)
        if output.resolve() not in path.parents:
            raise ValueError("rendered asset receipt escapes packet root")
        if path.stat().st_size != row["bytes"] or _sha256(path) != row["sha256"]:
            raise ValueError("sealed rendered PNG bytes changed")
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
        max_seconds=args.max_seconds))
    print(canonical_json(result), flush=True)


if __name__ == "__main__":
    main()
