"""Render a sealed P107 ``evaluation_only`` coverage handoff.

This is intentionally a separate entrypoint from the TRAIN renderer.  It
requires an externally pinned selector manifest and selection seal, validates
the selected events against the immutable full index, and calls the shared
packet/decode backend with an explicit evaluation role.  It never accepts a
TRAIN queue seal and never emits a training queue or outcome/action labels.
"""
from __future__ import annotations

import argparse
import os
import shutil
import tempfile
from pathlib import Path

import publish_memlite_coverage_render_handoff as handoff
import render_memlite_event_packets as renderer

DEFAULT_EVAL_OFFSETS = (-60, -45, -30, -15, 0, 1, 16, 31, 46, 60)


def render_evaluation_packets(*, index: Path, selector_output: Path, output: Path,
                              coverage_path: Path, protocol_path: Path,
                              expected_protocol_sha256: str,
                              expected_source_release_sha256: str,
                              expected_inventory_sha256: str,
                              expected_coverage_sha256: str,
                              expected_selector_manifest_sha256: str,
                              expected_selector_selection_seal_sha256: str,
                              questions: Path | None = None,
                              include_source_annotation_context: bool = False,
                              decode: bool = False, raw_root: Path | None = None,
                              contact_sheets: bool = False,
                              temporal_offsets: tuple[int, ...] = DEFAULT_EVAL_OFFSETS,
                              max_seconds: float = 1800.0) -> dict[str, object]:
    """Render EVAL packets and atomically publish the packet plus its receipt."""
    output = Path(output).absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"EVAL packet output already exists: {output}")
    if not output.parent.is_dir() or output.parent.is_symlink():
        raise ValueError("EVAL packet output parent must be an existing regular directory")
    selector = handoff.read_selector_handoff(
        selector_output, role="eval", expected_manifest_sha256=expected_selector_manifest_sha256,
        expected_selection_seal_sha256=expected_selector_selection_seal_sha256)
    parent = handoff._load_parent_index(
        index, selector, expected_source_release_sha256=expected_source_release_sha256,
        expected_inventory_sha256=expected_inventory_sha256, coverage_path=coverage_path,
        expected_coverage_sha256=expected_coverage_sha256, protocol_path=protocol_path,
        expected_protocol_sha256=expected_protocol_sha256)
    requests, request_sha = renderer._read_render_requests(
        selector.request_path, handoff._sha256(selector.request_path))
    if set(requests) != set(selector.event_ids):
        raise ValueError("EVAL render requests do not cover exactly the selected event IDs")
    if contact_sheets and not decode:
        raise ValueError("--contact-sheets requires --decode")

    # The shared packet backend defaults to the repository's TRAIN protocol
    # binding.  EVAL receipts pin their own protocol snapshot, so load that
    # exact binding before any packet assembly; otherwise a valid sealed
    # evaluation index could be rejected (or, worse, interpreted under a
    # different canonicalizer).
    protocol_binding = renderer._load_protocol(
        protocol_path, expected_sha256=expected_protocol_sha256)

    staging_parent = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=str(output.parent)))
    packet_stage = staging_parent / "packets"
    try:
        result = renderer.create_packets(
            Path(index), packet_stage, event_ids=set(selector.event_ids), limit=None,
            questions=renderer._questions(questions) if questions is not None else {},
            include_source_annotation_context=include_source_annotation_context,
            decode=decode, raw_root=raw_root, contact_sheets=contact_sheets,
            temporal_offsets_frames=temporal_offsets, render_requests=requests,
            render_requests_sha256=request_sha, queue_seal_sha256=selector.selection_seal_sha256,
            queue_seal_source_release_manifest_sha256=expected_source_release_sha256,
            expected_usage_role="evaluation_only", max_seconds=max_seconds,
            protocol_binding=protocol_binding)
        receipt = handoff.build_evaluation_render_receipt(
            selector=selector, parent=parent, packet_dir=packet_stage,
            packet_manifest_sha256=str(result["packet_manifest_sha256"]))
        handoff._write_json(packet_stage / "evaluation_render_receipt.json", receipt)
        receipt_sha256 = handoff._sha256(packet_stage / "evaluation_render_receipt.json")
        os.rename(packet_stage, output)
        staging_parent.rmdir()
        return {
            "status": "EVALUATION_ONLY_RENDER_READY",
            "output": str(output),
            "packet_manifest_sha256": result["packet_manifest_sha256"],
            "evaluation_receipt_sha256": receipt_sha256,
            "selector_selection_seal_sha256": selector.selection_seal_sha256,
            "selector_request_sha256": request_sha,
            "packets": result["packets"],
            "training_eligible": False,
            "usage_role": "evaluation_only",
            "immutable_split": "eval",
        }
    except BaseException:
        shutil.rmtree(staging_parent, ignore_errors=True)
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--selector-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--coverage-expectations", type=Path, required=True)
    parser.add_argument("--protocol-path", type=Path, required=True)
    parser.add_argument("--expected-protocol-sha256", required=True)
    parser.add_argument("--expected-source-release-manifest-sha256", required=True)
    parser.add_argument("--expected-inventory-seal-sha256", required=True)
    parser.add_argument("--expected-coverage-expectations-sha256", required=True)
    parser.add_argument("--expected-selector-manifest-sha256", required=True)
    parser.add_argument("--expected-selector-selection-seal-sha256", required=True)
    parser.add_argument("--questions", type=Path)
    parser.add_argument("--include-source-annotation-context", action="store_true")
    parser.add_argument("--decode", action="store_true")
    parser.add_argument("--raw-root", type=Path)
    parser.add_argument("--contact-sheets", action="store_true")
    parser.add_argument("--temporal-offset-frames", type=renderer._parse_temporal_offsets,
                        default=DEFAULT_EVAL_OFFSETS)
    parser.add_argument("--max-seconds", type=float, default=1800.0)
    return parser


def main() -> None:
    args = _parser().parse_args()
    result = render_evaluation_packets(
        index=args.index, selector_output=args.selector_output, output=args.output,
        coverage_path=args.coverage_expectations, protocol_path=args.protocol_path,
        expected_protocol_sha256=args.expected_protocol_sha256,
        expected_source_release_sha256=args.expected_source_release_manifest_sha256,
        expected_inventory_sha256=args.expected_inventory_seal_sha256,
        expected_coverage_sha256=args.expected_coverage_expectations_sha256,
        expected_selector_manifest_sha256=args.expected_selector_manifest_sha256,
        expected_selector_selection_seal_sha256=args.expected_selector_selection_seal_sha256,
        questions=args.questions, include_source_annotation_context=args.include_source_annotation_context,
        decode=args.decode, raw_root=args.raw_root, contact_sheets=args.contact_sheets,
        temporal_offsets=args.temporal_offset_frames, max_seconds=args.max_seconds,
    )
    print(handoff.base.canonical_json(result), flush=True)


if __name__ == "__main__":
    main()
