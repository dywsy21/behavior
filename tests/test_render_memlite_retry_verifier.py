from __future__ import annotations

import hashlib
import importlib.util
import json
import tempfile
import unittest
from fractions import Fraction
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "data" / "render_memlite_retry_verifier.py"
SPEC = importlib.util.spec_from_file_location("private_retry_verifier_test_module", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
verifier = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(verifier)


SELECTION_PATH = Path("/home/wsy/behavior-annotations/p107/natural-retry-pilot-selection-v1/private-rgb-selection.json")
SOURCE_RESULT_PATH = Path("/home/wsy/behavior-annotations/p107/natural-action-probe-v1/natural-retry-pilot-v1-results/retry-candidates.json")
SELECTION_SHA = "84c2aa626fddab3a36ac333c1fa2e13b59192ce9c9e12f8cb135e60e0c2971c2"
SOURCE_RESULT_SHA = "4eb0227cf785e6f201be837e1f263d0339869ac1228dbe530e83b44e611e97d6"


class _FakeFrame:
    def __init__(self, pts: int, view: str, colour: tuple[int, int, int]) -> None:
        self.pts = pts
        self._view = view
        self._colour = colour

    def to_image(self) -> Image.Image:
        image = Image.new("RGB", (18, 12), self._colour)
        image.info["view"] = self._view
        return image


class _FakeStream:
    average_rate = 30
    time_base = Fraction(1, 30)


class _FakeContainer:
    streams = type("Streams", (), {"video": [_FakeStream()]})()

    def __init__(self, path: Path, bias_ticks: int = 0) -> None:
        self.path = path
        self.bias_ticks = bias_ticks
        self.target = 0

    def __enter__(self) -> "_FakeContainer":
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def seek(self, offset: int, **_kwargs: object) -> None:
        self.target = offset

    def decode(self, _stream: object):
        # _decode_rgb seeks just before the requested tick.  Emit the exact
        # tick and a controllable biased tick so the real selector/rank logic
        # is exercised without installing PyAV.
        view = self.path.stem
        colour = (17, 71, 131) if "head" in view else ((41, 101, 53) if "left" in view else (151, 53, 47))
        yield _FakeFrame(self.target, view, colour)
        yield _FakeFrame(self.target + 1 + self.bias_ticks, view, colour)


class _FakeAV:
    def __init__(self, bias_ticks: int = 0) -> None:
        self.bias_ticks = bias_ticks

    def open(self, path: str) -> _FakeContainer:
        return _FakeContainer(Path(path), self.bias_ticks)


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


class PrivateRetryVerifierTests(unittest.TestCase):
    def _make_event_index(self, root: Path, raw_root: Path) -> tuple[Path, str]:
        root.mkdir(parents=True, exist_ok=True)
        selection = json.loads(SELECTION_PATH.read_text())
        source_result = json.loads(SOURCE_RESULT_PATH.read_text())
        episodes = {
            episode["source_identity"]["event_id"]: episode
            for episode in source_result["episodes"]
        }
        event_ids = []
        for row in selection["selections"]:
            event_id = row["source"]["event_id"]
            if event_id not in event_ids:
                event_ids.append(event_id)
        event_rows = []
        for event_number, event_id in enumerate(event_ids):
            episode = episodes[event_id]
            source_identity = episode["source_identity"]
            length = episode["length"]
            locators = []
            for view in verifier.VIEWS:
                relative = f"videos/{event_id}/{view}.mp4"
                path = raw_root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"synthetic tinyvideo placeholder")
                locators.append({
                    "camera_key": f"synthetic.{view}",
                    "episode_start_timestamp_s": 1000.0 + event_number * 100.0 + {
                        "head": 0.0, "left_wrist": 1000.0, "right_wrist": 2000.0,
                    }[view],
                    "expected_fps": 30,
                    "locator_status": "METADATA_ONLY_UNRESOLVED",
                    "relative_path": relative,
                    "requested_timestamp_s": 1000.0 + event_number * 100.0,
                    "view": view,
                })
            event_rows.append({
                "schema_version": "memlite-event-recovery-v1",
                "record_kind": "event_candidate",
                "event_id": event_id,
                "usage_role": "annotation_calibration",
                "source": {
                    **{key: value for key, value in source_identity.items() if key != "event_id"},
                    "episode_length": length,
                    "original_split": "train",
                    "source_annotation_sha256": episode["annotation_sha256"],
                    "source_release_manifest_sha256": source_result["inputs"]["release_manifest_sha256"],
                },
                "video_locators": locators,
            })
        event_bytes = b"".join(_json_bytes(row) for row in event_rows)
        event_path = root / "event_candidates.jsonl"
        event_path.write_bytes(event_bytes)
        receipt = {"sha256": hashlib.sha256(event_bytes).hexdigest(), "rows": len(event_rows), "bytes": len(event_bytes)}
        manifest = {
            "schema_version": "memlite-event-index-v1",
            "source_release_manifest_sha256": source_result["inputs"]["release_manifest_sha256"],
            "files": {"event_candidates.jsonl": receipt},
        }
        manifest_path = root / "manifest.json"
        manifest_path.write_bytes(_json_bytes(manifest))
        return root, hashlib.sha256(manifest_path.read_bytes()).hexdigest()

    def test_real_plan_is_161_frames_and_keeps_s08_split(self) -> None:
        selection = verifier._load_json(SELECTION_PATH, "selection")
        _meta, candidates = verifier._load_source_result(SOURCE_RESULT_PATH, SOURCE_RESULT_SHA, selection)
        normalized, wanted = verifier._validate_selection_and_candidates(selection, candidates, "a" * 64)
        self.assertEqual(len(normalized), 8)
        self.assertEqual(len(wanted), 6)  # s03/s06 and s01/s07 share source episodes.
        self.assertEqual(sum(len(row["plan"]) for row in normalized), 161)
        s08 = next(row for row in normalized if row["selection"]["selection_id"] == "s08")
        self.assertEqual(len(s08["selection"]["render_windows_local"]), 2)
        self.assertLess(s08["selection"]["render_windows_local"][0][1], s08["selection"]["render_windows_local"][1][0])

    def test_native_decoder_handles_three_cameras_boundaries_and_pts_bias(self) -> None:
        renderer = verifier._renderer()
        with tempfile.TemporaryDirectory() as folder:
            video = Path(folder) / "head.mp4"
            video.write_bytes(b"synthetic tinyvideo")
            start, end = 100.0, 100.0 + 9.0 / 30.0
            image, receipt = renderer._decode_rgb(
                _FakeAV(), video, start,
                episode_start_timestamp_s=start,
                episode_end_timestamp_s=end,
                actor_anchor_timestamp_s=None,
            )
            image.close()
            self.assertEqual(receipt["decoded_timestamp_s"], start)
            image, receipt = renderer._decode_rgb(
                _FakeAV(), video, end,
                episode_start_timestamp_s=start,
                episode_end_timestamp_s=end,
                actor_anchor_timestamp_s=None,
            )
            image.close()
            self.assertEqual(receipt["decoded_timestamp_s"], end)
            with self.assertRaises(ValueError):
                renderer._decode_rgb(
                    _FakeAV(bias_ticks=1), video, start + 3.0 / 30.0,
                    episode_start_timestamp_s=start,
                    episode_end_timestamp_s=end,
                    actor_anchor_timestamp_s=None,
                )

    def test_synthetic_metadata_run_writes_483_native_pngs_and_9_private_pages(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            event_index, event_manifest_sha = self._make_event_index(root / "index", root / "raw")
            output = root / "private-verifier-output"
            result = verifier.run_verifier(
                SELECTION_PATH,
                SOURCE_RESULT_PATH,
                event_index,
                root / "raw",
                output,
                expected_selection_sha256=SELECTION_SHA,
                expected_source_result_sha256=SOURCE_RESULT_SHA,
                expected_event_index_manifest_sha256=event_manifest_sha,
                av_backend=_FakeAV(),
            )
            self.assertEqual(result["native_png_count"], 483)
            self.assertEqual(result["overview_page_count"], 9)
            rows = [json.loads(line) for line in (output / "frames.jsonl").read_text().splitlines()]
            self.assertEqual(len(rows), 483)
            self.assertTrue(all(row["role"] == "private_verifier_only" for row in rows))
            self.assertTrue(all(row["actor_packet_included"] is False for row in rows))
            for row in rows:
                expected_request = row["camera_video_locator"]["episode_start_timestamp_s"] + row["local_frame"] / 30.0
                self.assertAlmostEqual(row["requested_timestamp_s"], expected_request, places=12)
                self.assertAlmostEqual(row["decoded_timestamp_s"], row["requested_timestamp_s"], places=12)
                self.assertAlmostEqual(row["pts_error_s"], 0.0, places=12)
            s08 = [row for row in rows if row["selection_id"] == "s08"]
            self.assertEqual({row["window_index"] for row in s08}, {0, 1})
            self.assertEqual(len({row["camera_view"] for row in s08}), 3)
            self.assertEqual(len(list((output / "native_rgb").rglob("*.png"))), 483)
            self.assertEqual(len(list((output / "private_verifier_overviews").glob("*.png"))), 9)
            with self.assertRaises(FileExistsError):
                verifier.run_verifier(
                    SELECTION_PATH,
                    SOURCE_RESULT_PATH,
                    event_index,
                    root / "raw",
                    output,
                    expected_selection_sha256=SELECTION_SHA,
                    expected_source_result_sha256=SOURCE_RESULT_SHA,
                    expected_event_index_manifest_sha256=event_manifest_sha,
                    av_backend=_FakeAV(),
                )

    def test_event_annotation_sha_is_bound_to_source_result(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            event_index, _manifest_sha = self._make_event_index(root / "index", root / "raw")
            event = json.loads((event_index / "event_candidates.jsonl").read_text().splitlines()[0])
            source_result = json.loads(SOURCE_RESULT_PATH.read_text())
            episode = next(
                row for row in source_result["episodes"]
                if row["source_identity"]["event_id"] == event["event_id"]
            )
            for annotation_value in ("0" * 64, None):
                bad_event = json.loads(json.dumps(event))
                if annotation_value is None:
                    bad_event["source"].pop("source_annotation_sha256")
                else:
                    bad_event["source"]["source_annotation_sha256"] = annotation_value
                with self.assertRaises(verifier.InputValidationError):
                    verifier._validate_video_locators(
                        bad_event,
                        episode["source_identity"],
                        episode["length"],
                        source_result["inputs"]["release_manifest_sha256"],
                        episode["annotation_sha256"],
                    )

            bad_event = json.loads(json.dumps(event))
            bad_event["event_id"] = "0" * 64
            with self.assertRaises(verifier.InputValidationError):
                verifier._validate_video_locators(
                    bad_event,
                    episode["source_identity"],
                    episode["length"],
                    source_result["inputs"]["release_manifest_sha256"],
                    episode["annotation_sha256"],
                )


if __name__ == "__main__":
    unittest.main()
