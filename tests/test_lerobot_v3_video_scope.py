"""Regression: RGB-only source metadata may still declare depth video streams."""
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import datasets
import pandas as pd
import torch

from g05.data.lerobot import lerobot_dataset_v3 as module
from g05.data.base_lerobot_dataset import BaseLerobotDataset

RGB = "observation.rgb.head"
DEPTH = "observation.depth.head"


class VideoScopeTests(unittest.TestCase):
    def patched(self, patcher):
        value = patcher.start()
        self.addCleanup(patcher.stop)
        return value

    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.meta = SimpleNamespace(
            video_keys=[RGB, DEPTH], camera_keys=[RGB, DEPTH], total_frames=3,
            total_episodes=1, fps=30,
            episodes=datasets.Dataset.from_dict({"dataset_from_index": [0], "dataset_to_index": [3]}),
            tasks=pd.DataFrame({"task_index": [0]}, index=["test task"]),
            features={RGB: {"dtype": "video"}, DEPTH: {"dtype": "video"},
                      "action": {"dtype": "float32"}, "observation.state": {"dtype": "float32"}},
            get_video_file_path=lambda ep, key: Path(f"videos/{key}/file-000.mp4"),
            get_data_file_path=lambda ep: Path("data/chunk-000/file-000.parquet"),
        )
        self.rgb_path = self.root / self.meta.get_video_file_path(0, RGB)
        self.rgb_path.parent.mkdir(parents=True)
        self.rgb_path.touch()
        self.hf = datasets.Dataset.from_dict({
            "episode_index": [0]*3, "task_index": [0]*3, "timestamp": [0., 1/30, 2/30],
            "index": [0, 1, 2], "frame_index": [0, 1, 2],
            "action": [[0.]*23]*3, "observation.state": [[0.]*61]*3,
        })
        self.hf.set_transform(module.hf_transform_to_torch)
        self.patched(patch.object(module, "LeRobotDatasetMetadata", return_value=self.meta))
        self.patched(patch.object(module.LeRobotDataset, "load_hf_dataset", return_value=self.hf))
        self.hub = self.patched(patch.object(module, "snapshot_download",
                                                  side_effect=AssertionError("Unexpected Hub access")))

    def create(self, **kwargs):
        return module.LeRobotDataset("test/source", self.root, episodes=[0], **kwargs)

    def test_selected_cache_and_real_getitem_do_not_touch_depth(self):
        ds = self.create(video_keys=[RGB], local_files_only=True,
                         delta_timestamps={RGB: [0.], "action": [0., 1/30]},
                         image_transforms=lambda x: x + 1)
        self.assertTrue(ds._check_cached_episodes_sufficient())
        self.assertEqual(ds.read_video_keys, [RGB])
        self.assertEqual(self.meta.video_keys, [RGB, DEPTH])
        self.assertTrue(all(DEPTH not in p for p in ds.get_episodes_file_paths()))
        with patch.object(ds, "_query_videos", return_value={RGB: torch.zeros(1, 3, 8, 8)}) as decode:
            sample = ds[0]
        self.assertEqual(set(decode.call_args.args[0]), {RGB})
        self.assertNotIn(DEPTH, sample)
        self.assertEqual(tuple(sample["action"].shape), (2, 23))
        self.assertTrue(torch.all(sample[RGB] == 1))
        self.hub.assert_not_called()

    def test_missing_requested_rgb_fails_without_download(self):
        self.rgb_path.unlink()
        with self.assertRaisesRegex(FileNotFoundError, "automatic Hub download is disabled"):
            self.create(video_keys=[RGB], local_files_only=True)
        self.hub.assert_not_called()

    def test_default_scope_still_requires_depth(self):
        with self.assertRaises(FileNotFoundError):
            self.create(local_files_only=True)
        self.hub.assert_not_called()

    def test_invalid_video_selections_fail_early(self):
        for keys in ([RGB, RGB], ["unknown"]):
            with self.assertRaises(ValueError):
                self.create(video_keys=keys, local_files_only=True)
        with self.assertRaisesRegex(ValueError, "outside video_keys"):
            self.create(video_keys=[RGB], local_files_only=True, delta_timestamps={DEPTH: [0.]})

    def test_filtered_download_allowlist_and_explicit_offline_guard(self):
        ds = self.create(video_keys=[RGB], local_files_only=True)
        with self.assertRaisesRegex(RuntimeError, "disabled"):
            ds.pull_from_repo()
        ds.local_files_only = False
        ds.episodes = None
        with patch.object(ds, "pull_from_repo") as download:
            ds.download()
        allowed = download.call_args.kwargs["allow_patterns"]
        self.assertIn(f"videos/{RGB}/**", allowed)
        self.assertTrue(all(DEPTH not in p for p in allowed))
        self.assertNotIn("videos/**", allowed)
        with patch.object(ds, "pull_from_repo") as download:
            ds.download(download_videos=False)
        self.assertEqual(download.call_args.kwargs["ignore_patterns"], "videos/**")

    def test_empty_video_selection_allows_action_only(self):
        ds = self.create(video_keys=[], local_files_only=True)
        self.assertEqual(ds._get_query_timestamps(0.), {})
        self.assertEqual(ds.get_episodes_file_paths(), ["data/chunk-000/file-000.parquet"])

    def test_multidataset_does_not_swallow_local_source_failure(self):
        self.rgb_path.unlink()
        with self.assertRaises(FileNotFoundError):
            module.MultiLeRobotDataset([str(self.root)], video_keys=[RGB], local_files_only=True)

    def test_cache_separates_scope_and_offline_policy(self):
        args = (["/dataset"], {RGB: [0.]}, {"/dataset": .01}, True)
        base = BaseLerobotDataset._make_cache_key(*args)
        rgb = BaseLerobotDataset._make_cache_key(*args, video_keys=[RGB])
        offline = BaseLerobotDataset._make_cache_key(*args, video_keys=[RGB], local_files_only=True)
        self.assertEqual(len({base, rgb, offline}), 3)


if __name__ == "__main__":
    unittest.main()
