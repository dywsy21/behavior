"""Fail-closed RGB-only reader for the compact MEM-Lite stage-1 release."""
from __future__ import annotations
from collections import OrderedDict
import hashlib
import json
from pathlib import Path
import random
import numpy as np
import torch
from torch.utils.data import Dataset

from g05.data.memlite_stage1_labels import anchor_records, projection
from g05.utils.training.stage1_model import make_processor


class Stage1Dataset(Dataset):
    def __init__(self, release, config, branch, split, *, epoch=0):
        self.release, self.config, self.branch, self.split = Path(release), config, branch, split
        self.manifest = json.loads((self.release / "manifest.json").read_text())
        self.root = Path(self.manifest["raw_root"])
        self.candidates = np.load(self.release / f"{split}_candidates.npy", mmap_mode="r")
        self.task_ids = np.load(self.release / f"{split}_tasks.npy", mmap_mode="r")
        self.offsets = np.load(self.release / "episode_offsets.npy", mmap_mode="r")
        self.epoch = int(epoch)
        self.processor = None
        self._episodes = OrderedDict()
        self._table = OrderedDict()
        self._file = None

    def __len__(self):
        return len(self.candidates)

    def episode(self, serial):
        if serial not in self._episodes:
            if self._file is None:
                self._file = (self.release / "episodes.jsonl").open("rb")
            self._file.seek(int(self.offsets[serial]))
            record = json.loads(self._file.readline())
            record["anchors"] = anchor_records(record["segments"], record["phase"])
            self._episodes[serial] = record
            if len(self._episodes) > 8:
                self._episodes.popitem(last=False)
        self._episodes.move_to_end(serial)
        return self._episodes[serial]

    def locate(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        serial, anchor = (int(x) for x in self.candidates[index])
        episode = self.episode(serial)
        frame, segment, previous, parent, history = episode["anchors"][anchor]
        return episode, frame, segment, previous, parent, history

    def _raw_arrays(self, row):
        import pyarrow.dataset as ds
        episode = int(row["episode_index"])
        if episode not in self._table:
            path = self.root / f"data/chunk-{row['data/chunk_index']:03d}/file-{row['data/file_index']:03d}.parquet"
            table = ds.dataset(path, format="parquet").to_table(columns=["frame_index", "timestamp", "action", "observation.state"],
                filter=ds.field("episode_index") == episode, use_threads=False).combine_chunks()
            frames = table["frame_index"].to_numpy()
            if not np.array_equal(frames, np.arange(row["length"])):
                raise ValueError(f"Episode {episode}: frame clock is noncontiguous or reordered")
            if not np.allclose(table["timestamp"].to_numpy(), frames / 30., rtol=0, atol=.4/30):
                raise ValueError(f"Episode {episode}: timestamp mismatch")
            arrays = {}
            for key, dim in (("action", 23), ("observation.state", 61)):
                col = table[key].chunk(0)
                arr = col.values.to_numpy(zero_copy_only=False).reshape(-1, dim).astype(np.float32, copy=False)
                if arr.shape != (row["length"], dim) or not np.isfinite(arr).all():
                    raise ValueError(f"Episode {episode}: invalid {key}")
                arrays[key] = arr
            self._table[episode] = arrays
            if len(self._table) > 4:
                self._table.popitem(last=False)
        self._table.move_to_end(episode)
        return self._table[episode]

    def raw(self, index, *, images=True):
        from g05.data.lerobot.datasets.video_utils import decode_video_frames_torchcodec
        ep, frame, segment_index, previous, parent, history = self.locate(index)
        row, segment = ep["row"], ep["segments"][segment_index]
        if row["task_index"] != int(self.task_ids[index]) or ep["split"] != self.split:
            raise ValueError("Manifest/locator mismatch; refusing random replacement")
        arrays = self._raw_arrays(row)
        valid = min(32, row["length"] - frame, segment["end"] - frame)
        if not 0 < valid <= 32 or frame % 16 != ep["phase"]:
            raise ValueError("Wrong stride phase or action clock")
        action_rows = np.minimum(np.arange(frame, frame + 32), frame + valid - 1)
        action, state = arrays["action"][action_rows], arrays["observation.state"][frame:frame + 1]
        shape = self.config["raw_shape"]
        result = dict(idx=int(index), task=ep["task_name"], embodiment="galaxea_r1pro", images={},
                      action_is_pad=torch.arange(32) >= valid, state_is_pad=torch.zeros(1, dtype=torch.bool),
                      image_is_pad=torch.zeros(1, dtype=torch.bool), frequency=30,
                      model_projection=projection(segment, branch=self.branch, task_name=ep["task_name"],
                          previous_intent=previous, previous_parent=parent, history=history))
        for domain, values in (("action", action), ("state", state)):
            result[domain] = {m["key"]: torch.from_numpy(values[:, m["start_index"]:m["start_index"] + m["raw_shape"]].copy())
                              for m in shape[domain]}
        if images:
            for meta in shape["images"]:
                key = "videos/" + meta["lerobot_key"]
                path = self.root / f"{key}/chunk-{row[key+'/chunk_index']:03d}/file-{row[key+'/file_index']:03d}.mp4"
                timestamp = row[key + "/from_timestamp"] + frame / 30.
                result["images"][meta["key"]] = decode_video_frames_torchcodec(
                    path, [timestamp], tolerance_s=.4/30, device="cpu")
        return result, dict(candidate=int(index), episode=int(row["episode_index"]), task=int(row["task_index"]),
                            frame=frame, segment_end=segment["end"], valid_action_steps=valid,
                            raw_episode=int(row["raw_episode_id"]), instance=int(row["task_instance_id"]))

    def __getitem__(self, index):
        if self.processor is None:
            self.processor = make_processor(self.config, training=self.split == "train")
        raw, identity = self.raw(int(index))
        # Transform randomness is a pure function of committed identity/epoch,
        # independent of worker scheduling and DataLoader prefetch on resume.
        key = f"stage1-17:{self.epoch}:{self.split}:{index}".encode()
        seed = int.from_bytes(hashlib.sha256(key).digest()[:4], "big")
        old_np, old_random = np.random.get_state(), random.getstate()
        try:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(seed)
                np.random.seed(seed)
                random.seed(seed)
                sample = self.processor.preprocess(raw)
        finally:
            np.random.set_state(old_np)
            random.setstate(old_random)
        if tuple(torch.nonzero(sample["action_dim_is_pad"]).flatten().tolist()) != (7, 8, 17, 18):
            raise ValueError("R1Pro base/trunk or padding mapping changed")
        if sample["action"].shape != (32, 27) or not torch.isfinite(sample["action"]).all():
            raise ValueError("Invalid normalized continuous target")
        sample["source_identity"] = identity
        return sample


def worker_init(_):
    torch.set_num_threads(1)


def collate_stage1(rows):
    cameras = tuple(rows[0]["pixel_values"])
    if any(tuple(r["pixel_values"]) != cameras for r in rows):
        raise ValueError("Camera order varies within batch")
    return dict(samples=[r["samples"] for r in rows],
        pixel_values={k: torch.stack([r["pixel_values"][k] for r in rows]) for k in cameras},
        source_identity=[r["source_identity"] for r in rows],
        **{k: torch.stack([r[k] for r in rows]) for k in ("action", "action_is_pad", "action_dim_is_pad")})


def to_device(batch, device):
    return dict(samples=batch["samples"],
        pixel_values={k: v.to(device, non_blocking=True) for k, v in batch["pixel_values"].items()},
        **{k: batch[k].to(device, non_blocking=True) for k in ("action", "action_is_pad", "action_dim_is_pad")})
