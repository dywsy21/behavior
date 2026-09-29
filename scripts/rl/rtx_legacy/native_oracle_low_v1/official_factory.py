"""Concrete evaluator-side factory for frozen native-FM oracle-low windows.

OmniGibson is imported only by :func:`_lazy_official_imports`, called from the
actual driver execution path.  CPU tests inject the small import bundle below
and never start Kit, CUDA, a policy, or an environment.
"""
from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping
from dataclasses import dataclass
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np

from .runtime import (
    FrozenOracleWindow,
    OfficialEvaluatorHarness,
    OracleLowV1Error,
    bootstrap_g05_source,
    install_frozen_semantic_subgoal,
    make_physical_oracle_evaluator,
    run_oracle_window_from_official_evaluator,
    sha256_file,
)


DEFAULT_OG_ROOT = Path("/mnt/sdc1/xhz/BEHAVIOR2026/BEHAVIOR-1K/OmniGibson")
DEFAULT_EVAL_ROOT = Path("/mnt/sdc1/robodojo/behavior_eval")
DEFAULT_TASKS_PATH = Path("/mnt/sdc1/xhz/BEHAVIOR2026/2026-challenge-demos/meta/tasks.jsonl")
_ALLOWED_WINDOW_FIELDS = {
    "kind", "immutable", "window_id", "task_name", "official_mode", "instance_id", "seed",
    "max_steps", "robot_config_path", "robot_config_sha256", "prefix_actions_path",
    "prefix_actions_sha256", "max_chunks", "execute_steps", "semantic_subgoal",
}
_SEMANTIC_SUBGOAL_FIELDS = {
    "parent_goal", "active_skills_semantic_json", "active_skills_text",
}


@dataclass(frozen=True)
class OfficialOracleLowWindow:
    window_id: str
    task_name: str
    official_mode: str
    instance_id: int
    seed: int
    max_steps: int | None
    robot_config_path: str
    robot_config_sha256: str
    prefix_actions_path: str
    prefix_actions_sha256: str
    max_chunks: int
    execute_steps: int
    semantic_subgoal: dict[str, str]

    def frozen_window(self) -> FrozenOracleWindow:
        actions_path = Path(self.prefix_actions_path).resolve(strict=True)
        if sha256_file(actions_path) != self.prefix_actions_sha256:
            raise OracleLowV1Error("frozen prefix actions differ from the window manifest")
        loaded = np.load(actions_path, allow_pickle=False)
        array = np.asarray(loaded, dtype=np.float32)
        if array.ndim != 2 or array.shape[1] != 23 or not np.isfinite(array).all():
            raise OracleLowV1Error("frozen prefix must be a finite [N,23] numpy array")
        return FrozenOracleWindow(
            self.window_id, tuple(row.copy() for row in array), self.max_chunks, self.execute_steps,
        )


def _sha(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise OracleLowV1Error(f"{field} must be SHA-256")
    try:
        int(value, 16)
    except ValueError as exc:
        raise OracleLowV1Error(f"{field} must be SHA-256") from exc
    return value


def load_official_oracle_window(path: str | Path) -> OfficialOracleLowWindow:
    """Load one immutable evaluator window without consuming source/audit rows."""
    source = Path(path).resolve(strict=True)
    try:
        raw = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise OracleLowV1Error(f"cannot read frozen oracle window {source}") from exc
    if not isinstance(raw, Mapping) or set(raw) != _ALLOWED_WINDOW_FIELDS:
        raise OracleLowV1Error("oracle window fields are not the exact frozen schema")
    if raw.get("kind") != "native_oracle_low_window" or raw.get("immutable") is not True:
        raise OracleLowV1Error("oracle window must be immutable native_oracle_low_window")
    task = raw.get("task_name")
    mode = raw.get("official_mode")
    if not isinstance(task, str) or not task or mode not in {"train", "public_test", "hidden_test"}:
        raise OracleLowV1Error("oracle window needs official task_name and official_mode")
    for key in ("instance_id", "seed", "max_chunks", "execute_steps"):
        if isinstance(raw.get(key), bool) or not isinstance(raw.get(key), int):
            raise OracleLowV1Error(f"oracle window {key} must be an integer")
    if raw["instance_id"] < 0 or raw["max_chunks"] < 1 or not 1 <= raw["execute_steps"] <= 16:
        raise OracleLowV1Error("oracle window instance/budget is invalid")
    max_steps = raw.get("max_steps")
    if max_steps is not None and (isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps < 1):
        raise OracleLowV1Error("oracle window max_steps must be positive or null")
    semantic = raw.get("semantic_subgoal")
    if not isinstance(semantic, Mapping):
        raise OracleLowV1Error("oracle window semantic_subgoal must be a mapping")
    # A frozen L1 window transports exactly the deployment semantic surface.
    # Object state / binding proof / audit must never be admitted merely to be
    # rejected later by the policy process.
    if set(semantic) != _SEMANTIC_SUBGOAL_FIELDS or not all(
        isinstance(key, str) and isinstance(value, str) and value
        for key, value in semantic.items()
    ):
        raise OracleLowV1Error("oracle window semantic subgoal is not the exact deployment schema")
    robot = Path(str(raw.get("robot_config_path", ""))).resolve(strict=True)
    prefix = Path(str(raw.get("prefix_actions_path", ""))).resolve(strict=True)
    if sha256_file(robot) != _sha(raw.get("robot_config_sha256"), field="robot_config_sha256"):
        raise OracleLowV1Error("robot config byte identity differs from frozen window")
    if sha256_file(prefix) != _sha(raw.get("prefix_actions_sha256"), field="prefix_actions_sha256"):
        raise OracleLowV1Error("prefix action byte identity differs from frozen window")
    return OfficialOracleLowWindow(
        window_id=str(raw["window_id"]), task_name=task, official_mode=mode,
        instance_id=int(raw["instance_id"]), seed=int(raw["seed"]), max_steps=max_steps,
        robot_config_path=str(robot), robot_config_sha256=str(raw["robot_config_sha256"]),
        prefix_actions_path=str(prefix), prefix_actions_sha256=str(raw["prefix_actions_sha256"]),
        max_chunks=int(raw["max_chunks"]), execute_steps=int(raw["execute_steps"]),
        semantic_subgoal=dict(semantic),
    )


def load_task_instructions(path: str | Path) -> dict[int, str]:
    rows: dict[int, str] = {}
    with Path(path).resolve(strict=True).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                item = json.loads(line)
                rows[int(item["task_index"])] = str(item["task"])
    if len(rows) != 100:
        raise OracleLowV1Error("official tasks file must contain exactly 100 task instructions")
    return rows


def _as_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _find_value_by_suffix(payload: Mapping[str, Any], suffix: str) -> Any:
    matches = [value for key, value in payload.items() if str(key).endswith(suffix)]
    if len(matches) != 1:
        raise OracleLowV1Error(f"official observation has no unique {suffix} stream")
    return matches[0]


def _chw_uint8(value: Any, *, camera: str) -> np.ndarray:
    image = _as_numpy(value)
    if image.ndim != 3:
        raise OracleLowV1Error(f"{camera} must be one RGB image")
    if image.shape[-1] in (3, 4):
        image = image[..., :3].transpose(2, 0, 1)
    elif image.shape[0] in (3, 4):
        image = image[:3]
    else:
        raise OracleLowV1Error(f"{camera} is not RGB(A)")
    if np.issubdtype(image.dtype, np.floating) and image.size and float(np.nanmax(image)) <= 1.0:
        image = image * 255.0
    return np.ascontiguousarray(np.clip(image, 0, 255).astype(np.uint8))


def behavior_obs_to_native_low(evaluator_obs: Mapping[str, Any], task_instructions: Mapping[int, str]) -> dict[str, Any]:
    """The reviewed RGB/proprio-only projection; no state predicates enter it."""
    proprio = _as_numpy(_find_value_by_suffix(evaluator_obs, "::proprio")).astype(np.float32).reshape(-1)
    if proprio.shape != (61,):
        raise OracleLowV1Error("official R1Pro proprio must be 61D")
    task_values = _as_numpy(evaluator_obs.get("task_id")).reshape(-1)
    if task_values.shape != (1,):
        raise OracleLowV1Error("official observation must carry exactly one task_id")
    task_id = int(task_values[0])
    if task_id not in task_instructions:
        raise OracleLowV1Error("official task_id has no frozen task instruction")
    camera_suffixes = {
        "head_rgb": "robot_r1:zed_link:Camera:0::rgb",
        "left_wrist_rgb": "robot_r1:left_realsense_link:Camera:0::rgb",
        "right_wrist_rgb": "robot_r1:right_realsense_link:Camera:0::rgb",
    }
    slices = {
        "base_qvel": slice(0, 3), "left_arm": slice(3, 10), "left_gripper": slice(24, 26),
        "right_arm": slice(28, 35), "right_gripper": slice(49, 51), "trunk_qpos": slice(53, 57),
    }
    return {
        "images": {key: _chw_uint8(_find_value_by_suffix(evaluator_obs, suffix), camera=key)
                   for key, suffix in camera_suffixes.items()},
        "state": {key: np.ascontiguousarray(proprio[part], dtype=np.float32) for key, part in slices.items()},
        "task": task_instructions[task_id], "embodiment_type": "galaxea_r1pro", "frequency": 30.0,
    }


@dataclass(frozen=True)
class _OfficialImports:
    OmegaConf: Any
    Evaluator: Any
    seed_everything: Callable[[int], Any]
    gm: Any


def _require_import_under(module: Any, root: Path, *, label: str) -> None:
    module_file = getattr(module, "__file__", None)
    if not isinstance(module_file, str) or not module_file:
        raise OracleLowV1Error(f"{label} has no verifiable module origin")
    try:
        Path(module_file).resolve(strict=True).relative_to(root)
    except (OSError, ValueError) as exc:
        raise OracleLowV1Error(f"{label} was not imported from the pinned official root") from exc


def _lazy_official_imports(*, og_root: str | Path, eval_root: str | Path, gpu: int) -> _OfficialImports:
    """Apply the reviewed chunk patch before importing official Evaluator."""
    og = Path(og_root).resolve(strict=True)
    evaluator_root = Path(eval_root).resolve(strict=True)
    if isinstance(gpu, bool) or not isinstance(gpu, int) or gpu < 0:
        raise OracleLowV1Error("official evaluator needs one non-negative physical GPU id")
    # These are the same process-local OmniGibson launch knobs used by the
    # reviewed evaluator launchers.  They are deliberately set only in the
    # real driver path, never while importing this module or running CPU tests.
    os.environ["OMNIGIBSON_GPU_ID"] = str(gpu)
    os.environ.setdefault("OMNIGIBSON_HEADLESS", "1")
    os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
    os.environ.setdefault("OMNIGIBSON_NO_OMNI_LOGS", "1")
    for entry in (str(og), str(evaluator_root)):
        if entry not in sys.path:
            sys.path.insert(0, entry)
    # Exact module import executes the public runner's wrapper patch.  It does
    # not launch its argparse main.
    chunk_runner = importlib.import_module("run_behavior_eval_chunked")
    _require_import_under(chunk_runner, evaluator_root, label="chunk-aware official evaluator runner")
    from omegaconf import OmegaConf
    from omnigibson.eval.evaluator import Evaluator
    from omnigibson.eval.utils.eval_utils import seed_everything
    from omnigibson.macros import gm
    _require_import_under(sys.modules[Evaluator.__module__], og, label="official Evaluator")
    return _OfficialImports(OmegaConf=OmegaConf, Evaluator=Evaluator, seed_everything=seed_everything, gm=gm)


class OfficialEvaluatorSession:
    """Context manager following official init → reset → load-instance order."""

    def __init__(
        self,
        window: OfficialOracleLowWindow,
        *,
        og_root: str | Path = DEFAULT_OG_ROOT,
        eval_root: str | Path = DEFAULT_EVAL_ROOT,
        gpu: int | None = None,
        imports: _OfficialImports | None = None,
        tasks_path: str | Path = DEFAULT_TASKS_PATH,
    ) -> None:
        self.window = window
        self.og_root = Path(og_root)
        self.eval_root = Path(eval_root)
        self.gpu = gpu
        self._imports = imports
        self.tasks = load_task_instructions(tasks_path)
        self.evaluator: Any | None = None
        self.harness: OfficialEvaluatorHarness | None = None

    def __enter__(self) -> OfficialEvaluatorHarness:
        if self._imports is None:
            if self.gpu is None:
                raise OracleLowV1Error("official evaluator factory requires an explicit physical --gpu")
            imports = _lazy_official_imports(og_root=self.og_root, eval_root=self.eval_root, gpu=self.gpu)
        else:
            imports = self._imports
        imports.gm.HEADLESS = True
        seed = imports.seed_everything(self.window.seed)
        if int(seed) != self.window.seed:
            raise OracleLowV1Error("official evaluator seed differs from frozen window")
        robot = imports.OmegaConf.load(self.window.robot_config_path)
        cfg = imports.OmegaConf.create({
            "env_wrapper": {"_target_": "omnigibson.eval.wrappers.RGBDFullResWrapper"},
            "policy_name": "native_oracle_low_local_harness",
            "model": {"_target_": "omnigibson.eval.policies.LocalPolicy", "action_dim": None},
            "headless": True, "partial_scene_load": True, "max_steps": self.window.max_steps,
            "write_video": False, "mode": self.window.official_mode, "seed": self.window.seed,
            "task": {"name": self.window.task_name}, "robot": robot,
        })
        evaluator = imports.Evaluator(cfg)
        enter = getattr(evaluator, "__enter__", None)
        if not callable(enter):
            raise OracleLowV1Error("official Evaluator lacks context manager entry")
        enter()
        try:
            # This exact order is used by the official evaluator main: reset
            # first initializes handles, then load the frozen instance, then
            # the L1 run helper resets again before its true prefix.
            evaluator.reset()
            evaluator.load_task_instance(self.window.instance_id)
        except BaseException:
            exit_fn = getattr(evaluator, "__exit__", None)
            if callable(exit_fn):
                exit_fn(*sys.exc_info())
            raise
        self.evaluator = evaluator
        self.harness = OfficialEvaluatorHarness(
            evaluator,
            behavior_obs_to_g05=lambda obs: behavior_obs_to_native_low(obs, self.tasks),
        )
        return self.harness

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        if self.evaluator is not None:
            exit_fn = getattr(self.evaluator, "__exit__", None)
            if callable(exit_fn):
                exit_fn(exc_type, exc_value, traceback)


class LowOnlyWebSocketClient:
    """Evaluator-side client for exactly one stateless native-low chunk."""

    def __init__(self, uri: str, *, source_root: str | Path) -> None:
        self.uri = str(uri)
        bootstrap_g05_source(source_root)

    @staticmethod
    def _wire_subgoal(installed: Any) -> dict[str, str]:
        values = {
            "parent_goal": getattr(installed, "parent_goal", None),
            "active_skills_semantic_json": getattr(installed, "active_skills_semantic_json", None),
            "active_skills_text": getattr(installed, "active_skills_text", None),
        }
        if not all(isinstance(value, str) and value for value in values.values()):
            raise OracleLowV1Error("only an InstalledLocalSubgoal may cross the low-only wire")
        return values

    async def _request_async(self, observation: Mapping[str, Any], installed: Any, execute_steps: int) -> Mapping[str, Any]:
        import websockets
        wire = importlib.import_module("g05.utils.websocket")
        packb, unpackb = getattr(wire, "packb", None), getattr(wire, "unpackb", None)
        if not callable(packb) or not callable(unpackb):
            raise OracleLowV1Error("atomic source lacks G05 websocket codec")
        request = {
            "kind": "native_low_chunk", "mode": "oracle_low", "observation": dict(observation),
            "installed_subgoal": self._wire_subgoal(installed), "execute_steps": execute_steps,
        }
        async with websockets.connect(self.uri, max_size=None) as websocket:
            _hello = unpackb(await websocket.recv())
            await websocket.send(packb(request))
            reply = unpackb(await websocket.recv())
        if not isinstance(reply, Mapping) or reply.get("ok") is not True or not isinstance(reply.get("response"), Mapping):
            raise OracleLowV1Error(f"low-only service rejected request: {reply}")
        return reply["response"]

    def request(self, observation: Mapping[str, Any], installed: Any, execute_steps: int) -> Mapping[str, Any]:
        return asyncio.run(self._request_async(observation, installed, execute_steps))


def run_official_oracle_low_window(
    *,
    window: OfficialOracleLowWindow,
    source_root: str | Path,
    policy_uri: str,
    og_root: str | Path = DEFAULT_OG_ROOT,
    eval_root: str | Path = DEFAULT_EVAL_ROOT,
    gpu: int | None = None,
    tasks_path: str | Path = DEFAULT_TASKS_PATH,
    imports: _OfficialImports | None = None,
) -> dict[str, Any]:
    """Concrete two-process official L1 run; never loads high/B in OG."""
    bootstrap_g05_source(source_root)
    installed = install_frozen_semantic_subgoal(window.semantic_subgoal)
    client = LowOnlyWebSocketClient(policy_uri, source_root=source_root)
    with OfficialEvaluatorSession(
        window, og_root=og_root, eval_root=eval_root, gpu=gpu, imports=imports, tasks_path=tasks_path,
    ) as harness:
        result = run_oracle_window_from_official_evaluator(
            harness, window=window.frozen_window(), installed_subgoal=installed,
            request_low_chunk=client.request,
            evaluate_subgoal=make_physical_oracle_evaluator(harness.evaluator.env),
        )
    return {
        **result,
        "official_window": {
            "task_name": window.task_name, "official_mode": window.official_mode,
            "instance_id": window.instance_id, "seed": window.seed,
            "robot_config_path": window.robot_config_path,
            "robot_config_sha256": window.robot_config_sha256,
        },
    }
