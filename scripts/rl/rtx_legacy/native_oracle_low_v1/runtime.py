"""Production-shaped, low-only native-FM oracle-subgoal adapter.

The module intentionally has no import-time G05 / torch / OmniGibson import.
The two process owners import it separately: the G05 process owns
``NativeLowOnlyService`` and the OmniGibson process owns ``OracleLowDriver``.
Only a frozen semantic subgoal crosses to the policy; privileged resolver and
physical-oracle evidence stay evaluator-side.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import importlib
import inspect
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np


class OracleLowV1Error(RuntimeError):
    """Raised before a low-only probe can become a misleading evaluation."""


OFFICIAL_ACTION_DIM = 23


def sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).resolve(strict=True).read_bytes()).hexdigest()


def _required_sha(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise OracleLowV1Error(f"{field} must be a SHA-256 string")
    try:
        int(value, 16)
    except ValueError as exc:
        raise OracleLowV1Error(f"{field} must be a SHA-256 string") from exc
    return value


def _verify_file(path: str | Path, expected_sha: str, *, field: str) -> Path:
    resolved = Path(path).resolve(strict=True)
    actual = sha256_file(resolved)
    if actual != _required_sha(expected_sha, field=field):
        raise OracleLowV1Error(f"{field} byte identity mismatch: {resolved}")
    return resolved


def bootstrap_g05_source(source_root: str | Path) -> Path:
    """Put one future atomic E7 source tree ahead of cwd/PYTHONPATH.

    The evaluator adapter does not borrow modules from the main checkout or a
    separately reviewed model tree.  A release integration has to provide one
    source root containing both the native runtime and public FM policy.
    """
    root = Path(source_root).resolve(strict=True)
    src = root / "src"
    if not src.is_dir():
        raise OracleLowV1Error(f"native source root lacks src/: {root}")
    # Reordering sys.path cannot repair a mixed process after a different G05
    # checkout was imported.  Refuse it before any loader/model construction.
    for name, module in tuple(sys.modules.items()):
        if name != "g05" and not name.startswith("g05."):
            continue
        module_file = getattr(module, "__file__", None)
        if not isinstance(module_file, str) or not module_file:
            continue
        try:
            Path(module_file).resolve(strict=True).relative_to(src)
        except (OSError, ValueError) as exc:
            raise OracleLowV1Error(
                f"foreign preloaded {name} prevents an atomic native source bootstrap: {module_file}"
            ) from exc
    for entry in (str(src), str(root)):
        while entry in sys.path:
            sys.path.remove(entry)
    sys.path[:0] = [str(src), str(root)]
    return root


def _module_identity(module: Any, *, source_root: Path, label: str) -> dict[str, str]:
    module_file = getattr(module, "__file__", None)
    if not isinstance(module_file, str) or not module_file:
        raise OracleLowV1Error(f"{label} lacks a module file")
    path = Path(module_file).resolve(strict=True)
    try:
        path.relative_to(source_root / "src")
    except ValueError as exc:
        raise OracleLowV1Error(f"{label} imported outside the atomic source root: {path}") from exc
    return {"file": str(path), "sha256": sha256_file(path)}


def _normalized_low(normalized: Mapping[str, Any]) -> Mapping[str, Any]:
    if normalized.get("mode") != "oracle_low":
        raise OracleLowV1Error("native oracle adapter accepts only mode=oracle_low")
    if normalized.get("high") is not None or normalized.get("outcome_runtime") is not None:
        raise OracleLowV1Error("oracle_low must not declare high/outcome runtime")
    constraints = normalized.get("runtime_constraints")
    expected = {"construct_components": ["low"], "load_components": ["low"], "call_components": ["low"]}
    if not isinstance(constraints, Mapping) or any(constraints.get(k) != v for k, v in expected.items()):
        raise OracleLowV1Error("oracle_low normalized runtime constraints are not low-only")
    low = normalized.get("low")
    if not isinstance(low, Mapping) or low.get("component") != "low":
        raise OracleLowV1Error("oracle_low normalized manifest lacks one low component")
    return low


def _load_snapshot(path: str | Path, *, expected_bridge_sha: str | None = None) -> Any:
    """Read the pre-existing NativeFMSnapshot transport, without new claims."""
    payload = json.loads(Path(path).resolve(strict=True).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise OracleLowV1Error("native snapshot JSON must be an object")
    bootstrap_g05_source(payload.get("source_root", ""))
    module = importlib.import_module("g05.models.g05.memlite_native_fm_inferencer")
    snapshot_cls = getattr(module, "NativeFMSnapshot", None)
    if snapshot_cls is None:
        raise OracleLowV1Error("atomic source lacks NativeFMSnapshot")
    required = (
        "source_root", "protocol_sha256", "processor_sha256", "builder_sha256",
        "policy_sha256", "normalizer_receipt_sha256", "bridge_sha256",
    )
    missing = [key for key in required if key not in payload]
    if missing:
        raise OracleLowV1Error(f"native snapshot misses {missing}")
    if expected_bridge_sha is not None and payload["bridge_sha256"] != expected_bridge_sha:
        raise OracleLowV1Error("native snapshot bridge hash differs from pinned bridge module")
    snapshot = snapshot_cls(**{key: payload[key] for key in required})
    snapshot.require_complete()
    return snapshot


def _resolve_stats_path(cfg: Any) -> Path:
    """Resolve only an explicitly configured/run-local normalizer asset."""
    candidates: list[Any] = []
    for key in ("datastatics_path", "dataset_stats_path"):
        try:
            candidates.append(cfg.get(key))
        except AttributeError:
            pass
    try:
        run_dir = cfg.get("run_dir")
    except AttributeError:
        run_dir = None
    if run_dir:
        candidates.append(Path(str(run_dir)) / "dataset_stats.json")
    try:
        ckpt_path = cfg.get("ckpt_path")
    except AttributeError:
        ckpt_path = None
    if ckpt_path:
        checkpoint = Path(str(ckpt_path))
        candidates.extend((checkpoint.parent / "dataset_stats.json", checkpoint.parent.parent / "dataset_stats.json"))
    for value in candidates:
        if value and Path(str(value)).is_file():
            return Path(str(value)).resolve(strict=True)
    raise OracleLowV1Error("low serving config does not resolve an approved dataset stats asset")


def _verify_actual_load_receipt(policy: Any, low: Mapping[str, Any]) -> dict[str, Any]:
    receipt = getattr(policy, "_coordination_base_load_receipt", None)
    if not isinstance(receipt, Mapping):
        raise OracleLowV1Error("real low checkpoint loader omitted _coordination_base_load_receipt")
    coverage = low.get("base_load_coverage")
    if not isinstance(coverage, Mapping) or coverage.get("complete") is not True:
        raise OracleLowV1Error("manifest lacks reviewed low base-load coverage")
    actual_missing = sorted(str(key) for key in receipt.get("truly_missing", []))
    actual_unexpected = sorted(str(key) for key in receipt.get("unexpected_keys", []))
    partial = sorted(str(key) for key in receipt.get("partial_loaded_keys", []))
    mismatched = receipt.get("mismatched_keys", [])
    if partial or mismatched:
        raise OracleLowV1Error("low checkpoint has partial or mismatched model parameters")
    allowed_missing = sorted(str(key) for key in coverage.get("allowed_missing_keys", []))
    allowed_unexpected = sorted(str(key) for key in coverage.get("allowed_unexpected_keys", []))
    if actual_missing != allowed_missing or actual_unexpected != allowed_unexpected:
        raise OracleLowV1Error(
            "actual low checkpoint coverage differs from immutable manifest allowlists"
        )
    return {
        "loaded_count": receipt.get("loaded_count"),
        "loaded_prefix_counts": dict(receipt.get("loaded_prefix_counts", {})),
        "missing_keys": actual_missing,
        "unexpected_keys": actual_unexpected,
        "partial_loaded_keys": partial,
        "mismatched_keys": list(mismatched),
    }


def load_native_low_component(
    normalized_manifest: Mapping[str, Any], *, source_root: str | Path, device: str = "cuda",
) -> Mapping[str, Any]:
    """Load exactly the evaluated A low checkpoint and its real processor.

    This is deliberately a loader, not a fallback to the old separated
    high+low v11 server.  It is not invoked by the CPU unit tests unless a
    real checkpoint/config has been published.
    """
    low = _normalized_low(normalized_manifest)
    root = bootstrap_g05_source(source_root)
    for field in ("checkpoint", "config", "parent_initialization"):
        value = low.get(field)
        if not isinstance(value, Mapping):
            raise OracleLowV1Error(f"normalized low component lacks {field}")
        _verify_file(value.get("path", ""), value.get("sha256", ""), field=f"low.{field}")

    from omegaconf import OmegaConf
    import torch
    from g05.data_processor.processor.mixture_processor import MixtureProcessor
    from g05.utils.checkpoint.checkpoint_utils import load_model_from_checkpoint
    from g05.utils.data.normalizer import load_dataset_stats_from_json
    from g05.utils.data.processor_utils import build_processors
    from g05.utils.eval.eval_utils import filter_embodiment

    config_path = Path(str(low["config"]["path"])).resolve(strict=True)
    cfg = OmegaConf.load(config_path)
    if not hasattr(cfg, "model") or not hasattr(cfg.model, "model_arch"):
        raise OracleLowV1Error("low config must be a resolved serving/training config with model.model_arch")
    OmegaConf.set_struct(cfg, False)
    cfg.ckpt_path = str(Path(str(low["checkpoint"]["path"])).resolve(strict=True))
    cfg.model.use_torch_compile = False
    OmegaConf.set_struct(cfg, True)
    filter_embodiment(cfg, "galaxea_r1pro")
    policy = load_model_from_checkpoint(
        cfg.model.model_arch,
        cfg.ckpt_path,
        device=device,
        extra_prefixes=["normalizer."],
        eval_mode=False,
    )
    if cfg.model.get("model_weights_to_bf16", True):
        policy = policy.to(torch.bfloat16)
    policy.apply_fp32_params()
    policy = policy.eval()
    if hasattr(policy, "action_tokenizer"):
        policy.action_tokenizer.to(device)
    if not callable(getattr(policy, "generate_low_level_action", None)):
        raise OracleLowV1Error("loaded low policy lacks public generate_low_level_action")

    stats_path = _resolve_stats_path(cfg)
    processor = build_processors(cfg)
    processor.set_normalizer_from_stats(load_dataset_stats_from_json(stats_path))
    processor.eval()
    processors = list(processor.processors.values()) if isinstance(processor, MixtureProcessor) else [processor]
    if len(processors) != 1:
        raise OracleLowV1Error("oracle_low requires exactly one selected R1Pro processor")
    sub_processor = processors[0]
    builder = getattr(sub_processor, "samples_builder", None)
    if not callable(getattr(sub_processor, "preprocess_for_inference", None)) or builder is None:
        raise OracleLowV1Error("selected low processor lacks public target-free inference builder")
    assets = low.get("runtime_assets")
    if not isinstance(assets, Mapping):
        raise OracleLowV1Error("normalized low component lacks runtime asset identities")
    processor_module = inspect.getmodule(type(sub_processor))
    builder_module = inspect.getmodule(type(builder))
    processor_identity = _module_identity(processor_module, source_root=root, label="low processor")
    builder_identity = _module_identity(builder_module, source_root=root, label="low builder")
    if processor_identity["sha256"] != assets.get("processor_sha256"):
        raise OracleLowV1Error("loaded processor differs from manifest runtime_assets.processor_sha256")
    if builder_identity["sha256"] != assets.get("builder_sha256"):
        raise OracleLowV1Error("loaded builder differs from manifest runtime_assets.builder_sha256")
    coverage = _verify_actual_load_receipt(policy, low)
    policy_module = inspect.getmodule(type(policy))
    return {
        "policy": policy,
        "processor": processor,
        "low_component_identity": {
            "checkpoint": dict(low["checkpoint"]),
            "config": dict(low["config"]),
            "parent_initialization": dict(low["parent_initialization"]),
            "runtime_assets": {"processor": processor_identity, "builder": builder_identity},
            "policy": _module_identity(policy_module, source_root=root, label="low policy"),
            "stats": {"path": str(stats_path), "sha256": sha256_file(stats_path)},
            "actual_load_coverage": coverage,
        },
    }


def build_production_low_only_service(
    manifest: Mapping[str, Any],
    *,
    source_root: str | Path,
    native_snapshot_path: str | Path,
    bridge_action_to_vector: Callable[[Mapping[str, Any]], Any],
    official23_support: Any,
    device: str = "cuda",
) -> "NativeLowOnlyService":
    """Construct the real low-only G05 service after atomic E7 integration.

    This factory is intentionally unavailable to a mixed candidate tree: it
    first forces all G05 imports under ``source_root``, then uses the
    training-owned public validator and the model-owned public FM API.  It
    never imports the separated v11 loader or any high policy.
    """
    root = bootstrap_g05_source(source_root)
    runtime_module = importlib.import_module("g05.utils.training.coordination_runtime")
    validate_manifest = getattr(runtime_module, "validate_native_fm_eval_manifest", None)
    validate_observation = getattr(runtime_module, "validate_native_fm_eval_runtime_observation", None)
    if not callable(validate_manifest) or not callable(validate_observation):
        raise OracleLowV1Error("atomic source lacks the published native-eval validator")
    inferencer_module = importlib.import_module("g05.models.g05.memlite_native_fm_inferencer")
    inferencer_factory = getattr(inferencer_module, "NativeFMPolicyInferencer", None)
    chunk_factory = getattr(inferencer_module, "NativeFMActionChunk", None)
    if not callable(inferencer_factory) or not callable(chunk_factory):
        raise OracleLowV1Error("atomic source lacks public native-FM low runtime classes")
    bridge_file = Path(inspect.getfile(bridge_action_to_vector)).resolve(strict=True)
    bridge_sha = sha256_file(bridge_file)
    snapshot = _load_snapshot(native_snapshot_path, expected_bridge_sha=bridge_sha)
    if Path(snapshot.source_root).resolve(strict=True) != root:
        raise OracleLowV1Error("native snapshot source_root does not equal the requested atomic source root")
    return NativeLowOnlyService(
        manifest,
        validate_manifest=validate_manifest,
        validate_runtime_observation=validate_observation,
        load_low=lambda normalized: load_native_low_component(normalized, source_root=root, device=device),
        inferencer_factory=inferencer_factory,
        chunk_factory=chunk_factory,
        snapshot=snapshot,
        bridge_action_to_vector=bridge_action_to_vector,
        bridge_identity={"file": str(bridge_file), "sha256": bridge_sha},
        official23_support=official23_support,
        device=device,
        allow_test_seam=False,
    )


def grouped_raw_to_official23(action: Mapping[str, Any]) -> np.ndarray:
    """Translate only postprocessed grouped raw actions into R1Pro wire-23D.

    This function deliberately has no normalized-27D input branch.  The
    normalizer/merger inverse belongs to ``NativeFMPolicyInferencer``.
    """
    if not isinstance(action, Mapping):
        raise OracleLowV1Error("official bridge needs a grouped raw action mapping")
    def part(key: str, width: int) -> np.ndarray:
        value = np.asarray(action[key], dtype=np.float32).reshape(-1)
        if value.shape != (width,) or not np.isfinite(value).all():
            raise OracleLowV1Error(f"postprocessed action {key} is not finite {width}D")
        return value
    if "base_qvel" in action and "trunk_qpos" in action:
        base, trunk = part("base_qvel", 3), part("trunk_qpos", 4)
    elif "lower_body" in action:
        lower = part("lower_body", 7)
        trunk, base = lower[:4], lower[4:]
    else:
        raise OracleLowV1Error("postprocessed action lacks base_qvel/trunk_qpos")
    vector = np.concatenate((base, trunk, part("left_arm", 7), part("left_gripper", 1),
                             part("right_arm", 7), part("right_gripper", 1)))
    if vector.shape != (OFFICIAL_ACTION_DIM,) or not np.isfinite(vector).all():
        raise OracleLowV1Error("official bridge did not produce finite wire-23D")
    return vector.astype(np.float32, copy=False)


class NativeLowOnlyService:
    """G05-process service that is structurally unable to load a high policy."""

    def __init__(
        self,
        manifest: Mapping[str, Any],
        *,
        validate_manifest: Callable[[Mapping[str, Any]], Mapping[str, Any]],
        validate_runtime_observation: Callable[[Mapping[str, Any], Mapping[str, Any]], Mapping[str, Any]],
        load_low: Callable[[Mapping[str, Any]], Mapping[str, Any]],
        inferencer_factory: Callable[..., Any],
        chunk_factory: Callable[..., Any],
        snapshot: Any,
        bridge_action_to_vector: Callable[[Mapping[str, Any]], Any],
        bridge_identity: Mapping[str, str],
        official23_support: Any,
        device: str = "cuda",
        allow_test_seam: bool = False,
    ) -> None:
        normalized = validate_manifest(manifest)
        low = _normalized_low(normalized)
        loaded = load_low(normalized)
        if not isinstance(loaded, Mapping) or "policy" not in loaded or "processor" not in loaded:
            raise OracleLowV1Error("low-only loader must return policy and processor")
        observation = validate_runtime_observation(normalized, {
            "constructed_components": ["low"],
            "loaded_components": ["low"],
            "called_components": ["low"],
        })
        if any(bool(observation.get(key)) for key in ("high_construction_attempted", "high_loaded", "high_called")):
            raise OracleLowV1Error("oracle_low runtime observation reports high activity")
        self.normalized_manifest = dict(normalized)
        self.low = dict(low)
        self.low_component_identity = dict(loaded.get("low_component_identity", {}))
        if not allow_test_seam:
            stats = self.low_component_identity.get("stats")
            actual_stats_sha = stats.get("sha256") if isinstance(stats, Mapping) else None
            support_stats_sha = getattr(official23_support, "normalizer_stats_sha256", None)
            if not isinstance(actual_stats_sha, str) or actual_stats_sha != support_stats_sha:
                raise OracleLowV1Error(
                    "loaded low normalizer stats do not equal the reviewed official23 support asset"
                )
        self.low_only_receipt = dict(observation)
        self.inferencer = inferencer_factory(
            loaded["policy"], loaded["processor"], device=device, snapshot=snapshot,
            allow_test_seam=allow_test_seam,
        )
        self._chunk_factory = chunk_factory
        self._snapshot = snapshot
        self._bridge = bridge_action_to_vector
        self._bridge_identity = dict(bridge_identity)
        self._official23_support = official23_support
        self._allow_test_seam = bool(allow_test_seam)

    def infer_chunk(
        self,
        observation: Mapping[str, Any],
        installed_subgoal: Any,
        *,
        execute_steps: int,
    ) -> dict[str, Any]:
        if not isinstance(execute_steps, int) or not 1 <= execute_steps <= 16:
            raise OracleLowV1Error("oracle_low may execute 1..16 fresh controls per chunk")
        # A WebSocket message can carry only a mapping, never the source-side
        # InstalledLocalSubgoal object.  Admit that mapping here -- at the one
        # service boundary before processor/model code -- rather than trusting
        # a caller to have removed audit, source, or physical-truth fields.
        # Already installed objects are accepted only for the evaluator-local
        # in-process seam and are never serialized over the wire.
        if isinstance(installed_subgoal, Mapping):
            installed_subgoal = install_frozen_semantic_subgoal(installed_subgoal)
        grouped, timing = self.inferencer.infer_native_low_with_timing(
            [observation], [installed_subgoal], num_obs_steps=int(self.low["num_obs_steps"]),
        )
        if not isinstance(grouped, Sequence) or len(grouped) != 1:
            raise OracleLowV1Error("native low inferencer did not return one grouped action horizon")
        chunk = self._chunk_factory(
            grouped[0], execute_steps=execute_steps, bridge_action_to_vector=self._bridge,
            bridge_identity=self._bridge_identity, snapshot=self._snapshot,
            official23_support=self._official23_support, allow_test_seam=self._allow_test_seam,
        )
        actions = [chunk.next_official_action() for _ in range(execute_steps)]
        vectors = [np.asarray(item, dtype=np.float32).reshape(-1) for item in actions]
        if any(vector.shape != (OFFICIAL_ACTION_DIM,) or not np.isfinite(vector).all() for vector in vectors):
            raise OracleLowV1Error("native low bridge returned invalid official 23D control")
        return {
            "actions": vectors,
            "chunk": chunk.receipt(),
            "timing": dict(timing),
            "low_only_receipt": dict(self.low_only_receipt),
            "low_component_identity": dict(self.low_component_identity),
        }


@dataclass(frozen=True)
class FrozenOracleWindow:
    """A fixed local-probe budget; no adaptive extension is allowed."""

    window_id: str
    prefix_actions: tuple[np.ndarray, ...]
    max_chunks: int
    execute_steps: int

    def __post_init__(self) -> None:
        if not isinstance(self.window_id, str) or not self.window_id:
            raise OracleLowV1Error("oracle window needs a nonempty id")
        if not isinstance(self.max_chunks, int) or self.max_chunks < 1:
            raise OracleLowV1Error("oracle window max_chunks must be positive and frozen")
        if not isinstance(self.execute_steps, int) or not 1 <= self.execute_steps <= 16:
            raise OracleLowV1Error("oracle window execute_steps must be in [1,16]")
        for action in self.prefix_actions:
            vector = np.asarray(action, dtype=np.float32).reshape(-1)
            if vector.shape != (OFFICIAL_ACTION_DIM,) or not np.isfinite(vector).all():
                raise OracleLowV1Error("frozen prefix must contain finite official 23D controls")


def _hash_vectors(actions: Sequence[np.ndarray]) -> str:
    return hashlib.sha256(b"".join(np.asarray(item, dtype=np.float32).tobytes() for item in actions)).hexdigest()


def _step_env(env: Any, vector: np.ndarray) -> Mapping[str, Any] | None:
    step = getattr(env, "step", None)
    if not callable(step):
        raise OracleLowV1Error("official oracle-low driver requires env.step")
    result = step(vector)
    if isinstance(result, Mapping):
        return result
    if isinstance(result, tuple) and len(result) == 5 and isinstance(result[4], Mapping):
        return result[4]
    if result is None:
        return None
    raise OracleLowV1Error("official env.step returned neither info mapping nor gymnasium five-tuple")


def _causal_physical_success(measurement: Mapping[str, Any]) -> bool:
    evidence = measurement.get("evidence")
    if not isinstance(evidence, Sequence) or not evidence:
        return False
    def completed(item: Any) -> bool:
        if isinstance(item, Mapping):
            return item.get("causal_completed") is True
        return getattr(item, "causal_completed", False) is True
    return measurement.get("local_subgoal_success") is True and all(completed(item) for item in evidence)


class OracleLowDriver:
    """Evaluator-side fixed-prefix, bounded multi-chunk physical probe."""

    def __init__(
        self,
        *,
        env: Any,
        observation_provider: Callable[[], Mapping[str, Any]],
        request_low_chunk: Callable[[Mapping[str, Any], Any, int], Mapping[str, Any]],
        evaluate_subgoal: Callable[[Any], Mapping[str, Any]],
    ) -> None:
        self.env = env
        self._observation_provider = observation_provider
        self._request_low_chunk = request_low_chunk
        self._evaluate_subgoal = evaluate_subgoal

    def run(self, window: FrozenOracleWindow, installed_subgoal: Any) -> dict[str, Any]:
        initial = self._evaluate_subgoal(installed_subgoal)
        prefix_infos: list[Mapping[str, Any] | None] = []
        for vector in window.prefix_actions:
            prefix_infos.append(_step_env(self.env, np.asarray(vector, dtype=np.float32).reshape(-1)))
        before_policy = self._evaluate_subgoal(installed_subgoal)
        policy_actions: list[np.ndarray] = []
        chunk_receipts: list[dict[str, Any]] = []
        observation_times_ns: list[int] = []
        last_info: Mapping[str, Any] | None = prefix_infos[-1] if prefix_infos else None
        after = before_policy
        for chunk_index in range(window.max_chunks):
            # This call is intentionally inside the loop: no camera/proprio
            # result may be cached across native action chunks.
            observation = self._observation_provider()
            observation_times_ns.append(time.monotonic_ns())
            response = self._request_low_chunk(observation, installed_subgoal, window.execute_steps)
            if not isinstance(response, Mapping):
                raise OracleLowV1Error("low-only service returned no response mapping")
            raw_actions = response.get("actions")
            if not isinstance(raw_actions, Sequence) or not raw_actions or len(raw_actions) > window.execute_steps:
                raise OracleLowV1Error("low-only service returned an invalid bounded action chunk")
            served: list[np.ndarray] = []
            for action in raw_actions:
                vector = np.asarray(action, dtype=np.float32).reshape(-1)
                if vector.shape != (OFFICIAL_ACTION_DIM,) or not np.isfinite(vector).all():
                    raise OracleLowV1Error("low-only service action is not finite official 23D")
                last_info = _step_env(self.env, vector)
                served.append(vector)
            policy_actions.extend(served)
            after = self._evaluate_subgoal(installed_subgoal)
            chunk_receipts.append({
                "chunk_index": chunk_index,
                "live_observation_captured_ns": observation_times_ns[-1],
                "served_actions": len(served),
                "service": {key: response[key] for key in ("chunk", "timing", "low_only_receipt") if key in response},
                "after_local_subgoal_outcome": after.get("local_subgoal_outcome"),
            })
            if _causal_physical_success(after):
                break
        before_success = before_policy.get("local_subgoal_success") is True
        newly_achieved = bool(policy_actions) and not before_success and _causal_physical_success(after)
        official_success: bool | None = None
        if isinstance(last_info, Mapping):
            done = last_info.get("done")
            if isinstance(done, Mapping) and type(done.get("success")) is bool:
                official_success = done["success"]
        return {
            "mode": "oracle_low",
            "window_id": window.window_id,
            "metric_kind": "L1_local_oracle_subgoal_not_full_task_sr",
            "initial_local_subgoal": initial,
            "before_policy_local_subgoal": before_policy,
            "after_policy_local_subgoal": after,
            "already_satisfied_before_policy": before_success,
            "newly_achieved_subgoal": newly_achieved,
            "policy_credit_for_subgoal": newly_achieved,
            "prefix_actions_executed": len(window.prefix_actions),
            "prefix_action_sha256": _hash_vectors(window.prefix_actions),
            "native_policy_actions_executed": len(policy_actions),
            "native_policy_action_sha256": _hash_vectors(policy_actions),
            "chunks_attempted": len(chunk_receipts),
            "max_chunks_frozen": window.max_chunks,
            "chunk_receipts": chunk_receipts,
            "termination_reason": "physical_subgoal_succeeded" if newly_achieved else "frozen_budget_exhausted_or_not_causal",
            # Budget / annotation end / model STOP are not outcome labels.
            "termination_outcome": "UNKNOWN",
            "official_info_done_success": official_success,
            "full_task_success_rate_claim": False,
            "used_snapshot_or_teleport": False,
        }


class OfficialEvaluatorHarness:
    """Small evaluator-side wrapper over the established official step route.

    The constructor receives a real ``omnigibson.eval.Evaluator`` from the
    official evaluator owner; it never imports or creates OmniGibson itself.
    It mirrors the reviewed chunk runner's per-step state refresh so the next
    policy request receives the *post-physics* evaluator observation.
    """

    def __init__(
        self,
        evaluator: Any,
        *,
        behavior_obs_to_g05: Callable[[Mapping[str, Any]], Mapping[str, Any]],
    ) -> None:
        self.evaluator = evaluator
        self._behavior_obs_to_g05 = behavior_obs_to_g05

    def reset(self) -> None:
        reset = getattr(self.evaluator, "reset", None)
        if not callable(reset):
            raise OracleLowV1Error("official evaluator lacks reset")
        reset()

    def observation(self) -> Mapping[str, Any]:
        obs = getattr(self.evaluator, "obs", None)
        if not isinstance(obs, Mapping):
            raise OracleLowV1Error("official evaluator does not expose current observation mapping")
        raw = self._behavior_obs_to_g05(obs)
        if not isinstance(raw, Mapping):
            raise OracleLowV1Error("official observation adapter returned non-mapping raw observation")
        return raw

    def step(self, action: Any) -> Mapping[str, Any] | None:
        vector = np.asarray(action, dtype=np.float32).reshape(-1)
        if vector.shape != (OFFICIAL_ACTION_DIM,) or not np.isfinite(vector).all():
            raise OracleLowV1Error("official evaluator harness accepts only finite wire-23D actions")
        wrapped_env = getattr(self.evaluator, "env", None)
        step = getattr(wrapped_env, "step", None)
        if not callable(step):
            raise OracleLowV1Error("official evaluator env lacks step")
        result = step(vector, n_render_iterations=1)
        if not isinstance(result, tuple) or len(result) != 5:
            raise OracleLowV1Error("official evaluator env.step must return gymnasium five-tuple")
        obs, _reward, _terminated, _truncated, info = result
        sync = getattr(self.evaluator, "_sync_lights_and_get_obs", None)
        preprocess = getattr(self.evaluator, "_preprocess_obs", None)
        if not callable(sync) or not callable(preprocess):
            raise OracleLowV1Error("official evaluator lacks reviewed observation refresh hooks")
        self.evaluator.obs = preprocess(sync(obs))
        if not isinstance(info, Mapping):
            raise OracleLowV1Error("official evaluator env.step emitted non-mapping info")
        return info


def install_frozen_semantic_subgoal(payload: Mapping[str, Any]) -> Any:
    """Install only the deployment semantic bundle, never a resolver row."""
    module = importlib.import_module("g05.utils.memlite_native_fm_contract")
    installer = getattr(module, "install_local_subgoal", None)
    if not callable(installer):
        raise OracleLowV1Error("atomic source lacks install_local_subgoal")
    # The source-side installer rejects audit, source-id, and physical-truth
    # keys.  This adapter deliberately does not strip fields before admission.
    return installer(payload)


def make_physical_oracle_evaluator(env: Any) -> Callable[[Any], Mapping[str, Any]]:
    """Bind the existing reviewed physical oracle to this official env only."""
    module = importlib.import_module("g05.utils.memlite_native_fm_contract")
    evaluator = getattr(module, "evaluate_installed_subgoal", None)
    if not callable(evaluator):
        raise OracleLowV1Error("atomic source lacks evaluate_installed_subgoal")

    def measure(installed_subgoal: Any) -> Mapping[str, Any]:
        result = evaluator(env, installed_subgoal)
        if not isinstance(result, Mapping) or result.get("metric_kind") != "local_subgoal_success":
            raise OracleLowV1Error("physical oracle returned an invalid local-subgoal metric")
        return result

    return measure


def run_oracle_window_from_official_evaluator(
    harness: OfficialEvaluatorHarness,
    *,
    window: FrozenOracleWindow,
    installed_subgoal: Any,
    request_low_chunk: Callable[[Mapping[str, Any], Any, int], Mapping[str, Any]],
    evaluate_subgoal: Callable[[Any], Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Reset one official instance, then run exactly one frozen L1 window."""
    harness.reset()
    return OracleLowDriver(
        env=harness,
        observation_provider=harness.observation,
        request_low_chunk=request_low_chunk,
        evaluate_subgoal=evaluate_subgoal or make_physical_oracle_evaluator(harness.evaluator.env),
    ).run(window, installed_subgoal)
