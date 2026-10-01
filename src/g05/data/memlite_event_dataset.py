"""Read one explicitly selected P107 event view from a sealed preparation pack.

This is not a compatibility layer for ``MemLiteSidecar`` or the old recovery
append dataset.  P107 records keep four labels separate, and a sealed package
created during preparation still cannot authorize stage3 training.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


REPO = Path(__file__).resolve().parents[3]
RELEASE_SCHEMA = "memlite-event-label-release-v1"


def _load_protocol() -> Any:
    path = REPO / "src/g05/data/memlite_event_protocol.py"
    spec = importlib.util.spec_from_file_location("p107_memlite_event_protocol_dataset", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load P107 protocol: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_actor_query_contract() -> Any:
    path = REPO / "src/g05/data/memlite_event_actor_query.py"
    spec = importlib.util.spec_from_file_location("p107_memlite_event_actor_query_dataset", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load P107 actor-query contract: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_protocol = _load_protocol()
_actor_query_contract = _load_actor_query_contract()
SCHEMA_VERSION = _protocol.SCHEMA_VERSION
LABEL_KINDS = frozenset(_protocol.LABEL_KINDS)
canonical_json = _protocol.canonical_json
canonical_sha256 = _protocol.canonical_sha256
actor_evidence_projection = _protocol.actor_evidence_projection
project_action_23_to_27 = _protocol.project_action_23_to_27
action_payload_sha256 = _protocol.action_payload_sha256
raw_action_payload_sha256 = _protocol.raw_action_payload_sha256
validate_event = _protocol.validate_event
validate_goal_satisfaction_view = _protocol.validate_goal_satisfaction_view
validate_attempt_outcome_view = _protocol.validate_attempt_outcome_view
validate_recovery_decision_view = _protocol.validate_recovery_decision_view
validate_corrective_action_view = _protocol.validate_corrective_action_view
load_corrective_action_authority = _protocol.load_corrective_action_authority
authority_raw_action_payload = _protocol.authority_raw_action_payload
actor_query_projection = _actor_query_contract.actor_query_projection
read_actor_query_sidecar = _actor_query_contract.read_actor_query_sidecar


def _read_json(path: Path) -> Any:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"sealed package input is missing or symlinked: {path}")
    if path.stat().st_mode & 0o222:
        raise ValueError(f"sealed package input remains writable: {path}")
    return json.loads(path.read_bytes())


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"sealed package input is missing or symlinked: {path}")
    rows = []
    for line in path.read_bytes().splitlines():
        if line.strip():
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError("sealed JSONL record is not an object")
            rows.append(value)
    return rows


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_view(view: Mapping[str, Any], event: Mapping[str, Any], *, authority: Any | None = None) -> None:
    validators = {
        "goal_satisfaction_counterfactual": validate_goal_satisfaction_view,
        "attempt_outcome": validate_attempt_outcome_view,
        "recovery_decision": validate_recovery_decision_view,
        "corrective_action": validate_corrective_action_view,
    }
    kind = view.get("label_kind")
    if kind not in validators:
        raise ValueError(f"unknown explicit P107 view: {kind}")
    if kind == "corrective_action":
        validators[kind](view, event, authority=authority)
    else:
        validators[kind](view, event)


def _assert_safe_actor_evidence(value: Any, path: str = "actor_evidence") -> None:
    forbidden = ("private", "pose", "predicate", "snapshot", "teacher", "future", "truth", "oracle",
                 "simulator", "provider", "contact_state", "robot_state")
    if isinstance(value, Mapping):
        for key, child in value.items():
            if any(token in str(key).casefold() for token in forbidden):
                raise ValueError(f"actor evidence leaks a provider/private field at {path}.{key}")
            _assert_safe_actor_evidence(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _assert_safe_actor_evidence(child, f"{path}[{index}]")


class MemLiteEventDataset(Sequence[dict[str, Any]]):
    """A read-only, view-specific P107 dataset.

    ``for_training=True`` is deliberately rejected until a separate future
    authorization changes the release contract.  The returned rows expose only
    actor-visible evidence plus the selected target; private provider evidence
    remains absent even for diagnostic reads.
    """

    def __init__(self, release_dir: str | Path, view_kind: str, *, expected_release_seal_sha256: str,
                 corrective_publisher_root: str | Path | None = None,
                 index_root: str | Path | None = None, for_training: bool = False):
        if view_kind not in LABEL_KINDS:
            raise ValueError("view_kind must be one of the four explicit P107 label views")
        self.release_dir = Path(release_dir)
        if self.release_dir.is_symlink() or not self.release_dir.is_dir():
            raise ValueError("release_dir must be a sealed regular directory")
        if self.release_dir.stat().st_mode & 0o222:
            raise ValueError("release_dir is not immutable")
        manifest_path = self.release_dir / "release_manifest.json"
        seal_path = self.release_dir / "release_seal.json"
        self.manifest = _read_json(manifest_path)
        seal = _read_json(seal_path)
        if (not isinstance(expected_release_seal_sha256, str) or len(expected_release_seal_sha256) != 64 or
                any(char not in "0123456789abcdef" for char in expected_release_seal_sha256)):
            raise ValueError("dataset needs an externally recorded release seal SHA-256")
        if _sha256(seal_path) != expected_release_seal_sha256:
            raise ValueError("externally recorded release seal SHA-256 does not match package")
        if self.manifest.get("schema_version") != RELEASE_SCHEMA or self.manifest.get("protocol_schema_version") != SCHEMA_VERSION:
            raise ValueError("not a compatible P107 event release")
        if seal.get("release_manifest_sha256") != _sha256(manifest_path):
            raise ValueError("release manifest seal mismatch")
        if seal.get("files") != self.manifest.get("files"):
            raise ValueError("release manifest/seal file receipts disagree")
        if (self.manifest.get("ready_for_training") is not False or self.manifest.get("training_run_authorized") is not False or
                self.manifest.get("stage3_training_authorized") is not False):
            raise ValueError("P107 preparation package must not self-authorize training")
        if for_training:
            raise ValueError("P107 data preparation is not a stage3 training authorization")
        files = self.manifest.get("files")
        if not isinstance(files, Mapping):
            raise ValueError("release manifest lacks immutable file receipts")
        paths = {name: self.release_dir / name for name in ("source_groups.jsonl", "events.jsonl", "views.jsonl", "actions.jsonl")}
        for name, path in paths.items():
            receipt = files.get(name)
            if not isinstance(receipt, Mapping) or receipt.get("sha256") != _sha256(path):
                raise ValueError(f"sealed package record receipt mismatch: {name}")
        events = _read_jsonl(paths["events.jsonl"])
        event_by_id = {row.get("event_id"): row for row in events}
        if len(event_by_id) != len(events):
            raise ValueError("duplicate event IDs in sealed package")
        group_rows = _read_jsonl(paths["source_groups.jsonl"])
        groups = {row.get("source_group_id"): row for row in group_rows}
        if len(groups) != len(group_rows):
            raise ValueError("duplicate source-group IDs in sealed package")
        if self.manifest.get("source_group_policy_sha256") != canonical_sha256(
                sorted(group_rows, key=lambda row: row["source_group_id"])):
            raise ValueError("sealed package source-group policy digest mismatch")
        capability = self.manifest.get("corrective_authority_capability")
        if capability is None:
            if corrective_publisher_root is not None or index_root is not None:
                raise ValueError("release has no corrective authority capability but publisher/index roots were supplied")
            authority = None
        else:
            if (not isinstance(capability, Mapping) or set(capability) != {
                    "publisher_manifest_sha256", "index_inventory_seal_sha256",
                    "accepted_live_runtime_acceptance_root_sha256"} or
                    capability.get("index_inventory_seal_sha256") != self.manifest.get("immutable_index_inventory_seal_sha256")):
                raise ValueError("sealed package corrective authority capability is malformed")
            if corrective_publisher_root is None or index_root is None:
                raise ValueError("release with positive-capable corrective evidence requires separate publisher and sealed index roots")
            authority = load_corrective_action_authority(
                Path(corrective_publisher_root),
                index_root=Path(index_root),
                expected_publisher_manifest_sha256=capability["publisher_manifest_sha256"],
                expected_index_inventory_seal_sha256=capability["index_inventory_seal_sha256"],
                expected_live_runtime_acceptance_root_sha256=capability["accepted_live_runtime_acceptance_root_sha256"])
        action_rows = _read_jsonl(paths["actions.jsonl"])
        actions = {row.get("view_id"): row for row in action_rows}
        if len(actions) != len(action_rows):
            raise ValueError("duplicate action payload IDs in sealed package")
        view_wrappers = _read_jsonl(paths["views.jsonl"])
        actor_query_by_view_id: dict[str, Mapping[str, Any]] = {}
        prelabel_registry: dict[str, Mapping[str, Any]] = {}
        if "actor_query_sidecar" in self.manifest:
            sidecar_metadata = self.manifest["actor_query_sidecar"]
            if not isinstance(sidecar_metadata, Mapping):
                raise ValueError("actor_query_sidecar manifest registration must be an object")
            # Validate every sidecar binding before selecting ``view_kind``.
            # A malformed OTHER-kind row must not become invisible merely
            # because this dataset instance projects a different kind.
            views_by_id: dict[str, Mapping[str, Any]] = {}
            for wrapper in view_wrappers:
                if not isinstance(wrapper, Mapping) or not isinstance(wrapper.get("view"), Mapping):
                    continue
                wrapper_view = wrapper["view"]
                wrapper_view_id = wrapper_view.get("view_id")
                if isinstance(wrapper_view_id, str):
                    if wrapper_view_id in views_by_id:
                        raise ValueError("duplicate view_id while loading actor-query sidecar")
                    views_by_id[wrapper_view_id] = wrapper_view
            actor_query_by_view_id, prelabel_registry = read_actor_query_sidecar(
                self.release_dir, sidecar_metadata, files=files,
                expected_view_ids=set(views_by_id), views_by_id=views_by_id,
                events_by_id=event_by_id)
        rows: list[dict[str, Any]] = []
        for wrapper in view_wrappers:
            view, gate = wrapper.get("view"), wrapper.get("dataset_quality_gates")
            if not isinstance(view, Mapping) or not isinstance(gate, Mapping):
                raise ValueError("sealed view wrapper lacks view/dataset-quality gates")
            event = event_by_id.get(view.get("event_id"))
            if event is None:
                raise ValueError("sealed view references no event")
            group = groups.get(view.get("source_group_id"))
            if not isinstance(group, Mapping) or wrapper.get("usage_role") != group.get("usage_role") or wrapper.get("original_split") != group.get("original_split"):
                raise ValueError("sealed view changed its source-group role or inherited split")
            validate_event(event)
            source = event["source"]
            if (source["source_group_id"] != group["source_group_id"] or source["original_split"] != group["original_split"] or
                    source["source_release_manifest_sha256"] != group["source_release_manifest_sha256"] or
                    source["task_index"] != group["task_index"] or source["task_instance_id"] != group["task_instance_id"]):
                raise ValueError("event changed its immutable source-group provenance")
            _validate_view(view, event, authority=authority)
            if view["label_kind"] != view_kind:
                continue
            row = self._project_row(
                view, event, gate, actions.get(view["view_id"]), authority=authority,
                actor_query=actor_query_by_view_id.get(view["view_id"]),
                prelabel_registry=prelabel_registry)
            rows.append(row)
        self.view_kind = view_kind
        self._rows = tuple(rows)
        self.training_allowed = False

    @staticmethod
    def _project_row(view: Mapping[str, Any], event: Mapping[str, Any], gate: Mapping[str, Any],
                     action: Mapping[str, Any] | None, *, authority: Any | None,
                     actor_query: Mapping[str, Any] | None = None,
                     prelabel_registry: Mapping[str, Mapping[str, Any]] | None = None) -> dict[str, Any]:
        # Deliberately reconstruct instead of passing the view through: this
        # prevents privileged_evidence, provider snapshots, teacher state, or
        # future truth from reaching a consumer by accidental dict expansion.
        result: dict[str, Any] = {
            "event_id": view["event_id"], "view_id": view["view_id"], "label_kind": view["label_kind"],
            "source_group_id": view["source_group_id"], "observation_frame": view["observation_frame"],
            "actor_evidence": actor_evidence_projection(view), "dataset_quality_gates": dict(gate),
            "action_start_frame": event["action"]["start_frame"],
        }
        _assert_safe_actor_evidence(result["actor_evidence"])
        if actor_query is not None:
            result["actor_query"] = actor_query_projection(
                actor_query, observation_frame=view["observation_frame"],
                view_kind=view["label_kind"], view=view, event=event,
                prelabel_registry=prelabel_registry)
        kind = view["label_kind"]
        if kind == "goal_satisfaction_counterfactual":
            result.update(goal_relation=view["goal_relation"], goal_satisfaction=view["goal_satisfaction"],
                          valid_goal_mask=view["valid_goal_mask"], low_action_supervision_mask=False)
        elif kind == "attempt_outcome":
            result.update(attempt_id=view["attempt_id"], attempt_outcome=view["attempt_outcome"],
                          valid_result_mask=view["valid_result_mask"], low_action_supervision_mask=False)
        elif kind == "recovery_decision":
            result.update(decision=view["decision"], target_bundle_id=view["target_bundle_id"],
                          decision_supervision_mask=view["decision_supervision_mask"], low_action_supervision_mask=False)
        else:
            if not isinstance(action, Mapping):
                raise ValueError("corrective action view has no actual 23D action payload")
            if (action.get("view_id") != view["view_id"] or
                    action.get("raw_action_sha256") != view["executed_action_receipt"]["raw_action_sha256"] or
                    action.get("action_payload_sha256") != view["action_payload_sha256"]):
                raise ValueError("corrective action payload/receipt identity mismatch")
            raw = action.get("raw_actions_23")
            raw_artifact_sha = action.get("raw_action_artifact_sha256")
            if (action_payload_sha256(raw) != view["action_payload_sha256"] or
                    raw_action_payload_sha256(raw, raw_action_artifact_sha256=raw_artifact_sha) != action["raw_action_sha256"]):
                raise ValueError("stored corrective raw actions do not match their sealed payload/artifact identity")
            if view["low_action_supervision_mask"]:
                if authority is None:
                    raise ValueError("positive corrective row lacks a sealed parent publisher authority")
                sealed = authority_raw_action_payload(authority, view["action_payload_sha256"])
                if canonical_json(sealed) != canonical_json({
                        "schema_version": sealed["schema_version"], "action_payload_sha256": action["action_payload_sha256"],
                        "raw_action_sha256": action["raw_action_sha256"],
                        "raw_action_artifact_sha256": raw_artifact_sha, "raw_actions_23": raw}):
                    raise ValueError("packaged positive action differs from sealed parent publisher payload")
            projected = project_action_23_to_27(raw, view["actual_executed_length"])
            expected = {"actions_27": projected["actions_27"], "action_is_pad": projected["action_is_pad"],
                        "action_dim_is_pad": projected["action_dim_is_pad"]}
            if {key: action.get(key) for key in expected} != expected:
                raise ValueError("stored 23D->27D action projection/padding receipt changed")
            result.update(action_intent_bundle_id=view["action_intent_bundle_id"],
                          actual_executed_length=view["actual_executed_length"], raw_actions_23=raw,
                          actions_27=expected["actions_27"], action_is_pad=expected["action_is_pad"],
                          action_dim_is_pad=expected["action_dim_is_pad"],
                          low_action_supervision_mask=view["low_action_supervision_mask"])
        return result

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return dict(self._rows[index])
