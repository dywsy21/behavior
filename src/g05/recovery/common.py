"""Small dependency-free validation helpers shared by recovery tooling."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from hashlib import sha256
import json
import math
from typing import Any, Iterable, Mapping, Sequence


class RecoveryContractError(ValueError):
    """A recovery receipt would be ambiguous, unsafe, or non-reproducible."""


def _normalise(value: Any) -> Any:
    """Return a deliberately small canonical JSON domain for receipt hashing.

    Opaque simulator state must never be passed here.  A live adapter has to
    provide a documented, explicit fingerprint for such state instead.
    """
    if is_dataclass(value):
        value = asdict(value)
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise RecoveryContractError("Receipt values must be finite")
        return value
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise RecoveryContractError("Receipt mapping keys must be strings")
        return {key: _normalise(value[key]) for key in sorted(value)}
    if isinstance(value, (list, tuple)):
        return [_normalise(item) for item in value]
    raise RecoveryContractError(
        "Opaque values need an adapter-provided fingerprint, not repr/pickle"
    )


def canonical_json(value: Any) -> str:
    return json.dumps(_normalise(value), sort_keys=True, separators=(",", ":"), allow_nan=False)


def canonical_sha256(value: Any) -> str:
    return sha256(canonical_json(value).encode("utf-8")).hexdigest()


def require_sha256(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise RecoveryContractError(f"{field} must be a lowercase SHA-256 hex digest")
    return value


def validate_raw23_action(action: Any) -> list[float]:
    if not isinstance(action, Sequence) or isinstance(action, (str, bytes, bytearray)):
        raise RecoveryContractError("An official action must be a 23-D sequence")
    if len(action) != 23:
        raise RecoveryContractError("Only official 23-D actions may enter recovery execution")
    result = []
    for value in action:
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise RecoveryContractError("Official actions must be finite numeric values")
        result.append(float(value))
    return result


def validate_raw23_actions(actions: Iterable[Any]) -> list[list[float]]:
    if isinstance(actions, (str, bytes, bytearray)):
        raise RecoveryContractError("Actions must be an ordered sequence, not text")
    result = [validate_raw23_action(action) for action in actions]
    if not result:
        raise RecoveryContractError("A recovery branch must execute at least one actual action")
    return result


def validate_state61(state: Any) -> list[float]:
    if not isinstance(state, Sequence) or isinstance(state, (str, bytes, bytearray)) or len(state) != 61:
        raise RecoveryContractError("Observed robot state must be the official 61-D vector")
    result = []
    for value in state:
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise RecoveryContractError("Observed robot state must be finite")
        result.append(float(value))
    return result
