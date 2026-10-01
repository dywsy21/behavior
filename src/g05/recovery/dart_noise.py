"""Noise primitives for candidate-only DART data collection.

This module deliberately operates on the native, physical 23-D action space.
The 27-D model representation (including its inactive padding positions) is a
later projection concern and is never accepted by these samplers.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import cos, isfinite, log, pi, sin, sqrt
from random import Random
from typing import Mapping, Sequence

from .common import RecoveryContractError, validate_raw23_action, validate_raw23_actions


RAW_ACTION_DIM = 23


def _finite_number(value: float, name: str) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError) as exc:
        raise RecoveryContractError(f"{name} must be numeric") from exc
    if not isfinite(value):
        raise RecoveryContractError(f"{name} must be finite")
    return value


def _validate_covariance(covariance: Sequence[Sequence[float]]) -> tuple[tuple[float, ...], ...]:
    if len(covariance) != RAW_ACTION_DIM:
        raise RecoveryContractError("DART covariance must be 23x23 native-action covariance")
    rows: list[tuple[float, ...]] = []
    for row_index, row in enumerate(covariance):
        if len(row) != RAW_ACTION_DIM:
            raise RecoveryContractError(f"DART covariance row {row_index} is not length 23")
        rows.append(tuple(_finite_number(value, f"covariance[{row_index}]") for value in row))
    for i in range(RAW_ACTION_DIM):
        for j in range(i):
            if abs(rows[i][j] - rows[j][i]) > 1e-10:
                raise RecoveryContractError("DART covariance must be symmetric")
    return tuple(rows)


def _cholesky_psd(covariance: Sequence[Sequence[float]]) -> tuple[tuple[float, ...], ...]:
    """Return a lower factor, including valid degenerate zero-covariance cases."""

    matrix = _validate_covariance(covariance)
    lower = [[0.0] * RAW_ACTION_DIM for _ in range(RAW_ACTION_DIM)]
    tolerance = 1e-10
    for i in range(RAW_ACTION_DIM):
        for j in range(i + 1):
            residual = matrix[i][j] - sum(lower[i][k] * lower[j][k] for k in range(j))
            if i == j:
                if residual < -tolerance:
                    raise RecoveryContractError("DART covariance must be positive semidefinite")
                lower[i][j] = sqrt(max(residual, 0.0))
            elif lower[j][j] > tolerance:
                lower[i][j] = residual / lower[j][j]
            elif abs(residual) > tolerance:
                raise RecoveryContractError("DART covariance is not positive semidefinite")
    return tuple(tuple(row) for row in lower)


def _standard_normals(rng: Random, count: int) -> tuple[float, ...]:
    """A local Box--Muller stream makes the seed receipt fully reproducible."""

    values: list[float] = []
    while len(values) < count:
        # random() is in [0, 1); guard the (astronomically unlikely) zero.
        u1 = max(rng.random(), 2.2250738585072014e-308)
        u2 = rng.random()
        radius = sqrt(-2.0 * log(u1))
        values.extend((radius * cos(2.0 * pi * u2), radius * sin(2.0 * pi * u2)))
    return tuple(values[:count])


@dataclass(frozen=True)
class CovarianceTrajectory:
    """Held-out train-group actions used for the paper's covariance estimate."""

    clean_supervisor_actions23: tuple[tuple[float, ...], ...]
    learner_actions23: tuple[tuple[float, ...], ...]

    def __post_init__(self) -> None:
        clean = validate_raw23_actions(self.clean_supervisor_actions23)
        learner = validate_raw23_actions(self.learner_actions23)
        if not clean or len(clean) != len(learner):
            raise RecoveryContractError("covariance trajectories require equal, nonempty clean and learner actions")
        object.__setattr__(self, "clean_supervisor_actions23", clean)
        object.__setattr__(self, "learner_actions23", learner)


def estimate_dart_covariance(trajectories: Sequence[CovarianceTrajectory]) -> tuple[tuple[float, ...], ...]:
    """Paper/code estimator: average each trajectory's error outer-products, then trajectories.

    This mirrors ``BerkeleyAutomation/DART/experiments/tools/noise.py`` rather
    than pooling time steps across source groups.  Callers must supply only
    held-out *training* calibration groups; split policy lives in collection.
    """

    if not trajectories:
        raise RecoveryContractError("at least one held-out training calibration trajectory is required")
    covariance = [[0.0] * RAW_ACTION_DIM for _ in range(RAW_ACTION_DIM)]
    for trajectory in trajectories:
        per_trajectory = [[0.0] * RAW_ACTION_DIM for _ in range(RAW_ACTION_DIM)]
        length = len(trajectory.clean_supervisor_actions23)
        for clean, learner in zip(trajectory.clean_supervisor_actions23, trajectory.learner_actions23):
            error = [clean[index] - learner[index] for index in range(RAW_ACTION_DIM)]
            for row in range(RAW_ACTION_DIM):
                for column in range(RAW_ACTION_DIM):
                    per_trajectory[row][column] += error[row] * error[column] / length
        for row in range(RAW_ACTION_DIM):
            for column in range(RAW_ACTION_DIM):
                covariance[row][column] += per_trajectory[row][column] / len(trajectories)
    return _validate_covariance(covariance)


def scale_dart_covariance(
    covariance: Sequence[Sequence[float]], *, alpha: float, horizon: int
) -> tuple[tuple[float, ...], ...]:
    """Equation (4): ``alpha / (T * tr(Sigma)) * Sigma``.

    It intentionally leaves a zero covariance at zero when ``alpha == 0`` and
    rejects a positive-noise request whose trace cannot be scaled.
    """

    matrix = _validate_covariance(covariance)
    alpha = _finite_number(alpha, "alpha")
    if alpha < 0:
        raise RecoveryContractError("alpha must be nonnegative")
    if not isinstance(horizon, int) or horizon <= 0:
        raise RecoveryContractError("DART horizon must be a positive integer")
    trace = sum(matrix[index][index] for index in range(RAW_ACTION_DIM))
    if trace < -1e-10:
        raise RecoveryContractError("DART covariance trace cannot be negative")
    if alpha == 0:
        return tuple(tuple(0.0 for _ in range(RAW_ACTION_DIM)) for _ in range(RAW_ACTION_DIM))
    if trace <= 1e-12:
        raise RecoveryContractError("cannot scale zero-trace covariance for positive DART alpha")
    scale = alpha / (horizon * trace)
    return tuple(tuple(scale * value for value in row) for row in matrix)


@dataclass(frozen=True)
class OriginalGaussianSample:
    requested_noisy23: tuple[float, ...]
    sampled_noise23: tuple[float, ...]


class OriginalGaussianNoise:
    """Exact unbounded Gaussian action sampling used by the original DART code."""

    algorithm_id = "dart_original_gaussian_unbounded"

    def __init__(self, covariance: Sequence[Sequence[float]], *, seed: int) -> None:
        if not isinstance(seed, int):
            raise RecoveryContractError("original DART seed must be an integer")
        self._covariance = _validate_covariance(covariance)
        self._lower = _cholesky_psd(self._covariance)
        self._rng = Random(seed)
        self.seed = seed

    @property
    def covariance(self) -> tuple[tuple[float, ...], ...]:
        return self._covariance

    def sample(self, clean_intended23: Sequence[float]) -> OriginalGaussianSample:
        intended = tuple(validate_raw23_action(clean_intended23))
        normals = _standard_normals(self._rng, RAW_ACTION_DIM)
        noise = tuple(
            sum(self._lower[row][column] * normals[column] for column in range(row + 1))
            for row in range(RAW_ACTION_DIM)
        )
        return OriginalGaussianSample(
            requested_noisy23=tuple(intended[index] + noise[index] for index in range(RAW_ACTION_DIM)),
            sampled_noise23=noise,
        )


@dataclass(frozen=True)
class ActionPart:
    """One physical native-action part, ordered exactly as supplied by metadata."""

    name: str
    dimension: int
    unit: str

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise RecoveryContractError("action part needs a nonempty metadata name")
        if not isinstance(self.dimension, int) or self.dimension <= 0:
            raise RecoveryContractError(f"action part {self.name!r} needs a positive dimension")
        if not isinstance(self.unit, str) or not self.unit:
            raise RecoveryContractError(f"action part {self.name!r} needs a declared unit")


@dataclass(frozen=True)
class Raw23ActionLayout:
    """Native layout derived from embodiment metadata, never raw-axis constants."""

    parts: tuple[ActionPart, ...]

    def __post_init__(self) -> None:
        if not self.parts or sum(part.dimension for part in self.parts) != RAW_ACTION_DIM:
            raise RecoveryContractError("metadata parts must describe exactly the 23 native action dimensions")
        names = [part.name for part in self.parts]
        if len(names) != len(set(names)):
            raise RecoveryContractError("metadata action part names must be unique")

    @property
    def slices(self) -> Mapping[str, slice]:
        offset = 0
        result: dict[str, slice] = {}
        for part in self.parts:
            result[part.name] = slice(offset, offset + part.dimension)
            offset += part.dimension
        return result


@dataclass(frozen=True)
class BoundedPartRule:
    """A safety rule for a named physical part in a DART-inspired sampler."""

    active: bool
    standard_deviation: float = 0.0
    cap: float = 0.0
    inactive_reason: str | None = None

    def __post_init__(self) -> None:
        std = _finite_number(self.standard_deviation, "bounded standard_deviation")
        cap = _finite_number(self.cap, "bounded cap")
        if std < 0 or cap < 0:
            raise RecoveryContractError("bounded noise standard deviation and cap must be nonnegative")
        if self.active and (std <= 0 or cap <= 0):
            raise RecoveryContractError("active bounded noise parts need positive standard deviation and cap")
        if not self.active and not self.inactive_reason:
            raise RecoveryContractError("every unperturbed physical part needs an explicit reason")


@dataclass(frozen=True)
class BoundedNoiseProfile:
    """Safety-bounded, temporally correlated noise; explicitly not original DART."""

    layout: Raw23ActionLayout
    rules_by_part: Mapping[str, BoundedPartRule]
    temporal_rho: float
    profile_id: str

    algorithm_id = "dart_inspired_bounded_correlated"

    def __post_init__(self) -> None:
        rho = _finite_number(self.temporal_rho, "temporal_rho")
        if rho < 0 or rho >= 1:
            raise RecoveryContractError("temporal_rho must be in [0, 1)")
        names = {part.name for part in self.layout.parts}
        if set(self.rules_by_part) != names:
            raise RecoveryContractError("bounded noise must explicitly declare every native action part")
        if not isinstance(self.profile_id, str) or not self.profile_id:
            raise RecoveryContractError("bounded profile needs a profile_id")


@dataclass(frozen=True)
class BoundedNoiseSample:
    requested_noisy23: tuple[float, ...]
    sampled_noise23: tuple[float, ...]


class DartInspiredBoundedNoise:
    """An auditable bounded sampler, with causal temporal correlation only."""

    def __init__(self, profile: BoundedNoiseProfile, *, seed: int) -> None:
        if not isinstance(seed, int):
            raise RecoveryContractError("bounded DART-inspired seed must be an integer")
        self.profile = profile
        self.seed = seed
        self._rng = Random(seed)
        self._previous_noise = (0.0,) * RAW_ACTION_DIM

    def sample(self, clean_intended23: Sequence[float]) -> BoundedNoiseSample:
        intended = tuple(validate_raw23_action(clean_intended23))
        standard_normals = iter(_standard_normals(self._rng, RAW_ACTION_DIM))
        noise = [0.0] * RAW_ACTION_DIM
        for part in self.profile.layout.parts:
            rule = self.profile.rules_by_part[part.name]
            action_slice = self.profile.layout.slices[part.name]
            for index in range(action_slice.start, action_slice.stop):
                normal = next(standard_normals)
                if rule.active:
                    candidate = (
                        self.profile.temporal_rho * self._previous_noise[index]
                        + sqrt(1.0 - self.profile.temporal_rho**2) * rule.standard_deviation * normal
                    )
                    noise[index] = max(-rule.cap, min(rule.cap, candidate))
        self._previous_noise = tuple(noise)
        return BoundedNoiseSample(
            requested_noisy23=tuple(intended[index] + noise[index] for index in range(RAW_ACTION_DIM)),
            sampled_noise23=tuple(noise),
        )
