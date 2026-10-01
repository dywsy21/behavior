from __future__ import annotations

try:
    import pytest
except ImportError:  # Allows the focused contract tests on the minimal CPU image.
    import re

    class _Approx:
        def __init__(self, expected: float) -> None:
            self.expected = expected

        def __eq__(self, actual: object) -> bool:
            return isinstance(actual, (int, float)) and abs(actual - self.expected) <= 1e-12

    class _Raises:
        def __init__(self, error: type[Exception], match: str) -> None:
            self.error = error
            self.match = match

        def __enter__(self) -> None:
            return None

        def __exit__(self, error_type: type[Exception] | None, error: Exception | None, traceback: object) -> bool:
            if error_type is None:
                raise AssertionError(f"expected {self.error.__name__}")
            if not issubclass(error_type, self.error) or not re.search(self.match, str(error)):
                return False
            return True

    class _PytestFallback:
        @staticmethod
        def approx(expected: float) -> _Approx:
            return _Approx(expected)

        @staticmethod
        def raises(error: type[Exception], *, match: str) -> _Raises:
            return _Raises(error, match)

    pytest = _PytestFallback()

from g05.recovery.common import RecoveryContractError
from g05.recovery.dart_noise import (
    ActionPart,
    BoundedNoiseProfile,
    BoundedPartRule,
    CovarianceTrajectory,
    DartInspiredBoundedNoise,
    OriginalGaussianNoise,
    Raw23ActionLayout,
    estimate_dart_covariance,
    scale_dart_covariance,
)


def _action(value: float, *, axis: int = 0) -> tuple[float, ...]:
    result = [0.0] * 23
    result[axis] = value
    return tuple(result)


def _diagonal(value: float) -> tuple[tuple[float, ...], ...]:
    return tuple(tuple(value if row == column else 0.0 for column in range(23)) for row in range(23))


def test_original_covariance_estimator_and_equation_four_scale_time_steps() -> None:
    # First trajectory mean square error is (1^2 + 3^2) / 2 = 5; the second
    # is (2^2 + 4^2) / 2 = 10; the author-code estimator averages trajectories.
    covariance = estimate_dart_covariance(
        [
            CovarianceTrajectory((_action(1), _action(3)), (_action(0), _action(0))),
            CovarianceTrajectory((_action(2), _action(4)), (_action(0), _action(0))),
        ]
    )
    assert covariance[0][0] == pytest.approx(7.5)
    assert covariance[1][1] == 0.0
    scaled = scale_dart_covariance(covariance, alpha=3.0, horizon=6)
    # Eq.4 gives alpha/(T*trace)*Sigma, so resulting trace is alpha/T.
    assert sum(scaled[index][index] for index in range(23)) == pytest.approx(0.5)
    assert scaled[0][0] == pytest.approx(0.5)


def test_original_gaussian_is_seeded_and_handles_degenerate_zero_covariance() -> None:
    first = OriginalGaussianNoise(_diagonal(0.25), seed=71).sample(_action(1.0))
    second = OriginalGaussianNoise(_diagonal(0.25), seed=71).sample(_action(1.0))
    assert first == second
    assert first.requested_noisy23 != _action(1.0)

    zero = OriginalGaussianNoise(_diagonal(0.0), seed=71).sample(_action(1.0))
    assert zero.requested_noisy23 == _action(1.0)
    assert zero.sampled_noise23 == _action(0.0)


def test_original_gaussian_rejects_non_psd_covariance() -> None:
    invalid = [list(row) for row in _diagonal(0.0)]
    invalid[0][1] = invalid[1][0] = 1.0
    with pytest.raises(RecoveryContractError, match="positive semidefinite"):
        OriginalGaussianNoise(invalid, seed=1)
    with pytest.raises(RecoveryContractError, match="finite"):
        OriginalGaussianNoise(_diagonal(1.0), seed=1).sample((float("nan"),) * 23)


def test_bounded_noise_uses_metadata_named_parts_and_explicit_unperturbed_parts() -> None:
    # Deliberately not the familiar raw-axis ordering: the sampler derives
    # offsets from these metadata parts instead of hard-coding 23-D indices.
    layout = Raw23ActionLayout(
        (
            ActionPart("right_gripper", 1, "position"),
            ActionPart("base_qvel", 3, "velocity"),
            ActionPart("left_arm", 7, "position"),
            ActionPart("trunk_qpos", 4, "position"),
            ActionPart("right_arm", 7, "position"),
            ActionPart("left_gripper", 1, "position"),
        ),
        embodiment_metadata_sha256="a" * 64,
        model_projection_manifest_sha256="b" * 64,
    )
    profile = BoundedNoiseProfile(
        layout=layout,
        rules_by_part={
            "right_gripper": BoundedPartRule(False, inactive_reason="not in initial hypothesis"),
            "base_qvel": BoundedPartRule(False, inactive_reason="declared safety bound for initial hypothesis"),
            "left_arm": BoundedPartRule(True, standard_deviation=2.0, cap=0.1),
            "trunk_qpos": BoundedPartRule(False, inactive_reason="declared safety bound for initial hypothesis"),
            "right_arm": BoundedPartRule(False, inactive_reason="not in initial hypothesis"),
            "left_gripper": BoundedPartRule(False, inactive_reason="not in initial hypothesis"),
        },
        temporal_rho=0.5,
        profile_id="named-parts-v1",
    )
    sample = DartInspiredBoundedNoise(profile, seed=4).sample(_action(0.0))
    slices = layout.slices
    assert all(sample.sampled_noise23[index] == 0.0 for index in range(slices["base_qvel"].start, slices["base_qvel"].stop))
    assert all(sample.sampled_noise23[index] == 0.0 for index in range(slices["trunk_qpos"].start, slices["trunk_qpos"].stop))
    assert any(sample.sampled_noise23[index] != 0.0 for index in range(slices["left_arm"].start, slices["left_arm"].stop))
    assert max(abs(value) for value in sample.sampled_noise23) <= 0.1


if __name__ == "__main__":
    for name, function in sorted(globals().items()):
        if name.startswith("test_") and callable(function):
            function()
    print("dart noise contract tests passed")
