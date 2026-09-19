"""Explicit OFFLINE collection capacity profiles; no motion/actor changes."""
from pathlib import Path
from native_teacher_artifacts import ArtifactBudget, MIB

LEGACY_ROOT = "/mnt/sdc1/robodojo/behavior_dev/vlm_sft_native_teacher_20260919"
H09W_ROOT = "/mnt/nvme_tmp/robodojo_vlm_sft_20260919/h09w_native_complete"
LEGACY_PROFILE = "near100"
COMPLETE_PROFILE = "near_complete384"


def capacity_limits(release):
    """No implicit enlargement, mixed profiles or bool-as-int budgets."""
    profile = release.get("capacity_profile", LEGACY_PROFILE)
    if type(profile) is not str or profile not in (LEGACY_PROFILE, COMPLETE_PROFILE):
        raise ValueError("Unknown native-teacher capacity profile")
    expected = {"run_MiB": 100, "total_MiB": 384}
    if profile == COMPLETE_PROFILE:
        expected = {"run_MiB": 384, "total_MiB": 512, "combined_total_MiB": 768}
        if (release.get("experiment_root") != H09W_ROOT or
                release.get("prior_experiment_roots") != [LEGACY_ROOT]):
            raise ValueError("Exact new NVMe and preserved original experiment roots required")
    elif any(k in release for k in ("prior_experiment_roots", "combined_total_MiB")):
        raise ValueError("Cross-root fields require the explicit complete capacity profile")
    for key, value in expected.items():
        if type(release.get(key)) is not int or release[key] != value:
            raise ValueError("Mixed or malformed native-teacher capacity limits")
    return expected["run_MiB"]*MIB, expected["total_MiB"]*MIB


class CombinedArtifactBudget(ArtifactBudget):
    """Recount old evidence on EVERY write/reservation, including cleanup.

    Old evidence is read-only and cannot disappear from the accounting by
    choosing a new output directory. This does not reserve an OS filesystem
    against unrelated users; the collector also retains its 80 GiB free gate.
    """
    def __init__(self, root, total_root, prior_root):
        super().__init__(root, total_root, 384*MIB, 512*MIB)
        self.prior_root = Path(prior_root).resolve()
        if (not self.prior_root.is_dir() or self.prior_root == self.total_root or
                self.prior_root.is_relative_to(self.total_root) or
                self.total_root.is_relative_to(self.prior_root)):
            raise ValueError("Preserved, separate original evidence root required")
        self.check(0)

    def check(self, size, cleanup=False):
        super().check(size, cleanup)
        if not self.prior_root.is_dir():
            raise RuntimeError("Original evidence root disappeared; stop, do not discount it")
        prior = self.used(self.prior_root)
        held = 0 if cleanup else self._held
        reserve = 0 if cleanup else 3*self.reserve
        if (prior > 384*MIB or
                prior+self.used(self.total_root)+held+size > 768*MIB-reserve):
            raise RuntimeError("Combined old/new experiment artifact budget exceeded BEFORE writing")


def near_artifact_budget(output, release):
    run_limit, total_limit = capacity_limits(release)
    if release.get("capacity_profile", LEGACY_PROFILE) == COMPLETE_PROFILE:
        return CombinedArtifactBudget(output, release["experiment_root"], release["prior_experiment_roots"][0])
    return ArtifactBudget(output, release["experiment_root"], run_limit, total_limit)
