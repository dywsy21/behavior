"""Shared, explicit provenance/configuration for the five-task FM adapter run."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path


TASK5_RUN = Path("/mnt/sdc1/robodojo/outputs/g05/behavior5_r1pro/behavior5_bs8_taskstrat5_trainstats_shuffle_5k_20260903_2215")
HIGH_RUN = Path("/mnt/sdc1/robodojo/outputs/g05/r1pro_memlite_ar_v10/behavior5_memlite_ar_v10_train_20260907T094716Z")
RECOVERY = Path("/mnt/sdc1/robodojo/datasets/memlite_recovery_v10_20260907")
SIDECAR = Path("/mnt/sdc1/robodojo/datasets/memlite_annotations_task0_4_causal_v9_final_20260906/meta/memlite_annotations.parquet")
STATS = Path("/mnt/sdc1/robodojo/stats/g05/behavior5_r1pro_trainonly_taskstrat5_stats_v2.json")
COHORT = Path("/mnt/sdc1/robodojo/behavior_dev/memlite_v10_impl.jO1FmC/heldout_motion_cohort")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for part in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def environment():
    return {"MEMLITE_FM_INIT_CKPT": str(TASK5_RUN / "checkpoints/step_5000.pt"),
            "MEMLITE_HIGH_CKPT": str(HIGH_RUN / "checkpoints/step_5000.pt"),
            "MEMLITE_TASK_EVENT_INDEX": str(RECOVERY / "motion_recovery_index_v2.json"),
            "MEMLITE_RECOVERY_MANIFEST": str(RECOVERY / "recovery_manifest.json"),
            "G05_OUTPUT_DIR": "/mnt/sdc1/robodojo/outputs/g05"}


def config(overrides=()):
    from hydra import compose, initialize_config_dir
    from g05.utils.config.config_resolvers import register_default_resolvers
    register_default_resolvers()
    os.environ.update(environment())
    repo = Path(__file__).resolve().parents[1]
    with initialize_config_dir(version_base=None, config_dir=str(repo / "configs")):
        cfg = compose(config_name="train", overrides=["task=r1pro_memlite_fm_v11",
            f"data.embodiment_datasets.galaxea_r1pro.memlite_sidecar={SIDECAR}",
            f"tokenizer.vq_config.ckpt_dir={TASK5_RUN / 'action_tokenizer.pt'}",
            "model.model_arch.attn_implementation=sdpa", *overrides])
    # compose() is not a running Hydra job; diagnostics must not pretend to
    # have a hydra.runtime output directory. Formal finetune has its own one.
    cfg.output_dir = str(repo.parent / "unlaunched_fm_diagnostic")
    cfg.logger.task = "r1pro_memlite_fm_v11"
    return cfg


def source_hashes(repo):
    return {str(path.relative_to(repo)): sha(path)
            for directory in ("src", "configs", "scripts")
            for path in sorted((repo / directory).rglob("*"))
            if path.is_file() and path.suffix in {".py", ".yaml", ".sh"}}
