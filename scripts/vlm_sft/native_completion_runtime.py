"""Exact warmstart/config/authorization/storage gates for isolated H09AB runs."""
import json
import hashlib
import os
from pathlib import Path
import subprocess

from common import sha
from native_actor_protocol import check_sha
from native_completion_protocol import VERSION, canonical, VARIANTS
from native_completion_training_data import DATA_SHA, STATES_SHA, REVIEW_SHA, MOTION_DATA_SHA
from native_motion_codec import ACTOR_VERSION, VERSION as MOTION_CODEC
from native_execution import CARRY_PROFILE, metadata as execution_metadata
from native_train import BASE, BASE_WEIGHT_SHA, base_identity
from native_reference_profile import ROOT
from native_storage import RuntimeStorage, runtime_environment, RUNTIME_ROOT, CACHE_SUBDIRS, WALL2100_SPEC, tree_bytes, NVME

INITIAL_ADAPTER_SHA = "4e993ff4f5ca7221e4fcdc61ad302ac644b3a27fe7ef2c25b57516ac21debf83"
INITIAL_ADAPTER_CONFIG_SHA = "62621ecba00f55f36658d3161c3e7ba38a977b3f5362ff51abee96fbc32e8a21"
INITIAL_RESULT_SHA = "c35ddebd558547088b9c84fa3565cc35fc6e569e886ab616b4596139a87314c3"
EXECUTOR_DIGEST = "8fdfcd3e1b8bdfd95530eec09710d1c44efc215885b49c64c1a82f55bbc501a8"
# Full original core identity, not a shortened branch-local cherry-pick id.
CORE_COMMIT = "0808ee8d811b78d1cd34d1da33c30abc705a66d1"
OUTPUTS = {"training": "training_completion_v1", "service": "service_completion_v1"}
CONFIG = {"protocol": VERSION, "actor_protocol": ACTOR_VERSION, "motion_codec": MOTION_CODEC,
    "base_model": BASE, "physical_gpu": 3, "seed": 41, "lora_rank": 16, "lora_alpha": 32,
    "lora_dropout": .05, "learning_rate": 5e-5, "effective_batch_size": 8, "microbatch": 2,
    "max_updates_including_gate": 120, "max_wall_seconds": 2700, "max_artifact_MiB": 384,
    "initial_adapter_sha256": INITIAL_ADAPTER_SHA, "per_update": [4, 2, 2],
    "motion_status_loss_weights": [.5, .5], "sampling": "trajectory_balanced_within_category_seed41",
    "terminal_checkpoint": 120}
SERVICE_CONFIG = {"protocol": VERSION, "actor_protocol": ACTOR_VERSION, "motion_codec": MOTION_CODEC,
    "physical_gpu": 3, "port": 8931, "max_calls": 64, "variants": list(VARIANTS), "max_artifact_MiB": 384}


def validate_config(value, *, service=False):
    if canonical(value) != canonical(SERVICE_CONFIG if service else CONFIG):
        raise ValueError("Exact registered H09AB recipe/service config required")


def clean_source():
    repo = Path(__file__).resolve().parents[2]
    if subprocess.check_output(["git", "-C", str(repo), "status", "--porcelain"], text=True).strip():
        raise ValueError("Immutable clean experiment source required")
    paths = sorted((repo/"src/semantic_robot/v2").glob("*.py")) + [repo/"scripts/semantic_robot/run_v2.py"]
    digest = hashlib.sha256(b"".join(path.name.encode()+path.read_bytes() for path in paths)).hexdigest()
    if digest != EXECUTOR_DIGEST: raise ValueError("Original public executor changed")
    return subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()


def initial_checkpoint(training):
    from native_serve import validate_checkpoint
    training = Path(training)
    if training.resolve() != ROOT/"training_v1" or sha(training/"result.json") != INITIAL_RESULT_SHA:
        raise ValueError("Only original reviewed120 result is a warmstart")
    folder, identity, checkpoint = validate_checkpoint(training, MOTION_DATA_SHA, protocol=ACTOR_VERSION)
    if checkpoint["adapter_sha256"] != INITIAL_ADAPTER_SHA: raise ValueError("Wrong parent adapter")
    cfg = json.loads((folder/"adapter_config.json").read_text())
    # PEFT serialized this actual checkpoint's targets as suffixes, not full
    # module paths. Pin its reviewed bytes; verify actual trainable names after
    # loading, instead of pretending the on-disk list contains full paths.
    if (sha(folder/"adapter_config.json") != INITIAL_ADAPTER_CONFIG_SHA or
            cfg.get("r") != 16 or cfg.get("lora_alpha") != 32 or cfg.get("lora_dropout") != .05 or
            cfg.get("bias") != "none" or cfg.get("task_type") != "CAUSAL_LM" or
            not cfg.get("target_modules")):
        raise ValueError("Parent checkpoint differs from language-only LoRA recipe")
    if any(identity.get(k) != v for k, v in base_identity().items()):
        raise ValueError("Parent base/config/tokenizer changed")
    return folder, identity


def require_authorization(auth, *, stage, code, config_sha, output):
    if stage not in OUTPUTS: raise ValueError("Unknown stage")
    expected = {"protocol": VERSION, "code_commit": code, "config_sha256": config_sha,
        "dataset_sha256": DATA_SHA, "states_sha256": STATES_SHA, "data_review_sha256": REVIEW_SHA,
        "initial_adapter_sha256": INITIAL_ADAPTER_SHA, "initial_result_sha256": INITIAL_RESULT_SHA,
        "physical_gpu": 3, "reviewer": "Codex-parent", "output": str(ROOT/OUTPUTS[stage]),
        "executor_digest": EXECUTOR_DIGEST, "carry_duration_core_commit": CORE_COMMIT,
        **execution_metadata(CARRY_PROFILE),
        "authorize_training": stage == "training", "authorize_service": stage == "service",
        "authorize_physical": False, "storage": WALL2100_SPEC}
    for key, value in expected.items():
        if key not in auth or canonical(auth[key]) != canonical(value): raise ValueError("Exact authorization mismatch: "+key)
    if Path(output).resolve() != ROOT/OUTPUTS[stage] or Path(output).exists() or Path(output).is_symlink():
        raise ValueError("Unique independent output required")
    review_path = Path(auth["parent_release"]); check_sha(auth["parent_release_sha256"])
    if sha(review_path) != auth["parent_release_sha256"]: raise ValueError("Independent release bytes changed")
    review = json.loads(review_path.read_text())
    for key in ("code_commit", "config_sha256", "dataset_sha256", "states_sha256", "initial_adapter_sha256", "reviewer", "authorize_"+stage):
        if canonical(review.get(key)) != canonical(expected[key]): raise ValueError("Independent exact-source release mismatch: "+key)
    if stage == "service":
        check_sha(auth.get("training_result_sha256"))
        if review.get("training_result_sha256") != auth["training_result_sha256"]:
            raise ValueError("Parent did not approve this actual trained checkpoint result")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "3": raise ValueError("Only handed-over physical GPU3")


class CompletionStorage(RuntimeStorage):
    """New model-only cache dirs, NO additional OG alias and unchanged global caps."""
    def __init__(self, auth, output):
        if canonical(auth.get("storage")) != canonical(WALL2100_SPEC): raise ValueError("Exact inherited storage")
        if Path(output).resolve() not in [ROOT/name for name in OUTPUTS.values()]: raise ValueError("Only two new model paths")
        super().__init__(auth, ROOT/"training_v1")
        self.output = Path(output).resolve(); self.root = RUNTIME_ROOT/self.output.name
        self.expected = runtime_environment(self.output)
    def check(self):
        receipt = super().check()
        size = tree_bytes(ROOT, NVME.stat().st_dev)
        if size >= 6*1024**3: raise RuntimeError("Whole H09Y result root exceeds6GiB")
        if self.output.exists() and tree_bytes(self.output, NVME.stat().st_dev) >= 384*1024**2:
            raise RuntimeError("Completion artifact cap384MiB")
        return {**receipt, "completion_model_only": True, "result_root_bytes": size, "new_og_alias": False}
    def create(self):
        receipt = self.check()
        for key in CACHE_SUBDIRS: Path(self.expected[key]).mkdir(parents=True, exist_ok=True)
        self.check(); return receipt


def completed_checkpoint(training, *, code, config_sha):
    training = Path(training); result = json.loads((training/"result.json").read_text())
    identity = json.loads((training/"identity.json").read_text())
    if (training.resolve() != ROOT/OUTPUTS["training"] or result.get("status") != "COMPLETE" or
            type(result.get("optimizer_updates")) is not int or result["optimizer_updates"] != 120 or
            result.get("identity_sha256") != sha(training/"identity.json") or
            identity.get("protocol") != VERSION or identity.get("code_commit") != code or
            identity.get("config_sha256") != config_sha or identity.get("dataset_sha256") != DATA_SHA or
            identity.get("states_sha256") != STATES_SHA or identity.get("review_sha256") != REVIEW_SHA or
            result.get("protocol") != VERSION or result.get("dataset_sha256") != DATA_SHA or
            result.get("initial_adapter_sha256") != INITIAL_ADAPTER_SHA or
            identity.get("initial_adapter_sha256") != INITIAL_ADAPTER_SHA or identity.get("warmstart_loaded") is not True):
        raise ValueError("Complete same-source warmstarted120 checkpoint required")
    validate_config(identity["config"])
    restore = json.loads((training/"restore_gate.json").read_text())
    native = json.loads((training/"native_loss_gate.json").read_text())
    if (restore.get("passed") is not True or type(restore.get("updates_included_in_total")) is not int or
            restore["updates_included_in_total"] != 2 or native.get("passed") is not True or
            native.get("all73_response_only_EOS") is not True):
        raise ValueError("Numeric save/reload gate missing")
    if [c["step"] for c in result["checkpoints"]] != [2, 120]: raise ValueError("No extra checkpoint selection")
    checkpoint = result["checkpoints"][-1]; folder = training/"adapter_0120"
    if (checkpoint["step"] != 120 or checkpoint["adapter_sha256"] != sha(folder/"adapter_model.safetensors") or
            checkpoint["adapter_config_sha256"] != sha(folder/"adapter_config.json")):
        raise ValueError("Terminal checkpoint bytes changed")
    return folder, identity, checkpoint
