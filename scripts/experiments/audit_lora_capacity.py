"""Read-only actual LoRA coverage/spectrum audit; no model or new training.

QR on the two skinny factors obtains the update singular values without ever
allocating a dense out_features x in_features update. A full rank spectrum is
not evidence of underfitting, nor a recommendation to increase rank by itself.
"""
from collections import defaultdict
import json
import math
import os
from pathlib import Path
import subprocess

from probe_ar_execution_codec import publish, sha
from train_fm_method_probe import BASE, PARENT, PARENT_SHA

REPO = Path(__file__).resolve().parents[2]
OUTPUT = BASE / "lora_capacity_a4_fm500_v1"
FM = BASE / "fm_action_control_v3/formal/checkpoints/step_500.pt"
FM_SHA = "efce4dfe232f85ac18f7fca66b562360ba23d839f3f74748f658b5911b5c33f9"


def low_rank_spectrum(a, b, *, scaling):
    import torch
    if (a.device.type != "cpu" or b.device.type != "cpu" or a.ndim != 2 or b.ndim != 2
            or a.shape[0] != b.shape[1] or not 1 <= a.shape[0] <= min(a.shape[1], b.shape[0])
            or not a.is_floating_point() or not b.is_floating_point()
            or not torch.isfinite(a).all() or not torch.isfinite(b).all()
            or not math.isfinite(scaling) or scaling <= 0):
        raise ValueError("Need finite compatible CPU low-rank factors and positive scaling")
    # B A = Q_B (R_B R_A^T) Q_A^T. The tall Q factors are orthonormal,
    # so its nonzero singular values are those of the rank x rank center.
    _, ra = torch.linalg.qr(a.double().T, mode="reduced")
    _, rb = torch.linalg.qr(b.double(), mode="reduced")
    singular = torch.linalg.svdvals(rb @ ra.T) * scaling
    energy = singular.square()
    total = float(energy.sum())
    if total:
        probabilities = energy / total
        rank95 = int(torch.searchsorted(probabilities.cumsum(0), .95)) + 1
        participation = float(1 / probabilities.square().sum())
        leading_fraction = float(probabilities[0])
    else:
        rank95, participation, leading_fraction = 0, 0., None
    return dict(rank_limit=a.shape[0], a_shape=list(a.shape), b_shape=list(b.shape), scaling=scaling,
        singular_values=singular.tolist(), update_frobenius_norm=total**.5,
        energy_rank95=rank95, energy_participation_rank=participation,
        leading_energy_fraction=leading_fraction, no_dense_update_materialized=True)


def adapter_pairs(state):
    suffix = ".lora_A.skill_fm.weight"
    modules = sorted(key.removesuffix(suffix) for key in state if key.endswith(suffix))
    expected_keys = {module + f".lora_{side}.skill_fm.weight" for module in modules for side in ("A", "B")}
    if not modules or {key for key in state if "lora_" in key} != expected_keys:
        raise RuntimeError("Missing, partial or different-named LoRA adapter")
    return [(module, state[module + suffix], state[module + ".lora_B.skill_fm.weight"]) for module in modules]


def scope_inventory(state, modules):
    """Only weight metadata is read here; frozen dense weights are not copied."""
    observed = []
    for name, tensor in state.items():
        if not name.startswith("model.vlm.") or not name.endswith(".weight") or "lora_" in name or tensor.ndim != 2:
            continue
        module = name.removesuffix(".weight").removesuffix(".base_layer")
        observed.append(dict(module=module, shape=list(tensor.shape), dtype=str(tensor.dtype),
                             has_lora=module in modules))
    if {row["module"] for row in observed if row["has_lora"]} != set(modules):
        raise RuntimeError("Every adapter must map to a real base two-dimensional weight")
    grouped = defaultdict(lambda: dict(total=0, with_lora=0))
    for row in observed:
        group = grouped[row["module"].rsplit(".", 1)[-1]]
        group["total"] += 1
        group["with_lora"] += int(row["has_lora"])
    return dict(base_2d_weights=observed, by_module_leaf=dict(sorted(grouped.items())))


def main():
    if OUTPUT.exists() or subprocess.check_output(["git", "-C", str(REPO), "status", "--porcelain"], text=True).strip():
        raise RuntimeError("Only a clean pinned source and a new one-time output")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import torch
    from omegaconf import OmegaConf

    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    if torch.cuda.is_available():
        raise RuntimeError("This audit is CPU-only")
    root = FM.parents[2]
    spec = json.loads((root / "method_spec.json").read_text())
    inspection = json.loads((root / "formal/checkpoint_inspection.json").read_text())
    status = json.loads((root / "status.json").read_text())
    if (status.get("state") != "complete" or status.get("verified_updates") != 500
            or not inspection.get("passed") or inspection.get("checkpoint_sha256") != FM_SHA
            or inspection.get("actual_adam_states") != 504 or not inspection.get("frozen_unchanged")
            or not inspection.get("full_model_optimizer_rng_roundtrip") or spec.get("route") != "fm"):
        raise RuntimeError("Require the already completed original M04 FM control, not a new candidate")
    # No resolvers/model construction needed to read this concrete saved node.
    config_path = root / "formal/config.yaml"
    cfg = OmegaConf.load(config_path).model.model_arch.low_vlm_lora
    config = OmegaConf.to_container(cfg, resolve=False)
    if config.get("r") != 8 or config.get("alpha") != 16 or config.get("dropout") != .05:
        raise RuntimeError("Actual saved LoRA recipe changed")
    parent_config_path = PARENT.parents[1] / "config.yaml"
    parent_config_sha = "4d45b4c2ae8872b8e4a88916c4143d922b8cf0e76eedaa6a4116d473d9ef2483"
    if sha(parent_config_path) != parent_config_sha:
        raise RuntimeError("Original A4 saved config changed")
    parent_lora = OmegaConf.to_container(OmegaConf.load(parent_config_path).model.model_arch.low_vlm_lora,
                                        resolve=False)
    if parent_lora != config:
        raise RuntimeError("Parent/control LoRA scaling or target configuration differs")
    receipts = []
    for path, digest, step in ((PARENT, PARENT_SHA, 2500), (FM, FM_SHA, 500)):
        if sha(path) != digest:
            raise RuntimeError("Declared saved checkpoint identity changed")
        payload = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
        if payload.get("step") != step:
            raise RuntimeError("Wrong parent or completed short-control step")
        state = payload["model_state_dict"]
        pairs = adapter_pairs(state)
        if len(pairs) != 96 or len(state) != 1138 or any(a.shape[0] != 8 or b.shape[1] != 8 for _, a, b in pairs):
            raise RuntimeError("Actual original192 rank8 LoRA coverage changed")
        rows = [dict(module=module, a_sha256=tensor_sha(a), b_sha256=tensor_sha(b),
                     **low_rank_spectrum(a, b, scaling=config["alpha"] / config["r"]))
                for module, a, b in pairs]
        receipts.append(dict(checkpoint=str(path), checkpoint_sha256=digest, step=step, rows=rows,
                             scope=scope_inventory(state, {module for module, _, _ in pairs})))
        del payload, state, pairs
    if [row["module"] for row in receipts[0]["rows"]] != [row["module"] for row in receipts[1]["rows"]]:
        raise RuntimeError("Parent/control adapter modules differ")
    comparison = [dict(module=a["module"], a_changed=a["a_sha256"] != b["a_sha256"],
        b_changed=a["b_sha256"] != b["b_sha256"], parent_rank95=a["energy_rank95"], control_rank95=b["energy_rank95"],
        parent_norm=a["update_frobenius_norm"], control_norm=b["update_frobenius_norm"])
        for a, b in zip(receipts[0]["rows"], receipts[1]["rows"])]
    OUTPUT.mkdir(exist_ok=False)
    publish(OUTPUT / "result.json", dict(complete=True, config=config, config_sha256=sha(config_path),
        parent_config_sha256=parent_config_sha,
        source_commit=subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip(),
        entry_sha256=sha(Path(__file__)), receipts=receipts, comparison=comparison,
        checkpoints_read=2, factor_pairs_analyzed=192, policy_forwards=0, backwards=0, optimizer_updates=0,
        training_samples_read=0, simulator_controls=0, no_dense_update_materialized=True,
        limitations=["Matrix spectra do not establish underfitting, task-gradient conflict or higher-rank benefit.",
            "No rank expansion, new random adapter, target-scope change, or training was performed.",
            "Layer-module coverage is read from actual saved tensor names, not assumed from target-name defaults."]))
    print(json.dumps(dict(complete=True, result_sha256=sha(OUTPUT / "result.json"), factor_pairs=192)), flush=True)


def tensor_sha(value):
    import hashlib
    import torch
    raw = value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


if __name__ == "__main__":
    main()
