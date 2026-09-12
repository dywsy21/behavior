"""Real-config/processor/codec/vision preflight; no VLM training or baseline."""
from pathlib import Path
from copy import deepcopy
from types import SimpleNamespace
import argparse
import json
import rootutils

rootutils.setup_root(__file__, indicator=".python-version", pythonpath=True)

import numpy as np
import torch
import pyarrow.parquet as pq
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from g05.utils.config.config_resolvers import register_default_resolvers
from g05.utils.data.processor_utils import build_processors
from g05.utils.data.normalizer import load_dataset_stats_from_json
from g05.models.g05.inferencer import PolicyInferencer
from g05.models.g05.g05_policy import G05Policy
from g05.tokenizer.interface.vq_base import VQActionTokenizer
from g05.utils.memlite_protocol import ActionDecodeError
from scripts.serve_policy_mem import build_obs_dict, _configure_action_execution_start_index, ChunkedPolicyWrapper


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--assets-root", required=True)
    parser.add_argument("--vision", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    register_default_resolvers()
    with initialize_config_dir(config_dir=str(root / "configs"), version_base="1.3"):
        cfg = compose(config_name="train", overrides=["task=r1pro_memlite_ar_v9"])
    for obj, key in ((cfg.model.model_arch, "hf_processor_path"), (cfg.model, "pretrained_ckpt"), (cfg.tokenizer.vq_config, "ckpt_dir")):
        obj[key] = str(Path(args.assets_root) / obj[key])
    processors = build_processors(cfg)
    processors.set_normalizer_from_stats(load_dataset_stats_from_json(cfg.datastatics_path))
    processors.eval()
    p = next(iter(processors.processors.values()))
    p.action_horizon = 32
    _configure_action_execution_start_index(processors, cfg)
    assert p.image_history_mode == "preserve_cameras" and p.num_obs_steps == 6
    assert not cfg.tokenizer.vq_config.dropout_noop_parts
    assert p.samples_builder._high.__class__.__name__ == "MEMLiteCausalHighLevelBuilder"

    dataset = Path("/mnt/sdc1/robodojo/datasets/behavior2026_g05_tasks_0_4")
    ep = pq.read_table(dataset / "meta/episodes/chunk-000/file-000.parquet").slice(0, 1).to_pylist()[0]
    table = pq.read_table(dataset / f'data/chunk-{ep["data/chunk_index"]:03d}/file-{ep["data/file_index"]:03d}.parquet',
                          columns=["episode_index", "frame_index", "action", "observation.state"])
    import pyarrow.compute as pc
    table = table.filter(pc.equal(table["episode_index"], 0))
    frame_rows = table.to_pylist()
    raw_actions = np.array([r["action"] for r in frame_rows], dtype=np.float32)
    raw_states = np.array([r["observation.state"] for r in frame_rows], dtype=np.float32)
    codec = VQActionTokenizer(cfg.tokenizer.vq_config, action_dim=27, device="cpu")
    codec_records = []
    frames = sorted(set([0, 16, 128, 256, 512, 1024] +
                        [int(i) for i in np.flatnonzero(np.any(np.diff(raw_actions[:, [14, 22]], axis=0) != 0, axis=1))[:24]]))
    for frame in frames:
        if frame + 32 > len(raw_actions):
            continue
        raw_obs = {"images": {meta["key"]: np.zeros((3, 256, 256), dtype=np.uint8) for meta in p.shape_meta["images"]},
                   "state": {meta["key"]: raw_states[frame, meta["start_index"]:meta["start_index"] + meta["raw_shape"]] for meta in p.shape_meta["state"]},
                   "embodiment_type": "galaxea_r1pro", "task": "Turn on the radio.", "frequency": 30.0}
        data = build_obs_dict(raw_obs, processors, mem_state=SimpleNamespace(memory_text="Task=0; Completed=none.", intent_text="Reach the radio", status="CONTINUE"))
        for meta in p.shape_meta["action"]:
            data["action"][meta["key"]] = torch.from_numpy(raw_actions[frame:frame + 32, meta["start_index"]:meta["start_index"] + meta["raw_shape"]].copy())
        data["action_is_pad"].zero_()
        data["memlite_branch"] = "low"
        for camera_index, (key, values) in enumerate(data["images"].items()):
            data["images"][key] = torch.stack([torch.full_like(values[0], 20 * camera_index + t) for t in range(6)])
        sample = p.preprocess(deepcopy(data))
        pixels = sample["pixel_values"]
        assert all(value.shape[:2] == (6, 3) for value in pixels.values())
        encoded = codec._encode_action_indices(sample["action"].unsqueeze(0), encode_kwargs={"action_dim_is_pad": sample["action_dim_is_pad"].unsqueeze(0)})
        decoded = codec.decode_token_ids_to_actions(encoded[0], time_horizon=32, action_dim=27,
                                                    decode_kwargs={"is_action_token_space": True})
        assert not decoded.absent_keys
        for index in (9, 19):
            target = sample["action"][:, index]
            torch.testing.assert_close(decoded.action[:, index], torch.where(target > 0, 1., -1.))
        post = PolicyInferencer._postprocess_single(
            {"action": decoded.action.unsqueeze(0), "proprio": sample["proprio"].unsqueeze(0),
             "action_dim_is_pad": sample["action_dim_is_pad"].unsqueeze(0),
             "proprio_dim_is_pad": sample["proprio_dim_is_pad"].unsqueeze(0),
             "selected_action_source": "ar", "ar_absent_keys": [decoded.absent_keys]},
            0, p)
        guard = SimpleNamespace(action_steps=16, _trace=lambda *a, **kw: None)
        ChunkedPolicyWrapper._admit_action(guard, post)
        diagnostics = post.get("_normalization_diagnostics", {})
        codec_records.append({"frame": frame, "raw_grippers_first": raw_actions[frame, [14, 22]].tolist(),
                              "absent_keys": [], "wire_action_shapes": {k: list(v.shape) for k, v in post.items() if not k.startswith("_")},
                              "max_normalized_excess": max((d["max_normalized_excess"] for d in diagnostics.values()), default=0)})

    try:
        codec._decode_action_indices([[1, 2, 3]], time_horizon=32, action_dim=27)
    except ActionDecodeError:
        pass
    else:
        raise AssertionError("Markerless action must not decode to a synthetic valid zero command")
    inferencer = PolicyInferencer(None, processors, device="cpu")
    high_obs = inferencer._with_memlite_runtime_branch([data], branch="high", memory_texts=["Task=0; Completed=none."])
    high, _ = inferencer._prepare_branch_batch(high_obs)
    high_sample = high[0].sample["samples"]
    assert high_sample["previous_intent"] == "Reach the radio"
    template = G05Policy._memlite_template(SimpleNamespace(), high_sample, "high")
    assert "<previous_intent_text_!>" in template
    result = {"config": "r1pro_memlite_ar_v9", "pixel_shapes": {k: list(v.shape) for k, v in pixels.items()},
              "explicit_gripper_codec_windows": codec_records, "causal_runtime_prompt": True}
    if args.vision:
        from g05.models.g05.qwen35.vision import Qwen3_5VisionModel
        from g05.models.g05.g05_model_qwen35 import G05ModelQwen35
        vision = Qwen3_5VisionModel(cfg.model.model_arch.vision)
        checkpoint = torch.load(cfg.model.pretrained_ckpt, map_location="cpu", mmap=True, weights_only=False)
        weights = checkpoint.get("model_state_dict", checkpoint.get("model", checkpoint.get("state_dict", checkpoint)))
        prefix = "model.vision_tower."
        selected = {key[len(prefix):]: value for key, value in weights.items() if key.startswith(prefix)}
        if not selected:
            raise ValueError(f"Cannot identify base vision weights; top-level keys: {list(checkpoint)[:8]}")
        vision.load_state_dict(selected, strict=True)
        facade = torch.nn.Module()
        facade.cfg = cfg.model.model_arch
        facade.vision_tower = vision.to("cuda").eval()
        facade.eval()
        pixel_batch = {key: value.unsqueeze(0).cuda() for key, value in pixels.items()}
        changed = {key: value.clone() for key, value in pixel_batch.items()}
        for value in changed.values():
            value[:, :-1] = torch.randn_like(value[:, :-1])
        calls = []
        hook = vision.register_forward_pre_hook(lambda module, positional, kw: calls.append(dict(num_frames=kw["num_frames"], bsz=kw["bsz"])), with_kwargs=True)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            first = G05ModelQwen35._forward_vision(facade, pixel_batch)
            second = G05ModelQwen35._forward_vision(facade, changed)
        hook.remove()
        assert all(call["num_frames"] == 6 for call in calls)
        assert first.shape == (1, 3 * 64, 2048), first.shape
        difference = float((first - second).abs().max())
        assert difference > 1e-5, "Vision ignored history while current frames were unchanged"
        result["pretrained_vision"] = {"calls": calls, "output_shape": list(first.shape), "history_change_max_effect": difference}
    print("MEMLITE_V9_PREFLIGHT=" + json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
