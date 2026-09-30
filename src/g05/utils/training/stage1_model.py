"""Production construction/restoration; never delegate to a benchmark loop."""
from __future__ import annotations
import gc
import json
from pathlib import Path

ASSETS = {
    "high": ("memlite-b-final-20260910", "B-training-config.yaml", "B-dataset-stats.json", "B-final-model.pt",
             "e7cd7bf738eb46901565088f829aa82c5c6ea95f610d4df1499e634959d29b13"),
    "low": ("memlite-a4-20260912", "A4-training-config.yaml", "A4-dataset-stats.json", "step_2500.pt",
            "6186704788c27c9fae3502c884df0e259de5242ee8690fe578dcbc1f2632f269"),
}


def configuration(root, branch, task_names):
    from omegaconf import OmegaConf
    from g05.utils.config.config_resolvers import register_default_resolvers
    register_default_resolvers()
    root = Path(root)
    directory, config_name, stats_name, weight_name, weight_sha = ASSETS[branch]
    assets = root / "models" / directory
    cfg = OmegaConf.load(assets / config_name)
    arch = cfg.model.model_arch
    arch.num_obs_steps = arch.cond_steps = 1
    arch.num_input_images = 3
    arch.coordination_train[f"{branch}_observation_steps"] = 1
    arch.hf_processor_path = str(root / "models/qwen3_5_2b_base_processor")
    cfg.tokenizer.vq_config.ckpt_dir = str(root / "models/action_tokenizer.pt")
    arch.pretrained_model_path = None
    cfg.model.processor.num_obs_steps = cfg.data.obs_size = 1
    if branch == "high":
        arch.planner_outcome.task_parent_format_by_task = {
            task: "Task goal: " + task for task in task_names.values()}
        if (arch.continuous_action or not arch.discrete_action or not arch.predict_cot
                or arch.coordination_train.trainability_profile != "high_planner_only"
                or arch.planner_outcome.memory_update_ce_weight != .25
                or arch.planner_outcome.outcome_loss_weight != 0):
            raise ValueError("High planner-only objective drift")
    elif (not arch.continuous_action or arch.discrete_action or arch.predict_cot
          or arch.fm.padding_action_weight != 0 or arch.fm.zero_pad_action_target
          or arch.fm.num_flow_samples != 4):
        raise ValueError("Low real-23-dim, four-noise FM objective drift")
    emb = cfg.data.processors.galaxea_r1pro
    raw_shape = OmegaConf.to_container(emb.shape_meta, resolve=True)
    merged = OmegaConf.merge(emb, cfg.model.processor)
    merged.action_state_merger = cfg.model.processor.action_state_merger
    merged._target_ = "g05.data_processor.processor.memlite_stage1.Stage1Processor"
    merged.samples_builder._target_ = ("g05.data_processor.processor.memlite_stage1." +
        ("PlannerOutcomeBuilder" if branch == "high" else "SkillFMActionBuilder"))
    merged.embodiment_type = "galaxea_r1pro"
    merged.image_history_mode = "preserve_cameras"
    merged.num_obs_steps = 1
    return dict(arch=OmegaConf.to_container(arch, resolve=True),
                processor=OmegaConf.to_container(merged, resolve=True), raw_shape=raw_shape,
                stats_path=str(assets / stats_name), initial_weights=str(assets / weight_name),
                initial_weights_sha256=weight_sha, source_config=str(assets / config_name))


def make_processor(config, training):
    from hydra.utils import instantiate
    from g05.utils.data.normalizer import load_dataset_stats_from_json
    processor = instantiate(config["processor"])
    stats = load_dataset_stats_from_json(config["stats_path"])["galaxea_r1pro"]
    processor.set_normalizer_from_stats(stats)
    processor.set_action_execution_start_index(0)
    processor.train() if training else processor.eval()
    # The original checkpoints' action coordinate systems are retained exactly.
    # Missing stats may not silently fall back to identity in this entry.
    for domain in ("action", "state"):
        # BehaviorPerKeyTransform consumes raw base_qvel/trunk_qpos and creates
        # lower_body before normalization. Validate that *post-transform* set.
        for key in ("left_arm", "left_gripper", "right_arm", "right_gripper", "lower_body"):
            if key not in stats[domain] or key not in processor.normalizer.normalizers[domain]:
                raise ValueError(f"Missing inherited normalizer key: {domain}/{key}")
            if processor.normalizer.normalizers[domain][key].mode == "dummy":
                raise ValueError(f"Unexpected identity normalization: {domain}/{key}")
    return processor


def restore_model(config, branch, state=None):
    import torch
    from hydra.utils import instantiate
    if state is None:
        saved = torch.load(config["initial_weights"], map_location="cpu", mmap=True, weights_only=False)
        state = saved["model_state_dict"]
        expected_step = 1500 if branch == "high" else 2500
        if saved["step"] != expected_step:
            raise ValueError("Wrong initialization lineage")
    expected = 950 if branch == "high" else 1138
    if len(state) != expected:
        raise ValueError("Incomplete checkpoint state")
    model = instantiate(config["arch"])
    if branch == "low":
        mapped = model.remap_checkpoint_state_dict(state)
        model.load_state_dict({k: v for k, v in mapped.items() if "lora_" not in k}, strict=True)
        receipt = model.post_checkpoint_load(state)
        if receipt.get("adapter_load_mode") != "resume" or receipt.get("restored") != 192:
            raise ValueError("Missing trained LoRA tensors")
    else:
        model.load_state_dict(state, strict=True)
    actual = model.state_dict()
    if set(actual) != set(state):
        raise ValueError("Restored state key coverage mismatch")
    for key, value in actual.items():
        reference = state[key]
        if (value.dtype != reference.dtype or value.shape != reference.shape
                or not torch.equal(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8),
                                   reference.detach().cpu().contiguous().reshape(-1).view(torch.uint8))):
            raise ValueError("Non-exact checkpoint restore: " + key)
    receipt = model.configure_coordination_trainability()
    groups = model.coordination_trainable_parameter_groups()
    wanted = {"planner_vlm": 326} if branch == "high" else {"action_expert": 322, "vlm_lora": 192}
    if {k: len(v) for k, v in groups.items()} != wanted:
        raise ValueError("Stage-1 trainability mismatch")
    del actual
    gc.collect()
    return model, dict(exact_model_tensors=expected, trainability=receipt)


def optimizer_groups(model, branch, recipe):
    if branch == "high":
        groups = model.get_optim_param_groups(lr=recipe["learning_rate"],
            weight_decay=recipe["weight_decay"], apply_decay_on_norm_and_bias=False)
    else:
        groups = [dict(params=[p for _, p in values], name=name,
                       lr=recipe["lora_learning_rate"] if name == "vlm_lora" else recipe["learning_rate"],
                       weight_decay=recipe["weight_decay"])
                  for name, values in model.coordination_trainable_parameter_groups().items()]
    ids = [id(p) for group in groups for p in group["params"]]
    expected = {id(p) for p in model.parameters() if p.requires_grad}
    if len(ids) != len(set(ids)) or set(ids) != expected:
        raise ValueError("Optimizer duplicates or omits trainable parameters")
    return groups
