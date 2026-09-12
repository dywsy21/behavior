# MEM-Lite Skill-FM early independent review

**Review type:** read-only, pre-integration and pre-training.  This is a narrow structural review of the experimental low-level controller.  It does **not** approve a GPU run, a checkpoint, an end-to-end rollout, or any success-rate claim.

## Source lock and reproducible checks

Reviewed in the isolated model tree:

`/mnt/sdc1/robodojo/behavior_dev/GalaxeaVLA_memlite_coordination_dev_20260908/model`

| File | SHA-256 at review | Scope |
|---|---|---|
| `src/g05/models/g05/g05_policy_memlite_skill_fm.py` | `f1c560efb31918a23f8d331edc421b3b86f112f19a2df4ad7a55e37c94c8300c` | continuous-FM policy/stage gate |
| `src/g05/models/g05/helpers/vlm_lora.py` | `aecfbe2165d5d7e9011f95579a622abdc5f19b0998b99e0bbca558398610680a` | PEFT lifecycle |
| `src/g05/data_processor/processor/samples_builder.py` | `d36fabb7b832c44f81fb55f8dc7b951d12ca26515e67f56fb00140b78f6a832c` | action/history template |
| `configs/task/r1pro_memlite_skill_fm_stage_a.yaml` | `931d142eac3a4f1bd6a0c8afbbcac02d6e21dad6540d3851ab92e571b5ebfd28` | action-expert-only stage |
| `configs/task/r1pro_memlite_skill_fm_stage_b.yaml` | `37c267aab5f5760fb24e34e89b66fa8e7e0b41c029794be1154e7309089ca80f` | LoRA/history stage |
| `src/g05/utils/memlite_skill_protocol.py` | `15da12ce9096500873a6b1dfbbb30e690b9ae08c33e69ac8b9750daae954677f` | **stale protocol copy** |

With `PYTHONDONTWRITEBYTECODE=1`, the absolute shared interpreter and `pytest -p no:cacheprovider`, `tests/test_memlite_skill_models.py` passed **10/10**.  These are CPU contract tests with a tiny FM model, not a full G0.5 weight or GPU-gradient test.

## What the static/CPU review establishes

1. The proposed low policy rejects AR routes (`discrete_action=false`, `continuous_action=true`, `predict_cot=false`) and refuses the old MEM-Lite CE-only branch.  Its `forward_train` validates a low v6 bundle then calls the inherited continuous path.  That path reaches `G05ModelQwen35.forward` and `FMHelper.train_step` with CE skipped, and the returned scalar is the real FM loss rather than an action-token loss.
2. Stage A is intended to update only `model.action_expert.*`; stage B adds only VLM parameter names containing `lora_`.  `FMHelper` detaches the VLM cache in stage A (`joint_training=false`) but preserves it in the stage-B config (`joint_training=true`).  The tiny CPU tests demonstrate action-expert gradients in A and nonzero cache gradient in B.
3. The FM loss masks padded time steps and padded dimensions.  The inherited G05 config sets `padding_action_weight=0.0`; the R1Pro base config supplies the 27-D merged action layout.  The policy verifies `[batch,horizon,action_dim]` before invoking FM.
4. The low template contains no `<action_action>` token, and `SkillFMActionBuilder` rejects terminal, unknown-skill and `low_action_supervision_mask=false` rows.  Its six-frame image-marker order is camera-major (`head_t*`, then `left_t*`, then `right_t*`), matching the Qwen35 pixel-dict concatenation order rather than the generic frame-major order.

## P1 blockers — do not start training

1. **The stage lifecycle is not connected to the real training entrypoint.**  At this source lock, `scripts/finetune.py` constructs/loads the model and proceeds toward device/DDP/optimizer construction without calling the policy's `remap_checkpoint_state_dict`, `post_checkpoint_load`, or `configure_coordination_trainability`.  Yet the policy requires the stage gate at forward time, and stage B requires LoRA injection before it.  Stage-A configuration even sets `freeze_parameter_prefixes: []`, so without that hook parameters are initially wrong.  Required order: checkpoint key remap → base checkpoint load → LoRA injection and restore → stage gate/parameter receipt → device/DDP/optimizer.  A missing adapter on a stage-B resume must fail rather than create random trainable LoRA silently.
2. **The model consumer is stale relative to the reviewed data contract.**  The final independently rechecked protocol is SHA `48d6f6d…`, but this tree imports the old `15da…` schema.  Until it syncs the final protocol and proves consumer projections, no sample is guaranteed to use the intended current contract.
3. **At this lock, high-prefix semantics are not yet faithfully consumed.**  `PlannerOutcomeBuilder` uses generic `command_text` rather than a tested `task_name` field, and constructs its EOC-after parent target from `parent_goal`, not explicit `target_parent_goal`.  This can collapse the task / previous-parent / target-parent distinction when those values diverge.  The model owner is preparing a separate fix; it must be independently rechecked after publication.

## P2 / required next evidence

- `VLMloraConfig.from_mapping` uses Python `bool(...)` for `enabled`; a malformed serialized string such as `"false"` would turn it on.  Hydra configs should provide native booleans, but the loader configuration validator should reject non-booleans.
- The PEFT helper's partial-load coverage test is sound only after the loader passes the original checkpoint adapter keys through the required lifecycle.  Add an explicit base-init receipt and a resume receipt, distinguishing the allowed zero-adapter stage-A-to-B initialization from a forbidden missing-adapter stage-B resume.
- After the loader fix, run an actual small, full-weight gradient audit: per-group gradients, optimizer membership, before/after parameter hashes, exact action/padding masks and camera history.  CPU synthetic cache gradients alone cannot prove G0.5 checkpoint keys, optimizer construction, DDP or memory behavior.

## Re-review gate

The next review must lock the updated model, loader and final protocol hashes; run the lifecycle/order and tokenizer-prefix tests; and inspect an actual small FM update receipt before any 5,000-step run.  Training may proceed only after all P1 blockers are closed and E2 data approval remains satisfied.

## Targeted model recheck: stable candidate published after the initial review

The following **new** source lock was independently read and tested in the same isolated model tree.  It supersedes the initial model-only hashes above; the history is retained so that the original P1s remain traceable.

| File | SHA-256 |
|---|---|
| `g05_policy_memlite_skill_fm.py` | `35eb2d8deb6f586440aba308b8dc709e723010b1b8ec413fcca4dc536729dd90` |
| `g05_policy_memlite_planner_outcome.py` | `90a5e60929465fa467d77fd796d4aa99bc61b0d211e5c79b389a8c523f6269d6` |
| `helpers/vlm_lora.py` | `90c4caa4af5e2882ef896d62f9f8020a15a871fa10a62f669a5337f7b40ebede` |
| `data_processor/processor/samples_builder.py` | `efe7b3258399cfe803d0c126ac5390d91b9261ae2b56ac3a6c0bf19e10f06dcb` |
| copied `utils/memlite_skill_protocol.py` | `48d6f6d263b2891fdbf6e837a86f2a64ee79effb8a1268e41bfd6368e5ed12a0` |
| `tests/test_memlite_skill_models.py` | `9f9a6c80d27653fa3a8e7447d4724b420a0bcf5627c10095d1dbcb0936680bd8` |

`PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$MODEL/src:$MODEL" ... pytest -q -p no:cacheprovider tests/test_memlite_skill_models.py` independently passed **13/13**.

### Closed model-side findings

- The final protocol is now copied at the exact reviewed hash.  Low samples project only semantic active-skill JSON/text, reject audit `active_skills_json`, and the policy rejects any unexpected audit field in its model sample.
- The high builder now requires and explicitly places `task_name` in the EOC-prefix.  It fails closed when the source task instruction differs from the sidecar task name.  It obtains the EOC-after current parent from `planner_target_projection(...)["target_parent_goal"]`, while the EOC-prefix accepts only `previous_parent_goal`.  The new deliberately divergent fixture and token-prefix test reject a target-parent/outcome leak.
- LoRA `enabled` is now strict boolean; its CPU test rejects `"false"`.  The helper also reports `base_init` versus `resume` adapter load mode and rejects an adapter namespace mismatch/partial restore.
- The strict stage profile (`low_ae` or `low_ae_lora_history`) is now checked in the low policy in addition to its action-expert/LoRA group check.  Existing tiny-FM tests still verify Stage-A detach and Stage-B joint-cache gradients.  The real inherited forward continues to route to FM loss, not action-token CE.

### Remaining P1 and scope limit

The **training-entrypoint lifecycle P1 remains open** until the independent training owner publishes and tests its hook: remap before base load, post-load LoRA injection/restore, then stage gate before device/DDP/optimizer, with an explicit distinction between Stage-A-to-B base initialization and a missing-adapter Stage-B resume.  No GPU full-weight one-step gradient audit was run in this review, and E2's human data approval has not happened.  Therefore this recheck closes neither the loader gate nor permission to start the 5,000-step training run.
