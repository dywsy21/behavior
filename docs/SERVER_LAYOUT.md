# robo服务器文件位置与保留规则

更新：2026-09-13。本文件中的服务器路径属于`ssh robo`，不是本地路径。**源码走GitHub push/pull，数据、权重、环境、完整实验结果不走Git。**

源码整理与GitHub同步已完成；按用户后来确认的安全范围，旧`hy_vla`实验归档已净释放约481.11GiB。下表区分实际运行位置、Git协作入口和冷归档；其他重要目录保留原位。

## 1. 总览

| 类别 | 当前服务器位置 | 用途与处理原则 |
| --- | --- | --- |
| 原主代码工作区 | `/mnt/sdc1/robodojo/GalaxeaVLA` | 用户指定迁移来源；源码已复制到本地，原31个已修改/90个未跟踪路径及`.git`保留。不直接pull覆盖旧改动 |
| 新GitHub协作入口 | `/mnt/sdc1/robodojo/behavior` | 已建立干净clone，核验时与本地/GitHub同为`c3367995234ce51964fff736f2544610c0320419`；后续文档commit会继续前进。不以它冒充最新A3实验快照 |
| 当前训练/推理解释器 | `/mnt/sdc1/robodojo/GalaxeaVLA/.venv/bin/python3.10` | 多个模型服务正在使用；不入Git，不迁移/删除 |
| 历史研究/运行副本 | `/mnt/sdc1/robodojo/behavior_dev` | 独立源码、run、评测、对齐诊断；不是一个可以整体删除的cache |
| 本轮研究根（下文记W） | `/mnt/sdc1/robodojo/behavior_dev/direct_execution_L6rqZ6_20260910` | A/B权重、运行配置、候选源码、视频/物理trace；按run/用途核验后归档，不按目录名删 |
| 五任务转换数据 | `/mnt/sdc1/robodojo/datasets/behavior2026_g05_tasks_0_4` | 当前实际训练数据，44个data parquet；raw action23/state61。不能误把它当临时转换文件删除 |
| train-only归一化统计 | `/mnt/sdc1/robodojo/stats/g05/behavior5_r1pro_trainonly_taskstrat5_stats_v2.json` | 五任务训练/部署共同依赖；SHA `846bcbeac181df5555cb5d40d4183d61743a556df5cf8c8a8fb1bc17e2a40b19` |
| 原始官方演示 | `/mnt/sdc1/xhz/BEHAVIOR2026/2026-challenge-demos` | xhz目录，不在此次robodojo清理范围；数据来源依赖 |
| BEHAVIOR/OmniGibson | `/mnt/sdc1/xhz/BEHAVIOR2026/BEHAVIOR-1K/OmniGibson` | 当前真实仿真框架；不升级或热改来“顺便清理” |
| 仿真解释器 | `/mnt/sdc1/xhz/miniconda3/envs/behavior/bin/python` | 与模型Python不同，当前Python3.11；不在清理范围 |
| 稳定动作/协议adapter | `/mnt/sdc1/robodojo/behavior_dev/GalaxeaVLA_memlite_coordination_dev_20260908/sim_runtime/production_native_oracle_low_v1` | 旧名字不代表闲置，多项正式服务仍依赖 |
| 版本化标签/审核 | `/mnt/sdc1/robodojo/behavior_dev/GalaxeaVLA_memlite_coordination_dev_20260908/data` | sidecar、release/审核清单和相应版本；有些实验另有overlay，实际入口以该run配置为准 |
| 本次归档盘 | `/mnt/tmp1` | 已核实是另一块XFS文件系统，约894GiB总容量，不能容纳2T归档；复制/校验/是否可释放源副本见末节 |

这不是说50任务数据已全部就绪。全任务数据清单、覆盖、切分和规模仍是大训练准备项，不可从五任务目录推断。

## 2. 当前A3与B-final：别拿错权重或源码

以下相对目录均在 **W = `/mnt/sdc1/robodojo/behavior_dev/direct_execution_L6rqZ6_20260910`** 下。

| 用途 | W下目录/文件 | 说明 |
| --- | --- | --- |
| A3最终低层权重 | `formal_a3_episodecoverage_5000_v1/checkpoints/step_5000.pt` | SHA `865193f1c8a257d72ea159bd7a46cebf945b0fd24905110b831f0e8a3438e940`；AE＋VLM LoRA，六帧，FM |
| A3训练原源码 | `a2_lora_history_candidate_v7_samplercoverage` | 名字含A2但实际是A3训练来源；以run receipt/source hash为准 |
| A3推理神经源码 | `a3_history_serving_candidate_v1` | 与训练神经实现核验；不要直接用主仓旧脚本加载代替 |
| A3推理配置/快照 | `a3_serving_composition_v2` | `serving_config.yaml`、`native_snapshot.json`、`source_receipt.json`共同固定来源 |
| B-final高层run | `formal_b_parent_format_1500_v1` | 本阶段1500更新，累计B谱系5000；不能把目录1500读成全部只训练1500 |
| B-final高层checkpoint | `formal_b_parent_format_1500_v1/checkpoints/step_1500.pt` | SHA `d4580d80cdc91a707c233a6c1e625f8fbdeca8896dd38a3d570c6fad340184ef`；UNKNOWN_ONLY，未训物理反馈头 |
| B-final当前serving源码 | `b_parent_format_serving_source_v2` | 实际高层服务源码；和B-final权重/父模型配置一起记录，不凭主仓默认导入替代 |
| 当前修正版完整runtime | `native_a3_aligned_full_runtime_v2` | 修补动作padding mask和执行起点0；不改权重、归一化及物理控制 |
| 当前修正版局部runtime | `native_a3_aligned_prefix_runtime_v2` | 仅用于有明确原演示前缀/固定技能的L1诊断，不能当完整SR |
| 完整runtime修复清单 | `a3_aligned_runtimes_v2_preparation.json` | 绑定两个runtime及动作对齐证据 |
| 旧A3完整runtime | `native_a3_history_runtime_v1` | 保留重现旧结果；含已定位的推理偏差，不作为新实验默认 |

**重要差距：复制`GalaxeaVLA`主工作区到GitHub不等于已将以上最新实验实现整合进去。** 后续按[任务P0-02](TEAM_PLAN.md)做逐模块diff/测试/合并；不要以复制完成冒称代码与A3运行快照一致。

## 3. 结果、视频和原始证据去哪找

### 2026-09-13新增：FM方法与AR重新验证

- **19:32 CoT GPU门完成/下一入口：** `ar_native_subtask_cot_gpu_gate_v1/result.json` SHA `54f83aff390626a0dfc009c5825d6a0769a36d5c3772402229d48477a622a92b`，2672389已退出，两临时更新通过但动作0/5完整组，未发布权重。新独立`git_worktrees/ar_cot_metrics_20260913`固定09ff64a/166 CPU passed，含动作/文本分项CE和两段生成回执、待用AR500只读诊断；尚无新训练/服务，不热改任何仍活跃队列源。

- **19:25原生Subtask-CoT：** 工作树`git_worktrees/ar_native_subtask_cot_v2_20260913`固定5aa3eff；`dual_track_fm_ar_20260913/ar_native_subtask_cot_input_gate_v2/result.json`十行complete，SHA `0d89930ff98ed90144e4aa592d2e0d7c8bc8ef6e3c207f47a610b85098141e6c`，v1检查器padding比较失败保留。`ar_native_subtask_cot_gpu_gate_v1` PID2672389、同级`.launch.log`，只两临时更新/五task自由生成；尚无正式CoT权重或仿真。活跃worktree禁止热改。

- **19:11 AR只读数值诊断完成：** `dual_track_fm_ar_20260913/ar_decode_consistency_a4_v1`，2604744已退出/代码039e268，独立worktree `git_worktrees/ar_decode_consistency_20260913`。46 CPU passed；两原train行/122同历史下一token比较，121 argmax一致、MRoPE与类型mask一致、96 LoRA模块三路径均调用，自由仍漏组，0更新/仿真。`result.json` SHA `161649772d3a5b8a519a6f5916b6dd1201d26df1de3bbca2e341e23f4cbea43b`；`.launch.log`在run同级，不以强制历史诊断冒称actor输出。

- **17:57 AR正式接续：** `ar_a4_fulltrain_v2/smoke/checkpoint_inspection.json`已passed（5更新/192 Adam），临时`smoke/checkpoints/step_5.pt` SHA `9f22d74f9886ee31a4bda9598a25de22e6680e90ce2b1176b5b02f4bf9f9a84c`。现supervisor1940901/formal trainer2316504重新从原A4初始化正式500，日志`formal.log`及`formal/`；不是从smoke续训，不能用旧2307504 PID判断当前任务。后继队列仍等待，171898c活跃worktree不变。

- **17:50最新阶段（覆盖下方启动历史）：** `fm_ae_lr2x_v1`于17:45:40完成500及回读；`formal/checkpoints/step_500.pt` SHA `7f1c9acdfe52d3ffd6e98038c46a6d743a07766b1188e20c7b45262048396753`。最终固定80=0.2011440，见[LR筛选报告](experiments/2026-09-13-fm-lr-screen.md)。`ar_a4_fulltrain_v2`已启动四卡smoke，supervisor1940901/trainer2307504；原生与其后五臂仍等待，不能将下文“0 AR更新等待”的旧快照当作最新阶段。

**17:44后续筛选队列（均已核验真实等待、0GPU/0更新）：** 根为`/mnt/sdc1/robodojo/behavior_dev/dual_track_fm_ar_20260913`；共用固定84110fa的`/mnt/sdc1/robodojo/behavior_dev/git_worktrees/method_screen_queue_20260913`，不可热pull。每臂`launch.json`/`method_spec.json`/`status.json`区分提交、等待与实际smoke/formal；checkpoint只有训练并验收后才存在。

| run子目录 | supervisor PID | 等待的前驱PID | method_spec SHA256 |
| --- | --- | --- | --- |
| `fm_beta_stratified_v1` | 2289674 | 2222863（native-task-AR） | `191a5393dfd57b8e6749bc93e1487328a6d5fed57fad26a76903f8203adde866` |
| `fm_exec_weight2_v1` | 2291489 | 2289674 | `a6b18a6c325b69b2609720ac95bd0b1ef6624eaf98ed4585faa2a80803cb4231` |
| `fm_action_control_v1` | 2294898 | 2291489 | `e16d8670fbf0f70193b670a7fb3d736552ab8d48e06f28b3ca16a6a87cd242eb` |
| `joint_a4_fulltrain_v1` | 2296750 | 2294898 | `2d8595a640def732d4891ed0497654fdcecd58e51be8d31866f80b9229bb6c03` |
| `ki_a4_fulltrain_v1` | 2297891 | 2296750 | `b9a7eec4cabe36688ff7badc6620bde93b16f3c3ba69cb92fe9823f3c4a9ff06` |

- `ar_native_omission_audit_v1/result.json`：只读CPU20目标/20已生成记录审计，SHA `429d8fa2e9d09e63477dd94433b58ca166442c4ca8b07e29ea1f0b456590b09e`。代码861942c的`git_worktrees/ar_native_omission_20260913`已退出，无新策略/仿真或release。

- **17:26接续：** `dual_track_fm_ar_20260913/ar_native_task_fulltrain_v1`已提交，supervisor2222863，spec SHA `2f01682d6912b14cd7c8d4d6371694cd81ea1303756968d3722bda68811c42af`；真实等待A4-AR1940901，0GPU/0训练更新。固定6e2587b的`git_worktrees/ar_native_task_20260913`现在是正式等待/训练活跃源，不得pull。task GPU v2已退出0，result SHA `09d6244d99053c8295fad280267044fabc0b8df8709ee33739a7c8e8efb5577c`；它仅为两临时更新门，没有可部署checkpoint。

- **原生AR新增（17:13）：** 原生base在`/mnt/sdc1/robodojo/checkpoints/G05/g05-base/checkpoints/model_state_dict.pt`，SHA `072211e5b2f5ef036729bae673f3f44da40adbea5c0044af55fe2fb8af654327`，不与A4父权重混用。新`dual_track_fm_ar_20260913/ar_native_input_gate_v1`是CPU20视图；`ar_native_skills_gpu_gate_v2`是已完成的两临时更新（不是正式策略），result SHA `4b527cbabe11f4770a32901f5a2146abe1d4666d83ad4a35a2384bf0fd514d6d`；`ar_native_skills_gpu_gate_v1`保留0更新词表加载失败。`ar_native_task_gpu_gate_v2`为当前task-only门，使用固定6e2587b的`git_worktrees/ar_native_task_20260913`，运行时不可pull。完整native训练尚未启动；原生词表不注册HL_END，A4默认词表不改。
- **16:35最新观测（覆盖下方启动历史）：** `dual_track_fm_ar_20260913/fm_control_v1`已500更新并验收，最终权重`formal/checkpoints/step_500.pt` SHA `def222a6674e6ac92e6ee982c22836b789240f1542c459d5de2111cd646a244e`。`fm_ae_lr2x_v1`是唯一AE LR×2候选，正式1902909/1912524运行；首个`formal/fixed_diagnostic/step_100.json`的原80均值0.1986116943，SHA `dca01b74cc0e9f488274b64b3d39bd02c546cce7cc33668bd14b9843146f1ad6`，不是最终500结果。仍固定下述fb40145 worktree，不能热pull。
- `dual_track_fm_ar_20260913/ar_a4_fulltrain_v2`：AR正式配方的串行任务，supervisor1940901等待上述候选实际完成/验收，当前0 AR更新；`method_spec.json` SHA `27d22490bb4ccaa9319cc37699a7754471dd91ba60d97f6f5995b0881c7cc930`。`status.json`区分waiting/smoke/formal/complete/failed，`smoke/`和`formal/`只有被真实启动后才产生；预定5步保存门→独立500，均用原A4父权重。`ar_a4_fulltrain_v1`是0更新等待API失败证据，不是训练权重。
- AR活跃Git worktree：`/mnt/sdc1/robodojo/behavior_dev/git_worktrees/action_queue_20260913`，固定171898c，等待也属于活跃使用，禁止切换/热改。入口` scripts/experiments/train_action_method_probe.py`；数据/环境仍复用原目录，模型代码仅feature分支，main目前同步文档。
- `dual_track_fm_ar_20260913/ar_reference_metric_gate_v1/result.json`：2次原train同输入参考前向逐位相等、0更新，SHA `7530aa9facead5b5d7b9ac4a7c4fb2f5f6f9e46eae4bdbb8faf72bea94d0d568`；`ar_input_gate_v2`为40视图真实codec检查，`ar_loader_gate_v1`为完整loader输入身份检查，`ar_gpu_gate_v2`/`ki_gpu_gate_v1`/`joint_gpu_gate_v1`为两临时更新工程门。以上不是已训练的AR/KI权重或成功率结果。

- `/mnt/sdc1/robodojo/behavior_dev/dual_track_fm_ar_20260913`：双路线小实验根。`ar_codec_gate_v1`/`v2`为失败证据，`ar_codec_gate_v3/result.json`为十条原train的CPU编码往返结果；不是AR策略或成功率。
- `fm_control_v1`：14:00编排启动，supervisor1499025、固定Git fb40145；14:14已通过GPU与四卡5步保存回读门，正式torchrun1508221开始从A4初始化。`method_spec.json`绑定扩展源码/父A4/预算，`status.json`为阶段，`gate/`与`smoke/`是工程门，`formal/`才是最多500次真实五任务更新。启动不是完成，须读阶段与checkpoint验收。
- Git固定worktree `/mnt/sdc1/robodojo/behavior_dev/git_worktrees/dual_track_fm_ar_20260913`；方法代码在独立feature，main仅同步说明。运行中不得pull，复用只读A3源码和A4旧编排的SHA明确记录；没有复制旧权重/数据，其他服务保留。

### 2026-09-12晚新增：A4过夜低层训练

- 编排/结果根：`/mnt/sdc1/robodojo/behavior_dev/overnight_a4_20260912`；`launch.json`记录准确commit/父权重/源码SHA，`status.json`给出真实阶段，不能只看PID判定成功。
- Git固定配方：`/mnt/sdc1/robodojo/behavior_dev/git_worktrees/a4_overnight_20260912`，commit `e463932740cbb3977a2b975be824c21d9dc96f45`；**不能在活跃副本pull**。它编排上述原A3训练快照，不代表主仓模型已完成P0-02整合。
- `gate/`为两次临时优化验收、不保存训练权重；`smoke/`为四卡5步验收，`smoke_checkpoint_inspection.json`回读验证后才放行正式阶段；`formal/`才是从原A3-5000初始化的新2500更新阶段，正式初始模型不来自smoke。
- 新正式权重：该根的`formal/checkpoints/step_N.pt`，每500更新及最终保存；固定80窗口结果在`formal/fixed_diagnostic/step_N.json`。2026-09-13 07:39完成，`step_2500.pt` SHA `6186704788c27c9fae3502c884df0e259de5242ee8690fe578dcbc1f2632f269`；`formal_checkpoint_inspection.json`通过完整模型/Adam/RNG/冻结参数验收，尚无A4成功率结论。
- 正式训练无墙钟截止（用户随后明确取消8小时限制）；有限2500步、不自动重试/追加训练。磁盘保留120GiB安全余量，不清理旧checkpoint；异常退出保留已有完整保存。恢复需绑定同run身份及原配置，不能重新执行`start`冒充断点恢复。

### 原有A/B评测和诊断

2026-09-13新结果根（不在W内）：

- `/mnt/sdc1/robodojo/behavior_dev/a4_radio_full_20260913_v1`：A4＋B-final自主完整收音机评测已完成，public_test301/302/303各3224控制，成功0/3；不是有前缀的L1。Git worktree `/mnt/sdc1/robodojo/behavior_dev/git_worktrees/a4_radio_full_20260913`实际固定be23b06；编排1479480、专用服务8783/8784和三个仿真均已退出，其他服务保留。`manifest.json`固定条件，`instance_301/`等各保留结果/视频，`summary.json`完整写出。三段视频及本人复核抽帧/小摘要已在本地`/home/wsy/behavior/artifacts/a4_radio_full_20260913`；完整结论见[评测报告](experiments/2026-09-13-a4-radio-full-eval.md)，不要把这些唯一证据当cache删除。

- `/mnt/sdc1/robodojo/behavior_dev/a4_paired_actions_20260913_v1`：170次A3/A4原train同输入FM对照已complete，`result.json`为摘要，`A3/`和`A4/`保留预测/完整恢复证据；没有physics或SR。
- `/mnt/sdc1/robodojo/behavior_dev/a4_radio_e121_l1_20260913_v1`：A4局部GRASP已完成，448原前缀＋464模型控制后满足指定对象稳定抓取；`actual_analysis.json`和[本轮报告](experiments/2026-09-13-a4-training-effectiveness.md)记录审核/限制，不是完整SR。`service/`为已退出的GPU0/8782临时服务证据，`actual_rollout/`为真实控制/物理trace/视频；supervisor已关闭自己创建的服务。Git worktree `/mnt/sdc1/robodojo/behavior_dev/git_worktrees/a4_prefix_20260913`在执行时固定c7fb287，确认全部进程退出后才更新至a3ce491运行只读后处理；不热pull。
- 本地A4视频及抽帧/小摘要：`/home/wsy/behavior/artifacts/a4_effectiveness_20260913`。视频源/本地SHA一致；不入Git，不视为可随意删的cache。

| 结果类别 | W下路径 | 结论/边界 |
| --- | --- | --- |
| 旧A3完整五任务 | `native_a3_final_development_pilot_v1/task_0` … `task_4` | 各有`result.json`、`rollout.mp4`与记录；完整0/5，重复开发实例 |
| 修正版完整五任务 | `native_a3_aligned_development_pilot_v3` | `task_N/`放结果/视频，序列receipt记录完成情况；2026-09-13核对task0–4均完整结束，官方0/5，属于A3+B-final而非A4 |
| 修正版完整评测清单 | `native_a3_aligned_five_task_development_v3.json` | 同A3/B、public_test301、env0/policy17；SHA `2722f9404b046ff69b443774d65fe1cb7b0bf4c25369d67b988cb7889924d730` |
| 旧A3固定GRASP | `a3_prefix_radio_e121_l1_v1/actual_rollout` | 448原前缀＋1280模型控制，无稳定抓取 |
| 修正版固定GRASP | `a3_aligned_radio_e121_l1_v2/actual_rollout` | 同权重/起点/条件仍未稳定抓取；`collection_result.json`不是完整任务SR |
| 修正版固定GRASP审计 | `a3_aligned_radio_e121_l1_v2/actual_analysis_v2.json` | 80段实际历史/动作/物理记录核对；80 IN_PROGRESS |
| 504次推理对照 | `a3_input_route_factorial_v2` | `result.json`及`execution_alignment_analysis_v1.json`；不再把mask/时间混杂误报为纯视觉域差异 |
| 42原训练锚点 | `a3_radio_e121_matched_original_inputs_v2` | 原radio121/138输入与目标逐值校验；大CPU缓存不进Git |
| 原radio真实回放 | `demo_radio_e121_alignment_v2_persist` | 真实原动作、观察、视频，非模型成功 |
| 第二原radio回放 | `demo_radio_e91_feedback_trace_v5` | 已核实精确目录名；原始成功演示回放/反馈，不是模型自主成功 |
| 原plates部件核验回放 | `demo_plates_e762_verified_parts_v4` | 精确部件映射/物理反馈；开门原谓词成立不等于交接就绪 |
| 原can-meat反馈回放 | `demo_canmeat_e807_feedback_trace_v5` | 11787原控制；包含原演示放置失误，不能自动准入全部成功标签 |
| 同状态纠正试验 | `c2_same_live_state_pilot_v1` | 真实接管实验及失败证据，不是已认证成功教师 |

查一个run先看`coordination_run_receipt.json`或`result.json`/序列状态，再看manifest、实际配置和checkpoint/source SHA；不能只看文件名、PID存在或step5000文件存在。

## 4. 活跃服务、端口和模拟器缓存

这是2026-09-12的状态快照，不是允许按PID直接kill的清单；操作前重新核对实际argv/用户/启动时间，防止PID复用。

23:54:39更新：为A4训练补足显存，仅在核验旧诊断已完成、完整启动身份和无连接后，以pidfd关闭旧A3前缀服务`3276541 / GPU0 / 8778`。该服务不再监听，旧权重、代码及`W/a3_prefix_radio_e121_l1_v1`结果全部保留；复现旧诊断时按其`service.launch.json`在新输出目录另行启动，不覆盖原回执。当前评测使用8773/8781，保持运行。下表的其他旧服务不因此自动获准停止。

| 用途 | 端口/GPU | W下证据目录 |
| --- | --- | --- |
| 修正版A3完整低层 | `127.0.0.1:8781` / GPU3 | `a3_aligned_full_low_service_v3`，外层同名`.log`/`.launch.json` |
| 修正版A3前缀诊断低层 | `127.0.0.1:8780` / GPU3 | `a3_aligned_radio_e121_l1_v2/service`及service日志 |
| B-final高层 | `127.0.0.1:8773` / GPU2 | `a2_final_eval_b_high_service_v1` |
| 原A2/旧A3等保留服务 | `8772/8776/8777` / GPU0或2 | 历史比较服务，是否可停由准确依赖及团队决定；8778已按上述核验关闭 |
| 上轮完整模拟器（已结束） | GPU1 | `native_a3_aligned_development_pilot_v3`；使用`kit_c1_gpu1_appdata_v1`私有缓存，2026-09-13核对campaign已退出 |

端口不能直接从外部访问时用SSH转发，不修改服务绑定扩大暴露。四张A100不等于四份可随意分配的空闲资源；先查实际进程和显存。渲染质量、IsaacSim版本与硬件兼容性仍需独立验证，吞吐不与正确性混为一谈。

## 5. 本地与GitHub结构

- 本地协作根：`/home/wsy/behavior`。
- GitHub：`https://github.com/dywsy21/behavior.git`；只同步代码、配置、tests和小文档。
- goal总计划与实时记录：`docs/plan.md`；三人任务板：`docs/TEAM_PLAN.md`；通用RL计划：`docs/RL_METHOD_PLAN.md`；开工先pull及实时更新规则：`AGENTS.md`。
- 根目录现保留`src/`、`configs/`、`tests/`、`scripts/`、`experiments/`、`tools/`、`assets/`、`skills/`、`licenses/`和项目元文件。
- 旧本地顶层Markdown报告在`/home/wsy/behavior/docs/archive/`；其中原goal计划已按用户要求移至`/home/wsy/behavior/docs/plan.md`作为实时维护入口。远端MAIN的7份未跟踪历史文档在`docs/archive/remote-main-20260912/`，继续作为历史记录。
- 147项旧本地实验、视频、附件、补丁和快照已可逆归档到`/home/wsy/behavior/artifacts/local-archive-20260912/root/`，约4.1GiB；`memlite-resume.L6rqZ6`也在此。没有删除，不把未完成研究草稿全当生产源码导入。
- 修正版radio视频本地位置：`/home/wsy/behavior/artifacts/local-archive-20260912/root/memlite-results-20260912/A3-aligned-radio-e121/`；其余历史文件按原目录名在同一归档根查找。
- 上游中文/英文文档入口保留在`docs/upstream/`，架构/数据/部署文档仍在`docs/architecture/`、`docs/data/`、`docs/deployment/`。
- 首次迁移commit为`69645b4220105ef1199fbbe90c889d4ae911ef6a`；2026-09-12的后续核验中，本地、GitHub和服务器新clone同为`c3367995234ce51964fff736f2544610c0320419`，两端工作树干净。该迁移检查点有474个跟踪文件，AGENTS及协作文档均跟踪；没有大于1MiB的跟踪文件。后续归档验收文档另行正常提交。
- 原源码身份、归档映射、扫描/验证及服务器安全checkout入口，以[仓库迁移记录](REPOSITORY_SYNC.md)为准。
- 新clone的`.venv`只是指向旧主仓环境的软链，已有editable安装可能仍导入旧`GalaxeaVLA/src`。本次未安装或切换服务；未来用独立环境或显式并核验`PYTHONPATH`/实际导入路径，不能仅看当前工作目录判断源码版本。禁止对活跃共享环境重新`pip install -e`改指向。

## 6. 本次安全归档结果

**用户已确认改为“先释放确认安全的部分，剩余暂不动”。最终仅迁移旧`hy_vla`实验，净释放516,587,864,064字节，即481.11GiB / 0.469834TiB / 0.516588TB（十进制）。** 不将最初两个候选的合计大小当作释放量。

| 项目 | 最终记录 |
| --- | --- |
| 源文件系统 | `/mnt/sdc1`，实际设备`/dev/sda1`，XFS；操作前可用302,917,308,416字节，操作后819,505,172,480字节（约763.22GiB） |
| 目标文件系统 | `/mnt/tmp1`，实际设备`/dev/nvme0n1p1`，XFS；约894GiB总量，操作后可用113,477,279,744字节（约105.68GiB） |
| 原入口 | `/mnt/sdc1/robodojo/experiments`，现为指向下列归档的兼容软链；**不是**主代码仓里的`GalaxeaVLA/experiments/` |
| 完整归档 | `/mnt/tmp1/robodojo-archive-20260912/experiments`；顶层为`hy_vla`，run最新时间2026-07-27 |
| 归档内容 | 353个普通文件、126个目录；逻辑大小516,607,863,251字节，源/目标核验时实际占用均516,608,692,224字节 |
| 内容验证 | 复制完成后独立`rsync --checksum --dry-run --itemize-changes --stats`，退出码0、变化数0，元数据匹配；这是完整内容比较，不冒称另算了每个大权重的SHA256 |
| 真正移除的副本 | 仅同盘隔离目录`/mnt/sdc1/robodojo/experiments.archived-source-20260912`；归档验证、原位软链/可读性、依赖检查通过后移除。归档中的完整文件保留，可恢复 |
| 未移动的范围 | HF源及本次形成的HF目标副本、`behavior_dev`、模型、数据、原`GalaxeaVLA`、xhz目录和现存服务；未为凑2T连带清理 |
| 操作后运行状态 | 7个原有服务监听与既有campaign保持，task2进程3632286仍在运行；它及排队task3/4的实际依赖均不在旧experiments目录 |

原始清单和校验/操作/恢复记录都在`/mnt/tmp1/robodojo-archive-20260912/manifests/`，最终记录为`experiments_migration_final_receipt_20260912.txt`。轻量、去除无关进程信息的协作副本见[归档验收材料](storage/README.md)。

HF源`/mnt/sdc1/robodojo/.cache/huggingface`原位保留。先前HF占用来自本次复制进程，不是已证实的生产占用；但其锁文件有当日活动，且未完成独立内容验收，因此未纳入迁移。`/mnt/tmp1/robodojo-archive-20260912/cache_huggingface`只是本次形成的未验收副本，不算新增已认证备份，也不算源盘清理成绩；本轮按用户决定不再动它。

恢复时先确认源盘空间充足及没有相关使用，在源盘新建**真实临时目录**，从归档复制、校验后再将原入口从软链切回真实目录，并保留归档/回滚入口。不要直接向当前软链入口rsync，否则会沿软链回到归档自身。详细边界见上述恢复记录。

`/mnt/tmp1`不自动意味着永久可靠的备份；冷归档的保存期限和磁盘健康仍要管理。未来若再清空间，重新按清单核验，不把此次放行扩大到其他目录。
