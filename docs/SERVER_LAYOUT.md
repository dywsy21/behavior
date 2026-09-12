# robo服务器文件位置与保留规则

更新：2026-09-12。本文件中的服务器路径属于`ssh robo`，不是本地路径。**源码走GitHub push/pull，数据、权重、环境、完整实验结果不走Git。**

本次目录整理/磁盘迁移正在核验；下表先记录实际实验已使用的位置。迁移完成后在末节追加真实目标路径、清单和df结果，不把计划当作已完成。

## 1. 总览

| 类别 | 当前服务器位置 | 用途与处理原则 |
| --- | --- | --- |
| 原主代码工作区 | `/mnt/sdc1/robodojo/GalaxeaVLA` | 用户指定迁移来源；源码已复制到本地，原31个已修改/90个未跟踪路径及`.git`保留。不直接pull覆盖旧改动 |
| 新GitHub协作入口 | `/mnt/sdc1/robodojo/behavior` | 已确认原先不存在；首次push后在此建立干净clone，完成状态见[迁移记录](REPOSITORY_SYNC.md)。不以它冒充最新A3实验快照 |
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
| 本次候选归档盘 | `/mnt/tmp1` | 必须先确认独立文件系统与容量，再迁移；实际位置/结果见末节 |

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
| 当前修正版完整runtime | `native_a3_aligned_full_runtime_v2` | 修补动作padding mask和执行起点0；不改权重、归一化及物理控制 |
| 当前修正版局部runtime | `native_a3_aligned_prefix_runtime_v2` | 仅用于有明确原演示前缀/固定技能的L1诊断，不能当完整SR |
| 完整runtime修复清单 | `a3_aligned_runtimes_v2_preparation.json` | 绑定两个runtime及动作对齐证据 |
| 旧A3完整runtime | `native_a3_history_runtime_v1` | 保留重现旧结果；含已定位的推理偏差，不作为新实验默认 |

**重要差距：复制`GalaxeaVLA`主工作区到GitHub不等于已将以上最新实验实现整合进去。** 后续按[任务P0-02](TEAM_PLAN.md)做逐模块diff/测试/合并；不要以复制完成冒称代码与A3运行快照一致。

## 3. 结果、视频和原始证据去哪找

| 结果类别 | W下路径 | 结论/边界 |
| --- | --- | --- |
| 旧A3完整五任务 | `native_a3_final_development_pilot_v1/task_0` … `task_4` | 各有`result.json`、`rollout.mp4`与记录；完整0/5，重复开发实例 |
| 修正版完整五任务 | `native_a3_aligned_development_pilot_v3` | `task_N/`放结果/视频，序列receipt记录完成情况；整理启动时task0失败、task1在跑 |
| 修正版完整评测清单 | `native_a3_aligned_five_task_development_v3.json` | 同A3/B、public_test301、env0/policy17；SHA `2722f9404b046ff69b443774d65fe1cb7b0bf4c25369d67b988cb7889924d730` |
| 旧A3固定GRASP | `a3_prefix_radio_e121_l1_v1/actual_rollout` | 448原前缀＋1280模型控制，无稳定抓取 |
| 修正版固定GRASP | `a3_aligned_radio_e121_l1_v2/actual_rollout` | 同权重/起点/条件仍未稳定抓取；`collection_result.json`不是完整任务SR |
| 修正版固定GRASP审计 | `a3_aligned_radio_e121_l1_v2/actual_analysis_v2.json` | 80段实际历史/动作/物理记录核对；80 IN_PROGRESS |
| 504次推理对照 | `a3_input_route_factorial_v2` | `result.json`及`execution_alignment_analysis_v1.json`；不再把mask/时间混杂误报为纯视觉域差异 |
| 42原训练锚点 | `a3_radio_e121_matched_original_inputs_v2` | 原radio121/138输入与目标逐值校验；大CPU缓存不进Git |
| 原radio真实回放 | `demo_radio_e121_alignment_v2_persist` | 真实原动作、观察、视频，非模型成功 |
| 第二原radio回放 | `demo_radio_e91_feedback_trace_v5` | 原始成功演示回放/反馈；完整准确名字以目录盘点为准 |
| 原plates部件核验回放 | `demo_plates_e762_verified_parts_v4` | 精确部件映射/物理反馈；开门原谓词成立不等于交接就绪 |
| 原can-meat反馈回放 | `demo_canmeat_e807_feedback_trace_v5` | 11787原控制；包含原演示放置失误，不能自动准入全部成功标签 |
| 同状态纠正试验 | `c2_same_live_state_pilot_v1` | 真实接管实验及失败证据，不是已认证成功教师 |

查一个run先看`coordination_run_receipt.json`或`result.json`/序列状态，再看manifest、实际配置和checkpoint/source SHA；不能只看文件名、PID存在或step5000文件存在。

## 4. 活跃服务、端口和模拟器缓存

这是2026-09-12的状态快照，不是允许按PID直接kill的清单；操作前重新核对实际argv/用户/启动时间，防止PID复用。

| 用途 | 端口/GPU | W下证据目录 |
| --- | --- | --- |
| 修正版A3完整低层 | `127.0.0.1:8781` / GPU3 | `a3_aligned_full_low_service_v3`，外层同名`.log`/`.launch.json` |
| 修正版A3前缀诊断低层 | `127.0.0.1:8780` / GPU3 | `a3_aligned_radio_e121_l1_v2/service`及service日志 |
| B-final高层 | `127.0.0.1:8773` / GPU2 | `a2_final_eval_b_high_service_v1` |
| 原A2/旧A3等保留服务 | `8772/8776/8777/8778` / GPU0或2 | 历史比较服务，是否可停由准确依赖及团队决定 |
| 当前完整模拟器 | GPU1 | `native_a3_aligned_development_pilot_v3`；使用`kit_c1_gpu1_appdata_v1`私有缓存 |

端口不能直接从外部访问时用SSH转发，不修改服务绑定扩大暴露。四张A100不等于四份可随意分配的空闲资源；先查实际进程和显存。渲染质量、IsaacSim版本与硬件兼容性仍需独立验证，吞吐不与正确性混为一谈。

## 5. 本地与GitHub结构

- 本地协作根：`/home/wsy/behavior`。
- GitHub：`https://github.com/dywsy21/behavior.git`；只同步代码、配置、tests和小文档。
- 当前计划：`docs/TEAM_PLAN.md`；通用RL计划：`docs/RL_METHOD_PLAN.md`；每次开始先pull的规则：`AGENTS.md`。
- 根目录现保留`src/`、`configs/`、`tests/`、`scripts/`、`experiments/`、`tools/`、`assets/`、`skills/`、`licenses/`和项目元文件。
- 41份旧本地顶层Markdown报告在`/home/wsy/behavior/docs/archive/`；远端MAIN的7份未跟踪历史文档在其`remote-main-20260912/`子目录。它们是历史记录，不是当前任务板。
- 147项旧本地实验、视频、附件、补丁和快照已可逆归档到`/home/wsy/behavior/artifacts/local-archive-20260912/root/`，约4.1GiB；`memlite-resume.L6rqZ6`也在此。没有删除，不把未完成研究草稿全当生产源码导入。
- 修正版radio视频本地位置：`/home/wsy/behavior/artifacts/local-archive-20260912/root/memlite-results-20260912/A3-aligned-radio-e121/`；其余历史文件按原目录名在同一归档根查找。
- 上游中文/英文文档入口保留在`docs/upstream/`，架构/数据/部署文档仍在`docs/architecture/`、`docs/data/`、`docs/deployment/`。
- GitHub首次同步、原源码身份、归档映射、扫描/验证及服务器安全checkout入口，以[仓库迁移记录](REPOSITORY_SYNC.md)为准。

## 6. 本次清理迁移记录（待实际完成后填入）

目标是源盘净释放约2T，同时保留重要文件。必须记录实际源/目标挂载、df前后值、每项原路径/新路径、校验、原位软链、恢复方法及活跃实验存活证据。若安全候选不足，说明实际释放量和保留原因，不为凑数删除重要材料。

不要根据本节的“目标”执行删除；实际迁移只按已核验清单逐项执行。`/mnt/tmp1`位置不自动意味着永久备份，后续保存期限和磁盘健康也要管理。
