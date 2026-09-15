# 已有执行视频精选及checkpoint对应表

2026-09-15 15:49（北京时间），Codex / V-01。按用户请求筛选已有结果；**三个不同任务的局部亮点，不是三个完整任务成功，也不是新评测或随机抽样的成功率。** 低层全部为FM，未选取原生AR或新500步方法候选冒充成功。

16:26按用户要求，三条带字幕视频另放到本地`artifacts/`根目录，已逐字节校验、原副本保留：

- [01-A2-trash-grasp-and-carry-intent.mp4](/home/wsy/behavior/artifacts/01-A2-trash-grasp-and-carry-intent.mp4)
- [02-A4-radio-grasp-policy-only-intent.mp4](/home/wsy/behavior/artifacts/02-A4-radio-grasp-policy-only-intent.mp4)
- [03-A3-halloween-candle-grasp-and-carry-intent.mp4](/home/wsy/behavior/artifacts/03-A3-halloween-candle-grasp-and-carry-intent.mp4)

## 16:13更新：实时意图字幕版已完成

用户追加要求后，已为同三条短片生成硬字幕版本，ckpt和原视频内容/速度不变。原720×720画面完整保留在720×1184视频中央，上方显示模型/仿真时钟/控制步，下方显示实时意图、原技能与对象ID、决策、命令记忆及来源/限制；未覆盖原片。

| 字幕版（本地） | 主要意图切换（短片时间） | 来源 |
| --- | --- | --- |
| [01 A2：抓桶并携行](/home/wsy/behavior/artifacts/selected-videos-20260915/subtitled/01-A2-trash-grasp-and-carry-intent.mp4) | 22.6667s，GRASP垃圾桶116 → NAVIGATE汽水罐114 | 18次已提交B-final结构化规划 |
| [02 A4：抓收音机](/home/wsy/behavior/artifacts/selected-videos-20260915/subtitled/02-A4-radio-grasp-policy-only-intent.mp4) | 固定GRASP收音机89，全段不变 | 原始标注条件；0高层调用，无高层记忆/CoT |
| [03 A3：拾蜡烛并携行](/home/wsy/behavior/artifacts/selected-videos-20260915/subtitled/03-A3-halloween-candle-grasp-and-carry-intent.mp4) | 17.20s，GRASP蜡烛91 → NAVIGATE下柜0 | 8次已提交B-final结构化规划 |

**字幕不是补写的CoT。** 原始日志有技能、目标、`decision`、`previous_outcome`、`memory_update`等结构化输出，没有另存的自由文本推理段。中文是忠实转述，不把画面观察改写成当时模型的意图。`issued_command_history`仅表示命令记录，不等于完成历史；`verified_world_facts=[]`与UNKNOWN_ONLY按原样标明，不能当作训练过的物理反馈。

时序以`consumed_actions_before_request`和实际安装的`active_subgoal`为准，短片开始前最后一次高层状态正确延续；不是将第一条截内更新提前到0秒。A2/A3日志本地与robo的SHA分别为`9151a1f7a7d5f01bc46e86c348dc445028b6206b928862e8747d9f9d8eb10162`/`7d88df90b0fb546c469f15a79d84f5bfca0e02cbaee7360b7fcaf136031c2eef`，现场核对一致。A2的132、A3的57个实际低层chunk均与字幕引用的已安装技能一致；1733个输出帧逐一检查字幕区间无重叠/缺口或未来状态。

三段成片1050/233/450帧、70/15.5333/30秒、15fps/H.264/yuv420p，完整解码全部exit0。Codex本人检查实际烧录成片的A2第339/340帧、A3第257/258帧及A4第130帧：切换前后一帧正确、中文字形可读、没有遮住原画面/文字越界。人工查看的是这5个成片视图，不宣称看完每一帧。

字幕目录`artifacts/selected-videos-20260915/subtitled/`同时保留`.ass`、逐段`.manifest.json`、`sources/*.planner-excerpt.json`（27条原始规划/固定条件，含可用raw_text）、`validation.json`和`review/`。媒体/原始日志/字体软链均被忽略，不入Git；本轮没有新模型、训练或仿真。

成片SHA256按01/02/03顺序：`8388ef5a762b8da4f459f592a093ae3893e5350255ebe00ced2fd2a883d8e503`、`6180dc6888c173dba17931ff0e2a82789e5f13d137c60c52791e068432928aa1`、`38c5c06d7cf630ebcfcbdcca636d964702eee216b17ab33785321fc8f85f7d2f`。

## 本地短片

目录：`/home/wsy/behavior/artifacts/selected-videos-20260915/`。原视频、失败后续和已有审核全部保留。以下均连续截取、没有加速、插帧或更改执行顺序。原视频15fps，每两个控制记录一帧，播放对应仿真时钟，不是慢速仿真实际消耗的墙钟时间。

| 短片 | 截自原视频 | 实际看点 | 限制 |
| --- | --- | --- | --- |
| [01：A2抓垃圾桶并携行](/home/wsy/behavior/artifacts/selected-videos-20260915/01-A2-trash-grasp-and-carry.mp4) | 00:20–01:30，70秒 | 从地面抓起垃圾桶，右手持桶从厨房走向客厅/电视柜附近 | 自动B-final，无演示前缀；后续未抓住汽水罐、未完成收垃圾 |
| [02：A4抓收音机](/home/wsy/behavior/artifacts/selected-videos-20260915/02-A4-radio-grasp-policy-only.mp4) | 视频帧224–456，即约00:14.93至结束，15.53秒 | 模型接管后伸手调整，最终右手抓持指定收音机 | 原回合先执行448步演示；短片从接管边界开始，不把前缀导航当模型能力。固定正确GRASP、没有自动高层；不证明离桌运输或开机 |
| [03：A3拾起蜡烛并走向柜子](/home/wsy/behavior/artifacts/selected-videos-20260915/03-A3-halloween-candle-grasp-and-carry.mp4) | 10:10–10:40，30秒 | 右手从地上拾起当前指定蜡烛，持物走回柜前 | 自动B-final，无演示前缀；发生在长回合后段，前面有反复调整/错目标，后续放置失败 |

这三条全片都不是完整任务成功。短片只是选择性展示执行较好的阶段；不能用其数量计算成功率。A4完整自主收音机的另三回合为0/3，见[独立完整评测](2026-09-13-a4-radio-full-eval.md)。

## 对应权重：均在robo，不在本机

### 01：低层A2-5000＋高层B-final

低层名称：`formal_a2_lora_history_5000_v5_arrowfilter_adamresume3000 / step_5000.pt`。

```text
/mnt/sdc1/robodojo/behavior_dev/direct_execution_L6rqZ6_20260910/formal_a2_lora_history_5000_v5_arrowfilter_adamresume3000/checkpoints/step_5000.pt
```

SHA256：`b57eed704179a1517f6582b7572c975f732d3bc3c49d80ce119eaa2f643ae640`。
推理配置：`/mnt/sdc1/robodojo/behavior_dev/direct_execution_L6rqZ6_20260910/a2_serving_composition_v3_adamresume/serving_config.yaml`。

### 02：低层A4-2500，固定技能，无高层ckpt参与

名称：`overnight_a4_20260912 / formal / step_2500.pt`。这是从A3-5000初始化后新增2500步的A4，不是仅从基座训练2500步。

```text
/mnt/sdc1/robodojo/behavior_dev/overnight_a4_20260912/formal/checkpoints/step_2500.pt
```

SHA256：`6186704788c27c9fae3502c884df0e259de5242ee8690fe578dcbc1f2632f269`。
训练配置：`/mnt/sdc1/robodojo/behavior_dev/overnight_a4_20260912/formal/.hydra/config.yaml`。
局部推理来源：`/mnt/sdc1/robodojo/behavior_dev/a4_radio_e121_l1_20260913_v1/service/`及同根manifest，执行commit `c7fb287`，后处理`a3ce491`；见[A4效果报告](2026-09-13-a4-training-effectiveness.md)。

### 03：低层A3-5000＋高层B-final

低层名称：`formal_a3_episodecoverage_5000_v1 / step_5000.pt`。

```text
/mnt/sdc1/robodojo/behavior_dev/direct_execution_L6rqZ6_20260910/formal_a3_episodecoverage_5000_v1/checkpoints/step_5000.pt
```

SHA256：`865193f1c8a257d72ea159bd7a46cebf945b0fd24905110b831f0e8a3438e940`。
推理配置：`/mnt/sdc1/robodojo/behavior_dev/direct_execution_L6rqZ6_20260910/a3_serving_composition_v2/serving_config.yaml`。

### 01、03共用的高层B-final

```text
/mnt/sdc1/robodojo/behavior_dev/direct_execution_L6rqZ6_20260910/formal_b_parent_format_1500_v1/checkpoints/step_1500.pt
```

SHA256：`d4580d80cdc91a707c233a6c1e625f8fbdeca8896dd38a3d570c6fad340184ef`。本阶段1500步，B谱系累计5000；UNKNOWN_ONLY，未训练物理反馈头。高层serving源码是同研究根的`b_parent_format_serving_source_v2`。

**复现注意：** checkpoint同名`step_5000.pt`不能互换。A2/A3录像来自下面列出的历史runtime，A3不是后来`native_a3_aligned_development_pilot_v3`；已有后续推理对齐修复，不能拿这些历史片段证明当前主仓默认推理能逐帧复现。加载时同时绑定原manifest、配置、源码和归一化统计，不只替换权重路径。

## 条件、物理证据和本人人工复核

- **A2抓桶：** `picking_up_trash`，public_test301/env seed0，0演示前缀，实际7901控制、官方失败。原逐步诊断1024确认右手持有`trash_can_116`，1029达到六帧稳定门；1280高层切导航汽水罐。本人本轮看全程拼图、16个局部面板及短片6个精确关键帧，图像显示抓起并携行；后段抓罐失败不能隐藏为成功。证据：研究根的`a2_final_trash_actual_episode_analysis_v1.json`及`A2_TRASH_FINAL_PERSONAL_REVIEW.md`。
- **A4收音机：** train episode121/instance138，env seed0/policy seed17，448未改原动作前缀＋464模型控制，固定正确GRASP/0高层调用。frame912的因果成功回执确认指定`radio_89`被右手持有，连续10个不同物理帧满足原六帧门。本人本轮重看24面板全程与最后12帧；没有从接触画面单独推断物体已完全离桌。证据：本地`artifacts/a4_effectiveness_20260913/A4-L1-analysis.json`及原run的`actual_analysis.json`、`actual_rollout/physical/e00000030.json`。旧JSON中的`personal_video_review=pending`是历史字段，后续人工报告已完成，不覆盖旧回执。
- **A3蜡烛：** `putting_away_Halloween_decorations`，public_test301/env seed0/policy seed17，0演示前缀，实际20682控制、官方失败。当前指令为`GRASP pillar_candle_91`，18731首次确认右手持有该对象、18736达到稳定门，18816切导航柜子；19200才进入PLACE_IN，后续未稳定放置。本人本轮看30面板全程、16局部面板及短片6精确关键帧，能看见拿起蜡烛并走回柜前。证据：研究根`a3_final_task2_actual_episode_analysis_v1.json`、`A3_EVAL_FINAL_PERSONAL_REVIEW.md`，及本地task_2的controller_events。原逐步分析覆盖全部动作、无diagnostic error，本轮只读核对，没有重跑物理判定。

public_test301已经反复诊断，是开发实例而非独立盲测。A4是训练实例上的局部诊断。以上抽帧检查不是宣称逐帧人工观看了所有长视频。

## 原始视频与文件校验

原始视频的本地位置：

1. `/home/wsy/behavior/artifacts/local-archive-20260912/root/memlite-results-20260911/A2-5000-Bfinal-trash/rollout.mp4`
2. `/home/wsy/behavior/artifacts/a4_effectiveness_20260913/A4-radio-e121-L1.mp4`
3. `/home/wsy/behavior/artifacts/local-archive-20260912/root/memlite-results-20260912/A3-5000-Bfinal-full/task_2/rollout.mp4`

对应robo原始视频：

1. `/mnt/sdc1/robodojo/behavior_dev/direct_execution_L6rqZ6_20260910/native_a2_final_development_pilot_v2_wirefix/task_1/rollout.mp4`
2. `/mnt/sdc1/robodojo/behavior_dev/a4_radio_e121_l1_20260913_v1/actual_rollout/rollout.mp4`
3. `/mnt/sdc1/robodojo/behavior_dev/direct_execution_L6rqZ6_20260910/native_a3_final_development_pilot_v1/task_2/rollout.mp4`

本轮现场重算三原视频SHA，本地与robo分别一致：

| 编号 | 原视频SHA256 | 短片SHA256 |
| --- | --- | --- |
| 01 | `5ae1dab348ae98b673b59b2c87c32ed239bfb3f4b7805089bbf24cafb230fc03` | `edb574c897eb19a7fbea1d09da159507fcbded82344e0ec089637004b984a6ab` |
| 02 | `a3dd6da39dc2417f8b591aac09a025c521a32063e4f700e4405984cbaaca7669` | `a937e3a27d17e271efb56a20cd0ae5bd08431c20ec05af6698d9d2dd98e06b59` |
| 03 | `bc12cfc2e4ae7dddd2a3c4cf424d5008dfc0a339863398bdad8443b06c8e36fa` | `164b9223a855c8533e1a5ccbcf097f0000822de9c1233c4178ea16f42153ba6e` |

短片分别1050/233/450帧，全部H.264、720×720、yuv420p、15fps，faststart；三个完整解码检查均exit0。使用FFmpeg连续裁剪并以libx264/CRF18重新编码，没有要求短片与原片压缩字节相同。原片已在本地，所以未重复下载视频；新增约19.4MB短片和少量审核拼图。

四份checkpoint在robo用只读stat确认存在，身份SHA取对应实际运行的权重加载回执，本轮没有重新读入/全量重哈希四个大权重。没有新训练、神经生成、仿真或数据release；没有删除/移动任何原始文件。
