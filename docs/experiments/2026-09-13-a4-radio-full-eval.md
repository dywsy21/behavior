# A4＋B-final：完整收音机任务评测

状态：2026-09-13 12:09（北京时间），3回合中2回合完成、303运行中。不是最终成功率报告。

## 预先固定的条件

- Owner：Codex，任务A-02/B-03；用户要求从局部抓取推进到全程成功率验证。
- 代码：`feat/low-fm-a4-effectiveness-20260913`，实际运行commit `be23b06b045fcc06e6fa8ab6dfb724aeebe95f31`；独立worktree运行，不热更新。完整物理/成功判据循环与既有审核版回归一致。
- 低层：A4新增2500更新最终权重，SHA `6186704788c27c9fae3502c884df0e259de5242ee8690fe578dcbc1f2632f269`；全量base/adapter恢复通过。高层：不变B-final，SHA `d4580d80cdc91a707c233a6c1e625f8fbdeca8896dd38a3d570c6fad340184ef`，仍UNKNOWN_ONLY。
- 任务：`turning_on_radio`。目标是官方指定收音机`toggled_on`，不是拿起物体；只接受环境成功判据。
- public_test初态301、302、303；环境seed0、policy seed17。每回合最多3224真实控制、墙钟5024秒；总预算9672控制，3回合后停止，不自动重试或改策略。
- 0演示前缀、0固定oracle技能；高层自主规划、命令记忆每回合清空，低层历史及随机状态重置。六帧三相机、23真实控制/27模型表示，padding `[7,8,17,18]`，预测32/执行0:16；7次真实wire检查通过。
- 原始物理状态仅用于诊断，未输入任何模型，评测轨迹不回流训练；本轮0优化。
- 301是已反复使用的开发对照，302/303是这轮额外初态；小样本不是官方榜单、50任务总体SR或严格统计证明。

## 实际结果（暂时）

| 初态 | 完成情况 | 实际控制 | 高层调用 | 官方成功 | 耗时 |
| --- | --- | --- | --- | --- | --- |
| 301 | 完整预算结束 | 3224 | 26 | 否 | 808.92秒 |
| 302 | 完整预算结束 | 3224 | 26 | 否 | 818.32秒 |
| 303 | 运行中 | — | — | 待完成 | — |

301结果SHA `5aa127330248caa4c19eefa6bbd7df7f59aea6d64af573ba007fa51e003ae4a9`；302结果SHA `ec049bf012b08c95e51d274bd2feede5ccc1e0e91a333429638426d0b0373643`。均正常退出，不是墙钟截断或基础设施错误。当前只有0/2已完成，不给尚未完成的回合补失败标签。

## 301失败位置与本人核验

高层0–1152控制发NAVIGATE，1152–3224发GRASP，始终指定正确`radio_89`，没有进入TOGGLE。3426行物理诊断包含全部3224真实控制，0诊断错误、0行确认持有对象。本人检查全程25张均匀抽帧与末帧：机器人转向后到桌前，右手在对象上方但没夹住；不是已经抓住后未切换。

完整视频已复制本地并与服务器校验相同SHA `000ca7d99ef98da01e337f0abab17ecb018bad76e627dc408b3d00692230db53`，时长107.533秒、1613帧。抽帧图中的0–24是抽样序号，不是原控制步号。

## 证据位置

302复核：高层第512控制切GRASP，之后持续到3224，未进入TOGGLE。3426行诊断同样包含3224真实控制，0错误、0确认持有对象。本人查看25张均匀抽帧＋末帧：右手未夹住，后段收音机倒在桌面上，但未拿起。视频本地/服务器SHA同为`4ea517c8d855a7dd2f0137f252131d80378c73f259fe08eae25c6a5224152ae8`，107.533秒/1613帧。不能把失败仅归因于高层没发抓取或切换太晚。

- 服务器run：`/mnt/sdc1/robodojo/behavior_dev/a4_radio_full_20260913_v1`；`manifest.json`固定条件，`launch.json`记录supervisor1479480，`wire_probe.json`为实际通信验收。
- `instance_301/`等：每回合`result.json`、`rollout.mp4`、`controller_events.json`、`official_step_trace.jsonl`、`diagnostic_physical_evidence.jsonl`。`summary.json`仅整轮结束后写出。
- Git运行副本：`/mnt/sdc1/robodojo/behavior_dev/git_worktrees/a4_radio_full_20260913`，不热pull。
- 本地视频：`/home/wsy/behavior/artifacts/a4_radio_full_20260913/instance_301/rollout.mp4`，忽略目录，不能当cache随意删除。

下一步：完成预定303后更新原始计数、失败阶段及视频；不因为当前0/2自动加训练或重跑开发实例。

