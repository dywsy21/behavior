# 当前评测日志中的 goal_status 索引

主agent于2026-09-06 21:16 UTC核对robo上实际安装源码。`Evaluator.load_env`调用`generate_basic_environment_config`，后者显式设置activity_definition_id=0；官方runner没有改这个设置。`CompiledTask.check_goal`经`get_goal_conditions`、`compile_state`、`evaluate_state`按problem0中顶层条件原顺序产生索引。`BehaviorTask`把PredicateGoal返回的索引原样写进info，诊断trace只记录它，没有把这些真值喂给policy。

| Task | 索引 | 顶层条件含义 |
| --- | ---: | --- |
| Halloween | 0 | 所有南瓜各自在某个柜子里 |
| Halloween | 1 | 所有蜡烛各自在某个柜子里 |
| Halloween | 2 | 坩埚在某个桌子旁边 |
| Halloween | 3 | 所有柜子均关闭 |
| plates | 0 | 披萨与盘子成对保持ontop关系 |
| plates | 1 | 所有披萨在同一个冰箱里 |
| plates | 2 | 所有碗在同一个水槽里 |
| plates | 3 | 所有冰箱关闭 |
| can meat | 0 | 所有铰链罐在指定柜子里 |
| can meat | 1 | 所有铰链罐关闭 |
| can meat | 2 | 每罐装有条件要求数量的香肠（forn 2） |
| can meat | 3 | 指定柜子关闭 |

20:18完整轨迹快照中，Halloween在step4524失去条件3，说明“所有柜子关闭”不再成立。不能仅凭这个聚合forall条件确认哪个柜门打开、是否是目标柜门、是否抓住把手或首intent已经完成；开柜门本身还可能是必要中间步骤。

plates的条件0反复短暂变成false后恢复，只说明披萨/盘子关系谓词发生变化，不足以确认搬盘、抓取或有意操作。can meat起始即满足0/1/3，之后保持不变；不能把这三个既有条件当成政策完成了75%的任务。

这些顶层bool条件的数量不等于官方q-score。正式成功率仍取最终官方JSON及正常退出状态，不能从截图或谓词数量自创分数。

安装源码均在`robo:/mnt/sdc1/xhz/BEHAVIOR2026/BEHAVIOR-1K`：`OmniGibson/omnigibson/eval/utils/eval_utils.py:185`、`OmniGibson/omnigibson/tasks/behavior_task.py:624`、`bddl3/bddl/knowledge_base/models.py:964`、`bddl3/bddl/condition_evaluation.py:690`及各task的`bddl3/bddl/activity_definitions/<task>/problem0.bddl`。
