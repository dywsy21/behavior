# 最低固定验证loss checkpoint：选择与有限闭环

Owner：Codex / EVAL-G05-BEST；分支`eval/g05-100k-20260928`。2026-09-28用户以本目标替代尚未完成的100k五任务评测。原run只保留已完成两条和中断证据，不续跑。

## 主要假设与选择口径

同一个Qwen3.5 noMEM训练run里，固定留出FM误差最低的已保存权重，是否比100k改善原始初态下的任务推进。FM是实际部署分支，**选择FM均值最低者**；CE只作参考/完全同分时优先级，其后取更早step。不是任意相加CE与FM，也不按训练loss选。

- 原run：`/mnt/sdc1/robodojo/outputs/g05/r1pro/behavior5_nomem_bs8_4gpu_20260923T135319Z`；候选`checkpoints/step_10000.pt`至`step_100000.pt`、每10k一个，共十个，均已确认存在。不引入别的架构/数据/训练run。
- 原日志每次仅32个轮转验证样本；checkpoint时刻FM：10k .0871825、20k .0920016、30k .1017845、40k .0953641、50k .0871086、60k .1005412、70k .1215389、80k .1102732、90k .0951086、100k .1259662。50k/10k几乎持平，不把这些非配对值当最终排名。
- 统一原逐task5%留出的50episode，各25%/75%两个32步完整窗口，共100例/每task20；原seed19000+i、原FP32参数＋BF16 autocast、相同processor/stats/codec，0训练。全部task/episode/index/fraction逐行配对，均值回对原始loss行。
- 40k/100k复用v4已核SHA的100行/各权重946严格恢复和FM-only一致性门；其余8个各100例，最多800新样本forward。CPU AST比对确认load_model/offline/选窗/23维转换/环境逻辑与060aeac不变。选择是在小固定开发验证样本上，不声称全量eval全局最优或独立盲测。

## 预算与停机

- 离线同时最多4worker/每GPU一个，阶段上限20min；任何候选加载、完整性、非有限值、样本不配对失败即停止，不挑剩余子集冒充全部十权重最佳。
- 选中者只加载一次部署：GPU0、端口8932；GPU3串行原radio301/3224控制与trash301/1024控制，0专家前缀、环境seed0/policy17、原始reset，仍A100 PathTracing。两场景与已完成100k相同，合计4248控制/266FM调用，每次32预测、从0执行前16，不改归一化/底盘/机器人/物理。
- 总墙钟75min，单模拟器≤40min且服从总剩余；0新增训练、0自动重试/加时/任务扩展。无官方成功就如实报告预算内未成功，不把两个开发回合称正式SR。
- 新代码先CPU测试、固定Git、新独立worktree后运行，旧源/run/视频/队友环境不热改。只清理新监管所拥有的子进程组；用户另改任务立即停止这条序列。

## 证据位置与当前状态

- 新入口：`scripts/experiments/eval_g05_best.py`；双端14 CPU回归已过，实际source **18acacf6ac81b1fa122fd0bfdcfaf2a18ed888ff**、新独立`behavior_dev/git_worktrees/eval_g05_best_18acacf`。
- 新输出：`/mnt/nvme_tmp/robodojo_g05_best_eval_20260928/v1`，`candidates/<step>/`为新离线记录，`selection.json`为全十排名/选择理由，`identity.json`为实际部署权重SHA，`task_0/`/`task_1/`为闭环。
- 私有runtime：`/mnt/nvme_tmp/robodojo_sim_runtime_20260925/eval_g05_best_20260928_v1`。
- 本地：`/home/wsy/behavior/artifacts/g05-best-eval-20260928/`（视频/原始行不入Git）。
- **12:09北京时间：监管3856250已唯一launch，最低者与闭环结果均待。** 原100k v4自有进程均已退出，task2中断192控制、其余未启动；不继续旧序列。启动前四卡空/849GiB RAM可用，原证据与decoder门通过，不据提交启动宣称模型效果。
