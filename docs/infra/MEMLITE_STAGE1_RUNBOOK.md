# MEM-Lite百任务阶段1操作手册

负责人：Codex / IMPL-MEM100-STAGE1、TRAIN-MEM100-STAGE1。最新状态以[plan](../plan.md)为准。
2026-09-30 21:30北京时间，用户明确授权后lc1高层一遍/lc2低层120小时正式进程各已完成≥5次真实更新、初始3200窗评测与step0保存，正在后台训练。数据准入、lc1高层16步/lc2低层32步及同run保存恢复此前已通过。下文命令是现有run的操作说明，**不要再次不带resume重复启动**。正式W&B高`8ecc6bb3908e`、低`e902c036e522`；链接与后续进度见plan。
原[9/30审查S1–S9](../experiments/2026-09-30-memlite-stage1-plan-and-readiness.md)是实施前快照；节点分配现改为lc1高层、lc2低层。

### 2026-09-30 GPU1 ECC恢复记录

21:54高层完成215更新后因lc1 GPU1 HBM双比特不可纠正ECC退出（Xid48/171/63/94/154），不是OOM；只有step0完整checkpoint，不能从日志215“续算”。22:38仅该UUID定点reset，pending row-remap已生效。22:45八卡各68GiB四模式读写、既有NCCL/BF16和ECC前后计数检查全通过后，按同源/同配置/同run `--resume`提交attempt2；最终真实恢复状态见plan。

- 健康工具：独立`src/high-ecc-health-20260930`（07ec13e）中的`scripts/infra/check_a800_gpu_health.py`，6个CPU合同测试。它不reset、不清ECC计数、不改训练权重，遇外部GPU进程/错误恢复标记/新ECC即失败。
- 证据：`runs/stage1_high_ecc_recovery_20260930/{memory-health-v1,nccl-v1}.json`；原故障日志`<high run>.supervisor/attempt_001.log`保留。aggregate历史错误不因reset抹除，不把计数非零一概视作本次新故障；pending/failure/recovery与增量必须核对。
- 不用减小batch或关闭ECC掩盖硬件错误，不整节点重启或批量reset。维护必须排空并精确核对UUID/PCI/进程，必要时协调管理员。重映射需reset生效，见[NVIDIA说明](https://docs.nvidia.com/deploy/a100-gpu-mem-error-mgmt/row-remapping.html)。短检通过不保证此卡以后不再故障，若重现应进入硬件维护而非无限重试。
- 本次训练仍使用d0528b4；健康工具不热改活跃源码或共享env。失败1787.20s继续由原supervisor预算累计；原W&B run中会保留失败attempt的曲线和从step0重放的新点，不删除旧证据。

## 路径与固定身份

共享根均为`/data/workspace/wsy/behavior2026`，四节点同一环境/数据，但两层分别单节点DDP，**没有跨节点梯度同步**。

| 内容 | 共享根下路径 |
| --- | --- |
| 已冻结训练源 | `src/stage1-d0528b4`，完整commit `d0528b4b0d553c252b6173355c1d4d567a9f9d76` |
| 实施分支 | `feat/memlite-stage1-lc12-20260930`，GitHub同步；不要pull活跃worktree |
| 共享环境 | `envs/g05-py310-cu128`；有作业期间不得安装/更新包 |
| 配置 | 源内`configs/memlite_stage1/stage1.yaml` |
| 原始RGB/动作/标注 | `datasets/2026-challenge-demos/datasets/fduTristin--2026-challenge-demos/snapshots/master` |
| 紧凑标签/候选v4 | `datasets/memlite-stage1-20260930-v4`；`acceptance.json`已发布，原manifest保持不改 |
| 低层动作界 | `manifests/memlite-stage1-v4-action-bounds/{stats,receipt}.json` |
| 高层初始化 | `models/memlite-b-final-20260910/B-final-model.pt`，950状态 |
| 低层初始化 | `models/memlite-a4-20260912/step_2500.pt`，1138状态/192 LoRA |
| 工程验收 | `runs/stage1_acceptance_20260930`；`high-v1`旧验收、`high-v2`/`low-v1`最终短验，均非正式模型 |
| W&B凭据 | `secrets/stage1-wandb.key`，0600，不能复制进Git/日志/命令行 |

原数据revision `4f50b44796641a4d526a19d9aeadc8aa51e2f2c2`。
训练文件26,347件/1,077,039,758,439字节全内容hash已通过，`source-hash-v1/result.json`保留。
旧B SHA `e7cd7bf738eb46901565088f829aa82c5c6ea95f610d4df1499e634959d29b13`；
旧A4 SHA `6186704788c27c9fae3502c884df0e259de5242ee8690fe578dcbc1f2632f269`。

数据manifest SHA `90ff0fa9334959dae5ff4368913add6c8a3858e9c124ca7b6c0b05abe85d6f23`；
低层新stats SHA `10dc04dc21f486fd3fd045c8d6114cf87b64b56cc7e038e581d60ff506cbd929`。
TRAIN12,299,471窗/18,895条演示，完整一遍48,045更新（最后207个真实观察）；EVAL645,793窗/994条演示。
两层候选身份相同但目标不同，不把低层4噪声当4倍独立样本。

## 训练合同

| 项目 | lc1高层 | lc2低层 |
| --- | --- | --- |
| 更新参数 | planner VLM 326张量；其他冻结 | action expert322＋LoRA192；单帧182项LoRA实际有梯度 |
| Loss | 规划/记忆CE，memory权重0.25 | 连续FM，4噪声/观察；不训练离散动作CE |
| Batch | 8卡×micro4×accum8=256 | 8卡×micro32=256 |
| 停止 | 合法TRAIN候选完整一遍，另168h事故上限 | 累计作业墙钟120h或200k更新先到 |
| LR | 1e-5，warmup1000，按实际一遍步数cosine至0.1倍 | AE/LoRA各1e-5，warmup2000，按200k cosine至0.1倍 |
| Eval/保存 | 每1000/2000更新 | 每2000/5000更新 |

共同：AdamW(0.9,0.95)、wd0.03、clip1、BF16 autocast＋FP32参数/Adam、gradient checkpointing、no EMA。
单帧三RGB、未来32动作（从0起）、执行前16，真实23控制→27模型位，padding只有[7,8,17,18]。
无物理反馈依据的outcome/terminal字段mask；不因演示段结束生成成功标签。

每条逻辑演示只抽一次固定offset0–15，其后+16。每遍全局无放回shuffle，低层下一遍改顺序不改phase。
每rank实际微批≥2task；尾两批重新分配真实样本，既不复制也不丢弃。完整遍数按真实候选计，不按原始视频帧数。
高层/低层在同一状态使用同一技能bundle；高层历史只有此前已发意图，最多3条，不传future/privileged memory。
可选parent标注缺失或不可靠时使用公开Task goal并不监督语义parent，不能把辅助NAV对象误当主目标。

## 数据准入与归一化

v4整源隔离111条：97时钟溢出、11非法leaf区间、2条旧歧义源、1条本次图审stale POUR。
原始资料均保留；不猜统一180帧偏移。原五任务950TRAIN/50eval来源按(task,raw episode,instance)保护，其余task按instance分组5%留出；public_test不入训练。

原B/A4采用五task TRAIN统计。低层沿用其mean/std/tail分位数/夹爪映射，状态输入坐标不改。
全TRAIN合法32步窗口仅扩展inverse action min/max安全界，并取消forward动作target±5裁剪，保留有限逆变换界。
不能用eval拟合范围；不是换成全量官方含eval统计。此后loss数值不能直接与旧裁剪目标横比。
部署低层必须绑定新stats和原始状态anchor，不能套旧五任务界导致合法动作被截断。

正式入口核`acceptance.json`绑定manifest SHA与六门：source_identity/protected_split/clock_alignment/human_review/normalizer/all_task_loader。
CPU契约通过不等于所有标签都正确；图审是覆盖百task/35技能/长短窗口的人工分层抽样，不是逐帧真值证明。
先完成实物输入＋8rank梯度＋原子保存/同run恢复/W&B验收，不能用`--preflight-stop-step`绕过正式数据准入跑长训。

## 启动、恢复、停止

**下面两个正式run已于2026-09-30 21:25北京时间获批提交。** 不要重复执行新建命令；先核对ledger/PID，确需恢复时必须遵守下方同run恢复协议。新实验须另有授权，并检查资源空闲、共享盘、Git准确版本；不要重登EC踢出队友。

```bash
source /data/workspace/wsy/behavior2026/src/infra-a5c9821/scripts/infra/activate_a800_training.sh
cd /data/workspace/wsy/behavior2026/src/stage1-d0528b4
export PYTHONPATH="$PWD/src"
git status --short
nvidia-smi
df -h /data/workspace
```

lc1正式高层（独立tmux窗口中执行）：

```bash
python -u scripts/infra/launch_memlite_stage1.py --component high \
  --output /data/workspace/wsy/behavior2026/runs/memlite_stage1_high_100task_v1
```

lc2正式低层（另一窗口）：

```bash
python -u scripts/infra/launch_memlite_stage1.py --component low \
  --output /data/workspace/wsy/behavior2026/runs/memlite_stage1_low_100task_v1
```

续跑用**相同源码、配置、output**加`--resume`。从最后原子发布的`checkpoints/latest.json`恢复模型/Adam、各rank RNG、epoch、下一已提交游标、LR进度及同一个W&B run。
DataLoader预取不算已消费；新试验从旧B/A4初始化用fresh optimizer，不将旧step1500/2500续算进本次预算。
恢复不允许代码/数据/参数悄悄变化，不允许预算清零；失败初始化、读取、验证和保存也计入supervisor累计墙钟。

停止前读`<output>.supervisor/ledger.json`确认该作业supervisor PID，只向**自己的supervisor**发TERM。
它发送保存请求，八rank在更新边界一致保存；不要kill队友进程、批量pkill或直接先杀torchrun。
没有最终回执/exit0不能称保存成功。`RUNNING`但原进程失踪的ledger需要人工核对，禁止删ledger重置120h。

工程短验使用独立output和`--preflight-stop-step 2`，退出后加`--resume --preflight-stop-step N`。
本轮最终高N=16、低N=32；旧high-v1的N=4证据单独保留。该flag是总更新终点不是再加N步；原每节点总60min/64更新预算不因换run重置，绝不自动接正式长训。

## W&B与证据读取

项目：[behavior2026-g05](https://wandb.ai/hanhanyy-fudan-university-school-of-management/behavior2026-g05)，group `memlite-100task-stage1-20260930`。
每层单独run，rank0写入；要求online，连接错误会失败，不静默offline。只记录数值/来源hash，不上传模型、数据、代码或secret。

- `train/update`是正式横轴；记录CE/FM、有效分母、观察数、LR、梯度范数、读取＋计算秒数、显存峰值、累计墙钟。
- `eval/loss`是全局有效监督加权均值，`eval/macro_task_loss`是task等权宏平均；`eval/task_0..99`逐task曲线。固定每task32窗/共3200，短验每task2窗/共200；不是全eval，更不是成功率。
- FM eval固定种子9183＋rank，并恢复训练RNG，避免eval改后续训练噪声。CE均值按有效token权重而非平均微批均值。
- 每run `recipe.json`、`wandb.json`、`restore_rank*.json`、`gradients_rank*.json`、`train.jsonl`、`eval_*.json`及`status.json`保留本地证据。
- 每checkpoint伴随SHA回执，`latest.json`最后原子切换；保存过程失败不会覆盖上一完整点。默认不删旧点，启动时须为约1.4TB最坏累计checkpoint量留余地；验收后共享余约3.8TB，不能视作未来始终足够。supervisor每2秒检查共享余量，低于256GiB拒绝新启动/通知自有trainer保存停止；不动任何队友文件。

新权重的效果须后续固定实例/预算仿真评测。短验loss/模型成功恢复只证明训练链运行，不证明MEM-Lite成功率改善。

## 本次实测与排期修订

最终源52项CPU回归通过；8rank Gloo不等分母/累积与单批参考梯度一致，真实两模型每rank恢复/梯度覆盖及global256也通过。
两节点最终run都先2更新退出，再同源同run恢复到高16/低32更新，final eval/save/exit0；高950状态、低1138状态/192 LoRA完整恢复。
W&B API读回均`finished`、最新update16/32、各100个task eval字段：
[高层短验](https://wandb.ai/hanhanyy-fudan-university-school-of-management/behavior2026-g05/runs/71f2c19d3165)、
[低层短验](https://wandb.ai/hanhanyy-fudan-university-school-of-management/behavior2026-g05/runs/00803c29ec18)。

| 真实数据短样本 | 高层（step4–16） | 低层（step4–32） |
| --- | --- | --- |
| 端到端更新吞吐 | 25.406观察/s | 70.409观察/s |
| 步时间中位数 / 最大 | 6.911 / 18.414秒 | 2.391 / 11.754秒 |
| rank0峰值allocated | 59.228GiB | 53.509GiB |
| 按本次速度外推一遍更新段 | 134.48小时 | 48.52小时 |

实际三RGB读取、随机长短标签和两节点共享盘并发都在内；不包含每次初始化/冷启动步、eval/保存。
这只有13/29个计时更新，不是长期稳定性保证；数据页缓存及首次kernel开销影响仍在，尚未分段profiler定位慢步来源。
150s监控中8卡observed used峰值≤66,730MiB/64,178MiB，0OOM；nvidia利用率包含collective等待，不能当纯计算利用率。
因此高层先按约6天排期、168h仅事故上限；低层120h按当前更新吞吐最多约2.47遍，扣eval/保存后更少，**不再沿缓存120.3观察/s承诺3–4遍**。
两节点可以正确并行训练；后续吞吐分解/等价优化另立有界任务，不在正式源码上热改或在本次混入新训练方法。

合main前仍须团队另一成员独立审查。本次是作者实施与验收，不冒称独审；正式作业使用已验独立分支/冻结源，未合main、无自动重试/续跑队列。运行状态见plan。
