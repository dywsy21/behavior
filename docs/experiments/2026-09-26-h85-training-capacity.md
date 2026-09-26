# H85动作数据训练容量验证

负责人：Codex主代理。目标是证明已准备的数据能被真实VLM动作训练消费，并核算至少3小时有效微调所需样本；不是本轮直接长训，也不是成功率实验。

## 固定数据与配方

数据为`h85_sft_v1`，manifest SHA `fb95625b265b564cb07cb481615a2f3fead194c26c18bfd65a70da77441fbad9`。212,500唯一TRAIN窗口/386实例，validation23,009、test20,705保持封存。CPU全量重数＋50实际getitem的QA SHA `b5f59acf8a7243d06131791bb183873dab0cfaaf2b6da27b99db4a238caea73a`，不重做原数据或改标签。

| 项目 | 固定选择 |
| --- | --- |
| 基础模型 | 已登记Qwen3.5-2B，原始权重SHA aa33250c…680b1；无旧adapter |
| 输入/目标 | 当前三RGB 448/320/320＋当前本体/任务/技能；r1-composite16-v1长JSON动作＋EOS |
| 可训练部分 | 仅语言模块线性层LoRA，rank16/alpha32/dropout0.05；视觉和原基座冻结 |
| 优化 | AdamW，LR5e-5，betas0.9/0.95，weight decay0.01，梯度裁剪1 |
| 批量/步数 | microbatch2×累积4＝有效batch8，32更新；前4更新预热，后28更新量吞吐 |
| 采样 | seed41，全TRAIN窗口均匀无放回；首两条插入全TRAIN最短/最长序列作数值门，其余随机，总256唯一窗口 |
| 读取 | 四个spawn CPU worker，prefetch2，三相机原视频按时钟读，私有样本ID不进入模型 |
| 资源/产物 | 仅空闲GPU2；1800秒＋30秒清理、32GiB自身显存、8CPU、1GiB含缓存；不写checkpoint、不改线上接口 |

这不是任务均衡或实例均衡采样：窗口多的任务自然占比大；256窗口也不是256个独立实例。新epoch使用新确定性排列，完整遍历时每个TRAIN窗口仅一次。不能顺序读取文件头几个任务，或靠重复几个样本凑时间。

变长动作回答的每个microbatch先算token平均CE，再按其监督token数占有效batch总数的比例反传，保证梯度等价完整batch的token平均；不是四个microbatch的CE机械等权。监督只含动作JSON＋EOS，prompt/padding无loss，视觉输入保持真实。

## 验证内容与容量口径

1. CPU准备：`prepare_trajectory_benchmark.py`读取全部TRAIN长度，固定256窗口清单；真实四进程加载全部768当前RGB、含极端长度样本，逐item核来源、token、mask/CPU device。≤600秒/4MiB、0GPU权重/训练，候选输出`h85_benchmark_cpu_v1`。这不是新增人工审查或训练吞吐。
2. GPU基准：`launch_trajectory_benchmark.py --run`在干净新源码运行；先确保已核队友3641677–3641680退出、GPU2没有计算/图形进程且≥64GiB可用，再加载权重。`benchmark_trajectory_training.py`核全TRAIN长短样本的原生CE与裁剪logit CE相等，再完成32更新，要求loss/gradient有限、LoRA确实改变、原基座无梯度。
3. 每个有效batch分别记录“已读好数据后的复制＋计算＋优化器时间”和“包括等待数据的时间”。剔除前4步后报告两种平均吞吐，以及最快实测计算步的samples/s。用这个较快速度核算三小时需要的窗口数；不以慢I/O来人为满足数据时长。
4. `212500 / 最快实测计算速度`若≥10,800秒，即在该固定单A100配方下，一遍不重复训练窗口已足够。否则如实报告不足，不能自动加epoch宣布完成。有限基准不是任何硬件/批量/优化方式的速度上界，也不能证明策略效果。

GPU2若出现外部新任务或资源/数值/时间超限，只终止本次新建的自有session。其他三卡允许队友照常工作，不能因为其占用高而操作他们的进程。无后台自动抢占、重试或新3小时长训。

## 当前状态

**19:33资源等待更新：** goal已标记blocked而非完成。19:30:59队友3641677–3641680仍live（elapsed22:23:54），固定9c38fec干净；实际GPU启动入口`--preflight`在占用门拒绝并exit1，0权重/worker/run创建。CPU工作无新增缺口，继续等待原训练自然结束；GPU forward/backward、吞吐和≥3h容量保持未证实，不因数据数量或预检拒绝改结论。

**2026-09-26 19:27北京时间实测：CPU准备已完成，GPU基准未运行。** 新冻结9c38fec96d974e4c6e7120d6a650d9a6ee5ab754，run `/mnt/nvme_tmp/robodojo_vlm_actions_20260926/h85_benchmark_cpu_v1`耗时32.907秒/exit0，服务器60回归4.985秒通过；全部256不同TRAIN窗口/768当前图经四spawn读取与batch核验，来自182实例，task0–4分别12/31/86/76/51条。最短/最长门实际1242/2481总tokens，后者回答1315，均完整加载。

本地完整launch/sampling/result及transfer在`artifacts/agentic-vlm-goal-20260918/h85_benchmark_cpu_v1`。result SHA `074759c50cb981b21e201c3cfeb564a489d5074446f3b1473e60687f45c0e01e`，sampling SHA `f3dc6a4a9a286df209044973afab18da443c1d92aac66cbad30c6a89d485ed66`。双端SHA、全部256ID/全顺序/来源计数及128 microbatch的token/长度在本地独立核过。0模型权重/训练/CUDA初始化，不把32.907秒视为训练速度。

19:27:06四卡原xhz3641677–3641680仍live、elapsed22:20:01，GPU基准目录尚不存在。下一无需重新准备数据，待资源自然释放后，在同一冻结源码按既定命令运行一次；运行前先核真实资源和源码干净，不能据以下命令存在当作已经启动：

```text
PYTHONPATH=/mnt/sdc1/robodojo/behavior_dev/semantic_agent_20260917/deps:<固定源码>/src
/mnt/sdc1/robodojo/GalaxeaVLA/.venv/bin/python scripts/vlm_sft/launch_trajectory_benchmark.py --run
```

启动时还需设置HF_HUB_OFFLINE=1/PYTHONDONTWRITEBYTECODE=1并用外层1850秒＋35秒清理兜底，内部监管限制1800＋30秒；不热改、重试或超预算继续。真实GPU forward/backward/吞吐/三小时容量仍未验证，候选`training_eligible:false`保持。

### 准备阶段记录

2026-09-26 19:16北京时间：代码和CPU测试准备中，初54项通过、独审中；原四卡队友任务19:00:29仍live。尚未运行上述CPU新入口或GPU基准，无吞吐实测、无3h容量结论。当前数据`training_eligible:false`保持。最后以真实结果和对应源码commit更新，不把文档中的配方当作已执行实验。

19:19增量58目标回归3.381秒通过；覆盖不等token微批的逐参数梯度等价、全局无放回/极值gate、spawn序列化排除AV句柄、掩码/审计ID隔离、容量公式及伪造回执拒收、外来GPU2任务只清本次child并保留主失败。两新worker入口本地CPU导入通过，不触发模型加载。仍待独审/真实四进程预检与GPU基准，不把CPU测试称forward/backward实测。

19:22全60定向回归3.224秒通过；独立只读终审另15新训练回归/四脚本编译/diff检查通过，仅放行已登记的一次CPU预检。GPU启动与实际训练容量仍未验证，必须自然等已核队友任务退出。
