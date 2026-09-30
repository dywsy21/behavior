# 单帧 MEM-Lite / 八卡 batch64、128 短测

2026-09-30，Codex / BENCH-MEM1F，分支 `bench/memlite-oneframe-a800-20260930`。用户授权短测速，不是全量训练。

## 最终实测结果

真实旧A4已经同步并验证。共享盘主权重：`/data/workspace/wsy/behavior2026/models/memlite-a4-20260912/step_2500.pt`，16,581,363,550B；源robo、本地中转、目的lc3完整SHA均为`6186704788c27c9fae3502c884df0e259de5242ee8690fe578dcbc1f2632f269`。保留完整FP32模型、LoRA、optimizer/scheduler/RNG，不作精度转换。ActionCodec、Qwen3.5 processor八件、A4配置/统计/回执五件也已完整同步/逐文件核验，无G0.5新下载。高层B-final不属于此次低层测速，未迁移。

### 八卡计算吞吐

lc3的8×A800 80GB PCIe；每相机只取当前一帧，共三张256×256，预测32步动作。**这里测低层SkillFM的动作专家＋VLM LoRA，高层独立训练不包含在内。** 两臂同一完整A4权重，BF16 autocast＋FP32权重，LoRA r8，4个FM噪声/观察，保留23真实控制和27维表示。AdamW LR1e-5、betas0.9/0.95、WD0.03、clip1；fresh optimizer，不恢复旧训练步数。vision沿用现有SDPA fallback，其余注意力/梯度checkpoint配置沿用旧A4，共享env未升级。

| 全局batch | 每卡microbatch / 累积 | 平均秒/更新 | 观察样本/秒 | 每卡PyTorch峰值allocated / reserved |
| --- | --- | ---: | ---: | ---: |
| 64 | 8 / 1 | 1.3421 | **47.6858** | 28.37 / 28.71 GiB |
| 128 | 16 / 1 | 1.5204 | **84.1870** | 36.11 / 36.56 GiB |

128吞吐为64的**1.765倍（+76.55%）**，相同观察数量的计算时间约少43.36%。均无OOM，无需累积fallback；不是batch翻倍就预设速度翻倍。allocated/reserved是PyTorch统计，不包含全部CUDA上下文/NCCL等额外占用。

每臂6步预热＋24步计时，64实测1536观察/32.210833s，128为3072/36.490185s。计时取每次更新八rank的最慢墙钟，包含组batch/H2D、tokenization、真实forward/backward、AdamW、梯度裁剪和DDP同步，不含初始化/JIT预热、原视频解码、正式dataset loader、checkpoint保存或eval。4个FM噪声没有重复算作4个观察样本。

八rank均逐字节验证1138模型状态和192LoRA完整恢复，322个AE参数与182个LoRA参数有实际梯度，冻结参数无梯度；LoRA存在非零FM梯度。十条输入来自原已审查的五任务TRAIN微批，按原SHA复用；单帧只裁历史图像和历史状态，**未来动作起点保持0**，未用观测长度推移目标。重复十条仅用于计算性能，不能拿loss下降证明泛化、单帧效果或成功率提升。

实际计算源`7af393b705bde65a485223d7e7cae0fa3eb86092`。`batch64-v1`在初始化比较CPU/GPU张量时报错，0更新；修复为跨设备逐字节比较后用新run重测，其约68s也计入预算。最后64-v2/128-v1均exit0，共60临时更新、0新checkpoint；不是正式训练。七项CPU回归在启动前通过；最终源`3f974fc6530d5730210af8aed58e73916aa250aa`补单帧限定门后，13:42北京时间在共享env/新冻结worktree完成八项CPU合同回归（3.076s）＋三项I/O边界回归，全部通过/exit0、禁用CUDA。未追加GPU重测，实际吞吐仍绑定原计算源。

### 百任务I/O短测

使用真实单帧TorchCodec读取函数（每样本新decoder、approximate seek、真实FPS与时间容差），32CPU worker，100任务各一episode/四个窗口，重复两轮共800窗口，三RGB＋32×23D原动作＋61D当前状态并resize。两GPU臂退出后才启动，0标签/模型更新/CUDA。

| I/O轮次 | 400窗口墙钟 | 窗口/秒 |
| --- | ---: | ---: |
| 首轮，包含worker启动 | 8.1347s | 49.1723 |
| 第二轮，磁盘/文件缓存已热 | 2.6505s | 150.9128 |

热轮高于batch128的84.19观察/s，说明这组短测中读取有余量，**不保证全量训练不会被I/O限制**：只覆盖100/20,000条episode，重复的是同一批窗口，没有清系统缓存；Parquet按episode分组读取，不是生产HF dataset/任务采样/MEM标签/归一化/增强的完整管线。仅几秒的热缓存值不能直接当1TB随机shuffle的持续带宽；也不能把它与计算吞吐简单相加来声称实测端到端速度。

### 一遍数据的计算量外推

公式：观察起点数 ÷ 实测观察吞吐 ÷ 86400。**这是按当前代表性输入长度外推的计算时间，不是已经完整跑了一遍100任务，也不是端到端交付工期。** 尚未计入真实全量loader、不同技能文本长度、保存、eval、负载波动；完整MEM-Lite合格窗口/holdout索引仍未发布。

| 采样口径 | 起点数 | batch64 | batch128 |
| --- | ---: | ---: | ---: |
| 全部原始帧，每帧一个起点 | 210,916,774 | **51.19天** | **29.00天** |
| 约95% train逐帧，仅近似 | 约200,370,935 | 48.63天 | 27.55天 |
| 明确改为stride16，全量候选 | 13,191,664 | 3.20天 | 1.81天 |
| 明确改为stride32，全量候选 | 6,600,830 | 1.60天 | 0.91天 |

后两行是**更换采样密度**，不等于让同样全部逐帧样本跑得更快；预测32步/执行16步也不会自动使训练stride为32/16。95%实例留出不严格等于95%帧，正式train索引要另算。本轮没有修改正式采样策略或启动百任务训练。

当前吞吐选择倾向global128，但不据此宣称收敛更好或自动增大学习率。此次同时涉及单帧、八A800和缓存数据路径，不能与旧四A100六帧正式训练的速度作单因素因果比较。

### 证据与未完成的准入

- GPU运行根：`/data/workspace/wsy/behavior2026/runs/memlite_oneframe_benchmark_20260930/`，`batch64-v2/result.json`、`batch128-v1/result.json`与各`restored_rank*.json`；日志在`logs/`。完整小结果已核SHA取回本地`artifacts/a800-memlite-oneframe-bench-20260930/results/`。
- [计算结果/源与输入SHA](../infra/results/2026-09-30-memlite-oneframe-compute.json)、[I/O结果](../infra/results/2026-09-30-memlite-oneframe-io.json)、[权重传输回执](../infra/results/2026-09-30-a4-checkpoint-transfer.json)。I/O源`61661f3`、run`cpu-io-v1`，逐task/episode/frame证据在`pass0.json`/`pass1.json`。
- GitHub push成功；lc3直接fetch遇GnuTLS -110，因此使用已push的同commit Git bundle，经verify/fetch导入对象后新建冻结worktree。没有覆盖活跃源码、重登录VPN或修改共享env。
- 这是**单帧计算入口的最小迁移**：最后加入显式单帧限定，防止把新增class误认为六帧vision/builder/通用checkpoint loader已全部移植。原robo冻结六帧运行源不改。35技能映射、完整标签/时钟QA、旧holdout保留、全量归一化、generic恢复钩子与正式loader整合仍属于P0-02/正式训练准入，本轮未完成。

## 预登记

- 假设：单帧与更大的每卡 microbatch 能提高旧 A4 低层的观察样本吞吐；不据测速判断任务效果。
- 范围：单节点 lc3、8×A80080GB PCIe，先 global64=8×8×1，再 global128=8×16×1。保留三相机256、32未来动作、23真实控制/27表示、FM四噪声、FM→VLM LoRA r8＋动作专家；高层不参与低层更新。
- 数据：固定官方 2026 revision `4f50b44796641a4d526a19d9aeadc8aa51e2f2c2`，ModelScope snapshot 的 RGB/动作/meta/annotation，不下 depth/raw。只读取有限代表样本，不构造/发布百任务正式监督集，不保存部署权重。
- **12:04北京时间用户覆盖：不使用随机初始化，优先同步旧模型，缺失才下载G0.5。** robo原A4完整checkpoint已找到，16,581,363,550B；本轮重新计算SHA256为`6186704788c27c9fae3502c884df0e259de5242ee8690fe578dcbc1f2632f269`，与历史一致。完整权重/LoRA迁至共享盘并验SHA后才允许测速；原随机初始化方案取消，0GPU更新。
- 实施：独立测速入口复用实际 G0.5 神经计算图和 A4 LoRA/冻结/技能前缀配置，区别缓存输入计算吞吐与真实视频/动作读取吞吐。若不能完整贯通原 SkillFM 的数据合同，明确标注性能代理与未验证项，不用伪造合格标签/削掉梯度换速度。
- 短测预算：每个 batch 至多40更新（包含预热与计时），两臂80；若128 OOM可一次 micro8×accum2 的128替代，额外≤25更新。单臂GPU墙钟≤20分钟、总GPU运行≤50分钟；CPU准备≤30分钟，有限读取≤10,000观察窗口。内存不足/非有限loss/梯度缺失/数据不匹配/外部占卡即停止当前臂，保留失败证据，不启动长训。
- 对照：两臂同 seed73、同模型/输入协议/训练参数集合；报告每卡batch、累积、真实样本/s、时间、峰值显存、数据等待和包含/排除的开销。不能只按更新/s比较batch，也不能把四次FM噪声算四个观察样本。
- 代码：启动前 commit/push，服务器独立固定 worktree；共享env与原infra源不改。单节点没有队友GPU进程才启动，绝不停止队友任务。结果和源SHA补入本文及plan。
- 全量外推：原始210,916,774帧与约95% train两种口径分开；原逐task留出、技能有效区间未完成正式发布，外推不冒充实际唯一全覆盖。

## 历史状态（最终结果见上方）

13:18（北京时间）GPU计算探针修复初始化后CPU/GPU权重比较问题，源`7af393b`、`batch64-v2`已八rank全部1138状态/192LoRA逐字节恢复；预热未完成、吞吐待。原`2193368`/v1失败0更新，约68s计入原预算，v2仅1100s超时、30更新。CPU七回归实际通过。百任务I/O采用独立只读代理：100任务×一episode×四窗口×两pass=800，32CPU worker/≤600s，须计算臂全部退出后才运行；保留“缓存计算≠完整训练、grouped读盘≠正式shuffle”的限制。

13:02（北京时间）**A4权重与14配套资产迁移完成**：主权重`models/memlite-a4-20260912/step_2500.pt`，16,581,363,550B，源/本地/目的完整SHA一致（`61867047…32f269`）。真实checkpoint已就位，旧原件和中转保留。新增单帧入口/六项CPU合同测试待服务器执行，0GPU更新；旧五任务十行TRAIN缓存仅用于计算吞吐，不当百任务端到端I/O证据。

12:43（北京时间）主A4在本地完整SHA通过后自动上传共享盘中，最终名尚未发布；`models/action_tokenizer.pt`源/本地/目的三个完整SHA已一致。完整14件配套资产已就位（原13件＋ActionCodec），原配置仅作档案，含robo绝对路径；后续须显式覆盖模型、processor、codec路径，不能原样开训。高层B-final不是本轮低层测速依赖，未迁移。还没有神经前向、梯度或batch吞吐结果。

12:31（北京时间）配套资产13件已完成源/目的逐文件SHA验收：A4五份配置/统计/回执＋Qwen3.5 processor八件；本地`assets/`、共享`models/memlite-a4-20260912/`及`models/qwen3_5_2b_base_processor/`。CPU内存解析单帧配置通过，但未做模型恢复/前向。加载器还需独立ActionCodec（506,886,775B，SHA `5088f64a5452a60bbc8cac90ee7d79c156f1d14540bc3139aa7060f862dddace`），已开始同步；该分支在纯FM执行中不生成动作token，不因此改训练目标。主A4同步仍进行，0GPU更新。

12:15（北京时间）单连接慢，转为固定`7960bb0`的校验式8连接/16MiB分块中转，PID193613、989块；旧rsync315MB partial保留。本地状态`weights/sync-status.json`，12:14:33累计503,316,480B/33.7s。脚本在完整SHA通过后才上传，远端再次完整SHA通过才发布最终文件；测试3项/7组边界、py_compile/diff通过，0GPU启动。源CPU mmap证1138模型条目/192 LoRA、全FP32；完整checkpoint保留optimizer/RNG，不改变数值精度。

12:04（北京时间）旧A4完整权重同步准备/传输中：源`robo:/mnt/sdc1/robodojo/behavior_dev/overnight_a4_20260912/formal/checkpoints/step_2500.pt`，本地可续传暂存`artifacts/a800-memlite-oneframe-bench-20260930/weights/A4-step2500.pt.partial`，计划目的`lc3:/data/workspace/wsy/behavior2026/models/memlite-a4-20260912/step_2500.pt`。robo不能直连lc3，使用本地受控中转；完整大小/SHA验收前不放行。共享盘余约4TiB、本地600GiB，原文件不移动/删除；无需G0.5替代。

11:55（北京时间）只读确认 lc3 八卡均0MiB/0计算进程，复用现有VPN连接，无重认证。小processor资产已同步，tokenizer.json与tokenizer_config.json双端SHA一致；完整资产清单验收待。尚未启动GPU训练。
