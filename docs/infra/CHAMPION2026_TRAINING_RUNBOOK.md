# 冠军权重 → 2026：训练链与操作边界

**2026-10-12用户已明确授权冠军正式大训练，MEM-Lite继续推进。**
`configs/champion2026/shared100_v1.json` 已记录授权，沿此前讨论配方执行：
lc1八A800、global256/stride16、最多8000更新或首次启动起24h。
新提交必须push并创建独立冻结worktree；先等原MEM餐盘完整回合正常退出、
核实八卡空闲后启动，不抢占队友、不修改活跃源码。真实启动/完成状态以计划为准。

## 数学与数据

- 父权重：公开 shared50 intermediate，revision `114902410a6439ad1a0b58a112b84159d3ab3f65`，不是最终四专家之一的 cp2。
- 保留原 50 个 task / 596 个 stage embedding 行；扩为 100 task / 1087 stage，其他原参数完整继承。真实动作 23 维、模型补齐为 32 维，不能沿用 G0.5 的 27 维填充位置。
- 作者原生 FM + 0.05 FAST CE + 0.1 stage CE，4 个 flow 噪声采样，不冻结视觉/语言参数，不开启 KI。保持原归一化、FAST codec、相关噪声、30 步动作目标。
- 原 TRAIN 准入 18,895 演示、stride16 共 12,443,010 原生锚。固定每视频原 phase；105 TRAIN 与 6 eval 整源隔离继续有效。MEM 技能边界不截断冠军原生动作目标。
- 每全局 batch256 都有全部 100 task，各 2 或 3 个不同锚；8,000 批共 2,048,000 唯一样本，各任务 20,480。按任务无放回，预取不能推进优化器游标；恢复后下一批身份 SHA 必须相同。
- AdamW、梯度 clip1、EMA0.99。旧参数 peak LR2.5e-5，新追加行在 **Adam 更新之后**乘4；warmup300、cosine 到8000步时的10%。这不是把梯度乘4，也不是完整过一遍数据。

## lc1 单节点八卡运行

共用根 `/data/workspace/wsy/behavior2026`，独立环境
`envs/champion-openpi-py311-v1`；不要更改 shared env、队友 lc2 或重新登录 VPN。
每次核实八卡确实空闲，固定源码及登记补丁。激活后必须显式增加确定性 flag：

```bash
source scripts/infra/activate_champion_a800.sh
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTHONPATH="$PWD/src"
export XLA_FLAGS="$XLA_FLAGS --xla_gpu_exclude_nondeterministic_ops=true"
```

入口为 `scripts/train/train_champion2026.py`；必填 `--recipe`（同一冻结源码里的
上述 JSON）、`--output`（新 run）及 `--credential-file`（现有 0600 私密文件，
绝不把内容放入命令/日志/Git）。W&B 固定已有 team 的 `behavior2026-g05`，仅 online。
`--engineering` 是另外一个 3 步/30 分钟验收模式，产物明确不可部署；
不把它当作长训练开关或绕过批准的方法。

24 小时从首次启动起计，**包含断线/暂停间隔**；恢复不会重置该时间窗。
每1000步及正常退出前保存。入口在加载模型前自动检查整个剩余保存计划：
实测全状态54.49GB，保守按56GiB/份；8000步保留8份，加1份任意步紧急保存、
15GiB EMA和256GiB余量，新run至少需要775GiB空闲。恢复只计算余下保存，
已有文件继续占实际磁盘；每次保存还有独立实时空间检查。系统不自动删除旧 checkpoint。
SIGINT/SIGTERM 只请求停止，当前更新结束后完整保存；不要第二次强杀保存进程。

## 保存、恢复、评估、导出

- `checkpoints/step_XXXXXXXX`：参数、Adam、EMA、相关噪声等 Intermediate、随机键、step/weak_type、采样游标及冻结运行合同。逐叶 SHA、实际 HOST→各物理分片回读都检查。只有 manifest 完成并原子提交的目录可恢复；`.incomplete-*`保留现场但不可恢复。
- `latest-checkpoint.json`：当前 run 的已提交恢复点。`--resume-latest` 仅恢复同代码/数据/环境/配方、同 W&B run；不得从另一个实验冒充继续。非有限更新等失败需诊断，入口不自动 retry。
- `events.jsonl`、`sampling-journal.jsonl`：记录真实完成步数、耗时、loss、批身份和 task 计数。云端 append history step 与优化器 step 分开，避免恢复旧 checkpoint 时云端静默丢日志。
- 固定原 eval：100 task 各16、共1600，旧/新50各800。每1000步、初始和末尾做 EMA forward，时间/停止条件不足时不伪造结果。无 augmentation，独立固定噪声；stage CE 单独计算，因为作者 `train=False` 原接口漏返回该项。逐任务与两组 FM/FAST/stage/总损失分开报告；不把它当作 Q/SR。
- `exports/ema_XXXXXXXX`：作者 Orbax `params/` + `assets/`，附 task-layout 与 manifest；每个 EMA 数组匹配其完整 checkpoint，导出后用作者 `restore_params` 逐字节回读。推理仍会按作者路径转换 BF16，此格式验收不等于物理控制成功。

### 不打断训练的已保存 EMA 导出

`scripts/infra/export_champion_saved_ema.py` 在新冻结源码/独立输出目录执行，设置
`JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES=''`，不恢复优化器、不读取活跃模型、不占GPU。
参数为 `--root`、完整 `--checkpoint`、已验签 `--manifest-sha256`、新的 `--output`；
默认保留775GiB，不能抢占其他正在保存作业的预留。只允许正式、已原子发布的checkpoint，
校验step/全叶账本/原100任务映射、全部EMA字节和对应模型shape/dtype，原资产严格验签。
复用原作者加载回读的 `export_ema`，不另造权重格式；NNX容器仅用于导出纯参数字典，
包括correlation等Intermediate值，不可用作训练恢复。该入口不修改原run/源码/checkpoint，
导出完成也不代表官方协议、24GB显卡、容器或物理成功率已通过。

## 尚未通过就不能写成“已就绪”

2026-10-12 已证：全部8000批CPU采样、W&B同run在线恢复、八卡默认scatter
非确定性与确定性微核、整模型同进程和新进程/新engine恢复后下一步全部302叶精确一致、
真实原留出1600条EMA前向有限且全覆盖、全模型工程EMA作者loader逐数组回读，60项CPU测试。
证据见[工程验收回执](results/2026-10-12-champion-training-chain.json)。这些不是正式训练效果。

新推理入口支持`formal_ema_v2`：完整文件SHA账本、100任务映射、训练/推理模型配置一致，
拒绝工程checkpoint和未登记参数。旧公开父权重路径有兼容回归。
2026-10-12晚，正式1000更新EMA v2已在CPU完成全部75数组作者回读、
7个原签名资产和100任务映射验签（`results/2026-10-12-champion-ema1000-export.json`）。
原来只有工程EMA v1实测的状态已被这一新证据覆盖；仍不能称v2物理验收已过。
还需：新正式权重的2026物理/提交容器验收、成员独立审查；用户已批准本轮大训。
正式CLI＋在线W&B现已实际超过1000步，首全保存与固定1600 EMA评估完成；
8000步/24h整轮终态仍待，不能把先前小模型与恢复测试冒充本轮已训练完成。
公开 cp2 的首个完整 DEV radio 回合仍是 Q0；不以工程通过掩盖这个效果问题。

## 2026-10-12 14:15 CST：提交接口复核（尚未实现/验收部署）

[当前官方提交说明](https://behavior.stanford.edu/challenge/submission.html)允许不足1000条的
部分提交，但缺项计零；禁止挑最好rollout。原JSON/MP4不能修改，字幕片只能是另外的
展示副本。须附wrapper、实际robot config、完整运行说明，Docker方案须实证单24GB GPU；
IP方案须至少50端口。现有`package_submission.py`是旧MEM全1000条专用草案器，不能冒充
冠军封包或自动放宽成成功筛选器；截止前部分提交须另登记覆盖范围与零分分母。

官网写v3.9.3；公开Git当前没有无后缀同名tag，实际最新公开
[v3.9.3-post2](https://github.com/StanfordVL/BEHAVIOR-1K/releases/tag/v3.9.3-post2)
指向a8247a8，与当前封存sim代码一致。不能凭旧tag名称改动活跃模拟器；最终提交写完整commit。
10383只读检查八卡空闲、Docker server24.0.9可访问，**未创建容器/未改环境/未启动模拟器**。

当前`serve_champion_episode.py`与`champion_episode_client.py`是有identity/token/真实执行ACK
的内部诊断协议，不是官方直接可连的服务。已读封存官方`network_utils.py`：标准客户端
先GET `/healthz`，连接后收metadata；`reset`只发送、不读取ACK；观测请求期待`action`，
可协商`__action_chunk_size__=20`并要求`action_chunk[...,0,:]`与`action`完全一致。
部署适配时每连接/每reset独立stage、RNG、inpainting；标准协议没有物理ACK，
不能捏造`simulator_apply_ack=True`，须显式区分已发送控制与已物理确认控制。
仍需完整协议实连、跨任务交错/重置、24GB和真实新EMA物理验收；当前无提交授权、未上传模型或视频。

## 2026-10-12 15:09 CST：标准协议桥实现（真实模型验收待）

`scripts/eval/serve_champion_official.py`是新的独立loopback工程入口，读取同一已验签父权重/EMA，
不替换已有诊断服务。`behavior_champion/official.py`消费官方msgpack，`reset`不回包；支持
逐控请求或固定20步chunk，其第0动作完全相同。每连接仅1环境：标准观测没有稳定环境ID，
故拒绝可能缩批/换槽的向量batch，不猜测哪个slot属于上一回合。task变化必须先reset；
每次reset重新创建stage投票、RNG、inpainting与动作队列，不让任务/连接共享记忆。

`ChampionSession`默认仍为真实`physical_ack`时钟；新官方模式显式`policy_delivery`，
仅在socket发送完成后计数，`physically_confirmed_controls=null`，绝不凭网络发送造物理ACK。
最后一块可能只执行一部分，reset时丢弃未用队列但不把发出20步冒称实际执行20步。

预登记`official_cp2_protocol_v1.json`只用空闲lc1 GPU3、公开cp2、两个原TRAIN观测，
`probe_champion_official.py`通过SHA锁定的原官方client代码实连，比较各任务独立与交错、
逐控与chunk20、半块reset/换任务。0优化/0仿真，全部断言通过后才记协议PASS。
本地新五项、原实控六项、资产六项通过；websockets10本机跳过原sync客户端，下一既有16环境验证。
服务还没有24GB实体GPU/容器验收，也没有新正式EMA的物理成绩；不称可提交、不开外网端口。

**15:30更新：** 原官方client在lc1通过全部6项CPU/真实socket测试；公开cp2的18次真实模型调用
完成10项逐值一致性检查（含60逐控vs3×20块、交错任务、半块reset与换task），46.91秒。
停止服务后参数before/after b1ebe3fc…102fc完全一致，0优化/0物理；小证据见
[协议验收回执](results/2026-10-12-champion-official-protocol.json)。此实测冻结3bdf24f7；后续
d793eb87只关闭自有服务/内部client的周期心跳，以避开同步库`ping_timeout=None`＋高延迟
触发的ping循环（安装原库的CPU反例已在两服务器复现）。官方client保持原60s/300s参数未改。
24GB、容器、正式新EMA及物理效果依然未验收；不能由这些协议检查推出成功率提升。

## 2026-10-12 16:04 CST：提交显存准备，非实体24GB验收

`serve_champion_official.py --profile-memory`只增加本进程GPU内存遥测：0.5秒采样NVML逐PID占用，
并在模型加载/同步推理完成后读取JAX allocator统计。日志独立`gpu-memory-samples.jsonl`，
成功关闭后`memory-profile.json`；采样错误单列，不假填零，不把其他进程占用算给模型。
采样最大值是下界，不能据A800测得低于24GiB就宣称3090/A5000/TitanRTX或Docker已通过。

本次根负责有限只读probe：沿原配置`official_cp2_protocol_v1.json`、lc1实空GPU3、原cp2和
两个TRAIN输入，复用18次真实模型调用/10项序列检查，在新冻源与新
`runs/champion_official_protocol_20261012/memory-{server,probe}-v1`目录运行；只为量实际峰值，
不是新增效果实验或正式训练。18调用完成后正常关闭绑定服务、复核权重摘要；模型/协议失败
则保留现场并停止，不扩为物理回合、不占队友GPU、不热改环境。实际结果尚待。

**16:11结果：** 本有限probe已完成18调用/10一致性检查，47.46s；253次采样无错误，
逐PID峰值8718MiB（约8.51GiB）、JAX live峰7,231,032,064B。与15:30原probe的十项
动作SHA也全部相同；权重摘要未变、服务已关闭、GPU已释放。原始小回执两端核验和
局限见[显存实测回执](results/2026-10-12-champion-serving-memory.json)。该结果支持显存有
余量的工程判断，不替代目标GPU/容器或未来正式100task模型的实测；不开新正式训练。
