---
title: "2026 BEHAVIOR Challenge：基座模型选择与 robo 准备情况深度研究"
subtitle: "面向 10 月 16 日截止期的技术决策、复现实验与交付风险报告"
author: "研究日期：2026-08-26"
date: ""
lang: zh-CN
toc: true
toc-depth: 2
---

\newpage

# 结论先行

**技术主线选择：G0.5。**在当前可公开核验的材料里，它是唯一给出过 2025 BEHAVIOR 标准赛道**直接、同任务集合**结果，并以单一 checkpoint 达到 0.3136 的开放基座；同表中的 pi0.5 为 0.2626，2025 冠军四 checkpoint 系统为 0.2605。这个证据的相关性远高于“在 LIBERO 上超过 pi0.5”的常见说法。

但 G0.5 不是无条件押注：它公开仓库尚无 BEHAVIOR/OmniGibson adapter 或 checkpoint，论文 BEHAVIOR 配置为 RGB 单帧而非今年可用的 RGBD，且公开模拟 checkpoint 的 AR+CoT 路径存在未澄清的复现疑点。**只有在第 1 周完成 adapter、500--1,000 step pilot 和闭环 smoke test，并拿到适用于本队参赛方式的书面许可后，才应把它升格为主线。**

**交付兜底：GR00T N1.7。**它是官方 2026 baseline，和 R1Pro/LeRobot 路径最接近，工程可控性最好；但没有公开的 BEHAVIOR 成绩可证明其超过 G0.5，且在 G0.5 与 LingBot 的各自 R1Pro 实验中均落后。它的价值是守住截止期，而非当前最高胜率。

**受控候选：LingBot-VLA 2.0。**它有 R1Pro、原生深度、60,000 小时预训练和较好的长程移动操控证据，许可证也更宽松；但没有 BEHAVIOR/OmniGibson 结果。把它放入相同预算、相同任务簇的 canary，而不是立即替换 G0.5。

这是一份技术判断，不是对最终排名的保证；2026 私有评测分数目前不存在，任何声称“G0.5 已在 2026 获胜”的说法都不成立。

## 最终决策表

| 角色 | 模型 | 现在应做什么 | 触发切换条件 |
|---|---|---|---|
| 技术主线（有条件） | G0.5 | 先做 R1Pro-23D/LeRobot/WebSocket adapter、短训和闭环冒烟 | adapter 或许可在第 1 周未过门槛；或相同预算被 LingBot 明显超过 |
| 交付兜底 | GR00T N1.7 | 并行保持官方 baseline 可训练、可提交 | G0.5 不可复现、显存/服务不稳定、许可不清 |
| 受控 canary | LingBot-VLA 2.0 | 跑 5--10 个代表任务的同预算对照 | 在移动、容器/家电、深度感知任务稳定胜出且部署合格 |
| 不选为主线 | Wall-OSS-0.5、ABot-M0.5、VLA-0 | 只借鉴其思想/留作后备研究 | 出现直接 BEHAVIOR/R1Pro 复现和可用实现后再评估 |

# 1. 问题边界与赛事约束

本报告的对象是 Stanford 的 [BEHAVIOR Challenge](https://behavior.stanford.edu/challenge/index.html)，截至 **2026-08-26** 的公开规则与仓库 tag **v3.9.2**。2026 赛季扩大为 100 个任务、7 个场景、约 20,000 条示范，并使用 R1Pro 全身双臂移动机器人；标准输入为单 RGB、深度和 proprioception，不能依赖语义分割或其它特权状态。[官方数据说明](https://behavior.stanford.edu/challenge/dataset.html)列出 LeRobot v3 数据；[评测规则](https://behavior.stanford.edu/challenge/evaluation.html)规定了受限观测与运行要求。

需要把三种“公开实例”严格分开：0--9 是可反复看 leaderboard 的实例；10--19 是公开但应作为 held-back test 的实例，避免被 leaderboard 调参污染；最终还有官方隐藏实例。最终策略必须在**一张 24 GB GPU**上运行。提交截止为 **2026-10-16 AoE**（北京时间为 10 月 17 日 19:59）。这些条件使“论文最高分”不足以直接回答“最适合比赛”。

本报告采用如下证据优先级：

1. 相同 BEHAVIOR 任务、相同或更严格观测的闭环成绩；
2. 相同 R1Pro/移动双臂的证据；
3. 长时程、导航、接触丰富任务的证据；
4. 24 GB 推理、公开权重与可复现实作；
5. LIBERO、RoboTwin 等异构基准。

其中第 1 项远比第 5 项重要。本文明确标注：**[事实]**为来源直接陈述，**[robo 观察]**为服务器只读审计，**[推断]**为基于事实的工程判断，**[建议]**为行动选择。

# 2. 为什么“爆杀 pi0.5”不能直接外推

pi0.5 是很好的历史参照，却不是全行业统一、可跨基准排序的标尺。大量新论文以 LIBERO 或 RoboTwin 为主实验：前者是短程、固定工作台任务，许多报告已经在 97--99% 区间；后者也主要评价受控的桌面双臂操作。相比之下，BEHAVIOR 的任务平均可长达数分钟，包含导航、移动底座、全身动作、取放顺序、柜门/抽屉/透明容器、状态变化和失败恢复。

因此，一篇工作即使在 LIBERO 比 pi0.5 高 1--10 个点，也不自动说明它能：

- 把 R1Pro 的 23 维动作、61 维 proprioception 与自身 canonical action/state 正确对齐；
- 在不使用分割图的前提下，跨房间维持数分钟任务状态；
- 适应 OmniGibson 的相机、接触动力学与 WebSocket 服务协议；
- 在 24 GB GPU 内完成稳定推理；
- 在 100 任务混合微调后不发生灾难性遗忘。

这不是否定那些论文，而是避免“基准转移谬误”。G0.5 之所以优先，靠的不是它在某个通用榜单第一，而是它已经跨过了最关键的一道相同任务验证。

# 3. 直接证据：G0.5 对 2025 BEHAVIOR 的意义与边界

G0.5 是 Galaxea 的自回归 VLA，基于 Qwen3.5 2B VLM 初始化，预训练覆盖 14 类机器人形态，并使用 27 维 canonical ActionCodec。模型同时生成语言化推理 token 与动作 token，可利用 Subtask、BBox、Trace、ActionHint 等 chain-of-thought（CoT）信号，并在预训练阶段使用约 5 秒的视觉记忆。[G0.5 论文](https://arxiv.org/abs/2608.11739)、[代码](https://github.com/OpenGalaxea/GalaxeaVLA)和[模型卡](https://huggingface.co/OpenGalaxea/G05)是本节的一手来源。

论文的 2025 BEHAVIOR 表格给出的数值如下（50 任务，每任务 10 个实例，两次评测均值）：

| 系统 | 训练/集成方式 | 分数 |
|---|---|---:|
| G0.5 | 1 epoch，单一 checkpoint | 0.2904 |
| G0.5 | 4 epochs，单一 checkpoint | **0.3136** |
| pi0.5 | 4 epochs，单一 checkpoint | 0.2626 |
| 2025 冠军 RLC | 4 个任务专用 checkpoint | 0.2605 |

按论文表格计算，0.3136 相比 pi0.5 的 0.2626 是约 **19.4% 相对提升**，相比冠军系统是约 **20.4% 相对提升**。这不是 2026 分数，也不能证明在 100 任务新增分布中仍有同样优势；但它是当前公开候选中最有决策价值的对应证据。

论文的直接 BEHAVIOR 对比使用官方 2025 standard track、默认低分辨率 RGB、单帧观测，并且**没有**移植冠军的显式 stage head；因此它是一个相当干净的单基座比较，但仍不是今年 RGBD、100 任务的最终设置。它表明 RGB-only 首版既合规又有直接先例，却不表明接入 depth 一定提升。

论文还报告 G0.5 在 50 个任务中领先 pi0.5 的 29 个、后者领先 15 个、约 6 个相近。它在“搬箱入储物处”“清理垃圾”“整理汽车”等导航+抓取放置任务有较大优势；但在容器/家电类任务不稳定，例如 microwave popcorn 的 pi0.5 更高。作者也承认短视觉记忆和半透明柜体/抽屉插入是弱点。这恰好说明：G0.5 是值得押注的基座，但需要有针对性的长程任务记忆和容器/家电数据策略。

## G0.5 的公开缺口：必须作为 go/no-go 条件

1. **没有公开 BEHAVIOR adapter/checkpoint。**仓库的 QUICK_START 支持 bridge、droid、libero、r1lite、r1pro、r1pro_wbc、robotwin、so100，没有 BEHAVIOR 或 OmniGibson 配置；模型卡列出的 checkpoint 也不含 BEHAVIOR。论文结果不能直接下载复现。
2. **公开模拟 checkpoint 的 AR+CoT 可复现性不清。**[Issue #57](https://github.com/OpenGalaxea/GalaxeaVLA/issues/57)指出发布的 LIBERO/RoboTwin checkpoint 看似禁用了 AR+CoT、仅使用 FM，询问论文结果与公开权重是否一致；截至研究日仍未见项目方正式澄清。不能把“论文里的 AR+CoT 故事”当作已验证可用的公开功能。
3. **论文 BEHAVIOR 配置是 RGB 单帧。**它不违背 2026（只用 RGB 是允许的），却没有利用今年可用的深度；不能假设深度一接入便提升。
4. **训练与部署资源不对称。**公开说明给出 checkpoint 约 11 GB、推理需大于 8 GB，故 24 GB 推理有余量；完整微调通常需大于 70 GB/GPU，推荐 A100 80 GB/H20 96 GB。最终推理约束通过，不代表本地可直接 full tune。
5. **许可不是默认安全。**G0.5 使用 [G0.5 Community License](https://github.com/OpenGalaxea/GalaxeaVLA/blob/main/LICENSE-G0.5)，而非 Apache-2.0。见第 8 节的决策树。

# 4. 候选横向比较：证据等级，而非虚假的总分

| 候选 | 与 BEHAVIOR 的证据 | R1Pro / 长程契合 | 交付性 | 本报告判断 |
|---|---|---|---|---|
| **G0.5** | **A：论文直接给 2025 BEHAVIOR 分数** | A-：R1Pro canonical 布局、移动操控强；长记忆仍短 | C+：无 adapter、许可需确认 | 技术主线，过 gate 后投入 |
| **GR00T N1.7** | B：官方 2026 baseline，未见公开 BEHAVIOR score | B：通用 canonical state/action、官方路径 | A-：官方集成，推理 16 GB+ | 交付兜底 |
| **LingBot-VLA 2.0** | C：没有 BEHAVIOR 结果 | A-：R1Pro、深度、长程移动证据 | B：Apache-2.0，仍需 adapter | 同预算 canary |
| Wall-OSS-0.5 | C：无 BEHAVIOR/R1Pro | C+：20+形态但无住宅移动证据 | B：Apache-2.0、LeRobot | 备选，不抢主线 |
| ABot-M0.5 | C：无 BEHAVIOR/R1Pro | B：移动+操作设计有吸引力 | C：重型、公开评测主要非本赛道 | 借鉴架构，不作为交付基座 |
| VLA-0 | C：无 BEHAVIOR/R1Pro | D：桌面操控为主 | B：开源 | 不适合当前任务 |
| pi0.7 | 无公开可用权重 | 未可验证 | D：不可取得 | 排除 |

证据等级含义：A 为同任务闭环量化结果；B 为官方集成或高度同构平台；C 为相关但异构基准；D 为不可取得或无法验证。本表不把不同论文的数字硬加成一个“综合分”，避免制造精确幻觉。

## 4.1 G0.5：为什么排第一

除直接 BEHAVIOR 成绩外，G0.5 的 27D canonical 结构把左右控制各分为 9D+gripper 1D，lower-body 为 7D。BEHAVIOR R1Pro 的活跃动作是 23D：base 3 + torso 4 + 左臂 7 + 左夹爪 1 + 右臂 7 + 右夹爪 1。**[推断]**可将每条 7D 手臂嵌入 9D 控制槽并补两个未用维，lower-body 对应 base+torso；输出时反向投影回精确 23D。映射很干净，却仍必须用闭环 smoke test 验证单位、顺序、四元数/关节定义和 action normalization，绝不能仅凭维数相同就认定正确。

论文对 R1Lite/R1Pro 的实机对照给出 G0.5 76.7、pi0.5 53.3、GR00T N1.7 24.4（同作者协议）。这一结果有价值但不能替代独立评测；它支持“移动双臂先验可能有迁移价值”，不应被写成跨论文的绝对排行榜。

## 4.2 LingBot-VLA 2.0：最值得做 canary 的替代

[LingBot-VLA 2.0 论文](https://arxiv.org/abs/2607.06403)与[代码](https://github.com/Robbyant/lingbot-vla-v2)报告 6B 模型、Qwen3-VL-4B-Instruct 骨干、约 60,000 小时预训练（50,000 小时机器人数据、10,000 小时第一视角数据）、原生深度路径与 55D canonical schema。论文列出 R1Pro；其 GM-100 九任务上，成功率为 LingBot 2.0 15.6%、pi0.5 8.9%、GR00T N1.7 5.6%。在其它平台的冰箱整理、灶台清洁等长程移动任务也超过 pi0.5。

它的弱点是证据没有落在 BEHAVIOR/OmniGibson，上述 R1Pro 结果是九个双臂/桌面任务，长程证据也不是 R1Pro。它在 4090D 上报告约 130 ms/动作块，但与 24 GB 官方环境、服务协议、100 任务混训的兼容性仍待实测。Apache-2.0 是其许可优势。

## 4.3 GR00T N1.7：最可靠的工程兜底

[NVIDIA Isaac GR00T](https://github.com/NVIDIA/Isaac-GR00T)的 [GR00T N1.7 3B 模型卡](https://huggingface.co/nvidia/GR00T-N1.7-3B)和 [BEHAVIOR baseline 文档](https://behavior.stanford.edu/challenge/baselines.html)提供了最直接的工程保证：它是官方赛道 baseline，使用通用 action/state 表示，公开建议推理 16 GB+、微调 40 GB+。代码为 Apache-2.0，但权重及其依赖还有 NVIDIA/ gated 的使用条件，仍需走账号与条款流程。

它应被保持为可随时训练和提交的并行线。G0.5 作者的实机数值、以及 LingBot 自己的 R1Pro 数值，都显示 GR00T 并非目前最高潜力；这些都不是独立 head-to-head BEHAVIOR 比赛，故结论是“兜底”而非“较差”。

## 4.4 其它候选为什么不抢主线

- [Wall-OSS-0.5](https://arxiv.org/abs/2605.30877)在自己的多形态实机设置中报告超过 pi0.5，且有 Apache-2.0 与 LeRobot 集成；但无 BEHAVIOR、R1Pro 或住宅级长程证据。
- [ABot-M0.5](https://arxiv.org/abs/2607.00678)将移动与操作拆解、用 latent actions/Dream Forcing，应对移动操控很有启发；公开实现和 checkpoint 的主覆盖为 LIBERO/RoboCasa，24 GB 单卡与 R1Pro/OmniGibson 均未验证。
- [VLA-0](https://arxiv.org/abs/2510.13054)在 LIBERO 上有很强结果，却主要是动作文本化的桌面基准证据，缺少全身移动、R1Pro、住宅场景与 BEHAVIOR 适配。
- Physical Intelligence 的公开 [openpi](https://github.com/Physical-Intelligence/openpi)截至研究日公开支持 pi0、pi0-FAST、pi0.5；关于 pi0.7 的[公开问题](https://github.com/Physical-Intelligence/openpi/issues/980)不能替代可下载权重。因此把 pi0.7 列作“今年最佳基座”没有操作意义。

# 5. 2025 冠军真正值得继承的部分

[2025 冠军代码与说明](https://github.com/IliaLarchenko/behavior-1k-solution)基于 pi0.5/openpi，提交分数约 26%。它做了 50 个任务 embedding、correlated flow-matching noise、从 VLM 多层向 action expert 注入特征、stage/system-2 特征、多步 flow matching、delta action、按时间戳归一化、FAST/subtask 辅助损失。推理端有 30 步预测执行 26 步的 soft inpainting、三次样条平滑、抓取失败纠正、任务分组的四个 checkpoint。作者的通用中间 checkpoint 未正式评测，只猜测约 15--20 Q。

应当移植的是**系统思想**，而不是把 pi0.5 的内部替换进 G0.5：

- **显式阶段记忆。**G0.5 的视觉记忆只有秒级；对分钟级 BEHAVIOR，在允许的 RGBD、proprioception、语言指令和 BDDL 任务结构上维护 episode-level stage state。模型自己不应获得额外特权状态。
- **失败恢复。**用失败抓取、物体遗失、门未开等可观测信号触发安全重试/重新观测；所有规则须经 held-back 实例验证。
- **动作后处理。**只在正确的动作坐标/频率/边界裁剪验证后使用平滑、短块执行和 soft-inpainting；错误的滤波可掩盖 adapter bug。
- **后期再做专家化。**先有单一 100-task generalist；仅在 10--19 held-back 证实稳定任务簇差异后，才引入少数 adapter/route（例如容器/家电），并保持最终服务单 24 GB GPU 可运行。

# 6. 推荐的系统设计与实验协议

## 6.1 最小可行的 G0.5 集成

官方 RGB / depth / 61D proprio 与任务指令进入系统后，RGB 先走 G0.5 原生视觉路径（首版不强接 depth）；depth 仅进入合规的短期几何/状态记忆并做消融；61D state 则经 selector 与 normalization。随后经由 **BEHAVIOR R1Pro 23D ↔ G0.5 canonical 27D** adapter，进入 G0.5 generalist、显式阶段记忆和已验证的恢复/动作安全层，最后由 WebSocket policy server 接至官方 v3.9.2 evaluator。

接口原则：多视角 RGB 依 G0.5 输入规范预处理；从 61D proprio 选择/拼出 canonical 所需状态；对 23D 动作严格建立双向映射和单元测试。每次输出都限制在训练数据统计范围内，记录 task、stage、原始/映射后 action、延迟与 reset 原因。**深度不应在第一个训练版本被强行拼入 VLM token**：先复现 RGB-only 直接证据，再以相同 seed、相同预算做深度记忆/adapter 消融，才知道它是信息还是噪声。

## 6.2 训练与数据策略

1. 把 2025 的 50 任务与新增 50 任务均衡采样，建立全 100 任务 generalist；不要按公开 0--9 榜单反复调参。
2. 从语言标注和 BDDL 生成**仅训练期**的子任务/阶段标签，保留它们作辅助监督或 memory target；线上只由允许观测更新状态。
3. 对 G0.5 已知薄弱簇过采样：打开/关闭、柜体/抽屉、半透明容器、插入/接触、烹饪/家电。新增任务中的整理、清洁、安装与容器交互很可能放大这些短板（**推断**，非已测结果）。
4. 先做 SFT；若已有稳定 SFT 闭环，再利用 BDDL 部分进度或阶段奖励探索 RL。G0.5 自回归 token 提供明确的 log probability，理论上比 flow-only path 更适合 GRPO；这仍是研究假设，不能拿 RL 代替基础 adapter 验证。
5. 所有模型选择以 10--19 held-back public instances 为准；0--9 只做最终 sanity check。最终隐藏实例前冻结 checkpoint、路由和恢复规则。

## 6.3 同预算 canary：G0.5 vs LingBot 2.0

在相同图像、示范子集、训练步数、种子数、动作频率和单 24 GB 推理预算下，选 5--10 个任务覆盖：

- 导航+搬运；
- 柜门/抽屉/容器；
- 烹饪/家电状态变化；
- 清洁/工具接触；
- 安装/布置的多阶段任务。

指标依次是：adapter 单测通过率、服务 P95 延迟、无崩溃 rollout 比率、官方 Q、阶段成功率、每次成功的推理成本。**切换条件**建议预先写死：若 LingBot 在 held-back 的平均 Q 提升至少 0.03，且所有部署门槛通过、没有显著更高 OOM/timeout，才投入更多预算；否则不因单个精彩 demo 改主线。

# 7. 从今天到截止日的 5 周路线图

| 周期 | 交付物 | Go / no-go 门槛 |
|---|---|---|
| 8/26--9/1（第 1 周） | G0.5 adapter、数据读取、1000-step pilot、5 任务闭环 smoke；GR00T 官方线可运行 | G0.5 完成 23D/27D 单测、无 NaN、服务稳定；许可问题有书面答复或明确保守路径 |
| 9/2--9/8（第 2 周） | 全 100 任务小规模 generalist；held-back 评价；LingBot canary | 主线必须显著优于其随机/未训基线且无协议错误；否则降级 G0.5、增加 GR00T 权重 |
| 9/9--9/22（第 3--4 周） | 主训练、阶段记忆/恢复与深度消融 | 只保留在 10--19 可重复提升的改动；容器/家电若显著短板才加 adapter |
| 9/23--10/6（第 5--6 周） | checkpoint 冻结候选、压力/长回合测试、Docker/service 验收 | 单 24 GB、冷启动、长时间稳定、无私有观测依赖 |
| 10/7--10/16（提交窗口） | 正式包、复跑、提交和备份 | 至少有 G0.5 主线和 GR00T 兜底各一套可复现、可提交产物 |

最重要的防线是**早设 kill switch**：第 1 周 G0.5 没有闭环，或第 2 周不能从 held-back 看出可信趋势，就停止把时间花在论文复刻细节上，转投官方 GR00T 路线。比赛剩余时间比“坚持某篇 SOTA”更稀缺。

# 8. G0.5 许可：行动决策树（非法律意见）

G0.5 的 Community License 允许部分研究、个人、教育和评估用途，但限制商业、对外服务、专利及特定分发方式。模型卡又要求经过其自定义/gated 的获取流程。仅靠“开源”二字，无法消除比赛奖金、参赛主体、Docker 交付和权重分发的歧义。特别要区分：评测方离线运行的 Docker、仅内网/本机的 policy server，和暴露为公网 IP 的 hosted/API service 不是同一风险等级；后者不应在未获书面许可前启用。赛事的 Docker/IP 提交通道（IP 方案还需满足官方端口要求）只说明 Stanford 接受何种技术交付，不会自动覆盖 Galaxea 的权利边界。

先问：队伍是否为非营利学术/个人研究，且用途仅为参赛评估？若答案是否或不确定，先向 Galaxea 书面申请竞赛许可，同时走 GR00T 兜底；若是，则继续问是否含现金奖、赞助商商业使用、公司雇佣关系或后续产品化。只要这一层存在不确定，就视为许可风险并要求书面确认；若没有，则按模型卡 gated 流程取得权重并保留条款版本与同意记录。最后问 Docker 是否会向赛事方交付权重或供外部托管运行：若会，明确询问交付、再分发与外部运行是否被允许；若不会，仍要核验第三方依赖与 challenge Docker 规则。

应同时给 Galaxea 与 Stanford 发出简短、可留档的问题：

1. 非营利学术队伍使用 G0.5 微调权重参加 BEHAVIOR 2026 是否受 Community License 允许？奖金是否改变结论？
2. 仅供评测方离线运行的 Docker，是否构成 prohibited external service、hosting 或 redistribution？
3. 是否允许提交微调后的权重，或只允许容器内推理？若评测必须经公网 IP 或外部 service 接入，是否被允许？需要何种 notices？
4. Stanford 的获奖、开源奖项/代码公开要求和 Docker 保存期限，是否会触发模型权重再分发？

在未收到书面确认前，本报告不作法律结论，也不建议把 G0.5 作为唯一可提交路径。保留模型卡、许可证 commit SHA、下载授权截图/邮件和最终 Docker 的 SBOM。

# 9. robo 服务器准备情况：只读审计

本节是对队友已完成工作的核对，不对 `robo` 执行任何写入或重启操作。证据来自 2026-08-26 的只读审计；路径、日志与硬件信息已去除令牌、用户名以外的敏感连接信息。下面的“PASS”仅说明对应的离线步骤通过，**绝不等于** BEHAVIOR 官方闭环已经通过。

## 9.1 已具备的基础条件

**[robo 观察，2026-08-26，只读]**主机 `llmvideo26`（用户 `robodojo`）是 4 × A100 80 GB PCIe、96 CPU core、866 GiB RAM（约 847 GiB 可用）的训练机；`/mnt/sdc1` 为 8.0 TB，已用 4.4 TB、余约 3.7 TB。操作系统为 Ubuntu 24.04.4，驱动 580.173.02。快照时 GPU0 被 Pi0.5 policy service 占用约 41,022 MiB，GPU1--3 空闲。没有运行中的 G0.5，端口 8765 未监听，`tmux` 的 `g05-server`/`g05-setup` 只是 idle bash。**Docker 未安装**，因此当前尚无挑战赛可交付镜像。

G0.5 主工作树为 `/mnt/sdc1/robodojo/GalaxeaVLA`，upstream `main` HEAD 为 `89f2322`（2026-08-13）。其中已有 **6 个已跟踪修改和 17 个未跟踪的 BEHAVIOR 文件，均未提交**。`BEHAVIOR2026_START_HERE.md` 记载了预定拓扑：A100 主机做 G0.5 service、RTX 主机 `kexin8` 跑 OmniGibson evaluator，并从 Windows 经过双跳 SSH 隧道连通。`.env.g05-a100` 存在但审计没有读取，避免泄露凭据。

项目 `.venv` 为 Python 3.10.18、PyTorch 2.7.1+cu128、CUDA 12.8、transformers 4.57.1、Hydra 1.3.2、websockets 16.0；G0.5 import 成功。系统没有 `uv` CLI。基础权重约 11.44 GB，ActionTokenizer 约 506.9 MB。数据软链视图为 `/mnt/sdc1/robodojo/datasets/behavior2026_g05`，实际数据位于 `/mnt/sdc1/xhz/BEHAVIOR2026/2026-challenge-demos`：100 tasks、20,000 episodes、210,916,774 frames、30 FPS、3 RGB、23D action、61D state。

| 审计项 | 已核验的结论 | 复现/验收动作 |
|---|---|---|
| GPU/存储 | 4 × A100 80 GB；训练余量充足，但最终仍必须按单 24 GB GPU验收 | 保存 GPU/驱动、CUDA/PyTorch 与磁盘快照 |
| 仓库与分支 | `GalaxeaVLA@89f2322` + 6 modified/17 untracked | 先保存 `git diff`、未跟踪文件、commit 与 manifest；再决定提交到私有分支 |
| Python 环境 | `.venv` 可 import G0.5 | 导出可重建的锁定文件，不读取/打包 `.env.g05-a100` |
| 数据与权重 | 100 任务数据与 G0.5 基础权重可读取 | 建立 hash/manifest；不把全量数据或受限权重混入镜像 |
| 服务交付 | `serve_behavior_policy.py` 已有但未启动验证；Docker 缺失 | 增加镜像/entrypoint，执行单卡冷启动和长回合验收 |

## 9.2 已经做了哪些 G0.5 接入工作

**[robo 观察，2026-08-26，只读]**已存在以下本地 BEHAVIOR 适配：

- `configs/data/behavior_r1pro.yaml` 覆盖全 100 任务；`configs/data/behavior5_r1pro.yaml` 是 5 任务缩小集。
- 三路 RGB 输入为头部 720²、两路腕部各 480²；运行时 processor 变成 `(1, 3, 256, 256)`。
- 官方 61D proprio 被选择为 base、双臂、双夹爪和 trunk，转换为 G0.5 27D state；官方 23D action（base 3 + trunk 4 + 左/右 arm 各 7 + 左/右 gripper 各 1）映射至 canonical 27D，并实现逆映射。
- `scripts/serve_behavior_policy.py` 已实现官方风格 WebSocket + msgpack、`/healthz`、默认 `127.0.0.1:8765`，action chunk 为 16。
- 本地修复包括：lower-body merger 不再丢字段；A100 的 SM80 使用非 Hopper-only FA4 backward；ActionTokenizer 的 OmegaConf interpolation；msgpack ndarray 的 byte keys。

这已经是实质性接入，而不只是下载模型；但由于这些代码未提交，必须把它们视为当前最重要的可复现性风险。

## 9.3 “部分训练已经跑通”究竟意味着什么

**[robo 观察，2026-08-26，只读]**五任务数据集位于 `/mnt/sdc1/robodojo/datasets/behavior2026_g05_tasks_0_4`，含 1,000 episodes、9,349,743 frames，任务为 `turning_on_radio`、`picking_up_trash`、`putting_away_Halloween_decorations`、`cleaning_up_plates_and_food`、`can_meat`。已完成 20-step smoke（多次）以及 batch 8/16/24 smoke。主 pilot 输出：

`/mnt/sdc1/robodojo/outputs/g05/behavior5_r1pro/behavior5_bs8_ckpt_pilot1000/last.pt`（指向 `step_1000.pt`）。

该 checkpoint 为 11,440,526,683 bytes，step=1,000，含 946 tensors 和 3,376,515,931 个 parameter values，sidecar 完整。CPU 的 `tools/check_behavior_checkpoint.py` 报告 `REQUIRED_SIDECARS=PASS`、`CUDA_VISIBLE_DEVICE_COUNT=0`、`G05_CHECKPOINT_INTEGRITY=PASS`。`tools/check_behavior_processor.py` 报告：3 × `(1,3,256,256)` RGB、61D→`(1,27)` proprio、action horizon `(32,27)`、postprocess→`(23,)`，`BEHAVIOR_G05_PROCESSOR_CHECK=PASS`。

这些结果能证明：数据至少能够被读取，batch 能经过 processor/模型，checkpoint 能保存/加载，训练循环没有在 1,000 step 前立即失败。对于 adapter 开发，这是很重要的里程碑。

但它**不能证明**：动作 23D↔27D 的物理语义正确、模型在全 100 任务收敛、policy server 能与 v3.9.2 evaluator 正确通讯、在完整回合不崩溃、官方 Q 有正增益，或满足最终单卡 Docker 规则。审计有两项关键 warning：YAML 写 224 的 image shape，但运行时 `camera_size_config` 覆盖为 256；`dataset_stats` 缺少 raw `base_qvel`/`trunk_qpos` 的 action/state 统计，当前走 dummy normalization fallback。前者是配置一致性问题，后者是数值尺度风险；两者都比 shape 通过更优先处理。

服务协议测试尚未执行（该测试会启动服务），也未做 G0.5 的 GPU 推理或 OmniGibson rollout。全 100 task config/stats 存在，但 `/mnt/sdc1/robodojo/outputs/g05/behavior_r1pro/test` 下没有全量 checkpoint。结论必须写为：**已跑通 5 任务的离线 processor/checkpoint/短训通路；尚未跑通 100 任务训练，更未证明 BEHAVIOR 闭环成功。**

审计还发现一个独立的 **Pi0.5 Flexiv 20,000-step** 训练活动/日志证据。实验简称为 “Flexiv insert-screw / 20260820”；完整实验标识为：

```text
flexiv_insert_screw_eef_3cam_
insert_screw_20260820
```

它在 8 月 22 日先跑 0→14,999，8 月 24 日 resume，8 月 25 日完成 20,000/20,000；step 19,900 的 loss 为 0.000897、grad_norm 为 0.0258（step 0 约 0.138）。最终 checkpoint 全路径为：

```text
/mnt/sdc1/robodojo/RoboDojo/XPolicyLab/policy/Pi_05/openpi/checkpoints/
pi05_flexiv_insert_screw_3cam_eef/
flexiv_insert_screw_eef_3cam_insert_screw_20260820/19999
```

参数约 12 GB + train state 约 31 GB；5k/10k/19999 三个 checkpoint 共约 125 GB。当前 `serve_policy.py` 加载 19999、端口 8000、占 GPU0 约 41 GB，但 util 为 0%。

这条 Pi0.5 数据是 Flexiv Rizon4 的单任务 “pick up the screw and insert it into the hole”：101 episodes、76,096 frames、3 RGB 480×640@30 Hz、10D EEF pose+gripper，无 validation split；模型是 32D action、horizon 50、delta EEF XYZ、bf16、batch 32、lr 5e-5、warmup 1,000、decay 30,000，使用两个 GPU 的 FSDP。命令脚本全路径为：

```text
/mnt/sdc1/robodojo/RoboDojo/XPolicyLab/policy/Pi_05/openpi/scripts/
run_flexiv_pi05_eef_3cam.sh
```

日志全路径为：

```text
/mnt/sdc1/robodojo/RoboDojo/XPolicyLab/policy/Pi_05/openpi/wandb/
run-20260824_124941-5h42j1h8/files/output.log
```

审计没有发现 8 月 20 日后的 rollout JSON/CSV/video/log；此前还发生过一次 JAX/NCCL OOM、一次 1-step smoke 成功和一次 KeyboardInterrupt。故它说明**离线训练完成**，不说明 Flexiv 闭环成功，更不是 G0.5/BEHAVIOR 成功。

## 9.4 当前缺口和优先级

1. **立即固化可复现证据。**把 6 modified/17 untracked G0.5 文件、精确 config、processor 版本、数据 manifest、权重 hash、命令行和 1,000-step log 纳入独立 experiment manifest；随后决定是否提交到私有分支。不要在未备份时清理工作树。
2. **修数值语义，再谈全训。**确认 YAML 224/运行 256 的唯一标准；补齐 `base_qvel`/`trunk_qpos` 的真实统计，禁止 dummy normalization 进入正式训练。逐通道构造已知 23D 动作，检查 27D 嵌入/反投影、normalize/denormalize、裁剪和时序方向，特别是 base/torso、左右臂、夹爪。
3. **升级为闭环最小验证。**先在 GPU1--3 之一启 G0.5 service，执行本地 protocol test；在 `kexin8` 确认 v3.9.2 evaluator 和跨机链路，五任务每个至少若干固定 seed，保存视频/轨迹/Q。若无正向信号，不进入昂贵全量训练。
4. **并行保存 GR00T N1.7 可提交路径。**机器上已有 LingBot_VLA、LingBot_VA、GR00T_N17 源码，但没有它们的 BEHAVIOR 配置、产物或专用权重证据；Hy-Embodied 0.5 有约 9.05 GB 权重、RoboTwin 权重不完整，均不能当现成 BEHAVIOR 线。不要等待 G0.5 完全修好。
5. **补 Docker。**当前没有 Docker，且 OmniGibson source/module 不在 `robo`，评测器在 `kexin8`。在代码冻结前完成无网络冷启动、单 24 GB、远程 evaluator 和长回合 smoke。

# 10. 风险登记与负责人建议

| 风险 | 影响 | 最早信号 | 处置 |
|---|---|---|---|
| G0.5 adapter 仅 shape 对齐、物理语义错 | 高 | action 抖动、base 反向、Q=0 | 单测+五任务闭环 gate |
| 公开 AR+CoT 与论文不一致 | 高 | issue 无回复、checkpoint config 冲突 | 优先稳定 FM 路径；把 AR/CoT 做消融而非前提 |
| G0.5 许可/权重交付不清 | 高 | 无书面回复 | 立刻咨询；GR00T 并行兜底 |
| 新 50 任务在柜体/家电上失分 | 中高 | held-back 分任务曲线低 | 过采样、stage memory、少量专用 adapter |
| 公开 0--9 过拟合 | 高 | leaderboard 升、10--19 降 | 预注册 split，冻结选择准则 |
| 最终 24 GB/Docker 失败 | 高 | OOM、冷启动下载、长回合泄漏 | 每周单卡镜像验收 |

建议责任分配：一人维护 G0.5 adapter/manifest；一人维护 GR00T 兜底；一人拥有 evaluator、held-back 指标和视频复盘；一人负责许可/交付清单。所有训练在同一实验台账里记录 commit、数据版本、seed、输入模态、checkpoint、Q 与资源。

# 11. 最终建议

截至 2026-08-26，最合理的技术选择不是“寻找一篇在 pi0.5 上分数最高的论文”，而是把候选放入 BEHAVIOR 的真实约束中筛选。**G0.5 是唯一应优先打通的高上限基座**：它对 2025 BEHAVIOR 的 0.3136 直接结果足以压过其余候选的异构 benchmark 优势。它的代价是 adapter、公开复现和许可风险，故一定要配 **GR00T N1.7 兜底**，并以 **LingBot-VLA 2.0 同预算 canary** 来检验深度/R1Pro 先验是否真的在 2026 任务簇上赢。

`robo` 已经跨过了“离线数据/processor/checkpoint/短训能走”的早期门槛，但尚未跨过“官方闭环成功”的门槛。接下来最有价值的不是立刻把训练拉到更多步，而是把已有五任务 1,000-step pilot 固化、验证动作语义，再用 v3.9.2 跑出可审计的 closed-loop Q。通过后再扩展到 100 任务，成功率和交付确定性都会更高。

\newpage

# 附录 A：关键一手来源

1. Stanford BEHAVIOR Challenge：[主页](https://behavior.stanford.edu/challenge/index.html)、[数据](https://behavior.stanford.edu/challenge/dataset.html)、[评测规则](https://behavior.stanford.edu/challenge/evaluation.html)、[baseline](https://behavior.stanford.edu/challenge/baselines.html)、[提交](https://behavior.stanford.edu/challenge/submission.html)。
2. 2025 冠军：[IliaLarchenko/behavior-1k-solution](https://github.com/IliaLarchenko/behavior-1k-solution)。
3. G0.5：[论文](https://arxiv.org/abs/2608.11739)、[代码](https://github.com/OpenGalaxea/GalaxeaVLA)、[模型卡](https://huggingface.co/OpenGalaxea/G05)、[快速开始](https://github.com/OpenGalaxea/GalaxeaVLA/blob/main/configs/QUICK_START.md)、[许可证](https://github.com/OpenGalaxea/GalaxeaVLA/blob/main/LICENSE-G0.5)、[公开复现疑问 #57](https://github.com/OpenGalaxea/GalaxeaVLA/issues/57)。
4. LingBot-VLA 2.0：[论文](https://arxiv.org/abs/2607.06403)、[代码](https://github.com/Robbyant/lingbot-vla-v2)。
5. NVIDIA：[Isaac GR00T](https://github.com/NVIDIA/Isaac-GR00T)、[GR00T N1.7 模型卡](https://huggingface.co/nvidia/GR00T-N1.7-3B)。
6. 其它候选：[Wall-OSS-0.5](https://arxiv.org/abs/2605.30877)、[ABot-M0.5](https://arxiv.org/abs/2607.00678)、[VLA-0](https://arxiv.org/abs/2510.13054)、[openpi](https://github.com/Physical-Intelligence/openpi)、[pi0.7 公开状态问题](https://github.com/Physical-Intelligence/openpi/issues/980)。

# 附录 B：主张—证据台账

| 编号／类型 | 主张 | 证据、核验日期与边界 |
|---|---|---|
| E1／事实 | 2026 是 100 任务、RGBD+proprio、R1Pro、单 24 GB GPU | BEHAVIOR 官方主页/数据/评测/提交页；2026-08-26。高，以 v3.9.2 为准。 |
| E2／事实 | 0--9 leaderboard，10--19 held-back public test，另有隐藏最终实例 | BEHAVIOR 官方评测/提交说明；2026-08-26。高；不把 10--19 用于反复调参。 |
| E3／事实 | G0.5 的 2025 分数为 0.2904、0.3136、pi0.5 0.2626、冠军 0.2605 | G0.5 论文 BEHAVIOR 表；2026-08-26。高；不是 2026 成绩。 |
| E4／事实 | G0.5 无公开 BEHAVIOR adapter/checkpoint | G0.5 QUICK_START、模型卡与仓库检索；2026-08-26。高；未来版本可能改变。 |
| E5／事实 | 发布模拟 checkpoint 与 AR+CoT 的一致性有公开疑问 | G0.5 GitHub issue #57；2026-08-26。中高；issue 不是项目方结论。 |
| E6／建议 | G0.5 应做主线但需 gate | 由 E3 的直接性与 E4/E5/许可证风险综合得出；条件性结论，不是预测。 |
| E7／建议 | GR00T N1.7 应作交付兜底 | 官方 baseline 集成与资源说明；工程价值高，不声称最高 Q。 |
| E8／建议 | LingBot 2.0 应做 canary | R1Pro/深度/长程结果，但无 BEHAVIOR 结果；中高置信。 |
| E9／推断 | 23D 至 27D 映射可行 | G0.5 R1Pro 27D schema 与 BEHAVIOR R1Pro action schema；必须闭环验证语义。 |
| E10／建议 | 2025 冠军的 stage/recovery/smoothing 值得迁移 | 冠军 README 与 G0.5 长程弱点；迁移系统思想，不复制内核。 |

### robo 远程只读审计条目

- **R1／robo 观察。**`robo` 为 4×A100 80 GB；G0.5 树、环境、数据与权重已就位，且有 6 modified/17 untracked。证据：`llmvideo26` 只读审计，2026-08-26；路径见第 9 节。置信高；修改未提交是复现风险。
- **R2／robo 观察。**五任务 1,000-step G0.5 pilot、processor/checkpoint 均 PASS。证据：checkpoint 与两项 check 脚本输出，2026-08-26。置信高；只说明离线通路。
- **R3／robo 观察。**两项 warning：224/256 不一致和 dummy normalization。证据：config、运行行为与 dataset stats，2026-08-26。置信高；必须在正式全训前修复。
- **R4／robo 观察。**Pi0.5 Flexiv 20k 训练完成，当前 GPU0 service 加载该 checkpoint。证据：作业日志/checkpoint/进程，2026-08-26。置信高；与 BEHAVIOR 闭环独立。
- **R5／robo 观察。**G0.5 未 GPU 推理/未协议测；OmniGibson 在 `kexin8`；Docker 未装。证据：端口/tmux/文件/模块只读审计，2026-08-26。置信高；最终交付尚不成立。

## 版本与可追溯性说明

研究结论仅覆盖 2026-08-26 可公开检索的材料；论文、代码和许可证都可能后续变更。PDF 交付前应将 robo 审计的准确路径、commit、日志行和硬件快照补入 R1--R4；不得把凭据、HF token、SSH 私钥、完整数据路径中的个人标识或受限权重链接写入报告。许可证部分仅为工程风险识别，不构成法律意见。
