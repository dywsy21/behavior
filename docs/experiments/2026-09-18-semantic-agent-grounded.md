# H-07：RGB-D grounding、可执行候选与有限搜索

2026-09-18，负责人Codex；分支`feat/semantic-agent-grounded-20260918`，当前实验源`d2da0ac`。改进的[接口设计](../SEMANTIC_AGENT_GROUNDED.md)、实时[执行计划](../plan.md)。没有训练或改动G05/FM/MEM-Lite、队友数据/RL，也未合main。

当前：实现已接线，90项CPU测试在本地和robo通过；首次真实初始化暴露新适配的重复相机配置错误，已经修复，工程复验进行中。**没有新闭环成功率证据。**

## 本轮假设与实现

旧H-06两条27B短测里，radio能认出来但手接近不了；task3不断小幅转向而没找到桌子。本轮测试“机器人参照、目标几何和动作可行性是否缺乏显式连接”，不是新的基座模型排名。

- 合法机载RGB-D反投影VLM指定的可见操作部位；深度洞/边界/多视角不一致拒绝。保留不确定性，目标表面点不冒称精确抓取pose。
- 用机器人自身指面几何算夹持中心，替代将EEF原点等同指尖；跨姿态与开合后必须检验。
- 每轮最多10个非HOLD候选经过原有限轨迹预检，明确预计收益/拒绝原因，VLM只能选可执行动作；保持micro、开夹、转轴、底盘和恢复退让选项。
- 24方位区间的实际观察覆盖，8°逐步搜索，有限换位；数据来自机载图像、本体局部速度积分，不是全局/物体真值。
- 3次普通恢复、最多2次策略重规划，不改原目标/持物记录；深度与真实微抬补充视觉随动验证。完整成功仍由隔离评估器确认。

## 固定预算与版本

原登记两个工程门＋两个策略短测，共4次场景reset额度；每次最多1536新控制/1200秒（初始化和原专家前缀另计），策略最多48决策。首两次初始化失败后，把两策略reset改为两工程复验，**不增reset，不再启动本轮独立策略回合**。保留3个保存状态、最多6次神经调用的静态schema测试，0权重更新、0付费API。复用Qwen3.8-27B `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`；只用GPU1/3，产物≤5GiB，保留80GiB磁盘。

| 版本 | 验证 | 状态 |
| --- | --- | --- |
| `40c1aee` | 本地87 CPU；robo Python3.10 87/87、3.792s | 基础实现，无物理调用 |
| `18b47ff` | 本地89 CPU；robo89/89、3.950s | 两场景初始化失败，各0新控制/前缀/神经调用 |
| `d2da0ac` | 本地90/90、1.587s；robo90/90、3.897s | 只读传感器修复后工程复验中 |

## 初始化失败的根因与修复

`OfficialEvaluatorSession`已经通过`run_behavior_eval_chunked`安装RGBDFullResWrapper、启用depth_linear，并在本体初始化后重载观测空间。新增适配错误地在reset后**再次给相机宽高赋相同值**；原生setter无相等检查，仍然销毁/重建render product并重新attach annotator，随后关节读取返回None。

两次相同异常都在`OnboardRGBD.__init__ → env.load_observation_space → proprioception_dim → get_joint_positions`，未开始控制/机器人标定。原始日志保留。修复为只读检查既有modalities/分辨率并从同一传感器读RGB和深度，禁止live reconfiguration；增加无setter/无reload的CPU回归。异常记录移至native退出前，避免Kit关闭解释器吞掉failure.json。

此问题是本次新增适配工程错误，不是VLM不会控制、深度不能用于竞赛或原始G05已经坏了。

## 待验/不可外推

- 两场景真实深度/FK/夹持中心/8°搜索转向门：运行中。
- 已保存状态三例新schema/目标像素与候选人工审查：等待门完成，最多6调用。
- 目标搜索、可信抓取、按键、稳定持盘与全任务成功率：**本轮无新闭环结果，不声称已解决。**
- 深度局部圆/胶囊否决不是完整环境碰撞规划；表面点不是可操作姿态；新实例、持物正例、正式OG3.9.2提交包装及独立成员review仍待。

服务器根`/mnt/sdc1/robodojo/behavior_dev/semantic_agent_grounded_20260918`；本地证据`artifacts/semantic-agent-grounded-20260918/`，原始视频/深度/日志不入Git。[路径索引](../SERVER_LAYOUT.md)
