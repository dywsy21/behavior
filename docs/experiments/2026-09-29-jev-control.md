# JEV-01：TypeSafe Jev 接入现有微动作框架

状态：接口实现＋真实API离线试验完成；**新仿真闭环未运行**。2026-09-29；负责人 Codex；分支 `feat/jev-control-20260929`。最终代码 `df50c4071ecfb4395dbdba1ce92a41d58c78ed73`，逐例摘要见[结果JSON](2026-09-29-jev-control-results.json)。

## 先看实际结果

最终六个既有task0开发状态，Jev两次请求选意图/完整动作全部完成：**6/6合法且非HOLD**，没有执行这些动作。五个选项具有原预检器给出的正距离改善预测（其中躯干微动只有0.4mm，仍可能停滞），一个是对已持收音机换视角找按钮；均不是接触/抓取/开机成功。

| 状态 | Jev选择 | 这说明什么/不说明什么 |
| --- | --- | --- |
| 导航时目标偏左 | 底盘左转粗档 | 预测朝向误差由42.91°降至34.91°，不等于导航完成 |
| 收音机还远 | 底盘前进细档 | 预测接近49.2mm，未物理执行 |
| 需要横向调整 | 底盘左移细档 | 预测接近19.1mm，未验证桌边接触安全 |
| 近处难以继续接近 | 躯干下降微档 | 仅0.4mm预测收益，不足以证明摆脱停滞 |
| 右手尚未对齐 | 右手前移细档 | 预测接近7.3mm，不是已抓住 |
| 已持物但看不见按钮 | 右手向前呈现物体给头相机 | 选择inspect而非盲按；预测改善视角，不证明看到了按钮 |

两次API合计选择延迟中位 **1.314s**，范围1.055–2.979s。同六份旧记录的VLM动作选择中位4.171s，仅是历史参考：时间/服务负载/输入模态不同，**不是严格同条件加速比，也不包含视觉观测/渲染/执行**。最终run13次请求、83,575 input tokens，按当日公开价约$0.00351；整体失败请求计费不完全可见。

保留全部失败：`saved_v1`前10例全HOLD、最后1例概率校验失败；改成明确stage条件/预计算事实/精简自然语言criteria后，`saved_v2`又因原生概率和0.99被拒。最终只兼容0.01网格、舍入包络内最多0.02质量误差；原值/argmax不改、不归一，所有概率边界仍校验。不能把这次提示改写归因成单因素实验或泛化提升。

总请求39/40（主线程真实key38次＋独审假key误请求1次），0新GPU/训练/仿真控制。最终757回归通过（5环境skip），独立审查已修问题；原始run在本地 `artifacts/jev-control-20260929/{saved_v1,saved_v2,saved_v3}`。本轮到此复盘，不再自动追加prompt搜索。

下一步是独立新版本task0/task3工程门，确认执行前exact RGB-D freshness不会因渲染噪声误停，然后只跑一个小预算task0闭环。**本次没有新任务成功率、视频或实际抓取改善证据。** 相机定位、微动作执行器和缺乏接触/恢复技能的问题仍可能决定最终效果，不能靠更快选项选择自动解决。

## 这次究竟替换什么

暂停本线程 VLM agentic 控制/微调推进。Jev 接管微动作选择、恢复策略和“按钮属于哪只手所持物体”的语义判断；现有视觉模型只负责观察/表面定位。任务步骤用显式、绑定 task ID 的 JSON 提供，不再让 VLM 规划或选动作。MEM-Lite/G0.5 权重不变，本轮不训练。

Jev 目前是文本决策模型，不能直接接收图片。输入来自既有 `actor_context`：RGB-D 定位、机器人本体感知、真实执行反馈、未知状态、当前安全候选。不能把模拟器对象坐标/接触真值传给 Jev，不能把感知模块漏检说成 Jev 看懂了图。

一次控制决策：先用 Choice 选当前意图，再用 Choice 选**一个完整的已预检微动作**。第二次请求显式包含第一次的意图；没有将同次独立问题的答案误当相互条件化。双臂/底盘/躯干、坐标系和幅度仍为原 `Action`，不独立组合 X/Y/Z/夹爪生成未经检查的动作。唯一 HOLD 候选不调用模型；`abstain` 停止，不偷偷改成未经授权的动作。

Jev 不计算 IK/碰撞/浮点几何。代码提供距离改善分类；原servo依然负责23维执行、载荷/保持水平、限位、RGB-D约束、执行后重观测。置信度与选项概率只记录，**不是物理成功率**，也不能解锁动作。既有抓取验证和官方结束判据不变。

网络两次调用后，先检查目标/阶段/候选上下文未变，再在 `servo.begin` 前重新读取同步RGB-D、本体、独立finger、control计数和simulation time；与决策快照不一致就停止。新 `jev_freshness.py` 是保守的同步仿真约定，不是现实机器人延迟补偿。渲染噪声若导致拒绝必须记录/诊断，不悄悄放松阈值；不能只重验stage就沿用陈旧深度候选。

## 实现入口

- `src/semantic_robot/v2/jev_client.py`：原生 TypeSafe API、固定 `jev-1.13.0`、禁跳转/隐藏重试，问题/概率/模型身份校验（保留有界原生两位小数精度），字节/调用/墙钟预算；失败计入调用数，晚到结果不得执行。
- `src/semantic_robot/v2/jev_policy.py`：两阶段决策、perception-only组合、显式任务计划、Jev恢复/held-reference，保留现有预检与授权。
- `scripts/semantic_robot/run_v2.py`：新增 `--controller jev --jev-task-plan ...`，`--uri/--expected-revision`只标识观测服务。旧VLM分支保持默认，便于明确对照，而不是悄悄覆盖已有结果。
- `scripts/semantic_robot/test_jev_saved.py`：最多16条旧actor请求，纯CPU/真实API复决策，不执行旧动作、不上传原图或事后诊断、不产生success rate。
- `configs/semantic_robot/jev_task0_plan.json`：右手拿收音机、左手按电源；步骤是策略，不是“现在已拿住”的证据。此固定示例不代表所有任务已适配，task3/其他任务须单独审查计划。

## 运行方式

凭据仅使用 `TYPESAFE_API_KEY` 或 `--typesafe-key-file` 指向当前用户拥有的0600普通文件，不能把key作为CLI参数值、写进Git、请求回执或日志。本次用户凭据保存在**仓库外**本地私有credentials目录，未复制至robo。

离线复决策（干净冻结源码）：

```bash
python scripts/semantic_robot/test_jev_saved.py \
  --state /absolute/saved/decision_048/action.json \
  --key-file /private/path/typesafe.key \
  --output /absolute/new/run
```

在原H75/H77同步RGB-D/安全profile的 `run_v2.py` 命令上增加：

```text
--controller jev
--jev-task-plan configs/semantic_robot/jev_task0_plan.json
--typesafe-key-file /private/path/typesafe.key
--jev-max-calls 104 --jev-timeout 20
```

仍须**本版本源码摘要匹配的task0/task3两个控制门**。新代码会改变摘要，旧门不能直接冒充通过；本次不放松这个条件。observer服务身份、私有仿真runtime、有限动作/时间预算仍必需。没有本次完整闭环时不得将离线选项合法率当BEHAVIOR效果。

## 验证与预算

假设：保持同观测/安全执行，通过Jev typed choices改善决策延迟与动作选择；不声称解决感知/控制器物理问题。基础源码 `3c23cbf`；本轮新权重/训练/重置/控制步均0。先≤40次API/15分钟（包括一次工程认证probe），最多16旧开发状态；认证/版本/schema/预算/安全错误即停，不自动重试。后续仿真另登记冻结commit与小预算。

- 工程认证probe真实通过：模型 `jev-1.13.0`，未授权状态选择HOLD，335 input/31 output tokens，0.843s RTT；这是未冻结开发源码的单请求接口检查，不是完整性能结论。
- 最终新19项Jev回归、配合8项deadline目标检查共27通过；完整757项harness回归29.504s通过（5 skip）。覆盖凭据/redirect/异常HTTP/概率精度/schema/超时/调用上限、两阶段绑定、整动作选择、单HOLD/弃权、无VLM规划/动作/恢复调用、双手reference/task绑定、上下文与输入freshness及来源SHA。没有执行新仿真工程门。
- 2026-09-29 16:04北京时间robo只读快照四卡显存/利用率为0；不代表后续资源自动预留。未连接A800/VPN，不改队友进程或共享环境。

## 与参考仓库的差别和限制

参考固定commit `7a4ed8b72c3c17d7aa790678ed9660df67c10dd3`，只借鉴意图→小动作→反馈的结构，不复制机器人资产或apple/plate脚本。参考是单xArm的结构化模拟器几何实验、单次trial，走OpenRouter；本实现为R1Pro、多肢体预检候选、官方TypeSafe直连和非特权actor输入，不把参考结果外推成BEHAVIOR成功率。小数计算、未知证据和大上下文恰是官方提醒的边界，因此在代码中做计算/约束并减少冗余。

来源：[参考实现](https://github.com/openroboto-ai/jev-robot-control/tree/7a4ed8b72c3c17d7aa790678ed9660df67c10dd3)、[官方API](https://docs.typesafe.ai/api)、[模型/输入/价格](https://docs.typesafe.ai/models)、[官方边界](https://docs.typesafe.ai/model-jaggedness/jev-1.13)。
