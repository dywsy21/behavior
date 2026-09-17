# H-02/03：R1Pro语义微动作与冻结小VLM初测

2026-09-17，Codex独自执行。状态：**运行中，尚未形成成功率结论**。
设计说明见[微动作接口](../SEMANTIC_AGENT_DESIGN.md)。不接旧MEM-Lite、不调用FM、不训练、不付费API。

## 预登记

- 主假设：机器人坐标明确、带实测回执的有限微动作可以可靠执行，并使冻结小VLM直接选择下一动作成为可检验路线。
- 代码：feature `feat/r1pro-semantic-agent-20260917`；第一版动作/服务`1680e01`，仿真入口`f18dd43`；远端独立worktree，不修改队友PPO或共享源。
- 接口门：task0 train instance138/envseed0，原episode121专家前缀448步；之后最多24个接口测试命令/600控制步/20分钟。它不是模型任务评测；前缀不算模型能力。原window SHA `93731a793ed550bae225e148d872b97d42e331a709e2fd0cd33bd1ccf76ccbd8`、原config SHA `a98fa8a81472adfecfb642778d4d7a01e20131be7f70824e0ae095d665cdbb93`。
- 初测候选：Qwen3.5-2B官方revision `15852e8c16360a2fea060d615a32b45270f8a8fc`；如需复核，4B revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`。bf16、batch1、三张320²图、greedy/不thinking/最多48生成token，HF5.7.0/SDPA；不是Show-Harness已微调adapter。
- GPU1模型/GPU3仿真，启动前真实检查；队友GPU0/2进程保留。最多两模型、总320次调用；闭环最多4个局部episode、各64决策/1536控制/20分钟；下载＋环境＋输出≤25GiB。不到门不扩展、不追加训练。
- 模型回合预备：task0同原448专家前缀作局部抓取，task3 train instance242/env0从真实reset开始（无4221步旧专家前缀），观察初始自主行为；不因不同起点比较任务胜负。正式盲测/提示词因果对照/FM配对不在本轮。
- 控制和场景版本：保留原23维R1Pro、assisted；OG v3.9.1开发环境，非官网v3.9.2正式成绩。任务结果只记录官方done字段，VLM不接收该字段。
- 停止条件：输入/控制契约不符、非有限、连续3次格式拒绝、严重方向错误、资源冲突或预算达到；保留失败，不伪造空动作/成功或无限重试。

## 已执行

- 本地12项CPU测试通过；方向、动作幅度、HOLD、夹爪命令锁存、底盘脉冲积分/停止、拒绝副作用和任务绑定均有回归。
- robo独立依赖overlay安装HF5.7.0，原G05 torch2.7.1和共享HF4.57.1未改。2B权重下载完成；路径`/mnt/sdc1/robodojo/behavior_dev/semantic_agent_20260917/models/Qwen3.5-2B`。
- 18:56启动模型服务`server_2b_v1`及仿真`gate_v1`，日志都在同一run根目录。实际初始化/物理结果待检查，提交启动不等于通过。

## 结果（待回填）

真实tracking误差、格式有效率、冷启动/暖态延迟、模型微动作、局部行为、官方终止以及视频，均以完成后的原始记录为准。当前0已确认新任务成功；不引用旧A4视频作为新方法结果。
