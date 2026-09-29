# JEV-02：原始起点模拟器闭环

负责人Codex；用户授权真实闭环/成功率；独立分支`feat/jev-control-20260929`。状态：独审和双端CPU验证通过，**gate0运行中**，尚无新SR。源码固定`ced02feb3a830cd3e0c83081f15e818810f106d0`、implementation `32020e102a10fffb106564ed640857090cf6dcd43fe6a63c98e1e2567553d5ed`。

## 假设与预算

固定Jev `jev-1.13.0`、视觉Qwen3.8-27B revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`，视觉仅observe/ground；显式任务计划，0训练、无VLM规划/控制/恢复。

- task0/3各一次工程门：24动作、≤1536控制、1200s动作阶段/1800s总墙时、0模型请求。
- 工程门后task0 TRAIN实例138/seed0，从原始场景起点最多3次，每次96决策/3072控制/2400s动作阶段/3600s含启动；Jev≤208请求、观察≤215次，每次新进程/new run，0专家前缀/回放。
- 先运行episode1，基础设施/协议故障即停扩展。episode2/3要求前一条完整且所有证据有效；物理策略失败仍进入分母，不选择性丢弃。预定3次未全完成时同时报告已完成n与未完成原因，不以“0/3”代替只跑了一次。
- 正式成功只读取现有官方evaluator的`done.success`及goal predicates；夹爪闭合/模型自报/局部接近不是成功。使用已冻结v3.9.1开发runtime，不是官方v3.9.2提交；已用TRAIN实例的重复成功率不是跨实例泛化率。不做同条件基线、不能据此声明超越G0.5/VLM。

源码commit/digest及实际启动/停止时间由run回执固定。窗口/机器人沿用`run_sim.py`原固定SHA，不调整任务实例或动作判据。

## 资源与入口

robo队友RL进程4022589（GPU0）、4022893/4022896（GPU2/3）保持不动。只占physical GPU1；已有两个各约200MiB跨卡上下文仅基线记录，不停止。GPU1新外部进程或旧小上下文增长>64MiB即停本次会话；自有observer≤58GiB、sim≤14GiB/其他卡≤512MiB，所有卡保留8GiB，NVMe保留80GiB。CPU感知48–51、sim88–91，避开队友sim72–87；不改共享环境、CUDA/驱动、安装源码。

`scripts/semantic_robot/launch_jev_closedloop.py --stage {gate0,gate3,episode1,episode2,episode3} --launch`；仅在新冻结干净Git worktree中运行，启动器不自动重试/追加。每阶段目录：

- `/mnt/nvme_tmp/robodojo_agentic_20260925/jev_20260929_<stage>_v1`：launch/supervisor、`gate/{manifest,result,trace,native_io,...}`、视频及actor回执；工程门沿用gate目录名，actor身份按manifest判断。
- `/mnt/nvme_tmp/robodojo_sim_runtime_20260925/jev_20260929_<stage>_v1`：独占cache/Kit portable，提前建立可读截图目录；旧缓存不删除/改权限。
- 用户凭据将只部署到robo拥有者0600仓库外`/mnt/sdc1/robodojo/.config/behavior/credentials/typesafe.key`；未部署前不启动actor。禁止打印内容/写Git/发往非TypeSafe地址。

## 先修正的确定性接口问题

原JEV-01要求两次RGB-D逐字节相同。实际H75 `gate/native_io.jsonl` SHA `cc55a99cd7fc1f77ac04c999d6c474c9e136b18e5b67c6e5f71916b166eb6453` 的六组连续同物理时钟采集（capture1→2、16→17、20→21、30→31、35→36、40→41）哈希全部不同。故静止场景的PathTracing重渲染不满足像素相等；先前约定会误拦真实动作。

新契约不比较跨render像素是否相等，而验证：控制计数、仿真时间/physics index、全部q/gripper/base_velocity/finger状态完全不变；三路RGB-D匹配实际read哈希与native收据；新render batch严格推进、绑定相同相机graph/产品/分辨率；每次capture实际0物理ticks。通过原native同步I/O事务证明同一物理世界未推进，允许渲染样本变化。不支持真实机器/异步仿真，不向actor输入对象真值，不放松collision/grasp/goal。两项工程门也在每次动作前验证此新契约。

显式GPU1选择同时传至SimulationApp renderer/physics、OG imports/session、GPU环境与启动bridge校验；旧GPU3默认保持。工程门结果绑定physical_gpu，不能拿旧GPU3 gate跳过GPU1验证。

## 验证与结果

- 本地36针对测试通过；最终完整semantic_robot **766项/29.724s，5skip**；原生启动相关41项通过，包含GPU1 bridge；robo同冻结源77项1.513s过。
- 独审修复阶段wall有限/上限、清理60s上限及env key隔离；新增逐请求Jev持久账本，对齐attempt/validated/error、token总数，凭据/headers/响应体不入账。有效abstain正常终止、无普通动作，可作为策略失败计入分母；真实协议/网络异常和时钟契约失败停止扩展。
- 独立最终36目标/代码增量审查通过。17:41北京时间唯一gate0监管4030656/actor4030665已启动，GPU1 renderer/physics实际均核为1，尚在场景加载；工程门结果、API闭环、成功率和视频仍待。
- 17:43用户key已私有传至上述robo路径，目录0700/文件0600/owner robodojo核过；没有向视觉服务/日志/Git暴露，尚无新API请求。
