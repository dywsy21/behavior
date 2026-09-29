# JEV-02：原始起点模拟器闭环

负责人Codex；用户授权真实闭环/成功率；独立分支`feat/jev-control-20260929`。18:48北京时间状态：**gate0/3均过、episode1官方失败、episode2运行中**。整体管线暂为0/1，但第一条0 Jev请求，不能视作Jev动作选择效果。源码固定`ced02feb3a830cd3e0c83081f15e818810f106d0`、implementation `32020e102a10fffb106564ed640857090cf6dcd43fe6a63c98e1e2567553d5ed`。

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
- 独立最终36目标/代码增量审查通过。gate0监管4030656/actor4030665已completed/exit0：1185.018s总、590.471s动作；24检查完成/440控制/112同步采集、gate_ok=true/无失败；native journal SHA `3df7c0bf7d09838e2f08577e79a8b1f57f4ccfa9ca47dbedfa8cdf0a7d6e1955`，GPU清理通过。实际GPU1 renderer/physics均核为1；初始头帧本人已看，双端SHA `af324a2b8616bdc00ca73e8ecd694e8e248f76b478510b24ddcfa3399c0979d0`。
- gate3监管4036929已completed/actor exit0，1098.185s总/568.489s动作；24检查、23执行完成/1预检拒绝、440控制112采集、清理0.368s。result SHA `128d4b1bd4b7490e2035b281021746dc48ddc4d11bc7908da66e491b8fe74240`，native journal `ec4e9d99614356e4327e3be5ff9d2e89d9f5fc3cbe5502fa22193c772bc71d02`。两门0模型/训练，不算SR。
- 18:21 episode1监管4042627启动，同冻结源/GPU1/原始TRAIN138 seed0，96决策/3072控制/2400s动作/3600s总；实际模型/控制/官方结果仍待。
- episode1已完整结束：监管completed/actor exit0、1283.362s总、373controls，官方success=false/goal未满足。16次observer调用，0 Jev请求；现有搜索控制器尚未找到目标即因VISUAL_ODOMETRY_UNCERTAIN及恢复预算耗尽停止。完整管线失败，不等于Jev选择了错误动作。result SHA `77bd8c74c3a2923e4ca0e699ba14c867ac6634e52d30aab498d3b70ffe840f60`，supervisor SHA `63f1b78657a3d10c8adde41b239224e9ecfd37f45e4738725bba7756e8a1b348`。清理0.689s通过，所有失败记录保留。
- 18:44 episode2监管4050934唯一启动，18:48核model4050986/actor4051071处于初始化；同源/模型/原始实例/预算，不在试验间修改策略。episode3仍待，不能将未跑项写进SR分母。
- gate0精确计数：24检查，23实际执行完成（19到达/4夹爪命令完成），1被可达性/碰撞预检拒绝；全24项freshness通过。result双端SHA `0b30649ff6d9df2573cf66f317144e52b67afe2b5e35c850a99f9056d4dab580`。
- 17:43用户key已私有传至上述robo路径，目录0700/文件0600/owner robodojo核过；没有向视觉服务/日志/Git暴露，尚无新API请求。

## 第一条失败定位（不是Jev模型失败归因）

16条选择来源全部为`measured_search_controller`，16次视觉observe全为目标不可见，13条转向完成、3条中断。第013/014次失败子段只有22/20个有效RGB-D匹配，第015次28个匹配去重后24个，均不足固定25点门；自身过滤均移除0点，未进入新的几何求解，因此这三处不能归因为优化器发散或机器人mask误剔除。原始证据在各`decision_013..015/action_motion.json`；静态hold后能通过，但下一转向仍丢跟踪。

两次恢复仅通过12个HOLD控制建立**新的局部**参考，不恢复丢失位移/旧覆盖，也不改变搜索方向；随后继续yaw_plus进入相似玻璃门视角。本人已检查决策004/014头帧，后者大部分为明亮玻璃。可确认瓶颈是低特征匹配＋恢复不改变视角；尚未逐帧人工证明全部视觉negative都正确，也未测试Jev真实选动作。只读证据不用于修改当前三次试验。

第一条action wall602.358s，其中视觉生成累计169.364s，16次中位7.221s、首48.414s/第二18.684s；Jev0次所以无本轮Jev延迟。总墙时1283.362s含冷启动。视频已归档`artifacts/jev-control-20260929/closedloop/episode1/rollout.mp4`，双端SHA `f4d8178d7a44697f0d2401c5670236e73fd874fa2b22bcb2613c6904e25a7e4e`，640×1088/15fps/12.4s（省略模型等待，非实时墙钟）。`result.json`含全16条decision；本run没有`trace.jsonl`，一次多文件复制因此exit1，但结果/监管/视频实际均完整且hash核同，不能把复制退出码误说成actor错误。
