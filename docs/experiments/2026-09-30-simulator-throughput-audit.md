# RL仿真吞吐审计：A100已经渲染，优先消除串行计算与重复观测

Owner：Codex / RL-SPEED-AUDIT；2026-09-30 18:46北京时间。用户要求先研究速度；本轮只读服务器、源码和公开一手资料，0新训练/仿真/测速/环境修改。E4接续器仍失败未修，不能把本报告视为RL已经恢复。独立分支`feat/rl-dense-aggressive-20260929`，共享主工作区及A800其他线程保持原样。

## 结论与证据边界

1. **不是纯CPU渲染。** E3实际Kit日志是Vulkan，GPU2/3分别被对应worker选中；实际renderer为PathTracing、OptiX去噪开启。官方入口明确关闭GPU dynamics，不能把`physics_gpu=2`当成GPU物理已开启，更不能把整个环境当成GPU端到端张量流水线。
2. **当前训练最耗时的是更新段，而不只是仿真。** 最新恢复段6批训练181.52分钟：课程回放24.68分钟、自主采样26.90分钟、更新桶约129.86分钟（71.54%）。更新桶包含存盘/reset尾部，非CUDA kernel独占时间。仅重复候选重试间隔就87.42分钟。
3. **仿真速度确实仍低。** 固定六回合final双环境合计7.435真实控制/s，平均每环境3.718控制/s；30Hz控制对应约0.124倍实时。更早的纯动作回放双环境9.130控制/s，不含模型/PPO/冷启动。两者不可相减来声称精确推理耗时。
4. 优先级：先补组件计时，批量化FM评分与更新回滚；仿真端测重复观测、NUMA/线程、纹理预算；随后做同画质RTX机器对照及并行数测试。不能承诺换卡/开某个选项就获得10×。

小体积可复算证据：[审计JSON](2026-09-30-simulator-throughput-audit.json)，保留相邻状态边界、计数、原日志SHA。没有伪造组件级profile或历史GPU占用曲线。

## A100、CUDA与RT Core到底是什么关系

- A100没有RT Core，但有通用GPU计算能力。光线求交、加速结构遍历可以用通用计算实现；NVIDIA在OptiX针对A100的说明明确支持CUDA路径。RTX的RT Core专门加速遍历和光线/三角形求交，不等于所有渲染都在RT Core上。Tensor Core长处是矩阵计算/部分去噪，不能把其低精度峰值直接当作光追速度。[NVIDIA OptiX/A100说明](https://forums.developer.nvidia.com/t/optix-7-on-a100/182509)、[NVIDIA OptiX各单元分工](https://forums.developer.nvidia.com/t/take-full-advantage-of-cuda-core-and-rt-core/241682)
- **算法能在GPU上计算≠软件厂商正式支持所有显卡。** 本实验固定Isaac Sim 5.1.0官方要求明确不支持无RT Core的A100/H100。已有本机适配能运行，是实测事实，但并不获得官方兼容/性能保障。[Isaac Sim 5.1要求](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/requirements.html)
- 本机日志证明Vulkan选择A100进行渲染；`optixDenoiser/enabled=true`只证明配置使用OptiX去噪，**不是证明整个Kit渲染器通过OptiX API追踪光线**。低层实现和每种GPU执行单元占比需profile，不能套用另一渲染库说明替代测量。
- 另查到公开[A100 Isaac Sim实测项目](https://github.com/asimfish/a100-issacsim)，它可证明此类运行并非逻辑上不可能，但场景、CPU、GPU型号和背景负载均非我们的配对条件；不采用其倍率预测BEHAVIOR或作为本项目测速结果。

## 实际运行位置与配置

审计run：`robo:/mnt/nvme_tmp/robodojo_g05_rl_20260928/e3_render_resume_v1`。
冻结源码：`/mnt/sdc1/robodojo/behavior_dev/git_worktrees/g05_50k_rl_render_resume_970896e`，commit `970896ede236851809573b02b48401f67f5e9dd2`。版本固定OG3.9.1/Isaac5.1.0，driver580.173.02。

| 部分 | 实际路径/行为 | 能与不能证明什么 |
| --- | --- | --- |
| learner | GPU0；冻结VLM BF16，AE FP32，TF32关闭 | 模型确实用GPU；不能因此认为小batch与串行评分高效 |
| sim0/sim1渲染 | GPU2/GPU3；Vulkan，PathTracing，4 spp/16 totalSpp，4/6 bounces，OptiX去噪、无temporal/DLSS | 不是CPU光追；配置值不等于每帧实测渲染时长 |
| 三相机 | head720×720，双腕各480×480，共979200像素；内部名`full_v1` | 已低于原wrapper3499200像素；不能将名字误读成未改官方原始分辨率 |
| 物理 | SDK `eval/evaluator.py:61`设置`gm.USE_GPU_DYNAMICS=False`；simulator按其关闭GPU dynamics并选MBP | 当前不是全GPU物理；下次需记录原生physics context实际值，不能只读cudaDevice |
| 控制与图像时钟 | 120Hz物理、30Hz控制；32预测/前16执行；每16控制末拍图 | 渲染已与控制解耦，不应重复宣称“改成chunk末渲染”是新优化 |
| CPU亲和 | sim0=72–79，sim1=80–87；都是NUMA3 | GPU2邻近48–71/NUMA2，GPU3邻近72–95/NUMA3；sim0跨NUMA，影响大小未测 |
| 进程 | 两常驻独立sim；启动TRAIN/final各约8分钟 | 常驻已实现；并行场景初始化未计入暖态吞吐 |

原始证据：`{training,final}/sim_{0,1}.stdout.log`、`{training,final}/worker_{0,1}/native_rl_profile.json`、`io.jsonl`、`steps.jsonl`。本次`nvidia-smi`四卡均空闲，因为15:33作业已结束；不能用当前0%反推此前没有GPU渲染。`nvidia-smi topo -m`与`lscpu -e=CPU,NODE,SOCKET,CORE`为亲和依据。没有停止其他进程或调整CPU/驱动。

## 时间究竟花在哪里

统计仅为09-30最后恢复段，不把继承55740控制或旧173次更新塞进本段分母。训练从173→208，共35新accepted；22596新物理控制中11496是专家前缀，仅11100是新自主控制。

| 训练阶段 | 墙钟 | 占训练时间 |
| --- | ---: | ---: |
| 专家前缀回放/起点检查 | 1480.93s / 24.68min | 13.60% |
| 自主采样、推理及rollout保存 | 1614.05s / 26.90min | 14.82% |
| updating桶 | 约7791.65s / 129.86min | 71.54% |
| 其余概率门 | 约4.47s | 0.04% |
| 总训练 | 10891.11s / 181.52min | 100% |

方法：解析1609条`RL_STATUS`，相邻phase切换作区间，按`training_result.seconds`裁掉最后收池/重载尾部。updating包括PPO/BC、candidate回滚与验证、checkpoint、下一批reset；collecting包括模型、物理、观测、IPC、rollout保存。不存在独立渲染秒数，不把这些墙钟桶误称组件profile。

仅本段TRAIN总墙钟吞吐为**1.019自主控制/s**；不能用包含前缀的2.075物理控制/s夸大新经验产量。这使100k自主控制在同配方/相同数据长度下约需27.25小时；这是该段吞吐外推，不是未来进度承诺，更非达到成功率所需样本数。

### 更新为什么慢：已由源码与日志相互印证

- `scripts/rl/learner.py:update`每组8条轨迹先逐条反向，再对每个候选参数**逐条评价整个rollout**。每次`G05FlowAdapter.score()`又循环10个FM时间步，batch1执行。
- 本段六批大小107/76/128/128/128/128；候选25/38/30/19/19/27，共158次（35接受、123拒绝）。据真实事件与源码循环推算，光候选评价就17723次轨迹评分、177230次AE velocity forward；这不是profile直接采到的调用次数，不包含采样与梯度计算。
- 123次拒绝全部触发max path KL，9次也触发mean KL，均非surrogate退化或log-ratio溢出。这是实质安全门，不应简单删掉来报速度。
- `trust_region.py`为每候选从CPU快照恢复AE参数/Adam，`learner.py`另做完整AE CPU拷贝/变更统计；单delta含状态约7.62GB。大量搬运是明确代码路径，但尚无独立带宽时间分解。
- 从相邻日志能单独圈出117个“同梯度、上一候选被拒、继续下一候选”的区间，累计5245.35s、每次中位48.54s；其中不包含仿真。这是全训练约48.16%，包含restore/evaluate/status，非纯前向kernel时长。
- 每个minibatch把LR重新拉回基础值，再做最多6档退避；可研究沿用最近合法尺度以减少无效重试，但那改变优化日程，必须另作方法对照。先做保留相同候选/接受判据的等价计算优化。

若保持更新段不动，即使假设其他训练阶段瞬间完成，这一段总训练加速上限也仅约`1 / 0.7154 = 1.40×`。把采样/课程整体加速2×约得1.17×；只加速渲染更低。这是当前同步流程的Amdahl示例，不预测更改算法/并行结构后的上限。

### 仿真/闭环已有基准

- 旧`e1_v3/throughput.json`：同两TRAIN1/138、相同256控制、同画质；串行53.31285s（4.802控制/s），并行28.04052s（9.130控制/s），1.901×。两个常驻进程各一张A100；含chunk末观测/IPC，排除初始化/模型/PPO。证明双进程并行有效，**不是单A100加第二实例的实测**。
- E3最终评测：public_test301/302 × seeds17/23/41，共19344控制，评测阶段2601.71s，合计7.435控制/s。包含模型、物理、图像、seed间reset与关闭尾部，不含473.34s final池初始化；完整SR仍0/6。
- 六回合各3224控制，在环境里约107.47秒；平均每对回合墙钟约14.45分钟。30Hz是模拟时间尺度，不是实测30FPS。不能把两环境总吞吐当单环境速度。

## 仿真端具体待测项，不是泛泛“开GPU”

### 1. 重复观测与RGB-D通道

`sim_worker.py`丢弃每次`env.step()`返回的obs；但安装SDK `envs/env_base.py:_post_step`仍会执行`get_obs()`。`VisionSensor._get_obs`调用annotator `get_data`。因此一个16控制chunk有16次未用于actor的环境观测获取；末尾`observe()`还先直接读三个sensor RGB+depth并做CPU复制/SHA，再通过`env.get_obs()`重复读图。缓存是否避免实际复制、各读的代价多大，需要计时，不能把次数当耗时比例。

候选：仅跳过中间不用的图像读取，保留每一控制的proprio/控制器/物体状态更新、官方task.step、终止与奖励；chunk末同一批图像读一次复用。不能直接跳过整个`_post_step`或`env.step`，否则会漏成功/接触与状态逻辑。actor实际只消费RGB，depth目前服务审计合同而非策略；RGB-only可作为下一独立模式，必须先重写并验证三相机完成批/时间戳合同，不能简单删depth绕过检查。

### 2. 取图同步、画质与录像

当前`SynchronousIO`每次capture包含Fabric同步`render()`和Replicator `rt_subframes=4`，reset后另prime；配置4 spp/16 totalSpp。新段TRAIN1445与final1220次capture、零重试，因此不是这次被重试风暴拖慢。逐层测render/Fabric、capture、get_data/readback和IPC，不能直接删同步fence：此前已发生过错帧故障。

viewer和headless viewport已关闭。训练每chunk拼图＋CPU libx264录像可以改抽样/异步队列，但须先量化，不预设它是主要瓶颈。head512/腕320的`shared_v1`已存在，像素约减少52.3%；仅说明像素数量，不承诺速度2×。改变分辨率、spp/子帧、去噪或RT/PT模式都可能改变视觉分布，先固定状态图像/动作与短闭环回归，保留标准评测profile。

### 3. CPU、NUMA、内存

GPU2的worker72–79与GPU本地48–71不匹配，是低成本优先测试项；按实际空闲CPU与节点内存绑定，不能抢占别人预留核。Kit仍继承早期保守`limit_cpu_threads=4`，先A/B 4与8，不默认线程越多越快。纹理预算继承旧共享卡profile：显存比例1%、每请求16MB；80GB不是全部供纹理使用。提高预算可能减少换入，但也可能增加stall，需看稳态和图像，不是直接关闭流式纹理。[NVIDIA CPU/显存优化说明](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/reference_material/sim_performance_optimization_handbook.html)、[纹理预算字段定义](https://docs.omniverse.nvidia.com/materials-and-rendering/latest/rtx-renderer_common.html)

### 4. GPU物理不是无风险的一键提速

BEHAVIOR官方速度页明确提醒GPU dynamics对其环境可能更慢，且流体/粒子需要它；PhysX官方建议小场景CPU、大批量场景GPU配batch API。当前每Python进程单场景、每控制大量状态逻辑/读回，不会因一项cuda配置自动变成Isaac Gym式GPU流水线。实际日志还出现stairs碰撞网格不兼容GPU的警告，不能靠更改碰撞形状换取不等价速度。[BEHAVIOR速度建议](https://behavior.stanford.edu/other/speed_optimization.html)、[PhysX107.3性能说明](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/107.3/dev_guide/guides/physics-performance.html)

保持120Hz/30Hz、碰撞体、摩擦、抓取模式与官方成功条件；GPU dynamics只作独立实验且对接触/滑落/终止一致性验收。

### 5. 已有VectorEnvironment可作为批量仿真的起点

本机固定SDK确有`omnigibson/envs/vec_env_base.py`，不是必须换到Isaac Lab。源码在同一进程创建多个`Environment(in_vec_env=True)`，逐环境`_pre_step`后**只调用一次全局`og.sim.step()`**，再逐环境`_post_step`；这比每进程一套Kit更接近集中式物理批处理，值得做2环境原型。[官方Vector Environment示例](https://behavior.stanford.edu/getting_started/examples.html)

边界：现有类接收同一config并deepcopy，不自动解决异质实例装载；观测/任务后处理仍是Python循环，官方示例是五个Franka而不是我们的双R1Pro厨房任务。本实验还需接入官方reset/实例/成功条件、隔离不同scene的BDDL状态与相机产品，确认reset一个环境不会推进另一个环境的物理时钟。不能直接拿示例FPS或`num_envs=4096`宣称我们的RGB复杂任务已批量跑通。优先原生VectorEnvironment小原型，而非先重写所有BEHAVIOR语义到另一个物理引擎。

## 下一步执行顺序（建议，尚未启动）

### A. 同目标计算加速，尽量不改算法

1. 补独立wall/CUDA事件计时：动作/物理、object/task/reward、render、capture、readback、IPC、VLM prefill、AE采样、score、backtracking restore、checkpoint/reset。GPU计时只在固定短profile窗口同步，避免常开同步本身改变吞吐；输出median/p95、控制/s、自主样本/s及更新/s。
2. **批量化固定轨迹评分。** 已存trajectory的10个去噪状态是给定数据，`score`各时间点评价独立，可沿trajectory×时间打batch；生成新轨迹的10步有依赖，不能用同样方式并行。正确保留每条轨迹10×32×23的logp/KL求和、mask与每样本归约，不能把当前全局`.sum()`直接套到批次上。
3. 80GB允许时将回滚参数/Adam/梯度和固定rollout context留GPU，减少CPU往返；先测峰值显存，不笼统承诺全部能常驻。复核restore后的Adam步数/随机数、FP32变化和原概率门。
4. 候选验证可分块先查max KL；某块已越硬上限即可拒绝并回滚，无需继续评价其他块。通过候选仍完整计算max/mean/surrogate。123次拒绝全触max KL，因此有具体依据；加回归证明接受判定不变，并标注提前拒绝的统计非全批均值。
5. 模型服务对多个env合批，保留每env独立seed/context/终止；随后考虑异步worker消除慢环境barrier。不要在同批rollout中途热更新actor。专家前缀回放与下一次起点检查可探索和当前PPO重叠，因为它们不依赖新actor；自主采样须等相应权重版本。

### B. 仿真小对照

先固定原同任务/同动作、同720/480/480/PT设置，以组件计时确认去重复读图、NUMA和4→8线程的收益；每项独立对照。之后才测RGB-only和较轻画质；它们不算纯硬件加速。保留官方任务/physics时钟/终止，不能为速度跳过接触计算。

独立比较双进程/双场景和原生VectorEnvironment 2场景，共同物理step下的吞吐与任务状态一致性；GPU dynamics开关单列，不与批量化同时混成一个因素。需要先适配当前官方评测入口与独立reset合同，不在运行环境热替换。

### C. RTX实测与规模

09-30 18:44已只读确认`user@10.162.152.173`单4090可SSH：`runs/bootstrap_v1/status.json`因7201秒准备限额失败；`assets_v2`因HTTP206分段请求失败，仅实例和robot两个归档完整；`runs/sim_speed_v1`不存在。未继续安装/下载，**没有4090对A100倍率**。旧“准备中”已更正。

建议完成原隔离环境/资产后，先每端1环境同画质，再测单卡1/2/4实例（显存准入逐级）；报告GPU/CPU整机组合，而非纯RT Core收益。另测RTX适用的实时渲染profile，单列视觉差异。若4090采样明显更快，优先RTX做环境、A100做learner；当前learner曾约28GiB，不能预设24GiB4090同时装下完整learner和场景。跨机器RGB传输/策略推理的实际网络开销也须计入；同机9控制/s不能直接搬到慢SSH隧道链路。

不要先铺开十几个环境：同步更新已占71%，更多环境可能只把逐条评分的耗时也放大。多GPU更适合独立环境分工起步，不假设一个环境开multiGPU就线性加速；官方说明单physics场景不会随GPU数线性加速。

### 验收口径

优先目标是同物理/同策略质量下**端到端自主样本吞吐至少2×**，而非GPU仪表盘100%；这是待测门槛，不是本轮结果。另记录单/双环境控制吞吐，期望后续冲到双环境20+控制/s但不承诺。需连续短运行无错帧/额外物理步、同观测动作数值回归及任务状态/抓取回归，才接回RL。

任何新实验启动前独立写明准确commit/权重SHA/数据与场景/seed/资源及停止条件，不在本研究记录中虚构已批准的新训练。完整任务成功率仍独立验收，速度提高不等于RL有效。
