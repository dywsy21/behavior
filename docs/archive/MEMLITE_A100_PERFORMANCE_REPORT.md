# robo：A100适配、性能验证与本地视频

更新：2026-09-07。主代码已部署到 `robo:/mnt/sdc1/robodojo/GalaxeaVLA`。本轮没有重新训练，也没有重新完成一轮成功率评测。

## 结论

已完成这台四卡A100服务器的运行配置适配，修复CPU线程过度并行造成的主要仿真开销，并去掉单样本AR解码中的空缓存操作。71项相关测试通过；四个任务各256步真实闭环验证完成、录像已下载。GPU仍未持续满载，剩余主要瓶颈在AR解码，不把“显存占用”或瞬时GPU百分比当作吞吐提升。

原v9 step5000五任务完整评测已于北京时间09:50:03结束，最终0/5成功，均超时，没有运行崩溃。新256步短测只验证速度与运行，不替代这个成功率结果，也不证明MEM-Lite有效。方法失败分析见[完整评测报告](/home/wsy/behavior/MEMLITE_V9_EVAL_REPORT.md)。

## 视频在本地

原评测启动时使用了`--no-write-video`，没有连续录像。以下是从真实三相机截图生成的**稀疏回放**：每128动作采样一次，约4.27模拟秒一组，6倍速播放；没有补造缺失画面，最后一张也不一定是终止帧。

- [收音机，18秒](/home/wsy/behavior/memlite-v9-playback.UFoeaM/turning_on_radio_SPARSE_6x.mp4)
- [垃圾，44秒](/home/wsy/behavior/memlite-v9-playback.UFoeaM/picking_up_trash_SPARSE_6x.mp4)
- [万圣节装饰，115秒](/home/wsy/behavior/memlite-v9-playback.UFoeaM/putting_away_Halloween_decorations_SPARSE_6x.mp4)
- [餐盘与食物，114秒](/home/wsy/behavior/memlite-v9-playback.UFoeaM/cleaning_up_plates_and_food_SPARSE_6x.mp4)
- [肉类装罐，99秒](/home/wsy/behavior/memlite-v9-playback.UFoeaM/can_meat_SPARSE_6x.mp4)

全部3285张原始JPEG、正式JSON和轨迹均已传到上述本地目录。远端/本地逐文件SHA256清单再哈希一致：`52599e81370614508927855d9b604f7eeab4d98878aa31ad5f49e755dab8ef3a`。采样与编码细节见[视频说明](/home/wsy/behavior/memlite-v9-playback.UFoeaM/README.md)。五片均检查编码和首帧，不声称逐帧看完。

另外，新A100短测确实开了录像，每片256动作、8.53模拟秒：

- [新短测：收音机](/home/wsy/behavior/memlite-perf.7cDl71/live_v2/turning_on_radio/turning_on_radio_NEW_short_chunk16.mp4)
- [新短测：万圣节装饰](/home/wsy/behavior/memlite-perf.7cDl71/live_v2/putting_away_Halloween_decorations/putting_away_Halloween_decorations_NEW_short_chunk16.mp4)
- [新短测：餐盘与食物](/home/wsy/behavior/memlite-perf.7cDl71/live_v2/cleaning_up_plates_and_food/cleaning_up_plates_and_food_NEW_short_chunk16.mp4)
- [新短测：肉类装罐](/home/wsy/behavior/memlite-perf.7cDl71/live_v2/can_meat/can_meat_NEW_short_chunk16.mp4)

这四段不是原五局的连续录像，也没有重新跑垃圾短测。ffprobe均确认H.264、672×448、256帧；人工另查看了收音机第5秒画面。既有仿真每16动作才刷新相机，视频中间保留重复画面：30fps编码不等于每秒30张独立渲染图。

## 到底改了什么

| 主仓库文件 | 作用 |
| --- | --- |
| `scripts/eval_memlite_v9.sh` | 新增`--a100`、录像开关、独立run标签；按GPU隔离并复用编译缓存；保留原评测排程和结果校验 |
| `scripts/run_with_cpu_budget.py` | 仿真Torch线程1、推理2，其他CPU运算线程池限1；启动时显式设置CPU亲和性 |
| `scripts/a100_eval_profile.py` | 检查四张A100 80GB，读取NUMA拓扑，不猜GPU对应的CPU组 |
| `src/g05/models/g05/helpers/ar_helper.py` | 单样本AR跳过空的已结束样本缓存快照；batch>1逻辑保持不变 |
| `docs/a100_evaluation.md` | 启动方式、性能证据和限制 |
| `tests/test_{cpu_budget,a100_profile,a100_launcher,ar_single_row_cache}.py` | 新增配置、启动器、缓存隔离及AR语义回归 |

这台机器GPU0/1/2/3附近CPU核分别为0–23、24–47、48–71、72–95。保持每卡一个仿真和一个policy服务，避免不同卡的进程共同争抢全机96个核。

CPU/NUMA/录像/缓存配置通过`--a100`显式开启，旧命令默认配置不变；AR空操作优化对所有单样本推理生效。新增缓存只共享local/cache与global/cache，日志、设置和episode状态隔离；不会覆盖旧结果。新共享缓存路径的首次生成仍需冷启动；本轮测试了链接安全性，没有把短测复用旧缓存的启动时间当作新缓存方案的严格测速。

没有改模型权重、训练数据、官方CPU物理模式、相机分辨率/画质、16动作渲染节奏、3相机×6帧历史、HL每128动作更新或官方timeout。原训练启动记录的18个源码/配置文件哈希仍一致；AR辅助文件不在这18项内，其修改另由回归和真实GPU输入核对覆盖。

## 测得多少提升

### 固定记录动作：单独测仿真

同一段MEM-Lite收音机记录动作，每个配置重新初始化，16步热身后测64步，保留观察、渲染和官方指标。不加载policy模型，因此以下不是完整推理速度。

| 配置 | 仿真步/秒 |
| --- | ---: |
| 原默认：Torch 96线程，其余线程池也很大 | 3.40 |
| 只将Torch限1 | 22.95 |
| 只将Torch限4 | 24.64 |
| 所有CPU运算线程池限1，不录像 | 23.41 |
| 所有CPU运算线程池限1，录像 | 18.23 |

各配置64×61维机器人proprio逐值相同。这里不证明所有场景状态或完整闭环轨迹都逐位一致。

8步CPU剖析中，接触缓存更新从约1.88秒这一主要热点显著缩短：原代码有大量很小的张量运算，开96线程时调度/同步成本超过计算本身。采用1线程是四进程并行下的保守配置，不宣称每个孤立场景的最优线程数都是1。

原始数据：[固定动作仿真测速](/home/wsy/behavior/memlite-perf.7cDl71/sim_replay_v1/timings.json)。

### 真实闭环：每任务256步，带录像

仍用原v9 step5000、实例301、seed0。四卡分别跑四个任务，排除各自前16步热身后统计：

| 任务 | 步/秒 | 完成动作数 |
| --- | ---: | ---: |
| 万圣节装饰 | 4.04 | 256 |
| 餐盘与食物 | 4.00 | 256 |
| 肉类装罐 | 3.87 | 256 |
| 收音机 | 3.94 | 256 |

四份结果均检查`complete=true`、动作数256和视频256帧。任务启动有错峰，不能把各自热身后速度相加，声称已测得整个窗口稳定四卡总吞吐。

旧完整评测开头对应区间约0.57–0.68步/秒；新旧闭环动作并非逐值一致，所以这个对比只是实际运行参考，不是严格固定负载加速比。上述四任务测速发生在AR空缓存优化部署前，不将收益重复相乘。

新闭环内policy请求往返占约73%–78%时间，仿真已不再是最大的耗时项。初版`live_v1`计时包装器有递归错误，已修正；其exit0不能证明成功，数据全部排除，仅`live_v2`有效。

### 纯AR推理：跳过无效缓存操作

batch=1时，解码循环已经检查过是否全部结束；还会继续forward，就意味着这个唯一的样本没有结束。因此不需要再为“已结束样本”逐层做空索引与缓存复制。batch>1仍保留原有冻结逻辑。

同一条真实观察、同权重和随机种子，旧实现两次重复平均低层2.364秒/次，候选2.224秒/次；切回旧实现重测2.404秒/次。所有生成动作及高层文本完全一致。

部署到主仓库后再次从磁盘加载同一权重，不用临时替换函数：低层均值2.222秒/次、高层1.574秒；33条动作及高层原文与旧实现完全一致，冷启动重复也一致。只有这一个固定输入，不声称覆盖全部任务或数值环境。

原始数据：[前后对照](/home/wsy/behavior/memlite-perf.7cDl71/policy_cache_v3/policy_timings.json)、[部署后核对](/home/wsy/behavior/memlite-perf.7cDl71/policy_deployed_v1/policy_timings.json)。71项相关回归通过，覆盖停止条件、多样本缓存冻结和MEM-Lite接口。性能剖析采集本身很重，`profiled_baseline_not_for_timing`行不用于延迟对比。

## A100硬件边界和剩余工作

A100没有RT Core和NVENC编码器，这是硬件属性，不能靠软件适配补出来。[NVIDIA架构说明](https://developer.nvidia.com/blog/nvidia-ampere-architecture-in-depth/)

Isaac Sim官方要求不支持无RT Core的A100/H100；但当前机器的已有OmniGibson环境确实能够运行。这里适配的是这套已验证运行环境，不等于获得了厂商正式支持，也没有换渲染器或模拟器。[Isaac Sim要求](https://docs.isaacsim.omniverse.nvidia.com/5.0.0/installation/requirements.html)

录像使用CPU libx264，没有调用不存在的NVENC。保留官方CPU物理，GPU仍负责当前渲染与模型计算；没有为了提高GPU百分比改变评测物理条件。

解码剖析仍见单次低层近9.8万次`cudaLaunchKernel`调用及大量同步，说明小kernel的CPU发起开销值得继续处理；这不是说同样数量的kernel都串行占满GPU。本轮没有实现CUDA Graph、跨环境动态批处理或替换注意力依赖。下一阶段应围绕真实AR解码路径减少发起/同步、验证批处理吞吐，并继续做动作一致性检查，而不是只看GPU占用百分比。

## 下一次完整评测如何启动

在robo的主仓库中运行：

```bash
bash scripts/eval_memlite_v9.sh \
  --train-output /mnt/sdc1/robodojo/outputs/g05/r1pro_memlite_ar_v9/behavior5_memlite_ar_v9_train_20260906T031051Z \
  --a100 --write-video --run-tag a100_eval_01
```

此命令会跑五个完整episode，不是256步短测；每次用新的run-tag，启动器拒绝覆盖旧结果。本轮没有额外启动这次完整重评，GPU诊断任务均已结束。

服务器上的全部诊断证据：`/mnt/sdc1/robodojo/behavior_dev/memlite_perf.wm4qXp/`。主仓库适配说明的本地副本：[a100_evaluation.md](/home/wsy/behavior/memlite-a100-deploy.RlGz5a/b/docs/a100_evaluation.md)。
