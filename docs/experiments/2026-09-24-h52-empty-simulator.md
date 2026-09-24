# H52：共享训练 GPU 上的空模拟器资源检查

## 预登记（2026-09-24，北京时间）

- 唯一负责人：Codex；代码在 `feat/semantic-agent-grounded-20260918`，启动时绑定实际完整 commit。
- 唯一假设：现有 Isaac Sim 5.1.0.0 / OG 3.9.1 的**空** Kit 实例，在独立进程设置较小纹理流缓存后，可在 GPU3 的约 7.3 GiB 空余显存内启动并保留训练余量。
- 不是完整场景可行性、不是模型/任务效果测试；模型/数据/任务/实例/seed 均不适用。没有新训练、专家前缀、reset、机器人控制或 VLM 调用。
- 只启动一次、主动工作至多 300 秒、8 次空 `app.update()`；超限后只给自有worker最多30秒终止/kill清理，不追加工作。失败不复用输出或自动重试。已有 H44 launcher/资源门不修改，不热改任何既有环境、训练、服务或源码。

## 实现和资源限制

入口 `scripts/semantic_robot/probe_simulator_startup.py`。独立父进程监管自己创建的 worker；只可终止该 Popen 子进程，不向训练发送任何信号。

- GPU3 物理编号，`multi_gpu=false`，不使用 CUDA 重编号；worker 限 CPU72–75，Kit 最多4线程。
- 四张卡启动前 free ≥7168 MiB，运行时 free ≥3072 MiB。相对于启动基线，GPU3 新增及自有 PID 用量各 ≤4096 MiB；其他卡各 ≤384 MiB。
- 完整 `nvidia-smi -q -x` 包含 compute **和 graphics**。只允许原训练 PID 3294346–3294349 与自有 worker；训练消失、未知进程或超限均停止新 worker。查询超时5秒、轮询间隔0.25秒，因此这**不是硬显存分区，也不能保证瞬时资源或吞吐不受影响**。
- 纹理 streaming 开启，`memoryBudget=0.01`、`streamingBudgetMB=16`；运行时读取设置核对（只容忍float32表示误差，拒绝NaN/Inf/错误类型）。这些设置只限制纹理缓存，不限制总VRAM。
- 私有缓存、临时目录、portable-root；直接用已存在且与 OG 原文件 SHA 相同的 experience，避开 OG 启动时向共享安装目录复制文件的路径。
- 320×320 是空 viewport 设置，且关闭 viewport updates；不宣称未来720相机、任务纹理、物理场景在同预算下可用。

外部依赖固定：

| 文件 | SHA256 |
| --- | --- |
| 已安装 `isaacsim/apps/omnigibson_5_1_0.kit`、OG原 `omnigibson_5_1_0.kit` | `1aec3eda2c9841a060070c16305ea90c72c92a56cf0c973084b88f61b23f140d` |
| `isaacsim/exts/isaacsim.simulation_app/isaacsim/simulation_app/simulation_app.py` | `7cbaa6f00e935a6f14bf1c28ec0db089fd924e931f3b0deee07a822f9b7d0090` |

解释器 `/mnt/sdc1/xhz/miniconda3/envs/behavior/bin/python`，不安装或升级依赖。计划输出 `/mnt/nvme_tmp/robodojo_agentic_20260924/h52_empty_kit_v1`；私有运行目录 `/mnt/nvme_tmp/robodojo_sim_runtime_20260924/h52_empty_kit_v1`。

## 验证与判据

本地15项CPU测试已通过：完整GPU/XML身份、graphics进程、启动/运行余量、自有/增量双上限、原训练保护、设置有限值/类型、显式空应用配置、只终止自有子进程、唯一提交、构造时独立Kit argv及异常恢复；新增nonce/父PID/一次性stage/私有环境检查、子进程及退出后资源失败传播、spawn中断清理、close异常覆盖成功阶段。额外核对实际导入的SimulationApp源码路径，清除继承的PYTHONPATH；防止脚本的`--worker`被Kit转发，并保证其私有portable-root判断读到正确参数。GPU3/physicsGPU3/禁多GPU均核实际Carb读回。

首轮独审发现失败未传播、直达内部入口、spawn中断窗口和Kit argv问题，现已修复。20:55 Epicurus修后窄独审与独立15 CPU通过，远端同解释器CPU尚待。20:47只读原四训练/7489MiB余量、三个外部依赖SHA再次核同；**未启动**。必须看到 worker `updates_complete`、实际设置一致、8 update、干净退出、全程资源记录以及原四训练仍在，才算空启动通过；进程退出0本身不是通过。通过后如需完整场景必须另登记、另验证；不能据此宣布官方 success rate >0%。

## 真实提交

代码`701abfad5b23daadd6388c32ba915d72a45fe926`已push，robo独立源码`/mnt/sdc1/robodojo/behavior_dev/git_worktrees/shared_simulator_701abfa`。双端15 CPU通过，实际启动前完整GPU/graphics快照只见四原训练，各free7489MiB。

唯一launch UTC `2026-09-24T12:56:57.903775+00:00`，supervisor `3464856`，worker `3464863`，输出/runtime如上。已经终止，不可重复提交。

## 实测结果：辅助卡额度中止，尚未完成启动

监管用时3.536819808s，worker退出-15，停在`constructing_app`，0 app.update/任务/reset/控制/模型/训练。第四个资源样本在3.222s看到GPU0自有C+G **422MiB**、卡增量 **436MiB**，超过384MiB辅助卡额度；GPU1自有12MiB，GPU2/GPU3尚无自有context。监管只对3464863发SIGTERM；退出后四原训练均在且各73644MiB，全卡free恢复7489MiB。4个完整样本已审。

这不是OOM，不证明主卡预算不够，更不证明真实场景不能共存。Kit日志表明实际argv包含`activeGpu=3`/`physics cudaDevice=3`/`multiGpu=False`、正确private portable-root且无`--worker`；但应用构造尚未返回，运行时设置读回和GPU选卡结果**未验收**。需查GPU枚举的辅助上下文和renderer编号，不直接放宽旧门强跑。

全7文件已取回 `artifacts/agentic-vlm-goal-20260918/h52_empty_kit_bundle_v1`，三项SHA双端一致：

- launch: `456cbc5289e8b02aada29cb0344901c84ca24df70b4656ab77edc464795425b8`
- supervisor: `dee50c8544ff92ccbdb71608bfc192cb8a1fc7962dc500fb249675728ca51e25`
- worker: `51fe420adbe276f9134df2e290b1cc63fb52ebed937853e93f5ae8e55ca25ad5`

完整goal未完成；静态/资源实验均不计完整success rate。

## H52b：唯一显式选卡后继（21:03北京时间预登记）

负责人仍Codex。H52首次运行已完整封存，**不复用run、不中途修改限额、不改旧701abfa源码**。最新脚本另登记一次 `h52b_explicit_gpu_v1`，仍300s主动工作＋30s终止清理、8 update、0task/reset/控制/模型/训练。

唯一假设：显式禁止自动多卡、最多一个渲染GPU后，Kit能够把主设备设为GPU3，并将启动枚举产生的非主卡上下文限制在512MiB。H52的422MiB发生在GPU表出现前，不能确定是临时枚举还是选错主设备；不先宣称根因已证实。

- `multi_gpu=False`之外增加`/renderer/multiGpu/autoEnable=false`、`max_gpu_count=1`，两项都严格运行时读回。
- 将**本次新探针**辅助上下文额度从384登记为512MiB，以观察已见422MiB之后的选卡日志；原GPU3 4096MiB、启动全卡free7168MiB、运行全卡free3072MiB、原训练身份和超限只停自有child全部保持。再越界不继续提高额度。
- 日志明确写入新OUTPUT的`kit.log`，开启INFO以审实际gpu.foundation GPU表；设置键读回并不等于真实设备选择验收。CUDA可见性不用于替代Vulkan隔离，不改驱动/安装/Xorg。
- 不再重跑旧H44门，不载入任务；旧H52/H44的预算/证据保留。新输出 `/mnt/nvme_tmp/robodojo_agentic_20260924/h52b_explicit_gpu_v1`，runtime `/mnt/nvme_tmp/robodojo_sim_runtime_20260924/h52b_explicit_gpu_v1`。

作者及独立15 CPU通过，Epicurus的delta独审通过；固定source `1483f2493c4dc450b7077ee7d1db418cf618d7c1`，远端`git_worktrees/shared_simulator_1483f24`同15 CPU/资源/依赖核验通过。唯一launch UTC13:05:18.053589，supervisor3465781，当前运行中。通过空应用仍不能放行完整场景或报告新SR。

依据：[Isaac Sim 5.1 Setup Tips](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/install_faq.html) 列出autoEnable/maxGpuCount；安装版SimulationApp源码第416–420行已读，max_gpu_count确实转为renderer设置。[NVIDIA Linux Troubleshooting](https://docs.omniverse.nvidia.com/dev-guide/latest/linux-troubleshooting.html) 提醒核gpu.foundation实际GPU表，CUDA可见性并不控制Vulkan渲染选卡。不同文档对编号描述不完全一致，因此需要本机实测表和UUID/PCI信息，不只信传参。

## H52b终态：空Kit/选卡/资源通过，任务渲染未测

worker3465792与监管3465781均退出0；启动20.065438164s、worker20.076754204s、监管23.388254539s。全部8 update与8项实际设置核同，29个完整资源样本通过。GPU表明确只有GPU3 `Yes: 0`，UUID前缀`c67cdb9d`与注册的物理GPU3一致；GPU0/1/2未激活，`llvmpipe`软件adapter被跳过。日志无`[Error]`，RTX Hydra engine已创建；仍未载场景/生成任务相机图片，因此不声称图像质量或完整场景可行已验证。

| 物理GPU | 自有PID采样峰MiB | 卡新增采样峰MiB | 最低free MiB |
| --- | ---: | ---: | ---: |
| 0 | 454 | 468 | 7020 |
| 1 | 416 | 422 | 7068 |
| 2 | 416 | 422 | 7068 |
| 3（唯一active） | 568 | 598 | 6891 |

所以前次GPU0先出现context**不等于最终用GPU0渲染**，可观察到非主卡仍有初始化上下文。不能单独归因于autoEnable/max1：H52在相应阶段前就被较小额度中止，两次并非等预算完成的比较。原四训练全程仍在、各73644MiB；退出全卡free恢复7489MiB，未发训练信号。无新训练、任务、reset、机器人控制或SR。

全8文件在 `artifacts/agentic-vlm-goal-20260918/h52b_explicit_gpu_bundle_v1`；四项SHA已双端核同：

- launch: `d3cfa67d268d51a642799feab823d269b1341f977fd94c4e4fbe248156e780a2`
- supervisor: `e2e27652facfc970ec95523d7b3b69ff466fd887a896fcad94167e3683e199ab`
- worker: `2377f7cd0be850181bbed13867992aa58b70278a5fe93e0e979e35d989acac00`
- kit.log: `ca275d3aed31091fe11d24a384ffa6eb287a230bc455d9af72fcde5a39e165c0`

下一：保留现有安装/四训练，在独立进程中将已验启动设置接到原OfficialEvaluatorSession；先仅一个原始task0场景及三相机RGB-D检查，另登记预算/版本/源审查，无actor动作。不能沿用空启动结论直接放行完整闭环。

## 依据

[NVIDIA RTX renderer settings](https://docs.omniverse.nvidia.com/materials-and-rendering/latest/rtx-renderer_common.html) 说明纹理流预算；[Isaac Sim 5.1 性能指南](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/reference_material/sim_performance_optimization_handbook.html) 提供相关性能设置。最终以本机固定安装源码与实际读回为准，不把文档预算当总显存隔离承诺。
