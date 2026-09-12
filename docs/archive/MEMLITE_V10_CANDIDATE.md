# 后续训练候选：尚未部署、尚未开训

当前正式运行仍是 v9 step5000 的五任务模拟器评测，必须保持权重、源码、seed和原timeout不变。原始主仓库为 `robo:/mnt/sdc1/robodojo/GalaxeaVLA`，21:10 UTC再次核验18项关键源码与本轮训练启动记录一致。

## 实测依据

同25个留出动作窗口的真实纯AR预测显示，大幅右臂动作被预测得过小、夹爪切换被漏掉，也有部分过大预测；不是所有动作均为零。CPU codec往返与完整训练/验证标签mask检查排除了这25窗口的夹爪表示损失和监督遗漏。详见 `memlite-v9-ll-probe-report.md`。

原训练集950个episode的CPU统计及实际旧sampler5000步回放：低层总140000行，夹爪切换窗口2669（1.91%）；大于0.05rad的机械臂窗口并不少见，因此不把问题描述为“全是静止数据”。原采样按任务帧数分配，而不是五任务等权。

| Task ID | 原LL计划样本 | 其中夹爪切换 | 候选LL计划样本 | 其中夹爪切换 |
| --- | ---: | ---: | ---: | ---: |
| 0 radio | 6138 | 134 | 28000 | 4000 |
| 1 trash | 14359 | 376 | 28000 | 4000 |
| 2 Halloween | 42113 | 754 | 28000 | 4000 |
| 3 plates | 41542 | 517 | 28000 | 4000 |
| 4 can meat | 35848 | 888 | 28000 | 4000 |

这是计划索引回放，不是未来DataLoader实际送达的逐样本收据，也不是成功率。

## 候选实现

独立远端：`robo:/mnt/sdc1/robodojo/behavior_dev/memlite_train_strategy.IofJdZ`。
本地副本：`/home/wsy/behavior/memlite-train-strategy.Rjv0lZ`。

- `src/g05/utils/common/task_event_sampler.py`：显式opt-in的新sampler/factory。五任务等权；每卡每批8行固定1HL+1关键LL+6其他LL。原sampler及验证sampler保持原默认行为。
- 关键LL只来自原TRAIN episode、当前primitive内完整前16步的真实夹爪命令切换。判断包括首动作与上一帧命令的切换；不是必须在输出16步内部再切换。其余LL保留原全部非关键行，包括导航、保持、小动作及边界附近行；不根据阈值删掉困难样本。
- `task_event_index_v1.json`：只记录已审核标签上的采样坐标，不生成/重写任何annotation。验证dataset roots、sidecar path、950个active episode、dataset长度、完整HL/LL指纹以及任务/关键/其他池的无重叠完备分割。
- DDP各rank有永久互斥的源样本分片。为等任务/事件采样，长epoch可在单rank内有意循环稀有池；真实5000步回放中140000个LL均唯一，HL少量有意重复。不能泛称“所有样本永不重复”。
- `scripts/finetune.py`只接入显式factory和采样日志；eval继续用旧branch sampler。
- `configs/task/r1pro_memlite_ar_v10.yaml`是待验证/待启用的第二阶段配置：从v9权重初始化、新optimizer/scheduler、5000步、LR2e-5、warmup100、最低LR比例0.1、seed17。不是从G05基座重训，也不是直接续接已降到LR0的旧scheduler。最终采用与否仍以本轮完整评测和后续准入为准。

模型架构、相机时序、action codec、归一化训练表示、CE部位权重、纯AR与FM冻结状态均不改变。没有为这次采样候选改图像结构或重造标签。20:44 UTC之后，raw-state anchor推理修复已合入这个隔离candidate做联合验证；原独立anchor副本仍保留，主仓库仍未修改。

## 已完成验证

- 新sampler及旧sampler第一轮39项，加入factory后43项；扩展相关回归最终91项通过。它们有重叠，不能相加。
- 扩展回归最初因隔离副本少了`.python-version`根标记及既有bridge import路径而在collection失败；补齐候选环境后91项全过。这不是修改正式训练/策略代码来绕过测试。
- 用真实Mixture/子Dataset、实际sidecar分支索引验证候选索引完全匹配；实际四rank各5000批回放得到上表精确配额，LL共140000唯一，跨rank源行互斥。
- 实际读取每task每类1条，共15条真实样本，禁用替代重采样：episode/frame/原Dataset索引/分支均正确，三相机各6帧、256×256，关键窗口前16步均未padding。本人读完15条文字输出；没有声称已看这15组训练图像。LL不带memory字段是高低层分工的既有设计，不是数据漏字段。
- 数据统计与索引构造都只使用原训练episode；source sidecar SHA仍为`a5f1026110cac3ed57682d8177d2dabcb2f30d6572791ebb27c2800100f5c5d2`。未启动GPU训练或另一轮模拟器。
- 实际Hydra组合v10配置通过：model architecture、processor、tokenizer逐项解析后与v9已保存配置完全相同；已核验从v9 step5000初始化、不重置扩展token embedding、fresh optimizer/scheduler、LR2e-5/warmup100/min-ratio0.1、每卡batch8/worker8/prefetch2、同5%holdout和同train-only统计。此检查不加载VLM，更不是训练smoke。
- 与raw-anchor合并后141项相关回归通过（与之前91/43项重叠，不相加）；联合candidate的105个真实示范窗口重建再次完成，缺失组0、准入拒绝0、夹爪重建错误0。候选仍没有加载真实VLM或运行模拟器。
- `combined-v10-candidate.patch`共874行、9个文件。对未改动主仓库的`git apply --check`及对联合candidate的`git apply --reverse --check`均通过；这里只检查，没有在主仓库应用或在candidate撤销。主仓库18项训练源码哈希仍一致。

## 21:11 UTC 兼容性复核补充

> 23:00 UTC新证据：另有25个高层自由生成选点诊断已完成，见`memlite-hl-probe.nIMizE/REPORT.md`。发现两处相对标注的切换滞后，也有合法替代顺序造成的逐字不匹配。现有candidate仍未变更/部署/开训；不能仅凭小样本DONE召回低就提高提前终止倾向，正式模拟器会独立判成功。最终训练方案仍要结合完整正式评测及必要的切换点复核，不能把此候选的准入测试说成最终方案已经定案。

共享inferencer还被普通`serve_policy.py`使用。新增raw-anchor诊断以及已有normalization诊断若进入其动作cache，会触发`AttributeError: 'dict' object has no attribute 'ndim'`。已先在隔离candidate加入4个回归场景（无诊断、anchor、normalization、两者），修复前实测3失败/1通过；随后仅在普通server排除这两个私有metadata字段。动作数组与缓存步序不改变，也没有启动普通server或被取消的基线评测。

修复后的联合回归145项通过（包含原141项，不能相加）。新补丁为`combined-v10-candidate-v2.patch`，918行、10文件；原874行版本保留为历史文件，后续应使用v2。v2再次通过主仓库的apply-check及candidate的reverse-check，均只检查未应用。普通server主仓库原始SHA为`77b942ac55cfd49fa306faa5a4c3b544ce136091329f4faf90aacd70a264eebb`。105窗口动作重建实现没有变化，没有把之前的CPU重建误称为再次模拟器验证。

原报告：`memlite-v9-activity-exposure.json`；真实索引验证：`memlite-train-strategy.Rjv0lZ/real_dataset_sampler_validation.json`。部署前仍需本轮五任务正式结束、临近部署时重查用户变更/源码哈希、主仓库增量应用后的回归、真实四卡smoke。启动器/正式新run尚未创建。不能把候选测试通过说成方法效果已经变好。

## 23:46 UTC：只读启动计划，不是训练

新增隔离辅助脚本`plan_only_v10.py`与6项针对性测试，未加入918行/10文件生产补丁，未改主仓库；生产候选145项回归的计数不变。这些准备辅助脚本不是运行中的启动器。

实际在robo、CUDA不可见条件下运行只读计划，复用已有`validate_config.py`，基础v10配置检查通过：architecture/processor/tokenizer与v9相同，v9 step5000初始化、fresh optimizer/scheduler、LR2e-5、warmup100、min-ratio.1、phase5000。只打印两种待执行spec，没有加载权重、调用GPU命令或创建run。

- 将来真正smoke的计划是4GPU、每卡batch8、5个optimizer step、warmup1、每步eval、第5步保存checkpoint；明确`DRY_RUN=0`，不把它叫纯配置检查。这组smoke覆盖配置只有构造/单元测试，尚无真实GPU证明。
- 正式第二阶段仍计划5000步、warmup100、每100步eval/500步checkpoint。smoke和正式训练都从原v9 step5000权重独立初始化，不把smoke产生的权重混入正式阶段。
- 实际执行前仍需完整v9结果、方案决策、精确data/index身份复验、用户变更/源码检查、主仓库回归、四卡空闲和磁盘余量、独立run/control记录。计划中的时间戳是明确占位符，不可直接当成正式run使用。

另有固定旧历史＋更晚真实图像/proprio的25次诊断，显示两个切换点滞后持续至+512，详见`memlite-hl-probe.nIMizE/matched_history_v1/REPORT.md`。它没有改动本候选高层数据/采样，不是已取消的非MEM基线，也不是新的模拟器成功率。
