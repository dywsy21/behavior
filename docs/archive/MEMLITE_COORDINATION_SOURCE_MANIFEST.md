# MEM-Lite 协同开发树 source manifest

**快照时间：2026-09-08；状态：初始快照已验证；source 现含仅供跨树复查的暂存候选，未启动 GPU 工作。48d6 parent semantic 泄漏已由人工审核确认，source 中的旧候选不可用于训练或物理运行，等待净化协议的原子替换。**

## 权威源快照

| 项目 | 值 |
|---|---|
| 主仓库 | `/mnt/sdc1/robodojo/GalaxeaVLA` |
| 基线提交 | `89f2322b4ad016e192437adc1a2c253b05bab246` |
| 已修改 tracked 文件 | 31 |
| 未跟踪源码/config/test 条目 | 96 |
| staged 文件 | 0 |
| dirty tracked binary patch SHA-256 | `d53dc15e93b1539196de29fa682df4d9f193be1d39125dc5cf15fbc6b236d24d` |
| 未跟踪源码归档 SHA-256 | `a508696159b0b9a5696a916769e3744818b921670d03dd4c31e615add6ad2947` |
| 工作区状态路径表 SHA-256 | `dd8615c5919baf10f13b738c39009e1bd30e731e4b3a1ec68ac40685049e5604` |
| 同步执行计划 SHA-256 | `5545f2fa94c613935cfee24a9d8d3d74803bec28146d9a42f692f4edf44a8f76` |

`dirty_tracked.patch` 与 `untracked_sources.tar` 是快照的内容载体；后者包含当前 Git 识别的未跟踪源码目录和文件，包含本执行计划，**不**包含 `.venv`、`venv`、`checkpoints`、`datasets`、`outputs`、模型权重或训练产物。

## 已建立的隔离树

根目录：

`/mnt/sdc1/robodojo/behavior_dev/GalaxeaVLA_memlite_coordination_dev_20260908`

| 名称 | 精确路径 | 可写责任 | 用途 |
|---|---|---|---|
| source | `/mnt/sdc1/robodojo/behavior_dev/GalaxeaVLA_memlite_coordination_dev_20260908/source` | 仅主仓库串行集成负责人 | 审查、准备集成 patch；不供并行实现直接写入 |
| data | `/mnt/sdc1/robodojo/behavior_dev/GalaxeaVLA_memlite_coordination_dev_20260908/data` | 数据/标签实现负责人 | E1–E2：事件协议、构造、全量机械检查、审阅产物 |
| model | `/mnt/sdc1/robodojo/behavior_dev/GalaxeaVLA_memlite_coordination_dev_20260908/model` | 低层及高层/结果头实现负责人 | E3–E6：训练通路、梯度审计、模型/损失与测试 |
| runtime_train | `/mnt/sdc1/robodojo/behavior_dev/GalaxeaVLA_memlite_coordination_dev_20260908/runtime_train` | 运行时/训练/评测实现负责人 | 训练 receipt、服务闭环、评测/日志与运行时测试 |
| sim_runtime | `/mnt/sdc1/robodojo/behavior_dev/GalaxeaVLA_memlite_coordination_dev_20260908/sim_runtime` | 仿真/serving 实现负责人 | 真实 physics 反馈采集、serving 接口、仿真时限与评测接线 |

五个目录均为 detached Git worktree；Git object database 可以共享，但没有共享可写源文件，且检查结果为零个多链接文件。各树在创建时均已验证：

1. `git diff --binary` 的 SHA-256 等于上表 patch；
2. `tar -d` 可同时验证主工作区与该树的未跟踪归档内容；
3. `git status --porcelain -z` 的路径表 SHA-256 等于上表；
4. 树内不存在 `.venv`；未复制数据、checkpoint 或 outputs；
5. 2026-09-08 初始四树验证时根目录总计约 27 MiB，主源树（排除 Git/环境/数据/输出）约 6.3 MiB，磁盘可用约 1.1 TiB；之后新增的 `sim_runtime` 由同一冻结 patch/archive 单独恢复并完成相同校验。

冻结 patch/archive 不会因后续实现者的工作而刷新。初始快照后的计划状态更新只允许写主仓库、本地镜像和 `source` 集成树，绝不覆盖已交付的 `data`、`model`、`runtime_train`、`sim_runtime` 工作树。因此 `source` 的计划文档可以较冻结 archive 更新；判定任一实现树的可集成基线仍以本节固定 HEAD、patch/archive SHA 和目标文件哈希为准。

### 已暂存的受审查组件（不等于主仓库集成或阶段放行）

2026-09-08 已将 data writer 的 v6/48d6 首批 10 个文件与随后仅 audit-provenance 的 3 文件增量暂存到 **source**，没有写主仓库：protocol、E1 builder、sidecar/LeRobot/recovery/processor consumer 及四个对应测试。每个目标在 source 的写入前均重新核对冻结源哈希或不存在性，写入后再核对发布 SHA-256；以绝对解释器和 `PYTHONPATH=$PWD/src:$PWD` 复跑指定四测试为 **34 passed**。依赖为 protocol `48d6f6d263b2891fdbf6e837a86f2a64ee79effb8a1268e41bfd6368e5ed12a0`。增量使实际 receipt 透传 `source_kind/expected_bundle_member_keys_json/requires_parallel/segment_start`，但这些 audit 字段不得进入 model semantic prefix。本暂存不覆盖后续 data writer 的更新；任何替换仍须以当前 source 目标 hash 为 precondition。

该验证只说明此时 source 中的代码/局部 frame-0 解码合同可运行。全量 sidecar、机械 split/mask 审计、120 项主代理视觉验收、真实 FM 一步和完整仿真均仍未通过，不得由此推断性能或成功率。

### 2026-09-08 source 串行暂存记录（全部未入主仓库）

已在 source 逐文件检查旧 hash/不存在性后，写入并复核以下三个作者发布组；这不是 git merge、主仓库写入、阶段放行或可运行版本承诺。

| 组 | 暂存范围与主要 hash | 当时的机械证据 | 当前限制 |
|---|---|---|---|
| data v6 | protocol `48d6f6d263b2891fdbf6e837a86f2a64ee79effb8a1268e41bfd6368e5ed12a0`；builder `d93fc4…`；sidecar `5d42ce…`；base/processor audit 增量 `99c9c3…` / `b8322a…`；4 测试 | data 指定套件 34 passed；局部 PyAV frame-0 解码 | **失效候选**：人工发现三个 parent 字段把 raw relation/`skill_idxes` repr 投入语义 prompt；不得用于 GPU 或 E1。等待数据作者发布净化协议、fixture、enrichment 和全量 hash。 |
| model | Skill-FM `4298aa19bc2784ec0b8d597308ebbcfec94215ab73267970979e2e1b873b426976`；planner/outcome/provider、LoRA、new samples builder、5 configs 与模型测试一并暂存（其余精确 hash 以 source 文件为准） | model 指定套件 16 passed | 依赖最终 data protocol；实际 selected processor、真实 FM 1-step、provider non-mock E2E 和18-ID train/serve映射均未通过。 |
| runtime/train | checkpoint `6e22bed9803fbb0e08dd148924b6655861700c8301fe13d72b965e2e1a6962f7`、finetune `465897…`、训练 receipt/sampler/launcher/3 stage scripts 与7测试；精确文件 hash 以本次 source 写入后的 `sha256sum` 为准 | cross-tree 旧组合为 80 passed + 1 expected data-hash fail | data 作者将协议改为净化候选后，runtime 的 identity/receipt/sampler/data-contract hash 已重新变化，source 暂存值不得视为最终冻结或启动许可。 |

旧组合交叉命令使用绝对解释器、`PYTHONDONTWRITEBYTECODE=1`、`PYTHONPATH=$PWD/src:$PWD` 及 `-p no:cacheprovider`，结果为 80 passed、1 failed；失败是 `tests/test_coordination_data_contract.py` 有意锁定 data protocol `48d6…`，而 data tree 已变为待审的 `1a458a85…`。它是防 torn integration 的有效失败，不可通过改期望 hash 伪造放行。

source 的运行/集成租约：最终 data clean hash、model consumer hash、runtime receipt/sampler hash 必须先组成一组完整 manifest，并在 source 重新运行 data/model/runtime 及真实 selected-processor CPU receipt；期间不得由任何作者从 source 启动 GPU。之后才可把 source 的确切文件 hash、配置、依赖和测试证据交给被授权的 GPU0 结构检查。主仓库仍然没有代码集成。

## 共享依赖与写入边界

- Python 环境只读共享：`/mnt/sdc1/robodojo/GalaxeaVLA/.venv/bin/python`。隔离树不创建 `.venv` 链接；使用该绝对解释器可以读包，**不得**在其中运行 `pip`、`uv pip`、升级依赖或修改其文件。
- datasets、checkpoints、已有权重仅通过绝对路径读取。新 data cache、训练输出、视频、临时 socket 和日志必须创建在各自 `behavior_dev` run 目录，不得写回主仓库或其它代理的树。
- 任何模型、数据或 runtime 改动先在所属树自测。提交给集成负责人的是带目标文件清单、测试日志、源/数据/权重/配置哈希的 patch/commit，而非直接写主仓库。
- 最终合并前，集成负责人必须重新检查主仓库 `HEAD`、dirty patch SHA、涉及目标文件 SHA 和 `git status`。若目标文件已被队友改动或哈希不符，停止并报告冲突；不得覆盖、reset、checkout 或假定快照仍然最新。

## 可恢复重建（使用新的、空的根目录）

不要删除任何现有树来“重建”。需要新副本时使用一个新的唯一目录，并且先校验当前主源是否仍代表该 manifest；如果主源已经改变，应基于本 manifest 的 `HEAD` 与归档，而不是覆盖主源。

```bash
repo=/mnt/sdc1/robodojo/GalaxeaVLA
snapshot=/mnt/sdc1/robodojo/behavior_dev/GalaxeaVLA_memlite_coordination_dev_20260908
new_tree=/mnt/sdc1/robodojo/behavior_dev/<new-unique-root>/source

git -C "$repo" worktree add --detach "$new_tree" 89f2322b4ad016e192437adc1a2c253b05bab246
git -C "$new_tree" apply --binary "$snapshot/dirty_tracked.patch"
tar -C "$new_tree" -xf "$snapshot/untracked_sources.tar"

sha256sum "$snapshot/dirty_tracked.patch"
git -C "$new_tree" diff --binary | sha256sum
tar -C "$new_tree" -df "$snapshot/untracked_sources.tar"
```

前两个 patch 哈希必须相同，且最后的 `tar -d` 不得输出差异。恢复过程不涉及 GPU、训练、数据重写或主仓库覆盖。
