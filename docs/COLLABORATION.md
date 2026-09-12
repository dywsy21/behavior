# Git与三人协作

共享仓库：[dywsy21/behavior](https://github.com/dywsy21/behavior)。本地工作根为`/home/wsy/behavior`；服务器当前源代码与运行副本并存，见[目录表](SERVER_LAYOUT.md)。

## 每次开始

先读`AGENTS.md`和[当前计划](TEAM_PLAN.md)，确认任务owner和预算。

```bash
git status --short --branch
git remote -v
git fetch origin --prune
git pull --ff-only
```

最后一条仅对干净且已设置upstream的当前分支执行。新任务先同步main，再建自己的分支：

```bash
git switch main
git pull --ff-only origin main
git switch -c feat/low-fm-short-description
```

不要在有别人未提交修改的目录执行切换；保留当前工作，使用独立clone/worktree。feature分支已存在时，pull它自己的upstream，并明确查看/合并main更新；**pull feature分支不自动包含main的新提交**。禁止为解决分叉force-push或丢弃修改。

## 分支和交接

| 主线 | 分支示例 | 提交前必须给队友什么 |
| --- | --- | --- |
| A低层/集成 | `feat/low-fm-alignment` | 动作接口、测试、权重/源身份和局部闭环结论 |
| B高层/反馈 | `feat/planner-outcome` | 数据schema/白名单、标签依据、校准与切换结果 |
| C通用RL/仿真 | `feat/multitask-fm-rl` | 共享策略/奖励接口、可重放起点、跨技能验证预算和配对指标 |

每个任务一个owner，跨模块接口变更先在计划写出影响，再与对应owner协商。main由集成人串行合入，不让三个人同时覆盖服务器同一目录。代码修改至少由另一成员独立review；文档/整理按相应校验验收。

## 运行实验时固定commit，不热更新

服务器从GitHub拉取对应分支；完成检查后把实验绑定到准确commit、配置、父权重和数据清单。在独立worktree或快照运行，记录实际导入的源码路径。调试目录、主仓根和正在运行的快照不能混为一谈。

运行中的训练/模型服务可能在稍后才读取某个文件，因此“这个文件现在没打开”不是可以pull/移动它的依据。旧进程完成后再切换；新版本用新运行目录/明确端口。Git同步不复制运行环境和TB级数据。

## 每次结束

1. 更新任务板的状态、实际结果、证据路径和下一步；失败也记录，未运行/未验证的草稿明确标注。
2. 检查差异和暂存范围，只提交需要的源码/配置/测试/摘要；不要无检查地暂存整个目录。
3. 运行`git diff --check`及相关测试，检查秘密和大文件；commit并push自己的分支。将分支/commit交给集成人或建PR。
4. 权重/视频等留服务器，以路径和SHA关联；如本地需要看视频，放被忽略的artifacts目录，不进Git。

初始化迁移时原远端GalaxeaVLA是脏仓。它的上游来源、既有修改及正在运行的实验要保留；不可直接用新GitHub分支覆盖。迁移实施结果和安全同步入口以[服务器目录表](SERVER_LAYOUT.md)的最终记录为准。

## 建议的轻量实验摘要

```text
ID / owner / branch / commit:
假设与本轮唯一改动:
父权重 / 数据版本与split:
任务 / 实例 / seeds / 前缀与重置规则:
预计上限 / 实际步数与耗时:
主要指标 / 原始计数 / 失败类别:
结论与局限（诊断、局部成功或完整SR）:
完整日志、视频、checkpoint的服务器路径与SHA:
下一步：合入 / 再做一个指定检验 / 停止 / 申请扩预算:
```
