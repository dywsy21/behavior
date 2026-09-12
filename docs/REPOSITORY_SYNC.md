# 仓库迁移与来源记录

状态：首次迁移已完成（2026-09-12）；本地首个迁移 commit 与后续证据 commit 均已 push，服务器干净 clone 已在证据核验 checkpoint 追到最新并核验。

本次迁移的可审计 receipt 与路径映射就是本文件（提交后固定为 `docs/REPOSITORY_SYNC.md`）；被忽略的归档内容位于 `artifacts/local-archive-20260912/root/`，不依赖 receipt 才能恢复。

## Git 入口

- `origin`：`https://github.com/dywsy21/behavior.git`，首次检查时仓库公开可见但页面显示为空，`git ls-remote --heads origin` 没有 refs；首次迁移 commit `69645b4220105ef1199fbbe90c889d4ae911ef6a` 与证据 commit `c384b0a2a2398baf80bb4f35a8beb4bd9dcdd296` 均通过普通 push 建立/更新 `origin/main`，没有覆盖或重写 GitHub 历史。
- `upstream`：`https://github.com/OpenGalaxea/GalaxeaVLA.git`，保存上游来源；当前 `upstream/main` 为 `89f2322b4ad016e192437adc1a2c253b05bab246`（`docs (readme): cite G0.5 arXiv paper`），其 push URL 已禁用。证据核验 checkpoint 的本地 `main` 与 `origin/main` 均为 `c384b0a2a2398baf80bb4f35a8beb4bd9dcdd296`。
- `robo` 不是本仓库的 Git remote。它只用于只读核对 `/mnt/sdc1/robodojo/GalaxeaVLA`；GitHub 协作不再通过 rsync/scp 覆盖服务器活跃源码。

## 迁移来源

来源目录：`robo:/mnt/sdc1/robodojo/GalaxeaVLA`。

核对时该目录为 `main`，基于上游 `origin/main`，工作树有 31 个已修改路径和 90 个未跟踪路径。复制的是该脏工作树中可追溯的源码、配置、测试、轻量脚本、文档、许可、assets 和项目元数据；因此这些改动被保留在本地，但不应被描述成已经合入 `behavior_dev` 的最新实验。

`/mnt/sdc1/robodojo/behavior_dev` 不是 Git 仓库，只包含多个实验快照/锁文件/运行目录；本次不从它直接合并或热改。主仓与 `behavior_dev` 的差距（包括新推理对齐 runtime 尚未进入 MAIN）必须在计划/服务器布局文档中明确记录。

## 本地布局与归档

- 上游项目源码直接位于仓库根的 `src/`、`configs/`、`scripts/`、`tests/`、`experiments/`、`assets/`、`tools/`、`skills/` 和 `licenses/`。
- 上游 `docs/architecture/`、`docs/data/`、`docs/deployment/` 保留原路径；上游的 `docs/README.md`、`docs/README_zh.md` 保留在 `docs/upstream/`，给协作仓库的 `docs/README.md` 让位。
- `docs/upstream/README*.md` 的正文保留，仅修正因目录迁移产生的 30 条相对入口（Quick Start、architecture、data、deployment）；相对链接核验为 30/30 可达。
- 本地旧顶层报告已移至 `docs/archive/`；迁移来源中未提交的协调/评估文档保留在 `docs/archive/remote-main-20260912/`。
- 本地实验、视频、诊断、补丁和附件整体可逆地移至被忽略的 `artifacts/local-archive-20260912/root/`；根代理完成读取后，`memlite-resume.L6rqZ6` 也已移入该目录（当前归档约 4.1 GiB）。归档目录不是可删除的垃圾桶。
- 服务器的 `.venv`、checkpoints、outputs、数据、模型/权重、缓存、日志、字节码和临时文件没有复制。服务器 `checkpoints` 绝对路径软链接也没有带入本地仓库。本地 `.gitignore` 还额外拦截常见权重/数组/数据集/视频后缀及 `receipts/`、`*.jsonl`，避免后续实验误入 Git。

首个 push 前核验服务器新协作入口 `/mnt/sdc1/robodojo/behavior` 不存在；原 `/mnt/sdc1/robodojo/GalaxeaVLA` 保持原位、原 `.git` 和脏工作区不变。首个 GitHub push 后才在该空路径 clone，并只通过共享原 `.venv`（不复制环境、不启动实验）。

首次 push 后已在原先确认为空的 `/mnt/sdc1/robodojo/behavior` 完成普通 clone，并在证据 commit 后用 `git pull --ff-only` 更新：证据核验 checkpoint 的 `main` 与 origin 均指向 `c384b0a2a2398baf80bb4f35a8beb4bd9dcdd296`，工作树干净；该 clone 的 `upstream` fetch URL 为 OpenGalaxea、push URL 为 `DISABLED`。`/mnt/sdc1/robodojo/behavior/.venv` 是指向 `/mnt/sdc1/robodojo/GalaxeaVLA/.venv` 的软链，仅复用原环境，不复制、不安装、不启动实验。clone 完成后重新核对原 MAIN 的 loopback listeners：`8772/8773/8776/8777/8778/8780/8781` 均仍监听（python/python3.10），未停止或重启服务。

## 运行环境边界

- 新 clone 的 `.venv` 只是复用原 MAIN 的环境；其中已有的 editable install 可能仍指向旧的 `GalaxeaVLA/src`，因此不能仅凭工作目录位于新 clone 就声称运行了新仓源码。
- 本次没有对共享环境执行 `pip install -e`，也没有启动实验或服务。后续若要运行，使用独立环境，或显式设置并核验 `PYTHONPATH`/实际 import 路径后再记录 receipt；不得在活跃任务上改写共享旧环境。

## 安全同步约定

1. 迁移完成前保留服务器原脏仓；不得在其上执行 pull、reset、checkout、force-push 或热改。服务器运行任务应使用停机后从 GitHub 建立的独立 checkout/worktree，并固定到明确 commit。
2. 本地同步前先查看 `git status --short --branch` 和 `git remote -v`，再 `git fetch origin --prune`；仅在工作树干净、分支有 upstream 且无分叉时使用 `git pull --ff-only`。发现脏改动时只 fetch、审查和协调，不覆盖内容。
3. 首次迁移提交 `69645b4220105ef1199fbbe90c889d4ae911ef6a` 只纳入经过检查的源码、配置、测试、轻量文档和协作规则；不纳入凭据、私人聊天、运行环境、数据、权重、视频、原始日志或海量诊断。首推前已复核暂存清单、秘密、大文件和 `git diff --cached --check`；证据追加 commit `c384b0a2a2398baf80bb4f35a8beb4bd9dcdd296` 已复核并普通 push。

## 最新核验 checkpoint

- 最终协作文档提交为 `49cced01e219c104155816c40ae3abc48e576034`，已普通 push 到 `origin/main`；本地仓库、GitHub `main` 和服务器 `/mnt/sdc1/robodojo/behavior` clone 均指向该 commit，工作树干净。服务器 clone 仍只用 `pull --ff-only` 更新，`upstream` 只读。
- 服务器冷实验归档的最终软链、内容校验、隔离副本移除、释放空间和恢复边界，见[轻量存储验收材料](storage/README.md)。原始 receipts/manifest 保留在 `/mnt/tmp1/robodojo-archive-20260912/manifests/`；公开仓库只纳入小型文本证据，不纳入权重、数组、视频或整机进程/环境日志。
- 原 `/mnt/sdc1/robodojo/GalaxeaVLA` 的 `.git`、31 个已修改路径和 90 个未跟踪路径仍保留；它不是本仓库的协作 checkout，也没有被 pull、reset、checkout 或 force-push。
