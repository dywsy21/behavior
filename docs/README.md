# 从这里开始

当前工作是**五任务小规模方法验证，为后续全任务大训练做准备**。不要把旧实验流水账中的“运行中/待办”当作当前状态。

- [Goal执行总计划与实时进度](plan.md)
- [三人分工、阶段进度和任务板](TEAM_PLAN.md)
- [Git同步、分支和交接](COLLABORATION.md)
- [面向50任务的通用RL方法计划](RL_METHOD_PLAN.md)
- [服务器代码/数据/权重/视频/归档位置](SERVER_LAYOUT.md)
- [本次源码迁移、归档映射与Git入口核验](REPOSITORY_SYNC.md)
- [安全磁盘归档、验收清单和恢复](storage/README.md)
- [仓库级工作规则](../AGENTS.md)

本地根目录保留协作代码；历史材料归档，重型实验文件被Git忽略但不代表可以删除。原GalaxeaVLA代码文档与上游许可保留；最新研究快照尚未完全集成的差异见任务板P0-02。

## 原项目技术入口

- [G0.5架构](architecture/g05_architecture_zh.md)与[输入输出](architecture/g05_io_zh.md)
- [训练数据结构](data/schema_zh.md)与[样本构造](data/samples_builders_zh.md)
- [服务部署](deployment/serve_policy_zh.md)与[MEM服务](deployment/serve_policy_mem_zh.md)

以上用于理解原仓代码，不代表其中每个默认值都适用于当前A3实验；准确运行约定仍以任务板和版本化运行记录为准。
