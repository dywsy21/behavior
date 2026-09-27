# A800集群准备（INFRA-A800）

负责人：Codex。开始：2026-09-27（北京时间）。代码分支：`infra/a800-cluster-20260927`。

## 范围与安全

- 用户授权训练基础设施准备、BEHAVIOR 2026数据下载及通信测试，不含正式大规模训练。
- lc1–lc3：`10.19.7.1`–`10.19.7.3`，SSH用户`user`、端口22。lc4（`.4`）不启动任务。
- 共享盘预期`/data/workspace`，由`.3`提供；用户报告剩余约5T，须现场核验。
- 凭据仅交互输入，不写命令行、配置、日志或Git。VPN SOCKS仅监听loopback；保留SSH主机密钥检查，不修改全局路由/信任库。VPN网关为用户指定地址，观测证书自签且2023年已过期，不将其描述为已验证可信证书。
- lc-connect v0.1.0 Linux amd64包SHA256：`4eaa5461d69800c5b61d9ed807b5b89feea90d53e87ffa05960d186b6dcc3571`，已与release核对。

## 数据范围

[官方2026数据说明](https://behavior.stanford.edu/challenge/dataset.html)：100任务、20,000演示，LeRobot demos约3.27TB，另有约1.44TB raw HDF5重放包。本次优先完整demos（含RGB/depth/动作/元数据/标注）；raw不是训练必需，不在剩余空间未经核算时额外下载。实际清单、revision、体积与可读性待核，不将官网近似大小当验收结果。

## 状态

客户端包已校验；VPN登录、节点/GPU库存、数据下载、环境安装、网络/NCCL性能均未完成。
