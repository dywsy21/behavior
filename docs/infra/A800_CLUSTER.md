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

## 实际硬件与TCP网络（2026-09-27 15:10北京时间）

前三台各8×A800 80GB PCIe，共24GPU；核验时无计算进程。Ubuntu22.04、驱动595.91.07；lc1为96逻辑核、lc2/3为128，内存约0.86–1TiB。GPU0–3/4–7分属两NUMA，均无NVLink。

bond0为2×10GbE的802.3ad LACP，但发送哈希是layer2。同一对节点的不同TCP连接仍可能落同一物理口。mlx5_0–3均DOWN/DISABLED；不擅自修改交换机、bond或启用未知IB网络。

以下为iperf3 TCP接收有效带宽（Gbps），每项10s＋2s预热，单项顺序执行，无GPU训练；“反向”是右边节点向左边节点发。下载环境安装有少量并发外网流量，故结果不是完全隔离实验。

| 节点对 | 单流正向 | 单流反向 | 四流正向 | 四流反向 |
| --- | ---: | ---: | ---: | ---: |
| lc1 / lc2 | 9.350 | 9.414 | 9.415 | 9.414 |
| lc1 / lc3 | 9.415 | 9.391 | 9.368 | 9.409 |
| lc2 / lc3 | 9.410 | 9.413 | 9.405 | 9.413 |

证据：共享`runs/network_20260927/*{,_server}.json`。iperf3 3.9由Ubuntu仓库包解压到自有`tools/iperf3`，没有全局apt install；每次服务只绑定指定内网IP/端口32163或32165，有timeout且一次连接后退出。**当前不能仅凭TCP带宽宣称DDP扩展率，待NCCL。**

## 数据身份和空间

官方非gated仓库`behavior-1k/2026-challenge-demos`，固定revision `4f50b44796641a4d526a19d9aeadc8aa51e2f2c2`。官方逐文件元数据共38,157件，**3,255,272,082,693B（3.255TB / 2.961TiB）**，不是HF历史版本累计usedStorage。

- videos：17,093文件，3,180,494,238,306B，含RGB/depth。
- data：955 Parquet，74,558,696,633B。
- annotations：20,002文件，213,164,921B；meta：104文件，5,977,742B；另README/LICENSE/.gitattributes。
- 现场共享盘余5,462,339,481,600B；全量demos可容纳且预留1TiB后仍有约1.1TB额外余量，raw不在本次下载清单。
- 完整官方文件清单：共享`manifests/hf_official_files.json`，SHA256 `502c11870eb0bce85e2ad94a75f32d30aad5ae2963480f7b5f73f6c8cabf5b22`。
- 下载器在当前free之上保守扣除所有在途文件全量大小；临近空间门可能提前退出，允许降低并发后续传，不冒险突破reserve。共享盘其他用户写入不受本进程控制；停止接纳后已有最多8个下载可完成。本次实际额外余量约1.1TB，远大于在途体积。

## 环境状态

隔离Python3.10.19、`envs/download`（huggingface-hub0.35.0）已可用。新Git checkout在`src/behavior`，固定b42c739安装期间不pull。`envs/g05-py310-cu128`按仓库uv.lock安装中；保留Torch2.7.1/cu128、datasets3.6.0、transformers4.57.1，排除本轮不需要的仿真包/deepspeed/FA4，用SDPA，不代表所有仿真/ZeRO后端已安装。第一次依赖下载30s超时，已保存旧日志，第二次设300s单请求、4并发并复用已校验缓存。

本轮暂无数据全集完成、NCCL通过或正式训练成功证据。
