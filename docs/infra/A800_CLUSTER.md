# A800集群准备（INFRA-A800）

负责人：Codex。开始：2026-09-27（北京时间）。代码分支：`infra/a800-cluster-20260927`。

## 范围与安全

- 用户授权训练基础设施准备、BEHAVIOR 2026数据下载及通信测试，不含正式大规模训练。
- lc1–lc3：`10.19.7.1`–`10.19.7.3`，SSH用户`user`、端口22。lc4（`.4`）不启动任务。
- 共享盘预期`/data/workspace`，由`.3`提供；用户报告剩余约5T，须现场核验。
- 凭据仅交互输入，不写命令行、配置、日志或Git。VPN SOCKS仅监听loopback；保留SSH主机密钥检查，不修改全局路由/信任库。VPN网关为用户指定地址，观测证书自签且2023年已过期，不将其描述为已验证可信证书。
- lc-connect v0.1.0 Linux amd64包SHA256：`4eaa5461d69800c5b61d9ed807b5b89feea90d53e87ffa05960d186b6dcc3571`，已与release核对。

## 数据范围（15:41按用户最新要求收窄为RGB）

[官方2026数据说明](https://behavior.stanford.edu/challenge/dataset.html)：100任务、20,000演示。用户明确**不下载深度视频**；当前选定三路RGB＋全部动作Parquet、标注、元数据，官方清单实算**26,350文件、1,077,039,763,530B（1.077TB / 约0.980TiB）**。不另下raw HDF5。下面3.255TB全模态信息只保留作原始来源对照，不再是执行目标。

RGB视频5,286件共1,002,261,919,143B；排除`videos/observation.depth_linear.*`共11,807件、2,178,232,319,163B。下载器默认`--video-mode rgb`且回执明确scope；保留官方meta原样（仍描述depth），后续训练配置只能请求RGB，不要把depth缺失误判为下载失败。全部meta/annotation不删减，仍覆盖100任务。

## 实际硬件与TCP网络（2026-09-27 15:10北京时间）

前三台各8×A800 80GB PCIe，共24GPU；核验时无计算进程。Ubuntu22.04、驱动595.91.07；lc1为96逻辑核、lc2/3为128，内存约0.86–1TiB。GPU0–3/4–7分属两NUMA，均无NVLink。

bond0为2×10GbE的802.3ad LACP，但发送哈希是layer2。同一对节点的不同TCP连接仍可能落同一物理口。mlx5_0–3均DOWN/DISABLED；不擅自修改交换机、bond或启用未知IB网络。

以下为iperf3 TCP接收有效带宽（Gbps），每项10s＋2s预热，单项顺序执行，无GPU训练；“反向”是右边节点向左边节点发。下载环境安装有少量并发外网流量，故结果不是完全隔离实验。

| 节点对 | 单流正向 | 单流反向 | 四流正向 | 四流反向 |
| --- | ---: | ---: | ---: | ---: |
| lc1 / lc2 | 9.350 | 9.414 | 9.415 | 9.414 |
| lc1 / lc3 | 9.415 | 9.391 | 9.368 | 9.409 |
| lc2 / lc3 | 9.410 | 9.413 | 9.405 | 9.413 |

证据：共享`runs/network_20260927/*{,_server}.json`。iperf3 3.9由Ubuntu仓库包解压到自有`tools/iperf3`，没有全局apt install；每次服务只绑定指定内网IP/端口32163或32165，有timeout且一次连接后退出。NCCL实测见下，TCP不能单独证明DDP训练扩展率。

## 数据身份和空间

官方非gated仓库`behavior-1k/2026-challenge-demos`，固定revision `4f50b44796641a4d526a19d9aeadc8aa51e2f2c2`。官方逐文件元数据共38,157件，**3,255,272,082,693B（3.255TB / 2.961TiB）**，不是HF历史版本累计usedStorage。

- videos：17,093文件，3,180,494,238,306B，含RGB/depth。
- data：955 Parquet，74,558,696,633B。
- annotations：20,002文件，213,164,921B；meta：104文件，5,977,742B；另README/LICENSE/.gitattributes。
- 现场共享盘余5,462,339,481,600B；全量demos可容纳且预留1TiB后仍有约1.1TB额外余量，raw不在本次下载清单。
- 完整官方文件清单：共享`manifests/hf_official_files.json`，SHA256 `502c11870eb0bce85e2ad94a75f32d30aad5ae2963480f7b5f73f6c8cabf5b22`。
- 下载器在当前free之上保守扣除所有在途文件全量大小；临近空间门可能提前退出，允许降低并发后续传，不冒险突破reserve。共享盘其他用户写入不受本进程控制；停止接纳后已有最多8个下载可完成。本次实际额外余量约1.1TB，远大于在途体积。

## 环境状态

所有环境、代码、工具和数据均在同一共享根`/data/workspace/wsy/behavior2026`，**没有四份环境**。lc1/2原已有整个workspace的NFS4.2；lc3是本地ext4；lc4的workspace实际上是有他人文件的本地盘，故只把lc3同名任务子目录NFS4.2挂到lc4新建的空`/data/workspace/wsy/behavior2026`，没有遮住其他人的workspace。四节点读取manifest SHA一致。lc4只CPU验收，不占GPU。lc4本次为运行时挂载，未写fstab；重启后如目录不可见，管理员可对这个空挂载点执行：

```bash
sudo mount -t nfs4 -o rw,vers=4.2,hard,timeo=600,retrans=2 \
  10.19.7.3:/data/workspace/wsy/behavior2026 /data/workspace/wsy/behavior2026
```

隔离Python3.10.19、`envs/download`（huggingface-hub0.35.0）已可用。`src/behavior`固定b42c739作为editable安装源，禁止热pull；新任务改代码用独立worktree和明确源码路径。`envs/g05-py310-cu128`已按uv.lock完成267包安装、Torch2.7.1+cu128实际CPU导入通过；datasets3.6.0/transformers4.57.1保持锁版本，完整入口/解码/新环境GPU验证仍待。排除本轮不需要的仿真包/deepspeed/FA4，以SDPA为后备，不代表OmniGibson/ZeRO后端已安装。

安装v1网络超时、v2传递仿真依赖egl-probe缺CMake、v3两个NVIDIA wheel中断；均保留日志。v4将`pypi.nvidia.com,pypi.nvidia.cn`放入进程级NO_PROXY后55.53s准备＋16.20s安装完成，缓存复用，不改版本/驱动。PyPI预取两包过慢已停止，最终走锁定NVIDIA官方URL和SHA。环境冻结清单`manifests/g05-environment-freeze.txt`。FFmpeg4.4.2相关Ubuntu包仅解压到自有tools；通过`scripts/infra/activate_a800_training.sh`设置局部动态库路径，不能改全局LD配置。共享env今后有运行任务时不能uv sync/升级。

## NCCL进展

临时只读复用已有starvla的Torch2.7.1+cu126/NCCL2.26.2进行通信诊断，不改该环境；自有cu128环境另验。lc1默认PCIe P2P在首通信阶段挂起（约147s后只终止自有torchrun）；P2P能力矩阵全OK并不能排除此故障。仅设`NCCL_P2P_DISABLE=1`后8卡通过；16卡也通过，所有rank BF16前反向有限值检查通过。当前仅确认P2P路径相关问题，不把ACS/驱动某一项写成已证明根因。

| 规模 | 1MiB延迟ms | 16MiB | 64MiB | 256MiB | 256MiB algorithm GB/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| lc1 / 8卡 | 0.458 | 5.776 | 22.797 | 90.409 | 2.969 |
| lc1+lc2 / 16卡 | 2.747 | 15.767 | 59.095 | 232.117 | 1.156 |
| lc1+lc2+lc3 / 24卡 | 6.246 | 27.881 | 110.262 | 440.490 | 0.609 |
| 同24卡，仅allreduce强制Tree | 7.185 | 34.246 | 89.300 | 346.344 | 0.775 |

每项3次预热＋10次测量，取每轮最慢rank再取中位数，CUDA结果逐次检查；实际峰值分配约320MiB/rank。`bus_GBps`是NCCL惯用换算，不等于物理以太网流量，不能拿16卡约2.17GB/s的bus值冒充17Gbps网卡实测。原始JSON含完整rank→hostname/local-rank映射。三种规模全通过。下载/安装有并发外网和共享盘流量，未声称隔离环境吞吐或真实训练加速比。默认建议每机独立8卡；LoRA/长计算配方仍可后续实测多机步耗时，不一概否定所有跨机训练。

Tree对照在大消息改善但没有解决链路上限；小消息更慢，不作为全局默认。首次误将`NCCL_ALGO=Tree`施加所有collective，导致不支持Tree的AllGather初始化失败，旧失败日志保留；改为官方2.24+支持的`NCCL_ALGO=allreduce:Tree`后通过。生产默认不强制算法。建议使用**单节点8卡DDP**；每节点独立配方/seed可以并行，不必让三机每步同步大梯度。跨机全参前需要管理员排查可用IB布线/交换机、P2P挂起；不能靠改训练batch就把10GbE变成高速互联。

## 下载进展

v2首次100件/160.6MB核hash通过后，约1.2GiB落盘处遇HTTP大流中断，进程已退出，旧日志/完整文件/.incomplete保留。下载器已加最多5次网络重试（2/4/8/16s退避，权限/hash/磁盘错误不重试）；7项CPU测试与独审验收。Xet头RGB206,429,692B已过官方SHA`9f73b8c262f655e4c39193fdc07fa928c17f98e186879cd9df8b21c52ab91692`。新冻结`src/infra-a9be0cb`/a9be0cb74aea31be42b53a84292fee46faf725ff在lc2 tmux`behavior-data-20260927-v3`运行（PID989140），`logs/dataset-download-v3.log`/`runs/dataset_download_20260927_v3`；Xet分块、8文件×8分块并发、额外chunk cache禁用、顺序写，不依赖本地VPN持续在线。**全集未完成，不启动正式训练。**
