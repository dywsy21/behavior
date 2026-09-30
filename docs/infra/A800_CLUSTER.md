# A800集群准备（INFRA-A800）

负责人：Codex。开始：2026-09-27（北京时间）。代码分支：`infra/a800-cluster-20260927`。

## 当前可用入口

**2026-09-30 13:31北京时间八卡短测：** 单帧三RGB、真实A4低层AE＋r8 LoRA、FM四噪声，global64/128的缓存计算吞吐47.69/84.19观察/s，峰值allocated28.37/36.11GiB/卡；两臂各30临时更新后退出，0新checkpoint。100任务CPU读取代理热轮150.91窗/s，不是正式MEM loader。源/结果/时间外推见[本轮报告](../experiments/2026-09-30-memlite-oneframe-a800-benchmark.md)；共享env未修改，百任务训练和完整数据/源码准入仍未放行。

**2026-09-30 13:02北京时间权重完成：** 完整A4已在`models/memlite-a4-20260912/step_2500.pt`原子发布；16,581,363,550B、SHA `6186704788c27c9fae3502c884df0e259de5242ee8690fe578dcbc1f2632f269`源/本地/目的全同。14配套资产也已校验。此更新覆盖下条上传状态，不代表模型前向/八卡测速已通过；高层B-final未迁，本轮没有G0.5替代下载。

**2026-09-30 12:43北京时间旧权重同步：** A4低层FM＋r8 LoRA已从robo完整中转到本地并核SHA，当前上传共享`models/memlite-a4-20260912/step_2500.pt.partial`；目的完整SHA通过才发布`step_2500.pt`，未到最终名之前不可用于测速。源/本地SHA `6186704788c27c9fae3502c884df0e259de5242ee8690fe578dcbc1f2632f269`、16,581,363,550B。`models/action_tokenizer.pt`已源/本地/目的完整SHA同（`5088f64a5452a60bbc8cac90ee7d79c156f1d14540bc3139aa7060f862dddace`，506,886,775B）；A4目录的五份配置/统计/回执及`models/qwen3_5_2b_base_processor/`八件也源/目的SHA同。原A4配置中的robo绝对路径仅归档，不可原样启动；后续须新源码和显式资产路径覆盖。此为资产同步，不是模型前向、吞吐或正式训练通过。

**2026-09-30 11:18北京时间更新：下载完成，0缺失。** lc2终端已显示26,350/26,350、`下载完成！`并返回bash，旧PID1032615退出。全部26,347件RGB/动作/标注/meta训练文件按官方清单路径和大小核对通过；仅仓库`.gitattributes`大小不同（2560/2504B）。约1.077TB已落盘，真实snapshot根不变。见[9/30状态](results/2026-09-30-dataset-status.json)；**全内容hash与正式训练入口准入尚未完成**，本轮未自动开训、重启下载或改环境。以下9/29下载中记录保留作历史对照。

**2026-09-29 21:53北京时间只读复查：数据尚未下完。** 后续任务已改ModelScope `fduTristin/2026-challenge-demos@master`，lc2 tmux`behavior-data`/PID1032615运行中；21:52逐路径/大小核约1.013/1.077TB（94.044%），尚缺337段头部RGB约64.147GB，21:53终端26,017/26,350且仍推进。动作/标注/meta/双腕RGB已按大小齐全，但未重做全内容hash；`.gitattributes`有一处大小不同。状态证据见[JSON](results/2026-09-29-dataset-status.json)。

真实数据根现在是`/data/workspace/wsy/behavior2026/datasets/2026-challenge-demos/datasets/fduTristin--2026-challenge-demos/snapshots/master`。不要把ModelScope的外层cache_dir当旧HF平铺数据根。下载完成后须按固定官方manifest核内容身份，再核reader根；本轮未改配置、未重启下载或训练。原alpha/后续hf-mirror任务无complete且已退出，下文9/27下载进展只作历史证据，不是实时状态。

本次lc-connect认证通过后，默认底层socket走另一VPN网卡超时；实测`curl --interface eth2`可达，再仅给lc-connect加`--bind-interface eth2 --disable-multi-line`恢复，SOCKS/HTTP仍只绑定loopback1080/1081。没有修改系统路由/其他VPN；网卡名是本机本次观测，不应在别的机器盲用。凭据只交互输入，不写本文件。

9/27已验收的训练基础环境、四节点共享路径、真实RGB reader和前三节点24卡烟测保留；**没有据本次状态检查启动正式训练，也未重新核验四节点环境**。9/27共享env editable源码固定`a5c9821`，当时四节点不额外设置PYTHONPATH时均导入同一路径：

```bash
cd /data/workspace/wsy/behavior2026/src/infra-a5c9821
source scripts/infra/activate_a800_training.sh
# 只激活环境，不启动训练
```

日常Git协作和最新文档入口是`/data/workspace/wsy/behavior2026/src/behavior`，跟踪`infra/a800-cluster-20260927`；在env已移走且确认没有运行者后才做干净ff-only同步。它不承担当前下载/训练运行，运行者继续使用冻结worktree，禁止对这些运行源pull。

数据配置：`configs/data/behavior2026_r1pro_rgb.yaml`，100任务、三RGB、本地只读禁止隐式Hub补齐、沿用23D原动作/61D原状态映射。正式训练前还要完整下载hash回执、顶层Mixture首样本、选定checkpoint/配方/预算；不把下方单episode reader门当完整训练step。9/27的`models/`原为空，9/30已按上方记录同步旧A4及配套资产，没有新G0.5下载。

历史alpha镜像v5日志`logs/dataset-rgb-alpha-v5.log`、回执`runs/dataset_rgb_alpha_20260927_v5/`保留。该旧run未完成，不能用它判断新ModelScope作业；新数据需独立对照固定官方RGB清单核路径/大小/内容，不按进程存在或文件数百分比当字节进度。

## 范围与安全

- 用户授权训练基础设施准备、BEHAVIOR 2026数据下载及通信测试，不含正式大规模训练。
- lc1–lc3：`10.19.7.1`–`10.19.7.3`，SSH用户`user`、端口22。lc4（`.4`）仅CPU导入/读取验证，不启动GPU负载。
- 共享盘由`.3`提供，已现场核验约5TiB可用；任务目录映射见环境状态。
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

隔离Python3.10.19、`envs/download`（huggingface-hub0.35.0）已可用。最初`src/behavior`固定b42c739用于安装；所有验收进程结束后已将自有env editable重新指向冻结`src/infra-a5c9821`。此后才将无运行者的`src/behavior`ff-only同步作为协作文档入口；b42c739历史仍在Git，其他旧实验worktree不改，不是热pull运行源。新任务用独立worktree并为该进程显式设置`PYTHONPATH=<new_source>/src`，不要为切代码反复重装大家共用的env。`envs/g05-py310-cu128`按uv.lock装267包，再补一个NPP运行库；Torch2.7.1+cu128、datasets3.6.0、transformers4.57.1、peft0.18.0、TorchCodec0.4.0+cu128保持版本。排除本轮不需要的仿真包/deepspeed/FA4，以SDPA为后备，不代表OmniGibson/ZeRO后端已安装。

安装v1网络超时、v2传递仿真依赖egl-probe缺CMake、v3两个NVIDIA wheel中断；均保留日志。v4将`pypi.nvidia.com,pypi.nvidia.cn`放入进程级NO_PROXY后55.53s准备＋16.20s安装完成，缓存复用，不改版本/驱动。PyPI预取两包过慢已停止，最终走锁定NVIDIA官方URL和SHA。环境冻结清单`manifests/g05-environment-freeze.txt`。FFmpeg4.4.2相关Ubuntu包仅解压到自有tools；通过`scripts/infra/activate_a800_training.sh`设置局部动态库路径，不能改全局LD配置。共享env今后有运行任务时不能uv sync/升级。

TorchCodec的cu128 wheel即使CPU解码也链接NPP，首次真实检查报`libnppicc.so.12`缺失；已补官方`nvidia-npp-cu12==12.3.3.100`，精确wheel/SHA在`configs/infra/a800-runtime-extra.txt`，仅加局部npp/lib路径，没有系统CUDA/驱动安装。重建env时在uv sync后执行`uv pip install --python <env>/bin/python --no-deps --require-hashes -r configs/infra/a800-runtime-extra.txt`。最终包清单`manifests/g05-environment-final-20260927.txt`保留全部版本和editable路径。

验收证据（共享`runs/environment_20260927`）：

- 四节点`lc{1,2,3,4}_components_v2.json`：真实依赖与两G05 policy导入、官方manifest/3文件SHA、32行23D动作/61D状态、TorchCodec三帧头RGB720×720，全部passed、零CUDA初始化。
- `lc3_rgb_reader_v2.json`：原始六视频meta不改，隔离symlink视图只给三RGB＋首Parquet；正式RGB配置的时间查询经过实际LeRobotDataset初始化和episode0首样本，动作32×23、状态1×61、头3×720×720/两腕3×480×480、0Hub调用/0CUDA全部passed，SHA`679e27f065c6461bb4ea5995fa01a661688285b2f1bd063b2c269f0dadb6e544`。首版验收脚本将单帧CHW误认成TCHW已纠正，失败日志/视图仍保留，不更改实际张量协议。
- 固定a5c9821服务器9 reader回归0.091s通过、独审通过；下载器8回归也通过。新RGB scope贯穿cache检查、Hub allowlist、实际解码，`local_files_only`使缺已选视频报错而不是自动补齐/静默跳过来源；None保持原全相机兼容。
- 本地仅存小结果`artifacts/a800-setup-20260927/{network_20260927,environment_20260927}`，关键回执双端SHA一致，没有复制训练数据。

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

新自有cu128环境最终各节点8卡复验（每尺寸3预热＋3测量，≤150s）：lc1/2/3均exit0，BF16前反向/归约数值全部通过，256MiB中位90.378/90.512/90.819ms，峰值320MiB/rank。`own_cu128_lc{1,2,3}.json`，三个probe已退出，没有正式训练；lc4GPU未测。

## 下载进展

**v5镜像直连运行中，v4已停止。** v5固定`075c5d36ba8c3601d218cac33491b64233ac30d8`、`src/infra-075c5d3`；lc2 tmux`behavior-rgb-alpha-20260927-v5`，python991067/pane991065，日志`logs/dataset-rgb-alpha-v5.log`、回执`runs/dataset_rgb_alpha_20260927_v5`。8并发/1TiB reserve/固定1.077TB RGB scope，SSH断开仍继续。启动前本地10测试0.046s、服务器10测试0.349s均过，独审通过。16:12已有新文件写入并通过官方hash，13,400件/2,199,773,897B完成校验；前13,200左右是复验旧完整文件，**不把启动后几秒的复验吞吐当互联网下载速度**。

v4冻结`src/infra-2023037`，原lc2 tmux`behavior-rgb-20260927-v4`/python990576已停止，旧run/log保留。用户16:06要求切镜像后TERM旧python，最后日志13,200件/2,198,439,397B，无depth；小标注件数多，按字节约0.20%。原RGB文件/partial全部保留复用。

新路线只将下载数据endpoint改为`https://alpha.hf-mirror.com`，不信任镜像提供的内容哈希。启动需`--official-manifest manifests/hf_official_files.json`，先核固定官方manifest SHA502c…5b22，再逐文件核size/SHA；没有该清单就拒绝镜像入口。该进程清除大小写HTTP/HTTPS/ALL_PROXY，`NO_PROXY=*`，`HF_HUB_DISABLE_XET=1`走HTTP直连（不另协商Xet CAS）。实际HF客户端2504B仓库文件在1.312s通过官方Git blob hash；用户多源速度对照不作为持续速度承诺。没有改系统代理/既有xray服务，不启动正式训练。

16:13真实进程环境和连接复核通过：代理环境键全无、NO_PROXY=*、禁Xet；8连接全部从10.19.7.2直达153.121.43.79:443，无127.0.0.1:10809。回执14,167件/2,205,646,710B，其中897新路径/6,747,826B不在v4回执中；证明镜像实际新下载，而不是只改配置/重验旧文件。当前在小annotation阶段，不用这一阶段字节率估算全部视频耗时。

旧v2遇HTTP中断后退出；旧v3在用户改范围时已由主线程TERM，保留3,627件/5.270GB回执。此前17件/3.194GB深度是旧遗留，不删除、不继续下载、不计入新scope；官方metadata仍声明depth属预期。全部旧日志/完整文件/`.incomplete`保留。新run完成只证明RGB选定集合齐全，不证明目录没有历史depth。
