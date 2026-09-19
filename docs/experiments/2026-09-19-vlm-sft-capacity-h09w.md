# H09W：原生教师完整记录容量（仅CPU，未新物理）

负责人Astra；父独占harness/总计划。2026-09-19 08:46:17 BJT开始，≤600s CPU、0reset/控制/模型/训练。父已依据H09V完整真实证据批准本CPU票：唯一假设是100MiB记录限额不足，新增显式384MiB run配置能容纳原12宏的保守证据上界；不修改教师/动作/成功门，不把容量验证叫方法A/B。

## 固定证据和容量依据

H09V已退出：396prefix＋267native含hold、8完整macro，第9条动作前预留失败；目标末升18.4077mm仍未到30mm，零完整GRASP正BC。全部246＋3文件/75depth数组hash及连续ledger审计见[H09V报告](2026-09-19-vlm-sft-near-grasp-h09v.md)。每完整macro实际约10.2–10.3MB，但未来容量不能仅按这个压缩率估计。

固定720² head/480²双腕与原记录格式：capture上界9,276,026B、macro后验预留19,879,156B。公式 `12*(capture_bound+action_evidence_bound)+396*4096+4*MiB+3,126,989 = 358,805,493B`，含全部36capture、保守逐控制/record额度、prefix和已知标定压缩件；384MiB有余量，192MiB不是12宏最坏保证。未知传感器布局/标定或写出内容超过原界仍按已有门停止，不能覆盖缺失证据。

## 最小兼容实现

`native_teacher_capacity.py`：缺省/显式`near100`仍run100MiB/root384MiB，旧H09V模板/已发布授权原字节不变。只有显式`near_complete384`才接受run384MiB/root512MiB/跨旧新总768MiB，固定新根 `/mnt/nvme_tmp/robodojo_vlm_sft_20260919/h09w_native_complete`、旧根 `/mnt/sdc1/robodojo/behavior_dev/vlm_sft_native_teacher_20260919`。未知/混搭/非整数容量、错误或重复/漏计旧根拒绝。旧根实际约209.90MB全部保留，不搬迁删除，也不借新目录从累计中消失。

新预算继承原完整after+settle事务预留：每次普通写/预留及cleanup都重数旧根、保留run/root独立上限、旧根384MiB上限及跨根768MiB，旧根缺失/重叠就拒绝；预留未花完的空间也参与计数。cleanup只使用既有单独余量，不能突破跨根总上限。外部进程仍可能占用磁盘，原80GiB实际空闲检查不变。

collector仅改profile验证和构造预算对象；仍12macro/420新控制含初12停稳与末hold/900s/1reset/0模型训练。未知载荷门、1cm/3°、原4mm精确门、格单元一次CLOSE、同239cb591执行器、GRASP contact/held/lift/stability/hold全部未改。新template为inactive，源commit待父审绑定；旧prepared b146、spec6c2、seed/review06f与TRAIN排除仍严格。未创建NVMe目录或启动新进程。

## 验证与下一步

87 SFT/0.299s、冻结331 harness/5.069s通过；6新增组覆盖默认兼容/混合profile、真bool授权/旧源/各非容量上限、跨根预留期间旧数据增长、保留cleanup但不越总额、原根消失/重叠、实际writer选择及342MiB公式。所有near非容量数值上限也要求真int，拒绝True/1.0冒充reset数，不改变既有合法授权数值。真实本地396×23来源＋父P3完整seed/review helper通过（本地manifest4b3cd89a；同prefix f92baf54），错来源/错SHA拒绝；远端b146来源在H09V已完整factory实证，未借本地副本冒充新的远端调用。当前0新物理/训练，稳定源待父独立终审后才另放唯一容量验证。

后继仍以真实原生完整GRASP正例为依赖，不把当前8条隔离方向动作凑成训练数据。若容量验证取得成功并父图审通过，可前瞻登记GRASP-only停顿纠正SFT：先按TRAIN实例分组确定至少2个训练、2个独立留出来源，核原5%/开发实例排除；采样必须来自合法全开停顿点，并保留同一源所有失败。最小数据门应同时要求多个成功完整轨迹、至少两个训练来源含CLOSE与lift、覆盖实际用到的方向/姿态，不能只有重复UP。新checkpoint预定不超过400更新，固定终点；配对评估至少两留出实例、同起点/同执行器/同决策与控制预算，base/adapter各一次，另报告仅history/proprio规则的增量。样本数/确切实例与重置总额须在首条完整轨迹真实产率和候选来源时钟核后另登记；当前不承诺未获得的成功数据，不等待旧六类240广覆盖门，也不以loss宣称局部闭环收益。

## 21:27恢复部署与21:37唯一物理提交（北京时间，运行中）

早晨`f9916dbb2f5679c0b9a2d6fbd19480e222aa7b07`已提交推送，随后额度/compact中断；不是原600秒块一直运行。21:27新独立CPU票在robo维护clone显式fetch，新增干净不可变`git_worktrees/vlm_sft_h09w_f9916db`，未pull/改活跃源。远端87 SFT/0.712s、331 harness/15.323s通过，executor仍`239cb591f178099f20e9a9ba6d7cd3ce04aa5ccfe183840ebe1ff78ac2f40b9b`。真实factory仅加载`picking_up_trash/train/192/seed0`，396×23 float32前缀逐值一致，SHA`f92baf548a56b651e13e99f3519a6a5b1df98177fefde4c74dc1012a5f4a1447`；原b146准备、完整P3实际轨迹/种子/父审核及两239工程门绑定均过。CPU回执SHA`e4ed15d5d5ec0bca6255f058acb3ccaebf59139dc8ddf0db36ad620c5961ceee`、inactive授权SHA`d7a439b0e17b416d1e773aada9d0e1dadd15b8d4453cdade5c83fb807ac77d4e`，均在旧SDA根且双端校验到本地`artifacts/h09w-deploy-f9916db/`；执行入口对inactive明确拒绝。

21:35父另放单次物理：GPU1已被队友RL占用，父归档停止自身GPU3旧模型后将GPU3主仿真交本轮，队友副context不动。**GPU硬件选择不同于H09V，不宣称严格容量单变量A/B。** 新active授权`authorization_h09w_native_task1_v1.json`SHA`ea4c255c4988b83334415cd6486918d55447c2936650a439251e3c17af3b4d57`绑定f991、父新票及CPU回执，旧inactive未覆盖。21:37:50.330924 BJT／13:37:50.330924 UTC，唯一PID317744提交，run为`/mnt/nvme_tmp/robodojo_vlm_sft_20260919/h09w_native_complete/native_task1_v1`，日志相邻`native_task1_v1.log`，`launch.json`记录完整命令/资源。启动前GPU3余80,939MiB、仅保留队友313698副context；两盘余128,473,370,624B／2,936,187,805,696B，旧根209,906,360B继续累计。原task1/e310/TRAIN192/seed0、396前缀＋≤420新控制含初12停稳与末hold、≤12宏/900秒、run384MiB/新根512MiB/跨根768MiB/余80GiB及全部动作/物理成功门不变；0模型/训练，无自动重试。此处仅为真实提交，尚无完成/成功；即使局部oracle通过也先隔离，待父完整图像与控制审后才可作为BC来源。
