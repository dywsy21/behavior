# JEV-05：Jev全策略控制的三任务原始起点评测

负责人：Codex；分支 `feat/jev-control-20260929`。截至14:11，首例协议错误保留，第二例正常弃权结束；另四例待首次提交。运行源commit和实际结果见下文与计划，不把本文件当作评测完成。

## 问题与边界

主要假设：统一的感知—类型化决策—受限执行闭环，能让Jev在不同官方任务上独立选择计划顺序、手臂、搜索、操作、参考系与恢复，并接受官方物理成功判据检验。**不是训练实验，也不预设会成功。**

Jev固定 `jev-1.13.0`，Qwen3.8-27B只提供图像感知/定位；代码可以拒绝不安全或不支持的命令，不能代替Jev选择策略动作。所有任务配置来自公开任务要求；特权对象状态不进入actor。完整future plan的选择不代表已观察到目标或完成前置动作。

JEV-04旧task0（TRAIN138，64决策/2048控制）已经真实运行：10策略动作均属于Jev，169控制中包含最后1次安全HOLD，官方0/1。它是独立旧协议诊断，**不混入本轮96决策的分母**。旧失败显示搜索时大量HOLD，原因尚待匹配API输入诊断；本轮不得靠替换失败实例得到好看结果。

## 冻结样本与统一预算

机器登记是 `configs/semantic_robot/jev_multitask_v1.json`，包含每个window、机器人配置、任务契约的SHA；加载时同时验证字节和task/instance/split/seed语义。

| task | 完整官方目标 | TRAIN实例（seed均0） |
| --- | --- | --- |
| 0 | 打开客厅桌上的收音机 | 138、97 |
| 1 | 客厅3个汽水罐全部放入厨房垃圾桶 | 141、190 |
| 3 | 两份披萨各自留在原盘上放入同一冰箱；两个碗放入同一水槽；冰箱关闭 | 242、102 |

每例原始重置、0演示/保存动作前缀；最多96决策、3072控制、2400s动作阶段、3600s总＋60s清理；Jev请求上限288、观察服务上限215。每例只提交一次，6例固定，不静默重试/更换seed/条件筛选。TRAIN为开发评测，不是独立盲测，也不代表100任务泛化；沿用v3.9.1开发环境的官方物理成功判据，不冒充v3.9.2官方榜单提交成绩。

新执行代码需要同源task0/task3各一次工程门：每门24检查/1536控制/1200s动作/1800s总＋60s清理。门不计成功率。API全局额度获准不限，但本批为三任务各3次future-plan探针、最多54请求/1200s，0仿真/动作。合法弃权记为有效响应，不要求模型先选出某个“正确答案”才获准进入模拟器。

## 任务槽与决策所有权

task0保留用户指定右手拿收音机、左手按键的v3契约。task1/3使用v4槽：每个next_goal选项同时表达目标槽与手臂，如 `goal_2_right`。可交换分支顺序由Jev决定；抓放绑定同一对象、同一手；当前执行器只支持同时搬一个对象，不暴露其做不到的双物运输。披萨连盘双手水平搬运是已登记能力约束，不让代码偷偷挑另一种策略。

task3开冰箱在pizza transport前；关冰箱只依赖两盘已放入，不强制先处理碗。单持物条件下，pick只解锁对应place；放下后再由Jev选择下一分支。目标参考解析上限按计划长度，不再固定4次。

完整计划绑定：task config SHA、Jev原始next_goal回答、选择顺序、目标、手臂、每次规划调用号及整体plan SHA。逐执行/命令弃权行携带同一goal proof。启动器从registry取得配置路径，不能信任result自报的另一个文件。所有物理control与选择账本逐一核对；最终安全HOLD单独统计，不算策略动作或成功。

初始规划弃权、命令弃权、或合法的首次安全停止，都保留零策略动作结果，integration=false，不冒充接口动作已通过；它们不是网络/协议错误，也不会被移出完整任务分母。

## 运行和报告

新干净Git源上先执行 `probe_jev_multitask.py`；证据通过后 `run_jev_campaign.py --launch` 启动独立后台协调器，按固定顺序gate0→gate3→task0a→task0b→task1a→task1b→task3a→task3b串行。协调器8小时硬上界，每阶段有独立资源/时间监管；页面/SSH断开不导致重复提交。策略失败继续下例；基础设施、来源或安全协议校验失败则停在原槽，先诊断，不自动重置。

GPU1上运行模拟器与观察服务，分别CPU88–91、48–51；模型≤59392MiB、模拟器≤14336MiB，保留各卡8GiB余量、NVMe80GiB余量，不终止队友。新run `jev_multi_20260930_<stage>_v1` 位于 `/mnt/nvme_tmp/robodojo_agentic_20260925`，同名私有runtime在 `/mnt/nvme_tmp/robodojo_sim_runtime_20260925`。旧JEV-04的源/run不改。

`summarize_jev_campaign.py`列全6槽，校验统一来源、原始实例、预算、计划和control归属。报告每task原始x/2与总x/6，同时给integration是否通过、实际移动/全部策略动作数、弃权和失败原因。缺失/基础设施失败单列，不能写成已完成策略成功率。每例完整结果/视频保留，最后人审抽帧。

## 本地验证

- v4槽与完整计划第一包61项检查通过；独立穷举task1/3分别157/2875合法前缀、48/960完整计划，无死锁。
- 统一launcher/parser/registry/window校验、完整/弃权计划归属、命令弃权goal篡改检查已补。更新后的全套811项/40.813s通过（5环境skip）；其后又补后台协调器不重试/基础设施失败停止/重复claim拒绝2检查，与API准入和统计共10项目标通过。第二包独审通过，无剩余实质阻断；独审也独立执行10项目标测试。
- 此处为启动前CPU验证记录；后续实际API/工程门/仿真状态见下文，不把CPU证据等同于任务成功率。

12:27最终全套813项/48.213s通过，5环境skip；implementation `403c74df8227e51b67203428e3033c3725f2bc56fe15e82fc4bbb54cbfd1ca69`，registry `40454b2a560960198fe2b9e745078ccf6d1f11ec5dfa3f2ede1c294268d760e2`。

## 实际执行进度

12:31：Git冻结4698b0aa37edf5b2758286fa3a2d5dec3777533e已robo部署，sim Python下71目标检查和六window实际SHA/身份全部通过。真实API九次规划均完成，54/54有效请求，result SHA `71cef4f9e7f2258b251b82dc5cd2132709ebe09db874c9be46b2ba15007729be`；19件回执在本地`artifacts/jev-control-20260929/multitask/api_v1`逐件双端SHA核同。后台协调器87628已运行，首门88005/88021正常实控；六例SR仍待。

12:52：首gate0已completed/24检查/440controls/112同步采集，全部freshness通过、gate_ok=true；总1217.852s/动作575.628s/清理0.372s。result SHA `46d31e8864626dd096e5b324bd3d64cf7f0109618d3025a8470ce4724a991a91`。协调器自动接gate3监管129152/actor129211，六例尚未提交；工程门不进入任务成功率分母。

13:10：gate3也completed/24检查/440controls/112同步采集，全部freshness通过；总1129.585s/动作573.501s/清理0.380s，result SHA `ef60c1ec22ff1febd86fa88d3bfdc28d0625fc693e8b7cb1b5dcd86f35cec8e2`。后台已自动启动首正式task0a监管140071，原始TRAIN138/96决策统一预算，实际结果待。

13:24：gate0/3各9件回执与视频均已取回本地`artifacts/jev-control-20260929/multitask/{gate0,gate3}`、双端SHA验全，本人各看7帧contact，无黑传感器帧；工程门不算SR。task0a已实控至control59/decision3，观察模型140499/actor140567正常，六例完整结果仍待。

## 首例协议失败（13:26结束，13:30核验）

task0a在decision6的第17次Jev调用（command）触发`Choice disagrees with distribution`。截至失败，16次响应通过、1次拒绝、7次观察；实际策略为1次底盘micro前进和5次HOLD，共102控制；异常处理再发1次安全HOLD并完成，所有自有进程退出。监管总947.842s、清理0.847s；`gate/failure.json` SHA `1b23475b7ae678ca9a8f657d93bb64a62bc4bc32cefceea98d470dd776d5f274`。

13:55额外核对原失败包的逐动作归属：6个已执行动作的command_call均匹配durable validated回答、action/proof一致、freshness过；trace实际103条control连续无遗漏，前102全部落在对应决策范围，最后1条明确after_exception safety_stop。仅证明失败前的部分执行由Jev选择，不将失败包拼成完整result或赋予官方成功值。

协调器按约定停止，其余五例未提交。**当前是1例接口/协议错误、0例完整任务终态，不是0/6成功率。** 同源统计器返回overall completed=0、infrastructure_failures=1、pending=5、success_rate=null。失败例不会被删除/替换为成功结果。

[TypeSafe Choice官方契约](https://docs.typesafe.ai/primitives/choice)和[HTTP API定义](https://docs.typesafe.ai/api)都要求choice对应最大概率，当前argmax校验有依据；文档提到的“跨不同问题概率不满足逻辑恒等式”不等于同一Choice可违背此契约。旧client只记录拒绝事件，未保存原概率，不能断言原差值、舍入或供应商内部原因。

补丁只增加脱敏数值诊断和请求/响应SHA，不改变选项、不降低校验、不重试执行。固定30次/1200s无执行probe从最后有效intent回执（SHA `1f1b0a47e22ed5c7e361c9d77af1ac0bb2977dfdfb24e29587b0e75343c220ec`）按原question builder重建command，核对原ledger的20184B请求长度。它复测的是重建输入产生的**新响应**，不是找回原拒绝响应；0仿真/动作/重置/训练，不记SR。

13:41失败全包326件已双端SHA核同，位于本地`artifacts/jev-control-20260929/multitask/task0a_infra_v1`，含全部RGB-D/逐决策/3.4s视频；本人看7帧，仅首次微前进＋保持，没有目标抓取。原state实际已RECOVER、选中的intent仍search（probe预检已纠正常量，不改变输入）；实际重建20184B/16选项、SHA `e1f9e1b252230df08ed9493cae794b86321d0c198cc5313d953799435de9d613`。30次API新结果待。

13:50诊断结果：独立ac74d9b源上的`jev_choice_contract_20260930_v1`完整30/30有效，27次HOLD、3次base forward micro，214500输入/5850输出tokens；全34件本地`multitask/choice_contract_v1`双端SHA核同，result `390e1db0d29aa35540323cfaaef65815ab453b85daf2a431fbe4bd175c2f6096`。全部请求SHA一致，未重现argmax不符，但不据此否定原错误或宣称供应商已修复。没有任何新动作/SR。

完成诊断后继续原未提交五槽：主线程逐例首次提交，不重启原协调器、不重跑/替换task0a、无新sim预算；仍4698b0a原源/同一准入证据，诊断补丁不进入本轮actor。每例官方终态、协议错误和未提交状态分开报告，task0完整率与六例总率在分母不齐时仍null，不用仅有效例的子集掩盖可靠性损失。

13:52接续task0b TRAIN97/seed0/0前缀已actor_running：监管183698、observer183786、actor184185，原GPU1/4698b0a/统一预算。task1/3四例尚未提交。旧task0a的16有效Jev请求总2.010s，7观察服务总133.633s/0输出截断，七次均报target不可见/无hazard；与图审一致，不能把此次原地等待主要归因于API太慢。

## 第二例task0b：正常弃权，官方未成功

14:11–14:19核验TRAIN97的唯一原始起点运行。监管completed、actor exit0，总906.578s/动作328.364s/清理0.796s；官方success=false、terminal=false、goal0未满足，stop_reason=JEV_ABSTAINED。21次Jev请求全部有效、9次视觉观察；8个策略动作（2底盘前进micro＋6 HOLD）全部经durable选择/完整计划/逐control归属复核，132策略controls＋1安全停止，integration=true。第9条decision是命令弃权，不算执行动作。

全部413件已本地`artifacts/jev-control-20260929/multitask/task0b_v1`逐文件双端SHA核同。result `443c46e6013545e7f4268e636ee34edb241638ecd961509655dcfe639e0fc6ec`，ownership `c6e86a95f0f893037bfc0b9f91a9fda55e1f095165709fd2bd77a5f974f89385`。本人查看4.4s视频7帧：头部面对电视/墙，腕部主要被本体遮挡，两次微前移后未搜索到收音机、未抓取。9次观察均不可见、无token上限截断，感知合计146.723s；与图审相符，不将失败归因为模型“看见了却不抓”。

当前task0为**1次完整失败＋1次协议中断**，原两例完整SR不报数值，不混入旧JEV-04。14:13仅首次提交原下一槽task1a TRAIN141/seed0，同4698b0a、统一预算；监管194597/observer194749/actor194814，结果待。余task1b/3a/3b尚未提交。

## 旁路：旧搜索停滞输入的敏感性，不修改本轮actor

只使用JEV-04 decision3保存请求，源SHA `eefd27fccc3cb868aa351404eb164e3701853116e9a1085521a1e2be90b33aa5`。当时Jev已选择search、目标不可见、15动作均通过预检；底盘yaw候选实际存在，通用rotation字段却为false，且1/3/8°转动均尚未跨24-bin覆盖的下一中心。

独立探针d40334879da1e407c44516820966d9336dc5490c保留原16选项（含abstain），交错A/B/C共9调用、240s、零控制/仿真/训练。A完整原输入；B仅附加“世界搜索以获取信息为目的、量化覆盖未跨bin不等于图像不变”的解释，保留所有安全条件、不指定方向；C仅根据现有yaw候选将泛化rotation标签改true。当前在跑的4698b0a actor未采用这些改动。

| 条件 | 原始计数 | 平均HOLD概率 | 平均yaw_plus micro概率 |
| --- | --- | --- | --- |
| A 原样 | 3/3 HOLD | 0.2967 | 0.0667 |
| B 世界搜索解释 | 3/3 yaw_plus micro（1°） | 0.1700 | 0.2800 |
| C 仅rotation标签 | 3/3 HOLD | 0.2467 | 0.0933 |

这是**同一真实停滞输入对问题表述的敏感性证据**，不是完整任务效果或泛化检验。B仍只选择1°微转；是否能在有限时间找到目标、改善成功率没有验证。C单独不足，不能将此标签认作唯一根因，也不能从概率直接断言内部推理机制。本轮不围绕这一实例无限调提示、不改变已登记六例版本。

本地v1在第2请求网络失败后停止（2 attempts/1有效，唯一A选择HOLD），原失败保留且不混作v2连续成功；同脚本改由robo网络路径的新v2全部9/9有效，60,471输入/1,755输出tokens。v2的12件完整证据在`/home/wsy/behavior/artifacts/jev-control-20260929/multitask/search_sensitivity_v2`，逐件双端SHA核同、durable ledger核过；result SHA `1829e9e85c80a7062b63b1ef02d4cc0efe433d2c394e8ca81ec0e47747147bc5`。原v1 SHA `812fe92eea196d671dd03b756096a2c72d900822fb57dc785a01fc8f93ef97fb`。
