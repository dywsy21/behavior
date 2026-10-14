# 内部全量评测：异常与续跑口径

2026-10-13，负责人Codex / SUBMISSION-FULL-EVAL。用户要求优先内部评测，模型交付暂缓。本页不授权外部上传、换权重或新训练。

## 冠军续评后的MEM依赖与原件收集

独立`queue_memlite_after_champion_continuation.py`只替换已退出、零阶段、明确因原冠军故障停止的旧等待队列；必须绑定原用户优先级授权、原冠军root manifest、诊断封存、当前冠军续评和原MEM v7，不修改这些登记。通过外部checkout只读验证c978冠军协调源与ae86 MEM协调源；真正启动MEM仍执行ae86原入口/原51c actor。新队列独立名称、共用MEM owner文件锁，既有MEM status/STOP或GPU被占用一律拒绝接管。等待时不导入模型、不用GPU。

冠军全部1000案例核账（真实有效原件＋明确缺失的执行异常）且八worker及各代模型退出、参数不变后，`package_champion_continuation.py`重审所有历史原命令/物理回执/seed/官方时限、原JSON和MP4 SHA/帧数、整批异常及行政中断。原字节ZIP只装真实官方JSON和原wrapper/robot/transport；缺失清单、固定1000分母统计、原配置和运行登记单独保留，不造JSON补零、不挑选重跑结果。归档完成且GPU实际空闲才启动原MEM576剩余，MEM终态亦调用原件收集工具；任何未知失败停止后继，无自动重试。此为内部资料收集，不是模型交付、视频托管或提交。实际部署、队列PID与双端测试见plan，不把合成测试称为真实全量完成。

## 2026-10-14 冠军：下一批实例恢复时物体不可用

18:44:50 CST，原冠军切洋葱首四例已经各9600控制正常完结；305–308开始官方`load_fn → load_batch → _load_instance_state`时，`object_scope[tro_key]`对应对象为None，`obj.load_state(tro_state, serialized=False)`抛AttributeError。日志没有具体slot/物体，因此不猜测缺失对象原因，也不把它当FM推理失败。18:45保护停止全局及MEM依赖队列。八份终态权重均不变，原STOP和所有原件保留。

独立入口`continue_champion_final.py`的`freeze/prepare/worker/supervise/report`采用与MEM相同的原件、行政中断及缺失披露口径，但保持冠军自己的原9f服务/模拟器、8份配置、final8000 EMA、FM20/chunk20和23维控制。不把MEM模型/observer混入冠军。已完成和已诊断错误不再执行；原任务顺序/2或4实例批量保留，未启动后缀独立登记。未知错误、混合异常、未提交官方JSON、改权重/配置/超时或缺终态回执必须拒绝。

新编排复用FIFO两加载槽，队列等待不是执行超时；原30分钟无模型进展门与官方episode时限均保留。每卡独立模型/端口/会话，已诊断的批次退出后核模型参数再为不同的未尝试实例新建服务，不允许让同一实例改答案。最终有执行异常时只能报告真实原件与缺失、completed-only统计和固定1000分母保守值，不能冒称1000有效官方JSON。目标机验收、实际封存、恢复及MEM后继/最终打包状态以plan为准；本机144回归不替代实际运行或另一成员review。

## 2026-10-14 用户选择冠军优先：行政暂停与反向依赖

13:42 CST用户明确先冠军全量、再续MEM。原resume_v6和零阶段冠军queue-resume-v6按各自STOP有序退出，所有worker/model/sim已退出，8份最终权重/observer不变。原冠军v2/final8000、9f actor、1177 FIFO编排于13:48实际启动；已有TRAIN2/4工程门只验签，不重复执行，不改seed/预算/模型。

新增`freeze-continuation --authorization PATH`专门验证停机前的授权收据：准确绑定两个manifest、原存活MEM/等待队列PID、授权先于STOP/终态、队列零阶段且已行政停止。未经此显式参数，原故障封存仍要求实际诊断错误；所有原件/权重/物理身份/未知异常拒绝门保留。独立物理错误即使与行政停止同时发生也不得列为可重跑。

新增`queue_memlite_after_champion.py`只接受上述seal派生的新MEM清单，冠军须完整1000、八worker终态/权重/原JSON与物理回执/内部归档相符、原PID全部退出且GPU无人占用才放行。冠军有错即停队列等待诊断；不会自动重试、抢卡、清STOP或截短任务。它不是通知服务，不承担公网模型交付。144项本地回归已过，目标机器验收、真实seal与反向队列启动待实时计划核实；根自测不替代另一成员审查。

14:01实测更新：ae869bfb独立部署后，RTX144回归通过；真实seal核390完整/34既有异常/4行政/572未启，新v7仍576剩余，PID3812967已waiting_for_champion。冠军首两task8环境已实际控制，200次chunk交付的任务/会话/初始状态/stage投票/inpainting长度复核通过；仅前缀证据，不冒称逐动作物理字节核验或完整成功率，详`results/2026-10-14-champion-first-public-prefix.json`。

## 2026-10-14 生火任务：关节状态视图丢失

resume_v5在11:39:17 CST的`setting_the_fire`305–308物理批次退出，11:40:46全局停止。唯一终止栈是官方evaluator的`_apply_actions`→env.step/get_obs→robot.get_proprioception→entity_prim.get_joint_positions，`_articulation_view.get_joint_positions()`返回None后`.view(self.n_dof)`抛AttributeError；此前有PhysX `unitDir is not valid`和`Articulation needs to be initialized`。模型服务因客户端退出而断连，不是高层解析错误、OOM或已知NaN四元数断言。日志SHA `89b099872561c6e4500eb8a410280643c18e2e1256b5708924c28b442d403bd9`。

独立编排增加此**精确组合**的异常分类：必须同时具备先行PhysX/关节警告、完整官方读取路径、唯一指定AttributeError；缺栈、其他异常、CUDA错误或混合故障仍停止。失败slot未在栈中标明，不猜测；该4例记缺官方终局的执行异常、不重试、不合成JSON。只是记录和隔离新的故障类型，不修剪动作、不更改官方物理或actor。结果草稿支持披露这个新类别，仍先覆盖1000案例。

原trace末次ACK为每例6448（part状态6400只是较早快照）；最后四个动作块未获ACK，不能伪造已执行回执。全部记录动作有限，但306最后块最大绝对值约3.946e12，其余三例约1.6，**有限不等于合理**。仅此仍不能确定物理状态先发散还是策略/动作转换先异常；保留原部分视频供进一步核帧，不为了赶评测把异常动作悄悄clip。八模型/observer终态权重均未变，0更新。本地回归及目标机封存/重启结果继续记实时计划，不能将这里的分类设计当评测已恢复。

## 原轮实际停止原因

RTX `runs/memlite_final100_20261013_v2`，推理/物理入口固定`51c11d32a5d2a73df79b4a6a87dd266db2ae6c61`。传真机任务第二个四实例批次（305–308）中的scene3机器人姿态成为NaN，官方`rigid_dynamic_prim.py`断言退出；原调度器随之全局STOP。八卡最终权重均逐字节不变。

- 64条已完成：按原完整物理ACK回执读取，保留原JSON/MP4，不重跑、不选分数。
- 4条物理异常批次：305–308均缺完整官方回执，按执行异常保留；308是日志指定的异常slot，其余三条受到同一物理batch终止影响。不能把四条都说成机器人各自失败，也不能补造官方Q/SR。
- 14条行政中断：其他GPU被全局STOP打断的已启动回合，保留旧partial及理由，在独立目录显式从头续跑；没有挑最佳结果。
- 918条未开始：仍用原注册任务、实例、种子、超时和2/4环境分组。早期口算18/914已由逐条封存回执更正为14/918；原未完18=4物理异常＋14行政中断。

传真机308在第5184控制附近raw23指令绝对值突增至395，其余七条实例最大低于1.84。仅凭动作trace无法确定“模型先异常”还是“物理/本体状态先发散再进入相对动作锚点”，故不改动作clip、归一化或物理参数掩盖现象。

**17:11 CST补充人工核帧：** 原308部分视频仍可读，5306帧，未把它称完整官方结果。按原视频0起始索引抽取5180–5185，5180仍贴近室内几何，**5181已出现明显场景外/天空和机器人部件异常位移**，早于5184处第一次数百量级输出；原5168动作块覆盖5168–5183，整块最大绝对值1.9687167，非NaN。官方`step_fn`在成功物理step之后写每一帧，因此不能把后来的395直接认作首次物理失稳的原因。更早的碰撞、关节/控制语义和数值状态首因仍未由现有记录确定；“失稳观测经raw-state anchor回灌”的因果方向与证据相符，但未保存该瞬间完整proprio，不能冒称逐维证明。未重跑公共实例、未修改actor/控制/物理。

亲审图在被忽略的本地`artifacts/final-eval-20261012/continuation-v1/fax308-{before-crash,boundary-5180-5185}.png`，服务器派生件在新续评目录`diagnostics/`；原MP4仍在旧run，未覆盖。第一图抽帧5100/5150/5170/5180/5190/5300，第二图5180–5185，按行从左至右。

## 独立续评编排

### 2026-10-14高层记忆生成拒绝

resume_v4于04:51:42 CST的`store_produce`305–308同一批，在四例均ACK到10240控制步后，高层解码被`_parse_target_free_planner_event`的K=3记忆递推校验拒绝；不是物理NaN。04:53全局保护退出，388完整保留，旧26物理批次异常另列。模型日志、客户端日志和服务终态三者均指向完全相同的`planner generation violates causal K=3 memory recurrence`，高低层与observer终态参数不变。服务没有把异常对象附带的原始生成文本写盘，因此无法定位四例中的具体slot或具体错误词；不得补造或断言是串环境导致。

新的独立编排只给这个精确签名增加分类：客户端栈、模型解析栈、终态错误必须配对一致；混有其他异常、CUDA错误、外来身份、未知格式仍保护停机。该四例列为**模型输出拒绝导致的批次执行异常**，不重跑、不填造官方分数、不执行错误意图；另七卡真正行政中断才可明示续跑。冻结actor/记忆验证/推理参数/物理均不改变。以后同签名只隔离这一批并封存旧模型的权重和服务回执，再用同权重新进程执行其他尚未尝试实例；不是让模型改答案重试。

只读trace审计新增显式`allow_incomplete`接口，用于停机现场：仍逐条验身份/高低层意图/observer时钟/raw23 ACK，但必须没有未ACK动作，返回未关闭回合清单且明确不是完整审计。默认完整审计仍拒绝任何未关闭回合，禁止捏造close事件。

入口`scripts/eval/continue_memlite_final.py`，由新的冻结Git checkout运行，**实际actor、会话、模型服务与模拟器入口仍来自原51c11d32 checkout**。

1. `freeze`：只读封存原64条完整结果/视频的SHA与帧数、所有attempt、八卡权重回执和trace；要求原进程都退出。
2. `prepare`：注册新的续评目录，绑定原manifest/封存回执/源commit，不覆盖原STOP或status。跳过64已完成及4已诊断异常，仅安排剩余932条；选取不依赖Q或success。
3. `supervise`：只有GPU实空才启动八worker。每卡独立模型、会话、端口；正常回合结束后原服务负责清空记忆。原runtime的任务/实例/slot/高低层/observer身份门不变。
4. 仅精确匹配“机器人四元数全NaN”的实际trace，才将**该物理批次**登记为执行异常；保持原件，验证模型未变，再创建全新服务处理不同的未试实例。同一part中尚未开始的后续batch单独留lineage，不重跑已尝试前缀。未知错误/权重变更/缺失回执仍停诊断。
5. 汇总分开给出：有真实官方JSON的completed-only Q/SR、完成数量、执行异常数量、未覆盖数量、异常与缺失按0的保守内部统计。只有1000条都具真实官方记录，才给`official_full_q_score`/`official_full_sr`；否则这两个字段为null。覆盖1000条不等于取得1000条有效官方JSON。

单个slot可能早于整个batch完成并已有官方JSON/视频。若之后另slot崩溃，禁止把前者也无声记成错误：检查到不属于已提交整批回执的官方JSON（包括半写文件）时停止，保留原件，等逐例trace/ACK审计，不自行降分或丢弃成功记录。

不声称进程重启与未中断的物理运行逐位一致。原任务顺序/种子/分组不变，行政中断来源逐条记录；公共301仍是已诊断开发实例，不称盲测。

### 第二次停止与续评链（19:10 CST）

`resume_v2`在咖啡台301–304批次的scene1（302）出现`left_gripper_finger_link1`全NaN四元数。callback两次报告，加上最终包装栈共三份相同异常；其后`read_from_physx`和pre-physics wrapper是同一次退出的伴随栈。原隔离器只接受单次`base_link`，因此漏接并全局保护停止。150条完整结果保留，咖啡4条需作为同批执行异常而非四个独立物理失败；22条其他已启动实例行政中断，蜂蜜任务仅加载、未开始回合。计数还需新封存器实读验签后确认。

- 精确诊断扩展到R1Pro机器人子link的全NaN四元数，允许同一slot/link的重复栈及上述已观察包装异常；多个不同slot/link、OOM、其他异常仍拒绝。只是故障分类，不修剪动作或改变仿真。
- `freeze-continuation`从已停续评封存继承结果＋新结果，验证每个原JSON/视频帧数与SHA、实际seed、任务/源码身份、所有模型代次终态未变与进程退出。存在未提交官方JSON时停止独立审计；不能默认为失败或行政中断。
- 每次`prepare`追溯原始1000实例manifest和同一51c runtime；保留完整父seal链、当前22行政中断的逐例来源。新目录不覆盖旧STOP/原件，排程仍与分数无关。
- 冠军队列允许明确指定新`--queue-name`和死去的`--previous-queue`，仅在前队列从未启动任何冠军阶段、两个原TRAIN预检均已验证、且新MEM是原依赖的显式续评时通过。保留旧队列状态并使用跨目录owner锁；不重跑冠军预检、不生成新模型。

本地51项定向回归通过，服务器真实数据封存和恢复待；根自审不冒称另一成员已独立review。

**服务器封存拒绝后的更正：** 八worker状态并不足以判定行政中断。原玫瑰插花305–308在19:09:36另有scene3/base_link NaN，早于该worker19:09:47读取STOP；虽然状态写Administrative stop，四条必须列入执行异常，不能续跑。故第二次停止新增物理异常实际8条（咖啡4＋玫瑰4），行政中断18条；累计150完整＋12异常，待续838。原eb5封存器正确拦住喇叭日志的独立traceback，未产生seal/未启动新回合。

喇叭traceback为19:01:59仅`omni.kit.property.usd/widgets.py`→`omni.kit.window.property/window.py`的空`scroll_y_max`界面回调；19:09:28原成功物理step后的状态仍为4实例各1920实际控制，模型trace各实例还确认至2032。加入仅此已查明UI栈＋更晚成功物理时刻＋正控制计数的封存诊断，其他异常仍拒绝；不改官方UI/物理/actor。新增回归明确验证“worker叫行政停止、原日志实际NaN”不能被重跑，以及没有后续物理证据不得豁免UI异常；合计53通过。

19:38真实封存/部署完成：6125b615目标53项回归通过，seal `a5ae63459cd62ac659e49f48675eb99908eca4dde3872768cae708604d5588a4`，150完整＋12异常＋18行政＋820未启。新resume_v3 manifest `e2152ded02b0844caf57bdafa13fa73b9c9295c7c3d5b9e6f349faef9f29ac10`实际开始838剩余，actor不变；新冠军`queue-resume-v3`只替换已停依赖编排，不替换模型/公共run/原预检。旧原件、行政partial、两次错误和全部源码版本保留。

### 20:10新增：加载准入饥饿不是模型超时

resume_v3的GPU6始终在第一处`loader_lock(job)`等待，1800s后旧门抛TimeoutError，全局停止。该worker没有启动模型或模拟器，也没有公共attempt；不能伪造第八份权重回执或给未试实例记物理失败。实际已启动七模型的终态均未变；新增两条玫瑰309/310已完成，保留至152条。

独立`loader_admission.py`保留两个原flock槽，但以受锁保护的顺序票据FIFO准入：后续任务不能越过未启动worker。只取消**排队等槽**的1800s截止，拿到槽后的模型ready/模拟无请求30分钟安全门和官方控制步时限不变。票据记录owner PID＋进程启动ticks，死亡/复用PID票据标记abandoned，STOP标记cancelled，释放记released；不删除票据证据。

封存器仅对原日志明确证明“等槽超时、0模型、0模拟器、0任务输出”的worker允许缺模型回执，单列`never_admitted_workers`，其余真正模型仍必须逐份参数不变；不会把未启动算物理失败。新增实际六子进程FIFO/两槽并发测试、跨3800s队列等待、STOP及过期PID等，合计64本地回归通过，真实目标封存待。冠军旧9f编排也采用旧准入函数，下一在独立编排层补同类修复，不能热改其actor源或伪造新预检。

### 冠军同类准入的独立修复

`champion_fifo_eval.py`在新的冻结checkout加载原9f `champion_full_eval.py`，先验原manifest、配置、源commit与终态回执。仅重绑定`loader_lock`及原supervisor用来启动子worker的`__file__`；原模块已计算的HERE/REPO、worker函数、模型服务和模拟器入口完全保持9f。新wrapper和FIFO文件SHA由后继队列绑定，父/子编排另写只增不覆的provenance回执；原模型/预检身份不改写成新commit。`inspect`只读验证绑定，不启动GPU或创建回合。

只准搭配已完成的原2/4 TRAIN门，禁止wrapper隐式重做preflight。后继显式`--fifo-champion --completed-preflight`，等待MEM1000已说明覆盖并核GPU空闲；未知失败仍停，不加入自动公共重试。2026-10-13 20:40本地69项通过；实际部署/原源绑定检查待。此为根自审与回归，不冒称已有另一成员独立审查。

20:41真实验收补充：858885e0独立RTX源目标69项通过，`inspect`验证原9f路径、19b056a7 manifest与原dispatcher SHA74f3a33d…46bf；新`queue-resume-v4`/PID3603438等待MEM resume_v4。原已完成TRAIN只验签，0新增rollout，actor/物理未改；完整公共评测仍待。

## 冠军TRAIN接口修复

原final8000权重成功GPU加载，首TRAIN在0控制步被客户端字段检查拦截。官方`evaluator.py::_preprocess_obs`额外加入`robot_r1::cam_rel_poses`。修复只在客户端明确丢弃此键，不读取其值、不发给模型；RGB/61D本体/任务编号的原协议不变，其他未知字段继续拒绝。纯CPU回归覆盖不读取该字段、未知/特权键拒绝、固定slot缩小/重置/实际ACK。必须在新冻结源做真实TRAIN复验后，才准冠军公共轮；旧失败目录保留。
