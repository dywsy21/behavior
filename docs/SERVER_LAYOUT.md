# robo服务器文件位置与保留规则

更新：2026-09-13。本文件中的服务器路径属于`ssh robo`，不是本地路径。**源码走GitHub push/pull，数据、权重、环境、完整实验结果不走Git。**

源码整理与GitHub同步已完成；按用户后来确认的安全范围，旧`hy_vla`实验归档已净释放约481.11GiB。下表区分实际运行位置、Git协作入口和冷归档；其他重要目录保留原位。

## 1. 总览

**2026-09-19 22:17当前：** H21唯一原起点354348/GPU3于22:16:17 BJT启动初始化，NVMe根`h21_search_reanchor/radio_h21_fullstart`/相邻log/launch_policy.json；353691/8925同cc服务ready、起始0/431。两源相同固定`semantic_reanchor_cc249c9`不可热pull；两旧门退出、完整本地归档仍传输。Astra无新GPU任务。

**2026-09-19 22:15当前：** H21原两门已完成全链/视频/首图审计；GPU3现在唯一新353691/8925模型加载，`h21_search_reanchor/server_h21`与相邻log/launch_server.json，源仍`semantic_reanchor_cc249c9`不可热改。策略尚未启，0新训练；本地h21_gates_bundle仍在完整传输。

**2026-09-19 22:14当前：** H21两门328327/328328已退出，原NVMe根两个result通过（417/418控制），固定cc源保留；完整本地`artifacts/agentic-vlm-goal-20260918/h21_gates_bundle/`在传输/核验，未启动新服务或策略。GPU3待父下一个原登记完整回合，Astra仍仅CPU。

**2026-09-19 21:56当前：** H09W317744已退出，`h09w_native_complete/native_task1_v1`保留107.34MB/724控制/局部GRASP成功但隔离，完整本地子`artifacts/h09w-native-task1-v1/complete/`正在全核。GPU3现在父唯一两H21门328327/328328，新`/mnt/nvme_tmp/robodojo_agentic_20260919/h21_search_reanchor`/两个相邻log/launch_gates.json，固定`semantic_reanchor_cc249c9`，不能热改。新VLM服务和完整策略尚未启，队友进程不触。

**2026-09-19 21:52当前：** H21新不可变`/mnt/sdc1/robodojo/behavior_dev/git_worktrees/semantic_reanchor_cc249c9`固定cc249c9/digest2445784a，仅381 CPU通过，无新物理/model。旧bd46同目录前缀的CPU副本保留。H09W仍原317744/GPU3/f991，初态完整本地子worktree`artifacts/h09w-native-task1-v1/initial/`，父3 RAW/7payload+3depth hash审过，后态待。

**2026-09-19 21:38当前：** H09W唯一PID317744/GPU3于21:37:50 BJT启动初始化，新NVMe `robodojo_vlm_sft_20260919/h09w_native_complete/native_task1_v1`与相邻log/launch.json；active授权位于原SDA根`authorization_h09w_native_task1_v1.json`SHAea4c255c…4d57，固定f991/239源不可热改。当前无本轮VLM服务/训练，父H21仅CPU379通过待独立审。队友GPU0/1/2与313698副context不动。

**2026-09-19 21:35当前：** 父旧模型209877已身份核验/21call与identity双端SHA归档后TERM并确认退出，本地`artifacts/agentic-vlm-goal-20260918/h19_server_complete_archive/server_h19_fullenv/`。GPU3余80,939MiB，队友313698副context不触；GPU1的247595/8790 `rl_radio_dense_20260919/code/model_service.py`与313698 `local_baseline.py --post-training`属于队友，禁止终止。新固定SFT源`git_worktrees/vlm_sft_h09w_f9916db`只CPU验收过；拟新NVMe `/mnt/nvme_tmp/robodojo_vlm_sft_20260919/h09w_native_complete/native_task1_v1`刚单次放行，真实PID待作者回执，不称运行/训练。父H21仅本地CPU，不热改原0d/f991。

**2026-09-19 21:24当前：** 中断恢复后核209877/GPU3/8924仍旧固定0d服务，211843/213880均已退出。H19完整本地`artifacts/agentic-vlm-goal-20260918/h19_fullstart_bundle/`三主SHA/视频解码过；H09V完整包在子worktree`artifacts/h09v-native-task1-v1/complete/`且父完整账本审通过，非成功训练数据。Astra新f9916db仅容量CPU源码；拟NVMe `robodojo_vlm_sft_20260919/h09w_native_complete`尚未放新物理。GPU1现15,277MiB、GPU2 4,035MiB，须核现进程而非沿用旧空闲；不触队友，sdc余128.49GB/NVMe2.94TB。

**2026-09-19 08:39当前：** H19原起点211843现已退出/官方false，原`radio_h19_fullstart`300,907,179B、469控制/21调用/617.439s保留，因462→468视觉运动段质量拒绝。GPU3仅209877/8924模型空闲21/431；不用旧192预算继续同run。父新只CPU诊断，完整H19本地包下载中。08:38“父仍运行”是较早轮询，已更正；H09V213880也已退出，Astra收尾完整证据/新容量设计。

**2026-09-19 08:38当前：** H09V213880已退出，原`h09v_native_task1_v1`89,949,654B/失败、8macro、396+267控制及真实末hold保留，完整本地包下载中；不是已训checkpoint/完整GRASP成功。GPU1释放待后继另票，原100MiB run不重跑或修改，Astra先收尾体积/物理证据。父209877模型及211843/H19仍原源运行。

**2026-09-19 08:28当前：** Astra GPU1唯一213880于08:27:09 BJT提交`/mnt/sdc1/robodojo/behavior_dev/vlm_sft_native_teacher_20260919/h09v_native_task1_v1`、相邻log；active授权SHA08162a72…5430d1，固定`vlm_sft_h09v_123837d`不热改，尚无正BC/新训练。父GPU3模型209877和原起点211843都在运行；211843在GPU1有约209MiB副context但主sim仍GPU3，不以此停队友。两路预算互不挪用，原root所有失败保留。

**2026-09-19 08:24当前：** H19实际服务209877/GPU3/8924、`server_h19_fullenv`已ready；唯一原起点211843于08:23:31 BJT提交`/mnt/nvme_tmp/robodojo_agentic_20260919/h19_combined_fullstart/radio_h19_fullstart`，同0d不可变源、零两类前缀，原192/6144/7200s。模型完整PYTHONPATH必须是`/mnt/sdc1/robodojo/behavior_dev/semantic_structured_20260919/deps_817f944:/mnt/sdc1/robodojo/behavior_dev/semantic_agent_20260917/deps:<固定source>/src`：前者decoder817，后者HF5.7.0；共享HF4.57.1不识别qwen3_5。**模拟器不加载这两个overlay**。208180/208920均加载前失败/0calls/已退出，两个日志保留。Astra新不可变`git_worktrees/vlm_sft_h09v_123837d`仅CPU81＋331过、near396准备78,412B，GPU1仍空待父放行，不是已采集/训练。

**2026-09-19 08:13当前：** H19首次208180因launcher遗漏decoder overlay在加载前退出/0调用，`server_h19.log`保留；新208920/GPU3/8924于08:12:35提交`server_h19_envfix`，源仍不可变0d。模型服务必须复用已固定`/mnt/sdc1/robodojo/behavior_dev/semantic_structured_20260919/deps_817f944`，以`PYTHONPATH=<该overlay>:<固定source>/src`加载；虚拟环境自身没有该依赖，不要重复安装/改共享环境。原起点agent未启动。

**2026-09-19 08:09当前：** H19两门203694/203695通过/退出，完整链/主SHA/视频已父核，新GPU3服务208180于08:09:01 BJT提交，端口8924、`h19_combined_fullstart/server_h19`及log、固定`semantic_combined_0d34ec7`源；原起点agent未启。完整门副本正传本地h19_gates_bundle；原server_h15/156556结束，不再调用。Astra仍≤600s格点教师CPU，GPU1无新物理/训练。

**2026-09-19 07:56当前：** H19新NVMe根已创建、`gate_radio_h19`203694与`gate_plates_h19`203695/GPU3于07:55:56 BJT提交，固定不可变`semantic_combined_0d34ec7`/621bd7c5，同名log与`launch_gates.json`，累计≤4GiB。两个门尚在初始化，不重启/热改。旧156556已确认退出；新8924模型和原起点策略均未启动。GPU1仍留Astra，不触GPU0队友。

**2026-09-19 07:54当前：** 新Git不可变`git_worktrees/semantic_combined_0d34ec7`仅CPU368通过/digest621bd7c5…c916；H19物理根仍未启动。旧156556空闲190账本已完整本地`h15_server_complete_archive/server_h15`双端SHA核后仅发TERM，需核退出；原server_h15所有结果保留，不再新调用。AstraGPU1仅CPU准备，无新模拟器或训练。

**2026-09-19 07:50当前：** H17 192585已退出，147,454,991B完整run在本地`artifacts/agentic-vlm-goal-20260918/h17_matched_bundle/radio_h17_matched_approach`；GPU3仅旧156556模型空闲190/215，GPU1当前空给Astra。P3完整70文件本地子worktree、双端SHA全过，父seed人工审通过但尚无native训练样本。H19新NVMe根`/mnt/nvme_tmp/robodojo_agentic_20260919/h19_combined_fullstart`仅预登记≤4GiB，尚未创建/运行，旧所有根及不可变源保留。

**2026-09-19 07:36当前：** P3 196049/196046已退出，原`h09u_reference_task1_v3`20,782,245B/465控制/reference局部成功，seed仍隔离、完整本地下载核验中。`/mnt/nvme_tmp/robodojo_agentic_20260919/h18_cpu`仅31,428B旧/新保存态CPU结果，无新模型/仿真；不可变ada源CPU/独立审通过。H17旧341/192585原回合未在此记录为结束，勿重复启动。

**2026-09-19 07:32当前：** GPU1新P3 `vlm_sft_native_teacher_20260919/h09u_reference_task1_v3`196049（壳196046），07:27:19 BJT提交，固定aff、授权00a798b1…594b4a1，原源/失败留存。GPU3 H17 192585已通过1392前缀并实际策略运行，仍同341；模型156556不变。新`git_worktrees/semantic_fastfk_ada39f7`只做CPU、不挂任何物理任务；H17双门完整本地`h17_gates_bundle`全SHA/运动链/视频通过。

**2026-09-19 07:16当前：** H17两门186948/186949退出且完整链/视频通过；唯一`radio_h17_matched_approach`192585/GPU3于07:15:32 BJT初始化，新NVMe同根/相邻log，固定`semantic_body_3411489`；原156556模型8923保持170/215起。局部是1392保存策略前缀，不是零前缀成功率。GPU1无新采集，Astra只核上游时钟；所有旧失败留存。

**2026-09-19 07:11当前：** H17 GPU3原186948/186949各已200控制、尚未门完成；NVMe输出原处，局部未启。GPU1 P2 185102已退出，原`h09u_reference_task1_v2`保留166控制/时序验证失败；完整本地作者`artifacts/h09u-reference-task1-v2/run`，双端全SHA审计进行中，非成功seed。模型156556保持旧源空闲。

**2026-09-19 07:02当前：** 新目录已真实创建：`/mnt/nvme_tmp/robodojo_agentic_20260919/h17_body_options`，仅H17有限新输出，累计1GiB；GPU3门`gate_radio_h17`186948/`gate_plates_h17`186949＋同名log初始化，runtime `git_worktrees/semantic_body_3411489`/digest b2d14390…c78df，禁止热改。旧root7,406,026,563B/旧模型156556保留原盘，局部run未启。GPU1 P2 Python185102（壳185097），原root `h09u_reference_task1_v2`＋log、独立26e源/新授权60d420…7d5ea；旧P1失败保留；尚未训练。

**2026-09-19 06:50当前：** H17新不可变`git_worktrees/semantic_body_4b56b03`354 CPU通过/digest a4ecacc0…2d499，无新物理/输出目录；GPU3旧156556/8923真实170/215，43剩余配额足。GPU1参考179879已退出，失败完整本地在子worktree `artifacts/h09u-reference-task1-v1/run`（17文件6,585,894B、全SHA过，165控制含hold）；原服务器目录/9b源保留不改。新NVMe拟输出仍未创建，余2.7TiB，原盘余约81GiB；旧父root7,406,026,563B，仍遵原7GiB，不把换盘抹成零历史。

**2026-09-19 06:39当前：** Astra GPU1唯一参考回放179879（壳179878），固定`git_worktrees/vlm_sft_h09u_9b52faa`，`vlm_sft_native_teacher_20260919/h09u_reference_task1_v1`及同级log；只TRAIN192、164原prefix＋288完整技能段＋13收尾，不是新VLM训练。授权同root `authorization_h09u_reference_task1_v1.json`/SHA44e2c2cd…8454f1。保持旧目录/源不热改，未启动task0。H16本地`h16_matched_bundle`已完整传完、三主SHA过。

**2026-09-19 06:36当前：** H16唯一175529已退出、218新控制/22调用/官方false；完整新结果171,088,570B正在本地`h16_matched_bundle`归档。GPU3仅原156556模型保留，累计170/215调用（待账本再核），不热改。H09U运行源`git_worktrees/vlm_sft_h09u_9b52faa`已CPU/父审，下一只task1参考回放GPU1，拟`vlm_sft_native_teacher_20260919/h09u_reference_task1_v1`，尚未PID；prepared仍原`h09u_reference_prepare_v2`，旧失败累计不抹除。双H16门本地`h16_gates_bundle`已完整SHA/288主hash/视频解码通过。

**2026-09-19 06:24盘核验：** `/mnt/nvme_tmp`实际挂载`/dev/mapper/vg_nvme-lv_nvme_tmp`、XFS/rw、3.5TiB总/约2.7TiB可用，目录1777，robodojo用户可写；**不是已消失的/mnt/tmp1**。拟新独立输出父目录`/mnt/nvme_tmp/robodojo_agentic_20260919`尚未创建/未迁移文件，后继实验需记录有限预算/真实路径；不把tmp名字当可删除依据、不改队友现有内容。原SDA盘仍保余80GiB，H16不换盘热改。

**2026-09-19 06:15当前：** GPU3唯一新局部策略`agentic_vlm_goal_20260918/radio_h16_matched_approach`175529（同名log）初始化，source仍`semantic_reorientation_2c546d2`；模型156556/8923旧5cf/148起。双H16门完整288主hash/304段签名-FK/8BASE通过，完整本地包仍下载；不要重启门或热改2c。H09U只有CPU准备新不可变7b源/本地草稿，无新采集或训练。

**2026-09-19 06:12当前：** `gate_radio_h16`172069与`gate_plates_h16`172070已退出、result两门通过（417/418控制），2c源不可变保留。完整副本传向本地`artifacts/agentic-vlm-goal-20260918/h16_gates_bundle`，链审计待；`radio_h16_matched_approach`尚未启动。GPU3仍仅旧模型156556/148calls，GPU1留给Astra。

**2026-09-19 06:06当前：** 新不可变`git_worktrees/semantic_reorientation_2c546d2`341远端CPU/真实source绑定通过，独立复审通过。父root下`gate_radio_h16`172069、`gate_plates_h16`172070/GPU3初始化中，日志同名`.log`，不得热改该源；模型156556/8923仍5cf/148calls空闲。H16局部保存前缀回合未启。Astra H09U只CPU修种子/因果与回放入口，GPU1空，无新SFT采集或训练。

**2026-09-19 05:53当前：** `git_worktrees/semantic_reorientation_1c7ddee`干净不可变、339 CPU与H15原1392前缀SHA/新SEARCH校验通过（digest c654f420…dedd229），仅待独立审，无H16模拟器进程。旧0e源码保留，其来源校验拒绝发生在reset前。GPU3模型156556/8923仍旧5cf源/148calls空闲，GPU1仍空给Astra H09T；新物理尚未放行。H15完整本地`h15_fullstart_bundle/radio_h15_fullstart`540主RGB-D hash/全视频/主SHA通过，旁边partial_scp是中断复制保留的部分副本，不是唯一结果。

**2026-09-19 05:23当前：** H15策略158357与task1采集163349均已退出；GPU1空闲，GPU3仅模型156556/8923空闲148调用。`radio_h15_fullstart`官方false/90决策/1914控制、完整1,073,873,438B正传本地`artifacts/agentic-vlm-goal-20260918/h15_fullstart_bundle`；服务器父root6,570,536,733B、磁盘余87,953,440,768B，仍须守7GiB/余80GiB。task1 `pilot_v2/task_1`最终164prefix/91native含finally hold，两完整接近标签，第三后验写前超量失败；全SFT root85,022,736B，task3未启动。源码/失败不删不热改。

**2026-09-19 05:01当前：** GPU1唯一新collector163349，04:59:09.866提交`vlm_sft_native_teacher_20260919/pilot_v2/task_1`/同级log，固定`git_worktrees/vlm_sft_native_68dacca`，只TRAIN192/164前缀。还未有教师样本/训练，task3未放行。GPU3同H15模型156556/策略158357仍运行，不热改任何活跃源。

**2026-09-19 04:58当前：** 父H15 GPU3模型156556/8923＋策略158357仍同源运行；局部d51/52拒底盘前进，未有最终成功。新采集不可变源`git_worktrees/vlm_sft_native_68dacca`已36测试/真实来源核验通过；`vlm_sft_native_teacher_20260919/authorization_remaining_two_68dacca.json`已激活SHA599298a3…78795，仅准原剩余`pilot_v2/task_1`/GPU1启动（待PID），task3未放行。旧失败`pilot_v1/task_0`304prefix/0native不删，完整压缩故障包在`artifact_budget_fix_68dacca/task0_failure_full.tar.gz`，本地作者artifacts同名包已SHA核验；累计root约54.62MB包含原47MB失败，100MiB不扩大。

**2026-09-19 04:43当前：** H15 GPU3模型156556/8923＋唯一策略158357正在原起点搜索，不热改5cfbb89；两r1完整本地`h15_gates_bundle`已288主RGB-D hash/两全视频解码、夹爪开闭人工面板通过。GPU1采集156956已体积故障中断退出，`vlm_sft_native_teacher_20260919/pilot_v1/task_0`保留大标定与`budget_operator_stop.json`，无final_hold/候选/训练，task1/3未启。下一只CPU修collector，旧1509源不改；父根5702020154B、盘余88910872576B，仍在7GiB/80GiB边界内。

**2026-09-19 04:36当前：** GPU3 `agentic_vlm_goal_20260918/server_h15`156556/8923同5cfbb89 ready0/215，唯一`radio_h15_fullstart`158357零前缀初始化；两r1门已过并退出，完整本地`h15_gates_bundle`传完。GPU1交给Astra：`vlm_sft_native_teacher_20260919/pilot_v1/task_0`156956及相邻log，源码`git_worktrees/vlm_sft_native_1509b7a`，授权`authorization_first_three.json`，只启动第一个TRAIN70，未训练。parent source`semantic_geometry_5cfbb89`与collector source均不可热改。旧132309已退出，104调用本地归档保留。

**2026-09-19 04:23当前：** 新`gate_radio_h15_r1`152337与`gate_plates_h15_r1`152335/GPU3已初始化提交，源`git_worktrees/semantic_geometry_5cfbb89`/digest239cb591，独立API复审通过；原H15失败目录保留。新model/fullstart/collector未启动，GPU1旧132309仍104空闲。下方“未启动”按时间保留为历史。

**2026-09-19 04:22当前：** H15两门149129/149131已reset后标定失败并退出，0控制；完整本地`h15_failed_gate_bundle`两failure SHA一致。API修正新源`git_worktrees/semantic_geometry_5cfbb89`仅331 CPU通过，独立审查待，`gate_radio_h15_r1`/`gate_plates_h15_r1`未启动。`h13_server_archive/server_h13`104调用及身份已完整下载/SHA核验；132309/GPU1/8922尚未停止，GPU3空。下方“初始化中”仅历史，禁止按旧PID复跑。

**2026-09-19 04:15当前：** GPU3新`agentic_vlm_goal_20260918/gate_radio_h15`149129/`gate_plates_h15`149131及同名日志已提交初始化，源`git_worktrees/semantic_geometry_b6f0845`不可变/digestb71737e4。尚无H15模型或原起点策略；旧132309/GPU1/8922仍104空闲。根4831846809B/余83.70GiB为启动前检查；不要复用下方旧工程门/旧运行状态。

**2026-09-19 04:07当前：** 新CPU-only不可变源`/mnt/sdc1/robodojo/behavior_dev/git_worktrees/semantic_geometry_e9abf39`，329测试通过，未新物理。H14完整本地包已三SHA/视频解码完成，下条“下载中”是历史。Astra独立`/mnt/sdc1/robodojo/behavior_dev/vlm_sft_native_teacher_20260919/prepare_v1`为三个TRAIN参考前缀/9视频（799441B），不是正标签；本地完整小包在`/home/wsy/behavior_worktrees/vlm-sft-native-teacher-20260919/artifacts/h09s-prepare-v1`。旧132309/GPU1仍104空闲，GPU3空，盘余约83.72GiB，未启动新训练。

**2026-09-19 03:42当前：** H14143363已退出，`radio_h14_contact_replay/{result.json,PRIVILEGED_CONTACTS.jsonl,rollout.mp4}`保留，1483控制/235.164s。GPU3释放；完整本地`artifacts/agentic-vlm-goal-20260918/h14_contact_bundle`下载中，未当已完整校验。132309/GPU1仍旧源104空闲，H09S仅CPU。下一禁止原样复跑；接触诊断不是actor可用传感。

**2026-09-19 03:36当前：** 新`agentic_vlm_goal_20260918/radio_h14_contact_replay`143363/GPU3已提交初始化，日志相邻`radio_h14_contact_replay.log`，不可变源`git_worktrees/contact_replay_8587e5e`。一次1482+1/1200s/0模型诊断，新artifact≤200MiB、根≤7GiB/余80GiB。旧62c12ed/0a3df02源仅CPU检查未物理使用。132309/GPU1仍空闲104，Astra H09S只CPU独立准备，不动GPU0队友/运行源。

**2026-09-19 03:28当前：** H13完整副本本地`artifacts/agentic-vlm-goal-20260918/h13_fullstart_bundle/radio_h13_fullstart`已传完805MiB、408主RGB-D hash/视频SHA与解码通过；人工面板不等于全帧。H14-contact新脚本仅本地CPU，拟远端`agentic_vlm_goal_20260918/radio_h14_contact_replay`尚不存在/未启动，待独立review与新固定Git源；原132309模型保持空闲104调用，不动源90a7c20。H09R计数/方案已Git187be9b，无新数据采集/训练。

**2026-09-19 03:03当前：** `radio_h13_fullstart`134323已退出，68决策/1483控制/104模型，官方false/NO_MOTION_PROGRESS，result/video保留。`server_h13`132309/GPU1/8922仍104/215空闲、源`semantic_joint_90a7c20`不热改；GPU3当前无本轮模拟器。全本地`artifacts/agentic-vlm-goal-20260918/h13_fullstart_bundle`下载中，不能当已完整校验；两H13门完整本地/hash/视频解码已过。父H14仅CPU保存几何诊断，Astra H09R仅训练来源计数，无新训练或物理提交。

**2026-09-19 02:32当前：** H13两门130412/130413均通过退出、Astra全段链验收通过；唯一`agentic_vlm_goal_20260918/radio_h13_fullstart`134323/GPU3已启动初始化。同90a7c20/digest627652f7、0前缀、新规划、96/3072/2400s；模型132309/GPU1/8922起始0/215。根3815699383字节、累计上限7GiB、余≥80GiB；门完整本地`h13_gates_bundle`仍下载中。不可热改活跃源或重复提交。

**2026-09-19 02:27当前：** `server_h13`132309/GPU1/8922已ready、同90a7c20/同27B revision、0/215调用；双门130412/130413仍GPU3初始化。Astra只读链审计中、唯一原起点策略未提交，代码仍固定不可热改。

**2026-09-19 02:26当前：** 固定`semantic_joint_90a7c20`新H13门130412（radio）/130413（plates）均GPU3初始化；旧121949已身份核验后退出、19ledger和identity本地SHA一致。新`agentic_vlm_goal_20260918/server_h13`/GPU1/8922加载提交中，尚未0前缀完整策略；活跃源不热改。根预算7GiB/余80GiB保持。

**2026-09-19 02:19当前：** 新不可变`git_worktrees/semantic_joint_90a7c20`固定90a7c20/digest627652f7，319CPU通过，仅保存帧复核运行，无物理/新服务。旧121949/GPU1/8921仍19/215空闲，旧H12全部策略/门已退出；不可热改其源。新H13尚待独立review，后继目录/8922预算已登记但未启动；新块artifact上限定累计7GiB并余≥80GiB，不据配置存在当进程运行。

**2026-09-19 02:03当前：** `radio_h12_fullstart`124062已退出/官方false，427控制、19调用；完整下载至本地`artifacts/agentic-vlm-goal-20260918/h12_fullstart_bundle`中。121949/GPU1/8921仍同8e270b1空闲19/215，不热改；GPU3无该策略。H12双门本地完整且144+144主观测hash/视频全解码过。根2.9GiB、盘余86GiB；H13目前只有保存数据CPU诊断，无新模型/物理。

**2026-09-19 01:52当前：** H12两门120081/120082均通过退出；唯一`agentic_vlm_goal_20260918/radio_h12_fullstart`124062/GPU3初始化，固定8e270b1/digest1f9c5b，不可热改；模型121949/GPU1/8921同源，起始0/215call。策略0前缀96/3072/2400s，artifact启动前按两门真实文件量改累计≤6GiB、仍余≥80GiB，根约2.7GiB。门完整本地`h12_gates_bundle/`下载中，旧H11服务已退出、18ledger本地SHA一致；不要依历史记录重复启动。

**2026-09-19 01:43当前：** H12不可变源`git_worktrees/semantic_substep_8e270b1`（8e270b1/digest1f9c5b），GPU3门`gate_radio_h12`120081/`gate_plates_h12`120082初始化；新GPU1模型`server_h12`121949/8921/≤215调用已提交加载，同27B revision，未启动策略。旧111660已核身份退出、18调用保留并下载本地`h11_server_archive/`；新源禁止热改。新run和日志在原`agentic_vlm_goal_20260918`，仍4GiB/80GiB边界。

**2026-09-19 01:36当前：** 新`git_worktrees/semantic_substep_14901a8`为14901a8/digest596c76a7独立源，309CPU过，尚无新模型或物理进程；旧111660/GPU1/8920保持2000d78/18调用空闲，GPU3空。H12各CPU诊断在本地`artifacts/agentic-vlm-goal-20260918/h12_*odometry*audit.json`，光流/相机合并均未部署，不把其正样本统计当策略改善。根约2.0GiB/盘余87GiB；不删除旧证据。

**2026-09-19 01:01当前：** 115556已正常退出，`radio_h11_fullstart/result.json`SHA882220ce…，361控制/官方false/视觉里程计不确定停止；GPU3释放，111660/GPU1/8920空闲18/217，不能热改服务源。完整H11策略本地`h11_fullstart_bundle/radio_h11_fullstart`已传完，96主RGB-D hash通过；H12仅CPU保存状态诊断，无新runtime/物理/训练。旧H11两门完整本地`h11_gates_bundle/`已传完，未完成额外人工/视频校验前不笼统标记全验。

**2026-09-19 00:54当前：** H11两门112456/112535通过退出（SHA9f21f7be…/c2312319…），唯一`agentic_vlm_goal_20260918/radio_h11_fullstart`115556/GPU3初始化；2000d78/ade09283源固定，0前缀、96/3072/2400s。服务111660/GPU1/8920同源、起始2/217，不能热改/重复提交；新结果待，H10所有失败保留。

**2026-09-19 00:49当前：** H11服务111660/GPU1/8920就绪、2/217调用；`static_plans_h11`已完成且保留两输出。GPU3双工程门`gate_radio_h11`112456、`gate_plates_h11`112535正在初始化，源均`semantic_structured_2000d78`不可热改，未启动策略。旧H10服务ledger已本地`h10_server_archive/`，完整原起点包`fullstart_failed_bundle/`，六初图hash与call13一致；r3双门完整视频解码与288主观测hash过。

**2026-09-19 00:44当前：** H10服务89280已核身份停止/退出（13调用保留）；H11新服务111660/GPU1/8920加载中，`agentic_vlm_goal_20260918/server_h11`、源`git_worktrees/semantic_structured_2000d78`/digestade09283，schema0308a127和decoder817f944。模型依赖只加独立`semantic_structured_20260919/deps_817f944` overlay，原不兼容`deps`保留，未动共享环境；GPU3无新物理。原H10所有失败与r3门均保留，本地门完整副本288主RGB-D hash通过；2静态及H11物理待。

**2026-09-19 00:28当前：** 108260因首规划JSON解析失败退出，`radio_h10_fullstart`仅初始图/失败记录，无已执行动作或完整视频；不要重复启动旧run。89280服务仍GPU1/8919、13/368调用，GPU3已释放；H11仅本地CPU实现/依赖检查，尚无新模型服务或物理。r3双门完整副本正传至本地`gates_r3_bundle/`，传完前不算核验。

**2026-09-19 00:24当前：** r3门105324/105398均已通过退出；唯一策略`agentic_vlm_goal_20260918/radio_h10_fullstart` PID108260/GPU3正在初始化，源`git_worktrees/semantic_prompt_1d93b29`，0前缀、96/3072/2400s。模型89280/GPU1/8919保持465bc85、同revision，从12/368调用起，不能热改或重复启动。r3 result SHA8c6641a9…/b2b4bc7e…；matched完整副本已在本地`matched_failed_bundle/`完成hash/全视频解码/人工六观察核验，下条“未核完”仅历史。盘余约87GiB，保留≥80GiB。

**2026-09-19 00:14当前：** H10旧matched101630已HTTP400退出，5动作/96新控制，证据`radio_h10_matched`保留；源码`git_worktrees/semantic_prompt_1d93b29`只压缩显示重复字段，digest428c339a。GPU3新双门`gate_radio_h10_r3`105324/`gate_plates_h10_r3`105398正在初始化，原fullstart未运行。模型89280/GPU1/8919仍465bc85同权重，12/368调用，不热改。`static_h10_r3`两次已完成0控制；新根1.1GiB/余88GiB。r2双门完整本地`artifacts/agentic-vlm-goal-20260918/gates_r2_bundle/`，失败matched下载到`matched_failed_bundle/`未核完前不当完整副本。

**2026-09-18 23:54当前：** H10证据根同下；r2双门98393/98469已通过并退出，`git_worktrees/semantic_gate_motion_dfe7c96`/digest128f5595固定，新唯一`radio_h10_matched`101630/GPU3正在初始化。模型服务89280/GPU1/8919仍旧465bc85源和同一冻结revision，身份分开记、不热改；原2静态调用。诊断前缀448＋362不计自主成功，64/2048/1800s上限；`radio_h10_fullstart`尚未运行。r1/最初失败门保留，旧H09不重启；盘余88GiB，原4GiB新增/80GiB余量边界不变。

**2026-09-18 23:33当前：** 新H10证据根`/mnt/sdc1/robodojo/behavior_dev/agentic_vlm_goal_20260918`，本地副本根`artifacts/agentic-vlm-goal-20260918/`。GPU1模型89280/8919使用不可变`git_worktrees/semantic_observer_465bc85`，同Qwen3.8-27B revision1d4bf0f，最大368调用/当前2；不可热改。初始`gate_radio_h10`/`gate_plates_h10`已失败退出，保留；新`git_worktrees/semantic_observer_5accf69`经CPU/独立review，`gate_radio_h10_r1`/`gate_plates_h10_r1`仅登记未启动，后续1诊断＋1原起点策略亦未运行。两个source身份分开记录；root351MiB/盘余89GiB，H09旧训练和证据不重启、不清理。

**2026-09-18 22:02当前：** H09所有训练/两门/四策略/专属模型已完成退出，GPU1/8918释放；H08 GPU3/8909亦无本轮进程，不重复启动。H09根`/mnt/sdc1/robodojo/behavior_dev/vlm_sft_showharness_20260918`：`data_v4`为实际1199/193/192数据；`train_v1/adapter_0600`为最终实验adapter（SHA `b5a125ed14dc06c82a7ae7fd288d8c7202d90e2210cc1d1c2f7195f3daac15e1`）；`eval_test_v1`静态，`radio_base_v1`/`radio_ft_v1`/`plates_ft_v1`/`plates_base_v1`四物理，`server_v1`调用证据。实际训练源140c47d、实际评测源975852a，后续035af65交接/父集成不替换这些历史身份。四视频及轻量回执副本已SHA核验到本仓`artifacts/vlm-sft-showharness-20260918/`；完整原始本地审阅材料仍在`/home/wsy/behavior_worktrees/vlm-sft-20260918/artifacts/`，含标明的partial校准下载，不当完整校准使用。H09根约2.8GiB、全盘余89GiB；不清理唯一证据，不将adapter当已改善策略。

**2026-09-18 21:56当前：** H08 B20模拟器71052/模型70673已确认退出，GPU3释放；`semantic_agent_effect_20260918/b20_complete.tar.gz`已完整下载并SHA验证，本地`artifacts/semantic-agent-effect-20260918/b20_verified_bundle/`包含策略/双门/服务及日志。策略视频214帧全解码、126 RGB-D hash和全部20后动作附着审计通过，人工4页面板完成；官方false。H09四配对已结束，专属74735/8918服务由Astra停止/核验，最终模型与结果位置以随后交接为准；不根据下文历史重新启动。

**2026-09-18 21:28当前：** B20策略71052已结束，官方false/429控制；70673核身份停止中，H08根`b20_complete.tar.gz`准备归档，源9f5957b保留。GPU3不再追加新回合；B20终态不要误作仍运行或任务成功。H09 GPU1独立微调后四短闭环继续，以owner记录为准，不动其服务。

**2026-09-18 21:17当前：** B20双工程门已退出/通过；同`semantic_inspection_budget_9f5957b`的唯一GPU3模型70673/8909和策略`radio_b20_matched`71052活跃，不热改或重新提交。策略64/2048/1800s，前缀448＋362另计。B19完整本地`artifacts/semantic-agent-effect-20260918/b19_verified_bundle`已SHA/174 RGB-D hash/全视频解码/本人5页核验。

**2026-09-18 21:10当前：** 旧58225已核身份退出；主代理GPU3双门66709/66710使用`git_worktrees/semantic_inspection_budget_9f5957b`不可变源，策略/新服务未启。H08根`static_b20_cpu_v1`/`static_b20_model_v1`已结束（4姿态/3神经/0控制），B19完整包本地下载中；后续服务日志另归档。H09正式600已完成/退出，最终adapter在其`train_v1`，Astra正在原192留出配对，不根据test另选checkpoint。

**2026-09-18 21:03当前：** B19策略58614已退出；完整`semantic_agent_effect_20260918/b19_complete.tar.gz` SHA `149ba57863ba37ac91e688b363a0ea60f074ce5a600fe7b4ac34998181b59f34`在下载到本地artifacts，未含仍活跃模型58225/GPU3/8909。模型保留只作B20最多3保存帧调用（当前54/151），旧源5df51be不得热改；B20仅本地实现无新物理。H09 GPU1唯一`vlm_sft_showharness_20260918/train_v1`61793实际470/600，源140c47d/最终data_v4，200/400 adapter已存；其目录名后缀不是commit，请以identity为准。盘余91GiB，仍保80GiB，不删除旧证据。

**2026-09-18 20:38当前：** B19双门已退出/通过；GPU3唯一模型58225/8909＋`radio_b19_matched`58614，固定`semantic_spatial_5df51be`，预算64/2048/1800s、≤151。不要照旧门状态重复提交或热改；GPU1 Astra H09在构造严格data_v3、未正式训练。

**2026-09-18 20:33当前：** GPU3 B19双门55607/55608，源`/mnt/sdc1/robodojo/behavior_dev/git_worktrees/semantic_spatial_5df51be`/digest6fe40b1a，不热改。新模型/策略未启动；同H08根`offline_b19_b18_matched`及`b19_actual_registered_b18_replay.json`是0控制离线证据。B18策略/服务均已退出，全包本地`b18_verified_bundle/`已SHA/24hash/全视频解码/人工4观察核验。

**2026-09-18 20:24当前：** B18策略47681已退出、55新控制/官方false；模型47250已核身份TERM，归档待。新诊断只在本地CPU，不重启旧run；GPU3暂无新物理。Astra GPU1独立H09正在构造严格数据release/真batch门，未正式训练，旧data_v1保留；主分支7dc0fdb已pull/fetch同步，main仍33677bd。

**2026-09-18 20:14当前：** B18双门已通过/退出；主代理GPU3 `server_b18_matched`47250/8909/≤151、唯一`radio_b18_matched`正在启动，源d807d3d不热改。Astra H09在独立源`vlm_sft_a201e09`准备CPU `data_v1`索引，训练尚未开始，GPU1留给其后续2B LoRA；各自新增盘量3/8GiB，合守80GiB余量。

**2026-09-18 20:08当前：** 主代理仅GPU3双门`gate_radio_b18`44434、`gate_plates_b18`44435运行，源`git_worktrees/semantic_persistent_d807d3d`固定d807d3d/b353df13/238CPU；H08根新增`offline_b18_radio_b17_matched`、`offline_b18_radio_b14_matched`为0控制旧帧候选核验。B18模型/策略未启动，不热改源。GPU1已交Astra H09独立SFT；其专属根`/mnt/sdc1/robodojo/behavior_dev/vlm_sft_showharness_20260918`，精确运行位置由其后续回报，不与H08混淆。

**2026-09-18 19:50当前：** B17唯一策略39229已结束（55新控制/4模型/官方false、保载安全停），38931核对专用命令后TERM；`b17_complete.tar.gz`SHA1c953bcd…约256MiB已归档、传本地中。下方“策略运行”为历史；不原样重启，新源/物理尚无，仅离线轨迹诊断。

**2026-09-18 19:44当前：** B17双门36340/36341均退出且通过；`server_b17_matched`38931/GPU1/8908/≤151已就绪，唯一`radio_b17_matched`GPU3已提交，仍用`semantic_reference_c9d3ea9`不可变源。不能照下方旧记录再启动另一run。双门初图本地`artifacts/semantic-agent-effect-20260918/b17_gate_previews/`已人工看；静态完整包在同根`b17_static_verified_bundle/`。

**2026-09-18 19:38当前：** `semantic_reference_c9d3ea9`固定c9d3ea9/2600725f正在供GPU3双门`gate_radio_b17`36340、`gate_plates_b17`36341使用，禁止热pull/修改。B17静态已完成、35667已请求TERM，`b17_static_complete.tar.gz`SHA8ee449e9…；策略/新模型尚未启动。所有目录均位于H08根`/mnt/sdc1/robodojo/behavior_dev/semantic_agent_effect_20260918`。

**2026-09-18 19:32当前：** B16静态34223已异常退出（2调用、0控制），held-reference unknown失败；34106核对专属命令后TERM，`b16_static_failed.tar.gz`SHA8129ffce…保留失败证据。B17新语义路由仅本地开发，无新物理或模型进程；旧f3f706d证据保留，不热修改。

**2026-09-18 19:21当前：** 唯一新静态服务`server_b16_static`34106/GPU1/8908/≤2调用，脚本`static_reference_b16`34223，固定源`git_worktrees/semantic_held_inspection_f3f706d`/digest8b001c6b/223CPU。这不是物理回合；两静态结果待验，源不热改。旧B14/15所有服务已退出；完整B14与部分B15都已本地SHA通过，B14另192hash/视频/本人5页核验。

**2026-09-18 19:19当前：** B14模拟器29038已结束、模型28821已请求TERM，`b14_complete.tar.gz` SHA25efcf4a…已在本地`b14_verified_bundle/`，745新控制/32生成、官方false但d3注册验证成功。B15静态32592异常退出（2实际调用，第三例0调用前检查失败），服务32343已请求TERM；`b15_static_partial.tar.gz` SHAd2bfae82…保留部分结果和日志。B16仅本地代码/223CPU，无服务器新源或运行；旧B13/B14策略都不重启。

**2026-09-18 18:55当前：** B14两门25757/25758已完整通过/退出，`radio_b14_matched`29038＋`server_b14_matched`28821/8909在GPU3，均固定`semantic_informative_probe_274479f`；源不可热改。B13标准完整本地`b13_standard_verified_bundle/`已6bcce747…SHA/168hash/全视频解码，旧回合/服务均结束。B15只有本地候选/保存帧CPU，无服务器新源或进程。

**2026-09-18 18:48当前：** B13两模拟器/两模型均确认退出。新固定`git_worktrees/semantic_informative_probe_274479f`（274479f/4fc431d0）正在被GPU3两个工程门`gate_radio_b14`25757、`gate_plates_b14`25758使用，源禁止热改；模型/匹配策略尚未运行。同H08根/同级日志。B13标准完整归档6bcce747…下载中，不能删除旧证据/旧B7前缀。

**2026-09-18 18:45当前：** 两B13策略19689/19690已结束，匹配服务18898已退出；标准服务18895已请求TERM、待最终退出核验。`b13_matched_complete.tar.gz` SHA588a57ae…已本地`artifacts/semantic-agent-effect-20260918/b13_matched_verified_bundle/`完整校验，含视频/24hash/人工4帧与track_review；`b13_standard_complete.tar.gz`正在归档，标准453新控制/官方false。旧af6283f源与旧B7前缀数据保留，不覆盖任何run。B14新源/两门/诊断尚未创建，后续GPU3有限预算见plan/config，不根据下方历史重启B13。

**2026-09-18 18:29当前：** af6283f/9c69bee4源正在被两个策略及两个服务使用，禁止热改。`radio_b13`19689/GPU1/8908与`radio_b13_matched`19690/GPU3/8909唯一运行；对应`server_b13_radio`18895、`server_b13_matched`18898，147/81限；目录/日志仍在`semantic_agent_effect_20260918`。两新工程门已通过且17003/17004退出，下面“等待门”为历史。匹配run读旧`radio_b7`四SHA固定文件，额外362动作前缀在manifest/结果/视频独立标注；不能清理旧B7数据或热改source。

**2026-09-18 18:24当前：** 新唯一源`/mnt/sdc1/robodojo/behavior_dev/git_worktrees/semantic_measured_approach_af6283f`固定af6283f/9c69bee4、双端206CPU；同H08根的`gate_radio_b13`17003/GPU1、`gate_plates_b13`17004/GPU3运行，不能热pull/改源码。`server_b13_radio`与`server_b13_matched`正在提交预载8908/8909，147/81模型调用上限；两个策略尚未运行，等待新门。B11及9792旧服务均已退出，完整本地`b11_verified_bundle/`已SHA/视频/120hash/人工7页核验。

**2026-09-18 17:58当前：** `radio_b11`12314已自然安全停/退出，result SHA83487814…、289新控制/41模型/官方false；仅闲置`server_b11_radio`9792已请求TERM，待核退出。完整`b11_complete.tar.gz`归档中，保留四工程门＋唯一策略＋服务/日志，旧source不改。没有任何新B12物理run。下方运行记录均为历史。

**2026-09-18 17:52当前：** 唯一`radio_b11`实际PID12314/GPU1，17:46启动，同源`server_b11_radio`9792/8908运行；两者固定`git_worktrees/semantic_goal_boundary_a4e3aab`、digest4cad025b，禁止热改。`gate_radio_b11_v2`9727/`gate_plates_b11_v2`9725已完整通过并退出，不再等待或重跑；新run/日志在原H08根。GPU3目前空闲，余盘97G；本地90c5913已push且fetch/pull无新团队更新。下方17:43“尚未启动”是历史。

**2026-09-18 17:43当前：** B11旧两门6944/6945已通过/退出，38个机器人link盒及实际四指位姿均已保存。新固定`git_worktrees/semantic_goal_boundary_a4e3aab`含189CPU通过的子目标边界/搜索记账修复；`gate_radio_b11_v2`9727、`gate_plates_b11_v2`9725在GPU3，不能热改。GPU1同源`server_b11_radio`9792/8908（147调用上限）预载，**radio_b11尚未启动**，必须等新两门精确digest通过。B8完整本地`b8_verified_bundle/`已SHA/视频/444hash/人工末段审查；B9/B10静态完整本地`b9_b10_static_bundle/`已SHA，旧服务3880在12次后已停，B9/B10均没有新物理策略。当前新run根仍同H08；不根据下方旧“尚无附着”重做B7。

**2026-09-18 17:17当前：** B8 task3已结束（1561控制/118模型/SEARCH_TRAVEL_BUDGET/官方false），服务4190951已停，`b8_complete.tar.gz`只读归档中；旧源4f48c60保留。B9两门完整通过但静态视觉缺陷未过，**没有radio_b9策略**；server_b9_radio3880/GPU1/8908同affcdec还在，供B10≤8静态（原4起）使用。B10新不可变源`/mnt/sdc1/robodojo/behavior_dev/git_worktrees/semantic_grasp_feedback_bca3d16`，176CPU；输出同H08根的`static_grasp_feedback_b10`、`static_grasp_geometry_b10`，0控制，物理门/策略均未启动。B7完整本地`b7_verified_bundle/`已SHA/视频/294hash核验，**d21–24确附着radio_89、d25恢复open释放**，下方旧“没有附着”判断由plan更正；operator_stop与全部原证据不覆盖。

**2026-09-18 16:56当前：** B7模拟器4185700经SIGINT人工早停、服务4184694已TERM；48完成/759新控制、服务104生成，没有result/failure（evaluator直接退出），`radio_b7/operator_stop.json`明确记录，不冒充完整终态。`b7_complete.tar.gz` SHA `b47e0b25e57c0dddab38a71173d9c594a519196b25d654ae0ce9a0c5195da0f9`开始下载。B9两门4193789/4193846在GPU1，源`semantic_grasp_geometry_affcdec`，未加载模型；B8 plates4191304及server4190951/8909仍在GPU3，源`semantic_navigation_4f48c60`。B6归档已在本地`b6_verified_bundle/`且744 RGB-D hash/两视频通过，人工选择时刻见plan；不要照下面旧状态重启任何旧run。

**2026-09-18 16:36当前：** `radio_b7`4185700/GPU1及`server_b7_radio`4184694/8908运行，源`semantic_probe_605512f`；第1次close空抓，未成功、不重启。`gate_radio_b8`4188030与`gate_plates_b8`4188040均GPU3，无B8模型，源`semantic_navigation_4f48c60`，通过/退出前不加载179调用的新服务。B6两策略及两旧服务均已停止，完整压缩归档SHA4c835b5b…正在传本地；B7/B8都仍同H-08输出根，不热改任一运行源码。

**2026-09-18 16:22当前：** B6两策略均结束、官方false（radio648新控制/89调用，task3 1711/128）；原9177016目录不修改。新源`git_worktrees/semantic_probe_605512f`固定605512f/159CPU，`static_probe_b7`对B6 d9/d19共2动作选择/0控制，复用旧radio服务4170243/8908（89起）；旧plates服务4170359/8909闲置待停。B7工程门/策略尚未启动，不能将旧B6门用来验新source。B5完整归档本地已SHA与两视频/216hash验证，位置`artifacts/semantic-agent-effect-20260918/b5_verified_bundle/`。

**2026-09-18 15:53当前：** B6两工程门完成、对应进程4167916/4167977退出；唯一`radio_b6`4172010/GPU1、`plates_b6`4172067/GPU3运行中，同源`semantic_identity_b6`固定9177016。服务`server_b6_radio`4170243/8908、`server_b6_plates`4170359/8909同源0起/147与179调用上限，不热改。先前B5都已接口失败退出，不能按旧运行状态重启；B5完整归档c43131f0…传回本地中。源/配置/两门SHA与完整预算见plan、h08_b6.json及H-08报告。

**2026-09-18 15:46当前：** B5两个策略均因输入token超限退出（radio72新控制/11生成，task3 720/33），无方法成功结论。旧服务4136279/4157077已停止（93/33总生成）。新源`/mnt/sdc1/robodojo/behavior_dev/git_worktrees/semantic_identity_b6`固定9177016/150CPU，`gate_radio_b6`、`gate_plates_b6`与`server_b6_radio`/`server_b6_plates`正在启动/预载，GPU1/3、端口8908/8909、147/179调用上限；输出根保持`semantic_agent_effect_20260918`，策略尚未启动。`static_context_b6`保存2原prompt/精确token与动作，`static_identity_b6`2观察中第二是冰箱，`static_identity_positive_b6`才是餐桌正例。不可把下面旧B5“运行中”当现状。

**2026-09-18 15:22当前：** B5两门已完整通过（385/396新控制、48/48在线视觉位移有效、digest0a3d0085…），4155037/4155097已退出。唯一`radio_b5`4158141/GPU1/8908、`plates_b5`4158203/GPU3/8909正在初始化，均源`semantic_contact_gate_ec1a63e`且OMP/MKL/OpenBLAS=4，不能热改。radio服务4136279保留77/224起，旧plates4136398已TERM115次，替为新同权重`server_b5_plates`4157077/8909/0起≤179次。完整B4归档已本地SHA匹配/视频全解码，结果见实验文档；下文待换新/待启动描述均是历史。

**2026-09-18 15:16当前（覆盖下面历史启动状态）：** B4两策略已结束，无官方成功；旧radio服务4136279/GPU1/8908仍在（77/224调用，源39881cf），task3旧服务4136398/GPU3/8909（115/224）待核身份后替换。B5不可变源`/mnt/sdc1/robodojo/behavior_dev/git_worktrees/semantic_contact_gate_ec1a63e`已145 CPU，两`gate_radio_b5`/`gate_plates_b5`启动，根`/mnt/sdc1/robodojo/behavior_dev/semantic_agent_effect_20260918`，日志同级。策略尚未启动；每个新仿真进程OMP/MKL/OpenBLAS=4。两主接触静态及SHA已在本地artifacts，B4完整归档仍传输中。禁止热改这些源/把下文旧PID当现状。

**2026-09-18 14:22 H-08策略运行：** `radio_b4`4143759/GPU1/8908、`plates_b4`4143760/GPU3/8909已唯一启动，正在初始化，run与日志均在同H-08根；各自原预算/前缀不变，仿真14a33ec、服务39881cf，不热改。下条的准备提交已更新为运行中，结果仍待验。

**2026-09-18 14:21 H-08当前：** 运行源`/mnt/sdc1/robodojo/behavior_dev/git_worktrees/semantic_selfdepth_14a33ec`，两`gate_radio_b4_fast`/`gate_plates_b4_fast`已完整通过，385/396新控制、digest ac733fff…；旧39881cf两慢版门中断保留（无完整result）。B4服务4136279/8908/GPU1与4136398/8909/GPU3仍固定`semantic_selfdepth_39881cf`（仅仿真优化，HTTP源码未改），同27B各0/224调用。策略`radio_b4`/`plates_b4`准备唯一提交；根仍`semantic_agent_effect_20260918`、同级日志。B3完整归档SHA16cca5a0…和静态归档SHA40aa687e…均已到本地并校验，B3两视频已全解码与人工选帧核验。旧启动/传输记录均为历史，不热改任何在用源。

**2026-09-18 13:56 H-08最新：** B3两策略均结束（radio738新控制/105调用，task3 1580/128，官方false），两模型服务已停止，radio另4静态合计109。完整`b3_complete.tar`SHA16cca5a0…传回本地中；`b4_static.tar`SHA40aa687e…为4静态及服务调用证据。新不可变源`/mnt/sdc1/robodojo/behavior_dev/git_worktrees/semantic_selfdepth_39881cf`已robo123CPU，唯一`gate_plates_b4`4134086/GPU3启动核验本体visual mesh/self_depth；radio新门与B4策略尚未提交。新旧run都在原H-08根，日志同级；禁止热改或把下面旧启动记录当现状。

**2026-09-18 13:19 H-08 B3运行：** `gate_radio_b3`/`gate_plates_b3`已通过（385/396新控制、48/48视觉运动、a2f598f3…）。唯一`radio_b3`和`plates_b3`在H-08根启动，预算分别64/2048/1800s、80/2400/2400s；`server_b3_radio`4125196/GPU1/8908、`server_b3_plates`4125250/GPU3/8909各224调用，均同0e79cd6/27B revision。所有源码从`semantic_odometry_0e79cd6`运行，禁止热改；当前策略尚无终态，新日志同级`.log`。

**2026-09-18 13:15 H-08 B3当前：** 不可变源`/mnt/sdc1/robodojo/behavior_dev/git_worktrees/semantic_odometry_0e79cd6`固定0e79cd692d936c56081a6b25ded0c2b17218ba63，robo113CPU通过。H-08根下`gate_radio_b3`/`gate_plates_b3`真实工程门正在初始化；`server_b3_radio`GPU1/8908、`server_b3_plates`GPU3/8909各同27B/≤224调用开始加载。旧`server_b2`已停（仅2静态）；B2两策略没有启动。`odometry_probe_b2`217控制/0神经完成，归档SHA71c1e712…已在本地，物理真值只存诊断。B3两策略仍等新门，不热改/重复提交。[实时计划](plan.md)

**2026-09-18 12:50 H-08最新：** B1两策略以`failure.json`终止，server_b1停止，82生成；511MiB `b1_complete.tar`/SHA bfc7f22f…保留所有旧门/策略/服务，正在下载。B2新源`git_worktrees/semantic_refined_160abd5`（105CPU），`gate_radio_b2`4117180/GPU1与`gate_plates_b2`4117232/GPU3正在跑；`server_b2`4115322/GPU1/8908仍固定`semantic_refined_6d18204`，2/256调用，两源都不能热改。`static_surface_b2`为完成的2神经/0控制选点，tar SHA94f3080d…与本地一致。B2短闭环未启动，不复用B1旧门。[H-08报告](experiments/2026-09-18-semantic-agent-effect.md)

**2026-09-18 12:36 H-08当前：** 新根`/mnt/sdc1/robodojo/behavior_dev/semantic_agent_effect_20260918`；B1不可变源仍`git_worktrees/semantic_grounded_e6925d8`。`gate_radio_b1`/`gate_plates_b1`两门已通过；`server_b1`（4111632/GPU3/8907，固定27B、≤220调用）、`radio_b1`（4111945/GPU1）、`plates_b1`（4112002/GPU3）正在跑，日志同级`.log`。每策略48决策/1536新控制/1200s、0训练。本地证据目标`/home/wsy/behavior/artifacts/semantic-agent-effect-20260918/`。这是用户新授权后的独立H-08，下面H-07等待/停止记录保留为历史，勿重复提交或热改源。

**2026-09-18 12:10 H-07修复源码：** `/mnt/sdc1/robodojo/behavior_dev/git_worktrees/semantic_grounded_e6925d8`固定e6925d87f90d12a7ab6237cb99daf3279cb3b646，模型环境Python3.10 96/96 CPU通过（3.919s）；仅CPU验证，未启动新神经/模拟器，不拿9993fab旧门当新源回执。旧失败结果/最新控制门/全部下载证据仍在下方H-07根，本地artifacts约988MiB。新物理复验预算待用户答复。

**2026-09-18 12:06 H-07真实状态：** `radio_27b_v1`与`plates_27b_v1`因首轮/次轮观察字段契约失败退出，分别18 HOLD新控制/0控制；以各`failure.json`为终态，没有`result.json`，不是仍在运行/有效SR。`server_27b_v2`6次生成已停（4104006退出，两模拟器4104443/4104495也退出），没有新模型权重。`pilots_v1.tar`与本地SHA8817a4c2…一致，含两run/日志/服务逐调用；`plates_pilot_v1.tar`为此前单独保存的同次task3失败证据。新字段兼容修复只在Git本地进行，尚无新物理run；下方启动记录为历史。

**2026-09-18 11:54覆盖下方H-07运行状态：** 9993fab最新两门已完成（共同digest88a8c268…），首帧人工核验通过。新`server_27b_v2`（4104006/GPU3/8907、最多218调用）正在服务；H-07根`radio_27b_v1`/`plates_27b_v1`及同级`.log`为本轮新两策略短测，已提交、结果待验，勿与H-06同名run混淆。`semantic_grounded_9993fab`正在使用，禁止热pull/修改。`gates_v3.tar`新两门证据已与本地SHA一致，旧gates/static不覆盖。[实时计划](plan.md)

**2026-09-18 11:48 H-07（Codex，首帧修复复验运行中）：** 根`/mnt/sdc1/robodojo/behavior_dev/semantic_agent_grounded_20260918`，最新独立源`/mnt/sdc1/robodojo/behavior_dev/git_worktrees/semantic_grounded_9993fab`，分支`feat/semantic-agent-grounded-20260918`。`gate_radio_v1`/`gate_plates_v1`初始化失败、0控制；d2da0ac的`gate_radio_v2`/`gate_plates_v2`控制完成，但task3初图陈旧，不能放行新源。新增`gate_radio_v3`/`gate_plates_v3`及同级`.log`正在重新验证RGB-D同步。`static_27b_v1`已6次静态调用/0执行，task3陈旧帧只算schema检查；`server_27b_v1`已停止。`gates_and_static_v2.tar`为233MiB证据归档，本地传输目标`/home/wsy/behavior/artifacts/semantic-agent-grounded-20260918/`，产物均不入Git。权重只读复用下方H-06的27B，未复制/下载；新策略尚未启动。[设计](SEMANTIC_AGENT_GROUNDED.md)

**2026-09-17/18 H-06语义agent v2（Codex，本轮有界评估结束）：** 源码分支`feat/semantic-agent-v2-20260917`，根为`/mnt/sdc1/robodojo/behavior_dev/semantic_agent_v2_20260917`。所有本轮模型/模拟器已停止，结果与下载的官方权重保留；两条策略短测均未成功，不覆盖下面v1。[最终报告](experiments/2026-09-17-semantic-agent-v2.md)

| v2位置（相对新根，除非写绝对路径） | 内容/边界 |
| --- | --- |
| `models/Qwen3.8-27B`、`download_reference.log` | 官方冻结revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`，55,586,036,737字节，下载回执/逐文件大小核验；不是微调权重 |
| `gate_radio_v1` / `gate_radio_v2`及同名`.log` | 分别为libGLU初始化失败、COM/link原点参考点失败，原证据保留；不能当成功门 |
| `gate_radio_v3` | `3869ed8`修正质心Jacobian后复验通过，349控制；不是最终有限轨迹版本，校准JSON是机器人资产，不含场景对象状态 |
| `gate_plates_v1` | 旧短窗口渐进IK失败，37控制后停止；保留与修复后同姿态的物理对照，不是通过门 |
| `gate_radio_v4` / `gate_plates_v2` | `b620b3b`有限轨迹两门已通过：24到达/361控制、22到达＋2运动前拒绝/371控制；前者448前缀另计，后者0；与26de8ad同digest，不适用于e7478cf新harness |
| `static_review_inputs_v1` | 15原始状态及三视角人工审核拼图；此版本的几何引导来自失败校准，不可作为正确标尺 |
| `static_4b_v2` | 修正校准后15状态/30调用完成，radio仅2/10识别，静态门未通过、未放行4B物理回合；不作为新训练集 |
| `static_27b_v2` / `static_27b_v3_transport` | 前者保留原18调用/12围栏解析失败，后者校验hash后复用原观察＋3动作、仅补12动作；合计30独立生成，不重复模型采样 |
| `server_4b_v1`、同名`.log` | 已停，30调用；原v2协议服务8907，复用旧4B权重，不与原v1服务/队友服务混淆；调用及输入hash留证 |
| `server_27b_v1` / `server_27b_v2` | 分别18/12静态调用；两服务均已停止，后者于22:17释放GPU供双场景验收。若后续重载使用新目录，不覆盖这些身份/调用日志 |
| `server_27b_v3`、同名`.log` | 26de8ad/GPU3/8907，闭环48调用（2规划＋24观察＋22动作），原PID4030268已停；不是新微调权重 |
| `radio_27b_v1` / `plates_27b_v1`及同名`.log` | 26de8ad最后两条闭环已结束：task0 train138/env0/seed0、448专家前缀＋98新控制；task3 train242/env0/seed0、0前缀＋235控制。均恢复耗尽、官方成功false；result/manifest/plan/逐决策JSON、原图、rollout.mp4全部保留 |
| `/mnt/sdc1/robodojo/behavior_dev/git_worktrees/semantic_v2_e7478cf` | 最终micro接近/恢复计时修复的Git独立源；2026-09-18模型环境Python3.10全部64 CPU通过，0新闭环/训练。不要用旧控制门绕过新digest检查 |
| `/mnt/sdc1/robodojo/behavior_dev/git_worktrees/semantic_v2_*` | e19940f下载、be00be7服务、47b992d旧门、3869ed8修复门等独立不可变源，禁止热pull/修改 |
| 本地`/home/wsy/behavior/artifacts/semantic-agent-v2-20260917/` | 原/修复校准、15状态审核、两条闭环同名完整目录及server_27b_v3调用日志；最终两门在`gate-radio-v4`/`gate-plates-v2`。两条rollout.mp4已SHA/全解码/人工抽帧检查；不入Git，不因ignore而删除 |

v2预算≤65GiB新增并至少80GiB可用，0训练；共享解释器和依赖仍见下表。模拟器入口仅在本私有进程re-exec时补既有pymeshlab的libGLU目录，不升级共享OG/conda。最新进程/资源状态只以plan与实查为准。

**2026-09-17新增语义微动作试验（Codex）：**

19:35收尾：下面4个物理run均结束，模型服务3982269/3982591已停止，GPU1/3本轮工作已释放；队友3898152/3898758仍在。总目录14,231,956,955字节/约13.25GiB，0清理旧文件。三条VLM短测无任务成功，并暴露IK限位边界，不可把目录complete当方法已验收。[报告](experiments/2026-09-17-semantic-agent-pilot.md)

| 位置 | 内容/边界 |
| --- | --- |
| `/mnt/sdc1/robodojo/behavior_dev/semantic_agent_20260917` | H-02/03 run根；`gate_v1`是24命令控制门，`radio_2b_v2`/`radio_4b_v2`为448前缀局部闭环，`plates_4b_v2`为0前缀task3；均有result和video，失败亦保留 |
| 同根`models/Qwen3.5-2B`和`models/Qwen3.5-4B` | 官方冻结权重，revision分别`15852e8c…`/`851bf6e8…`，不是新训MEM-Lite或Show-Harness adapter，不入Git |
| 同根`deps` | transformers5.7.0依赖overlay，经显式PYTHONPATH供新服务使用；未升级共享G05的4.57.1。模型Python仍复用原torch2.7.1；仿真Python不使用此overlay |
| 同根`server_*` / `bench_*` / `sanity_*` | 身份、调用清单/图像prompt hash、同图延迟与无执行输入检查；旧8897/8898私有服务已停止，8＋60＋108调用日志保留 |
| 同根`audit_saved_inputs_v1` / `audit_saved_inputs_v1.log` | H-05保存状态审计：3状态×6配方=18静态调用，0控制/更新；原始prompt/媒体hash/token/输出完整保留，进程3993804已于20:07核验退出；结果SHA `0882a4b2…` |
| `/mnt/sdc1/robodojo/behavior_dev/git_worktrees/semantic_agent_*` | `1680e01`/`f18dd43`/`a7880ec`/`fa9db1b`/`d312380`/`1d872d7`不可变Git快照；运行已结束，仍保留用于重现，不用新代码覆盖旧版本 |
| `/mnt/sdc1/robodojo/behavior_dev/git_worktrees/semantic_audit_a870cff` | H-05诊断脚本不可变源码；不连接模拟器，原生产控制器没有改动 |
| 本地`/home/wsy/behavior/artifacts/semantic-agent-20260917/` | 已传回的控制门/闭环视频、审核图和结果副本；被忽略不代表可删除 |

本次查`/mnt/sdc1`约181G可用，`/mnt/tmp1`不再存在；下表的归档盘是9月12日历史记录，后续清理必须重新核对挂载，不能照旧路径移动。本次未清理/迁移任何旧文件，新增试验限定25GiB内。

| 类别 | 当前服务器位置 | 用途与处理原则 |
| --- | --- | --- |
| 原主代码工作区 | `/mnt/sdc1/robodojo/GalaxeaVLA` | 用户指定迁移来源；源码已复制到本地，原31个已修改/90个未跟踪路径及`.git`保留。不直接pull覆盖旧改动 |
| 新GitHub协作入口 | `/mnt/sdc1/robodojo/behavior` | 已建立干净clone，核验时与本地/GitHub同为`c3367995234ce51964fff736f2544610c0320419`；后续文档commit会继续前进。不以它冒充最新A3实验快照 |
| 当前训练/推理解释器 | `/mnt/sdc1/robodojo/GalaxeaVLA/.venv/bin/python3.10` | 多个模型服务正在使用；不入Git，不迁移/删除 |
| 历史研究/运行副本 | `/mnt/sdc1/robodojo/behavior_dev` | 独立源码、run、评测、对齐诊断；不是一个可以整体删除的cache |
| 本轮研究根（下文记W） | `/mnt/sdc1/robodojo/behavior_dev/direct_execution_L6rqZ6_20260910` | A/B权重、运行配置、候选源码、视频/物理trace；按run/用途核验后归档，不按目录名删 |
| 五任务转换数据 | `/mnt/sdc1/robodojo/datasets/behavior2026_g05_tasks_0_4` | 当前实际训练数据，44个data parquet；raw action23/state61。不能误把它当临时转换文件删除 |
| train-only归一化统计 | `/mnt/sdc1/robodojo/stats/g05/behavior5_r1pro_trainonly_taskstrat5_stats_v2.json` | 五任务训练/部署共同依赖；SHA `846bcbeac181df5555cb5d40d4183d61743a556df5cf8c8a8fb1bc17e2a40b19` |
| 原始官方演示 | `/mnt/sdc1/xhz/BEHAVIOR2026/2026-challenge-demos` | xhz目录，不在此次robodojo清理范围；数据来源依赖 |
| BEHAVIOR/OmniGibson | `/mnt/sdc1/xhz/BEHAVIOR2026/BEHAVIOR-1K/OmniGibson` | 当前真实仿真框架；不升级或热改来“顺便清理” |
| 仿真解释器 | `/mnt/sdc1/xhz/miniconda3/envs/behavior/bin/python` | 与模型Python不同，当前Python3.11；不在清理范围 |
| 稳定动作/协议adapter | `/mnt/sdc1/robodojo/behavior_dev/GalaxeaVLA_memlite_coordination_dev_20260908/sim_runtime/production_native_oracle_low_v1` | 旧名字不代表闲置，多项正式服务仍依赖 |
| 版本化标签/审核 | `/mnt/sdc1/robodojo/behavior_dev/GalaxeaVLA_memlite_coordination_dev_20260908/data` | sidecar、release/审核清单和相应版本；有些实验另有overlay，实际入口以该run配置为准 |
| 本次归档盘 | `/mnt/tmp1` | 已核实是另一块XFS文件系统，约894GiB总容量，不能容纳2T归档；复制/校验/是否可释放源副本见末节 |

这不是说50任务数据已全部就绪。全任务数据清单、覆盖、切分和规模仍是大训练准备项，不可从五任务目录推断。

## 2. 当前A3与B-final：别拿错权重或源码

**2026-09-14 19:30重要更正：** 以下“CoT2032144/EMA1707751/tail1820879活跃”为19:17以前快照。原CoT在19:18:02被账号ssy手动kill进程组，19:18:07写failed；正式最后39/624抽取、没有`formal/checkpoints`。后继EMA/tail各因未完成前驱退出、0更新，原source/worktree仍保留，不能把它们当正在排队。事故证据/恢复边界见`docs/experiments/2026-09-14-cot-external-stop.md`；三份v1 run与status/日志不覆盖，不重启旧服务3543247。协调GPU使用前不提交CoT替代轮或恢复EMA/tail；现有已完成marker/FM/LoRA审计结果全部保留。

**2026-09-14 19:17更新：** `dual_track_fm_ar_20260913/lora_capacity_a4_fm500_v1/result.json`只读审计已完成（SHA `0bb9958df841577b77285c115119f421d73786a567cf8e06f5213e45a370b06e`，2权重/192对/0神经与训练）；`git_worktrees/capacity_audit_20260914`最终固定e50e7b5/17 CPU，CPU任务已退出，不再重跑。a077cac仅是此前错误配置路径的历史入口；原A4实际配置在`overnight_a4_20260912/formal/.hydra/config.yaml`，SHA4d45b4c2…。新容量训练尚未登记/启动；原CoT2032144/EMA1707751/tail1820879活跃源不能pull。

**2026-09-14 19:08更新：** `dual_track_fm_ar_20260913/ar_marker_actions_v1/result.json`完整十窗结束（SHA `831cbd40586ba5c1fca9481485072fe964030c46b3cc9f84c47ef32f6eb64330`），2021126退出；该完整输出/原动作留服务器，轻量摘要已在docs/experiments/results。所有10自由格式正确但总体动作误差未胜FM，无marker新物理视频。原CoT220792已正式2032144，source d114581仍活跃不能pull，smoke权重SHA `12ae130c87d73525c4269ca6c9fe6684ad94aff345a339db68ffb3ed8cfbad1a`；EMA/tail维持原等待。新`git_worktrees/capacity_audit_20260914`固定a077cac/16 CPU，仅用于2权重/192对LoRA只读审计、无训练；结果尚待验，不重复启动。

**2026-09-14 18:58更新：** 原`ar_a4_marker_fulltrain_v3/formal/checkpoints/step_500.pt`完整验收（SHA `3801388d71381c4cd586dac4bc19b07164e8922b8de6b5ea869bdcc52a56b52b`，inspection `b848f38026e5583f0cfa4d3471cd8647ed0eddb70a36a366617a5e5f73ce43a4`），不是下面历史302/500状态。唯一最终十窗`dual_track_fm_ar_20260913/ar_marker_actions_v1`于18:57:18启动2021126；源`git_worktrees/marker_actions_20260914`固定81810be97b108d1e843844139535c86931c60a80、42 CPU passed，正在使用不可pull；同级`.launch.json/.launch.log`记录原10 AR/0训练仿真。输出结果仍待验，不重启。CoT原220792已smoke2019434，EMA/tail仍原等待，不改三处活跃源。

**2026-09-14 18:10新后继：** `/mnt/sdc1/robodojo/behavior_dev/git_worktrees/tail_lr_screen_20260914`固定2300c50/299 CPU，现在由1820879活跃等待器引用，禁止热pull。唯一`dual_track_fm_ar_20260913/fm_tail_lr_v1`的spec SHA `55bdce00df92b30b8d75dd3b465134e4dc9664907037de64a558cc22a55932d4`、launch SHA `6f4ca2522dd63e75048a4659fd439c5e9f4c63bead4bc6bc139353233d176f32`；18:09真实等待EMA1707751/ticks336359196，0GPU/0更新。原marker1563012训练、CoT220792→EMA1707751继续原队列，尾端候选不接前驱权重/不叠EMA；smoke/formal尚未开始，不重复提交。marker200阶段结果原文件在`ar_a4_marker_fulltrain_v3/formal/eval_step_200.json`（059b0b4d…），不是新500权重。

**2026-09-14 17:46新队列：** `/mnt/sdc1/robodojo/behavior_dev/git_worktrees/ema_screen_20260914`现在固定d28581a（264 CPU通过）并已用于活跃EMA等待器1707751，**不可热pull**；ee970b0是同目录较早已结束CPU检查的历史commit。唯一run `dual_track_fm_ar_20260913/fm_trainable_ema_v1`含`method_spec.json`（SHA `d195fd580899e5d947df5daa8d768d6be44315ca61a7c441beab159e2d8fd2b7`）、launch/status/supervisor.log；当前0GPU/0更新，等待CoT220792/ticks334478266。smoke/formal尚未创建，不重排；将来每阶段`ema/`保存配对评估，完整online+EMA+Adam同checkpoint，`ema_*_rank*.json`记录实际内存/影子/还原。原marker1563012真实训练、CoT原等待，旧9f26b45/d114581均不变。

**2026-09-14 17:16更新：** `dual_track_fm_ar_20260913/m04_actions_ki_v1/result.json`真实十窗complete（SHA `8e5434b647bdf40117bf16abf5b950508941d9f85805c3a9b7e0c54c6ca6c29c`），1547266退出；三臂30动作已全部结束，不重复生成。marker原4129562已进入正式1563012（9f26b45），CoT220792仍等marker；这是训练中的后继，不是旧KI仍存活。完整结果/输入配对见docs的M-04报告，未启动该方法的模拟器。

**2026-09-14 17:08更新：** 双路线根`ki_a4_fulltrain_v3/formal/checkpoints/step_500.pt`已完整验收，SHA `3efc6d1ab77cd02b01fa0a3db92262d522636fa2085e107dca04d7f3163e45ea`；原4129539/1031803退出，只读监控38073正常结束。`m04_actions_ki_v1`最后十窗17:07:41启动1547266，固定44255a4、同级`.launch.json`/`.launch.log`；尚无最终动作结果，不重复启动。marker4129562已进入smoke1543690，CoT220792仍等待；各活跃源不热pull。

**2026-09-14 16:40候选CPU源：** `/mnt/sdc1/robodojo/behavior_dev/git_worktrees/module_clip_20260914`固定dc2fe36，新增分模块裁剪helper/44 CPU tests已通过（0.18s）；不是新训练run，尚无真实策略更新/权重或GPU-DDP效果。旧9f26b45训练、d114581等待、44255a4动作入口均未更改，不能把新helper自动带入旧运行。

**2026-09-14 16:27更新：** 下述双路线根`m04_actions_joint_v1/result.json`已真实十窗complete（SHA `3e1f5eb6b3c2f9437aae4083c1a0d56fb0f8c647fad6aa032eabdb68de706dec`），1324567退出，完整恢复与FM实际输入指纹配对通过；以下16:19“结果待验”为历史启动快照。FM/joint不重复生成，`m04_actions_ki_v1`尚未提交，待KI完整500保存门；原AR等待器保持。

**2026-09-14 16:19更新：** 双路线根`/mnt/sdc1/robodojo/behavior_dev/dual_track_fm_ar_20260913`中，`joint_a4_fulltrain_v3/formal/checkpoints/step_500.pt`已完整验收（SHA `e1a67647570ae0706503625b0891086e1a15474e335177aed6b44a35553b9fd5`），对应eval/inspection均已写出；`m04_actions_fm_v1/result.json`真实完成十窗（`d3502794…`），原两PID退出。现有44255a4的`git_worktrees/m04_actions_20260914`已唯一启动joint原十窗1324567；输出`m04_actions_joint_v1`与同级`.launch.json`/`.launch.log`，结果待验，不重做FM/CPU门。KI原1031803/4129539实际训练，marker4129562/CoT220792等待；活跃源不热pull。

**2026-09-14 13:05最新：** `fm_action_control_v3`13:02:59完整500验收完成、旧4092082/4092140退出；最终权重SHA `efce4dfe232f85ac18f7fca66b562360ba23d839f3f74748f658b5911b5c33f9`，固定80=0.19756705752806739。首个`m04_actions_fm_v1`于13:04:38启动410326，使用原44255a4的`m04_actions_20260914`，同级`.launch.json`/`.launch.log`为启动证据；十原状态/0训练/仿真，结果待验。此源现活跃，不可pull；不要重复启动。

**2026-09-14 12:59已完成：** `git_worktrees/action_groups_20260914`固定64e9ff8/51 CPU；`dual_track_fm_ar_20260913/fm_saved_action_groups_v1`保存十原目标/40已有预测的组误差，358858及子进程已退出，result SHA `3669ad81ba2b2fe66fc35cef0df8d9c132312ebfe96f08abd84fb04f6397edc1`。0新模型调用/训练/仿真；完整run保留，轻量报告已入docs，不重复此诊断。

**2026-09-14 12:39准备完成：** `git_worktrees/m04_actions_20260914`固定44255a4/161 CPU，新增`probe_m04_actions.py --run <fm_action_control_v3|joint_a4_fulltrain_v3|ki_a4_fulltrain_v3> --gpu 1`；每臂完整500保存/SHA通过才允许在`dual_track_fm_ar_20260913/m04_actions_{fm,joint,ki}_v1`各一次十状态连续动作生成，当前三个输出尚未创建/无进程。不会重跑M-02或A4缓存，完整预算见plan。

**2026-09-14 12:33最新：** 新`git_worktrees/ar_cot_screen_20260914`固定d114581/165 CPU，仅供`dual_track_fm_ar_20260913/ar_native_subtask_cot_fulltrain_v1`，supervisor220792已真实等待marker4129562，spec SHA `3dffcd78de30bff9d19c3e2b1658433bda29e0a215d4b48bf7630bf3d2d37863`。这是第五个既定串行训练/等待器，不是已训练CoT权重；旧FM8fcf61f/后三臂9f26b45源不变，三处活跃源均不可热pull。先核验status/launch，不重复提交。

**2026-09-14 12:17补充：** 三臂机器/本人审核JSON及六张拼图在服务器`dual_track_fm_ar_20260913/fm_screen_prefix512_v1/review_by_main_20260914/<arm>/`亦有完整校验副本，6 JSON/6 PNG与本地SHA相同；不覆盖原completion历史。当前仅方法训练链继续，FM已270/500，三个原后继等待器不重复启动。

**2026-09-14 12:11最新：** `fm_screen_prefix512_v1`三臂各512全部完成，1536动作及本人87视图核验完成；`recovery`supervisor34585、Beta34590/39113、exec59091/64006均退出，原control三进程此前也退出。三臂各`completion.json`和root completion的`personal_review_pending`是当时不可变历史；实际已完成的后续人工/机器回执在本地`/home/wsy/behavior/artifacts/experiments/2026-09-14-fm-screen-prefix512-v1/<arm>/review/`，详见[报告](experiments/2026-09-14-fm-prefix512-screen.md)，不覆盖旧回执。三视频已本地，非cache可随意删。当前活跃仍是FM v3 4092082/4092140和9f26b45的后三训练等待器，不热pull它们。

**2026-09-14 11:57最新：** `fm_screen_prefix512_v1/fm_control_v1`已完整512并完成本地审查；本地`/home/wsy/behavior/artifacts/experiments/2026-09-14-fm-screen-prefix512-v1/fm_control_v1`含视频和`review/`。原三臂supervisor4177428在第二臂前端口检查停止，旧三进程均退出；保留`supervise_failure.json`。新显式恢复34585使用`git_worktrees/fm_screen_prefix_resume_20260914`固定d5f0aec/145 CPU，只接Beta服务34590/8787→exec/8788，原manifest不改、补充`recovery.json`/`recovery.launch.json`，源不可热pull。原A4只读服务文件也在被引用，勿修改。FM v3正式168/500。

**2026-09-14 11:44最新：** `dual_track_fm_ar_20260913/fm_screen_prefix512_v1`为新三FM500局部闭环，supervisor4177428；其`fm_control_v1/`下专用服务4177433/8786正在初始化，后两臂待串行执行，均最多512模型控制。编排源`git_worktrees/fm_screen_prefix_20260914`固定7b806b6/137 CPU，神经服务继续只读使用原`git_worktrees/a4_prefix_20260913/scripts/experiments/serve_low_fm_prefix.py`（`c4bc099d…`）；两处运行期均禁止热pull。各臂service/socket/rollout/completion分别给出真实阶段；manifest SHA `d37907aa…`，不是全任务SR或数据release。FM v3正式100/500，后三臂等待。

**2026-09-14 11:34最新：** 新`git_worktrees/reference_queue_20260914`固定9f26b45/117 CPU，现被joint4129515、KI4129539、marker4129562三个v3等待器使用，禁止热pull；三者依次等待FM4092082→joint→KI，原5＋500。各`launch.json`/`method_spec.json`/`status.json`已存在并核验，FM仍使用独立8fcf61f源、formal42/500。下方“未创建”是11:32历史，已由此次真实提交覆盖。

**2026-09-14 11:32更正：** `fm_action_control_v3`4092082/4092140已进入真实更新（11:30为22），初始80为原精确0.19694399407017044。joint/KI/marker三个v3目录尚不存在；第一次串行提交在创建joint目录前因前驱名单遗漏安全拒绝，待新固定源修正后补提交。8fcf61f活跃源禁止热改；旧结果/服务均保留。

**2026-09-14 11:19最新：** `fm_initial_reference_probe_v1`（8前向）、`fm_full_reference_probe_v1`（四卡80窗/160前向）和`fm_screen_actions_v1`（30 FM生成）均已complete退出，result SHA分别`379f7945…`、`ddedb3b7…`、`a9906fa5…`；0新优化/仿真，不是新成功率。`git_worktrees/reference_recovery_20260914`固定8fcf61f/111 CPU，现在用于新恢复训练，禁止热pull；`fm_action_control_v3`4092082已提交（spec `5169c020…`），后继三臂提交中，真实阶段见各status/launch。旧v1/v2及两个已完成FM500全部保留。

**2026-09-14 10:43实际状态覆盖以下旧记录：** 六个恢复supervisor均已退出。`fm_beta_stratified_v2`与`fm_exec_weight2_v2`正式500完成，checkpoint SHA分别`05ea17bc…`/`101c4b32…`；`fm_action_control_v2`smoke5通过，formal仅到`eval_step_0.json`即参考值断言失败、0正式更新；joint/KI/marker未训练即前驱失败退出。旧7572ce2源及失败目录都保留，不在旧run覆盖/重启；后续修复用新Git固定源，准确新run另记。新FM报告见[结果](experiments/2026-09-14-fm-method-screen.md)，约461GiB磁盘可用；旧六服务不动。

**22:44最新状态：** `ar_native_prefix_pilot_v1`唯一448＋256控制已完成，3191511/3191531/3196694全部退出，8785私有服务已正常停下；完整结果SHA `5a200786…`，视频SHA `e845f688…`。约36MB结果已复制本地`/home/wsy/behavior/artifacts/experiments/2026-09-13-native-ar-prefix-v1`，`actual_rollout/rollout.mp4`可看，`review/`有全动作/物理校验及29视图人工记录；[摘要](experiments/2026-09-13-native-ar-prefix-review.md)区分短测未成功与SR。原wire单次3140666也早已complete退出；两个结束源保留证据，不再写成活跃服务。只有恢复队列7572ce2及其Beta正式3141061正在训练/等待，不得热pull；旧六服务仍保留。

**22:21新实际物理作业：** `git_worktrees/ar_native_prefix_pilot_20260913`固定989a575/290 CPU passed；双路线根`ar_native_prefix_pilot_v1`于22:19:21提交3191511，manifest `eb7f99d6…`，端口8785仅loopback，native500＋schema＋task-only。当前服务初始化，后续socket/资源门通过才进唯一448prefix＋最多256模型控制；实际进度看`service.log`、`socket_gate.json`、`rollout.launch.json`、`actual_rollout/collection_result.json`，不是已完成结果。全部旧服务保留、该源码不可热pull。

**22:02续记：** `fm_beta_stratified_v2/smoke/checkpoints/step_5.pt`已实际保存/回读（`5bc38337…`，504 Adam/冻结不变/80来源），正式500尚待验。新`git_worktrees/ar_native_actor_wire_20260913`固定0701fd1/274 CPU通过；双路线根`ar_native_actor_wire_probe_v1`于22:01:12启动3140666，日志为同级`.launch.log`，原native500/一次生成/0训练仿真。该独立源正在使用，不热pull；旧六服务/旧v1证据保留。

**21:53六个显式恢复run：** 固定源码`/mnt/sdc1/robodojo/behavior_dev/git_worktrees/method_queue_recovery_20260913`（7572ce2，267 CPU passed），已活跃，禁止热pull。run仍在双路线根，原v1失败保留，均新建v2、原A4独立5+500、不追加formal预算。Beta在真实gate，后五个已核验等待/0更新；阶段见各自status.json、日志与launch.json。

| run | supervisor | method_spec.json SHA256 |
| --- | --- | --- |
| fm_beta_stratified_v2 | 3131633 | `16dfb7ded0b1773fd755d1e5cb9f05d19d093e65c385233fe9de56d69bb8a0a1` |
| fm_exec_weight2_v2 | 3131858 | `c8ab308b18dd7784139a8e4b2e16f81528eece38951064f0b17f827a4b3c38b1` |
| fm_action_control_v2 | 3131867 | `09b3bb2a385b7b45266238f097327bcbf49c95c0670ac589da889692bb398c0c` |
| joint_a4_fulltrain_v2 | 3131878 | `02a01661ad86667291195e6b631c5d68c1c0d3a624d3945204f4c9ea425abb93` |
| ki_a4_fulltrain_v2 | 3131887 | `1850c78e2d729c476871c67a3e5bfa13a73934adfc591374f8c70d1f123bc285` |
| ar_a4_marker_fulltrain_v2 | 3131895 | `eb2036b66c3045ac25f2764d2104a6c8c3cbd4f35052857eb80385dc145223c2` |

**21:44最新状态（覆盖以下旧进程快照）：** 双路线根`/mnt/sdc1/robodojo/behavior_dev/dual_track_fm_ar_20260913`内，`ar_native_task_fulltrain_v1/formal/checkpoints/step_500.pt`已完整验收（SHA `639e64aeeb251113b807751f234077595165e65e9dd9e3b66cd7c4661f9df963`），2222863/2724340退出。`ar_schema_reference_a4_v1`同十窗完成（result `6fdd96d8…`），`ar_native_observation_gate_v3`十状态全通过（`6e970017…`），3077705/3126741退出。`git_worktrees/ar_native_actor_20260913`固定0406b53/239 CPU passed，尚未启动神经服务或仿真。

`fm_beta_stratified_v1`单GPU gate/result.json已通过（`d1c249a3…`），四卡smoke.log记录CUDA不可见的启动错误/0更新；其后exec_weight2/FM control/joint/KI/marker五个v1因前驱未验收依次failed。2289674/2291489/2294898/2296750/2297891/2994307均已退出，所有v1证据保留，不能再当作运行队列。拟修环境后另建对应v2，未覆盖旧run/源/服务；实际新位置随后登记。最近磁盘余量约602.8GB，未清理任何文件。

**21:20诊断续接（覆盖下面20:56启动快照）：** `ar_schema_ar500_v2`已10/10完成，2994300退出，result SHA `1c9ad37c043182d71db1fd0ad91f855376eb09daefa4342ec523bab331d9bb9d`，0训练/仿真。新`ar_schema_reference_a4_v1`3077705使用固定f7c0c51的`git_worktrees/ar_schema_reference_20260913`，同十窗原A4-FM/codec参考运行；新`ar_native_observation_gate_v2`3095966使用a343a61的`git_worktrees/ar_native_observations_v2_20260913`，229 CPU通过后真实十train观察门运行。前次`ar_native_observation_gate_v1`3051919在相机别名检查错误处退出，日志保留、0VLM/更新/仿真。新run均在下述同一双路线根，同级`.launch.log`；所有活跃工作树不可热pull。

**20:56新AR诊断/候选位置：** 实验根仍为`/mnt/sdc1/robodojo/behavior_dev/dual_track_fm_ar_20260913`。新活跃源码`/mnt/sdc1/robodojo/behavior_dev/git_worktrees/ar_marker_queue_20260913`固定73e2914，200 CPU passed；同时被schema诊断与等待进程使用，禁止热pull。

- `ar_schema_ar500_v2`：2994300于20:54:34启动，同级`.launch.log`；原AR500的10次静态格式约束生成、0更新/仿真，结果待验。`ar_schema_ar500_v1`首条61 tokens保留在`train_task0_raw.json`（SHA `50d01cf0b897288d2a41cdc8e225fd788667dd06f284e2bb61b7186ced2319e1`），随后评估CPU/CUDA相减失败，2910095已退出，不是成功完成十窗口。
- `ar_a4_marker_fulltrain_v1`：supervisor2994307已核验真实等待KI2297891，0GPU/0更新；`method_spec.json` SHA `c14c484b88c40e99ffb68bc570b656dfa1bcc5b6a50ff02decc75506a4212c21`，`launch.json`/`status.json`记录依赖身份与阶段。预定原A4＋零初始化8行共享delta、原切分5保存门→独立500；尚无新checkpoint，不能拿20步缓存adapter冒充它。

以下相对目录均在 **W = `/mnt/sdc1/robodojo/behavior_dev/direct_execution_L6rqZ6_20260910`** 下。

| 用途 | W下目录/文件 | 说明 |
| --- | --- | --- |
| A3最终低层权重 | `formal_a3_episodecoverage_5000_v1/checkpoints/step_5000.pt` | SHA `865193f1c8a257d72ea159bd7a46cebf945b0fd24905110b831f0e8a3438e940`；AE＋VLM LoRA，六帧，FM |
| A3训练原源码 | `a2_lora_history_candidate_v7_samplercoverage` | 名字含A2但实际是A3训练来源；以run receipt/source hash为准 |
| A3推理神经源码 | `a3_history_serving_candidate_v1` | 与训练神经实现核验；不要直接用主仓旧脚本加载代替 |
| A3推理配置/快照 | `a3_serving_composition_v2` | `serving_config.yaml`、`native_snapshot.json`、`source_receipt.json`共同固定来源 |
| B-final高层run | `formal_b_parent_format_1500_v1` | 本阶段1500更新，累计B谱系5000；不能把目录1500读成全部只训练1500 |
| B-final高层checkpoint | `formal_b_parent_format_1500_v1/checkpoints/step_1500.pt` | SHA `d4580d80cdc91a707c233a6c1e625f8fbdeca8896dd38a3d570c6fad340184ef`；UNKNOWN_ONLY，未训物理反馈头 |
| B-final当前serving源码 | `b_parent_format_serving_source_v2` | 实际高层服务源码；和B-final权重/父模型配置一起记录，不凭主仓默认导入替代 |
| 当前修正版完整runtime | `native_a3_aligned_full_runtime_v2` | 修补动作padding mask和执行起点0；不改权重、归一化及物理控制 |
| 当前修正版局部runtime | `native_a3_aligned_prefix_runtime_v2` | 仅用于有明确原演示前缀/固定技能的L1诊断，不能当完整SR |
| 完整runtime修复清单 | `a3_aligned_runtimes_v2_preparation.json` | 绑定两个runtime及动作对齐证据 |
| 旧A3完整runtime | `native_a3_history_runtime_v1` | 保留重现旧结果；含已定位的推理偏差，不作为新实验默认 |

**重要差距：复制`GalaxeaVLA`主工作区到GitHub不等于已将以上最新实验实现整合进去。** 后续按[任务P0-02](TEAM_PLAN.md)做逐模块diff/测试/合并；不要以复制完成冒称代码与A3运行快照一致。

## 3. 结果、视频和原始证据去哪找

### 2026-09-13新增：FM方法与AR重新验证

- **20:38 schema只读GPU：** `dual_track_fm_ar_20260913/ar_schema_ar500_v1` PID2910095，20:34:30启动，固定220312c的`git_worktrees/ar_schema_generation_v2_20260913`（191 CPU passed）。五train＋五原heldout、最多10次AR生成/0更新/仿真，不加载marker20，不热改源；结果尚待验。b8c638d旧worktree只做过CPU测试，没有启动GPU。

- **20:23 marker20完成：** `ar_marker_rows20_v1/result.json` SHA `860e06515a99a48aa7f6c8f243a04990bf3455d39ebe1f9ae62d20e0903e8d31`，20/193 Adam/恢复通过，五task仍漏body。其`trainable_state.pt` SHA `2f1d0fdd7ddebc5b3c03fa74f13fb70f06e62a0fa101db2d52694e0e00580805`只含实验adapter与优化器，依赖AR500父权重，不作为独立发布策略。

- **20:17 marker接续：** `ar_marker_control20_v1`已完成20/回载passed、2818602退出，result `a2386e90…`；同源8ff16c1的`ar_marker_rows20_v1`于20:16:32启动2835113，最多20更新/16384新增参数。两者都只是十train缓存诊断，原始日志和小体积adapter保留原位，不部署或自动追加。

- **20:13 marker短对照：** `git_worktrees/ar_marker_learning_20260913`固定8ff16c1/183 CPU passed；`dual_track_fm_ar_20260913/ar_marker_control20_v1` PID2818602，20:12:46真实启动、GPU1/最多20更新，日志为同级`.launch.log`。markers臂还未启动；活跃worktree不热pull，小体积adapter实验文件不等于可独立部署checkpoint。

- **19:59最新阶段：** `dual_track_fm_ar_20260913/ar_marker_embedding_audit_v1/result.json` CPU完成，SHA `88c4a83e5f11e5df8f967966de1aaa92db2be1ecec1d33b3948c3d7f49778edb`，源码`git_worktrees/ar_marker_audit_20260913`/33d739c，0更新/仿真。此前AR500探针2715623已退出，result SHA `c17363d9…`，不再活跃。原生task smoke通过（`8ef61d98…`），formal2724340已由2222863接续；固定6e2587b不可热改，后继五臂仍等待。

- **19:39 AR500只读探针运行：** `dual_track_fm_ar_20260913/ar_decode_consistency_ar500_v1` PID2715623，09ff64a的`git_worktrees/ar_cot_metrics_20260913`；已完成500权重SHA `51bacc1d…`，两原train同历史/分项CE/独立自由生成，0更新/仿真，结果待验。该worktree现活跃，不热改。

- **19:37 A4-AR500完成/原生接续：** `ar_a4_fulltrain_v2/formal/checkpoints/step_500.pt` SHA `51bacc1d9ec1f6d19c7e82e93bed60b8a1eb5d5ffbab5337c9b7ba47a84e90aa`，inspection通过，原1940901/2316504已退出；eval500 SHA `20600c174ae6d7ce454604ba3a5f9c9a31bb1dbce39f130c9ccbfb15ecd7cb6a`、CE6.7695，仍缺组。原生`ar_native_task_fulltrain_v1`2222863现smoke2714930、源6e2587b；不要按旧“waiting”记录重启或pull它。

- **19:32 CoT GPU门完成/下一入口：** `ar_native_subtask_cot_gpu_gate_v1/result.json` SHA `54f83aff390626a0dfc009c5825d6a0769a36d5c3772402229d48477a622a92b`，2672389已退出，两临时更新通过但动作0/5完整组，未发布权重。新独立`git_worktrees/ar_cot_metrics_20260913`固定09ff64a/166 CPU passed，含动作/文本分项CE和两段生成回执、待用AR500只读诊断；尚无新训练/服务，不热改任何仍活跃队列源。

- **19:25原生Subtask-CoT：** 工作树`git_worktrees/ar_native_subtask_cot_v2_20260913`固定5aa3eff；`dual_track_fm_ar_20260913/ar_native_subtask_cot_input_gate_v2/result.json`十行complete，SHA `0d89930ff98ed90144e4aa592d2e0d7c8bc8ef6e3c207f47a610b85098141e6c`，v1检查器padding比较失败保留。`ar_native_subtask_cot_gpu_gate_v1` PID2672389、同级`.launch.log`，只两临时更新/五task自由生成；尚无正式CoT权重或仿真。活跃worktree禁止热改。

- **19:11 AR只读数值诊断完成：** `dual_track_fm_ar_20260913/ar_decode_consistency_a4_v1`，2604744已退出/代码039e268，独立worktree `git_worktrees/ar_decode_consistency_20260913`。46 CPU passed；两原train行/122同历史下一token比较，121 argmax一致、MRoPE与类型mask一致、96 LoRA模块三路径均调用，自由仍漏组，0更新/仿真。`result.json` SHA `161649772d3a5b8a519a6f5916b6dd1201d26df1de3bbca2e341e23f4cbea43b`；`.launch.log`在run同级，不以强制历史诊断冒称actor输出。

- **17:57 AR正式接续：** `ar_a4_fulltrain_v2/smoke/checkpoint_inspection.json`已passed（5更新/192 Adam），临时`smoke/checkpoints/step_5.pt` SHA `9f22d74f9886ee31a4bda9598a25de22e6680e90ce2b1176b5b02f4bf9f9a84c`。现supervisor1940901/formal trainer2316504重新从原A4初始化正式500，日志`formal.log`及`formal/`；不是从smoke续训，不能用旧2307504 PID判断当前任务。后继队列仍等待，171898c活跃worktree不变。

- **17:50最新阶段（覆盖下方启动历史）：** `fm_ae_lr2x_v1`于17:45:40完成500及回读；`formal/checkpoints/step_500.pt` SHA `7f1c9acdfe52d3ffd6e98038c46a6d743a07766b1188e20c7b45262048396753`。最终固定80=0.2011440，见[LR筛选报告](experiments/2026-09-13-fm-lr-screen.md)。`ar_a4_fulltrain_v2`已启动四卡smoke，supervisor1940901/trainer2307504；原生与其后五臂仍等待，不能将下文“0 AR更新等待”的旧快照当作最新阶段。

**17:44后续筛选队列（均已核验真实等待、0GPU/0更新）：** 根为`/mnt/sdc1/robodojo/behavior_dev/dual_track_fm_ar_20260913`；共用固定84110fa的`/mnt/sdc1/robodojo/behavior_dev/git_worktrees/method_screen_queue_20260913`，不可热pull。每臂`launch.json`/`method_spec.json`/`status.json`区分提交、等待与实际smoke/formal；checkpoint只有训练并验收后才存在。

| run子目录 | supervisor PID | 等待的前驱PID | method_spec SHA256 |
| --- | --- | --- | --- |
| `fm_beta_stratified_v1` | 2289674 | 2222863（native-task-AR） | `191a5393dfd57b8e6749bc93e1487328a6d5fed57fad26a76903f8203adde866` |
| `fm_exec_weight2_v1` | 2291489 | 2289674 | `a6b18a6c325b69b2609720ac95bd0b1ef6624eaf98ed4585faa2a80803cb4231` |
| `fm_action_control_v1` | 2294898 | 2291489 | `e16d8670fbf0f70193b670a7fb3d736552ab8d48e06f28b3ca16a6a87cd242eb` |
| `joint_a4_fulltrain_v1` | 2296750 | 2294898 | `2d8595a640def732d4891ed0497654fdcecd58e51be8d31866f80b9229bb6c03` |
| `ki_a4_fulltrain_v1` | 2297891 | 2296750 | `b9a7eec4cabe36688ff7badc6620bde93b16f3c3ba69cb92fe9823f3c4a9ff06` |

- `ar_native_omission_audit_v1/result.json`：只读CPU20目标/20已生成记录审计，SHA `429d8fa2e9d09e63477dd94433b58ca166442c4ca8b07e29ea1f0b456590b09e`。代码861942c的`git_worktrees/ar_native_omission_20260913`已退出，无新策略/仿真或release。

- **17:26接续：** `dual_track_fm_ar_20260913/ar_native_task_fulltrain_v1`已提交，supervisor2222863，spec SHA `2f01682d6912b14cd7c8d4d6371694cd81ea1303756968d3722bda68811c42af`；真实等待A4-AR1940901，0GPU/0训练更新。固定6e2587b的`git_worktrees/ar_native_task_20260913`现在是正式等待/训练活跃源，不得pull。task GPU v2已退出0，result SHA `09d6244d99053c8295fad280267044fabc0b8df8709ee33739a7c8e8efb5577c`；它仅为两临时更新门，没有可部署checkpoint。

- **原生AR新增（17:13）：** 原生base在`/mnt/sdc1/robodojo/checkpoints/G05/g05-base/checkpoints/model_state_dict.pt`，SHA `072211e5b2f5ef036729bae673f3f44da40adbea5c0044af55fe2fb8af654327`，不与A4父权重混用。新`dual_track_fm_ar_20260913/ar_native_input_gate_v1`是CPU20视图；`ar_native_skills_gpu_gate_v2`是已完成的两临时更新（不是正式策略），result SHA `4b527cbabe11f4770a32901f5a2146abe1d4666d83ad4a35a2384bf0fd514d6d`；`ar_native_skills_gpu_gate_v1`保留0更新词表加载失败。`ar_native_task_gpu_gate_v2`为当前task-only门，使用固定6e2587b的`git_worktrees/ar_native_task_20260913`，运行时不可pull。完整native训练尚未启动；原生词表不注册HL_END，A4默认词表不改。
- **16:35最新观测（覆盖下方启动历史）：** `dual_track_fm_ar_20260913/fm_control_v1`已500更新并验收，最终权重`formal/checkpoints/step_500.pt` SHA `def222a6674e6ac92e6ee982c22836b789240f1542c459d5de2111cd646a244e`。`fm_ae_lr2x_v1`是唯一AE LR×2候选，正式1902909/1912524运行；首个`formal/fixed_diagnostic/step_100.json`的原80均值0.1986116943，SHA `dca01b74cc0e9f488274b64b3d39bd02c546cce7cc33668bd14b9843146f1ad6`，不是最终500结果。仍固定下述fb40145 worktree，不能热pull。
- `dual_track_fm_ar_20260913/ar_a4_fulltrain_v2`：AR正式配方的串行任务，supervisor1940901等待上述候选实际完成/验收，当前0 AR更新；`method_spec.json` SHA `27d22490bb4ccaa9319cc37699a7754471dd91ba60d97f6f5995b0881c7cc930`。`status.json`区分waiting/smoke/formal/complete/failed，`smoke/`和`formal/`只有被真实启动后才产生；预定5步保存门→独立500，均用原A4父权重。`ar_a4_fulltrain_v1`是0更新等待API失败证据，不是训练权重。
- AR活跃Git worktree：`/mnt/sdc1/robodojo/behavior_dev/git_worktrees/action_queue_20260913`，固定171898c，等待也属于活跃使用，禁止切换/热改。入口` scripts/experiments/train_action_method_probe.py`；数据/环境仍复用原目录，模型代码仅feature分支，main目前同步文档。
- `dual_track_fm_ar_20260913/ar_reference_metric_gate_v1/result.json`：2次原train同输入参考前向逐位相等、0更新，SHA `7530aa9facead5b5d7b9ac4a7c4fb2f5f6f9e46eae4bdbb8faf72bea94d0d568`；`ar_input_gate_v2`为40视图真实codec检查，`ar_loader_gate_v1`为完整loader输入身份检查，`ar_gpu_gate_v2`/`ki_gpu_gate_v1`/`joint_gpu_gate_v1`为两临时更新工程门。以上不是已训练的AR/KI权重或成功率结果。

- `/mnt/sdc1/robodojo/behavior_dev/dual_track_fm_ar_20260913`：双路线小实验根。`ar_codec_gate_v1`/`v2`为失败证据，`ar_codec_gate_v3/result.json`为十条原train的CPU编码往返结果；不是AR策略或成功率。
- `fm_control_v1`：14:00编排启动，supervisor1499025、固定Git fb40145；14:14已通过GPU与四卡5步保存回读门，正式torchrun1508221开始从A4初始化。`method_spec.json`绑定扩展源码/父A4/预算，`status.json`为阶段，`gate/`与`smoke/`是工程门，`formal/`才是最多500次真实五任务更新。启动不是完成，须读阶段与checkpoint验收。
- Git固定worktree `/mnt/sdc1/robodojo/behavior_dev/git_worktrees/dual_track_fm_ar_20260913`；方法代码在独立feature，main仅同步说明。运行中不得pull，复用只读A3源码和A4旧编排的SHA明确记录；没有复制旧权重/数据，其他服务保留。

### 2026-09-12晚新增：A4过夜低层训练

- 编排/结果根：`/mnt/sdc1/robodojo/behavior_dev/overnight_a4_20260912`；`launch.json`记录准确commit/父权重/源码SHA，`status.json`给出真实阶段，不能只看PID判定成功。
- Git固定配方：`/mnt/sdc1/robodojo/behavior_dev/git_worktrees/a4_overnight_20260912`，commit `e463932740cbb3977a2b975be824c21d9dc96f45`；**不能在活跃副本pull**。它编排上述原A3训练快照，不代表主仓模型已完成P0-02整合。
- `gate/`为两次临时优化验收、不保存训练权重；`smoke/`为四卡5步验收，`smoke_checkpoint_inspection.json`回读验证后才放行正式阶段；`formal/`才是从原A3-5000初始化的新2500更新阶段，正式初始模型不来自smoke。
- 新正式权重：该根的`formal/checkpoints/step_N.pt`，每500更新及最终保存；固定80窗口结果在`formal/fixed_diagnostic/step_N.json`。2026-09-13 07:39完成，`step_2500.pt` SHA `6186704788c27c9fae3502c884df0e259de5242ee8690fe578dcbc1f2632f269`；`formal_checkpoint_inspection.json`通过完整模型/Adam/RNG/冻结参数验收，尚无A4成功率结论。
- 正式训练无墙钟截止（用户随后明确取消8小时限制）；有限2500步、不自动重试/追加训练。磁盘保留120GiB安全余量，不清理旧checkpoint；异常退出保留已有完整保存。恢复需绑定同run身份及原配置，不能重新执行`start`冒充断点恢复。

### 原有A/B评测和诊断

2026-09-13新结果根（不在W内）：

- `/mnt/sdc1/robodojo/behavior_dev/a4_radio_full_20260913_v1`：A4＋B-final自主完整收音机评测已完成，public_test301/302/303各3224控制，成功0/3；不是有前缀的L1。Git worktree `/mnt/sdc1/robodojo/behavior_dev/git_worktrees/a4_radio_full_20260913`实际固定be23b06；编排1479480、专用服务8783/8784和三个仿真均已退出，其他服务保留。`manifest.json`固定条件，`instance_301/`等各保留结果/视频，`summary.json`完整写出。三段视频及本人复核抽帧/小摘要已在本地`/home/wsy/behavior/artifacts/a4_radio_full_20260913`；完整结论见[评测报告](experiments/2026-09-13-a4-radio-full-eval.md)，不要把这些唯一证据当cache删除。

- `/mnt/sdc1/robodojo/behavior_dev/a4_paired_actions_20260913_v1`：170次A3/A4原train同输入FM对照已complete，`result.json`为摘要，`A3/`和`A4/`保留预测/完整恢复证据；没有physics或SR。
- `/mnt/sdc1/robodojo/behavior_dev/a4_radio_e121_l1_20260913_v1`：A4局部GRASP已完成，448原前缀＋464模型控制后满足指定对象稳定抓取；`actual_analysis.json`和[本轮报告](experiments/2026-09-13-a4-training-effectiveness.md)记录审核/限制，不是完整SR。`service/`为已退出的GPU0/8782临时服务证据，`actual_rollout/`为真实控制/物理trace/视频；supervisor已关闭自己创建的服务。Git worktree `/mnt/sdc1/robodojo/behavior_dev/git_worktrees/a4_prefix_20260913`在执行时固定c7fb287，确认全部进程退出后才更新至a3ce491运行只读后处理；不热pull。
- 本地A4视频及抽帧/小摘要：`/home/wsy/behavior/artifacts/a4_effectiveness_20260913`。视频源/本地SHA一致；不入Git，不视为可随意删的cache。

| 结果类别 | W下路径 | 结论/边界 |
| --- | --- | --- |
| 旧A3完整五任务 | `native_a3_final_development_pilot_v1/task_0` … `task_4` | 各有`result.json`、`rollout.mp4`与记录；完整0/5，重复开发实例 |
| 修正版完整五任务 | `native_a3_aligned_development_pilot_v3` | `task_N/`放结果/视频，序列receipt记录完成情况；2026-09-13核对task0–4均完整结束，官方0/5，属于A3+B-final而非A4 |
| 修正版完整评测清单 | `native_a3_aligned_five_task_development_v3.json` | 同A3/B、public_test301、env0/policy17；SHA `2722f9404b046ff69b443774d65fe1cb7b0bf4c25369d67b988cb7889924d730` |
| 旧A3固定GRASP | `a3_prefix_radio_e121_l1_v1/actual_rollout` | 448原前缀＋1280模型控制，无稳定抓取 |
| 修正版固定GRASP | `a3_aligned_radio_e121_l1_v2/actual_rollout` | 同权重/起点/条件仍未稳定抓取；`collection_result.json`不是完整任务SR |
| 修正版固定GRASP审计 | `a3_aligned_radio_e121_l1_v2/actual_analysis_v2.json` | 80段实际历史/动作/物理记录核对；80 IN_PROGRESS |
| 504次推理对照 | `a3_input_route_factorial_v2` | `result.json`及`execution_alignment_analysis_v1.json`；不再把mask/时间混杂误报为纯视觉域差异 |
| 42原训练锚点 | `a3_radio_e121_matched_original_inputs_v2` | 原radio121/138输入与目标逐值校验；大CPU缓存不进Git |
| 原radio真实回放 | `demo_radio_e121_alignment_v2_persist` | 真实原动作、观察、视频，非模型成功 |
| 第二原radio回放 | `demo_radio_e91_feedback_trace_v5` | 已核实精确目录名；原始成功演示回放/反馈，不是模型自主成功 |
| 原plates部件核验回放 | `demo_plates_e762_verified_parts_v4` | 精确部件映射/物理反馈；开门原谓词成立不等于交接就绪 |
| 原can-meat反馈回放 | `demo_canmeat_e807_feedback_trace_v5` | 11787原控制；包含原演示放置失误，不能自动准入全部成功标签 |
| 同状态纠正试验 | `c2_same_live_state_pilot_v1` | 真实接管实验及失败证据，不是已认证成功教师 |

查一个run先看`coordination_run_receipt.json`或`result.json`/序列状态，再看manifest、实际配置和checkpoint/source SHA；不能只看文件名、PID存在或step5000文件存在。

## 4. 活跃服务、端口和模拟器缓存

这是2026-09-12的状态快照，不是允许按PID直接kill的清单；操作前重新核对实际argv/用户/启动时间，防止PID复用。

23:54:39更新：为A4训练补足显存，仅在核验旧诊断已完成、完整启动身份和无连接后，以pidfd关闭旧A3前缀服务`3276541 / GPU0 / 8778`。该服务不再监听，旧权重、代码及`W/a3_prefix_radio_e121_l1_v1`结果全部保留；复现旧诊断时按其`service.launch.json`在新输出目录另行启动，不覆盖原回执。当前评测使用8773/8781，保持运行。下表的其他旧服务不因此自动获准停止。

| 用途 | 端口/GPU | W下证据目录 |
| --- | --- | --- |
| 修正版A3完整低层 | `127.0.0.1:8781` / GPU3 | `a3_aligned_full_low_service_v3`，外层同名`.log`/`.launch.json` |
| 修正版A3前缀诊断低层 | `127.0.0.1:8780` / GPU3 | `a3_aligned_radio_e121_l1_v2/service`及service日志 |
| B-final高层 | `127.0.0.1:8773` / GPU2 | `a2_final_eval_b_high_service_v1` |
| 原A2/旧A3等保留服务 | `8772/8776/8777` / GPU0或2 | 历史比较服务，是否可停由准确依赖及团队决定；8778已按上述核验关闭 |
| 上轮完整模拟器（已结束） | GPU1 | `native_a3_aligned_development_pilot_v3`；使用`kit_c1_gpu1_appdata_v1`私有缓存，2026-09-13核对campaign已退出 |

端口不能直接从外部访问时用SSH转发，不修改服务绑定扩大暴露。四张A100不等于四份可随意分配的空闲资源；先查实际进程和显存。渲染质量、IsaacSim版本与硬件兼容性仍需独立验证，吞吐不与正确性混为一谈。

## 5. 本地与GitHub结构

- 本地协作根：`/home/wsy/behavior`。
- GitHub：`https://github.com/dywsy21/behavior.git`；只同步代码、配置、tests和小文档。
- goal总计划与实时记录：`docs/plan.md`；三人任务板：`docs/TEAM_PLAN.md`；通用RL计划：`docs/RL_METHOD_PLAN.md`；开工先pull及实时更新规则：`AGENTS.md`。
- 根目录现保留`src/`、`configs/`、`tests/`、`scripts/`、`experiments/`、`tools/`、`assets/`、`skills/`、`licenses/`和项目元文件。
- 旧本地顶层Markdown报告在`/home/wsy/behavior/docs/archive/`；其中原goal计划已按用户要求移至`/home/wsy/behavior/docs/plan.md`作为实时维护入口。远端MAIN的7份未跟踪历史文档在`docs/archive/remote-main-20260912/`，继续作为历史记录。
- 147项旧本地实验、视频、附件、补丁和快照已可逆归档到`/home/wsy/behavior/artifacts/local-archive-20260912/root/`，约4.1GiB；`memlite-resume.L6rqZ6`也在此。没有删除，不把未完成研究草稿全当生产源码导入。
- 修正版radio视频本地位置：`/home/wsy/behavior/artifacts/local-archive-20260912/root/memlite-results-20260912/A3-aligned-radio-e121/`；其余历史文件按原目录名在同一归档根查找。
- 上游中文/英文文档入口保留在`docs/upstream/`，架构/数据/部署文档仍在`docs/architecture/`、`docs/data/`、`docs/deployment/`。
- 首次迁移commit为`69645b4220105ef1199fbbe90c889d4ae911ef6a`；2026-09-12的后续核验中，本地、GitHub和服务器新clone同为`c3367995234ce51964fff736f2544610c0320419`，两端工作树干净。该迁移检查点有474个跟踪文件，AGENTS及协作文档均跟踪；没有大于1MiB的跟踪文件。后续归档验收文档另行正常提交。
- 原源码身份、归档映射、扫描/验证及服务器安全checkout入口，以[仓库迁移记录](REPOSITORY_SYNC.md)为准。
- 新clone的`.venv`只是指向旧主仓环境的软链，已有editable安装可能仍导入旧`GalaxeaVLA/src`。本次未安装或切换服务；未来用独立环境或显式并核验`PYTHONPATH`/实际导入路径，不能仅看当前工作目录判断源码版本。禁止对活跃共享环境重新`pip install -e`改指向。

## 6. 本次安全归档结果

**用户已确认改为“先释放确认安全的部分，剩余暂不动”。最终仅迁移旧`hy_vla`实验，净释放516,587,864,064字节，即481.11GiB / 0.469834TiB / 0.516588TB（十进制）。** 不将最初两个候选的合计大小当作释放量。

| 项目 | 最终记录 |
| --- | --- |
| 源文件系统 | `/mnt/sdc1`，实际设备`/dev/sda1`，XFS；操作前可用302,917,308,416字节，操作后819,505,172,480字节（约763.22GiB） |
| 目标文件系统 | `/mnt/tmp1`，实际设备`/dev/nvme0n1p1`，XFS；约894GiB总量，操作后可用113,477,279,744字节（约105.68GiB） |
| 原入口 | `/mnt/sdc1/robodojo/experiments`，现为指向下列归档的兼容软链；**不是**主代码仓里的`GalaxeaVLA/experiments/` |
| 完整归档 | `/mnt/tmp1/robodojo-archive-20260912/experiments`；顶层为`hy_vla`，run最新时间2026-07-27 |
| 归档内容 | 353个普通文件、126个目录；逻辑大小516,607,863,251字节，源/目标核验时实际占用均516,608,692,224字节 |
| 内容验证 | 复制完成后独立`rsync --checksum --dry-run --itemize-changes --stats`，退出码0、变化数0，元数据匹配；这是完整内容比较，不冒称另算了每个大权重的SHA256 |
| 真正移除的副本 | 仅同盘隔离目录`/mnt/sdc1/robodojo/experiments.archived-source-20260912`；归档验证、原位软链/可读性、依赖检查通过后移除。归档中的完整文件保留，可恢复 |
| 未移动的范围 | HF源及本次形成的HF目标副本、`behavior_dev`、模型、数据、原`GalaxeaVLA`、xhz目录和现存服务；未为凑2T连带清理 |
| 操作后运行状态 | 7个原有服务监听与既有campaign保持，task2进程3632286仍在运行；它及排队task3/4的实际依赖均不在旧experiments目录 |

原始清单和校验/操作/恢复记录都在`/mnt/tmp1/robodojo-archive-20260912/manifests/`，最终记录为`experiments_migration_final_receipt_20260912.txt`。轻量、去除无关进程信息的协作副本见[归档验收材料](storage/README.md)。

HF源`/mnt/sdc1/robodojo/.cache/huggingface`原位保留。先前HF占用来自本次复制进程，不是已证实的生产占用；但其锁文件有当日活动，且未完成独立内容验收，因此未纳入迁移。`/mnt/tmp1/robodojo-archive-20260912/cache_huggingface`只是本次形成的未验收副本，不算新增已认证备份，也不算源盘清理成绩；本轮按用户决定不再动它。

恢复时先确认源盘空间充足及没有相关使用，在源盘新建**真实临时目录**，从归档复制、校验后再将原入口从软链切回真实目录，并保留归档/回滚入口。不要直接向当前软链入口rsync，否则会沿软链回到归档自身。详细边界见上述恢复记录。

`/mnt/tmp1`不自动意味着永久可靠的备份；冷归档的保存期限和磁盘健康仍要管理。未来若再清空间，重新按清单核验，不把此次放行扩大到其他目录。
