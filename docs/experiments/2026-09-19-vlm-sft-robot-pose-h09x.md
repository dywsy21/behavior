# H09X：当前机器人姿态协议与窄范围GRASP实验准备

负责人Astra，2026-09-19 22:01–22:31 BJT，≤1800s CPU，0新reset/控制/模型权重加载/模型forward/训练。父独占harness/GPU调度与独立审查。唯一假设：原输入遗漏关节及EEF朝向，使3°旋转标签的当前机器人状态不完整；补合法robot-only姿态，再用多实例完整native GRASP轨迹验证小规模SFT。**RGB可能已含朝向线索，当前没有证据证明其充分或不充分，不能把输入缺字段直接当旧训练失败的因果解释。**

## 已实现并验证的协议

稳定代码`564d08c2237c9d6a1568940c756e43239fc7a154`，版本`h09x-current-robot-pose-v1`。新`native_actor_protocol.py`从当前18D q（4 torso＋7 left＋7 right）和固定robot-only calibration计算双EEF在机体系的旋转矩阵，同时保留旧位置/指口/本体速度/torso输入。矩阵列是局部EEF轴在base坐标的方向，动作仍是原41符号的base轴1cm/3°，没有修改执行器/目标/成功门。

离线只读当前capture：绑定独立记录的capture SHA、精确prefix/native时钟、canonical calibration SHA和七payload SHA；核当前q/grip生成的旧proprio一致。当前q只以6位小数进入actor，矩阵6位；有限/形状/真数值/正交/det+1/齐次检查拒绝畸形值。未读取private teacher ledger或对象位姿。时钟、标定SHA、实例、未来、seed/目标位姿、success都不进入模型文本；图像hash仅用于传输校验。新请求传当前原始PNG字节并核SHA，服务端统一缩到256²，基础和adapter完全同协议。

`modeling.messages`只对显式版本使用新SYSTEM且要求text等于白名单actor重新生成的文本；默认旧协议行为不变。`training_row`本身不是数据release/正确性证书。旧live/collector、239cb591执行器与所有正在运行的源均未改。

本地95 SFT/2.051s、旧331 harness/5.968s过；robo固定564的95/2.663s过。新增8组覆盖3°自转而旧位置相同、隐私字段、NaN/反射/缩放、错时钟/标定/陈旧位置、真实PNG绑定、base/FT/train同prefix、真实CPU tensor EOS和混长assistant-only mask、实例分组、source action index时钟。真实H09W全部10 before capture转换1.238674s，base/FT/train前缀逐字相同；当前q/FK不是事后对象位置。

实际2B AutoProcessor CPU门15.355442s（没有加载权重或执行forward）：10个前缀1059–1089 tokens，原1800prefix/2048context上限足够，EOS248046、目标3或5tokens、混长padding和所有prompt/image位置不监督均过。证据在原NVMe H09W根及本地`artifacts/h09w-native-task1-v1/complete/`：

- `h09x_real_protocol_v1.json` SHA `ee5cf1e24bce0981cec019e0b179ee34b259faf83b91064c2cb1abe4073e88ef`。
- `h09x_real_token_gate_v1.json` SHA `b68e9512fcd1235a803ad84e21675aa5c70d121d67be705c110379913cbd1e66`。

父固定564独立完整审查通过：95/2.124s；另真实10capture及50个隐私/旧时钟负例0.759s通过。**后继接线仍须**：dataset builder逐行核`row.images`实际字节与actor原RGB SHA（通用`load_images`不自动做此事）；新服务身份强制新protocol，禁止漏字段回落旧SYSTEM。此次没有启动/接通新服务，也没有生成可训练发布清单。

## 四个真实来源，不把注释段尾当成功

`native_grasp_sources.py`在看动作/标签之前固定分组：已反复开发192只能TRAIN；其余H09R additional_train按`sha256(h09x-grasp-instance-split-v1:task:instance)`排序，前两留出、第三训练。全实例排除原逐task5%、旧H09 val/test和开发138/242；两留出源虽属于原官方TRAIN池，但本实验永久不进训练。原数组/全部高低层重复标签行完整SHA再核，低层分支冲突拒绝。没有随机拆帧/把同源近邻分进两组。

| 角色 | task / episode / instance | 完整GRASP段[start,end) | 原动作首次CLOSE（1基控制） | 近抓取精确prefix长度 | 手 | 成功pose seed |
|---|---|---|---|---|---|---|
| TRAIN，已开发 | 1 / 310 / 192 | 164–452 | 397 | 396 | right | P3已父审，W完整native已父审 |
| TRAIN，新 | 1 / 264 / 114 | 596–1186 | 994 | 993 | right | 无，需一次真实完整参考回放 |
| HELDOUT，新 | 1 / 200 / 1 | 642–981 | 833 | 832 | left | 无；VLM评测不应读取pose seed |
| HELDOUT，新 | 1 / 247 / 71 | 408–1357 | 1039 | 1038 | right | 无；VLM评测不应读取pose seed |

四源均有同一GRASP注释、可审原23D动作和先双开再单手CLOSE；目标原名都是trash_can_116（来源身份，仅教师/审计，actor用trash can）。这是**来源可用性，不是停顿后可抓、原段真实成功或当前native正标签**。原段/前缀SHA和完整选中标签索引在`h09x_source_feasibility_v2.json`，SHA `5cb0ed5a5c4ca48e5148dbeab4406c373242bcd3764c4ebb5a8d9864c7934278`，1.568273s、0物理。192新计算prefix SHA精确等于已执行f92baf54…1447。

首计数v1把0基首次CLOSE index再减1，得到395/992/831/1037；只读候选错误已保留（SHA03a2054e…549e），564明确修为prefix长度=首次CLOSE action index，1基控制再+1并加fixture。**没有按错误候选reset或重写已有396前缀。**

额外左手TRAIN只读扫描3秒、8个未占用additional_train完整来源，均排除四个已分配组和原排除集合；没有找到满足本先导“首完整GRASP前双开、单手CLOSE”的左手来源。`h09x_optional_left_source_v1.json` SHA `7ec4ec23ef640d48e2317766632a84ae5da6ea434fcec63c56e8922f35a2e7d1`保留扫描项、分支数、完整来源SHA和拒绝原因。**不继续无界找左手，不移动heldout1；将heldout71作为右手同分布测试，heldout1单列跨手OOD，不因左手缺训练样本永久阻断这次窄实验。**

## 最小后继路线与前瞻上限（本票不授权执行）

结论范围是“给定合法GRASP意图、付费专家近抓取前缀之后的停顿纠正”，不是phase-start、原任务起点、完整SR、50任务泛化或视觉能力认证。H09W已提供1条完整轨迹/10宏，而不是10个独立实例。旧H09负训练、H09V容量失败和所有隔离数据永久保留；失败动作不能正BC。

1. **下一块先只做114的一个成功seed参考。** 由已核原episode264构造通用完整参考：596prefix＋590原动作＋12稳定＋1hold=1199 controls；1reset，reset后≤1200s，初始化另≤900s，run≤80MiB。原192/P3不重做。当前原prepare入口绑定早期三源，须先以已核counts/所有source SHA构造新明确实例准备清单；复用完整参考runner，保持真实oracle/末hold/父图审。若无success或来源/接触/期限异常，停此块，不原样retry、不造seed。
2. **seed过后最多5个新TRAIN原生回合，不一次发无限采集。** 预登记同一通用first-close索引偏移：192再取392、388；114取993、989、985。前三者验证顺序是114/993，然后192/392、114/989；仅剩已登记两不同prefix可补覆盖，不改姿态/物理门。合计3747付费prefix、≤2100新controls（5×420，含所有settle/hold）、≤60macro、5reset；每回合≤12macro/reset后1200s/384MiB，初始化另900s。新1200秒只是拟议新profile，**原900s授权不改、不直接复用**。教师仍同一个PoseTeacher和已审物体局部seed，非逐实例手绘动作。后继可只先放114/993，观察真实产率后决定是否消耗剩余登记量。
3. **窄数据门。** 至少4条完整实际GRASP成功且末hold仍成功的轨迹，两个TRAIN实例各≥2条，含已父审W；≥32个真实macro，至少4次因果CLOSE、8次lift和8个闭爪前纠正，至少4种非HOLD动作，包含真实旋转。每条完整control/capture/oracle和父抽审绑定；同实例不同prefix若停顿后EEF初姿近重复（位置<3mm且角度<0.75°），不充作第二条独立覆盖。近重复及所有恢复/分支仍归原实例。达不到则如实停，不拿距离下降/合法输出补数；旧六类240广覆盖门保留但本窄结论不冒充达到它。
4. **训练一次小2B，不续旧adapter。** 复用原Qwen3.5-2B权重，language-only LoRA r16/alpha32/dropout.05，LR5e−5、有效batch8/microbatch2、seed41，固定120总更新（含最初2步真实loss/梯度/原生CE和reload门）、单GPU≤45min、adapter/小状态≤256MiB；固定终点、不按heldout表现挑checkpoint/反复调参。假设产率近W的10宏/轨迹，6条最多约60条，不足以支持广泛泛化；学习/过拟合都需要报告。
5. **留出同预算物理效果。** 两留出原TRAIN实例的832/1038精确prefix，base、FT和预登记非视觉proprio＋已执行history最近邻各一次，共6reset；每例同12决策/420新controls/reset后1200s/初始化900s，合计5610prefix＋≤2520新controls、最多48个神经动作调用。最近邻只由TRAIN合法本体/历史拟合，不能读目标真值、phase index或留出教师pose；另零模型静态报重复上一动作/仅history规则，防把固定动作序列背诵称视觉收益。左手OOD与右手同分布分开给原始1例/策略计数，不作统计显著性结论。局部oracle只作事后审计，不反馈actor、不因hidden grasp truth提前结束/改变动作；完整任务SR另报，不因本局部任务成功冒称完成捡垃圾。

整个后继最多12个新reset（1参考＋5采集＋6评测），15176个新实际控制上限（1199参考＋3747采集prefix＋2100采集native＋5610评测prefix＋2520评测native，含所有settle/hold），0自动retry；初始化与reset后上限合计≤7h，训练45min，另CPU/传输不冒充GPU时间。新NVMe独立root拟≤6GiB，所有旧根保留另计；5×384MiB采集＋6×384MiB评测＋80MiB参考＋256MiBadapter＋余量仍可容纳，双盘余≥80GiB。这个总额是**分阶段上限建议，不是已授予/必须跑满**。H09W单例828秒含初始化只给吞吐参照，不能保证993前缀同速；给新实例余量不等于降低任何动作/物理质量门。

## 下一块必须补齐的小接线，不掩盖为“命令已可运行”

- 从真实已核新episode输出完整reference/near manifest及明确TRAIN/HELDOUT用途。不能套旧192 identity，不能让heldout只因来自官方TRAIN池就进训练release。VLM两heldout不需要物体局部pose seed，合法固定意图/所选执行手须事前注册；来源标签只定义评测起点，不给actor专家动作。
- 独立数据release同时绑定父整轨迹审核、末物理成功、实际前后记录与协议派生SHA；对训练原图内容逐行核hash，再交processor。旧quarantine原件不改。
- 新服务和训练identity强制H09X版本；旧`train.py`仍硬绑定GPU1/旧VERSION、旧`serve_local.py`输入也未升级，**不能把此刻旧脚本直接作为GPU3新训练命令**。仅需有界新profile/路由、同新数据门和资源授权检查；未授权之前不加载模型。手臂旋转在评测中也须以合法命令历史/校准全开等公开资格和同安全门明确启用，不得把TRAIN teacher的privileged held/contact判断接到部署actor。

父可按已审协议先登记“新来源prepare＋114一次参考”小块，再推进原生数据和真实120步训练及六回合对照；本CPU票没有自行新增物理。当前唯一完整native成功仍是开发TRAIN192的W，未宣称新SFT有效。
