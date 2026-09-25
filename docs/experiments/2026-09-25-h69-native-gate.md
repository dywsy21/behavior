# H69：空闲卡上的同源完整场景基础gate

## 真实终态与归档（2026-09-25 14:33北京时间，Codex）

唯一5e4ce75运行failed，519.4327845s。worker退出0但没有result，监管正确拒绝通过；根异常为初始化后`Registered PathTracing/OptiX settings changed`，尚缺报错时具体actual，不能断言改写字段或触发者。原官方reset1和load138各1且completed（证据`gate/worker.json`），最终外层reset0、gate/control/VLM/训练0，不是任务失败样本或press验收。

完整10件4441121B已从原run取回`artifacts/agentic-vlm-goal-20260918/h69_native_gate_bundle_v1`并逐件双端SHA一致。214次资源样本：worker主GPU3峰5247MiB、辅助0/1/2峰456/416/416MiB，未触显存门；退出后三卡空、GPU0队友3564916/12546MiB仍在。不能把原较早native_profile快照中的0reset当终态；不重启H69。

| 相对文件 | SHA256 |
| --- | --- |
| launch.json | 0bd58c0cad051036c232ce9697ad26467672e9b5cec25b6f6499123a310c5bdc |
| supervisor.log | 8721c4f35052b8f21247f6c0f31ffe82777c0e158311bf4234f0fa53647fcc4a |
| supervisor.claim.json | eec7d964b1b7897930ecdbc6ae02081e493a6734cdf406d3e750de2f11ad43da |
| worker.log | a74e63c514f87b30db0b040958309137a2d6f84834b33d9a451d3b48afbee300 |
| supervisor.json | 3a3bee89eeb1d3eddf2af00f3814808496afcc97c3122fc882b7952077c64ee6 |
| gate/manifest.json | 17c61ee037df89fa73ffd9feaaa5e8e1b9fe8a157438578568c018bd96556338 |
| gate/steps.jsonl | e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855 |
| gate/worker.json | 172fe348547cf48fe7ff98a884b6c7cf6ff0488c44585c46319680e6b4bce6ed |
| gate/native_profile.json | b87ae4ee51e6985a15543ffe0924b1c8ded23be59981d4eb2b1ca02b257dd68c |
| gate/kit.log | 2fcdfe7ee595651bdc549045a0db34130969a7ad1d4fc374d4ffcdb92c10deee |

以下为原始预登记/执行史，不覆盖上方终态。

2026-09-25 13:46北京时间预登记，owner Codex；基线7a1b203（H68），运行源码待实现/独审后固定。四卡原训练已自然退出，不是获准停止它们；GPU0/1仍留团队。

唯一假设：解除共享显存限制后，已修正的真实手指/本体排除/动作内RGB-D反馈能在完整原场景下通过现有基础动作gate。它不验证VLM语义、press接触或任务SR；不把gate固定动作当agent策略。

- 原task0 `turning_on_radio` TRAIN138/seed0，原window/机器人SHA及reset→load138→reset序列；专家/旧policy前缀均0。
- 一次session/场景实例，现有24个基础动作、≤1536实际控制步、动作阶段≤1200秒，外部总墙钟≤2400秒（包括冷初始化，另至多30秒拥有进程清理）。失败停止，不自动重试/切实例。
- GPU3主显存≤24576MiB、启动空闲≥32768MiB/运行余量≥8192MiB；其他卡仅本进程辅助上下文≤512MiB，不占用其训练计算。每2秒检查compute+graphics，未知新进程停止本次子进程，不给他人发信号。CPU72–75，0模型请求/训练。
- 采用已独审的进程私有PathTracing/OptiX 4spp/16total、无viewer、三路原720/480 RGB-D在创建前配置。图像分布与原RT不同，必须记录并人工检查；不降低相机尺寸/改物理。不能用旧renderer的gate给新profile放行。
- 显式源依赖摘要绑定profile及其复用模块；新源、新run与cache，不热改共享SDK、旧运行源或环境。资产沿H64 SHA `5f2cbf18b989eb22bc98665f48538c6cc306679d3be94ab4de46f2af7b4c6f04`。
- 对照是现有gate的测量门而非旧模型SR。通过须真实24动作、原FK/运动/独立finger检查和gate_ok，父人工看初始三视角及录像；退出0/目录存在不当通过。
- H68 flag虽随同源记录，gate不构造VLM/press goal，因此即使通过也不批准真实press策略。后续press/模型票另写预算。

13:52启动前资源约束细化：05:49:21 UTC GPU0出现队友3561374/12462MiB，GPU1/2/3仍空；尚0新launch。0/1原本归团队，预检改为登记其既有PID/显存baseline，GPU2/3仍须无既有进程。每卡运行保留8GiB；我们的辅助每卡≤512MiB，连同总显存相对baseline增量也≤512MiB。未知新进程/余量越界停自己，既有队友可以自然退出，不阻止/不停止它们；不以别人的既有12GiB拒绝自己的400MiB辅助上下文。该细化在新实验启动前固定，主卡/时间/控制预算不变。

预计run `/mnt/nvme_tmp/robodojo_agentic_20260925/h69_native_gate_v1`；runtime `/mnt/nvme_tmp/robodojo_sim_runtime_20260925/h69_native_gate_v1`，固定源码worktree待commit。当前尚未创建或启动。

13:49实现：`native_full_profile.py`仅显式profile启用时包装单个官方session；复用H56安装身份/私有startup、三相机创建前配置、PT与reset后检查。runner摘要新增全部复用profile文件，结果/gate比较renderer profile，原默认保持。单次launcher不复用旧四训练PID或4GiB共享门，unknown process/墙钟/显存停自己的Popen child。9新用例与83旧邻接共92 CPU通过0.970s；完整semantic及独审待，未部署。

13:57审查增补：独审发现退出码0可漏末次wall、只清child可能留Kit后代、启动器/全部配置未完整绑定，均已修并加反例。启动器现纳入digest；manifest完整参数/source/实例/零训练模型和result全部开关严格比对。清理只对自己新建session的PGID，leader刚退出也继续清同组，12+12s＋最后采样≤30s。原session现在实际记录reset→load(window.instance_id)→reset及完成状态，不用源码推断事件。100邻接CPU过1.249s，最后竞态delta/全量/独审待；0物理启动。

13:58最终回归：含cleanup竞态的100邻接过1.311s；修后semantic703中698过/5本地SDK skip（28.959s）。新增直接执行runner实际argparse并与manifest全参数对比后，14个本票目标用例过0.066s，diff clean。原703初失败是测试闭包沿用旧session注入名，已修测试到实际调用，不更改异常/安全停规则。独审最终结论待，仍0launch。

14:01独审闭合：最后补helper实际来源绑定（在ADAPTER插入前CPU预载，use时核spec.origin和__file__），同名旧模块反例拒绝。102邻接过1.318s、703全量中698过/5SDK skip（30.638s）；独立59过、另真实无GPU孤儿进程组清理过，无剩余实质代码阻塞。下一固定源/robo预检/唯一启动；仍不是原生、press或SR验收。

14:05启动前归属更正（覆盖13:52中对0/1总增量和新PID的约束）：d45c487新worktree/robo102 CPU及23实依赖已过，但仍0launch；GPU0队友任务已自然轮换成3563718、GPU2/3仍空。0/1不是本票分配卡，允许外部PID自然启停/增长，仍只限制我们的准确PID每卡≤512MiB并保留全卡8GiB余量，不把队友增长算到我们头上。GPU2/3仍启动无外部PID，运行未知PID停止自己，原增量主24GiB/辅助512保持。没有新增本进程额度、时间/样本或改变其他任何门；小delta独审/新固定源后才启动。旧d45c487仅预检，不曾创建场景。

14:08明确“我们”的归属是Popen创建的隔离SID及其所有PID，而不只是leader：按每卡合计受主24GiB/辅助512MiB限制，退出后带原SID拒残留。清理枚举这个SID的所有PGID，处理leader已退出和新子组；不对外部SID发信号。独审反例已补，104邻接CPU过1.290s，最终delta独审待；0launch。

14:10最终delta闭合：空session不发旧数字PGID信号的反例加入后，105邻接过1.294s，独立18目标及diff过、无剩余实质阻塞。新运行源待固定；原d45c487仅CPU预检，0物理启动，预算不追加。

14:14真实运行：源码`5e4ce75623c9b658977a14ad0c6dfb5f6ebe9abb`、双端105（robo2.496s）/23依赖过；唯一launch UTC06:11:57.224739，supervisor3564979/worker3564992。77.45s监管running/原生loading_scene，1 app/原startup、PT before-scene、GPU3实际配置及720/480相机配置已过，无failure；还没有完成reset/真实gate动作或RGB-D验收。主自有852MiB、辅助240/200/200，GPU0队友3564916/12462MiB共存。路径/原预算保持。
