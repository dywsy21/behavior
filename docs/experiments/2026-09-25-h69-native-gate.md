# H69：空闲卡上的同源完整场景基础gate

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
