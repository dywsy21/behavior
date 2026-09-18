# H11：初始规划的结构化输出与完整任务覆盖

负责人Codex；分支`feat/semantic-agent-grounded-20260918`；执行源`2000d784f9dab49b56d2833259cbcb2172ba49f6`，digest`ade09283b5683c8dc3451a3cf70e7e48fbc5dc794e70738d56c83f1d882f403d`。状态以[plan.md](../plan.md)为准；预算见[H11登记](../../configs/semantic_robot/h11_structured_planning_block.json)。零训练，不改机器人动作/安全阈值。

## 实际反例

H10唯一无前缀尝试在初始规划失败：模型先输出解释，再输出JSON围栏，未截断却违反严格协议；其中唯一目标还是“导航直到看见收音机”。这有两个问题：数据格式不合法，以及任务计划不完整。直接找花括号取JSON只能掩盖第一个问题，还会接受第二个问题。旧尝试和服务call13保留，零控制不是从失败分母删除它的理由。

## 实现理念

- `--structured-planning`默认关闭。新服务在生成每个token时，按固定JSON schema约束task-plan/recovery；不是生成后猜测/补写答案。两种调用共享旧`kind=plan`但使用不同白名单schema名。
- 复用tokenizer索引，不复用跨请求parser/enforcer。原act/ground/reference有限动作词表、所有输入/输出/时间/调用硬上限、严格字段和持物依赖解析均保留。新客户端在reset前核对schema hash及decoder commit。
- 完整任务提示区分“指令要求的未来结果”和“眼下画面”。目标当前不可见时仍列后续任务，各子目标自己从SEARCH开始；不能以只找到目标替代完整操作，也不能编造任务/图像未提供的目的地。
- **语法不是语义证书。** 不禁止合法纯导航，不靠模型`complete=true`自证。此次两任务静态计划由人逐项核对，尚没有通用50任务完整性认证器；这同样不证明真实执行能成功。
- 解析或输出截断失败时，先保存原请求/响应与调用计数。初始规划标记`INITIAL_PLAN`，外层异常不覆盖首个错误；不放宽JSON解析、不隐藏重试。

格式约束采用[LM Format Enforcer官方实现](https://github.com/noamgat/lm-format-enforcer/tree/817f944fcc8851917c40300a47f62d1defc3ffc3)。PyPI0.11.3实际无法导入本机Transformers5.7的tokenizer类型；仅在独立overlay安装上游固定817f944的兼容修复，不改共享环境。服务检查包的`direct_url.json`固定commit，不能仅以相同版本号冒充相同源码。

## 已核验与局限

299本地/服务器semantic测试通过；Astra独立299＋21 SFT及实际runner异常路径检查通过。真实processor六张起点图、完整2163token前缀、2/16子目标与7种恢复再切回任务共10例的逐tokenCPU检查通过，20.840秒，无模型生成/物理动作。EOS248046属于实际模型终止集合[248046,248044]；合法JSON闭合前不能提前EOS，解释/围栏起始token被拒。这些是有限协议样本，不是所有语义或所有输入的完备证明。

当前只登记2次保存起点神经规划（300秒）、两个同源工程门及一个新原始起点回合；没有重训/重复匹配前缀。原始图片的历史真实性依赖保留原包和ledger，静态工具额外证明读取图片到服务没有变化。开发实例与v3.9.1模拟器结果，不冒充官方v3.9.2竞赛分数。
