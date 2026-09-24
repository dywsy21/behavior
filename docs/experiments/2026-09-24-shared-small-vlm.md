# H45：训练期间的小VLM共享显存可行性

## 问题与边界

最新active goal允许在现有训练运行时尝试小VLM。当前四张A100各有约7489MiB可用显存，训练为xhz的3294346–3294349，不能停止或改配置。历史2B小输入峰值约4570MiB，不能直接代表当前6–9图/长上下文harness，更不能代表模拟器也放得下。

唯一主要假设：冻结Qwen3.5-2B能在保留训练显存余量的条件下，处理当前harness的真实公开语义请求。负责人Codex-parent。本实验只筛查资源与输出，不测新成功率、不训练，不取代官方完整回合。

## 冻结条件

- 配置：`configs/semantic_robot/h45_shared_small_vlm_probe.json`；代码：`scripts/semantic_robot/probe_shared_vlm.py`，必须干净Git独立worktree；实际commit记入结果。
- 权重：Qwen3.5-2B revision `15852e8c16360a2fea060d615a32b45270f8a8fc`，完整10文件SHA与根目录清单固定。可选`generation_config.json`当前不存在，新增它也会拒绝，不能只记录版本字符串。
- 输入：原H38 task0 TRAIN-development138、seed0保存态，4个请求为规划、d0观察、d91观察、d91动作；30图顺序/RGB SHA逐一匹配原记录。只传原actor公开字段，不读取特权诊断标签来修动作。
- 所有原视角和文本保留，图像明确从最大640降到320，不声称这是与27B的模型单变量对照。
- 固定greedy/BF16/SDPA、Transformers5.7.0、原结构解码commit817f944；规划/观察/动作最大输出1024/320/64，原输入上限12000token。
- GPU2精确UUID/训练PID；PyTorch allocator≤4864MiB、另计512MiB非Torch开销、至少保留2048MiB。allocator限制不是整个进程的硬显存隔离，2s采样不能保证瞬时零风险，也不承诺共享计算无性能影响。
- 只运行一次，4调用；模型阶段600s、独立监管最多900s（含hash/CPU准备），无重试/新reset/训练。未知进程、低余量、OOM或到时，只结束新建worker。独占launch回执避免SSH断连后重复提交。

## 当前结果

2026-09-24 17:37北京时间：19 CPU测试过；4个真实保存态请求/30图CPU恢复及原图hash核验过。独立审查中，未调用GPU、未生成新输出、未启动模拟器。原H44两工程门已通过，不为新模型复跑那些门。

17:41：完整模型manifest及资源TOCTOU问题经独审指出后修复；首次CUDA和CPU模型加载后分配前两次全预算核验，已占自有context扣账，进程总显存上限5376MiB；采用UUID可见性。21 CPU与修后独立复审通过，无剩余实质发现。仍0 GPU调用，下一固定commit/远端单次探针。

## 后续判断

实际记录输入token、延迟、allocator/进程显存、语法和人工语义检查。格式合法不等于空间判断正确；保存态输出正确不等于闭环成功。只有探针资源/语义可用，且另有模拟器资源依据，才登记一个有限无专家/旧策略前缀完整回合。训练资源不足时不偷降保护余量或把局部抓取算成官方成功。

## H45实际失败与H45b修复

H45由99caf682在17:46提交，supervisor3438643/worker3438651。worker18.823s、监管27.538s后exit1，0次生成：在停止符一致性门拒绝。采样自有峰值4810MiB、最低空闲2674MiB，退出后7489MiB；四训练保持原PID。仅权重装载通过余量门，推理显存/语义仍未知。

完整本地证据`artifacts/agentic-vlm-goal-20260918/h45_shared_bundle_v1`。result SHA `02775d9c61a5df9ac92fd38cac2431e71ea5b4a3f74a53bd275093a0525f89b5`，supervisor SHA `88325174c6cecf5aa75838b8712fd8e1428f4e03e02d06bfb9352217436475d1`。

实际CPU读取显示，缺独立generation配置时默认只用text_config的248044（文本结束），而tokenizer的对话结束是248046。H45b显式保留二者作为生成停止符，且三项准确身份都要匹配；不改本地模型文件或删改生成文字。原v1严格检查不放松，新配置单独opt-in，原输出保留。26 CPU已过，独立窄审待；后续单次复验预算/输入与H45相同，没有新模型搜索/训练/物理。

17:53独立窄审通过：LMFE结束符在有效集合内，有限动作trie只在完整候选后放行两个EOS，无截断修补；26 CPU过，下一冻结源码后单次复验。原v1及现有训练不动。
