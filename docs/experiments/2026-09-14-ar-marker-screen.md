# AR marker行训练：阶段证据

## 最新：最终十窗口内容验收（2026-09-14 19:08北京时间）

原A4父权重重训marker500的十窗口自由AR验收已完成，result SHA `831cbd40586ba5c1fca9481485072fe964030c46b3cc9f84c47ef32f6eb64330`。[逐窗/机器摘要](results/2026-09-14-ar-marker-actions.json)。source81810be/42 CPU，2021126已退出，原10生成预算用完；0新FM/teacher encode/训练/物理。每条60动作tokens＋终止248044，均原自由生成、实际AR、8组完整，不是静态schema强制，也不是任务SR。完整1140状态/192 LoRA/2 marker逐位恢复passed（receipt SHA `140e70e4e667b14d30ec5081e19382e4fdbdb1493f218aedb6e72a986f837d36`）。

十条来源、实际像素/本体/mask均与已完成M04 FM十窗逐项匹配；对照动作与原A4缓存均复算目标/有效支持，不额外生成。以下按SSE/有效scalar合并，越低越好；全10有效，无失败筛选偏差。短train窗口的有效长度保留，不补到16。

| 诊断划分/区间 | 有效scalar | marker AR500 | 同入口FM500 | A4缓存参考 |
| --- | ---: | ---: | ---: | ---: |
| train前0:16 | 1403 | 0.8206035 | 0.4470648 | 0.4525187 |
| heldout前0:16 | 1840 | 1.1197821 | 0.8818891 | 0.9016779 |
| train后16:32 | 1104 | 0.7751956 | 0.6300357 | 0.6505368 |
| heldout后16:32 | 1840 | 1.2282899 | 0.9744828 | 0.9909080 |

前段AR只在train_task1、heldout_task2、heldout_task4三窗胜FM，其余七窗更差；heldout前段合并误差高26.98%。因此格式学习有实际进展，**内容尚未总体胜出，本marker候选不升默认、不据个别好窗扩物理或追加5000**。这不是完整成功率，不是否定所有AR或证明marker单因素因果（本配方与旧AR另有eager/fused CE差异）。下一原native CoT/其余FM方法继续，marker这十窗不再重跑。

## 训练终点与历史阶段证据

2026-09-14 18:56北京时间更新，Codex / AR-01。**正式500已完整完成，动作配对/方法验收仍待做。** run `robo:/mnt/sdc1/robodojo/behavior_dev/dual_track_fm_ar_20260913/ar_a4_marker_fulltrain_v3`，source9f26b45；原spec `9401ca0c526853e756e7f44b5a7fdf6879346cef5dfc96509f84b8d1ea897bd0`。从原A4独立训练LoRA＋8行共享marker delta，纯AR/无FM更新，不叠静态schema强制解码；不是原生task-only或CoT那两条路线。

最终500/8000抽取/193 Adam/冻结及完整模型优化器四rank RNG回读passed；checkpoint `3801388d71381c4cd586dac4bc19b07164e8922b8de6b5ea869bdcc52a56b52b`，inspection `b848f38026e5583f0cfa4d3471cd8647ed0eddb70a36a366617a5e5f73ce43a4`；未实测新进程续训。原固定80 CE4.619393786787986、action4.696353414654732、text/boundary0.0018457102428556028，FM仅参考0.19695782312192023，eval SHA `36a9fc0947e285d828f42e683aab962e6f3594302c44513830e6e6a2753c619f`。原五诊断自由动作全部完整/source AR，task0–4前段RMSE依次0.9996753931、1.7443532944、1.0976165533、1.0016080141、0.1395004243。**结构5/5不是成功率，不能据此部署**；与旧FM未核实同实际像素，不先写严格内容胜负。以下300/200保留为历史。

最新300点：固定80 CE=4.7525360107（action=4.8317308754，text/boundary=0.00087947054），自由完整4/5；task0–3各61tokens、source均AR，前16归一化RMSE依次1.0272169113、1.7956352234、1.3895500898、1.2618654966；task4生成长度到300仍不完整。task1/3的同名诊断窗口记录比200点误差更高，不把结构改善/多出有效样本或更低CE当作内容收益。原`formal/eval_step_300.json` SHA `75938b004cc100cb9bac7d497c141b74f586abac74e4017cc9ee9293a234c59d`；尚未核对与旧FM生成的实际像素逐位身份，不作严格跨模型动作胜负结论。以下200点表保留作历史过程证据。

## 当前结果：格式开始学到，动作还不能据此判好

原固定80的CE从step0的18.6797801到100的5.5262159、200的4.9585251。100→200的自由完整动作从0/5到2/5：task1的GRASP和task3的CLOSE_DOOR各生成完整8组，共60个动作tokens＋1个终止符；实际`selected_action_source`均为AR，无FM或GT补组。

| 原留出诊断窗口 | step200生成 | 有效执行0:16的归一化RMSE |
| --- | --- | ---: |
| task0-00-190-289 / GRASP | 43 tokens；缺右臂/下身残差层 | 不计有效动作分数 |
| task1-16-390-1007 / GRASP | 61 tokens；8组完整 | 1.5051903725 |
| task2-32-590-5263 / CLOSE_DRAWER | 91 tokens；重复body1、未通过完整动作检查 | 不计有效动作分数 |
| task3-48-790-9174 / CLOSE_DOOR | 61 tokens；8组完整 | 1.0019207001 |
| task4-64-990-10974 / CLOSE_DOOR | 70 tokens；重复body1、结构非法 | 不计有效动作分数 |

200点action-token CE=5.0410959482，text/boundary CE=0.0043092482。结构/边界学习不能冒充动作内容正确，更不能把2/5写成任务success rate。两条已解码动作的误差仍较大；未用仅这两条有效子集的均值与FM全五条比较，也没有把失败三条填零/自动补动作。当前无marker分支的仿真结果。

原完整文件分别为`formal/eval_step_100.json`（SHA `12082a24442df7d0b2368974b40fafeef1b73927b2ae48cfb8f0d98920e5256a`）与`formal/eval_step_200.json`（`059b0b4d07edf8384c3c84c2d5c5fa53b12e0bd46e36820f8adedfe21b5fc217`）。这些是已发生的原评估回放读取，本次没有新增生成、teacher forcing、截断上限或训练步数。

## 后续

18:58执行更新：81810be97b108d1e843844139535c86931c60a80独立源的新42 CPU测试passed（0.42s），旧推理/数据源码身份全部匹配；唯一runtime差异只是未使用EMA helper注册，去除该单行后整文件SHA还原15fbd79f…。`ar_marker_actions_v1`18:57:18启动2021126，原10 AR/0更新仿真；同级launch/log有回执，内容结果待验，不重提。原CoT220792已smoke2019434。下段“入口待实现”是18:48历史状态。

18:48新增有限验收预登记：`ar_marker_actions_v1`由Codex负责，原marker500完整保存门后最多10自由AR调用，原5train/5heldout、seed17/300 token上限、0新训练/新FM或A4参考/仿真。以M-04 FM500缓存（`m04_actions_fm_v1/result.json`，SHA `d3502794bb1d3ff7c27d8838c55c331a8faba22b23dd32ff82c4fdf2d87193c7`）验证实际输入指纹并配对内容；A4十输出仅只读复用。完整1140状态恢复和193 Adam/8000抽取门先通过，格式失败不填零或补组，内容只和相同有效子集比较并报覆盖。独立源/CPU后执行，GPU1需40GiB空闲/40%显存上限；输出已存在、源/状态/配对不符立即停，不自动重试。此刻入口待实现、0新增神经调用，不意味着marker方法已通过。

继续原500及完整保存门，核对300/500自由完整性与内容后再决定严格有限动作/闭环验收；不部署中途权重、不因出现2/5就追加5000。CoT220792仍按原依赖等待marker4129562；EMA、尾端LR也各有独立源/父权重/预算，不重排或混称同一个模型。
