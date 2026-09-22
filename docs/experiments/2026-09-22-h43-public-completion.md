# H43：复用真实动作历史，核验完成请求

负责人：Codex父；2026-09-22 11:39登记，CPU最多1800秒、0模型调用/训练/reset。此实现没有接入任何正在运行的服务或模拟器。

## 要修的实际问题

原motion-only SFT和不用图像的近邻策略都在开发i71中抓住过垃圾桶，但继续UP，最后碰到躯干而失败。原34条训练行只有动作前状态，遗漏了5条成功终态；H09AA补了独立的REQUEST_VERIFY标签。不能靠追加一个HOLD动作标签解决：原HOLD会继续决策，末态真实清理HOLD又只有1个控制tick，不能编造18tick示范。

H09AB负责学习“现在应请求验收吗”，H43只回答“已有的公开传感器证据是否支持持握”。二者都不是官方成功判据。

## 接口与边界

`src/semantic_robot/v2/skill_completion.py`新增默认不调用的独立函数，不修改旧run_v2、native45控制器或旧六槽评测。

- `CaptureRef(directory, sha256, clock)`是外部证据绑定，不是送入VLM的文本。
- `ExecutedMotionRef(token, execution_path, execution_sha256, before, after)`指向本回合实际完成的微动作及前后机载capture。
- `verify_grasp_request(model, goal, records, request_capture, *, localize, deadline, requested_status)`只接受明确REQUEST_VERIFY、单手pick、1次实际CLOSE后连续2–4次实际UP。最多5次定位回调、0新动作。其他技能/不完整历史返回UNKNOWN，不能假称覆盖所有技能。
- `localize(CapturedState, Goal)`回调只发送当前RAW三视图和公开目标描述；文件路径、实例编号、时钟、自体几何和任何结果字段都不送入模型。严格返回只定位表面的GroundedEvidence；enclosed/co_moving/supported/effect必须为null，不消费模型自报持握。
- 返回`PUBLIC_HOLDING_VERIFIED`或`UNKNOWN`，始终`not_skill_success=true`和`not_official_task_success=true`。调用方仍需单独决定保持/交接和评分。

## 为什么不是又做一次“试着抬手”

验证器复用此前已执行的抬升及图像，没有控制接口。每个capture校验全部7个文件SHA、深度数组与单位、三视图同一snapshot、render barrier、实际关节/夹爪/FK、机器人自体排除几何。动作时钟必须连续且与真实12tick settle匹配，失败/缺失/CLOSE手臂不符都拒绝。

请求必须绑定最新状态：clock/q/gripper/base velocity与最后after_settle一致。可以在同tick重新render，但使用请求的当前图像，并且不把新render算作新动作。图像缺失、不确定目标、无有效深度或超时不能成为成功；最终返回只看请求终点，不能截取前面偶然通过的一帧。

仍使用原`RGBDMotion(rgbd_joint)`、实际机器人自体排除、`GraspMotionVerifier`持续特征/空间采样和原阈值；没有调低稳定性、位移或跟踪门。

## 作者验证（独审/部署未完成）

2026-09-22 11:53：13专项/0.708s；543公共/20.388s；174旧native/6.170s通过。旧native协议与训练代码未修改。

完整保存态复算14.304s，复用H42已经付费并保存的12次定位响应，实际新模型调用和控制均为0：

| 保存态 | 各终点公开持握 | 脚本将最后目标设为不可见 |
| --- | --- | --- |
| TRAIN114 prefix989 | false,false,true,true | UNKNOWN |
| i71原FT | false,false,false,true | UNKNOWN |
| i71原NN | false,false,true,true | UNKNOWN |

全部与H42原数值过程的布尔结果相同，FT首无有效深度仍保留。负例是脚本改定位证据的接口测试，不是新模型的误报率。

证据：本地`artifacts/agentic-vlm-goal-20260918/h43_saved_replay_author_v2.json`，SHA `309dabac76708c1a39afc9ac5f2939e9d38eac28c430898c13ff14f1e4593c38`；早期无snapshot检查的v1保留，不能替代最终代码证据。

关键限制：NN公开持握早于原严格局部技能完成。因此不能收到这个布尔值就自动认定GRASP完成。下一步是独立源码审查、精确版本的学习请求＋公开验证接线，再用新的同起点预算比较原0120和completion增量权重。旧i71终态失败和官方0保持不变。
