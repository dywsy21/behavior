# 开合爪命令与抓取成功分离（默认关闭）

负责人：Codex；SFT接线与独立审查：Astra。2026-09-21 13:47 BJT登记，≤1800s CPU，无新增物理重置、模型调用或训练。线上H22固定d7不热改。

## 证据与问题

H09Y TRAIN192/p392新v2教师第一个RIGHT_CLOSE发完18步后被标为TRACKING_FAILED。完整32文件10,424,297B、423含末hold控制、20物理判据帧与6张RAW经父独立核验。实际419–423夹持/指接触为真，右手末误差2.983310mm/1.299236°，左手52.2μm；没有提起，局部GRASP仍IN_PROGRESS，0正BC。

旧OPEN/CLOSE始终保持起始关节命令，并没有使用定位IK修正，却和位移动作一样要求末态2.5mm/1.5°。因此接触后的细小弹性变形会中断已经执行完的夹爪命令；反过来，手腕没移动也不证明手指开合或抓取成功。不能用将2.5调成3毫米的个例门槛来掩盖这个语义错误。

## 接口

- 显式`ServoLimits(gripper_completion_v1=True)`；部署入口`--gripper-completion-v1`默认关闭，旧回合及源不变。
- 完整OPEN/CLOSE命令得到`GRIPPER_COMMAND_COMPLETED`，不是`TARGET_REACHED`。原名义定位结果仍在`pose_tracking_status`，两手位置/角度误差原样保留。
- 必须完整18控制，末态有限值、原关节界限、自身碰撞近似检查通过；活动手不得超出原发散包络18mm/9°，非活动手仍须2.5mm/1.5°。新增末控制之后的检查，不以最后一次命令之前的安全状态替代末态。途中硬中止、未完成序列不会被改判。
- 这些是执行与姿态约束，不是环境碰撞认证；保持原深度/实际几何预检。不使用目标位置、接触、held、官方结果等特权量。
- `execution_completed(action, feedback)`统一核验新版本receipt（精确动作/控制数/末态检查/数值包络），兼容旧TARGET_REACHED。不能只给成功状态白名单加一个字符串。
- `holding=UNKNOWN`、`success_claim=false`。空夹、开爪卡住等仍是需要后继观测判定的任务状态，不能靠该receipt当抓取/放置成功。

TaskHarness和GroundedHarness只据此进入VERIFY_GRASP/VERIFY_PLACE并保留可能接触状态；原持握、抬升、稳定与官方成功门不变。采集、数据准入、base/FT/NN配对评测必须使用同一显式执行profile，旧失败不能重新写成正样本。

## 验证与下一步

15组新回归覆盖默认旧行为、左右/双手开闭、活动/非活动手、末态关节/自碰撞/非有限值、短序列/途中硬停、receipt篡改、禁止增大发散包络、非夹爪动作与harness只进入验证。加入包络上限及视觉测量失败不能覆盖的回归后，全407测试通过（6.720s）。

真实192392保存态18个逐步q/开度的新旧命令对照已完成（.731s）：全部18条23维命令逐位相同、末态误差独立复现；旧TRACKING_FAILED、新GRIPPER_COMMAND_COMPLETED并保留pose_tracking_status=TRACKING_FAILED，所有末态执行检查通过，holding仍UNKNOWN。来源32文件全SHA再核；逐tick底盘速度未存，此离线对照用有限零占位，只影响夹爪完成不使用的base_integral，**不冒充完整动力学重演或新物理成功**。脚本在父忽略目录`artifacts/agentic-vlm-goal-20260918/gripper_saved_state_comparison.py`。原失败仍失败/0正BC。

固定提交独审与SFT全链接线通过后，才使用原剩余采集预算；新120更新及六物理配对仍待完成。

原失败完整证据：`/mnt/nvme_tmp/robodojo_vlm_sft_20260919/h09y_grasp_only/native_t1_i192_p0392`；本地子worktree `artifacts/h09y-resume-20260921/native_complete/native_t1_i192_p0392`。原inventory SHA69298bcaac0f988e9d184d212e24e58528b5b7eb97cc7e69d56723d7ce803e15。
