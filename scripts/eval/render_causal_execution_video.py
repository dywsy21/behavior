"""Burn actual online planner/observer events into a continuous recorded MP4.

CPU-only media export. Never generates robot footage from sparse observations,
reruns a model, changes labels, or puts a future result into an earlier frame.
The audited collector records one post-action frame every two controls at15fps.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess

OUTCOMES = ('IN_PROGRESS', 'SUCCEEDED', 'FAILED', 'UNKNOWN')
ZH = {'IN_PROGRESS': '进行中', 'SUCCEEDED': '成功', 'FAILED': '失败', 'UNKNOWN': '未知'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def jsonl(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines()]


def probabilities(logits, temperature=1.):
    if (len(logits) != 4 or any(not math.isfinite(x) for x in logits)
            or not math.isfinite(temperature) or temperature <= 0):
        raise ValueError('Finite four-class logits and a positive temperature required')
    z = [v / temperature for v in logits]
    p = [math.exp(v - max(z)) for v in z]
    return [x / sum(p) for x in p]


def frame_states(entries, *, job_id, start, controls, nframes):
    """Attach the context that actually emitted each recorded action.

    A frame at the end of a chunk precedes the ACK/next model invocation at
    that same clock. Follow journal order through its ACTION event, not just
    `prediction_step <= image_step`, which would show the next intent early.
    """
    if nframes != len(controls) // 2 or not nframes:
        raise ValueError('Video count does not match the collector control clock')
    rows = [r for r in entries if r['job']['id'] == job_id]
    jobs = {(r['job']['case'], r['job']['seed']) for r in rows}
    if len(jobs) != 1 or not rows or rows[-1]['kind'] != 'close':
        raise ValueError('Mixed or unclosed video episode')
    for i, row in enumerate(controls, start + 1):
        if row['control_step'] != i or row['simulator_apply_ack'] is not True:
            raise ValueError('Missing physical control/ACK in video timeline')
    actions = [(i, r) for i, r in enumerate(rows) if r['kind'] == 'action']
    result = []
    for frame in range(nframes):
        step = start + 2 * (frame + 1)
        eligible = [(i, r) for i, r in actions if r['control_step'] < step]
        if not eligible:
            raise ValueError('No issued action before this video frame')
        pos, action = eligible[-1]
        if not step <= action['control_step'] + action['max_controls']:
            raise ValueError('Video frame is outside the actual action chunk')
        prefix = rows[:pos]
        plans = [r for r in prefix if r['kind'] == 'plan' and not r['result']['reused']]
        checks = [r for r in prefix if r['kind'] == 'observer']
        raw = [r for r in checks if r['result'].get('members')]
        if not plans or not checks or not raw:
            raise ValueError('Requires actual online high and observer logs, not inferred captions')
        plan, last, prediction = plans[-1], checks[-1], raw[-1]
        if (plan['result']['goal'] != action['goal']
                or plan['result']['revision'] != action['revision']
                or prediction['attempt_generation'] != last['attempt_generation']
                or any(r['control_step'] >= step for r in (last, prediction))):
            raise ValueError('Cross-intent, stale-attempt or future caption')
        skill = json.loads(action['goal']['semantic_bundle'])
        if len(skill) != 1 or len(prediction['result']['members']) != 1:
            raise ValueError('This layout explicitly supports one issued skill member')
        member = prediction['result']['members'][0]
        p = probabilities(member['raw_logits'])
        predicted = member['uncalibrated_prediction']
        if (predicted['outcome'] != OUTCOMES[max(range(4), key=p.__getitem__)]
                or abs(predicted['confidence'] - max(p)) > 2e-6):
            raise ValueError('Logged head result is inconsistent with its raw logits')
        result.append(dict(frame=frame, control_step=step, action_step=action['control_step'],
            revision=action['revision'], skill=skill[0], plan=plan['result'],
            observer=prediction, feedback=json.loads(last['estimated_feedback']),
            physical=controls[step-start-1], probabilities=p))
    return result


def media_info(path):
    data = subprocess.check_output(['ffprobe', '-v', 'error', '-select_streams', 'v:0',
        '-count_frames', '-show_entries', 'stream=codec_name,width,height,pix_fmt,r_frame_rate,nb_read_frames,duration',
        '-of', 'json', str(path)], text=True)
    return json.loads(data)['streams'][0]


def stamp(frame):
    # Floor to centiseconds. At15fps this boundary lies strictly after the
    # preceding display frame, so no event is shown on an earlier frame.
    centis = frame * 100 // 15
    return f'{centis//360000}:{centis//6000%60:02d}:{centis//100%60:02d}.{centis%100:02d}'


def clean(text):
    return str(text).replace('\\', '/').replace('{', '(').replace('}', ')').replace('\n', ' ')


def line(y, text, *, size=23, color='E8E8E8', x=28):
    return r'{\an7\pos(' + f'{x},{y}' + r')\fs' + str(size) + r'\c&H' + color + '&}' + clean(text)


def minimal_texts(goal, members, decision='EXECUTE'):
    """Only the two requested fields; probabilities are raw model outputs."""
    skills=json.loads(goal['semantic_bundle'])
    if len(skills)!=len(members) or [m['member'] for m in members]!=list(range(len(skills))):
        raise ValueError('Intent and per-member head outputs must be aligned')
    verbs={'GRASP':'抓取','OPEN_DOOR':'打开','PLACE':'放置','RELEASE':'松开',
        'TOGGLE_ON':'开启','TOGGLE_OFF':'关闭','NAVIGATE':'移动','REACH':'伸手靠近'}
    arms={'LEFT':'左手','RIGHT':'右手','BOTH':'双手','UNSPECIFIED':''}
    intents=[];outcomes=[]
    for skill,member in zip(skills,members):
        arm=arms.get(skill['arm'],skill['arm'])
        target=skill['target']
        intention=f"{arm}{verbs.get(skill['verb'],skill['verb'])} {target}"
        if skill.get('destination'):intention+=' → '+skill['destination']
        intents.append(intention)
        p=probabilities(member['raw_logits']);i=max(range(4),key=p.__getitem__)
        outcomes.append(f"{arm or target}：{ZH[OUTCOMES[i]]} {p[i]*100:.1f}%")
    prefix='重试 · ' if decision=='RETRY' else ''
    return ['当前意图：'+prefix+'； '.join(intents),'结果头预测：'+'； '.join(outcomes)]


def minimal_caption(state, **unused):
    texts=minimal_texts(state['plan']['goal'],state['observer']['result']['members'],state['plan']['event']['decision'])
    return [line(462,texts[0],size=25,color='FFFF8A'),line(507,texts[1],size=25)]


def caption(state, *, title, job, speed, temperature, models, terminal=None):
    skill = state['skill']; plan = state['plan']; event = plan['event']
    raw = state['observer']; member = raw['result']['members'][0]
    p = state['probabilities']; label = OUTCOMES[max(range(4), key=p.__getitem__)]
    feedback = state['feedback']; consumed = json.loads(plan['causal_input']['execution_feedback'])
    verb = {'GRASP': '抓取', 'OPEN_DOOR': '打开门'}.get(skill['verb'], skill['verb'])
    arm = {'UNSPECIFIED': '手臂未指定', 'RIGHT': '右手', 'LEFT': '左手'}.get(skill['arm'], skill['arm'])
    decision = {'RETRY': '重试', 'EXECUTE': '执行', 'REPLAN': '重新规划'}.get(event['decision'], event['decision'])
    memory = json.loads(event['memory_update'])
    physics = state['physical']; evidence = physics['physical_evidence']
    if 'directed_open_fraction' in evidence:
        physical_text = f"门开度={evidence['directed_open_fraction']:.3f}；原通过门为 >=0.350 且稳定6控制步"
    else:
        physical_text = '目标被对应手持有=' + str(evidence['target_held_by_requested_arm']) + '；稳定6控制步才算局部成功'
    if skill['verb'] == 'GRASP':
        cp = probabilities(member['raw_logits'], temperature)
        calibrated = f'T={temperature:.3f} 后：{OUTCOMES[max(range(4),key=cp.__getitem__)]} {max(cp)*100:.1f}%；仍需 >=85% / 两次新观测'
    else:
        calibrated = '当前仅 GRASP 通过校准；开门结果头原始分数仅作诊断，不接入高层作为已知结果'
    rows = [line(16, title, size=31, color='FFF0CF'),
        line(62, f"实例{job['case'].rsplit('_',1)[-1]} / seed{job['seed']}  |  播放 {speed:g}x（无生成/插值画面）  |  局部技能，非完整任务", size=20),
        line(99, '头部相机', size=20, x=180), line(99, '左腕相机', size=20, x=628),
        line(99, '右腕相机', size=20, x=1076),
        line(596, f"当前意图：{event['decision']}（{decision}） / {skill['verb']}（{verb}） {skill['target']} · {arm}", size=24, color='FFFF8A'),
        line(634, f"结果头原始输出：{label}（{ZH[label]}）{max(p)*100:.1f}%  |  检查@控制{raw['control_step']}，当前{state['control_step']}，非逐帧新预测", size=22),
        line(669, '原始四类概率：' + '  |  '.join(f'{ZH[k]} {v*100:.1f}%' for k,v in zip(OUTCOMES,p)), size=21),
        line(704, calibrated, size=21, color='85CFFF'),
        line(740, f"过滤后可用结果：{feedback['estimated_bundle_outcome']}  |  高层上次实际消费：{consumed['estimated_bundle_outcome']}（规划@{plan['control_step']}）", size=23, color='A9EDAE'),
        line(778, f"当前控制步 {state['control_step']}  |  意图版本 {state['revision']}  |  已发命令记录 {len(memory['issued_command_history'])}；已证世界事实 {len(memory['verified_world_facts'])}", size=21),
        line(814, '物理核验（不输入模型）：' + physical_text, size=21, color='E6C3AF'),
        line(851, f"高层 {models['planner'][:8]} / 低层FM {models['low'][:8]} / 结果头 {models['observer_adapter'][:8]}；字幕来自该次在线日志，不是补写的推理。", size=19),
        line(888, '原片15fps，每2个实际控制步录1帧；保留连续镜头。UNKNOWN不能替换为物理成功。', size=19)]
    if terminal:
        tail_note = '最后1控制步未录入原视频' if terminal['scored_controls']%2 else '全部控制步均有视频采样'
        rows[-1] = line(888, f"片尾日志终态：{terminal['outcome']}，共{terminal['scored_controls']}控制步；末帧定格，{tail_note}。", size=21, color='85CFFF')
        rows.append(line(924, '这是局部物理判据，不是结果头预测，也不是完整任务SR。', size=18))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('episode', 'physical-audit', 'causal-audit', 'service', 'journal', 'calibration', 'config', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--title', required=True)
    parser.add_argument('--speed', type=float, choices=(1., .5, .25), default=1.)
    parser.add_argument('--layout', choices=('minimal','diagnostic'), default='minimal')
    a = parser.parse_args()
    if a.output.exists(): raise FileExistsError(a.output)
    result, start, audit, causal, service, config = map(read,
        (a.episode/'result.json', a.episode/'start.json', a.physical_audit, a.causal_audit, a.service, a.config))
    job = result['job']; job_id = job['id']; controls = jsonl(a.episode/'controls.jsonl')
    row = [r for r in audit['episodes'] if r['job_id'] == job_id]
    if len(row) != 1: raise ValueError('An exact audited physical episode is required')
    row = row[0]
    bindings = [(a.config, audit['config_sha256']), (a.episode/'result.json', row['result_sha256']),
        (a.episode/'controls.jsonl', row['controls_sha256']), (a.episode/'policy.mp4', row['policy_video_sha256']),
        (a.episode/'restore-diagnostics.json', row['restore_sha256']),
        (a.physical_audit, causal['physical_audit_sha256']), (a.service, causal['service_result_sha256']),
        (a.journal, causal['journal_sha256']), (a.calibration, service['calibration_sha256'])]
    if (any(sha(p) != digest for p,digest in bindings)
            or audit['status'] != 'machine_control_reward_reset_audit_passed'
            or causal['status'] != 'all_joint_observations_intents_actual_actions_and_clocks_replayed'
            or service['status'] != 'completed_read_only_skill_probes'
            or service['optimizer_updates'] != 0 or start['job'] != job
            or result['policy_sha256'] != config['model']['sha256']
            or result['identity'] != start['identity']
            or row['case'] != job['case'] or row['seed'] != job['seed']
            or row['actual_controls'] != result['scored_controls']):
        raise ValueError('Changed/unverified source, model, episode or audit')
    info = media_info(a.episode/'policy.mp4')
    if (info['codec_name'],info['width'],info['height'],info['r_frame_rate']) != ('h264',672,224,'15/1'):
        raise ValueError('This export binds the audited15fps three-camera collector format')
    states = frame_states(jsonl(a.journal), job_id=job_id, start=config['cases'][job['case']]['start_control'],
        controls=controls, nframes=int(info['nb_read_frames']))
    a.output.mkdir(parents=True)
    repeats = round(1/a.speed); body_frames = len(states)*repeats; tail_frames = 30
    height=558 if a.layout=='minimal' else 960
    render_caption=minimal_caption if a.layout=='minimal' else caption
    ass = ['[Script Info]', 'ScriptType: v4.00+', 'PlayResX: 1344', f'PlayResY: {height}', 'WrapStyle: 2',
        '[V4+ Styles]', 'Format: Name,Fontname,Fontsize,PrimaryColour,SecondaryColour,OutlineColour,BackColour,Bold,Italic,Underline,StrikeOut,ScaleX,ScaleY,Spacing,Angle,BorderStyle,Outline,Shadow,Alignment,MarginL,MarginR,MarginV,Encoding',
        'Style: Default,Noto Serif CJK SC,23,&H00EEEEEE,&H00EEEEEE,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,0,0,7,0,0,0,1',
        '[Events]', 'Format: Layer,Start,End,Style,Name,MarginL,MarginR,MarginV,Effect,Text']
    def emit(first,last,texts):
        for text in texts: ass.append(f'Dialogue: 0,{stamp(first)},{stamp(last)},Default,,0,0,0,,{text}')
    temperature = read(a.calibration)['temperature']; models = service['joint_model_identity']
    for i,state in enumerate(states):
        emit(i*repeats,(i+1)*repeats,render_caption(state,title=a.title,job=job,speed=a.speed,temperature=temperature,models=models))
    emit(body_frames,body_frames+tail_frames,render_caption(states[-1],title=a.title,job=job,
        speed=a.speed,temperature=temperature,models=models,terminal=result))
    subtitle = a.output/'overlay.ass'; subtitle.write_text('\n'.join(ass)+'\n')
    # A real decoded video is the sole visual input. Slow playback duplicates
    # existing frames; no sparse RGB/AI-generated/interpolated frame is used.
    path = str(subtitle.resolve())
    if any(c in path for c in "':,[]\\"): raise ValueError('Unsupported filter-path character')
    video = a.output/'execution.mp4'
    top=0 if a.layout=='minimal' else 132
    filters = f'scale=1344:448:flags=lanczos,pad=1344:{height}:0:{top}:color=0x101722,setpts=PTS/{a.speed},tpad=stop_mode=clone:stop_duration=3,fps=15,ass=filename={path}'
    cmd = ['ffmpeg','-nostdin','-v','warning','-n','-i',str(a.episode/'policy.mp4'),'-an',
        '-vf',filters,'-frames:v',str(body_frames+tail_frames),'-c:v','libx264','-crf','18',
        '-preset','medium','-pix_fmt','yuv420p','-movflags','+faststart',str(video)]
    subprocess.run(cmd,check=True)
    final = media_info(video)
    if int(final['nb_read_frames']) != body_frames+tail_frames or final['r_frame_rate'] != '15/1':
        raise ValueError('Encoded video timing differs from caption timing')
    subprocess.run(['ffmpeg','-nostdin','-v','error','-i',str(video),'-f','null','-'],check=True)
    manifest = dict(status='encoded_mp4_full_decode_passed', title=a.title, job=job,
        source_video_sha256=sha(a.episode/'policy.mp4'), video_sha256=sha(video),
        physical_audit_sha256=sha(a.physical_audit),causal_audit_sha256=sha(a.causal_audit),
        service_sha256=sha(a.service),journal_sha256=sha(a.journal),config_sha256=sha(a.config),models=models,
        source_path=str(a.episode),original_frames=len(states),output_frames=body_frames+tail_frames,
        playback_speed=a.speed,terminal_freeze_seconds=2,source_real_controls=result['scored_controls'],
        source_final_controls_not_recorded=result['scored_controls']%2,
        original_fps=15,raw_prediction_is_not_calibrated_physical_truth=True,
        all_captions_bound_to_actual_emitting_action=True,synthetic_robot_images=0,
        newly_run_model_inferences=0,new_physical_controls=0,whole_task_sr=False,
        encoder_command=cmd,stream=final,layout=a.layout,owner_output_media_review_pending=True)
    (a.output/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    timeline=[dict(source_frame=x['frame'],control_step=x['control_step'],action_step=x['action_step'],
        revision=x['revision'],observer_step=x['observer']['control_step'],
        raw_outcome=x['observer']['result']['members'][0]['uncalibrated_prediction'],
        accepted_outcome=x['feedback']['estimated_bundle_outcome'],
        actual_high_input_outcome=json.loads(x['plan']['causal_input']['execution_feedback'])['estimated_bundle_outcome']) for x in states]
    (a.output/'timeline.json').write_text(json.dumps(timeline,indent=2)+'\n')
    print(json.dumps(manifest,ensure_ascii=False))


if __name__ == '__main__': main()
