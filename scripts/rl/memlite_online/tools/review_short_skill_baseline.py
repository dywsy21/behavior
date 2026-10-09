"""Recompute every real baseline control/reward and render review panels.

This produces evidence, never an automatic BASELINE_ACCEPTED authorization.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
from recovery_corpus import file_sha,digest
from skill_aligned_reward import SkillIdentity,SkillReward,skill_measurement


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True);p.add_argument('--runs',type=Path,nargs='+',required=True)
    p.add_argument('--policy-sha256',required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();cfg=json.loads(a.config.read_text())
    if a.output.exists():raise FileExistsError(a.output)
    import numpy as np
    import imageio.v2 as iio
    from PIL import Image,ImageDraw
    expected={(c,s) for c in cfg['cases'] for s in cfg['evaluation_seeds']}
    seen=set();audits=[];a.output.mkdir(parents=True)
    for directory in a.runs:
        result=json.loads((directory/'result.json').read_text());start=json.loads((directory/'start.json').read_text())
        reset=json.loads((directory/'restore-diagnostics.json').read_text())
        process=json.loads((directory.parent/'result.json').read_text());job=result['job']
        key=(job['case'],job['seed']);case=cfg['cases'][job['case']]
        if (key not in expected or key in seen or job['phase']!='evaluation' or job['round']!=0
                or process['status']!='completed_single_episode' or process['completed_episodes']!=1
                or process['fresh_process_per_episode'] is not True or process['config_sha256']!=file_sha(a.config)
                or any(x['policy_sha256']!=a.policy_sha256 for x in (result,start))
                or start['job']!=job or reset['job']!=job or result['whole_task_sr']
                or reset['fresh_process_per_episode'] is not True):
            raise ValueError('Mixed policy, worker, job or baseline identity')
        seen.add(key);identity=SkillIdentity(**result['identity']);skill=json.loads(case['semantic_bundle'])[0]
        if (start['identity']!=asdict(identity) or identity.task!=case['task']
                or identity.instance!=case['instance_id'] or identity.episode!=job['id']
                or identity.context_id!=case['context_id'] or identity.bundle_sha256!=digest([skill])):
            raise ValueError('Mixed task, instance, intent or episode')
        if reset['proprio_error']>1e-3 or reset.get('target_position_max_error',0)>.005:
            raise ValueError('Cold baseline pose gate failed')
        controls=[json.loads(x) for x in (directory/'controls.jsonl').read_text().splitlines()]
        if (file_sha(directory/'controls.jsonl')!=result['controls_sha256']
                or len(controls)!=result['scored_controls'] or not controls):raise ValueError('Control ledger changed')
        measurement=skill_measurement(skill,start['initial_physical_evidence'])
        reward=SkillReward(identity,measurement,control_step=case['start_control'])
        for i,row in enumerate(controls):
            t=case['start_control']+i+1;command=np.asarray(row['action_executed_raw23'])
            if (row['control_step']!=t or row['simulator_apply_ack'] is not True or command.shape!=(23,)
                    or not np.isfinite(command).all() or row['action_source']!='evaluation'
                    or row['policy_version']!=result['policy_version']):raise ValueError('Actual action/ACK/clock differs')
            measured=skill_measurement(skill,row['physical_evidence'])
            value=reward.advance(identity,t,measured,protected_values={},official_terminal=row['official_terminal'],
                time_limit=row['official_truncated'] or t==case['end_control'])
            if dict(value,identity=asdict(identity))!=row['skill_reward']:raise ValueError('Skill reward does not recompute')
        if (value['skill_success']!=result['skill_success'] or value['outcome']!=result['outcome']
                or not (value['terminated'] or value['truncated'])):raise ValueError('False episode completion')
        for filename,sha in start['original_rgb'].items():
            if file_sha(directory/filename)!=sha:raise ValueError('Original start image changed')
        frames=iio.mimread(directory/'policy.mp4',memtest='1GB')
        if not frames:raise ValueError('No actual policy video')
        indices=sorted({0,len(frames)//3,2*len(frames)//3,len(frames)-1})
        canvas=Image.new('RGB',(672,250*(len(indices)+1)),'white');draw=ImageDraw.Draw(canvas)
        draw.text((4,4),job['case']+f" seed={job['seed']} original start",fill='black')
        for col,camera in enumerate(('head','left_wrist','right_wrist')):
            with Image.open(directory/(camera+'_rgb.png')) as im:canvas.paste(im.convert('RGB'),(col*224,23))
        for row,index in enumerate(indices,1):
            draw.text((4,row*250+4),f'actual rollout encoded frame={index}, final={result["outcome"]}',fill='black')
            canvas.paste(Image.fromarray(frames[index]),(0,row*250+23))
        sheet=a.output/(job['case']+f'-seed{job["seed"]}.jpg');canvas.save(sheet,quality=95,subsampling=0)
        audits.append(dict(case=job['case'],seed=job['seed'],job_id=job['id'],run=str(directory),
            result_sha256=file_sha(directory/'result.json'),controls_sha256=result['controls_sha256'],
            restore_sha256=file_sha(directory/'restore-diagnostics.json'),policy_video_sha256=file_sha(directory/'policy.mp4'),
            actual_controls=len(controls),outcome=result['outcome'],success=result['skill_success'],
            reset_proprio_error=reset['proprio_error'],reset_target_error=reset.get('target_position_max_error'),
            final_physical=controls[-1]['physical_evidence'],sheet_sha256=file_sha(sheet),sheet=sheet.name))
    if seen!=expected:raise ValueError('Incomplete fixed-seed baseline, preserve partial audit')
    result=dict(status='machine_control_reward_reset_audit_passed',episodes=audits,
        policy_sha256=a.policy_sha256,config_sha256=file_sha(a.config),reviewed_by_human=False,
        optimizer_authorized=False,whole_task_sr=False)
    (a.output/'audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__=='__main__':main()
