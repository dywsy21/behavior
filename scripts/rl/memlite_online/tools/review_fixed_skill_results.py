"""Audit same-skill physics ledgers and render original RGB; never sign success."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
from recovery_corpus import file_sha
from skill_aligned_reward import SkillIdentity,SkillReward,skill_measurement


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config',type=Path,required=True);ap.add_argument('--runs',type=Path,nargs='+',required=True)
    ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    from PIL import Image,ImageDraw
    cfg=json.loads(a.config.read_text());a.output.mkdir(parents=True);audits=[]
    for directory in a.runs:
        receipt=json.loads((directory/'result.json').read_text())
        case=cfg['cases'][receipt['task']+'_'+str(receipt['instance_id'])]
        if (receipt['status']!='completed_fixed_skill_policy' or receipt['policy_config_sha256']!=file_sha(a.config)
                or receipt['model_sha256']!=cfg['models'][receipt['model']]['sha256']
                or receipt['optimizer_steps']!=0 or receipt['whole_task_success_evaluated']):
            raise ValueError('Changed policy/config/status')
        rows=[json.loads(x) for x in (directory/'reward-ledger.jsonl').read_text().splitlines()]
        identity=SkillIdentity(**rows[0]['reward']['identity']);skill=json.loads(case['semantic_bundle'])[0]
        if (identity.context_id!=case['context_id'] or identity.task!=receipt['task']
                or identity.instance!=receipt['instance_id']):raise ValueError('Mixed task or context')
        reward=SkillReward(identity,receipt['start_measurement'],control_step=case['start_control'])
        for i,row in enumerate(rows):
            t=case['start_control']+i+1
            if (row['control_step']!=t or row['simulator_apply_ack'] is not True
                    or row['action_source']!='learned_native_fm' or len(row['action_executed_raw23'])!=23):
                raise ValueError('Control ACK/clock/source mismatch')
            measurement=skill_measurement(skill,row['physical_evidence'])
            if row['measurement']!=measurement:raise ValueError('Wrong measured skill')
            # Native full-task terminal is not inferred from an elapsed skill
            # interval; this fixed probe only scores actual stable skill success.
            recomputed=reward.advance(identity,t,measurement,protected_values={},time_limit=t==case['end_control'])
            if dict(recomputed,identity=asdict(identity))!=row['reward']:raise ValueError('Reward mismatch')
        if rows[-1]['reward']['skill_success']!=receipt['learned_policy_success']:raise ValueError('Incorrect success receipt')
        for group in receipt['original_rgb'].values():
            for name,sha in group.items():
                if file_sha(directory/name)!=sha:raise ValueError('Changed original image')
        frames=sorted(int(k.removeprefix('policy_control_')) for k in receipt['original_rgb'] if k.startswith('policy_control_'))
        # Include start, intermediate progress, pre-success and final camera views.
        wanted=sorted({frames[0],frames[len(frames)//3],frames[2*len(frames)//3],frames[-1]})
        canvas=Image.new('RGB',(672,250*len(wanted)),'white');draw=ImageDraw.Draw(canvas)
        for i,t in enumerate(wanted):
            draw.text((4,i*250+4),directory.name+f' control={t}',fill='black')
            for col,camera in enumerate(('head','left_wrist','right_wrist')):
                path=directory/f'control-{t:06d}-{camera}.png'
                with Image.open(path) as original:canvas.paste(original.convert('RGB').resize((224,224)),(col*224,i*250+23))
        sheet=a.output/(directory.name+'.jpg');canvas.save(sheet,quality=95,subsampling=0)
        audits.append(dict(run=str(directory),result_sha256=file_sha(directory/'result.json'),
            ledger_sha256=file_sha(directory/'reward-ledger.jsonl'),actual_policy_controls=len(rows),
            success=receipt['learned_policy_success'],final_physical=rows[-1]['physical_evidence'],
            last_six=[r['measurement']['achieved'] for r in rows[-6:]],sheet=str(sheet),
            sheet_sha256=file_sha(sheet),reviewed_by_human=False))
    (a.output/'audit.json').write_text(json.dumps(audits,indent=2)+'\n')
    print(json.dumps(audits))


if __name__=='__main__':main()
