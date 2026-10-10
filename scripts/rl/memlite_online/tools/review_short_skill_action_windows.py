"""Review every native boundary of an audited RL trajectory, not auto-BC.

The old short-skill recorder has dense actions but only boundary proprio/RGB.
Never invent per-control observations or relabel learner actions as an expert.
This tool emits immutable review evidence and exact 32-action window metadata;
it neither exports/adopts training examples nor drops failed ledger episodes.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import sys

REPO=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(REPO/'scripts/rl/memlite_online/code'))
from recovery_corpus import canonical,digest,file_sha
from skill_aligned_reward import SkillIdentity,SkillReward,skill_measurement
from skill_observation_archive import load_observation


def action_windows(controls,observations):
    """Observation at s[t] pairs with ACKs t+1 through t+32, not t..t+31."""
    by_step={r['control_step']:r for r in controls}
    if len(by_step)!=len(controls) or not controls:
        raise ValueError('Unique real control sequence required')
    steps=[r['control_step'] for r in controls]
    if steps!=list(range(steps[0],steps[-1]+1)):
        raise ValueError('Nonconsecutive applied controls')
    clocks=[r['control_step'] for r in observations]
    if not clocks or len(set(clocks))!=len(clocks) or clocks!=sorted(clocks) or clocks[0]!=steps[0]-1 or clocks[-1]!=steps[-1]:
        raise ValueError('Missing, duplicated or shifted boundary observation')
    result=[]
    for obs in observations:
        t=obs['control_step']
        future=[by_step.get(s) for s in range(t+1,t+33)]
        complete=all(r is not None for r in future)
        if complete:
            if any(r['simulator_apply_ack'] is not True for r in future):
                raise ValueError('Unacknowledged command')
            if any(r['skill_reward']['terminated'] or r['skill_reward']['truncated'] for r in future[:-1]):
                raise ValueError('Window crosses a real terminal or truncation')
            if len({r['policy_version'] for r in future})!=1:
                raise ValueError('Policy changed inside one window')
        result.append(dict(observation_control_step=t,complete_32_applied_actions=complete,
            first_action_ack=t+1 if complete else None,last_action_ack=t+32 if complete else None,
            action_sha256=digest([r['action_executed_raw23'] for r in future]) if complete else None,
            observation_sha256=obs['observation_sha256'],admitted_for_training=False))
    return result


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for key in ('config','audit','output'):ap.add_argument('--'+key,type=Path,required=True)
    a=ap.parse_args()
    if a.output.exists() or subprocess.check_output(['git','status','--porcelain'],cwd=REPO,text=True).strip():
        raise ValueError('New output and frozen clean source required')
    cfg=json.loads(a.config.read_text());audit=json.loads(a.audit.read_text())
    if (audit['status']!='diagnostic_closed_trajectory_audit_not_baseline'
            or audit['config_sha256']!=file_sha(a.config) or audit['optimizer_authorized'] is not False):
        raise ValueError('Require the exact previously recomputed physical/observation audit')
    from PIL import Image,ImageDraw
    a.output.mkdir(parents=True);reports=[]
    for proof in audit['episodes']:
        directory=Path(proof['run']);case=cfg['cases'][proof['case']]
        if case['original_split']!='train' or case['recovery_split']!='train':
            raise ValueError('Do not use heldout/protected sources for learner-action candidates')
        for name,key in [('result.json','result_sha256'),('controls.jsonl','controls_sha256')]:
            if file_sha(directory/name)!=proof[key]:raise ValueError('Changed actual episode')
        controls=[json.loads(x) for x in (directory/'controls.jsonl').read_text().splitlines()]
        result=json.loads((directory/'result.json').read_text())
        obs_root=directory/'observations';archive=proof['observation_archive']
        for name,key in [('manifest.json','manifest_sha256'),('observations.jsonl','observations_sha256')]:
            if file_sha(obs_root/name)!=archive[key]:raise ValueError('Changed native observation archive')
        observations=[json.loads(x) for x in (obs_root/'observations.jsonl').read_text().splitlines()]
        windows=action_windows(controls,observations)
        start=json.loads((directory/'start.json').read_text());initial=start['initial_physical_evidence']
        identity=SkillIdentity(**result['identity']);skill=json.loads(case['semantic_bundle'])[0]
        if (start['identity']!=asdict(identity) or start['job']!=result['job']
                or result['policy_sha256']!=audit['policy_sha256']):
            raise ValueError('Changed original initial evidence or model identity')
        # The older audit did not bind start.json bytes separately. Recompute
        # all rewards again so changing the initial physical state cannot
        # silently change reported progress; pin these exact bytes below.
        reward=SkillReward(identity,skill_measurement(skill,initial),control_step=case['start_control'])
        for row in controls:
            t=row['control_step']
            measured=skill_measurement(skill,row['physical_evidence'])
            value=reward.advance(identity,t,measured,protected_values={},official_terminal=row['official_terminal'],
                time_limit=row['official_truncated'] or t==result['policy_end_control'])
            if dict(value,identity=asdict(identity))!=row['skill_reward']:
                raise ValueError('Actual initial/control physics no longer reconstructs rewards')
        physical={controls[0]['control_step']-1:initial,**{r['control_step']:r['physical_evidence'] for r in controls}}
        sheets=[]
        for page,begin in enumerate(range(0,len(observations),6)):
            selected=observations[begin:begin+6]
            canvas=Image.new('RGB',(672,250*len(selected)),'white');draw=ImageDraw.Draw(canvas)
            for i,record in enumerate(selected):
                obs=load_observation(obs_root,record);t=record['control_step']
                phi=skill_measurement(skill,physical[t])['potential']
                windows[begin+i]['measured_potential_at_observation']=phi
                if windows[begin+i]['complete_32_applied_actions']:
                    end=skill_measurement(skill,physical[t+32])['potential']
                    windows[begin+i]['potential_change_after_actual_32_controls']=end-phi
                draw.text((4,i*250+4),f"{proof['case']} seed{proof['seed']} s[{t}] potential={phi:.5f}",fill='black')
                for col,camera in enumerate(('head_rgb','left_wrist_rgb','right_wrist_rgb')):
                    pixels=obs['images'][camera].transpose(1,2,0)
                    canvas.paste(Image.fromarray(pixels).resize((224,224)),(224*col,250*i+23))
            name=f"{proof['job_id'][:12]}-page-{page:03d}.png";canvas.save(a.output/name)
            sheets.append(dict(path=name,sha256=file_sha(a.output/name),steps=[r['control_step'] for r in selected]))
        reports.append(dict(job_id=proof['job_id'],case=proof['case'],seed=proof['seed'],
            original_run=str(directory),actual_controls=len(controls),original_split='train',recovery_split='train',
            source_start_sha256=file_sha(directory/'start.json'),source_controls_sha256=proof['controls_sha256'],
            source_result_sha256=proof['result_sha256'],
            action_source='learner_stochastic_fm_not_expert',physical_final_success=proof['success'],
            source_policy_sha256=audit['policy_sha256'],windows=windows,sheets=sheets,
            human_review='pending',admitted_for_training=False))
    report=dict(schema='native_rl_boundary_action_review_v1',source_commit=subprocess.check_output(
        ['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),config_sha256=file_sha(a.config),
        audit_path=str(a.audit),audit_sha256=file_sha(a.audit),episodes=reports,
        model_or_optimizer_invoked=False,constructed_dense_observations=False,
        training_admission=False,physical_failure_denominators_unchanged=True)
    with (a.output/'index.json').open('x') as stream:stream.write(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(status='review_material_only',episodes=len(reports),
        observations=sum(len(r['windows']) for r in reports),index_sha256=file_sha(a.output/'index.json'))))


if __name__=='__main__':main()
