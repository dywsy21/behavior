"""Bind the owner's explicit exact-branch reviews to separate objective rows.

This tool does not select approvals itself. No whole-clip promotion, on-policy
claim, injected-action BC, or cross-branch reuse of a review is allowed.
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
from recovery_corpus import digest,file_sha
from recovery_teacher_corpus import physical_proposal,validate_branch
from recovery_terminal_corpus import load_terminal


def verify_action_window(t, rows, manifest, anchor, reviewed_frames):
    if (len(rows[t:t+32])!=32 or any(r['label_kind']!='same_state_local_teacher_candidate' for r in rows[t:t+32])
            or not anchor['label_audit']['full_executed_32_step_target_available']
            or not {s for s in map(int,manifest['anchors']) if t<=s<=t+32}<=set(reviewed_frames)
            or min(reviewed_frames)>t or max(reviewed_frames)<t+32):
        raise ValueError('Missing real clean actions or full 32-control media review')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('corpus','decisions','evidence-root','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    root=a.evidence_root.resolve();decisions=json.loads(a.decisions.read_text())
    if decisions['schema']!='owner_local_recovery_review_v1' or not decisions['reviewer']:
        raise ValueError('Missing explicit owner review')
    queue=json.loads((a.corpus/'review-queue.json').read_text())
    inventory=json.loads((a.corpus/'audit/inventory.json').read_text())
    anchors=[json.loads(x) for x in (a.corpus/'audit/anchors.jsonl').read_text().splitlines()]
    approvals=[]
    for decision in decisions['branches']:
        choices=[q for q in queue if q['case']==decision['case'] and q['branch']==decision['branch']]
        if len(choices)!=1:raise ValueError('Review does not uniquely identify a real collected branch')
        q=choices[0];branch=Path(q['source_path']);manifest=json.loads((branch/'manifest.json').read_text())
        if file_sha(branch/'manifest.json')!=decision['manifest_sha256']:raise ValueError('Review from another attempt')
        review=root/decision['review_directory'];materials=json.loads((review/'review.json').read_text())
        if (materials['manifest_sha256']!=decision['manifest_sha256']
                or materials['transitions_sha256']!=manifest['transitions_sha256']):raise ValueError('Unbound original media review')
        for s in materials['sheets']:
            if file_sha(review/s['path'])!=s['sha256']:raise ValueError('Changed reviewed media')
        rows=[json.loads(x) for x in (branch/'transitions.jsonl').read_text().splitlines()]
        plans=json.loads((branch/'plans.json').read_text());validate_branch(rows,manifest,plans)
        terminal=load_terminal(branch,manifest,rows,plans)
        refs=[dict(path=str((review/s['path']).resolve().relative_to(root)),sha256=s['sha256'],kind='original_media_review')
              for s in materials['sheets']]
        refs.append(dict(path=str((review/'review.json').resolve().relative_to(root)),
                         sha256=file_sha(review/'review.json'),kind='physical_semantic_review'))
        refs.append(dict(path=str(a.decisions.resolve().relative_to(root)),sha256=file_sha(a.decisions),kind='owner_decision'))
        indexed={r['control_step']:r for r in anchors if r['actor_input']['rgb']['archive']==q['archive']}
        event=digest([q['source_group'],'first_demonstrated_grasp'])
        def approve(t,pool,label):
            row=indexed[t]
            if t not in materials['frames']:raise ValueError('Exact observation not visually inspected')
            approvals.append(dict(sample_id=row['sample_id'],pool=pool,reviewer=decisions['reviewer'],
                evidence=refs,event_id=event,reviewed_start=min(materials['frames']),reviewed_end=max(materials['frames']),
                source_group=q['source_group'],label=label))
        for outcome in decision['outcomes']:
            t=outcome['control_step'];latest=[p for p in plans if p['control_step']<=t][-1]
            if terminal is not None and t==terminal['control_step']:
                if (materials.get('terminal_observation_sha256')!=manifest['terminal_observation_sha256']
                        or not indexed[t]['actor_input'].get('observation_only')):
                    raise ValueError('Missing actual final-observation review/corpus evidence')
                measured=terminal['outcome_candidate']
            else:measured=physical_proposal(rows,t,manifest['arm'],attempt_start=latest['control_step'])
            if measured!=outcome['value']:raise ValueError('Owner outcome contradicts causal physical evidence')
            approve(t,'outcome',dict(value=measured,member_index=0,available_control_step=t,evidence_end_control_step=t))
        if decision['action_steps'] or decision['planner_steps']:
            if (not manifest['physical_recovery_candidate'] or manifest['failure'] is not None
                    or not decision['corrective_execution_visually_verified']):
                raise ValueError('No verified real successful correction for positive action/plan')
        for t in decision['action_steps']:
            verify_action_window(t,rows,manifest,indexed[t],materials['frames'])
            approve(t,'action',dict(quality='verified_correct_execution',executed_controls=32))
        for t in decision['planner_steps']:
            target=a.corpus/'proposed-plans'/(indexed[t]['sample_id']+'.json')
            if not any(e['control_step']==t and e['decision']=='RETRY' for e in plans):raise ValueError('No actual RETRY issued')
            approve(t,'planner',dict(verified_plan_path=str(target.resolve().relative_to(root)),
                verified_plan_sha256=file_sha(target),kind='verified_recovery_continuation'))
    output=dict(schema='recovery_sample_approvals_v1',inventory_sha256=digest(inventory),
        anchors_sha256=file_sha(a.corpus/'audit/anchors.jsonl'),approvals=approvals,
        review_note='Exact physically executed offline local corrections. Not full-task success or on-policy policy improvement.')
    a.output.write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps(dict(approvals=len(approvals),output=str(a.output),sha256=file_sha(a.output))))


if __name__=='__main__':main()
