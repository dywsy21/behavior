"""Normalize real offline teacher branches into UNAPPROVED training candidates.

No generated approval. The output deliberately distinguishes simulated teacher
corrections from previous on-policy RL candidates. Reviewers sign exact rows.
"""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import zipfile

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
from recovery_corpus import canonical,digest,file_sha,group_key,split_group,anchor_candidate
from recovery_teacher_corpus import validate_branch,branch_histories,physical_proposal,episode_identity


def write_json(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def write_lines(path,rows):path.write_text(''.join(json.dumps(r,sort_keys=True,allow_nan=False)+'\n' for r in rows))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--collection',type=Path,required=True);p.add_argument('--sources',type=Path,required=True)
    p.add_argument('--protected',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--previous-collection',type=Path)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    protected=set(json.loads(a.protected.read_text())['groups']);source_rows=[]
    inventory=[];anchors=[];histories=[];bindings={};queue=[];excluded=[]
    a.output.mkdir(parents=True)
    for folder in ('raw','audit','history','proposed-plans'):(a.output/folder).mkdir()
    for entry in json.loads((a.sources/'manifest.json').read_text())['cases']:
        proposal=a.sources/entry['directory'];directory=a.collection/entry['directory']
        previous=a.previous_collection/entry['directory'] if a.previous_collection else None
        if previous is not None and (previous/'result.json').exists():
            # Keep the FIRST recorded attempt, including its failures. Do not
            # select a luckier repetition or count warm-engineering duplicates.
            directory=previous
        if not (directory/'result.json').exists():
            excluded.append(dict(case=entry['directory'],reason='No completed collection receipt'));continue
        source=json.loads((proposal/'manifest.json').read_text())
        if (file_sha(proposal/'manifest.json')!=entry['manifest_sha256'] or source['source_group'] in protected
                or source['source_episode']['split']!='train' or source['protected_groups_sha256']!=file_sha(a.protected)
                or json.loads((directory/'source.json').read_text())!=source):
            raise ValueError('Unverified source / protected instance / changed copied source')
        for name,sha in source['files'].items():
            if Path(name).name!=name or file_sha(proposal/name)!=sha:raise ValueError('Changed original source')
        result=json.loads((directory/'result.json').read_text())
        if result['status']!='completed_candidates_only':
            excluded.append(dict(case=entry['directory'],reason=result['status']));continue
        if file_sha(directory/'full_snapshot.pt')!=result['full_snapshot_sha256']:
            raise ValueError('Changed complete state snapshot')
        seed=json.loads((directory/'seed.json').read_text())
        prefix=[json.loads(x) for x in (directory/'prefix.jsonl').read_text().splitlines()]
        if (len(prefix)!=seed['source_frame'] or len(prefix)<6 or
                not all(r['physical_audit']['grasp'][seed['arm']]=='TRUE' for r in prefix[-6:])):
            raise ValueError('Seed was not physically verified as same-target same-arm grasp')
        source_rows.append(dict(case=entry['directory'],source_manifest_sha256=entry['manifest_sha256'],
            collection_result_sha256=file_sha(directory/'result.json'),source_commit=result['source_commit'],
            original_release_sha256=source['original_release_sha256'],source_group=source['source_group'],
            full_snapshot_sha256=result['full_snapshot_sha256'],prefix_sha256=file_sha(directory/'prefix.jsonl')))
        for branch in result['branches']:
            branch_path=directory/branch['kind'];manifest=json.loads((branch_path/'manifest.json').read_text())
            if manifest!=branch:raise ValueError('Changed finalized branch')
            if (file_sha(branch_path/'transitions.jsonl')!=manifest['transitions_sha256']
                    or file_sha(branch_path/'plans.json')!=manifest['plans_sha256']):raise ValueError('Changed physical branch')
            rows=[json.loads(x) for x in (branch_path/'transitions.jsonl').read_text().splitlines()]
            plans=json.loads((branch_path/'plans.json').read_text())
            if not rows:
                excluded.append(dict(case=entry['directory'],branch=branch['kind'],reason=branch['failure']));continue
            validate_branch(rows,manifest,plans)
            episode=episode_identity(source,result,manifest,branch['kind'])
            clip_id=digest(episode)[:24];relative=clip_id+'.zip'
            binding={seed['physics']['target_name']:seed['physics']['entity']}
            bindings[canonical([episode['run'],episode['episode_id']])]=binding
            normalized=deepcopy(rows)
            for row in normalized:
                audit=row['physical_audit'];audit['grasp_states']={audit['entity']:audit['grasp']}
                audit['entity_bindings']=binding
            raw=''.join(json.dumps(r,sort_keys=True,allow_nan=False)+'\n' for r in normalized).encode()
            header_anchors={k:v for k,v in manifest['anchors'].items() if int(k)<len(rows)}
            header=dict(schema='recovery_teacher_candidate_v2',clip_id=clip_id,label_kind='offline_local_teacher_candidate',
                bc_eligible=False,human_review='pending',episode=episode,provenance=source_rows[-1],
                start_control_step=0,end_control_step=len(rows),transitions_sha256=hashlib.sha256(raw).hexdigest(),
                rgb_anchors=header_anchors,branch_evidence=manifest,plans=plans,
                excluded_unapplied_anchors=sorted(set(manifest['anchors'])-set(header_anchors),key=int),
                events=[dict(kind='offline_intervention_and_correction_candidate',physical_recovery_candidate=branch['physical_recovery_candidate'])],
                action_mapping='raw23 base/trunk/left/gripper/right/gripper; no masked controls',
                policy_memory='fresh isolated curriculum start; not an inherited full-task history')
            with zipfile.ZipFile(a.output/'raw'/relative,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=3) as archive:
                archive.writestr('manifest.json',json.dumps(header,indent=2,allow_nan=False))
                archive.writestr('transitions.jsonl',raw)
                for t,anchor in manifest['anchors'].items():
                    if int(t)>=len(rows):continue
                    for camera,checksum in anchor['sha256'].items():
                        name=f'rgb/{int(t):08d}/{camera}.jpg';path=branch_path/name
                        if file_sha(path)!=checksum:raise ValueError('Changed original observed RGB')
                        archive.write(path,name,compress_type=zipfile.ZIP_STORED)
            # An attempted action may fail before its first complete control;
            # never expose a dangling pre-action frame as an applied sample.
            item=dict(path=relative,sha256=file_sha(a.output/'raw'/relative),bytes=(a.output/'raw'/relative).stat().st_size,
                episode=episode,controls=len(rows),images=3*len(header_anchors),events=header['events'],
                structural_validation='passed',split=split_group(source['task'],source['instance_id'],protected))
            inventory.append(item);mapping={r['control_step']:r for r in normalized};branch_anchors=[]
            for t,anchor in sorted(header_anchors.items(),key=lambda x:int(x[0])):
                t=int(t);row=anchor_candidate(episode,mapping,t,dict(archive=relative,sha256=anchor['sha256'],control_step=t),binding,item['split'])
                latest=[p for p in plans if p['control_step']<=t][-1]
                proposal_label=physical_proposal(rows,t,seed['arm'],attempt_start=latest['control_step'])
                row['label_audit']['offline_teacher']=dict(kind=branch['kind'],action_label_kind=rows[t]['label_kind'],
                    physical_recovery_candidate=branch['physical_recovery_candidate'],not_on_policy=True,
                    source_event_id=digest([source['source_group'],'first_demonstrated_grasp']),
                    branch_failure=branch['failure'])
                row['label_audit']['outcome'].update(value=proposal_label,reason='causal_offline_physical_proposal_pending_review',
                    evidence_end_control_step=t)
                if any(r['label_kind']!='same_state_local_teacher_candidate' for r in rows[t:t+32]):
                    row['label_audit']['full_executed_32_step_target_available']=False
                branch_anchors.append(row);anchors.append(row)
            context=branch_histories(branch_anchors,plans);histories.extend(context)
            for row,h in zip(branch_anchors,context):
                if h['predecision'] is not None and h['observable']['issued_decision']=='RETRY':
                    target=dict(schema='recovery_verified_plan_v1',sample_id=row['sample_id'],source_episode=row['source_episode'],
                        control_step=row['control_step'],parent_goal=h['observable']['parent_goal'],
                        active_skills_semantic_json=h['observable']['issued_skills_semantic_json'],
                        decision=h['observable']['issued_decision'],memory_update=h['observable']['memory'])
                    write_json(a.output/'proposed-plans'/(row['sample_id']+'.json'),target)
            queue.append(dict(source_group=source['source_group'],split=item['split'],case=entry['directory'],branch=branch['kind'],
                source_path=str(branch_path),archive=relative,arm=seed['arm'],physical_recovery_candidate=branch['physical_recovery_candidate'],
                proposed_outcomes={v:sum(r['label_audit']['outcome']['value']==v for r in branch_anchors)
                                   for v in ('SUCCEEDED','FAILED','IN_PROGRESS')},failure=branch['failure']))
    write_json(a.output/'audit/inventory.json',inventory);write_lines(a.output/'audit/anchors.jsonl',anchors)
    write_json(a.output/'source-manifest.json',source_rows);write_json(a.output/'bindings.json',bindings)
    summary=dict(schema='recovery_corpus_audit_v1',source_manifest_supplied=True,protected_groups_supplied=True,
        bindings_sha256=file_sha(a.output/'bindings.json'),source_manifest_sha256=file_sha(a.output/'source-manifest.json'),
        protected_groups_sha256=file_sha(a.protected),conflicting_evidence=[],quarantined=[],
        inventory_sha256=digest(inventory),archives_seen=len(inventory),unique_visual_anchors=len(anchors),
        tasks=len({r['task'] for r in anchors}),source_groups=len({r['source_group'] for r in anchors}),
        source_kind='offline_local_teacher_candidates_not_on_policy',training_ready=False,excluded=excluded)
    write_json(a.output/'audit/summary.json',summary);write_lines(a.output/'history/contexts.jsonl',histories)
    write_json(a.output/'history/receipt.json',dict(inventory_sha256=digest(inventory),
        contexts_sha256=file_sha(a.output/'history/contexts.jsonl'),anchors_sha256=file_sha(a.output/'audit/anchors.jsonl')))
    write_json(a.output/'review-queue.json',queue)
    print(json.dumps(dict(**summary,review_queue=len(queue)),indent=2))


if __name__=='__main__':main()
