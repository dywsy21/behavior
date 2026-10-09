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
from recovery_corpus import canonical,digest,file_sha,split_group,anchor_candidate,same_intent_starts
from recovery_teacher_corpus import validate_branch,branch_histories,physical_proposal,episode_identity,PhysicalProposalIndex
from recovery_reference_binding import verify_saved_binding
from recovery_terminal_corpus import load_terminal,normalized_terminal,terminal_anchor
from recovery_articulation_corpus import verify_articulation_seed,validate_articulation_branch,CLEAN


def write_json(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def write_lines(path,rows):path.write_text(''.join(json.dumps(r,sort_keys=True,allow_nan=False)+'\n' for r in rows))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--collection',type=Path,required=True);p.add_argument('--sources',type=Path,required=True)
    p.add_argument('--protected',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--previous-collection',type=Path,action='append',default=[])
    p.add_argument('--mechanism',choices=('grasp','articulation'),default='grasp')
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    protected=set(json.loads(a.protected.read_text())['groups']);source_rows=[]
    inventory=[];anchors=[];histories=[];bindings={};queue=[];excluded=[]
    a.output.mkdir(parents=True)
    for folder in ('raw','audit','history','proposed-plans'):(a.output/folder).mkdir()
    for entry in json.loads((a.sources/'manifest.json').read_text())['cases']:
        proposal=a.sources/entry['directory'];directory=a.collection/entry['directory']
        priors=[root/entry['directory'] for root in a.previous_collection
                if (root/entry['directory']/'result.json').exists() or (root/entry['directory']/'status.json').exists()]
        if len(priors)>1:raise ValueError('Conflicting first attempts; cannot select a preferred result')
        previous=priors[0] if priors else None
        if previous is not None:
            # Keep the FIRST recorded attempt, including its failures. Do not
            # select a luckier repetition or count warm-engineering duplicates.
            directory=previous
        if not (directory/'result.json').exists():
            state=json.loads((directory/'status.json').read_text()) if (directory/'status.json').exists() else {}
            excluded.append(dict(case=entry['directory'],reason=state.get('status','No completed collection receipt'),
                original_error=state.get('error'),receipt_sha256=file_sha(directory/'status.json') if state else None));continue
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
        articulation=a.mechanism=='articulation'
        if articulation:
            verified_binding=verify_articulation_seed(directory,source,result,prefix,seed)
            target_physics=seed['physical'];arm='not_applicable';clean_label=CLEAN
        else:
            verified_binding=verify_saved_binding(directory,source,result,prefix)
            if (len(prefix)!=seed['source_frame'] or len(prefix)<6 or
                    not all(r['physical_audit']['grasp'][seed['arm']]=='TRUE' for r in prefix[-6:])):
                raise ValueError('Seed was not physically verified as same-target same-arm grasp')
            target_physics=seed['physics'];arm=seed['arm'];clean_label='same_state_local_teacher_candidate'
        source_rows.append(dict(case=entry['directory'],source_manifest_sha256=entry['manifest_sha256'],
            collection_result_sha256=file_sha(directory/'result.json'),source_commit=result['source_commit'],
            original_release_sha256=source['original_release_sha256'],source_group=source['source_group'],
            full_snapshot_sha256=result['full_snapshot_sha256'],prefix_sha256=file_sha(directory/'prefix.jsonl')))
        if articulation and any('simulator_apply_ack' not in r for r in prefix):
            source_rows[-1]['prefix_proof']='reviewed_legacy_nonterminal_after_apply_'+result['source_commit']
        if verified_binding:
            source_rows[-1]['reference_binding_sha256']=file_sha(directory/'reference-binding.json')
            source_rows[-1]['binding_reference_sha256']=file_sha(directory/'binding-reference.jsonl')
        for branch in result['branches']:
            branch_path=directory/branch['kind'];manifest=json.loads((branch_path/'manifest.json').read_text())
            if manifest!=branch:raise ValueError('Changed finalized branch')
            if (file_sha(branch_path/'transitions.jsonl')!=manifest['transitions_sha256']
                    or file_sha(branch_path/'plans.json')!=manifest['plans_sha256']):raise ValueError('Changed physical branch')
            rows=[json.loads(x) for x in (branch_path/'transitions.jsonl').read_text().splitlines()]
            plans=json.loads((branch_path/'plans.json').read_text())
            if verified_binding and any(json.loads(plan['active_skills_semantic_json'])!=verified_binding['resolved_skills'] for plan in plans):
                raise ValueError('Corrective branch intent does not match the actually bound reference target')
            if not rows:
                excluded.append(dict(case=entry['directory'],branch=branch['kind'],reason=branch['failure']));continue
            if articulation:
                outcome_proposals,predecision_proposals=validate_articulation_branch(rows,manifest,plans,seed)
                terminal=None  # This immutable collector has no separate final RGB.
            else:
                validate_branch(rows,manifest,plans)
                terminal=load_terminal(branch_path,manifest,rows,plans)
            episode=episode_identity(source,result,manifest,branch['kind'])
            if articulation:episode['teacher_kind']='offline_local_measured_reference_path_servo_v1'
            clip_id=digest(episode)[:24];relative=clip_id+'.zip'
            binding={target_physics['target_name']:target_physics['entity']}
            bindings[canonical([episode['run'],episode['episode_id']])]=binding
            normalized=deepcopy(rows)
            for row in normalized:
                audit=row['physical_audit'];audit['grasp_states']={audit['entity']:audit['grasp']}
                audit['entity_bindings']=binding
                if articulation:row['policy_update']=0
            raw=''.join(json.dumps(r,sort_keys=True,allow_nan=False)+'\n' for r in normalized).encode()
            header_anchors={k:v for k,v in manifest['anchors'].items() if int(k)<len(rows)}
            if terminal:
                t=terminal['control_step']
                header_anchors[str(t)]=dict(control_step=t,observation_only=True,
                    sha256={k:v['sha256'] for k,v in terminal['images'].items()},
                    dimensions={k:[3,224,224] for k in terminal['images']})
            header=dict(schema='recovery_teacher_candidate_v2',clip_id=clip_id,label_kind='offline_local_teacher_candidate',
                bc_eligible=False,human_review='pending',episode=episode,provenance=source_rows[-1],
                start_control_step=0,end_control_step=len(rows),transitions_sha256=hashlib.sha256(raw).hexdigest(),
                rgb_anchors=header_anchors,branch_evidence=manifest,plans=plans,
                excluded_unapplied_anchors=sorted(set(manifest['anchors'])-set(header_anchors),key=int),
                events=[dict(kind='offline_intervention_and_correction_candidate',physical_recovery_candidate=branch['physical_recovery_candidate'])],
                action_mapping='raw23 base/trunk/left/gripper/right/gripper; no masked controls',
                policy_memory='fresh isolated curriculum start; not an inherited full-task history')
            if terminal:header['terminal_observation']=normalized_terminal(terminal,binding)
            with zipfile.ZipFile(a.output/'raw'/relative,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=3) as archive:
                archive.writestr('manifest.json',json.dumps(header,indent=2,allow_nan=False))
                archive.writestr('transitions.jsonl',raw)
                for t,anchor in manifest['anchors'].items():
                    if int(t)>=len(rows):continue
                    for camera,checksum in anchor['sha256'].items():
                        name=f'rgb/{int(t):08d}/{camera}.jpg';path=branch_path/name
                        if file_sha(path)!=checksum:raise ValueError('Changed original observed RGB')
                        archive.write(path,name,compress_type=zipfile.ZIP_STORED)
                if terminal:
                    archive.write(branch_path/'terminal-observation.json','terminal-observation.original.json')
                    for camera,ref in terminal['images'].items():
                        archive.write(branch_path/ref['path'],f'rgb/{terminal["control_step"]:08d}/{camera}.jpg',
                                      compress_type=zipfile.ZIP_STORED)
            # An attempted action may fail before its first complete control;
            # never expose a dangling pre-action frame as an applied sample.
            item=dict(path=relative,sha256=file_sha(a.output/'raw'/relative),bytes=(a.output/'raw'/relative).stat().st_size,
                episode=episode,controls=len(rows),images=3*len(header_anchors),events=header['events'],
                structural_validation='passed',split=split_group(source['task'],source['instance_id'],protected))
            inventory.append(item);mapping={r['control_step']:r for r in normalized};branch_anchors=[]
            proposal_index=None if articulation else PhysicalProposalIndex(rows,arm)
            intent_starts=same_intent_starts(mapping)
            for t,anchor in sorted(header_anchors.items(),key=lambda x:int(x[0])):
                t=int(t);image_ref=dict(archive=relative,sha256=anchor['sha256'],control_step=t)
                if terminal and t==terminal['control_step']:
                    row=terminal_anchor(episode,terminal,image_ref,item['split'],branch)
                    branch_anchors.append(row);anchors.append(row);continue
                row=anchor_candidate(episode,mapping,t,image_ref,binding,item['split'],intent_starts=intent_starts)
                latest=[p for p in plans if p['control_step']<=t][-1]
                proposal_label=(outcome_proposals[t] if articulation else physical_proposal(
                    rows,t,arm,attempt_start=latest['control_step'],index=proposal_index))
                row['label_audit']['offline_teacher']=dict(kind=branch['kind'],action_label_kind=rows[t]['label_kind'],
                    physical_recovery_candidate=branch['physical_recovery_candidate'],not_on_policy=True,
                    source_event_id=digest([source['source_group'],'first_demonstrated_'+a.mechanism]),
                    branch_failure=branch['failure'])
                row['label_audit']['outcome'].update(value=proposal_label,reason='causal_offline_physical_proposal_pending_review',
                    evidence_end_control_step=t)
                if articulation and t in predecision_proposals:
                    row['label_audit']['predecision_outcome']=predecision_proposals[t]
                if any(r['label_kind']!=clean_label for r in rows[t:t+32]):
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
                source_path=str(branch_path),archive=relative,arm=arm,mechanism=a.mechanism,
                physical_recovery_candidate=branch['physical_recovery_candidate'],
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
