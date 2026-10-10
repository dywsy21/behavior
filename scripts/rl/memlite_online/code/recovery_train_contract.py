"""CPU gates shared by the launcher and actual recovery trainer."""
import json
from pathlib import Path

from recovery_corpus import file_sha
from recovery_sft_data import require_training_pool


def validate_launch(ticket_path,component,*,engineering=False):
    ticket_path=Path(ticket_path);ticket=json.loads(ticket_path.read_text())
    if ticket.get('schema')!='recovery_sft_launch_v1' or ticket['component']!=component:
        raise ValueError('Wrong explicit launch ticket')
    if not engineering and ticket.get('formal_training_authorized') is not True:
        raise ValueError('Formal training not authorized by this preparation ticket')
    if engineering and ticket.get('engineering_smoke') is not True:
        raise ValueError('No engineering-only ticket')
    for key,entry in ticket['files'].items():
        if file_sha(entry['path'])!=entry['sha256']: raise ValueError('Changed pinned launch input: '+key)
    recipe=json.loads(Path(ticket['files']['recipe']['path']).read_text())
    if component=='H1' and recipe.get('H1',{}).get('feedback')=='frozen_source_disjoint_observer_v1':
        from recovery_postfit_feedback import validate_training_release
        validate_training_release(ticket,recipe)
        processor=json.loads(Path(ticket['files']['processor_audit']['path']).read_text())
        if (processor.get('schema')!='accepted_recovery_processor_audit_v1' or processor.get('status')!='passed'
                or processor.get('source_commit')!=ticket['source_commit']
                or processor.get('recipe_sha256')!=ticket['files']['recipe']['sha256']
                or processor.get('feedback_sha256')!=ticket['files']['feedback']['sha256']
                or processor.get('admission_sha256')!=ticket['files']['admission']['sha256']
                or processor.get('optimizer_steps')!=0 or processor.get('oracle_inputs') is not False):
            raise ValueError('Fixed feedback requires actual all-row processor acceptance')
    if component=='H1' and recipe.get('H1',{}).get('original_feedback')=='causal_expert_unknown_v1':
        if not {'expert_feedback_audit','expert_feedback_human_review'}<=ticket['files'].keys():
            raise ValueError('New expert feedback needs actual processor and human visual acceptance')
        audit=json.loads(Path(ticket['files']['expert_feedback_audit']['path']).read_text())
        review=json.loads(Path(ticket['files']['expert_feedback_human_review']['path']).read_text())
        if (audit.get('schema')!='expert_causal_feedback_audit_v1' or audit.get('status')!='AUTOMATED_PASSED'
                or audit.get('source_commit')!=ticket['source_commit']
                or audit.get('recipe_sha256')!=ticket['files']['recipe']['sha256']
                or audit.get('original_feedback_repeat_stride',16)!=recipe['H1'].get('original_feedback_repeat_stride',16)
                or audit.get('normal_noninitial_control_tasks')!=100 or audit.get('processed_raw_samples')!=24
                or audit.get('clocks_match_runtime') is not True or audit.get('targets_unchanged') is not True
                or audit.get('physical_outcomes_added')!=0):
            raise ValueError('Invalid new feedback audit')
        if (review.get('schema')!='expert_causal_feedback_human_review_v1' or review.get('decision')!='APPROVED'
                or review.get('audit_sha256')!=ticket['files']['expert_feedback_audit']['sha256']
                or review.get('rows_sha256')!=audit['rows_sha256'] or len(review.get('sheets',[]))!=24
                or len({r.get('sha256') for r in review['sheets']})!=24
                or any(not r.get('note') or r.get('decision')!='APPROVED' for r in review['sheets'])):
            raise ValueError('Human feedback/image review is missing or not bound to this audit')
    if ticket['event_passes'] > 5 and (recipe.get('extended_event_fit') is not True
            or recipe.get('training_authorized_by_this_file') is not True or not recipe.get('authorization')):
        raise ValueError('Extended event fitting requires a separately pinned authorized recipe')
    if (ticket['maximum_updates']<1 or ticket['maximum_updates']>recipe['maximum_updates_per_line']
            or not 1<=ticket['event_passes']<=recipe['maximum_event_passes']
            or not 60<=ticket['wall_seconds']<=recipe['maximum_wall_seconds_per_line']
            or ticket['world_size'] not in (1,8) or ticket['micro_batch']<1):
        raise ValueError('Expanded/invalid preparation budget')
    if engineering:
        if (ticket['maximum_updates']>2 or ticket['wall_seconds']>1800 or component not in ('H1','L0')
                or ticket['formal_training_authorized'] or ticket['world_size']!=8
                or ticket['micro_batch']!=1 or ticket.get('global_batch')!=16):
            raise ValueError('Engineering is two original-expert updates at most, not recovery SFT')
    else:
        pool=recipe[component]['pool']
        approved=require_training_pool(ticket['admission'],pool,ticket['files']['admission']['sha256'])
        if component=='L0':
            from recovery_native_actions import validate_native_training_supplement
            validate_native_training_supplement(ticket,recipe,[r for r in approved[1] if r['candidate']['split']=='train'])
        if component=='H0' and ticket['world_size']!=1: raise ValueError('Tiny cached observer uses one worker')
        if component=='H1' and not {'history','feedback','feedback_receipt'}<=ticket['files'].keys():
            raise ValueError('H1 needs real causal history and OOF prediction artifacts')
    return ticket,recipe


def validate_h0_event_budget(counts, ticket):
    """The combined OOF + final fit must fit this ticket, not only the ceiling."""
    if not counts or max(counts.values()) > min(5, ticket['event_passes']):
        raise ValueError('Combined H0 event exposures exceed the explicit ticket budget')
