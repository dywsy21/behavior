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
        require_training_pool(ticket['admission'],pool,ticket['files']['admission']['sha256'])
        if component=='H0' and ticket['world_size']!=1: raise ValueError('Tiny cached observer uses one worker')
        if component=='H1' and not {'history','feedback','feedback_receipt'}<=ticket['files'].keys():
            raise ValueError('H1 needs real causal history and OOF prediction artifacts')
    return ticket,recipe


def validate_h0_event_budget(counts, ticket):
    """The combined OOF + final fit must fit this ticket, not only the ceiling."""
    if not counts or max(counts.values()) > min(5, ticket['event_passes']):
        raise ValueError('Combined H0 event exposures exceed the explicit ticket budget')
