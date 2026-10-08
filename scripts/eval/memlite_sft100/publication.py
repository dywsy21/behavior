"""Command provenance for legacy and administratively resumed evaluations."""
import json
from pathlib import Path


def evaluation_commands(job, manifest):
    if manifest.get('kind')!='native_sft100_admin_resume':
        return [json.loads(p.read_text()) for p in sorted((job/'tasks').glob('*/command.json'))]
    inherited=json.loads((job/'prior_commands.json').read_text())
    current=[dict(path=str(p),command=json.loads(p.read_text()))
             for p in sorted((job/'parts').glob('*/command.json'))]
    return ([row|dict(provenance='prior_stopped_run; completed and interrupted attempts retained')
             for row in inherited]+[row|dict(provenance='optimized_remaining_cases') for row in current])


def command_task_names(commands, manifest):
    if manifest.get('kind')=='native_sft100_admin_resume':
        return {row['command']['part']['task'] if 'part' in row['command']
                else Path(row['path']).parent.name for row in commands}
    return {row['argv'][row['argv'].index('--task')+1] for row in commands}
