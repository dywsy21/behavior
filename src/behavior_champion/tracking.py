"""Online-only W&B lifecycle, with credentials kept outside code and receipts."""
import json
import os
from pathlib import Path
import stat


class OnlineRun:
    def __init__(self, directory, *, credential_file, entity, project, name, config,
                 resume_id=None, resume_update=None):
        credential = Path(credential_file)
        if credential.is_symlink() or stat.S_IMODE(credential.stat().st_mode) != 0o600:
            raise ValueError('Expected a private mode-0600 credential file, not a symlink')
        key = credential.read_text().strip()
        if not key or '\n' in key:
            raise ValueError('Invalid private W&B credential')
        self.directory = Path(directory).resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        os.environ['WANDB_API_KEY'] = key
        # A previously inherited disabled/offline setting must not silently turn
        # a requested monitored training into an unobserved local run.
        os.environ['WANDB_MODE'] = 'online'
        import wandb
        self.run = wandb.init(entity=entity, project=project, name=name,
            id=resume_id, resume='must' if resume_id else 'never', mode='online',
            dir=str(self.directory), config=config,
            settings=wandb.Settings(init_timeout=90))
        if self.run is None or self.run.offline or self.run.entity != entity or self.run.project != project:
            raise RuntimeError('W&B online team/project identity was not established')
        self.run.define_metric('train/completed_updates')
        self.run.define_metric('train/*', step_metric='train/completed_updates')
        self.run.define_metric('eval/*', step_metric='train/completed_updates')
        self.identity = dict(entity=self.run.entity, project=self.run.project,
                             run_id=self.run.id, url=self.run.url, mode='online')
        if resume_id is not None:
            if self.run.id != resume_id or resume_update is None or resume_update < 0:
                raise ValueError('Exact run and checkpoint step required for online resume')
            self.run.log({'lifecycle/event': 'checkpoint_restored',
                          'lifecycle/restored_update': resume_update})

    def log(self, metrics, *, completed):
        if completed < 0 or any(not isinstance(key, str) for key in metrics):
            raise ValueError('Invalid training metrics')
        # W&B's append-only history step is NOT the optimizer step. On a crash,
        # cloud history may be ahead of the last durable checkpoint; making the
        # optimizer step a metric preserves the visible replay instead of
        # silently dropping resumed metrics with an older internal `_step`.
        self.run.log({'train/completed_updates': completed, **metrics})

    def finish(self, status, *, completed, error=False):
        self.run.summary.update({'terminal_status': status, 'completed_updates': completed})
        self.run.finish(exit_code=1 if error else 0)
        return dict(**self.identity, terminal_status=status, completed_updates=completed)
