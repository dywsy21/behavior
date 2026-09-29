"""Finish ONLY the preregistered E2 final matrix after an audited pool failure."""
import json
import signal
import time
import traceback
from pathlib import Path

from common import OUT, save, sha
from learner import Experiment
from method import MethodExperiment
from g05.rl.protocol import EVAL_SEEDS, paired_summary
from g05.rl.recovery import validate_baseline


class FinalEvaluation(MethodExperiment):
    def __init__(self):
        # Bypass the training experiment's entry guard, not its execution contracts.
        Experiment.__init__(self)
        if self.manifest['entry'] != 'final_eval' or self.manifest['max_new_actor_updates'] != 0:
            raise ValueError('Explicit evaluation-only manifest required')
        self.pool = None
        self.evaluations = list(self.manifest['prior_evaluations'])
        validate_baseline(self.evaluations)
        self.reset_references = {int(k): v for k, v in self.manifest['reset_references'].items()}
        self.pool_history = []
        self.outstanding = {}
        self.training_started = None
        self.start_updates = 10
        self.eval_log = (OUT/'evaluations.jsonl').open('x', buffering=1)
        for row in self.evaluations:
            self.eval_log.write(json.dumps(row)+'\n')

    def forbidden(self, *args, **kwargs):
        raise RuntimeError('Training, BC, replay and checkpoint selection are disabled in final evaluation')

    train = update = rollout = replay = prepare_bc = checkpoint = gates = forbidden

    def spawn_pool(self, pool):
        if pool != 'final':
            raise ValueError('Only the original final evaluation pool is allowed')
        return super().spawn_pool(pool)

    def restore_delta(self, path, expected_sha, *, optimizer, receipt_name=None):
        if optimizer:
            raise ValueError('Final evaluation cannot restore a training optimizer')
        return super().restore_delta(path, expected_sha, optimizer=False, receipt_name=receipt_name)

    def run(self):
        try:
            for path, expected in self.manifest['audit_input_sha256'].items():
                if sha(path) != expected:
                    raise ValueError('Stopped-run evidence changed since audit: '+path)
            chosen = Path(self.manifest['final_checkpoint'])
            expected = self.manifest['final_checkpoint_sha256']
            # New process: construct original parent graph, then strict-load SAVED delta.
            self.load()
            self.restore_delta(chosen, expected, optimizer=False)
            if self.actor_updates != 94 or self.critic_updates != 16:
                raise ValueError('Loaded checkpoint update counters differ from frozen selection')
            self.spawn_pool('final')
            self.evaluate([('rl_fp32', list(EVAL_SEEDS), 'float32')])
            self.close_pool()
            result = paired_summary(self.evaluations)
            result.update(checkpoint=str(chosen), checkpoint_sha256=expected,
                actor_updates=94, new_actor_updates=84, new_training_updates=0,
                controls=self.budget.used, seconds=time.monotonic()-self.started,
                cumulative_controls=self.manifest['prior_controls']+self.budget.used,
                cumulative_active_seconds=self.manifest['prior_active_seconds']+time.monotonic()-self.started,
                prior_run=self.manifest['continued_from'], training_stop_reason='registered_training_deadline')
            save(OUT/'result.json', result)
            self.phase = 'completed'
            self.status(paired_result=result)
        except BaseException as error:
            self.phase = 'failed'
            self.status(error=repr(error))
            save(OUT/'failure.json', dict(error=repr(error), traceback=traceback.format_exc(), controls=self.budget.used))
            raise
        finally:
            self.close_pool(strict=False)
            self.eval_log.close()
            self.log.close()


if __name__ == '__main__':
    def stop(signum, frame):
        raise SystemExit(f'Owned final evaluation stopped by signal {signum}')
    signal.signal(signal.SIGTERM, stop)
    FinalEvaluation().run()
