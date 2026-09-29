"""E3 continuation: automatic curriculum, signed dense reward, stronger PPO."""
import json
from pathlib import Path
import signal
import time
import traceback

from common import OUT, save, sha
from learner import cpu_tree
from method import MethodExperiment, TrainingCutoff
from g05.rl.dense_recipe import validate_recipe, reward_batch_audit
from g05.rl.protocol import EVAL_SEEDS, paired_summary
from g05.rl.recovery import validate_baseline


class DenseExperiment(MethodExperiment):
    def __init__(self):
        super().__init__()
        validate_recipe(self.manifest)
        self.evaluations = list(self.manifest['prior_evaluations'])
        validate_baseline(self.evaluations)
        self.reset_references = {int(k): v for k, v in self.manifest['reset_references'].items()}
        for row in self.evaluations:
            self.eval_log.write(json.dumps(row)+'\n')

    def spawn_pool(self, pool):
        if pool not in ('training', 'final'):
            raise ValueError('E3 reuses sealed prior baselines; no extra baseline sampling')
        return super().spawn_pool(pool)

    def update(self, rows, advantages, returns):
        audit = reward_batch_audit(rows)
        # A constant gamma offset alone is not a demonstrated dense learning signal.
        if not audit['state_changing_potential_controls']:
            raise ValueError('Dense potential did not change on any real autonomous control')
        save(OUT/f'reward_audit_{self.batches:03d}.json', audit)
        return super().update(rows, advantages, returns)

    def run(self):
        try:
            for path, expected in self.manifest['audit_input_sha256'].items():
                if sha(path) != expected:
                    raise ValueError('Sealed E2 evidence changed: '+path)
            self.load()
            self.parent_ae = cpu_tree(self.policy.model.action_expert.state_dict())
            self.restore_delta(self.manifest['resume_checkpoint'], self.manifest['resume_sha256'], optimizer=True)
            if self.actor_updates != 94 or self.critic_updates != 16:
                raise ValueError('Wrong E2 continuation counts')
            self.start_updates = self.actor_updates
            self.prepare_bc()
            self.spawn_pool('training')
            try:
                self.train()
            except TrainingCutoff:
                for w, n in list(self.outstanding.items()): self.collect_reply(w, n)
                self.actor_optimizer.zero_grad(set_to_none=True)
                self.critic_optimizer.zero_grad(set_to_none=True)
                completed_batches = self.batches
                self.batches += 1
                self.checkpoint()
                self.last_checkpoint = OUT/f'rl_batch_{self.batches:03d}_updates_{self.actor_updates:04d}.pt'
                self.stop_reason = 'registered_training_deadline'
                save(OUT/'training_result.json', dict(reason=self.stop_reason,
                    completed_batches=completed_batches, checkpoint_counter=self.batches,
                    actor_updates=self.actor_updates, new_actor_updates=self.actor_updates-self.start_updates,
                    controls=self.budget.used-self.training_start_controls,
                    seconds=time.monotonic()-self.training_started, checkpoint=str(self.last_checkpoint),
                    partial_rollout_not_used_for_extra_updates=True))
            self.close_pool()
            chosen = self.last_checkpoint or Path(self.manifest['resume_checkpoint'])
            receipt = json.loads(chosen.with_suffix('.json').read_text())
            save(OUT/'frozen_final_selection.json', dict(checkpoint=str(chosen), sha256=receipt['sha256'],
                selection='last legal checkpoint, fixed before evaluation', eval_results_used=False))
            self.policy.model.action_expert.load_state_dict(self.parent_ae, strict=True)
            self.restore_delta(chosen, receipt['sha256'], optimizer=False)
            del self.parent_ae
            self.parent_ae = None
            self.spawn_pool('final')
            self.evaluate([('rl_fp32', list(EVAL_SEEDS), 'float32')])
            self.close_pool()
            result = paired_summary(self.evaluations)
            against_e2 = [dict(r, variant='parent_fp32') for r in self.manifest['resume_evaluations']]
            against_e2 += [r for r in self.evaluations if r['variant'] == 'rl_fp32']
            result.update(vs_e2=paired_summary(against_e2), checkpoint=str(chosen),
                checkpoint_sha256=receipt['sha256'], actor_updates=self.actor_updates,
                new_actor_updates=self.actor_updates-self.start_updates, controls=self.budget.used,
                seconds=time.monotonic()-self.started, training_stop_reason=self.stop_reason,
                reward_shaping_used_for_success_metric=False, prior_baselines_reused=True)
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
    def stop(signum, frame): raise SystemExit(f'Owned E3 stopped by signal {signum}')
    signal.signal(signal.SIGTERM, stop)
    DenseExperiment().run()
