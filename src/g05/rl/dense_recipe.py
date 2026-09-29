"""Explicit E3 opt-in; historical E1/E2 defaults remain unchanged."""
from math import isfinite

REWARD = dict(kind='goal_geometry_potential_v1', weight=.2, gamma=.9998,
              goal_weight=.7, distance_scale=.25)


def validate_recipe(manifest):
    expected = dict(entry='method_dense', ae_precision='float32', curriculum_admission='automatic',
        bounded_backtracking=True, reset_candidate_lr_each_minibatch=True,
        clip=.1, target_path_kl=.1, target_mean_path_kl=.02, ppo_epochs=4,
        critic_steps_per_batch=4, bc_weight=.1, gamma_control=.9998, lambda_chunk=.95,
        episode_controls=1024, max_controls=100000, max_active_wall_seconds=43200,
        max_training_controls=80000, max_training_seconds=28800, max_batches=64,
        max_new_actor_updates=2000, max_actor_updates=2094,
        final_eval_reserved_controls=19344, final_eval_reserved_seconds=10800)
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(f'Unregistered E3 setting {key}')
    if (manifest.get('reward') != REWARD
            or manifest.get('learning_rates') != dict(action_expert=1e-7, noise=1e-6, critic=1e-4)
            or manifest.get('backtracking_scales') != [1., .5, .25, .125, .0625, .03125]):
        raise ValueError('Unregistered reward or optimization settings')
    if manifest['max_training_controls'] + manifest['final_eval_reserved_controls'] > manifest['max_controls']:
        raise ValueError('Final evaluation control reserve missing')


def reward_batch_audit(rows):
    official = dense = 0.
    transitions = changed = 0
    potentials = []
    for row in rows:
        if row.get('split') != 'train':
            raise ValueError('Non-TRAIN reward sample')
        keys = ('rewards', 'official_rewards', 'shaping_rewards', 'reward_details')
        if any(key not in row for key in keys) or len({len(row[key]) for key in keys}) != 1:
            raise ValueError('Misaligned dense reward audit')
        for total, base, shaping, detail in zip(*(row[key] for key in keys)):
            if (not all(isfinite(x) for x in (total, base, shaping)) or base not in (0., 1.)
                    or abs(total-base-shaping) > 1e-8 or not isinstance(detail, dict)
                    or detail['total'] != total or detail['official'] != base or detail['shaping'] != shaping):
                raise ValueError('Dense reward arithmetic or official reward invalid')
            before, after = detail['potential_before'], detail['potential_after']
            if not all(isfinite(x) and 0 <= x <= 1 for x in (before, after)):
                raise ValueError('Invalid recorded potential')
            potentials.extend((before, after))
            transitions += 1
            changed += abs(after-before) > 1e-8
            official += base
            dense += shaping
    if not transitions:
        raise ValueError('No autonomous transitions to audit')
    return dict(policy_controls=transitions, official_reward=official, shaping_reward=dense,
                state_changing_potential_controls=changed,
                potential_min=min(potentials), potential_max=max(potentials),
                actor_observation_contains_reward=False)
