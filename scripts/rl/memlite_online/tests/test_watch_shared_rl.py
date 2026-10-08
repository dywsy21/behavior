import importlib.util
from pathlib import Path
import unittest


path = Path(__file__).resolve().parents[1] / 'tools/watch_shared_rl.py'
spec = importlib.util.spec_from_file_location('watch_shared_rl', path)
watch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(watch)


class ObserverTests(unittest.TestCase):
    def test_explicit_team_can_differ_from_default_entity(self):
        team = dict(name='requested-team', isTeam=True,
                    organization=dict(name='requested-org', orgEntity=dict(name='org-entity')))
        viewer = dict(username='owner', entity='other-default',
                      teams=dict(edges=[dict(node=team)]))
        self.assertEqual(watch.verify_destination(viewer, 'owner', 'requested-team',
                                                 'requested-org'), team)
        self.assertEqual(watch.verify_destination(viewer, 'owner', 'requested-team',
                                                 'org-entity'), team)
        for expected_user, entity, org in [('wrong-user', 'requested-team', 'requested-org'),
                ('owner', 'requested-team', 'other-org'), ('owner', 'unlisted', None)]:
            with self.subTest(expected_user=expected_user, entity=entity, org=org):
                with self.assertRaises(ValueError):
                    watch.verify_destination(viewer, expected_user, entity, org)

    def test_org_entity_is_not_a_project_team(self):
        viewer = dict(username='owner', teams=dict(edges=[
            dict(node=dict(name='org-entity', isTeam=False))]))
        with self.assertRaises(ValueError):
            watch.verify_destination(viewer, 'owner', 'org-entity')

    def test_error_redaction(self):
        body = {'errors': [{'message': 'bad fake-key and wandb_v1_fake_alternative_token'}]}
        self.assertEqual(watch.safe_api_errors(body, 'fake-key'),
                         ['bad [REDACTED] and [REDACTED]'])

    def test_no_completed_is_not_zero_sr(self):
        result = watch.train_metrics(dict(controls=10, completed_episodes=0,
            macro_q_covered=None, macro_sr_covered=None, versions=[2] * 8))
        self.assertNotIn('train/macro_sr_covered', result)
        self.assertEqual(result['train/control_steps'], 10)

    def test_nonfinite_and_text_never_uploaded_as_metrics(self):
        self.assertEqual(watch.numeric_fields(dict(a=float('nan'), b='secret', c=3),
                                             ('a', 'b', 'c'), 'x/'), {'x/c': 3})

    def test_ppo_axes_and_phases(self):
        row = dict(update=7, accepted_actor_lr=1e-7,
            before=dict(mean_approx_kl=0), post_update=dict(mean_approx_kl=.001))
        result = watch.ppo_metrics(row)
        self.assertEqual(result['ppo/update'], 7)
        self.assertEqual(result['ppo/before/mean_approx_kl'], 0)
        self.assertEqual(result['ppo/post_update/mean_approx_kl'], .001)

    def test_watch_findings_are_not_effect_claims(self):
        snapshot = dict(summary=dict(updated=14400, completed_episodes=0,
            episodes_with_q_increase=0, successes=0),
            state=dict(started=0, status='training'), aborted=False)
        keys = dict(watch.findings(snapshot, 14400))
        self.assertEqual(set(keys), {'four_hours_no_q_increase'})
        snapshot['summary'].update(completed_episodes=1, episodes_with_q_increase=1, successes=1)
        keys = dict(watch.findings(snapshot, 14400))
        self.assertIn('first_train_success', keys)
        self.assertNotIn('four_hours_no_q_increase', keys)

    def test_failure_and_stale_are_separate(self):
        snapshot = dict(summary=dict(updated=0, completed_episodes=0,
            episodes_with_q_increase=0, successes=0),
            state=dict(started=0, status='failed'), aborted=True)
        self.assertEqual(set(dict(watch.findings(snapshot, 400))), {'training_failed'})
        snapshot['aborted'] = False
        snapshot['state']['status'] = 'training'
        self.assertEqual(set(dict(watch.findings(snapshot, 400))), {'monitor_stale'})


if __name__ == '__main__':
    unittest.main()
