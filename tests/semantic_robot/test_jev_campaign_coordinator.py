import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import run_jev_campaign as campaign


class CoordinatorTests(unittest.TestCase):
    def run_fixture(self, fail=None):
        temporary=tempfile.TemporaryDirectory();self.addCleanup(temporary.cleanup)
        root=Path(temporary.name)/'coordinator';root.mkdir()
        identity=dict(code_commit='frozen',implementation_digest='digest')
        (root/'reservation.json').write_text(json.dumps(dict(identity=identity,stages=list(campaign.ORDER))))
        observed=[]
        def configure(stage,kind):
            self.assertEqual(kind,'multitask')
            campaign.launch.ROOT=Path(temporary.name)/stage
            return SimpleNamespace(PYTHON=Path('/python'))
        def run(command,**kwargs):
            stage=command[command.index('--stage')+1];observed.append(stage)
            folder=campaign.launch.ROOT;folder.mkdir()
            (folder/'launch.json').write_text(json.dumps(dict(source_commit='frozen',campaign='multitask',
                implementation_digest='digest',supervisor_pid=123,wall_seconds=3600)))
            (folder/'supervisor.json').write_text(json.dumps(dict(status='failed' if stage==fail else 'completed',
                exit_codes={'actor':0},official_success=False,controls=1,stop_reason='JEV_PLAN_ABSTAINED')))
            return SimpleNamespace(returncode=0)
        with patch.object(campaign,'ROOT',root),patch.object(campaign,'source_identity',return_value=identity), \
                patch.object(campaign.launch,'ROOT',None),patch.object(campaign.launch,'configure',side_effect=configure), \
                patch.object(campaign.subprocess,'run',side_effect=run):
            if fail:
                with self.assertRaises(RuntimeError):campaign.supervise()
            else:campaign.supervise()
            with self.assertRaises(FileExistsError):campaign.supervise()  # no duplicate coordinator/reset
        return observed,json.loads((root/'campaign.json').read_text())

    def test_policy_failures_do_not_trigger_retry_or_filter_later_slots(self):
        observed,receipt=self.run_fixture()
        self.assertEqual(observed,list(campaign.ORDER))
        self.assertEqual(receipt['status'],'completed')
        self.assertEqual(receipt['retries'],0)

    def test_infrastructure_failure_retains_slot_and_stops_next_reset(self):
        observed,receipt=self.run_fixture('task1a')
        self.assertEqual(observed,['gate0','gate3','task0a','task0b','task1a'])
        self.assertEqual(receipt['status'],'stopped')
        self.assertEqual(receipt['stages'][-1]['stage'],'task1a')
        self.assertEqual(receipt['stages'][-1]['status'],'failed')


if __name__=='__main__':unittest.main()
