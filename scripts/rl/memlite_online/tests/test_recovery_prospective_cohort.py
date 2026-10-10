import importlib.util
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

spec=importlib.util.spec_from_file_location('prospective_cohort',Path(__file__).resolve().parents[1]/'tools/prepare_prospective_calibration_cohort.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


class ProspectiveCohortTests(unittest.TestCase):
    def test_complete_denominator_preserves_unavailable_sources_and_rejects_early_subset(self):
        with TemporaryDirectory() as temporary:
            root=Path(temporary)
            def write(path,value):
                path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))
                return module.file_sha(path)
            cfg=dict(schema='prospective_observer_calibration_collection_v1',
                declared_before_physical_collection_and_model_predictions=True,
                selected_observer=dict(sha256='d'*64),selection=dict(metadata_fresh_sources=2))
            config=root/'config.json';config_sha=write(config,cfg)
            old=root/'old.json';write(old,dict(schema='recovery_independent_cohort_v1',
                frozen_test_groups=[dict(source_group='old:1')]))
            sources=root/'sources';collection=root/'collection';corpus=root/'corpus'
            originals=[];attempts=[];groups=[]
            for ordinal,(case,status) in enumerate((('new_1','completed_candidates_only'),
                                                    ('new_2','reference_grasp_not_reproduced'))):
                group='new:'+str(ordinal+1)
                sha=write(sources/case/'manifest.json',dict(source_group=group,recovery_split='dev'))
                originals.append(dict(directory=case,manifest_sha256=sha));groups.append(dict(source_group=group))
                result_sha=write(collection/case/'result.json',dict(source_group=group,pid=ordinal+1,
                    proposal_sha256=sha,status=status))
                attempts.append(dict(case=case,pid=ordinal+1,returncode=0,closed_result_sha256=result_sha))
            inventory_sha=write(sources/'manifest.json',dict(cases=originals))
            audit=root/'audit.json';write(audit,dict(config_sha256=config_sha,
                source_inventory_sha256=inventory_sha,selected_observer_sha256='d'*64,
                status='source_closure_and_group_isolation_passed_not_labels',groups=groups))
            write(collection/'result.json',dict(status='inventory_attempts_finished',results=attempts))
            write(corpus/'review-queue.json',[dict(source_group='new:1',case='new_1',arm='left',split='dev',branch='clean')])
            output=root/'out.json'
            args=['cohort','--config',str(config),'--source-audit',str(audit),'--external-cohort',str(old),
                  '--sources',str(sources),'--collection',str(collection),'--corpus',str(corpus),'--output',str(output)]
            def git(command,**kwargs):return '' if 'status' in command else 'a'*40
            with patch.object(sys,'argv',args),patch.object(module.subprocess,'check_output',side_effect=git):
                module.main()
            got=json.loads(output.read_text())
            self.assertEqual(got['declared_sources'],2)
            self.assertEqual(got['unavailable_source_groups'],['new:2'])
            self.assertFalse(got['calibration_ready']);self.assertFalse(got['model_predictions_read'])
            self.assertEqual(len(got['calibration_groups']),2)
            args[-1]=str(root/'not_written.json')
            write(collection/'result.json',dict(status='inventory_attempts_finished',results=attempts[:1]))
            with patch.object(sys,'argv',args),patch.object(module.subprocess,'check_output',side_effect=git):
                with self.assertRaisesRegex(ValueError,'Lost, duplicated or skipped'):module.main()
            self.assertFalse(Path(args[-1]).exists())


if __name__=='__main__':unittest.main()
