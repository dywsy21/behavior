import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

spec=importlib.util.spec_from_file_location('prep_ticket',Path(__file__).resolve().parents[1]/'tools/prepare_recovery_sft_ticket.py')
ticket=importlib.util.module_from_spec(spec);spec.loader.exec_module(ticket)


class PreparationTicketTests(unittest.TestCase):
    def test_fixed_success_states_admission_binding_and_all_checks(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);values={};checks={}
            for name,(schema,status) in ticket.DATA_CHECKS.items():
                value=dict(schema=schema,status=status,admission_sha256='a'*64,inventory_sha256='b'*64,
                    high_sha256='c'*64,optimizer_steps=0,head_consumption_and_insulated_backward=True,
                    formal_training=False,actor_oracle_inputs=False,manual_review_complete=True,
                    accepted=[dict(split='train'),dict(split='dev')])
                path=root/(name+'.json');path.write_text(json.dumps(value))
                values[name]=value;checks[name]=dict(path=str(path),sha256=ticket.file_sha(path))
            preflight=root/'preflight.json'
            def check():
                preflight.write_text(json.dumps(dict(schema='recovery_accepted_data_preflight_v1',
                    admission_sha256='a'*64,optimizer_steps=0,formal_training_authorized=False,checks=checks)))
                return ticket.validate_data_checks(preflight,'a'*64,dict(inventory_sha256='b'*64),'c'*64)
            self.assertEqual(len(check()),6)
            for field,value in [('status','failed'),('admission_sha256','d'*64),('inventory_sha256','d'*64)]:
                bad=dict(values['processor']);bad[field]=value
                path=Path(checks['processor']['path']);path.write_text(json.dumps(bad))
                checks['processor'].update(sha256=ticket.file_sha(path),accepted_statuses=['failed'])
                with self.assertRaises(ValueError):check()
            path.write_text(json.dumps(values['processor']));checks['processor']['sha256']=ticket.file_sha(path)
            del checks['transfer']
            with self.assertRaises(ValueError):check()


if __name__=='__main__':unittest.main()
