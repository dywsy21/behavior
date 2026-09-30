import copy
import hashlib
import json
import unittest
from unittest.mock import patch
import probe_jev_search_context as probe


class SearchProbeTests(unittest.TestCase):
    def fixture(self):
        request=dict(model='jev-1.13.0',images=[],questions={'command':{'instructions':'original rules','criteria':{
            'command_000':'hold','command_001':'yaw','abstain':'stop'}}},state=dict(
            harness={'stage':'SEARCH','search':{'total_bins':24}},facts={'target_visible':False,
                'rotation_option_available':False,'immediate_hazard':False},chosen_tactic_not_new_evidence='search',
            commands={'command_000':{'command':{'part':'all','move':'hold'}},
                      'command_001':{'command':{'part':'base','move':'yaw_plus'}}}))
        raw=json.dumps({'request_without_pixel_duplicates':request}).encode()
        return request,raw

    def test_exact_preserved_and_each_variant_has_only_its_named_change(self):
        request,raw=self.fixture()
        with patch.object(probe,'SOURCE_SHA',hashlib.sha256(raw).hexdigest()):outputs=probe.variants(raw)
        self.assertEqual(outputs['exact'],request)
        b=copy.deepcopy(outputs['search_explanation'])
        b['questions']['command']['instructions']=b['questions']['command']['instructions'].removesuffix(probe.EXPLANATION)
        self.assertEqual(b,request)
        c=copy.deepcopy(outputs['rotation_label']);c['state']['facts']['rotation_option_available']=False
        self.assertEqual(c,request)
        self.assertTrue(outputs['rotation_label']['state']['facts']['rotation_option_available'])
        self.assertEqual(request['facts'] if 'facts' in request else request['state']['facts'],outputs['exact']['state']['facts'])

    def test_unpinned_or_non_search_input_rejected_before_api(self):
        request,raw=self.fixture()
        with self.assertRaises(ValueError):probe.variants(raw)
        request['state']['harness']['stage']='APPROACH'
        bad=json.dumps({'request_without_pixel_duplicates':request}).encode()
        with patch.object(probe,'SOURCE_SHA',hashlib.sha256(bad).hexdigest()):
            with self.assertRaises(ValueError):probe.variants(bad)


if __name__=='__main__':unittest.main()
