import copy
import json
from pathlib import Path
import sys
import unittest

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"scripts/vlm_sft"))
from native_reference_profile import (PROFILE,PURPOSE,ROOT as EXPERIMENT,SOURCE,HELDOUT,BUDGET,
    validate_profile,validate_execution_location,check_initialization_deadline)
from native_teacher_group_prepare import select_training_source,COUNTS_SHA
from native_grasp_sources import reserve_groups


class ReferenceProfileTests(unittest.TestCase):
    def setUp(self):
        self.release=json.loads((ROOT/"configs/vlm_sft/h09y_reference_release_template.json").read_text())
        self.manifest={"reference_profile":PROFILE,"purpose":PURPOSE,"held_out_instance_groups":HELDOUT}
        self.row=dict(zip(("task","episode","instance"),SOURCE))
        self.row.update(start=596,end=1186,verb="GRASP",hand="right",support_hand=None,payloads=[])

    def test_exact_new_and_unchanged_legacy(self):
        self.assertEqual(validate_profile(self.release,self.manifest,self.row),PROFILE)
        validate_execution_location(self.release,EXPERIMENT/"reference_train114_v1",3)
        self.assertIsNone(validate_profile({}, {}, {}))
        validate_execution_location({},"unused",1)
        with self.assertRaises(ValueError):validate_execution_location({},"unused",3)
        with self.assertRaises(ValueError):validate_profile({},self.manifest,self.row)

    def test_mixed_unknown_boolean_and_extra_budget_rejected(self):
        variants=[("reference_profile","unknown"),("purpose","NATIVE_BC"),("physical_gpu",True),
                  ("physical_gpu",1),("resets",True),("model_calls",False),("source",[1,200,1]),
                  ("held_out_instance_groups",[]),("experiment_root","/tmp/elsewhere")]
        for key,value in variants:
            with self.subTest(key=key,value=value):
                release=copy.deepcopy(self.release);release[key]=value
                with self.assertRaises(ValueError):validate_profile(release,self.manifest,self.row)
        for key in BUDGET:
            release=copy.deepcopy(self.release);release["budget"][key]=float(BUDGET[key])
            with self.assertRaises(ValueError):validate_profile(release,self.manifest,self.row)
        release=copy.deepcopy(self.release);release["budget"]["extra"]=1
        with self.assertRaises(ValueError):validate_profile(release,self.manifest,self.row)
        for output,gpu in ((EXPERIMENT/"retry",3),(EXPERIMENT/"reference_train114_v1",1)):
            with self.assertRaises(ValueError):validate_execution_location(self.release,output,gpu)

    def test_initialization_not_legacy_time_change(self):
        check_initialization_deadline(self.release,899.99)
        with self.assertRaises(TimeoutError):check_initialization_deadline(self.release,900)
        check_initialization_deadline({},100000)

    def test_reserved_training_source_not_heldout_or_branch_alias(self):
        counts=json.loads((ROOT/"configs/vlm_sft/h09r_train_feasibility_counts.json").read_text())
        audit={"schema":"h09x-grasp-instance-split-v1","counts_sha256":COUNTS_SHA,"groups":[]}
        for split,source in reserve_groups(counts):
            audit["groups"].append({"split":split,"source":source,"selected_earliest_eligible":{
                "start":596,"end":1186,"hand":"right","skill":{"verb":"GRASP"},"near_prefix_controls":993}})
        self.assertEqual(select_training_source(counts,audit)["source"]["instance"],114)
        for mutation in ("split","duplicate","near"):
            altered=copy.deepcopy(audit)
            if mutation=="split":altered["groups"][1]["split"]="heldout"
            if mutation=="duplicate":altered["groups"].append(altered["groups"][1])
            if mutation=="near":altered["groups"][1]["selected_earliest_eligible"]["near_prefix_controls"]=992
            with self.assertRaises(ValueError):select_training_source(counts,altered)


if __name__=="__main__":unittest.main()
