import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/"scripts/vlm_sft"),str(ROOT/"src")]
from common import TOKENS, actor_state, classify_window, clean_object, prompt, token_to_action


def quiet():
    states=np.zeros((17,61))
    states[:,23]=1
    states[:,48]=1
    states[:,24:26]=.05
    states[:,49:51]=.05
    actions=np.zeros((16,23));actions[:,[14,22]]=1
    return states,actions


class ContractTests(unittest.TestCase):
    def test_cartesian_signs_and_part(self):
        for arm,start in (("LEFT",17),("RIGHT",42)):
            for axis,pos,neg in ((0,"FORWARD","BACK"),(1,"LEFT","RIGHT"),(2,"UP","DOWN")):
                for sign,name in ((1,pos),(-1,neg)):
                    s,a=quiet();s[:,start+axis]=np.linspace(0,.02*sign,17)
                    self.assertEqual(classify_window(s,a)[0],f"{arm}_{name}")

    def test_mixed_is_not_fake_hold(self):
        s,a=quiet();s[:,42]=np.linspace(0,.02,17);s[:,44]=np.linspace(0,.018,17)
        self.assertIsNone(classify_window(s,a)[0])
        self.assertIsNone(classify_window(*quiet())[0])

    def test_base_not_masked_and_transition_refused(self):
        s,a=quiet();a[:,2]=.2
        self.assertEqual(classify_window(s,a)[0],"BASE_YAW_PLUS")
        a[:8,2]=-.2
        self.assertIsNone(classify_window(s,a)[0])

    def test_late_base_reversal_rejected(self):
        s,a=quiet();a[:,2]=.4;a[12:,2]=-.1
        self.assertIsNone(classify_window(s,a)[0])

    def test_arm_excursion_cannot_hide_inside_base_or_gripper(self):
        for gripper in (False,True):
            s,a=quiet();s[4:10,42]=.08
            if gripper:
                s[:,49:51]=np.linspace(.05,.02,17)[:,None];a[:,22]=-1
            else:a[:,0]=.3
            self.assertIsNone(classify_window(s,a)[0])

    def test_curved_side_excursion_not_pure_translation(self):
        s,a=quiet();s[:,42]=np.linspace(0,.02,17);s[4:10,43]=.05
        self.assertIsNone(classify_window(s,a)[0])

    def test_torso_not_mislabelled_as_both_arms(self):
        s,a=quiet();s[:,53]=np.linspace(0,.04,17)
        s[:,19]=np.linspace(0,.02,17);s[:,44]=np.linspace(0,.02,17)
        self.assertEqual(classify_window(s,a)[0],"TORSO_UP")

    def test_gripper_is_command_not_success(self):
        s,a=quiet();s[:,49:51]=np.linspace(.05,.02,17)[:,None];a[:,22]=-1
        self.assertEqual(classify_window(s,a)[0],"RIGHT_CLOSE")
        self.assertNotIn("DONE",TOKENS)
        self.assertNotIn("SUCCEEDED",TOKENS)

    def test_contract_strips_ids_and_refuses_privileged_keys(self):
        self.assertEqual(clean_object("coffee_table_koagbh_0"),"coffee table")
        self.assertEqual(clean_object("radio_89"),"radio")
        self.assertEqual(clean_object("radio_89, coffee_table_koagbh_0"),"radio, coffee table")
        state=actor_state(quiet()[0][0])
        text=prompt("turn on radio","pick radio",state,[])
        self.assertNotIn("future",text)
        state["goal_status"]=True
        with self.assertRaises(ValueError):prompt("task","pick",state,[])

    def test_every_native_symbol_has_unmodified_harness_action(self):
        for token in TOKENS:
            value=token_to_action(token)
            self.assertEqual(type(value).parse(value.text()),value)


if __name__ == "__main__":unittest.main()
