from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/"scripts/vlm_sft"),str(ROOT/"src")]
from common import CAMERAS,actor_state,prompt
from live import ACTIVE,parse_request,request_payload,runtime_proprio,validate_proprio
from run_local import guard_release,implementation_digest,GATE,write_privileged_audit
from stratify_static import low_velocity


class LiveContractTests(unittest.TestCase):
    def test_velocity_audit_counts_use_json_native_booleans(self):
        row={"proprio":{"base_velocity_local":[0,0,0]}}
        self.assertIs(type(low_velocity(row)),bool)
        self.assertIs(type(sum(low_velocity(row) for _ in range(3))),int)

    def fixture(self):
        state=np.zeros(61);state[17:20]=[.1,.2,.3];state[42:45]=[.4,-.2,.3]
        state[24:26]=.04;state[49:51]=.03;state[:3]=[.1,-.2,.05];state[53:57]=[.2,.3,.4,.5]
        views={v:Image.new("RGB",(256,256),(10,20,30)) for v in CAMERAS}
        return state,views

    def test_live_and_offline_proprio_and_prompt_are_identical(self):
        source,views=self.fixture()
        state=SimpleNamespace(base_velocity=source[:3],q=np.r_[source[53:57],np.zeros(14)],
            poses={"left":(source[17:20],None),"right":(source[42:45],None)},gripper=np.array([.04,.03]))
        value=runtime_proprio(state);self.assertEqual(value,actor_state(source))
        payload=request_payload("Turn on the radio.",value,["RIGHT_UP"],views,"finetuned",0)
        row,images=parse_request(payload)
        self.assertEqual(row["text"],prompt(payload["task"],ACTIVE[0],actor_state(source),["RIGHT_UP"]))
        self.assertEqual(set(images),set(CAMERAS))

    def test_no_privileged_field_or_future_history_allowed(self):
        source,views=self.fixture();value=actor_state(source)
        payload=request_payload("radio",value,[],views,"base",0)
        for field in ("goal_status","object_pose","expert_skill"):
            with self.assertRaises(ValueError):parse_request({**payload,field:True})
        with self.assertRaises(ValueError):parse_request({**payload,"history":["TORSO_UP"]})
        with self.assertRaises(ValueError):parse_request({**payload,"active_instruction":"oracle future move"})
        with self.assertRaises(ValueError):validate_proprio({**value,"holding_object_id":3})

    def test_unknown_latched_close_cannot_be_released(self):
        self.assertFalse(guard_release("RIGHT_OPEN",{"right":True,"left":False}))
        self.assertTrue(guard_release("LEFT_OPEN",{"right":True,"left":False}))
        self.assertTrue(guard_release("RIGHT_UP",{"right":True,"left":False}))
        self.assertEqual(len(implementation_digest()),64)
        self.assertFalse(any(t.startswith("TORSO") for t in GATE))

    def test_diagnostic_world_pose_is_write_only_not_actor_state(self):
        robot=SimpleNamespace(grasping_mode="physical",get_position_orientation=lambda:(np.array([1,2,3]),np.array([0,0,0,1])))
        with patch("run_local.write_json") as write:
            self.assertIsNone(write_privileged_audit("audit.json",robot))
        value=write.call_args.args[1]
        self.assertTrue(value["not_available_to_actor"])
        self.assertEqual(value["robot_world_position_m"],[1,2,3])
        with self.assertRaises(ValueError):validate_proprio(value)


if __name__=="__main__":unittest.main()
