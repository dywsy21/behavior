import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
import numpy as np
from semantic_robot.v2.saved_prefix import load_saved_prefix,check_endpoint,bootstrap_unverified_pick
from semantic_robot.v2.grounded_harness import GroundedHarness


class SavedPrefixTests(unittest.TestCase):
    def make(self,root,change=None):
        goal={"kind":"pick","target":"object","hand":"right","done_when":"moves with hand","level":False}
        manifest={"task":0,"args":{"prefix":448},"window_sha":"window","robot_sha":"robot","code_commit":"code"}
        rows=[]
        for i in range(4):
            a=[0.]*23;a[14]=1;a[22]=-1 if i>=2 else 1
            rows.append({"control":i+1,"decision":i//2,"action23":a,"terminal":False})
        for i,move in enumerate(("up","close")):
            rows.append({"decision":i,"action":{"part":"right","move":move,"scale":"fine","frame":"base"},
                "feedback":{"control_ticks":2},"control_end":2*(i+1)})
        state={"q":[0.]*18,"gripper":[.05,.035]}
        if change:change(manifest,rows,goal,state)
        files={"manifest.json":json.dumps(manifest),"steps.jsonl":"\n".join(map(json.dumps,rows)),
            "plan.json":json.dumps([goal]),"decision_002/proprio.json":json.dumps(state)}
        hashes={}
        for name,text in files.items():
            path=root/name;path.parent.mkdir(exist_ok=True);path.write_text(text)
            hashes[name]=hashlib.sha256(path.read_bytes()).hexdigest()
        spec={"schema":1,"source_run":str(root),"source_code_commit":"code","replay_decisions":2,"replay_controls":4,"sha256":hashes}
        path=root/"spec.json";path.write_text(json.dumps(spec));return path

    def load(self,p):return load_saved_prefix(p,task=0,prefix=448,window_sha="window",robot_sha="robot")

    def test_hash_pinned_replay_only_bootstraps_unverified_latch(self):
        with tempfile.TemporaryDirectory() as root:
            replay=self.load(self.make(Path(root)))
        state=SimpleNamespace(q=np.zeros(18),gripper=np.array([.05,.035]))
        self.assertTrue(check_endpoint(replay,state)["passed"])
        h=GroundedHarness(replay["plan"]);bootstrap_unverified_pick(h,replay,state)
        self.assertEqual(h.stage,"VERIFY_GRASP");self.assertTrue(h.pending_grasp["right"])
        self.assertFalse(any(h.hold_verified.values()));self.assertEqual(h.completed,[]);self.assertIsNone(h.feedback)
        self.assertFalse(replay["receipt"]["autonomous_progress"])
        state.q[0]=.02;self.assertFalse(check_endpoint(replay,state)["passed"])

    def test_modified_source_is_rejected_before_replay(self):
        with tempfile.TemporaryDirectory() as root:
            p=self.make(Path(root));(Path(root)/"steps.jsonl").write_text("{}")
            with self.assertRaises(ValueError):self.load(p)

    def test_wrong_task_missing_controls_terminal_nonfinite_or_wrong_last_action_rejected(self):
        changes=[lambda m,r,g,s:m.update(task=3),lambda m,r,g,s:r[0].update(control=2),
                 lambda m,r,g,s:r[1].update(terminal=True),lambda m,r,g,s:r[0]["action23"].__setitem__(0,float("nan")),
                 lambda m,r,g,s:r[-1]["action"].update(move="open"),lambda m,r,g,s:g.update(kind="navigate")]
        for change in changes:
            with tempfile.TemporaryDirectory() as root:
                with self.assertRaises(ValueError):self.load(self.make(Path(root),change))

    def make_approach(self,root,change=None):
        def make_unloaded(m,r,g,s):
            m["args"]["prefix"]=0
            for row in r:
                if "action23" in row:row["action23"][22]=1
                elif row["action"]["move"]=="close":row["action"]["move"]="up"
            s["gripper"]=[.05,.05]
        path=self.make(root,make_unloaded)
        spec=json.loads(path.read_text());spec.update(schema=2,purpose="matched_unladen_approach_diagnostic")
        h={"goal_index":0,"goal":json.loads((root/"plan.json").read_text())[0],"stage":"APPROACH",
           "stop_reason":None,"carry_constraints":False,
           "unverified_close_latches":{"left":False,"right":False},
           "holding_verified_by_observation_and_proprio":{"left":False,"right":False},
           "possible_contact_after_any_close":{"left":False,"right":False}}
        if change:change(spec,h)
        name="decision_002/harness.json";(root/name).write_text(json.dumps(h))
        spec["sha256"][name]=hashlib.sha256((root/name).read_bytes()).hexdigest()
        path.write_text(json.dumps(spec));return path

    def test_unladen_diagnostic_does_not_copy_old_evidence_or_completion(self):
        with tempfile.TemporaryDirectory() as root:
            p=self.make_approach(Path(root))
            replay=load_saved_prefix(p,task=0,prefix=0,window_sha="window",robot_sha="robot")
        h=GroundedHarness(replay["plan"])
        bootstrap_unverified_pick(h,replay,SimpleNamespace(q=np.zeros(18),gripper=np.array([.05,.05])))
        self.assertEqual(h.stage,"SEARCH");self.assertEqual(h.completed,[])
        self.assertIsNone(h.observation);self.assertIsNone(h.feedback);self.assertIsNone(h.last_action)
        self.assertFalse(any(h.pending_grasp.values()));self.assertFalse(any(h.hold_verified.values()))
        self.assertFalse(replay["receipt"]["autonomous_progress"])

    def test_unladen_source_rejects_wrong_stage_missing_load_state_goal_and_budget(self):
        changes=[lambda s,h:h.update(stage="VERIFY_GRASP"),lambda s,h:h.pop("unverified_close_latches"),
                 lambda s,h:h["possible_contact_after_any_close"].update(right=True),
                 lambda s,h:h.update(goal_index=1),lambda s,h:s.update(replay_controls=2049),
                 lambda s,h:s.update(purpose="arbitrary_resume")]
        for change in changes:
            with tempfile.TemporaryDirectory() as root:
                p=self.make_approach(Path(root),change)
                with self.assertRaises(ValueError):load_saved_prefix(p,task=0,prefix=0,window_sha="window",robot_sha="robot")


if __name__=="__main__":unittest.main()
