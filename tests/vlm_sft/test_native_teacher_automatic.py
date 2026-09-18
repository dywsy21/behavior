"""CPU counterexamples. Synthetic outcomes are NOT real collection evidence."""
import copy
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/"src"), str(ROOT/"scripts/vlm_sft")]
from native_teacher_artifacts import ArtifactBudget, capture_layout, capture_upper_bound, action_evidence_bound
from native_teacher_outcomes import LocalOutcome, rigid
from native_teacher_policy import validate_spec, validate_train_group, PoseTeacher
from native_teacher_og import truth
from native_teacher_seed import extract_seed
from native_teacher_contract import actor_input
from native_teacher_collect import capture_snapshot
import test_native_teacher


def frame(tick=0):
    return {"tick": tick, "target_uid": "target", "target_pose": np.eye(4).tolist(),
            "hand_poses": {a: np.eye(4).tolist() for a in ("left", "right")},
            "held": {"left": False, "right": False}, "finger_contact": {"left": False, "right": False},
            "contacts_known": True, "payload_ok": True, "forbidden_contacts": [], "toggled": False,
            "linear_velocity": [0.,0.,0.], "angular_velocity": [0.,0.,0.],
            "relation": True, "supported": True, "corners_inside": True,
            "finger_opening": {"left": .05, "right": .05}}


class AutomaticTeacherTests(unittest.TestCase):
    def outcome(self, verb): return LocalOutcome({"verb": verb, "hand": "right", "target": "target"})

    def test_reservation_refuses_before_any_action_and_keeps_cleanup(self):
        with tempfile.TemporaryDirectory() as d:
            b=ArtifactBudget(d,run_limit=100,total_limit=1000,reserve=10)
            b.write_bytes(Path(d)/"before",b"a"*30)
            executed=[]
            with self.assertRaises(RuntimeError):
                with b.transaction(61): executed.append(True)
            self.assertFalse(executed)
            with b.transaction(60) as r:
                r.write_bytes(Path(d)/"after", b"b"*40)
                self.assertEqual(r.remaining,20)
                with self.assertRaises(RuntimeError): b.write_bytes(Path(d)/"steal",b"c")
            self.assertEqual(b._held,0)
            b.write_bytes(Path(d)/"cleanup",b"d"*20,cleanup=True)
            with self.assertRaises(RuntimeError): r.check(0)

    def test_raw_capture_bound_does_not_assume_compressibility(self):
        rng=np.random.default_rng(7)
        images={v+"_rgb":rng.integers(0,256,(3,24,32),dtype=np.uint8) for v in ("head","left_wrist","right_wrist")}
        depths={v:rng.random((24,32),dtype=np.float32) for v in ("head","left_wrist","right_wrist")}
        layout=capture_layout(images,depths)
        from PIL import Image
        total=0
        for raw in images.values():
            buf=io.BytesIO();Image.fromarray(raw.transpose(1,2,0)).save(buf,format="PNG");total+=len(buf.getvalue())
        buf=io.BytesIO();np.savez_compressed(buf,**depths);total+=len(buf.getvalue())
        self.assertLess(total,capture_upper_bound(layout))
        self.assertGreater(action_evidence_bound(layout),2*capture_upper_bound(layout))
        images["head_rgb"]=images["head_rgb"].astype(float)
        with self.assertRaises(ValueError): capture_layout(images,depths)

    def test_grasp_close_only_and_duplicate_clock_never_success(self):
        o=self.outcome("GRASP");f=frame();o.update(f)
        for t in range(1,20):
            f=frame(t);f["held"]["right"]=f["finger_contact"]["right"]=True
            out=o.update(f,"RIGHT_CLOSE")
        self.assertNotEqual(out["outcome"],"SUCCEEDED")
        self.assertEqual(o.update(f),out)
        conflict=copy.deepcopy(f);conflict["target_pose"][2][3]=.1
        with self.assertRaises(ValueError):o.update(conflict)
        with self.assertRaises(ValueError):o.update(frame(21))

    def test_grasp_requires_actual_lift_and_stable_relative_pose(self):
        o=self.outcome("GRASP");o.update(frame())
        f=frame(1);f["held"]["right"]=f["finger_contact"]["right"]=True;o.update(f,"RIGHT_CLOSE")
        for t in range(2,14):
            f=frame(t);f["held"]["right"]=f["finger_contact"]["right"]=True
            f["target_pose"][2][3]=f["hand_poses"]["right"][2][3]=.04
            out=o.update(f)
        self.assertEqual(out["outcome"],"SUCCEEDED")
        self.assertIsNone(out["official_task_success"])
        f=frame(14);f["forbidden_contacts"]=[["finger","torso"]]
        self.assertEqual(o.update(f)["outcome"],"FAILED")

    def test_press_requires_new_edge_with_contact_and_twelve_ticks(self):
        o=self.outcome("PRESS");o.update(frame())
        for t in range(1,13):
            f=frame(t);f["toggled"]=True;f["finger_contact"]["right"]=True;out=o.update(f)
            if t<12:self.assertNotEqual(out["outcome"],"SUCCEEDED")
        self.assertEqual(out["outcome"],"SUCCEEDED")
        for initial_on,contact in ((True,True),(False,False)):
            o=self.outcome("PRESS");f=frame();f["toggled"]=initial_on;o.update(f)
            for t in range(1,14):
                f=frame(t);f["toggled"]=True;f["finger_contact"]["right"]=contact;out=o.update(f)
            self.assertNotEqual(out["outcome"],"SUCCEEDED")

    def test_place_release_support_payload_and_unknown_volume(self):
        for mutation in (None,"supported","payload_ok","corners_inside"):
            o=self.outcome("PLACE_IN");f=frame();f["held"]["right"]=True;o.update(f)
            for t in range(1,14):
                f=frame(t)
                if mutation:f[mutation]=None if mutation=="corners_inside" else False
                out=o.update(f,"RIGHT_OPEN")
            self.assertEqual(out["outcome"]=="SUCCEEDED",mutation is None)

    def test_unknown_grasp_enum_is_not_truthy(self):
        class Unknown: name="UNKNOWN"
        self.assertIsNone(truth(Unknown()));self.assertIsNone(truth(-1));self.assertIsNone(truth(True))
        o=self.outcome("GRASP");f=frame();f["held"]["right"]=None
        self.assertEqual(o.update(f)["outcome"],"UNKNOWN")

    def test_rigid_and_supported_hand_corruption_fail(self):
        p=np.eye(4);p[0,0]=.1
        with self.assertRaises(ValueError):rigid(p)
        o=LocalOutcome({"verb":"PRESS","hand":"left","support_hand":"right","target":"target"})
        f=frame();f["held"]["right"]=True;o.update(f)
        f=frame(1);f["held"]["right"]=True;f["target_pose"][0][3]=.02
        self.assertEqual(o.update(f)["outcome"],"FAILED")

    def test_train_only_uses_real_counts_and_instance_exclusions(self):
        counts=json.loads((ROOT/"configs/vlm_sft/h09r_train_feasibility_counts.json").read_text())
        row=next(r for r in counts["sources"] if r["task"]==1 and r["episode"]==310)
        ref={"source":copy.deepcopy(row)}
        self.assertEqual(validate_train_group(ref,counts,[]),(1,192))
        with self.assertRaises(ValueError):validate_train_group(ref,counts,[[1,192]])
        ref["source"]["extracted_arrays_and_labels_sha256"]="0"*64
        with self.assertRaises(ValueError):validate_train_group(ref,counts,[])

    def test_private_fields_cannot_be_projected_by_actor_contract(self):
        request,_,_,_=test_native_teacher.NativeTeacherTests().fixture()
        legal=request["actor"]
        private={"target_uid":"radio_89","target_pose":np.eye(4).tolist(),"goal_status":True}
        projected=actor_input(legal["task"],legal["active_instruction"],legal["proprio"],legal["current_rgb_sha256"],[])
        self.assertFalse(set(private)&set(projected))
        bad=copy.deepcopy(legal["proprio"]);bad.update(private)
        with self.assertRaises(ValueError):actor_input(legal["task"],legal["active_instruction"],bad,legal["current_rgb_sha256"],[])

    def test_seed_requires_physical_completion_not_annotation_tail(self):
        spec={"verb":"PRESS","hand":"right","target":"target"}
        rows=[{"frame":frame(),"actual_action23":None}]
        with self.assertRaises(ValueError):extract_seed(spec,rows)
        for t in range(1,13):
            f=frame(t);f["toggled"]=True;f["finger_contact"]["right"]=True
            f["goal_parent_pose"]=np.eye(4).tolist()
            rows.append({"frame":f,"actual_action23":[0.]*23})
        seed=extract_seed(spec,rows)
        self.assertFalse(seed["is_native_bc_label"]);self.assertEqual(seed["physics_controls"],12)
        rows[5]["frame"]["tick"]=8
        with self.assertRaises(ValueError):extract_seed(spec,rows)

    def test_teacher_keeps_frozen_carry_rotation_restriction(self):
        spec={"verb":"GRASP","hand":"right"}
        t=PoseTeacher(spec);f=frame();goal=np.eye(4);goal[0,3]=.03
        ranked=t.ranked(None,f,goal,np.eye(4))
        self.assertEqual(ranked[0],"RIGHT_FORWARD")
        self.assertFalse(any("ROLL" in s or "PITCH" in s or "YAW" in s for s in ranked))

    def test_spec_cannot_change_original_intent(self):
        source={"verb":"GRASP","target":"target","destination":"","arm":"UNSPECIFIED"}
        ref={"private_original_semantic_json":json.dumps([source])}
        spec={"schema":"h09t-private-teacher-v1","verb":"GRASP","hand":"right","support_hand":None,
              "target":"target","destination":"","payloads":[],"goal_pose_local":np.eye(4).tolist(),
              "goal_frame":"target","pose_reviewer":"reviewer","pose_evidence_sha256":"a"*64}
        validate_spec(spec,ref)
        spec["verb"]="PRESS"
        with self.assertRaises(ValueError):validate_spec(spec,ref)


if __name__=="__main__":unittest.main()
