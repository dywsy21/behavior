import copy
from dataclasses import replace
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from scipy.spatial import ConvexHull

from semantic_robot.v2.finger_collision_asset import SOURCE
from semantic_robot.v2.press_reference import FingerSurfaceAsset, MARGIN_M, compile_piece, segment_intersects_piece
from test_finger_kinematics import calibrated_fixture, current


def box(low=(-.01, -.008, -.03), high=(.01, .008, .03)):
    vertices = np.array([[x, y, z] for x in (low[0], high[0]) for y in (low[1], high[1]) for z in (low[2], high[2])])
    return {"vertices_link_m": vertices.tolist(), "triangles": ConvexHull(vertices).simplices.tolist(),
            "authored_approximation": "convexHull", "collision_enabled": True}


def surface_spec():
    return {"version": 1, "source": SOURCE, "frame": "named_finger_link_local", "scene_truth": False,
            "current_joint_positions": None, "runtime_cooked_collision_verified": False,
            "physical_contact_evidence": False,
            "robot_definition": {a: {"links": [a+"_f0", a+"_f1"], "joints": [a+"_j0", a+"_j1"]} for a in ("left", "right")},
            "links": {a+"_f"+str(i): {"arm": a, "meshes": {"box": box()}} for a in ("left", "right") for i in (0, 1)}}


class PressReferenceTests(unittest.TestCase):
    def setUp(self):
        self.model, self.state = calibrated_fixture()
        self.asset = FingerSurfaceAsset(surface_spec(), "a"*64)
        _, self.transforms = self.asset.context(self.model, self.state.q, self.state.finger_qpos)
        self.target = (self.transforms["right_f0"] @ [.06, 0, 0, 1])[:3]

    def bind(self, asset=None, target=None):
        return (asset or self.asset).bind(self.model, self.state.q, self.state.finger_qpos,
                                         arm="right", goal_key=(0, "press", "visible switch", "right"),
                                         target_base=self.target if target is None else target)

    def test_coplanar_triangles_merge_and_point_uses_inset_planar_surface(self):
        self.assertEqual(len(compile_piece(box())["patches"]), 6)
        ref = self.bind()
        self.assertEqual(ref.link, "right_f0")
        np.testing.assert_allclose(ref.point_link_m, [.01, 0, 0], atol=1e-12)
        row = self.asset.resolve(ref, self.model, self.state.q, self.state.finger_qpos)
        self.assertTrue(row["not_physx_face_identity"])
        self.assertFalse(row["physical_contact_evidence"])
        self.assertFalse(row["runtime_cooked_collision_verified"])

    def test_nearest_patch_point_stays_one_mm_from_real_patch_edge(self):
        target = (self.transforms["right_f0"] @ [.1, .0078, .025, 1])[:3]
        ref = self.bind(target=target)
        self.assertLessEqual(ref.point_link_m[1], .008-MARGIN_M+1e-12)

    def test_buried_piece_surface_is_not_selected(self):
        spec = surface_spec()
        spec["links"]["right_f0"]["meshes"]["cover"] = box((.005, -.012, -.02), (.025, .012, .02))
        asset = FingerSurfaceAsset(spec, "b"*64)
        ref = self.bind(asset)
        self.assertEqual(ref.mesh, "cover")
        np.testing.assert_allclose(ref.point_link_m, [.025, 0, 0], atol=1e-12)
        buried = (self.transforms["right_f0"] @ [.01, 0, 0, 1])[:3]
        self.assertFalse(asset.exposed(buried, "right", "right_f0", "box", self.transforms))

    def test_target_inside_any_finger_abstains_instead_of_selecting_back_surface(self):
        target = self.transforms["right_f0"][:3, 3]
        self.assertIsNone(self.bind(target=target))

    def test_thin_patches_have_no_reference_not_box_center_fallback(self):
        spec = surface_spec()
        for row in spec["links"].values(): row["meshes"] = {"tiny": box((-.0004,)*3, (.0004,)*3)}
        self.assertIsNone(self.bind(FingerSurfaceAsset(spec, "b"*64)))

    def test_same_mean_different_named_joint_moves_same_local_point(self):
        ref = self.bind()
        first = self.asset.resolve(ref, self.model, self.state.q, current(right=(.03, .05)))
        second = self.asset.resolve(ref, self.model, self.state.q, current(right=(.04, .04)))
        self.assertEqual(first["reference_id"], second["reference_id"])
        np.testing.assert_allclose(np.subtract(second["point_base_m"], first["point_base_m"]), [0, .01, 0], atol=1e-12)

    def test_real_arm_plan_fk_moves_reference_not_closing_center(self):
        ref = self.bind(); q = self.state.q.copy(); q[11] += .02
        before = self.asset.resolve(ref, self.model, self.state.q, self.state.finger_qpos)
        after = self.asset.resolve(ref, self.model, q, self.state.finger_qpos)
        np.testing.assert_allclose(np.subtract(after["point_base_m"], before["point_base_m"]), [.02, 0, 0], atol=1e-12)
        self.assertEqual(before["reference_id"], after["reference_id"])

    def test_current_closure_hiding_reference_is_rejected(self):
        ref = self.bind()
        with self.assertRaisesRegex(ValueError, "obscured"):
            self.asset.resolve(ref, self.model, self.state.q, current(right=(0, 0)))

    def test_missing_or_mean_only_proprio_never_reuses_open_reference(self):
        for positions in (None, {}, {"right": .05, "left": .05}, {**current(), "fake": 0}):
            with self.subTest(positions=positions), self.assertRaises(ValueError):
                self.asset.context(self.model, self.state.q, positions)

    def test_whole_robot_calibration_and_asset_identity_are_bound(self):
        ref = self.bind()
        self.model.links["right"][0][0, 3] += .001
        with self.assertRaisesRegex(ValueError, "Exact asset/calibration"):
            self.asset.resolve(ref, self.model, self.state.q, self.state.finger_qpos)
        with self.assertRaises(ValueError):
            self.asset.resolve(replace(ref, asset_sha256="c"*64), self.model, self.state.q, current())

    def test_forged_interior_point_and_wrong_hand_are_rejected(self):
        ref = self.bind()
        for changed in (replace(ref, point_link_m=(0, 0, 0)), replace(ref, arm="left")):
            with self.assertRaises(ValueError): self.asset.resolve(changed, self.model, self.state.q, current())

    def test_geometry_flags_ownership_and_topology_are_checked(self):
        edits = [lambda s: s.update(version=True), lambda s: s.update(scene_truth=True),
                 lambda s: s.update(physical_contact_evidence=True),
                 lambda s: s["links"]["right_f0"].update(arm="left"),
                 lambda s: s["links"]["right_f0"]["meshes"]["box"].update(collision_enabled=False),
                 lambda s: s["links"]["right_f0"]["meshes"]["box"]["triangles"].pop()]
        for edit in edits:
            spec = surface_spec(); edit(spec)
            with self.subTest(edit=edit), self.assertRaises(ValueError): FingerSurfaceAsset(spec, "a"*64)

    def test_segment_clipping_checks_interior_parallel_and_clearance(self):
        piece=compile_piece(box())
        self.assertTrue(segment_intersects_piece([-.1,0,0],[.1,0,0],piece))
        self.assertTrue(segment_intersects_piece([-.1,.0085,0],[.1,.0085,0],piece))
        self.assertFalse(segment_intersects_piece([-.1,.0101,0],[.1,.0101,0],piece))
        self.assertFalse(segment_intersects_piece([.02,0,0],[.03,0,0],piece))

    def test_thin_piece_without_patch_still_blocks_target_channel(self):
        original=self.bind();spec=surface_spec()
        spec["links"]["right_f0"]["meshes"]["thin_blocker"]=box((.020,-.0004,-.0004),(.022,.0004,.0004))
        asset=FingerSurfaceAsset(spec,"a"*64)
        self.assertEqual(asset.links["right_f0"]["thin_blocker"]["patches"],[])
        self.assertTrue(asset.exposed((self.transforms["right_f0"]@[.01,0,0,1])[:3],
                                     "right","right_f0","box",self.transforms))
        with self.assertRaisesRegex(ValueError,"segment obscured"):
            asset.resolve(original,self.model,self.state.q,self.state.finger_qpos,target_base=self.target)
        candidate=self.bind(asset)
        if candidate is not None:
            self.assertNotEqual(candidate,original)
            asset.resolve(candidate,self.model,self.state.q,self.state.finger_qpos,target_base=self.target)


class PressControllerTests(unittest.TestCase):
    setUp = PressReferenceTests.setUp

    def controller(self, kind="press", enabled=True):
        from semantic_robot.v2.grounded_harness import GroundedController, GroundedHarness
        from semantic_robot.v2.harness import Goal
        from semantic_robot.v2.servo import SafeServo
        manager=GroundedHarness([Goal(kind,"visible switch","right","visible effect")])
        servo=SafeServo(self.model,self.state,[-1,-1])
        return GroundedController(self.model,servo,manager,press_finger_surfaces=self.asset if enabled else None)

    def observe(self, controller, target=None, **evidence):
        from test_grounded import grounded_evidence
        depths={v:np.full((100,100),1.2,np.float32) for v in ("head","left_wrist","right_wrist")}
        receipt={v:{"valid_fraction":1.} for v in depths}
        point=self.target if target is None else target
        localized={"valid":True,"point_base_m":point.tolist(),"views":[]}
        # Synthetic perception only; the actual controller/FK/servo run unchanged.
        with patch("semantic_robot.v2.grounded_harness.localize_target",return_value=localized):
            controller.observe(grounded_evidence(**evidence),self.state,depths,receipt)

    def test_actual_observe_and_joint_plan_use_the_same_surface_point(self):
        from semantic_robot.v2.protocol import Action
        c=self.controller();self.observe(c)
        ref=c.press_reference;self.assertIsNotNone(ref)
        row=self.asset.resolve(ref,self.model,self.state.q,self.state.finger_qpos)
        expected=float(np.linalg.norm(self.target-row["point_base_m"]))
        self.assertAlmostEqual(c.target["distance_to_active_reference_m"],expected)
        self.assertNotIn("distance_to_active_closing_center_m",c.target)
        self.assertIn("target_minus_reference_tool_m",c.target)
        q=self.state.q.copy();q[11]+=.02
        after=self.asset.resolve(ref,self.model,q,self.state.finger_qpos)
        trial=SimpleNamespace(joint_plan=np.array([self.state.q,q]))
        distances=c._expected_distances(Action("right","forward"),self.state,trial)
        self.assertAlmostEqual(distances["right"],float(np.linalg.norm(self.target-after["point_base_m"])))
        self.assertIs(c.press_reference,ref)

    def test_new_target_pixel_does_not_rebind_to_other_finger(self):
        c=self.controller();self.observe(c);ref=c.press_reference
        second=(self.transforms["right_f1"] @ [.06,0,0,1])[:3]
        self.observe(c,second)
        self.assertIs(c.press_reference,ref)
        self.assertEqual(c.target["press_reference"]["reference_id"],ref.record()["reference_id"])

    def test_same_goal_target_behind_fixed_face_blocks_without_rebinding(self):
        from semantic_robot.v2.protocol import HOLD,Action
        c=self.controller();self.observe(c);ref=c.press_reference
        behind=(self.transforms[ref.link]@[-.06,0,0,1])[:3]
        self.observe(c,behind,effect=True)
        self.assertIs(c.press_reference,ref)
        self.assertFalse(c.target["valid"])
        self.assertEqual(c.candidates(self.state),(HOLD,))
        self.assertFalse(c.press_execution_check(self.state,Action("right","back"))["eligible"])
        self.assertIsNone(c.harness.observation.effect)

    def test_same_goal_new_target_channel_is_checked_again(self):
        from semantic_robot.v2.protocol import HOLD
        spec=surface_spec()
        spec["links"]["right_f0"]["meshes"]["thin_blocker"]=box((.020,-.0004,-.0004),(.022,.0004,.0004))
        self.asset=FingerSurfaceAsset(spec,"a"*64)
        c=self.controller()
        self.observe(c,(self.transforms["right_f0"]@[.06,-.006,0,1])[:3]);ref=c.press_reference
        np.testing.assert_allclose(ref.point_link_m,[.01,-.006,0],atol=1e-12)
        self.observe(c,(self.transforms["right_f0"]@[.06,.024,0,1])[:3])
        self.assertIs(c.press_reference,ref)
        self.assertFalse(c.target["valid"])
        self.assertIn("segment obscured",c.target["press_reference"]["reason"])
        self.assertEqual(c.candidates(self.state),(HOLD,))

    def test_real_plan_behind_face_is_rejected_not_just_missing_prediction(self):
        from semantic_robot.v2.protocol import Action
        c=self.controller();self.observe(c)
        q=self.state.q.copy();q[11]+=.08
        trial=SimpleNamespace(joint_plan=np.asarray([q]))
        action=Action("right","forward")
        self.assertIsNone(c._expected_distances(action,self.state,trial))
        self.assertFalse(c.press_trial_check(self.state,action,trial)["eligible"])
        good=self.state.q.copy();good[11]+=.01
        # An acceptable endpoint cannot hide a bad intermediate servo point.
        loop=SimpleNamespace(joint_plan=np.asarray([q,good]))
        self.assertIsNotNone(c._expected_distances(action,self.state,loop))
        self.assertFalse(c.press_trial_check(self.state,action,loop)["eligible"])

    def test_press_gripper_change_and_unmodeled_arm_plan_are_vetoed(self):
        from semantic_robot.v2.protocol import Action
        c=self.controller();self.observe(c)
        trial=SimpleNamespace(joint_plan=None)
        for action in (Action("right","close"),Action("right","open"),Action("right","forward")):
            self.assertFalse(c.press_trial_check(self.state,action,trial)["eligible"])

    def test_next_semantic_goal_creates_its_own_reference(self):
        from semantic_robot.v2.harness import Goal
        c=self.controller();self.observe(c);ref=c.press_reference
        c.harness.goals.append(Goal("press","second switch","right","visible effect"));c.harness.index=1
        c.harness.stage="SEARCH"
        second=(self.transforms["right_f1"] @ [.06,0,0,1])[:3]
        self.observe(c,second)
        self.assertNotEqual(c.press_reference.record()["reference_id"],ref.record()["reference_id"])
        self.assertEqual(c.press_reference.link,"right_f1")

    def test_active_loaded_hand_blocks_and_cannot_confirm_effect(self):
        from semantic_robot.v2.protocol import Action,HOLD
        for field in ("held","hold_verified","pending_grasp","possible_contact_after_close"):
            c=self.controller();getattr(c.harness,field)["right"]="object" if field=="held" else True
            c.harness.stage="VERIFY_EFFECT";c.harness.confirmations=1;c.harness.last_action=Action("right","forward")
            self.observe(c,effect=True)
            self.assertFalse(c.target["valid"])
            self.assertIsNone(c.centers["right"])
            self.assertEqual(c.candidates(self.state),(HOLD,))
            self.assertEqual(c.harness.completed,[])
            self.assertIsNone(c.harness.observation.effect)

    def test_other_hand_load_is_preserved_and_not_used_as_press_hand(self):
        c=self.controller();c.harness.held["left"]="radio";c.harness.hold_verified["left"]=True
        self.observe(c)
        self.assertTrue(c.target["press_reference"]["valid"])
        self.assertEqual(c.press_reference.arm,"right")
        self.assertEqual(c.harness.held["left"],"radio")
        self.assertTrue(c.harness.hold_verified["left"])

    def test_gripper_change_and_unplanned_rotation_do_not_guess_future_surface(self):
        from semantic_robot.v2.protocol import Action
        c=self.controller();self.observe(c)
        for action in (Action("right","open"),Action("right","close"),Action("right","yaw_plus","micro","tool")):
            self.assertIsNone(c._expected_distances(action,self.state))

    def test_missing_finger_proprio_is_hold_not_closing_center_fallback(self):
        from semantic_robot.v2.protocol import HOLD
        c=self.controller();self.state=replace(self.state,finger_qpos=None)
        self.observe(c,effect=True)
        self.assertFalse(c.target["valid"])
        self.assertIsNone(c.centers["right"])
        self.assertEqual(c.candidates(self.state),(HOLD,))

    def test_execution_rechecks_same_mean_changed_fingers_and_late_load(self):
        from semantic_robot.v2.protocol import Action,HOLD
        self.state=replace(self.state,finger_qpos=current(right=(.04,.04)))
        c=self.controller();self.observe(c)
        action=Action("right","forward")
        self.assertTrue(c.press_execution_check(self.state,action)["eligible"])
        changed=replace(self.state,finger_qpos=current(right=(.03,.05)))
        self.assertFalse(c.press_execution_check(changed,action)["eligible"])
        self.assertTrue(c.press_execution_check(changed,HOLD)["eligible"])
        c.harness.pending_grasp["right"]=True
        self.assertFalse(c.press_execution_check(self.state,action)["eligible"])

    def test_execution_rechecks_arm_drift_and_goal_even_with_same_reference(self):
        from semantic_robot.v2.protocol import Action
        c=self.controller();self.observe(c)
        q=self.state.q.copy();q[11]+=.001
        self.assertFalse(c.press_execution_check(replace(self.state,q=q),Action("right","forward"))["eligible"])
        c.harness.goals.append(copy.deepcopy(c.harness.goal));c.harness.index=1
        self.assertFalse(c.press_execution_check(self.state,Action("base","back"))["eligible"])

    def test_actual_runner_checks_after_model_latency_before_servo(self):
        import ast
        from pathlib import Path
        tree=ast.parse((Path(__file__).resolve().parents[2]/"scripts/semantic_robot/run_v2.py").read_text())
        checks=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)
                and n.func.attr=="press_execution_check"]
        begins=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)
                and n.func.attr=="begin" and isinstance(n.func.value,ast.Name) and n.func.value.id=="servo"]
        self.assertEqual(len(checks),1)
        self.assertTrue(any(checks[0].lineno<b.lineno<checks[0].lineno+45 for b in begins))

    def test_disabled_press_retains_old_action_policy_prompt_path(self):
        c=self.controller(enabled=False);self.observe(c)
        self.assertFalse(c.uses_press_surface)
        self.assertIsNone(c.press_snapshot)

    def test_nonpress_and_disabled_press_preserve_legacy_geometry(self):
        for kind,enabled in (("pick",True),("press",False)):
            c=self.controller(kind,enabled);self.observe(c)
            self.assertNotIn("press_reference",c.target)
            self.assertIn("distance_to_active_closing_center_m",c.target)
            for arm,p in self.model.grasp_centers(self.state.q).items():
                np.testing.assert_array_equal(c.centers[arm],p)


if __name__ == "__main__": unittest.main()
