import copy
import unittest

import numpy as np

from semantic_robot.v2.finger_collision_asset import declared_fingers, export_stage, transform_mesh

try:
    from pxr import Gf, Usd, UsdGeom, UsdPhysics
except ImportError:
    Usd = None


class FingerCollisionAssetTests(unittest.TestCase):
    def setUp(self):
        self.points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
        self.counts = np.array([3, 3, 3, 3])
        self.indices = np.array([0, 2, 1, 0, 1, 3, 0, 3, 2, 1, 2, 3])
        self.T = np.diag([.01, .02, .03, 1.])
        self.definition = {"manipulation": {"n_arms": 2, "arm_names": ["left", "right"],
            "finger_link_names": {a: [a + "_f0", a + "_f1"] for a in ("left", "right")},
            "finger_joint_names": {a: [a + "_j0", a + "_j1"] for a in ("left", "right")}}}

    def test_nonuniform_scale_rotation_and_translation_applied_once(self):
        R = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
        T = self.T.copy(); T[:3, :3] = R @ T[:3, :3]; T[:3, 3] = [.04, -.02, .01]
        result = transform_mesh(self.points, self.counts, self.indices, T)
        expected = np.array([[.04, -.02, .01], [.04, -.01, .01], [.02, -.02, .01], [.04, -.02, .04]])
        np.testing.assert_allclose(result["vertices_link_m"], expected, atol=1e-14)
        np.testing.assert_array_equal(result["triangles"], self.indices.reshape(-1, 3))
        self.assertNotIn("contact", result)

    def test_bad_transform_or_units_are_not_replaced_by_a_box(self):
        edits = [lambda T: T.__setitem__((0, 0), -.01), lambda T: T.__setitem__((0, 0), 0),
                 lambda T: T.__setitem__((3, 0), 1), lambda T: T.__setitem__((0, 3), np.nan),
                 lambda T: T.__setitem__((0, 3), 10)]
        for edit in edits:
            T = self.T.copy(); edit(T)
            with self.subTest(edit=edit), self.assertRaises(ValueError):
                transform_mesh(self.points, self.counts, self.indices, T)

    def test_invalid_topology_and_degenerate_triangles_rejected(self):
        for counts, indices in (([4] * 4, self.indices), (self.counts, self.indices[:-1]),
                                (self.counts, self.indices.astype(float)),
                                (self.counts, [-1] + self.indices[1:].tolist()),
                                (self.counts, [4] + self.indices[1:].tolist()),
                                (self.counts, [0, 0, 1] + self.indices[3:].tolist())):
            with self.subTest(indices=indices), self.assertRaises(ValueError):
                transform_mesh(self.points, counts, indices, self.T)

    def test_exact_named_two_hand_definition_and_no_alias(self):
        result = declared_fingers(self.definition)
        self.definition["manipulation"]["finger_link_names"]["left"][0] = "modified"
        self.assertEqual(result["left"]["links"][0], "left_f0")
        for kind in ("finger_link_names", "finger_joint_names"):
            value = copy.deepcopy(self.definition)
            value["manipulation"][kind]["left"] = ["same", "same"]
            with self.assertRaises(ValueError): declared_fingers(value)
            value["manipulation"][kind]["left"] = ["scene/object", "finger"]
            with self.assertRaises(ValueError): declared_fingers(value)


@unittest.skipIf(Usd is None, "Real USD SDK is validated separately on robo, never replaced by a mock")
class NativeUSDExportTests(unittest.TestCase):
    def fixture(self):
        case = FingerCollisionAssetTests(); case.setUp()
        stage = Usd.Stage.CreateInMemory()
        root = UsdGeom.Xform.Define(stage, "/r1pro")
        stage.SetDefaultPrim(root.GetPrim()); UsdGeom.SetStageMetersPerUnit(stage, 1.)
        root.AddTranslateOp().Set(Gf.Vec3d(3, 4, 5)); root.AddRotateZOp().Set(30.)
        meshes = []
        for arm in ("left", "right"):
            for name in case.definition["manipulation"]["finger_link_names"][arm]:
                link = UsdGeom.Xform.Define(stage, "/r1pro/" + name)
                UsdPhysics.RigidBodyAPI.Apply(link.GetPrim())
                link.AddTranslateOp().Set(Gf.Vec3d(.1, -.2, .3)); link.AddRotateZOp().Set(90.)
                parent = UsdGeom.Xform.Define(stage, str(link.GetPath()) + "/collisions")
                parent.AddTranslateOp().Set(Gf.Vec3d(.001, .002, 0))
                mesh = UsdGeom.Mesh.Define(stage, str(parent.GetPath()) + "/tetrahedron")
                mesh.CreatePointsAttr().Set([Gf.Vec3f(*v) for v in case.points.tolist()])
                mesh.CreateFaceVertexCountsAttr().Set(case.counts.tolist())
                mesh.CreateFaceVertexIndicesAttr().Set(case.indices.tolist())
                mesh.AddTranslateOp().Set(Gf.Vec3d(.04, -.02, .01))
                mesh.AddScaleOp().Set(Gf.Vec3f(.01, .02, .03))
                UsdPhysics.CollisionAPI.Apply(mesh.GetPrim()).CreateCollisionEnabledAttr(True)
                UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr("convexHull")
                meshes.append(mesh)
        return stage, case.definition, meshes

    def export(self, stage, definition):
        return export_stage(stage, definition, Usd=Usd, UsdGeom=UsdGeom, UsdPhysics=UsdPhysics)

    def test_real_usd_parent_transforms_cancel_and_nonuniform_scale_is_applied_once(self):
        stage, definition, _ = self.fixture()
        result = self.export(stage, definition)
        self.assertEqual(len(result["links"]), 4)
        expected = np.array([[.041, -.018, .01], [.051, -.018, .01], [.041, .002, .01], [.041, -.018, .04]])
        for row in result["links"].values():
            self.assertEqual(set(row["meshes"]), {"collisions/tetrahedron"})
            np.testing.assert_allclose(row["meshes"]["collisions/tetrahedron"]["vertices_link_m"], expected, atol=2e-9, rtol=0)
        self.assertFalse(result["physical_contact_evidence"])
        self.assertFalse(result["runtime_cooked_collision_verified"])

    def test_real_usd_disabled_or_unexpected_approximation_is_rejected(self):
        for field in ("enabled", "approximation"):
            stage, definition, meshes = self.fixture()
            if field == "enabled":
                UsdPhysics.CollisionAPI(meshes[0].GetPrim()).GetCollisionEnabledAttr().Set(False)
            else:
                UsdPhysics.MeshCollisionAPI(meshes[0].GetPrim()).GetApproximationAttr().Set("none")
            with self.subTest(field=field), self.assertRaises(ValueError): self.export(stage, definition)

    def test_real_usd_visual_or_other_link_geometry_never_enters_finger_export(self):
        stage, definition, meshes = self.fixture()
        UsdGeom.Mesh.Define(stage, "/r1pro/left_f0/visuals")  # No collision API, intentionally no topology.
        cube = UsdGeom.Cube.Define(stage, "/r1pro/other_robot_link/not_a_finger")
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
        result = self.export(stage, definition)
        self.assertEqual(sum(len(v["meshes"]) for v in result["links"].values()), 4)
        UsdPhysics.CollisionAPI.Apply(stage.GetPrimAtPath("/r1pro/left_f0/visuals"))
        with self.assertRaises(ValueError): self.export(stage, definition)

    def test_real_usd_missing_named_link_or_wrong_unit_is_rejected(self):
        stage, definition, _ = self.fixture()
        UsdGeom.SetStageMetersPerUnit(stage, .01)
        with self.assertRaises(ValueError): self.export(stage, definition)
        UsdGeom.SetStageMetersPerUnit(stage, 1.)
        stage.RemovePrim("/r1pro/right_f1")
        with self.assertRaises(ValueError): self.export(stage, definition)

    def test_real_usd_nested_rigid_body_is_not_owned_by_named_finger(self):
        for path in ("/r1pro/left_f0/collisions", "/r1pro/left_f0/collisions/tetrahedron"):
            stage, definition, _ = self.fixture()
            UsdPhysics.RigidBodyAPI.Apply(stage.GetPrimAtPath(path))
            with self.subTest(path=path), self.assertRaisesRegex(ValueError, "nested rigid body"):
                self.export(stage, definition)


if __name__ == "__main__": unittest.main()
