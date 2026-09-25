"""Authored R1Pro collision mesh export, NOT a live PhysX/contact certificate.

This module never imports a simulator. USD bindings are supplied by the reader;
only named robot finger links are read, with all mesh scales applied exactly.
"""
import numpy as np


SOURCE = "robot_asset_authored_convex_finger_meshes"


def transform_mesh(points, counts, indices, T_link_mesh):
    points = np.asarray(points, dtype=float)
    counts, indices = np.asarray(counts), np.asarray(indices)
    T = np.asarray(T_link_mesh, dtype=float)
    if (points.ndim != 2 or points.shape[1] != 3 or not 4 <= len(points) <= 4096 or
            not np.isfinite(points).all() or T.shape != (4, 4) or not np.isfinite(T).all() or
            not np.allclose(T[3], [0, 0, 0, 1], atol=1e-9, rtol=0) or
            np.linalg.det(T[:3, :3]) <= 1e-12):
        raise ValueError("Finite non-reflected, non-singular robot mesh transform required")
    if (counts.ndim != 1 or not 4 <= len(counts) <= 8192 or
            not np.issubdtype(counts.dtype, np.integer) or not np.all(counts == 3) or
            indices.ndim != 1 or len(indices) != 3 * len(counts) or
            not np.issubdtype(indices.dtype, np.integer) or
            np.any(indices < 0) or np.any(indices >= len(points))):
        raise ValueError("Exact indexed triangle topology required; no box/triangulation fallback")
    vertices = points @ T[:3, :3].T + T[:3, 3]
    if np.max(np.abs(vertices)) > .5:
        raise ValueError("Robot finger link-local mesh outside half-metre sanity bound")
    faces = indices.reshape(-1, 3)
    triangles = vertices[faces]
    areas2 = np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]), axis=1)
    if np.any(areas2 <= 1e-14):
        raise ValueError("Degenerate authored collision triangle")
    return {"vertices_link_m": vertices.tolist(), "triangles": faces.tolist(),
            "T_link_mesh": T.tolist(), "bounds_link_m": [vertices.min(axis=0).tolist(), vertices.max(axis=0).tolist()]}


def declared_fingers(definition):
    manipulation = definition["manipulation"]
    if manipulation["arm_names"] != ["left", "right"] or manipulation["n_arms"] != 2:
        raise ValueError("Audited two-arm R1Pro robot definition required")
    links = manipulation["finger_link_names"]
    joints = manipulation["finger_joint_names"]
    if set(links) != {"left", "right"} or set(joints) != {"left", "right"}:
        raise ValueError("Both hands must be explicitly named")
    for mapping in (links, joints):
        names = []
        for arm in ("left", "right"):
            if not isinstance(mapping[arm], list) or len(mapping[arm]) != 2:
                raise ValueError("Two independently named fingers per hand required")
            names.extend(mapping[arm])
        if (any(not isinstance(n, str) or not n or "/" in n for n in names) or len(set(names)) != 4):
            raise ValueError("Four distinct direct robot finger names required")
    return {arm: {"links": list(links[arm]), "joints": list(joints[arm])} for arm in ("left", "right")}


def export_stage(stage, definition, *, Usd, UsdGeom, UsdPhysics):
    fingers = declared_fingers(definition)
    root = stage.GetDefaultPrim()
    if not root or root.GetName() != "r1pro" or UsdGeom.GetStageMetersPerUnit(stage) != 1.:
        raise ValueError("Metre-scale R1Pro asset root required")
    cache = UsdGeom.XformCache()
    result = {"version": 1, "source": SOURCE, "frame": "named_finger_link_local",
              "scene_truth": False, "current_joint_positions": None,
              "runtime_cooked_collision_verified": False, "physical_contact_evidence": False,
              "robot_definition": fingers, "links": {}}
    for arm in ("left", "right"):
        for name in fingers[arm]["links"]:
            link = root.GetChild(name)
            if not link or not link.HasAPI(UsdPhysics.RigidBodyAPI):
                raise ValueError("Named articulated robot finger link is missing")
            world_link = np.asarray(cache.GetLocalToWorldTransform(link), dtype=float).T
            meshes = {}
            for prim in Usd.PrimRange(link):
                if not prim.HasAPI(UsdPhysics.CollisionAPI):
                    continue
                owner = prim
                for _ in range(16):
                    if owner == link:
                        break
                    if not owner or owner.HasAPI(UsdPhysics.RigidBodyAPI):
                        raise ValueError("Collision has a nested rigid body, not the named finger owner")
                    owner = owner.GetParent()
                else:
                    raise ValueError("Collision ancestry does not reach its declared finger link")
                enabled = UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get()
                approximation = UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get()
                if enabled is not True or not prim.IsA(UsdGeom.Mesh) or approximation != "convexHull":
                    raise ValueError("Expected enabled authored convex-hull finger meshes only")
                mesh = UsdGeom.Mesh(prim)
                world_mesh = np.asarray(cache.GetLocalToWorldTransform(prim), dtype=float).T
                T = np.linalg.solve(world_link, world_mesh)
                record = transform_mesh(mesh.GetPointsAttr().Get(), mesh.GetFaceVertexCountsAttr().Get(),
                                        mesh.GetFaceVertexIndicesAttr().Get(), T)
                key = str(prim.GetPath().MakeRelativePath(link.GetPath()))
                if key in meshes:
                    raise ValueError("Duplicate finger collision identity")
                meshes[key] = {**record, "authored_approximation": str(approximation),
                               "collision_enabled": enabled}
            if not 1 <= len(meshes) <= 64:
                raise ValueError("Bounded nonempty finger collision geometry required")
            result["links"][name] = {"arm": arm, "meshes": meshes}
    return result
