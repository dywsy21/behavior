"""Stable authored finger-surface references, NOT contact or cooked geometry.

The point is explicit link-local XYZ, not a PhysX face identity. We use inset
coplanar patches and reject points near other pieces, including the other jaw.
"""
import copy
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial import ConvexHull

from semantic_robot.control import finite
from .finger_collision_asset import SOURCE, transform_mesh
from .finger_kinematics import FingerKinematics, rigid


EXPORT_SHA = "5f2cbf18b989eb22bc98665f48538c6cc306679d3be94ab4de46f2af7b4c6f04"
MARGIN_M = .001  # Engineering clearance, not a physical contact/success threshold.


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def inset_polygon(polygon, equations, margin):
    polygon = np.asarray(polygon, dtype=float)
    for equation in equations:
        output = []
        if not len(polygon): break
        previous = polygon[-1]
        before = previous @ equation[:2] + equation[2] + margin
        for point in polygon:
            after = point @ equation[:2] + equation[2] + margin
            if (after <= 0) != (before <= 0):
                output.append(previous + before / (before-after) * (point-previous))
            if after <= 0: output.append(point)
            previous, before = point, after
        polygon = np.asarray(output, dtype=float).reshape(-1, 2)
    return polygon


def nearest_polygon(point, polygon, equations):
    if np.all(equations[:, :2] @ point + equations[:, 2] + MARGIN_M <= 1e-12):
        return point
    ends = np.roll(polygon, -1, axis=0)
    edges = ends-polygon
    lengths2 = np.einsum("ij,ij->i", edges, edges)
    valid = lengths2 > 1e-20
    if not valid.any(): raise ValueError("Degenerate inset patch")
    starts, edges, lengths2 = polygon[valid], edges[valid], lengths2[valid]
    t = np.clip(np.einsum("ij,ij->i", point-starts, edges)/lengths2, 0, 1)
    candidates = starts + t[:, None]*edges
    return candidates[np.argmin(np.sum((candidates-point)**2, axis=1))]


def compile_piece(row):
    if row.get("authored_approximation") != "convexHull" or row.get("collision_enabled") is not True:
        raise ValueError("Enabled authored convex mesh required")
    faces = np.asarray(row["triangles"])
    if faces.ndim != 2 or faces.shape[1] != 3: raise ValueError("Triangle indices required")
    mesh = transform_mesh(row["vertices_link_m"], np.full(len(faces), 3), faces.ravel(), np.eye(4))
    vertices, faces = np.asarray(mesh["vertices_link_m"]), np.asarray(mesh["triangles"])
    edges = np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1)
    if np.any(np.unique(edges, axis=0, return_counts=True)[1] != 2):
        raise ValueError("Closed authored mesh required")
    triangles = vertices[faces]
    normals = np.cross(triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0])
    normals /= np.linalg.norm(normals, axis=1)[:, None]
    inward = np.einsum("ij,ij->i", vertices.mean(axis=0)-triangles[:, 0], normals) > 0
    normals[inward] *= -1
    offsets = np.einsum("ij,ij->i", normals, triangles[:, 0])
    if np.max(normals @ vertices.T-offsets[:, None]) > 2e-8:
        raise ValueError("Authored mesh is not a convex boundary")
    groups = []
    for index, (normal, offset) in enumerate(zip(normals, offsets)):
        group = next((g for g in groups if np.linalg.norm(normal-g[0]) <= 1e-6 and abs(offset-g[1]) <= 1e-8), None)
        if group is None: groups.append([normal.copy(), offset, [index]])
        else: group[2].append(index)
    patches = []
    for normal, offset, indices in groups:
        axis = np.eye(3)[np.argmin(np.abs(normal))]
        u = np.cross(axis, normal); u /= np.linalg.norm(u)
        basis = np.column_stack([u, np.cross(normal, u)])
        points = vertices[np.unique(faces[indices])]
        xy = points @ basis
        hull = ConvexHull(xy)
        polygon = inset_polygon(xy[hull.vertices], hull.equations, MARGIN_M)
        if len(polygon) < 3: continue
        area2 = abs(np.sum(polygon[:, 0]*np.roll(polygon[:, 1], -1)-polygon[:, 1]*np.roll(polygon[:, 0], -1)))
        if area2 < 1e-12: continue
        patches.append({"normal": normal, "offset": offset, "basis": basis,
                        "polygon": polygon, "equations": hull.equations})
    return {"normals": normals, "offsets": offsets, "patches": patches}


def segment_intersects_piece(start, end, piece, margin=MARGIN_M):
    """Clip a segment against outward halfspaces expanded by margin.

    This expansion conservatively contains the Euclidean clearance shell. A
    tangent is rejected too; this is not an environment/swept-robot certificate.
    All collision pieces matter, including those with no usable inset patch.
    """
    start, end = finite(start, (3,)), finite(end, (3,))
    values = piece["normals"] @ start-piece["offsets"]-margin
    slopes = piece["normals"] @ (end-start)
    lo, hi = 0., 1.
    for value, slope in zip(values, slopes):
        if abs(slope) <= 1e-14:
            if value > 1e-12: return False
        elif slope > 0:
            hi = min(hi, -value/slope)
        else:
            lo = max(lo, -value/slope)
        if lo > hi+1e-12: return False
    return True


@dataclass(frozen=True)
class SurfaceReference:
    asset_sha256: str
    calibration_sha256: str
    goal_key: tuple
    arm: str
    link: str
    mesh: str
    authored_patch: int
    point_link_m: tuple

    def record(self):
        row = asdict(self)
        return {**row, "reference_id": digest(row), "frame": "named_finger_link_local",
                "source": "authored_finger_union_exposed_patch",
                "margin_m": MARGIN_M, "not_physx_face_identity": True,
                "runtime_cooked_collision_verified": False, "physical_contact_evidence": False}


class FingerSurfaceAsset:
    def __init__(self, surfaces, asset_sha256):
        if (surfaces.get("source") != SOURCE or type(surfaces.get("version")) is not int or surfaces.get("version") != 1 or
                surfaces.get("frame") != "named_finger_link_local" or surfaces.get("scene_truth") is not False or
                surfaces.get("current_joint_positions") is not None or
                surfaces.get("runtime_cooked_collision_verified") is not False or
                surfaces.get("physical_contact_evidence") is not False):
            raise ValueError("Robot-only authored finger surfaces required")
        if not isinstance(asset_sha256, str) or len(asset_sha256) != 64 or any(c not in "0123456789abcdef" for c in asset_sha256):
            raise ValueError("Explicit geometry source SHA required")
        self.asset_sha256 = asset_sha256
        self.definition = copy.deepcopy(surfaces["robot_definition"])
        if set(self.definition) != {"left", "right"}: raise ValueError("Both named hands required")
        names = [n for a in ("left", "right") for n in self.definition[a]["links"]]
        if len(names) != 4 or len(set(names)) != 4 or set(names) != set(surfaces["links"]):
            raise ValueError("Four uniquely named finger links required")
        self.links = {}
        for arm in ("left", "right"):
            if len(self.definition[arm]["links"]) != 2 or len(self.definition[arm]["joints"]) != 2:
                raise ValueError("Two named links/joints per hand required")
            for name in self.definition[arm]["links"]:
                row = surfaces["links"][name]
                if row["arm"] != arm or not 1 <= len(row["meshes"]) <= 64:
                    raise ValueError("Bounded meshes owned by the declared hand required")
                self.links[name] = {key: compile_piece(mesh) for key, mesh in row["meshes"].items()}

    def context(self, model, q, positions):
        spec = model.spec.get("metadata", {}).get("finger_kinematics")
        calibrated = FingerKinematics(spec)
        for arm, (names, _, _, _, links) in calibrated.arms.items():
            if (list(names) != self.definition[arm]["joints"] or
                    set(links) != set(self.definition[arm]["links"])):
                raise ValueError("Asset and named finger calibration do not match")
        geometry = calibrated.geometry({a: model.forward(q, a) for a in ("left", "right")}, positions)
        transforms = {name: rigid(item["T_base_link"]) for arm in geometry["arms"].values() for name, item in arm.items()}
        # The source identity plus ACTUAL active FK arrays catches mutated
        # control geometry without re-hashing a many-MB visual mesh per trial.
        identity={"source_calibration_sha256":model.sha,"finger":spec,
                  "q_reference":model.reference.tolist(),
                  "eef_links":{a:[x.tolist() for x in model.links[a]] for a in ("left","right")}}
        return digest(identity), transforms

    def exposed(self, point_base, arm, own_link, own_mesh, transforms):
        for name in self.definition[arm]["links"]:
            T = transforms[name]
            local = T[:3, :3].T @ (point_base-T[:3, 3])
            for key, piece in self.links[name].items():
                if (name, key) == (own_link, own_mesh): continue
                # max plane separation is a conservative lower bound on distance
                # to this convex body. It also rejects buried and seam points.
                if np.max(piece["normals"] @ local-piece["offsets"]) < MARGIN_M-1e-10:
                    return False
        return True

    def check_target(self, point_base, normal_base, target_base, arm, own_link, own_mesh, transforms):
        point, target = finite(point_base, (3,)), finite(target_base, (3,))
        if (target-point) @ finite(normal_base, (3,)) <= 1e-8:
            raise ValueError("Current target is behind or tangent to the fixed press face")
        for name in self.definition[arm]["links"]:
            T = transforms[name]
            start = T[:3, :3].T @ (point-T[:3, 3])
            end = T[:3, :3].T @ (target-T[:3, 3])
            for key, piece in self.links[name].items():
                if (name, key) == (own_link, own_mesh): continue
                if segment_intersects_piece(start, end, piece):
                    raise ValueError("Fixed press point-to-target segment obscured by another finger piece")

    def bind(self, model, q, positions, *, arm, goal_key, target_base):
        if arm not in ("left", "right"): raise ValueError("A press primitive uses one free hand")
        target = finite(target_base, (3,))
        calibration, transforms = self.context(model, q, positions)
        candidates = []
        for name in self.definition[arm]["links"]:
            T = transforms[name]; local = T[:3, :3].T @ (target-T[:3, 3])
            for key, piece in self.links[name].items():
                if np.max(piece["normals"] @ local-piece["offsets"]) <= 2e-8:
                    return None  # The observed target is inside the finger union.
                for index, patch in enumerate(piece["patches"]):
                    point2 = nearest_polygon(local @ patch["basis"], patch["polygon"], patch["equations"])
                    point = patch["normal"]*patch["offset"] + patch["basis"] @ point2
                    # Use a target-facing patch; no approach through this body.
                    if (local-point) @ patch["normal"] <= 1e-8: continue
                    world = T[:3, :3] @ point+T[:3, 3]
                    if self.exposed(world, arm, name, key, transforms):
                        try:
                            self.check_target(world, T[:3, :3] @ patch["normal"], target, arm, name, key, transforms)
                        except ValueError:
                            continue
                        ref = SurfaceReference(self.asset_sha256, calibration, tuple(goal_key), arm, name, key, index, tuple(point))
                        candidates.append((float(np.linalg.norm(world-target)), name, key, index, ref))
        return min(candidates, key=lambda x: x[:4])[-1] if candidates else None

    def resolve(self, reference, model, q, positions, *, target_base=None):
        calibration, transforms = self.context(model, q, positions)
        if (not isinstance(reference, SurfaceReference) or reference.asset_sha256 != self.asset_sha256 or
                reference.calibration_sha256 != calibration or reference.arm not in self.definition or
                reference.link not in self.definition[reference.arm]["links"] or
                reference.mesh not in self.links[reference.link] or type(reference.authored_patch) is not int or
                not 0 <= reference.authored_patch < len(self.links[reference.link][reference.mesh]["patches"])):
            raise ValueError("Exact asset/calibration reference required")
        patch = self.links[reference.link][reference.mesh]["patches"][reference.authored_patch]
        point = finite(reference.point_link_m, (3,)); xy = point @ patch["basis"]
        if (abs(point @ patch["normal"]-patch["offset"]) > 2e-8 or
                np.max(patch["equations"][:, :2] @ xy+patch["equations"][:, 2]+MARGIN_M) > 1e-9):
            raise ValueError("Reference left its authored inset patch")
        T = transforms[reference.link]; world = T[:3, :3] @ point+T[:3, 3]
        if not self.exposed(world, reference.arm, reference.link, reference.mesh, transforms):
            raise ValueError("Reference obscured by another current finger piece")
        normal = T[:3, :3] @ patch["normal"]
        if target_base is not None:
            self.check_target(world, normal, target_base, reference.arm, reference.link, reference.mesh, transforms)
        return {**reference.record(), "point_base_m": world.tolist(),
                "normal_base": normal.tolist(), "current_target_direction_checked": target_base is not None,
                "same_hand_segment_checked": target_base is not None}


def load_pinned_asset(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 2*1024**2:
        raise ValueError("Bounded regular H64 export required")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != EXPORT_SHA:
        raise ValueError("Exact reviewed H64 export required")
    value = json.loads(raw)
    if value["status"] != "completed": raise ValueError("Completed authored export required")
    return FingerSurfaceAsset(value["surfaces"], EXPORT_SHA)
