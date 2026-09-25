"""Inspect H64's saved robot meshes on CPU; no simulator/contact certificate."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
from scipy.spatial import ConvexHull


def inspect_mesh(mesh):
    vertices = np.asarray(mesh["vertices_link_m"], dtype=float)
    faces = np.asarray(mesh["triangles"], dtype=int)
    triangles = vertices[faces]
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    twice_area = np.linalg.norm(normals, axis=1)
    normals /= twice_area[:, None]
    signed = np.einsum("fvi,fi->fv", vertices[None] - triangles[:, :1], normals)
    # A convex boundary face cannot have vertices on BOTH sides of its plane.
    support_error = np.minimum(signed.max(axis=1), -signed.min(axis=1))
    edges = np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1)
    _, multiplicity = np.unique(edges, axis=0, return_counts=True)
    hull = ConvexHull(vertices)
    volume = abs(float(np.einsum("fi,fi->f", triangles[:, 0],
                                np.cross(triangles[:, 1], triangles[:, 2])).sum() / 6))
    T = np.asarray(mesh["T_link_mesh"])
    return {"vertices": len(vertices), "triangles": len(faces),
            "bounds_link_mm": (np.array([vertices.min(axis=0), vertices.max(axis=0)]) * 1000).tolist(),
            "mesh_scale_singular_values": np.linalg.svd(T[:3, :3], compute_uv=False).tolist(),
            "transform_determinant": float(np.linalg.det(T[:3, :3])),
            "minimum_triangle_area_m2": float(twice_area.min() / 2),
            "edges_not_shared_twice": int(np.count_nonzero(multiplicity != 2)),
            "max_convex_support_error_m": float(support_error.max()),
            "signed_mesh_volume_absolute_m3": volume, "convex_hull_volume_m3": float(hull.volume),
            "relative_volume_difference": float(abs(volume-hull.volume) / hull.volume)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    os.sched_setaffinity(0, set(sorted(os.sched_getaffinity(0))[:2]))
    started = time.monotonic()
    if args.output.exists() or args.output.is_symlink():
        raise ValueError("New inspection output only")
    if args.input.stat().st_size > 2 * 1024**2:
        raise ValueError("Two MiB input bound")
    raw = args.input.read_bytes()
    if hashlib.sha256(raw).hexdigest() != args.sha256:
        raise ValueError("Actual exported asset bytes do not match")
    data = json.loads(raw)
    if (data["status"] != "completed" or data["source_commit"] != "0f8321d5679b9d2bab6b360e20d2dec2914c9e2f"
            or data["surfaces"]["physical_contact_evidence"] is not False
            or data["surfaces"]["runtime_cooked_collision_verified"] is not False):
        raise ValueError("Only the real authored H64 export is inspected")
    links = data["surfaces"]["links"]
    report = {"input_sha256": args.sha256, "scope": "SAVED_AUTHORED_MESH_NUMERICS_ONLY",
              "physical_contact_evidence": False, "runtime_cooked_collision_verified": False,
              "model_calls": 0, "simulator_resets": 0, "training_steps": 0, "links": {}}
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    fig = plt.figure(figsize=(15, 14))
    colors = plt.get_cmap("tab10")
    for row, (name, item) in enumerate(links.items()):
        report["links"][name] = {key: inspect_mesh(mesh) for key, mesh in item["meshes"].items()}
        combined = np.concatenate([np.asarray(m["vertices_link_m"]) for m in item["meshes"].values()]) * 1000
        low, high = combined.min(axis=0), combined.max(axis=0)
        for col, (a, b) in enumerate(((0, 1), (0, 2), (1, 2))):
            ax = fig.add_subplot(4, 4, row * 4 + col + 1)
            for number, mesh in enumerate(item["meshes"].values()):
                verts = np.asarray(mesh["vertices_link_m"]) * 1000
                ax.scatter(verts[:, a], verts[:, b], s=3, color=colors(number), alpha=.7)
            ax.set(xlabel="xyz"[a]+" (mm)", ylabel="xyz"[b]+" (mm)")
            ax.set_aspect("equal"); ax.grid(alpha=.25)
            if col == 0: ax.set_title(name.replace("_gripper_finger_", " / "), fontsize=10)
        ax = fig.add_subplot(4, 4, row * 4 + 4, projection="3d")
        for number, mesh in enumerate(item["meshes"].values()):
            verts = np.asarray(mesh["vertices_link_m"]) * 1000
            ax.add_collection3d(Poly3DCollection(verts[np.asarray(mesh["triangles"])],
                                               facecolors=colors(number), edgecolors="none", alpha=.65))
        ax.set(xlim=(low[0]-1, high[0]+1), ylim=(low[1]-1, high[1]+1), zlim=(low[2]-1, high[2]+1),
               xlabel="x (mm)", ylabel="y (mm)", zlabel="z (mm)")
        ax.set_box_aspect(high-low+2); ax.view_init(elev=22, azim=-50)
    fig.suptitle("R1Pro authored finger collision pieces — link-local millimetres\n"
                 "Colours identify 8 pieces per finger. NOT live cooked geometry or contact evidence.", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, .96))
    args.output.mkdir(parents=True, exist_ok=False)
    fig.savefig(args.output / "four_fingers.png", dpi=150)
    plt.close(fig)
    report["seconds"] = time.monotonic()-started
    report["manual_review"] = "PENDING_MAIN_AGENT_VISUAL_INSPECTION"
    with (args.output / "numerical.json").open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    print(json.dumps({"links": len(links), "meshes": sum(len(r) for r in report["links"].values()),
                      "seconds": report["seconds"], "output": str(args.output)}))


if __name__ == "__main__": main()
