"""Lossless robot calibration evidence and pre-write pilot byte budgets."""
import gzip
import hashlib
import json
from contextlib import contextmanager
from pathlib import Path

from semantic_robot.v2.kinematics import RobotModel

MIB = 1024**2


def json_bytes(value):
    # Exactly the old common.write_json bytes: preserve the original raw SHA.
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+"\n").encode("utf-8")


class ArtifactBudget:
    def __init__(self, root, total_root=None, run_limit=30*MIB, total_limit=100*MIB, reserve=MIB):
        self.root = Path(root).resolve()
        self.total_root = Path(total_root or root).resolve()
        if not self.root.is_relative_to(self.total_root):
            raise ValueError("Pilot outputs must stay within the counted experiment root")
        self.run_limit, self.total_limit, self.reserve = run_limit, total_limit, reserve
        self._held = 0

    @staticmethod
    def used(root):
        return sum(p.stat().st_size for p in root.rglob("*") if p.is_file())

    def check(self, size, cleanup=False):
        reserve = 0 if cleanup else self.reserve
        held = 0 if cleanup else self._held
        if (self.used(self.root)+held+size > self.run_limit-reserve or
                self.used(self.total_root)+held+size > self.total_limit-3*reserve):
            raise RuntimeError("Pilot artifact budget exceeded BEFORE writing; stop without another action")

    def write_bytes(self, path, content, cleanup=False):
        path = Path(path)
        if not path.resolve().is_relative_to(self.root):
            raise ValueError("Artifact outside this run")
        self.check(len(content), cleanup=cleanup)
        with path.open("xb") as stream:
            stream.write(content)

    def write_json(self, path, value, cleanup=False):
        self.write_bytes(path, json_bytes(value), cleanup=cleanup)

    def append_text(self, stream, line):
        self.check(len(line.encode("utf-8")))
        stream.write(line)

    @contextmanager
    def transaction(self, size):
        """One collector owns this root; reserve COMPLETE post-action evidence.

        Normal writes cannot spend another transaction's allowance. Cleanup has
        a separate existing reserve. No reservation changes a disk/run limit.
        This is not an OS free-space guarantee against other disk users.
        """
        if not isinstance(size, int) or size <= 0:
            raise ValueError("Positive integer transaction bound required")
        self.check(size)
        self._held += size
        reservation = ArtifactReservation(self, size)
        try:
            yield reservation
        finally:
            self._held -= reservation.remaining
            reservation.remaining = 0
            reservation.closed = True


class ArtifactReservation:
    def __init__(self, owner, remaining):
        self.owner, self.remaining, self.closed = owner, remaining, False
        self.root = owner.root

    def check(self, size, cleanup=False):
        if cleanup or self.closed or size < 0 or size > self.remaining:
            raise RuntimeError("Evidence exceeded its pre-action reservation")
        self.owner.check(0)

    def _consume(self, size):
        self.remaining -= size
        self.owner._held -= size

    def write_bytes(self, path, content, cleanup=False):
        path = Path(path)
        if not path.resolve().is_relative_to(self.root):
            raise ValueError("Artifact outside this run")
        self.check(len(content), cleanup)
        with path.open("xb") as stream:
            stream.write(content)
        self._consume(len(content))

    def write_json(self, path, value, cleanup=False):
        self.write_bytes(path, json_bytes(value), cleanup)

    def append_text(self, stream, line):
        size = len(line.encode("utf-8"))
        self.check(size)
        stream.write(line)
        self._consume(size)


def capture_layout(images, depths):
    """Hardware-sized contract, independent of this frame's compressibility."""
    import numpy as np
    result = {}
    for view in ("head", "left_wrist", "right_wrist"):
        image, depth = np.asarray(images[view+"_rgb"]), np.asarray(depths[view])
        if (image.dtype != np.uint8 or image.ndim != 3 or image.shape[0] != 3 or
                depth.ndim != 2 or depth.dtype.kind != "f" or depth.dtype.itemsize > 8 or
                min(*image.shape, *depth.shape) <= 0):
            raise ValueError("Expected raw CHW uint8 RGB and floating HW depth")
        result[view] = {"rgb_shape": list(image.shape), "depth_shape": list(depth.shape),
                        "depth_dtype": str(depth.dtype), "rgb_bytes": image.nbytes,
                        "depth_bytes": depth.nbytes}
    return result


def capture_upper_bound(layout):
    """Conservative DEFLATE/PNG/ZIP bound, including all metadata (1 MiB).

    9/8 + 1/64 of raw bytes plus 64 KiB per payload exceeds DEFLATE's
    worst-case overhead, PNG row filters/chunk headers and NumPy/ZIP headers.
    Actual encoded bundles are also checked against this bound before writing.
    """
    total = MIB
    for view in ("head", "left_wrist", "right_wrist"):
        item = layout[view]
        for raw in (item["rgb_bytes"]+item["rgb_shape"][1], item["depth_bytes"]+4096):
            if not isinstance(raw, int) or raw <= 0:
                raise ValueError("Invalid raw capture size")
            total += raw+(raw+7)//8+(raw+63)//64+65536
    return total


def action_evidence_bound(layout, max_controls=52, teacher_bytes_per_control=16384):
    """Immediate + settled snapshots, trace, teacher telemetry and record.

    The owner's separate cleanup reserve is never spent on this transaction.
    A rejected action consumes no physics. Unknown image layouts fail closed.
    """
    if not 1 <= max_controls <= 256:
        raise ValueError("Finite registered control bound required")
    return (2*capture_upper_bound(layout) + max_controls*(4096+teacher_bytes_per_control)
            + 256*1024)


def write_calibration(writer, model):
    """Online FK keeps the complete in-memory model; only storage is compressed."""
    raw = json_bytes(model.spec)
    compressed = gzip.compress(raw, compresslevel=6, mtime=0)
    if gzip.decompress(compressed) != raw:
        raise ValueError("Calibration compression roundtrip failed")
    receipt = {"compression": "gzip", "path": "robot_calibration.json.gz",
               "uncompressed_json_bytes": len(raw), "compressed_bytes": len(compressed),
               "uncompressed_json_sha256": hashlib.sha256(raw).hexdigest(),
               "compressed_sha256": hashlib.sha256(compressed).hexdigest(),
               "canonical_model_sha256": model.sha, "complete_model_no_fields_removed": True}
    writer.check(len(compressed)+len(json_bytes(receipt)))
    writer.write_bytes(writer.root/receipt["path"], compressed)
    writer.write_json(writer.root/"robot_calibration_receipt.json", receipt)
    return receipt


def load_calibration(folder):
    """Offline replay only; validate compressed, raw and canonical identities."""
    folder = Path(folder)
    receipt = json.loads((folder/"robot_calibration_receipt.json").read_text())
    if receipt.get("compression") != "gzip" or receipt.get("path") != "robot_calibration.json.gz":
        raise ValueError("Unsupported calibration artifact")
    compressed = (folder/receipt["path"]).read_bytes()
    if (len(compressed) != receipt["compressed_bytes"] or
            hashlib.sha256(compressed).hexdigest() != receipt["compressed_sha256"]):
        raise ValueError("Compressed calibration identity mismatch")
    raw = gzip.decompress(compressed)
    if len(raw) != receipt["uncompressed_json_bytes"] or hashlib.sha256(raw).hexdigest() != receipt["uncompressed_json_sha256"]:
        raise ValueError("Uncompressed calibration identity mismatch")
    model = RobotModel(json.loads(raw))
    if model.sha != receipt["canonical_model_sha256"]:
        raise ValueError("Canonical calibration identity mismatch")
    return model
