"""Lossless robot calibration evidence and pre-write pilot byte budgets."""
import gzip
import hashlib
import json
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

    @staticmethod
    def used(root):
        return sum(p.stat().st_size for p in root.rglob("*") if p.is_file())

    def check(self, size, cleanup=False):
        reserve = 0 if cleanup else self.reserve
        if (self.used(self.root)+size > self.run_limit-reserve or
                self.used(self.total_root)+size > self.total_limit-3*reserve):
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
