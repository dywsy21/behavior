"""Bind explicit AR/KI Git extensions to the immutable A3 runtime.

No shared environment changes or hot edits. Import only the declared modules
from this Git worktree. The input processor has one explicit native-vocabulary
opt-out, defaulting to the identical A3 MEM-Lite layout; other code stays A3.
"""
from __future__ import annotations

import importlib
import importlib.util
from pathlib import Path
import sys

from probe_ar_execution_codec import SOURCE, SOURCE_SHA, INPUT, INPUT_SHA, sha

REPO = Path(__file__).resolve().parents[2]
EXTENSIONS = {
    "g05.models.g05.io.input_preprocessor": "src/g05/models/g05/io/input_preprocessor.py",
    "g05.utils.training.fm_training_methods": "src/g05/utils/training/fm_training_methods.py",
    "g05.utils.training.ar_training_methods": "src/g05/utils/training/ar_training_methods.py",
    "g05.models.g05.helpers.action_marker_rows": "src/g05/models/g05/helpers/action_marker_rows.py",
    "g05.models.g05.g05_policy_memlite_action": "src/g05/models/g05/g05_policy_memlite_action.py",
    "g05.models.g05.g05_policy_memlite_action_rows": "src/g05/models/g05/g05_policy_memlite_action_rows.py",
}


def bootstrap():
    sys.path[:0] = [str(SOURCE / "src"), str(SOURCE / "scripts"), str(SOURCE)]
    import g05
    from g05.utils.training.coordination_runtime import source_tree_sha256

    if Path(g05.__file__).resolve().parent != SOURCE / "src/g05":
        raise RuntimeError("Existing g05 import is outside the immutable A3 runtime")
    if source_tree_sha256(SOURCE) != SOURCE_SHA:
        raise RuntimeError("A3 source identity changed")
    for name, relative in EXTENSIONS.items():
        path = REPO / relative
        importlib.import_module(name.rsplit(".", 1)[0])
        if name in sys.modules:
            previous = Path(sys.modules[name].__file__).resolve()
            if previous == path:
                continue
            # Importing the g05 package loads the source policy/processor.
            # Replace precisely this declared default-compatible processor
            # and its policy constructor binding, not arbitrary old modules.
            if not (name == "g05.models.g05.io.input_preprocessor" and previous == SOURCE / relative):
                raise RuntimeError("An undeclared action extension is already imported: " + name)
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        setattr(sys.modules[name.rsplit(".", 1)[0]], name.rsplit(".", 1)[1], module)
        if name == "g05.models.g05.io.input_preprocessor":
            core = importlib.import_module("g05.models.g05.g05_policy")
            if Path(core.__file__).resolve() != SOURCE / "src/g05/models/g05/g05_policy.py":
                raise RuntimeError("Undeclared base policy for input-processor binding")
            core.InputPreprocessor = module.InputPreprocessor
    return dict(source=str(SOURCE), source_sha256=SOURCE_SHA,
                extensions={name: dict(path=str(REPO / relative), sha256=sha(REPO / relative))
                            for name, relative in EXTENSIONS.items()})


if __name__ == "__main__":
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    bootstrap()
    import pytest
    raise SystemExit(pytest.main(sys.argv[1:]))
