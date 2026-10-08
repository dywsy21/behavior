"""Load only the pinned, immutable baseline support modules; no hot patching."""
import hashlib
import json
import os
from pathlib import Path
import sys


def bootstrap():
    source = Path(__file__).resolve().parents[1]
    manifest = json.loads((source / 'baseline_manifest.json').read_text())
    baseline = Path(os.environ.get('RL_BASE_SNAPSHOT', manifest['source'])).resolve()
    for relative, expected in manifest['sha256'].items():
        path = baseline / relative
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise RuntimeError('Baseline dependency changed: ' + str(path))
    # Our replacements first; unchanged reward/ledger/builders from the
    # hash-pinned snapshot, never the live tools directory.
    sys.path[:0] = [str(source / 'code'), str(baseline / 'code'), str(baseline / 'tools')]
    return source
