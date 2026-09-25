"""H71: one full-scene gate for the non-DLSS AA prerequisite fix."""
import argparse
from pathlib import Path

import launch_h69 as gate


def configure():
    gate.ROOT = Path('/mnt/nvme_tmp/robodojo_agentic_20260925/h71_aa_prerequisite_v1')
    gate.RUNTIME = Path('/mnt/nvme_tmp/robodojo_sim_runtime_20260925/h71_aa_prerequisite_v1')
    gate.WALL_SECONDS = 1200
    base = gate.configure()
    base.ENTRYPOINT = Path(__file__).resolve()
    return base


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument('--launch', action='store_true')
    modes.add_argument('--supervise', action='store_true')
    args = parser.parse_args()
    base = configure()
    (gate.launch if args.launch else gate.supervise)(base)
