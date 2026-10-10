"""Build the approved union and freeze calibration inputs; never start a GPU job."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

REPO = Path(__file__).resolve().parents[4]
sys.path[:0] = [str(REPO/'src'), str(Path(__file__).resolve().parents[1]/'code')]
from recovery_calibration_launch import prepare
from recovery_admission import local_file
from recovery_corpus import file_sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--declaration', type=Path, required=True)
    args = parser.parse_args()
    if subprocess.check_output(['git', 'status', '--porcelain'], cwd=REPO, text=True).strip():
        raise ValueError('Use a clean, separately frozen source worktree')
    declaration = local_file(REPO, args.declaration.resolve())
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip()
    result = prepare(json.loads(declaration.read_text()), REPO,
        source_commit=commit, declaration_sha256=file_sha(declaration))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
