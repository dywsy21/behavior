"""Mechanically relocate only an extracted, owned uv cache, never a live env."""
import argparse
import json
import os
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--old-root', type=Path, required=True)
    args = ap.parse_args()
    allowed = {
        Path('/data/workspace/wsy/behavior2026/tools/champion-uv-transferred-cache-v1'):
            Path('/home/wsy/behavior/artifacts/champion-wheelcache-v1'),
        Path('/data/workspace/wsy/behavior2026/tools/champion-extra-transferred-cache-v1'):
            Path('/home/wsy/behavior/artifacts/champion-extra-cache-v1'),
    }
    if allowed.get(args.root) != args.old_root or not (args.root/'CACHEDIR.TAG').is_file():
        raise ValueError('Only the two registered extracted cache copies can be relocated')
    links, alternates = [], []
    for path in args.root.rglob('*'):
        if path.is_symlink():
            target = Path(os.readlink(path))
            if target.is_absolute() and target.is_relative_to(args.old_root):
                relocated = args.root/target.relative_to(args.old_root)
                if not relocated.exists():
                    raise ValueError('Incomplete extracted cache target: '+str(relocated))
                links.append((path, os.path.relpath(relocated, path.parent)))
        elif path.name == 'alternates' and path.is_file() and 'git-v0' in path.parts:
            previous = path.read_text()
            updated = previous.replace(str(args.old_root)+'/', str(args.root)+'/')
            if previous != updated:
                for value in updated.splitlines():
                    if not Path(value).is_dir():
                        raise ValueError('Missing relocated Git object store')
                alternates.append((path, updated))
    # All targets checked before mutation. Original archives remain recoverable;
    # only cache-link addressing changes, never a downloaded package payload.
    for path, target in links:
        path.unlink()
        path.symlink_to(target)
    for path, text in alternates:
        path.write_text(text)
    print(json.dumps(dict(status='cache_addressing_relocated', symlinks=len(links),
                          git_alternates=len(alternates), root=str(args.root))))


if __name__ == '__main__':
    main()
