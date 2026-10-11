"""Verify downloaded assets against pinned official HF blob/LFS hashes.

The metadata JSON must be fetched from the official model API at the supplied
revision, not inferred from the mirror's progress or filenames.
"""
import argparse
import fnmatch
import hashlib
import json
from pathlib import Path


def verify(metadata, root, *, repo, revision, patterns):
    if metadata['id'] != repo or metadata['sha'] != revision:
        raise ValueError('Wrong official model identity/revision')
    selected = [r for r in metadata['siblings'] if any(fnmatch.fnmatchcase(r['rfilename'], p) for p in patterns)]
    if not selected:
        raise ValueError('Empty asset selection')
    rows = []
    for row in selected:
        name = row['rfilename']
        path = root / name
        if Path(name).is_absolute() or '..' in Path(name).parts or not path.resolve().is_relative_to(root.resolve()):
            raise ValueError('Asset outside explicit download directory')
        if not path.is_file() or path.stat().st_size != row['size']:
            raise ValueError('Missing/incomplete asset: ' + name)
        lfs = row.get('lfs')
        algorithm = 'sha256' if lfs else 'git-blob-sha1'
        digest = hashlib.sha256() if lfs else hashlib.sha1()
        if not lfs:
            digest.update(f"blob {row['size']}\0".encode())
        sha256 = hashlib.sha256()
        with path.open('rb') as stream:
            while block := stream.read(8*1024*1024):
                digest.update(block)
                sha256.update(block)
        wanted = lfs['sha256'] if lfs else row['blobId']
        if digest.hexdigest() != wanted:
            raise ValueError('Official hash mismatch: ' + name)
        rows.append(dict(path=name, size=row['size'], official_algorithm=algorithm,
                         official_hash=wanted, sha256=sha256.hexdigest()))
        print(json.dumps(dict(verified=name, bytes=row['size'])), flush=True)
    return dict(status='all_selected_assets_verified', repo=repo, revision=revision,
                selected_patterns=patterns, files=rows, bytes=sum(r['size'] for r in rows),
                verification='official pinned blob/LFS hashes; no downloaded Python executed')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for key in ('metadata', 'root', 'output'):
        ap.add_argument('--'+key, type=Path, required=True)
    for key in ('repo', 'revision'):
        ap.add_argument('--'+key, required=True)
    ap.add_argument('--pattern', action='append', required=True)
    args = ap.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = verify(json.loads(args.metadata.read_text()), args.root, repo=args.repo,
                    revision=args.revision, patterns=args.pattern)
    result['metadata_sha256'] = hashlib.sha256(args.metadata.read_bytes()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    main()
