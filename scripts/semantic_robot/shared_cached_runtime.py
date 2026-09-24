"""Copy only frozen, stopped-run binary caches into an empty private runtime.

No shared installation, process lock, temporary USD, generated Python, source
cache or original run is modified. Copying does not establish a cache hit.
"""
import hashlib
import json
from pathlib import Path
import shutil
import stat


CACHE_TREES = ('cache/texturecache', 'cuda', 'portable/cache/shadercache',
               'portable/cache/nv_shadercache')
MAX_BYTES = 8 * 1024**3
MAX_FILES = 5000


def file_sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(4*1024**2), b''):
            digest.update(chunk)
    return digest.hexdigest()


def digest_manifest(rows):
    return hashlib.sha256(json.dumps(rows,sort_keys=True,separators=(',', ':')).encode()).hexdigest()


def manifest(root):
    root = Path(root)
    if not root.is_absolute() or root.resolve() != root or not root.is_dir():
        raise ValueError('Exact absolute non-symlink runtime required')
    rows, total = [], 0
    for rel in CACHE_TREES:
        tree = root/rel
        if not tree.is_dir() or tree.resolve() != tree:
            raise ValueError('Missing or redirected cache tree: '+rel)
        for path in sorted(tree.rglob('*')):
            info = path.lstat()
            if path.is_symlink() or path.resolve() != path:
                raise ValueError('Cache symlinks forbidden')
            if stat.S_ISDIR(info.st_mode): continue
            if not stat.S_ISREG(info.st_mode): raise ValueError('Only ordinary cache files allowed')
            total += info.st_size
            if total > MAX_BYTES or len(rows) >= MAX_FILES:
                raise ValueError('Bounded cache seed exceeded')
            fingerprint = (info.st_dev,info.st_ino,info.st_size,info.st_mtime_ns)
            digest = file_sha(path)
            after = path.stat()
            if fingerprint != (after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns):
                raise ValueError('Cache changed while hashing')
            rows.append({'path':str(path.relative_to(root)),'bytes':info.st_size,'sha256':digest})
    return sorted(rows,key=lambda row:row['path'])


def copy_frozen(source, destination, *, expected_sha, expected_count, expected_bytes):
    source, destination = Path(source), Path(destination)
    if (destination.resolve() != destination or not destination.is_dir() or
            source == destination or source in destination.parents or destination in source.parents):
        raise ValueError('Separate existing private runtime required')
    for rel in CACHE_TREES:
        tree = destination/rel
        if tree.exists() and (tree.resolve() != tree or not tree.is_dir() or any(tree.iterdir())):
            raise ValueError('Never overwrite a populated cache tree')
        # Reject redirected ancestors even when the final subtree is absent.
        if tree.resolve() != tree: raise ValueError('Redirected cache destination')
    rows = manifest(source)
    if (digest_manifest(rows) != expected_sha or len(rows) != expected_count or
            sum(row['bytes'] for row in rows) != expected_bytes):
        raise ValueError('Frozen source cache manifest changed')
    if shutil.disk_usage(destination).free < 80*1024**3 + expected_bytes:
        raise ValueError('Preserve runtime disk headroom')
    for rel in CACHE_TREES:(destination/rel).mkdir(parents=True,exist_ok=True)
    for row in rows:
        original, target = source/row['path'], destination/row['path']
        target.parent.mkdir(parents=True,exist_ok=True)
        # Exclusive regular-file creation; no hardlinks or writable source reuse.
        with original.open('rb') as src, target.open('xb') as dst:
            shutil.copyfileobj(src,dst,length=4*1024**2)
        shutil.copystat(original,target,follow_symlinks=False)
        if original.stat().st_ino == target.stat().st_ino and original.stat().st_dev == target.stat().st_dev:
            raise ValueError('Cache copies must not share source inodes')
    copied = manifest(destination)
    if copied != rows or manifest(source) != rows:
        raise ValueError('Source or copied cache content changed')
    return rows
