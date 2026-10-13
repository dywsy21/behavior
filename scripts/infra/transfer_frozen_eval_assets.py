"""Hash-bound, parallel SSH range transport for already frozen model files.

No private keys, optimizer states, implicit retries, source mutation or model
conversion. Parts live outside the final pack. Original file SHA is required
again before publication; partial prior rsync attempts remain recoverable.
"""
import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import uuid

BUFFER = 4 << 20


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(BUFFER), b''):
            h.update(block)
    return h.hexdigest()


def save_new(path, value):
    with Path(path).open('x') as f:
        json.dump(value, f, indent=2)
        f.write('\n')


def check_plan(plan):
    if plan['schema'] != 'frozen_eval_range_transport_v1':
        raise ValueError('Unknown transfer plan')
    names = set()
    for row in plan['files']:
        if row['name'] not in ('low.pt', 'observer_backbone.pt') or row['name'] in names:
            raise ValueError('Unexpected or duplicate model')
        names.add(row['name'])
        offset = 0
        for i, chunk in enumerate(row['chunks']):
            if (chunk['offset'] != offset or chunk['bytes'] <= 0
                    or chunk['name'] != f"{Path(row['name']).stem}-{i:03d}.part"
                    or re.fullmatch('[0-9a-f]{64}', chunk['sha256']) is None):
                raise ValueError('Invalid, overlapping or gapped range')
            offset += chunk['bytes']
        if offset != row['bytes'] or not offset:
            raise ValueError('Incomplete file coverage')
    if names != {'low.pt', 'observer_backbone.pt'}:
        raise ValueError('Both fixed model files required')
    return plan


def prepare(pack, manifest_sha, output, chunk_bytes=1 << 30):
    pack = Path(pack).resolve()
    if digest(pack / 'manifest.json') != manifest_sha or not 1 <= chunk_bytes <= 2 << 30:
        raise ValueError('Unbound source or invalid chunk size')
    manifest = json.loads((pack / 'manifest.json').read_text())
    rows = []
    for name in ('low.pt', 'observer_backbone.pt'):
        signed = [r for r in manifest['files'] if r['path'] == name]
        if len(signed) != 1:
            raise ValueError('Missing/ambiguous source file')
        signed = signed[0]
        path = pack / name
        if path.is_symlink() or path.stat().st_size != signed['bytes']:
            raise ValueError('Changed source size/type')
        full = hashlib.sha256()
        chunks = []
        offset = 0
        with path.open('rb') as f:
            while offset < signed['bytes']:
                length = min(chunk_bytes, signed['bytes'] - offset)
                h = hashlib.sha256()
                left = length
                while left:
                    block = f.read(min(BUFFER, left))
                    if not block:
                        raise ValueError('Truncated source')
                    h.update(block)
                    full.update(block)
                    left -= len(block)
                chunks.append(dict(name=f'{path.stem}-{len(chunks):03d}.part',
                    offset=offset, bytes=length, sha256=h.hexdigest()))
                offset += length
            if f.read(1):
                raise ValueError('Source grew while preparing')
        if full.hexdigest() != signed['sha256']:
            raise ValueError('Source SHA changed')
        rows.append(dict(name=name, source=str(path), bytes=offset,
            sha256=full.hexdigest(), chunks=chunks))
    plan = check_plan(dict(schema='frozen_eval_range_transport_v1',
        source_manifest_sha256=manifest_sha, files=rows))
    save_new(output, plan)
    return dict(plan_sha256=digest(output), parts=sum(len(r['chunks']) for r in rows))


def receive(directory, name, length, sha, stream):
    directory = Path(directory).resolve()
    if (re.fullmatch(r'(low|observer_backbone)-\d{3}\.part', name) is None
            or not 0 < length <= 2 << 30 or re.fullmatch('[0-9a-f]{64}', sha) is None):
        raise ValueError('Invalid explicit part')
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    if path.exists():
        raise FileExistsError('A part already exists; verify it before explicit resume')
    tmp = directory / ('.' + name + '.' + uuid.uuid4().hex)
    h = hashlib.sha256()
    left = length
    with tmp.open('xb') as f:
        while left:
            block = stream.read(min(BUFFER, left))
            if not block:
                raise ValueError('Truncated incoming range; partial retained')
            f.write(block)
            h.update(block)
            left -= len(block)
        if stream.read(1) or h.hexdigest() != sha:
            raise ValueError('Oversized/corrupt incoming range; partial retained')
        f.flush()
        os.fsync(f.fileno())
    # Exclusive publication: never overwrite an earlier successful part.
    os.link(tmp, path)
    tmp.unlink()
    return dict(part=name, bytes=length, sha256=sha)


def push(plan, *, host, port, remote_script, remote_parts, workers):
    if not 1 <= workers <= 16:
        raise ValueError('Explicit bounded transport concurrency required')
    def one(row, chunk):
        argv = ['python3', remote_script, 'receive', '--directory', remote_parts,
            '--name', chunk['name'], '--length', str(chunk['bytes']), '--sha256', chunk['sha256']]
        process = subprocess.Popen(['ssh', '-p', str(port), '-o', 'BatchMode=yes',
            '-o', 'StrictHostKeyChecking=yes', '-o', 'ConnectTimeout=30', host, shlex.join(argv)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        h = hashlib.sha256()
        try:
            with Path(row['source']).open('rb') as source:
                source.seek(chunk['offset'])
                left = chunk['bytes']
                while left:
                    block = source.read(min(BUFFER, left))
                    if not block:
                        raise ValueError('Source truncated during transfer')
                    process.stdin.write(block)
                    h.update(block)
                    left -= len(block)
            process.stdin.close()
            process.stdin = None
            stdout, stderr = process.communicate(timeout=600)
            if process.returncode or h.hexdigest() != chunk['sha256']:
                raise RuntimeError('Part transfer failed: ' + chunk['name'] + ': ' + stderr.decode()[-2000:])
            receipt = json.loads(stdout)
            if receipt != dict(part=chunk['name'], bytes=chunk['bytes'], sha256=chunk['sha256']):
                raise ValueError('Receiver receipt differs')
            print(json.dumps(receipt), flush=True)
            return receipt
        except BaseException:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            raise
    check_plan(plan)
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(one, row, chunk) for row in plan['files'] for chunk in row['chunks']]
        return dict(status='all_parts_transferred', parts=[f.result() for f in futures])


def assemble(plan, directory, pack):
    check_plan(plan)
    directory, pack = Path(directory), Path(pack)
    if digest(pack / 'manifest.json') != plan['source_manifest_sha256']:
        raise ValueError('Destination pack identity differs')
    receipts = []
    for row in plan['files']:
        target = pack / row['name']
        if target.exists():
            raise FileExistsError('Never overwrite a prior final file')
        temp = directory / (row['name'] + '.assembling-' + uuid.uuid4().hex)
        full = hashlib.sha256()
        with temp.open('xb') as out:
            for chunk in row['chunks']:
                part = directory / chunk['name']
                if part.is_symlink() or part.stat().st_size != chunk['bytes']:
                    raise ValueError('Wrong range size/type')
                h = hashlib.sha256()
                with part.open('rb') as f:
                    for block in iter(lambda: f.read(BUFFER), b''):
                        h.update(block)
                        full.update(block)
                        out.write(block)
                if h.hexdigest() != chunk['sha256']:
                    raise ValueError('Corrupt part during assembly')
            out.flush()
            os.fsync(out.fileno())
        if temp.stat().st_size != row['bytes'] or full.hexdigest() != row['sha256']:
            raise ValueError('Whole-file identity failed; assembly retained')
        os.link(temp, target)
        temp.unlink()
        receipts.append(dict(name=row['name'], bytes=row['bytes'], sha256=full.hexdigest()))
    return dict(status='exact_original_files_published', files=receipts)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='mode', required=True)
    a = sub.add_parser('prepare')
    a.add_argument('--pack', required=True); a.add_argument('--manifest-sha256', required=True)
    a.add_argument('--output', required=True)
    a = sub.add_parser('receive')
    a.add_argument('--directory', required=True); a.add_argument('--name', required=True)
    a.add_argument('--length', type=int, required=True); a.add_argument('--sha256', required=True)
    for mode in ('push', 'assemble'):
        a = sub.add_parser(mode)
        a.add_argument('--plan', required=True); a.add_argument('--plan-sha256', required=True)
        if mode == 'push':
            a.add_argument('--host', required=True); a.add_argument('--port', type=int, required=True)
            a.add_argument('--remote-script', required=True); a.add_argument('--remote-parts', required=True)
            a.add_argument('--workers', type=int, default=16)
        else:
            a.add_argument('--directory', required=True); a.add_argument('--pack', required=True)
    a = p.parse_args()
    if a.mode == 'prepare':
        result = prepare(a.pack, a.manifest_sha256, a.output)
    elif a.mode == 'receive':
        result = receive(a.directory, a.name, a.length, a.sha256, sys.stdin.buffer)
    else:
        if digest(a.plan) != a.plan_sha256:
            raise ValueError('Changed transport manifest')
        plan = json.loads(Path(a.plan).read_text())
        if a.mode == 'push':
            result = push(plan, host=a.host, port=a.port, remote_script=a.remote_script,
                remote_parts=a.remote_parts, workers=a.workers)
        else:
            result = assemble(plan, a.directory, a.pack)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
