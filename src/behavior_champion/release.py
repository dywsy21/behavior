"""Read the accepted source split WITHOUT reviving whole-episode quarantines.

Champion action chunks need no semantic skill boundaries, but that does not
authorize using a source episode previously excluded by a human/data audit.
Only compact metadata is read here; no RGB, actions or privileged state.
"""
from collections import Counter
import hashlib
import json
from pathlib import Path


RELEASE_SHA256 = '90ff0fa9334959dae5ff4368913add6c8a3858e9c124ca7b6c0b05abe85d6f23'


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while block := stream.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


class AcceptedRelease:
    def __init__(self, directory, *, manifest_sha256=RELEASE_SHA256):
        self.root = Path(directory)
        if sha256(self.root / 'manifest.json') != manifest_sha256:
            raise ValueError('Unregistered source release')
        self.manifest = json.loads((self.root / 'manifest.json').read_text())
        self.acceptance = json.loads((self.root / 'acceptance.json').read_text())
        if (self.acceptance['status'] != 'ACCEPTED' or
                self.acceptance['manifest_sha256'] != manifest_sha256):
            raise ValueError('Original data acceptance missing or mismatched')
        for name in ('episodes.jsonl', 'exclusions.json'):
            self.verify_file(name)
        rows = json.loads((self.root / 'exclusions.json').read_text())
        self.exclusions = {int(row['episode_index']): row for row in rows}
        if len(self.exclusions) != len(rows) or len(rows) != self.manifest['excluded_episodes']:
            raise ValueError('Repeated or mismatched source quarantine')
        self.excluded_splits = Counter(row['split'] for row in rows)
        if set(self.excluded_splits) - {'train', 'eval'}:
            raise ValueError('Invalid quarantine split')

    def verify_file(self, name):
        if (Path(name).name != name or
                sha256(self.root / name) != self.manifest['files'][name]):
            raise ValueError('Changed accepted source metadata: ' + name)

    def episodes(self, split=None):
        """Yield (original row serial, eligible record); consume to EOF to audit."""
        if split not in (None, 'train', 'eval'):
            raise ValueError('Explicit original train/eval split required')
        seen, seen_excluded, groups, counts = set(), set(), {}, Counter()
        with (self.root / 'episodes.jsonl').open() as stream:
            for serial, line in enumerate(stream):
                ep = json.loads(line)
                row = ep['row']; eid = int(row['episode_index'])
                task, instance = int(row['task_index']), int(row['task_instance_id'])
                if (eid in seen or ep['split'] not in ('train', 'eval') or
                        not 0 <= task < 100 or not 0 <= instance < 301):
                    raise ValueError('Duplicate, foreign or public-test source episode')
                seen.add(eid)
                if groups.setdefault((task, instance), ep['split']) != ep['split']:
                    raise ValueError('Cross-split source instance')
                excluded = self.exclusions.get(eid)
                if excluded is not None:
                    identity = [task, int(row['raw_episode_id']), instance]
                    if excluded['identity'] != identity or excluded['split'] != ep['split'] or ep['segments']:
                        raise ValueError('Quarantine identity, split or empty-marker mismatch')
                    seen_excluded.add(eid)
                    continue
                if not ep['segments']:
                    raise ValueError('Unregistered empty/quarantined source episode')
                counts[ep['split']] += 1
                if split is None or ep['split'] == split:
                    yield serial, ep
        if (len(seen) != self.manifest['episodes'] or seen_excluded != set(self.exclusions) or
                dict(counts) != self.acceptance['eligible_episodes']):
            raise ValueError('Accepted source coverage or exclusions changed')
