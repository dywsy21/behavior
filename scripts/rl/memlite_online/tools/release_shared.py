"""Release an already paused run only with pinned machine and human-QA receipts."""
import argparse
import hashlib
import json
from pathlib import Path
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--job', type=Path, required=True)
    parser.add_argument('--review', type=Path, required=True)
    args = parser.parse_args()
    job = args.job.resolve()
    review = json.loads(args.review.read_text())
    manifest = json.loads((job/'manifest.json').read_text())
    raw = (job/'first_two_updates.audit.json').read_bytes()
    audit = json.loads(raw)
    assert review['job'] == str(job) and review['training_source'] == manifest['source_commit']
    assert review['audit_sha256'] == hashlib.sha256(raw).hexdigest()
    assert audit['passed'] is True and audit['update'] == review['audited_update']
    assert review['release_training'] is True and review['media_review_passed'] is True
    assert review['bc_eligible'] is False
    assert audit['checkpoint']['sha256'] == review['checkpoint_sha256']
    latest = json.loads((job/'policies/gpu_0/checkpoints/direct_latest.receipt.json').read_text())
    assert latest['sha256'] == review['checkpoint_sha256'] and latest['updates'] == review['audited_update']
    assert not (job/'ABORT').exists() and not (job/'STOP_TRAINING').exists()
    assert time.time() < manifest['absolute_deadline']-300
    for rank in range(8):
        state = json.loads((job/'policies'/f'gpu_{rank}'/'policy_status.json').read_text())
        assert state['status'] == 'awaiting_acceptance' and state['optimizer_updates'] == review['audited_update']
    with (job/'operator_acceptance.json').open('x') as stream:
        json.dump(dict(review=review, released_at=time.time(), review_path=str(args.review.resolve())), stream, indent=2)
    # Last operation: all rank watchdogs see this after the receipt is closed.
    with (job/'ACCEPTED').open('x') as stream:
        stream.write(review['checkpoint_sha256']+'\n')
    print(json.dumps(dict(released=str(job), update=review['audited_update'], deadline=manifest['absolute_deadline'])))


if __name__ == '__main__':
    main()
