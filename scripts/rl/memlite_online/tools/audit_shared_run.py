"""Read-only real shared-round acceptance; writes a receipt, never releases training."""
# ruff: noqa: E402
import argparse
import json
from pathlib import Path
import time

from bootstrap import bootstrap
bootstrap()
from checkpoint_io import atomic_json, validate_checkpoint
from synchronous import state_digest
from validate_recovery import validate_archive
import torch


def audit(job):
    job = Path(job)
    manifest = json.loads((job/'manifest.json').read_text())
    expected = manifest['training']['acceptance_pause_updates']
    rows, sessions, candidates = [], [], []
    for rank in range(8):
        root = job/'policies'/f'gpu_{rank}'
        status = json.loads((root/'policy_status.json').read_text())
        row = json.loads((root/'latest_update.json').read_text())
        assert status['status'] == 'awaiting_acceptance', (rank, status['status'])
        assert row['update'] == expected and row['collected_policy_version'] == expected-1
        assert row['rank'] == rank and row['world_size'] == 8
        assert row['before']['mean_approx_kl'] <= .005
        assert row['actor_updated'] and row['actor_max_delta'] > 0
        assert 0 < row['accepted_actor_lr'] <= manifest['training']['actor_lr']
        assert row['post_update']['mean_approx_kl'] <= manifest['training']['target_kl']
        assert row['post_update']['clip_fraction'] <= manifest['training']['max_clip_fraction']
        if rank:
            assert row['replica_identity'] == rows[0]['replica_identity']
            assert row['checkpoint']['sha256'] == rows[0]['checkpoint']['sha256']
            assert not list((root/'checkpoints').glob('*.pt')), 'Only rank zero may write weights'
        rows.append(row)
        local_sessions = [json.loads(line) for line in (root/'session_begin.jsonl').read_text().splitlines()]
        assert local_sessions
        for session in local_sessions:
            assert session['rank'] == rank and session['all_memories_empty'] and session['independent_ledgers']
        sessions.extend(local_sessions)
        # Bound media verification to the first candidate per sampling rank.
        archives = sorted((job/'recovery'/f'gpu_{rank}').glob('*.zip'))
        if archives:
            candidate = validate_archive(archives[0])
            assert candidate['episode']['gpu'] == rank
            assert candidate['episode']['task'] in {t['task'] for t in manifest['groups'][rank]['tasks']}
            candidates.append(candidate)
    ids = [identity for row in sessions for identity in row['episode_ids']]
    assert len(ids) == len(set(ids)), 'Cross-rank/episode identity collision'
    path = Path(rows[0]['checkpoint']['latest'])
    checkpoint = validate_checkpoint(path, expected)
    assert checkpoint['sha256'] == rows[0]['checkpoint']['sha256']
    payload = torch.load(path, map_location='cpu', mmap=True, weights_only=False)
    shared = payload['distributed_state']
    assert shared['world_size'] == 8 and shared['committed_update'] == expected
    assert shared['task_weight_map'] == manifest['task_weights']
    assert [r['rank'] for r in shared['rank_states']] == list(range(8))
    for saved, reported in [('action_expert', 'actor'), ('critic', 'critic'),
                            ('actor_optimizer', 'actor_optimizer'), ('critic_optimizer', 'critic_optimizer')]:
        assert state_digest(payload[saved]) == rows[0]['replica_identity'][reported], saved
    return dict(passed=True, verified=time.time(), source=manifest['source_commit'],
        update=expected, world_size=8, independent_episodes=len(ids),
        checkpoint=checkpoint, replica_identity=rows[0]['replica_identity'],
        rank_summaries=[{k:r[k] for k in ['rank','experiences','tasks','actor_grad_norm',
            'critic_grad_norm','actor_max_delta','accepted_actor_lr','before','post_update',
            'communication_seconds','total_seconds']} for r in rows],
        candidates=candidates, human_media_review='pending',
        not_independent_evaluation=True, auto_release=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--job', type=Path, required=True)
    args = parser.parse_args()
    receipt = args.job/'first_two_updates.audit.json'
    if receipt.exists():
        raise ValueError('Existing acceptance receipt must not be overwritten')
    result = audit(args.job)
    atomic_json(receipt, result)
    print(json.dumps({k:result[k] for k in ['passed','update','world_size','independent_episodes']}))


if __name__ == '__main__':
    main()
