"""Verify the intended online team/run lifecycle; zero model/optimizer work."""
import argparse
import json
from pathlib import Path
import subprocess

from behavior_champion.tracking import OnlineRun


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--credential-file', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    source = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    config = dict(purpose='champion2026_engineering_preparation_not_training',
                  optimizer_updates=0, source_commit=source)
    tracker = OnlineRun(args.output, credential_file=args.credential_file,
        entity='hanhanyy-fudan-university-school-of-management', project='behavior2026-g05',
        name='champion2026-online-lifecycle-preflight', config=config)
    try:
        tracker.log({'engineering/handshake': 1, 'engineering/optimizer_updates': 0}, completed=0)
        first = tracker.finish('handshake_saved', completed=0)
        # Reopen the exact same online run, rather than accidentally producing
        # a second run or changing team during a training resume.
        resumed = OnlineRun(args.output, credential_file=args.credential_file,
            entity=first['entity'], project=first['project'], name='champion2026-online-lifecycle-preflight',
            config=config, resume_id=first['run_id'], resume_update=0)
        resumed.log({'engineering/resume_handshake': 1, 'engineering/optimizer_updates': 0}, completed=0)
        final = resumed.finish('engineering_online_and_resume_verified', completed=0)
        import wandb
        remote = wandb.Api(timeout=60).run(f"{first['entity']}/{first['project']}/{first['run_id']}")
        if (remote.config.get('source_commit') != source or
                remote.summary.get('terminal_status') != final['terminal_status'] or
                remote.summary.get('engineering/resume_handshake') != 1):
            raise ValueError('Remote W&B summary does not confirm the resumed terminal run')
        result = dict(status='online_and_exact_run_resume_verified', identity=final,
                      optimizer_updates=0, physical_controls=0, source_commit=source,
                      remote_terminal_summary_confirmed=True)
        (args.output / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps(result), flush=True)
    except Exception:
        # No credential or environment dump on failure; W&B retains its own
        # ordinary run diagnostics under the private experiment directory.
        raise


if __name__ == '__main__': main()
