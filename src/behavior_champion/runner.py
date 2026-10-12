"""Durable optimization loop, independent of JAX/model construction.

The engine owns math; this loop owns finite budgets, checkpoint boundaries,
signals, online logging, and resume provenance. It never auto-retries an error,
deletes an earlier checkpoint, or changes an approved recipe.
"""
import json
import math
import os
from pathlib import Path
import signal
import time


def write_json(path, value):
    """Atomically replace our small mutable run pointer, not model artifacts."""
    import tempfile
    path = Path(path)
    descriptor, temporary = tempfile.mkstemp(prefix='.' + path.name + '.', dir=path.parent)
    with os.fdopen(descriptor, 'w') as stream:
        json.dump(value, stream, indent=2); stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, path)


class StopRequest:
    def __init__(self):
        self.requested = False
        self.signum = None
        self.previous = {}

    def __enter__(self):
        def handle(signum, _):
            self.requested, self.signum = True, signum
        for signum in (signal.SIGINT, signal.SIGTERM):
            self.previous[signum] = signal.signal(signum, handle)
        return self

    def __exit__(self, *args):
        for signum, handler in self.previous.items():
            signal.signal(signum, handler)


def validate_launch(recipe, *, engineering):
    if recipe.get('schema') != 'champion100-training-v1':
        raise ValueError('Unknown recipe schema')
    if not engineering and recipe.get('formal_launch_approved') is not True:
        raise PermissionError('Formal large training still requires the user-approved recipe in Git')
    if (recipe.get('updates') != 8000 or recipe.get('batch_size') != 256 or
            recipe.get('max_hours') != 24 or recipe.get('checkpoint_every') != 1000 or
            recipe.get('eval_every') != 1000 or recipe.get('eval_per_task') != 16):
        raise ValueError('Changed recipe; explicitly implement and review a new training contract')


def run_updates(engine, tracker, directory, *, updates, max_seconds, checkpoint_every,
                eval_every, stop, prior_seconds=0., started=None, clock=time.monotonic):
    """Engine: completed, step(i), save(), evaluate(), export(receipt).

    Record time at each completed update. Resume must use the maximum journal
    elapsed time, including updates later replayed from an older checkpoint.
    Evaluation at step0/end is fixed, not chosen from observed loss or Q.
    """
    directory = Path(directory)
    if not (0 <= engine.completed <= updates and 0 <= prior_seconds < max_seconds):
        raise ValueError('Invalid completed cursor or exhausted cumulative budget')
    started = clock() if started is None else started
    elapsed = lambda: prior_seconds + clock() - started
    last_saved = None
    last_eval = None

    def journal(row):
        with (directory / 'events.jsonl').open('a') as stream:
            stream.write(json.dumps(dict(completed_updates=engine.completed,
                                         elapsed_seconds=elapsed(), **row)) + '\n')
            stream.flush(); os.fsync(stream.fileno())

    def save(reason):
        nonlocal last_saved
        if last_saved is None or last_saved['completed_updates'] != engine.completed:
            last_saved = engine.save()
            write_json(directory / 'latest-checkpoint.json', last_saved)
            journal(dict(event='checkpoint_committed', reason=reason, receipt=last_saved))
        return last_saved

    def evaluate():
        nonlocal last_eval
        metrics = engine.evaluate()
        tracker.log(metrics, completed=engine.completed)
        journal(dict(event='fixed_original_heldout_evaluated', metrics=metrics))
        last_eval = engine.completed

    try:
        if engine.completed == 0 and not stop.requested and elapsed() < max_seconds:
            evaluate()
        while engine.completed < updates:
            if stop.requested or elapsed() >= max_seconds:
                break
            previous = engine.completed
            metrics = engine.step(previous)
            if engine.completed != previous + 1:
                raise ValueError('Optimizer cursor must advance by exactly one completed update')
            if any(not math.isfinite(float(value)) for value in metrics.values()):
                raise ValueError('Nonfinite training metrics; do not commit a bad state')
            tracker.log({'train/' + key: value for key, value in metrics.items()}, completed=engine.completed)
            journal(dict(event='optimizer_update', metrics=metrics))
            if engine.completed % checkpoint_every == 0:
                save('periodic')
            if (engine.completed % eval_every == 0 and engine.completed != updates
                    and not stop.requested and elapsed() < max_seconds):
                evaluate()
        status = ('completed_target_updates' if engine.completed == updates else
                  'saved_on_signal' if stop.requested else 'saved_at_cumulative_time_limit')
        # A received signal forbids new optimization. Commit the current finite
        # state; don't kill mid-write just to hit a wall-clock number exactly.
        receipt = save(status)
        if not stop.requested and elapsed() < max_seconds and last_eval != engine.completed:
            evaluate()
        export = engine.export(receipt) if not stop.requested and elapsed() < max_seconds else None
        terminal = dict(status=status, completed_updates=engine.completed,
                        elapsed_seconds=elapsed(), checkpoint=receipt, export=export)
        write_json(directory / 'result.json', terminal)
        tracker.finish(status, completed=engine.completed)
        return terminal
    except Exception as error:
        # Never save a possibly corrupt state after a failed optimizer step.
        # The earlier durable checkpoint is retained for explicit diagnosis.
        terminal = dict(status='failed_no_retry', error_type=type(error).__name__,
                        completed_updates=engine.completed, elapsed_seconds=elapsed(),
                        last_checkpoint=last_saved)
        write_json(directory / 'failure.json', terminal)
        tracker.finish('failed_no_retry', completed=engine.completed, error=True)
        raise
