"""Persist EVERY acknowledged control, including the official terminal one.

Native termination is a control-flow boundary, never a skill outcome label.
Call this only AFTER the simulator applied the action and observed s[t+1].
Raise/stop only AFTER it returns; otherwise the final observation gets the
clock of s[t] even though the simulator already advanced to s[t+1].
"""
import json


def persist_applied_control(stream, row, *, terminated, truncated):
    if (type(terminated) is not bool or type(truncated) is not bool
            or type(row.get('control_step')) is not int or row['control_step'] < 0
            or row.get('simulator_apply_ack') is not True):
        raise ValueError('Require a real acknowledged control and native Boolean end flags')
    final = dict(row, terminated=terminated, truncated=truncated)
    stream.write(json.dumps(final, allow_nan=False) + '\n')
    stream.flush()
    return row['control_step'] + 1


def require_nonterminal(terminated, truncated):
    if terminated or truncated:
        raise RuntimeError('Official terminal after recorded control; not a skill failure label')
