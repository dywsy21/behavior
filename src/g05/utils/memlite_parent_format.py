"""Known task-level text is format supervision, not a fine-grained goal label.

The declaration is derived from public task instructions/slugs, never from an
episode, future action, simulator predicate, or held-out outcome. The original
parent semantic-evidence flag remains false. This module cannot create targets.
"""
from collections.abc import Mapping

from g05.utils.memlite_planner_fields import planner_field_text
from g05.utils.memlite_skill_protocol import validate_semantic_parent_goal


def validate_task_parent_formats(enabled, formats, *, planner_only):
    if type(enabled) is not bool or not isinstance(formats, Mapping):
        raise ValueError('Task-parent format configuration needs a boolean and mapping')
    if enabled and not planner_only:
        raise ValueError('Task-parent format supervision is limited to high_planner_only')
    if enabled and not formats:
        raise ValueError('Format supervision needs a frozen public-task mapping')
    if not enabled and formats:
        raise ValueError('Do not configure a dormant task-parent format mapping')
    checked = {}
    for task, parent in formats.items():
        if not isinstance(task, str) or not task.strip() or task != task.strip():
            raise ValueError('Task-parent mapping keys must be exact task instruction text')
        if not isinstance(parent, str) or not parent.startswith('Task goal: ') or not parent[11:].strip():
            raise ValueError('Only literal task-level fallback text is eligible')
        validate_semantic_parent_goal(parent, field='task_parent_format')
        checked[task] = parent
    return checked


def is_declared_task_parent_format(sample, formats):
    """True only for an existing, exact task fallback with semantic mask=false.

    The target and its tokenizer-facing text must already agree with the
    declared public-task fallback. No sample field or evidence mask is changed.
    Unknown detailed parent goals remain unsupervised.
    """
    if sample.get('current_parent_goal_supervision_mask') is not False:
        return False
    expected = formats.get(sample.get('task_name'))
    if expected is None or sample.get('target_parent_goal') != expected:
        return False
    if sample.get('parent_goal_supervision_mask') is not False:
        raise ValueError('Raw and tokenizer-facing parent evidence flags disagree')
    if sample.get('current_parent_goal_value') != expected:
        raise ValueError('Parent format value differs from its existing target')
    if planner_field_text(sample, 'current_parent_goal') != 'Parent goal: ' + expected:
        raise ValueError('Actual tokenized parent text differs from declared task fallback')
    return True
