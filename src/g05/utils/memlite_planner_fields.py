"""Separate schema values from rendered planner template slots.

Exact-19 values remain unmodified for validation/audit. Rendered strings use
disjoint names, so processor transport cannot erase AR grammar or change the
known-outcome prefix between training and target-free inference.
"""

PLANNER_RENDERED_FIELDS = (
    "known_previous_outcome", "outcome_target", "next_decision",
    "active_skills_semantic_json", "memory_update", "task_complete",
)


def _render(field, value):
    if field == "task_complete":
        if not isinstance(value, bool):
            raise ValueError("planner task_complete schema value must be boolean")
        value = str(value).lower()
    labels = {
        "known_previous_outcome": "Known previous outcome",
        "outcome_target": "Previous outcome", "next_decision": "Decision",
        "active_skills_semantic_json": "Active skills", "memory_update": "Memory update",
        "task_complete": "Task complete",
    }
    return f"{labels[field]}: {value}"


def bind_planner_rendered_fields(sample, projection):
    """Preserve builder text before the processor installs raw schema values."""
    if projection["memlite_branch"] != "high" or "outcome_target_value" not in sample:
        return
    for field in PLANNER_RENDERED_FIELDS:
        rendered = _render(field, projection[field])
        if sample.get(field) != rendered:
            raise ValueError(f"planner builder field {field!r} disagrees with raw projection")
        alias = f"planner_{field}"
        marker = f"<{field}_text"
        if marker not in sample["template"]:
            raise ValueError(f"planner template lacks rendered field {field!r}")
        sample[alias] = rendered
        sample["template"] = sample["template"].replace(marker, f"<{alias}_text")


def planner_field_slot(sample, field):
    return f"planner_{field}" if f"planner_{field}" in sample else field


def planner_field_text(sample, field):
    slot = planner_field_slot(sample, field)
    text = str(sample[slot])
    if slot != field and text != _render(field, sample[field]):
        raise ValueError(f"planner rendered field {field!r} differs from raw schema value")
    return text


def split_planner_event_fields(text: str) -> list[str]:
    """Split field separators, preserving pipes inside JSON strings."""
    fields, start, quoted, escaped, depth = [], 0, False, False, 0
    for index, char in enumerate(text):
        if quoted:
            if escaped:
                escaped = False
            elif char == chr(92):
                escaped = True
            elif char == '"':
                quoted = False
        elif char == '"':
            quoted = True
        elif char in '[{':
            depth += 1
        elif char in ']}':
            depth -= 1
            if depth < 0:
                raise ValueError('Unbalanced planner JSON')
        elif char == '|' and depth == 0:
            fields.append(text[start:index].strip())
            start = index + 1
    if quoted or depth:
        raise ValueError('Incomplete planner JSON')
    tail = text[start:].strip()
    if tail:
        fields.append(tail)
    return fields
