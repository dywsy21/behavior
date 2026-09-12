"""Shared, strict MEM-Lite text protocol for generation, training and serving.

Parsing is validation, not repair: an invalid proposal must never mutate live
memory or become an action-network conditioning string.
"""
from __future__ import annotations

import re


class MemLiteProtocolError(ValueError):
    """The planner did not produce one complete, admissible proposal."""


class ActionDecodeError(ValueError):
    """AR output cannot be interpreted as an explicit grouped action."""


VALID_STATUSES = frozenset({"CONTINUE", "DONE", "FAILED", "REPLAN"})
TERMINAL_INTENTS = frozenset({"task complete", "task completed", "done"})


def is_terminal_intent(intent: str) -> bool:
    return str(intent).strip().rstrip(".").casefold() in TERMINAL_INTENTS


def parse_memlite_output(text: str) -> dict[str, str]:
    if not isinstance(text, str) or not text.strip():
        raise MemLiteProtocolError("Empty high-level output")
    raw = text.strip()
    if len(raw) > 8192 or "\x00" in raw:
        raise MemLiteProtocolError("Oversized or invalid high-level output")
    if "<HL_END>" in raw:
        body, suffix = raw.split("<HL_END>", 1)
        if suffix.strip() or "<HL_END>" in suffix:
            raise MemLiteProtocolError("Unexpected text after HL_END")
    else:
        body = raw
    parts = [part.strip() for part in body.split("|")]
    if parts and parts[-1] == "":
        parts.pop()  # canonical protocol has one separator before HL_END
    if len(parts) != 3:
        raise MemLiteProtocolError("Expected exactly Intent, Updated Memory and Status")
    patterns = (
        r"(?:Intent|Subtask|Next intent)\s*:\s*(.+)",
        r"(?:Updated Memory|Memory update|Memory)\s*:\s*(.+)",
        r"(?:Status|Intent status)\s*:\s*([A-Za-z]+)",
    )
    values = []
    for index, (part, pattern) in enumerate(zip(parts, patterns)):
        match = re.fullmatch(pattern, part, flags=re.IGNORECASE | re.DOTALL)
        if match is None or not match.group(1).strip():
            raise MemLiteProtocolError(f"Missing, reordered or malformed field {index + 1}")
        values.append(match.group(1).strip())
    intent, memory, status = values[0], values[1], values[2].upper()
    if len(intent) > 2048 or len(memory) > 4096:
        raise MemLiteProtocolError("High-level content exceeds its protocol limit")
    if status not in VALID_STATUSES:
        raise MemLiteProtocolError(f"Unknown high-level status: {status!r}")
    if status == "CONTINUE" and is_terminal_intent(intent):
        raise MemLiteProtocolError("Terminal intent cannot be an executable CONTINUE")
    if status == "DONE" and not is_terminal_intent(intent):
        raise MemLiteProtocolError("DONE must identify task completion, not an executable skill")
    return {"intent": intent, "memory": memory, "status": status, "raw": raw}
