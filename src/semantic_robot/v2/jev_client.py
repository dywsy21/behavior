"""Bounded TypeSafe native Choice API. No OpenRouter, redirects or hidden retries.

Credentials are deliberately absent from receipts, reprs and exceptions. The
client selects symbols; it never creates robot commands or certifies safety.
"""
import json
from http.client import HTTPException
import math
import os
import stat
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import HTTPRedirectHandler, Request, build_opener

from .wall_budget import require_time

ENDPOINT = "https://api.typesafe.ai/v1/systemone"
MODEL = "jev-1.13.0"
MAX_REQUEST_BYTES = 100_000
MAX_RESPONSE_BYTES = 1_000_000


class JevError(RuntimeError):
    """A safe, credential-free failure; the caller must not actuate."""


class JevAbstained(JevError):
    """A valid no-action decision, not a transport or schema failure."""


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise JevError("TypeSafe redirect refused; credentials not forwarded")


def load_key(path=None):
    if path is None:
        key = os.environ.get("TYPESAFE_API_KEY", "")
    else:
        # A private regular file, not a symlink or a CLI argument containing a key.
        fd = os.open(Path(path), os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        try:
            info = os.fstat(fd)
            if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
                raise JevError("TypeSafe credential file must be owner-only (0600)")
            key = os.read(fd, 4097).decode("ascii").strip()
        finally:
            os.close(fd)
    if not isinstance(key, str) or not 16 <= len(key) <= 4096 or any(c.isspace() for c in key):
        raise JevError("Set TYPESAFE_API_KEY or provide a private credential file")
    return key


def choice(instructions, criteria):
    if not isinstance(instructions, str) or not instructions.strip():
        raise ValueError("Explicit Choice instructions required")
    if not isinstance(criteria, dict) or not 2 <= len(criteria) <= 255:
        raise ValueError("Choice requires 2..255 explicit options")
    if any(not isinstance(k, str) or not k or not isinstance(v, (str, dict)) for k, v in criteria.items()):
        raise ValueError("Choice requires named, described options")
    return {"type": "choice", "instructions": instructions, "criteria": criteria}


def strict_response(raw):
    def pairs(values):
        result = {}
        for key, value in values:
            if key in result:
                raise JevError("Duplicate JSON response field")
            result[key] = value
        return result
    try:
        return json.loads(raw, object_pairs_hook=pairs,
                          parse_constant=lambda _: (_ for _ in ()).throw(JevError("Nonfinite response")))
    except (ValueError, UnicodeError):
        raise JevError("TypeSafe returned invalid JSON") from None


def probability(value):
    return type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1


def distribution_receipt(probs):
    """Native API has returned two-decimal probabilities summing to .99.

    Preserve raw values/argmax. Permit at most two percentage points ONLY for
    values on the .01 grid, further bounded by their rounding envelope. This is
    transport compatibility, never a physical-success or confidence gate.
    """
    valid = all(probability(p) for p in probs.values())
    quantized = valid and all(abs(p * 100 - round(p * 100)) <= 1e-9 for p in probs.values())
    mass = sum(probs.values()) if valid else None
    tolerance = min(.02, .005 * len(probs)) + 1e-9 if quantized else 1e-6
    return {"raw_probability_sum": mass, "two_decimal_grid": quantized,
            "mass_tolerance": tolerance, "locally_renormalized": False,
            "passed": valid and abs(mass - 1) <= tolerance}


def validate_response(value, questions, model):
    if not isinstance(value, dict) or value.get("model") != model:
        raise JevError("Unexpected TypeSafe model identity")
    answers = value.get("answers")
    if not isinstance(answers, dict) or set(answers) != set(questions):
        raise JevError("Missing or unexpected TypeSafe answers")
    for name, question in questions.items():
        answer = answers[name]
        if not isinstance(answer, dict) or answer.get("type") != "choice":
            raise JevError("Expected Choice answer")
        probs = answer.get("probabilities")
        selected = answer.get("choice")
        if not isinstance(selected, str) or selected not in question["criteria"]:
            raise JevError("Choice outside current options")
        if not isinstance(probs, dict) or set(probs) != set(question["criteria"]):
            raise JevError("Choice probability options mismatch")
        if (not distribution_receipt(probs)["passed"] or
                not probability(answer.get("confidence"))):
            # Numeric-only diagnostics, never echoed server strings/headers.
            finite_probs = all(type(p) in (int, float) and math.isfinite(p) for p in probs.values())
            mass = sum(probs.values()) if finite_probs else None
            confidence = answer.get("confidence")
            safe_conf = confidence if type(confidence) in (int, float) and math.isfinite(confidence) else None
            raise JevError("Invalid Choice probability distribution: " + json.dumps({
                "option_count": len(probs), "probability_sum": mass,
                "invalid_probability_count": sum(not probability(p) for p in probs.values()),
                "confidence": safe_conf}, allow_nan=False))
        if probs[selected] + 1e-7 < max(probs.values()):
            raise JevError("Choice disagrees with distribution")
    usage = value.get("usage")
    if not isinstance(usage, dict) or any(type(usage.get(k)) is not int or usage[k] < 0
                                          for k in ("input_tokens", "output_tokens")):
        raise JevError("Missing or invalid TypeSafe token accounting")
    # Explicitly select fields; never persist arbitrary echoed server content.
    return {"model": model, "answers": {
        name: {k: answers[name][k] for k in ("type", "choice", "probabilities", "confidence")}
        for name in questions}, "usage": {k: usage[k] for k in ("input_tokens", "output_tokens")},
        "probability_validation": {name: distribution_receipt(answers[name]["probabilities"]) for name in questions}}


class JevClient:
    def __init__(self, *, key_file=None, api_key=None, max_calls=40, timeout=20.0,
                 model=MODEL, opener=None, journal=None):
        if type(max_calls) is not int or max_calls < 1 or not 0 < timeout <= 60:
            raise ValueError("Finite request count and 0..60s timeout required")
        if model != MODEL:
            raise ValueError("This adapter pins jev-1.13.0; qualify another version explicitly")
        if api_key is not None and key_file is not None:
            raise ValueError("Choose one credential source")
        self._key = api_key if api_key is not None else load_key(key_file)
        if not isinstance(self._key, str) or not self._key or any(c.isspace() for c in self._key):
            raise JevError("Invalid credential")
        self.model, self.timeout, self.max_calls = model, timeout, max_calls
        self.calls, self.input_tokens, self.output_tokens = 0, 0, 0
        self.validated_responses = 0
        self.deadline, self.last_call = None, None
        self._opener = opener if opener is not None else build_opener(NoRedirect())
        if journal is not None and not callable(journal): raise ValueError("Callable request journal required")
        self._journal = journal

    def record(self, event, **fields):
        if self._journal is not None:
            self._journal({"call":self.calls,"model":self.model,"event":event,**fields})

    @property
    def identity(self):
        return {"provider": "typesafe", "endpoint": ENDPOINT, "model": self.model,
                "input": "structured_text_only", "retries": 0,
                "probability_is_not_physical_success": True}

    def evaluate(self, state, questions):
        self.last_call = None
        left = require_time(self.deadline)
        if self.calls >= self.max_calls:
            raise JevError("TypeSafe request budget exhausted")
        if not isinstance(questions, dict) or not 1 <= len(questions) <= 8:
            raise ValueError("Bounded question map required")
        for name, value in questions.items():
            if not isinstance(name, str) or not isinstance(value, dict) or value.get("type") != "choice":
                raise ValueError("This robot adapter accepts named Choice questions only")
            if set(value) != {"type", "instructions", "criteria"}:
                raise ValueError("Unexpected Choice question fields")
            choice(value["instructions"], value["criteria"])
        payload = {"model": self.model, "state": state, "questions": questions}
        raw = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode()
        if len(raw) > MAX_REQUEST_BYTES:
            raise ValueError("TypeSafe robot context too large; no silent truncation")
        if self._key in raw.decode():
            raise JevError("Credential must not appear in state or questions")
        request = Request(ENDPOINT, raw, {"Authorization": "Bearer " + self._key,
                          "Content-Type": "application/json", "Accept": "application/json"}, method="POST")
        self.calls += 1  # failed/timeout requests consume budget too
        started = time.perf_counter()
        self.record("attempt", request_bytes=len(raw), question_names=list(questions))
        try:
            with self._opener.open(request, timeout=min(self.timeout, left) if left is not None else self.timeout) as response:
                if response.geturl() != ENDPOINT:
                    raise JevError("Unexpected TypeSafe response URL")
                data = response.read(MAX_RESPONSE_BYTES + 1)
            if len(data) > MAX_RESPONSE_BYTES:
                raise JevError("TypeSafe response exceeded byte budget")
            if self._key.encode() in data:
                raise JevError("TypeSafe response contained credential; receipt suppressed")
            result = validate_response(strict_response(data), questions, self.model)
        except HTTPError as exc:
            self.record("http_error", http_status=exc.code)
            raise JevError(f"TypeSafe HTTP {exc.code}; no retry or actuation") from None
        except (URLError, TimeoutError, OSError, HTTPException):
            self.record("network_error")
            raise JevError("TypeSafe network/timeout failure; no retry or actuation") from None
        except (JevError, ValueError):
            self.record("rejected_response")
            raise
        self.input_tokens += result["usage"]["input_tokens"]
        self.output_tokens += result["usage"]["output_tokens"]
        self.validated_responses += 1
        result["roundtrip_s"] = time.perf_counter() - started
        self.record("validated", usage=result["usage"], roundtrip_s=result["roundtrip_s"],
                    selections={name: answer["choice"] for name, answer in result["answers"].items()})
        # Match the harness receipt convention; no headers/key/raw response.
        self.last_call = {"result": result, "request": {**payload, "images": []}}
        require_time(self.deadline)  # late answers must never authorize motion
        return result, self.last_call
