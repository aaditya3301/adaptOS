"""Reference-free quality checks for live answers.

In production there is no correct answer to compare against, so each task gets cheap checks
that only look at the request and the answer. They are heuristics, calibrated against the
offline evaluation (see README), and combined with user feedback by the monitor.

Every scorer returns ``{"quality": 0..1, <individual checks>}``.
"""

from __future__ import annotations

import re
from typing import Any

from adaptos.data.formatters import sql_executes
from adaptos.evaluation.metrics import extract_answer, extract_sql

# --------------------------------------------------------------------------- SQL

SCHEMA_STATEMENT = re.compile(r"CREATE\s+TABLE\b[^;]*?\)", re.IGNORECASE | re.DOTALL)
AGG_INTENT = re.compile(
    r"\b(how many|number of|count|total|sum|average|avg|mean|highest|lowest|maximum|minimum|"
    r"max|min|most|least|largest|smallest|biggest|fewest|greatest|top|best|worst|earliest|latest)\b",
    re.IGNORECASE,
)
AGG_SQL = re.compile(r"\b(COUNT|SUM|AVG|MIN|MAX)\s*\(|\bGROUP\s+BY\b|\bORDER\s+BY\b", re.IGNORECASE)


def score_sql(user_input: str, output: str) -> dict[str, Any]:
    """Valid SQL for the given schema, and aggregation used when the question asks for one."""
    schema = "; ".join(SCHEMA_STATEMENT.findall(user_input))
    question = user_input.split("Question:", 1)[-1]
    sql = extract_sql(output)
    valid = bool(schema) and sql_executes(schema, sql)
    intent_ok = bool(AGG_INTENT.search(question)) == bool(AGG_SQL.search(sql))
    quality = 0.0 if not valid else (1.0 if intent_ok else 0.4)
    return {"quality": quality, "valid_sql": valid, "intent_match": intent_ok}


# --------------------------------------------------------------------- Summarize

_WORD = re.compile(r"#?[A-Za-z0-9]+#?")
_SPEAKER = re.compile(r"#Person\d+#")
# Summary length relative to the conversation, from the training targets (10th-90th pct).
_MIN_RATIO, _MAX_RATIO = 0.10, 0.45


def score_summary(user_input: str, output: str) -> dict[str, Any]:
    """Length proportional to the conversation, speakers referenced, not a copy."""
    dialogue = user_input.split("\n\n", 1)[-1] if "#Person" in user_input else user_input
    n_dialogue, n_summary = len(_WORD.findall(dialogue)), len(_WORD.findall(output))
    ratio = n_summary / n_dialogue if n_dialogue else 0.0
    if ratio < _MIN_RATIO:
        length_score = ratio / _MIN_RATIO
    elif ratio > _MAX_RATIO:
        length_score = max(0.0, 1 - (ratio - _MAX_RATIO) / _MAX_RATIO)
    else:
        length_score = 1.0
    speakers_in = set(_SPEAKER.findall(dialogue))
    speakers_out = set(_SPEAKER.findall(output))
    speakers_ok = bool(speakers_out) and speakers_out <= speakers_in if speakers_in else True
    copied = n_summary > 0 and output.strip() in dialogue
    quality = 0.6 * length_score + 0.3 * float(speakers_ok) + 0.1 * float(not copied)
    return {
        "quality": quality,
        "length_ratio": round(ratio, 4),
        "speakers_ok": speakers_ok,
        "copied": copied,
    }


# -------------------------------------------------------------------------- Math

_NUM = r"-?\$?\d[\d,]*(?:\.\d+)?"
_TIMES, _DIVIDE = chr(0xD7), chr(0xF7)  # multiplication / division signs models sometimes write
_EQUATION = re.compile(rf"({_NUM})\s*([+\-*/x{_TIMES}{_DIVIDE}])\s*({_NUM})\s*=\s*({_NUM})")


def _num(text: str) -> float:
    return float(text.replace("$", "").replace(",", ""))


def _equation_ok(a: str, op: str, b: str, c: str) -> bool:
    x, y, z = _num(a), _num(b), _num(c)
    if op in ("x", "*", _TIMES):
        v = x * y
    elif op in ("/", _DIVIDE):
        if y == 0:
            return False
        v = x / y
    elif op == "+":
        v = x + y
    else:
        v = x - y
    return abs(v - z) <= max(1e-6, 1e-4 * abs(z)) or abs(round(v, 2) - z) < 1e-6


def score_math(user_input: str, output: str) -> dict[str, Any]:
    """Explicit final answer, and every 'a op b = c' step is arithmetically correct."""
    answer, explicit = extract_answer(output)
    equations = _EQUATION.findall(output)
    correct = sum(_equation_ok(*e) for e in equations)
    arithmetic = correct / len(equations) if equations else 0.5
    quality = 0.3 * float(explicit and answer is not None) + 0.7 * arithmetic
    return {
        "quality": quality,
        "has_final_answer": explicit,
        "equations": len(equations),
        "equations_correct": correct,
    }


ONLINE_SCORERS = {"sql": score_sql, "summarize": score_summary, "math": score_math}


def score_online(task: str | None, user_input: str, output: str) -> dict[str, Any] | None:
    """Quality checks for an answer, or None for the base model (no task-specific check)."""
    if task is None or task not in ONLINE_SCORERS:
        return None
    return ONLINE_SCORERS[task](user_input, output)
