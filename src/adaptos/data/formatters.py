"""Per-task conversion, validation and drift rules.

Each formatter turns one raw dataset row into an ``Example`` or rejects it with a reason.
The reasons are counted and logged to MLflow, so data quality is visible per dataset version.
"""

from __future__ import annotations

import re
import sqlite3
from typing import Any, ClassVar

from adaptos.data.schema import Example, example_id


class Rejected(Exception):
    """Raised by a formatter when a raw row should be dropped."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


class TaskFormatter:
    task: ClassVar[str]
    drift_rules: ClassVar[tuple[str, ...]]

    def format(self, row: dict[str, Any]) -> Example:
        raise NotImplementedError

    def is_drift(self, ex: Example, rule: str, params: dict[str, Any]) -> bool:
        raise NotImplementedError

    def _check_rule(self, rule: str) -> None:
        if rule not in self.drift_rules:
            raise ValueError(f"unknown drift rule '{rule}' for {self.task}: {self.drift_rules}")


def _clean(text: Any) -> str:
    return str(text or "").strip()


# --------------------------------------------------------------------------- SQL

_AGGREGATION = re.compile(r"\b(COUNT|SUM|AVG|MIN|MAX)\s*\(|\bGROUP\s+BY\b", re.IGNORECASE)


_SCHEMA_STMT = re.compile(r"^\s*CREATE\s+TABLE\b", re.IGNORECASE)
_SELECT_STMT = re.compile(r"^\s*(SELECT|WITH)\b", re.IGNORECASE)


def sql_executes(schema: str, query: str) -> bool:
    """True if ``query`` runs against an empty in-memory database built from ``schema``.

    Only CREATE TABLE statements and a single SELECT are allowed, so dataset content can
    never ATTACH files, change PRAGMAs or run anything else.
    """
    statements = [s for s in schema.split(";") if s.strip()]
    if not statements or not all(_SCHEMA_STMT.match(s) for s in statements):
        return False
    if not _SELECT_STMT.match(query) or ";" in query.rstrip().rstrip(";"):
        return False
    conn = sqlite3.connect(":memory:")
    try:
        conn.executescript(schema)
        conn.execute(query).fetchall()
        return True
    except (sqlite3.Error, sqlite3.Warning, ValueError):
        return False
    finally:
        conn.close()


class SqlFormatter(TaskFormatter):
    task = "sql"
    drift_rules = ("aggregation",)

    def format(self, row: dict[str, Any]) -> Example:
        question, schema, query = (_clean(row.get(k)) for k in ("question", "context", "answer"))
        if not (question and schema and query):
            raise Rejected("empty_field")
        if not sql_executes(schema, query):
            raise Rejected("sql_does_not_execute")
        text = f"{schema}\n\nQuestion: {question}"
        return Example(
            id=example_id(self.task, text),
            task=self.task,
            input=text,
            target=query,
            meta={"schema": schema, "question": question},
        )

    def is_drift(self, ex: Example, rule: str, params: dict[str, Any]) -> bool:
        self._check_rule(rule)
        return bool(_AGGREGATION.search(ex.target))


# --------------------------------------------------------------------- Summarize

_SPEAKER = re.compile(r"#Person\d+#")


class SummarizeFormatter(TaskFormatter):
    task = "summarize"
    drift_rules = ("long_dialogue",)

    def format(self, row: dict[str, Any]) -> Example:
        dialogue, summary = _clean(row.get("dialogue")), _clean(row.get("summary"))
        if not (dialogue and summary):
            raise Rejected("empty_field")
        if len(summary) >= len(dialogue):
            raise Rejected("summary_not_shorter")
        text = f"Summarize this conversation:\n\n{dialogue}"
        return Example(
            id=example_id(self.task, text),
            task=self.task,
            input=text,
            target=summary,
            meta={
                "topic": _clean(row.get("topic")),
                "turns": dialogue.count("\n") + 1,
                "speakers": len(set(_SPEAKER.findall(dialogue))),
            },
        )

    def is_drift(self, ex: Example, rule: str, params: dict[str, Any]) -> bool:
        self._check_rule(rule)
        return ex.meta["turns"] >= int(params.get("min_turns", 14))


# -------------------------------------------------------------------------- Math

_CALC_ANNOTATION = re.compile(r"<<[^>]*>>")
_NUMBER = re.compile(r"^-?\d+(\.\d+)?$")


def parse_gsm8k_answer(answer: str) -> tuple[str, str]:
    """Split a GSM8K answer into (reasoning, final numeric answer)."""
    if "####" not in answer:
        raise Rejected("no_final_answer")
    reasoning, final = answer.rsplit("####", 1)
    final = final.strip().replace(",", "").rstrip(".")
    if not _NUMBER.match(final):
        raise Rejected("final_answer_not_numeric")
    return _CALC_ANNOTATION.sub("", reasoning).strip(), final


class MathFormatter(TaskFormatter):
    task = "math"
    drift_rules = ("multi_step",)

    def format(self, row: dict[str, Any]) -> Example:
        question, answer = _clean(row.get("question")), _clean(row.get("answer"))
        if not (question and answer):
            raise Rejected("empty_field")
        reasoning, final = parse_gsm8k_answer(answer)
        if not reasoning:
            raise Rejected("empty_reasoning")
        return Example(
            id=example_id(self.task, question),
            task=self.task,
            input=question,
            target=f"{reasoning}\nFinal answer: {final}",
            meta={"final_answer": final, "steps": reasoning.count("\n") + 1},
        )

    def is_drift(self, ex: Example, rule: str, params: dict[str, Any]) -> bool:
        self._check_rule(rule)
        return ex.meta["steps"] >= int(params.get("min_steps", 5))


FORMATTERS: dict[str, TaskFormatter] = {
    f.task: f for f in (SqlFormatter(), SummarizeFormatter(), MathFormatter())
}


def get_formatter(task: str) -> TaskFormatter:
    try:
        return FORMATTERS[task]
    except KeyError:
        raise ValueError(f"no formatter for task '{task}'; known: {sorted(FORMATTERS)}") from None
