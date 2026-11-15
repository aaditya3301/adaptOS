"""Task metrics used by offline evaluation and the promotion gate.

Every scorer returns a dict of metrics in [0, 1]; the key named in the adapter's
``evaluation.primary_metric`` drives the gate.
"""

from __future__ import annotations

import random
import re
import sqlite3
import time
from collections import Counter
from collections.abc import Sequence
from typing import Any

from adaptos.data.formatters import is_safe_sql
from adaptos.data.schema import Example

# ------------------------------------------------------------------------------ SQL

_FENCE = re.compile(r"```(?:sql)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)
_STRING_LIT = re.compile(r"""'([^']*)'|"([^"]*)\"""")
_NUMBER_LIT = re.compile(r"(?<![\w.])-?\d+(?:\.\d+)?(?![\w.])")
_WORDS = ("alpha", "beta", "gamma", "delta", "north", "south", "red", "blue", "x1", "y2")
_SEEDS = (0, 1, 2, 3, 4)
_ROWS_PER_TABLE = 30
_QUERY_TIMEOUT_SEC = 2.0


def extract_sql(text: str) -> str:
    """Pull the query out of a model reply (code fences, trailing text, extra statements)."""
    m = _FENCE.search(text)
    sql = (m.group(1) if m else text).strip()
    return sql.split(";")[0].strip()


def _normalize_sql(sql: str) -> str:
    return " ".join(sql.replace('"', "'").lower().rstrip(";").split())


def _literals(sql: str) -> list[Any]:
    """String and number literals in the reference query, used to seed synthetic rows."""
    values: list[Any] = [a or b for a, b in _STRING_LIT.findall(sql)]
    for n in _NUMBER_LIT.findall(_STRING_LIT.sub("", sql)):
        values.append(float(n) if "." in n else int(n))
    return values


_CONDITION = re.compile(
    r"""(\w+)\s*(<=|>=|<>|!=|=|<|>|\bLIKE\b)\s*('[^']*'|"[^"]*"|-?\d+(?:\.\d+)?)""",
    re.IGNORECASE,
)


def _satisfying_values(gold: str) -> dict[str, Any]:
    """Column -> a value that satisfies the reference query's condition on that column."""
    values: dict[str, Any] = {}
    for col, op, raw in _CONDITION.findall(gold):
        if raw[0] in "'\"":
            value: Any = raw[1:-1].replace("%", "")
        else:
            value = float(raw) if "." in raw else int(raw)
        op = op.upper()
        if op in ("<>", "!="):
            continue
        if isinstance(value, int | float) and op in ("<", ">"):
            value = value - 1 if op == "<" else value + 1
        values.setdefault(col.lower(), value)
    return values


def _build_db(
    schema: str, literals: Sequence[Any], satisfying: dict[str, Any], seed: int
) -> sqlite3.Connection:
    """Fill every table with synthetic rows. About a third of the rows satisfy all of the
    reference query's conditions at once, so WHERE clauses are actually exercised."""
    rng = random.Random(seed)
    conn = sqlite3.connect(":memory:")
    conn.executescript(schema)
    tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")]
    for table in tables:
        cols = conn.execute(f'PRAGMA table_info("{table}")').fetchall()  # (cid, name, type, ...)
        rows = []
        for _ in range(_ROWS_PER_TABLE):
            row = []
            match_row = satisfying and rng.random() < 0.35
            for _cid, name, ctype, *_ in cols:
                if match_row and name.lower() in satisfying:
                    row.append(satisfying[name.lower()])
                elif literals and rng.random() < 0.5:
                    row.append(rng.choice(literals))
                elif any(t in (ctype or "").upper() for t in ("INT", "REAL", "NUM", "FLOA")):
                    row.append(rng.randint(0, 300))
                else:
                    row.append(rng.choice(_WORDS))
            rows.append(row)
        placeholders = ",".join("?" * len(cols))
        conn.executemany(f'INSERT OR IGNORE INTO "{table}" VALUES ({placeholders})', rows)
    return conn


def _run(conn: sqlite3.Connection, sql: str) -> list[tuple] | None:
    deadline = time.monotonic() + _QUERY_TIMEOUT_SEC
    conn.set_progress_handler(lambda: int(time.monotonic() > deadline), 10_000)
    try:
        return conn.execute(sql).fetchall()
    except (sqlite3.Error, sqlite3.Warning, ValueError, OverflowError):
        return None


def execution_match(schema: str, gold: str, pred: str) -> tuple[bool, bool]:
    """(results match on every synthetic database, prediction is valid SQL)."""
    if not is_safe_sql(schema, pred):
        return False, False
    literals = _literals(gold)
    satisfying = _satisfying_values(gold)
    ordered = "ORDER BY" in gold.upper()
    valid = True
    for seed in _SEEDS:
        try:
            conn = _build_db(schema, literals, satisfying, seed)
        except sqlite3.Error:  # unusual schema we cannot populate: fall back to exact match
            return _normalize_sql(pred) == _normalize_sql(gold), True
        try:
            g, p = _run(conn, gold), _run(conn, pred)
        finally:
            conn.close()
        if p is None:
            return False, False
        if g is None:
            return False, valid
        if (g != p) if ordered else (Counter(g) != Counter(p)):
            return False, valid
    return True, valid


def score_sql(ex: Example, prediction: str) -> dict[str, float]:
    pred = extract_sql(prediction)
    match, valid = execution_match(ex.meta["schema"], ex.target, pred)
    return {
        "execution_match": float(match),
        "exact_match": float(_normalize_sql(pred) == _normalize_sql(ex.target)),
        "valid_sql": float(valid),
    }


# ------------------------------------------------------------------------ Summarize

_TOKEN = re.compile(r"#?[a-z0-9]+#?")


def _lcs(a: Sequence[str], b: Sequence[str]) -> int:
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b, 1):
            cur.append(prev[j - 1] + 1 if x == y else max(prev[j], cur[j - 1]))
        prev = cur
    return prev[-1]


def rouge_l(prediction: str, reference: str) -> float:
    """ROUGE-L F1 on lowercase word tokens (speaker tags like #person1# kept as tokens)."""
    p, r = _TOKEN.findall(prediction.lower()), _TOKEN.findall(reference.lower())
    if not p or not r:
        return 0.0
    lcs = _lcs(p, r)
    if lcs == 0:
        return 0.0
    precision, recall = lcs / len(p), lcs / len(r)
    return 2 * precision * recall / (precision + recall)


def score_summary(ex: Example, prediction: str) -> dict[str, float]:
    n_pred, n_ref = len(_TOKEN.findall(prediction.lower())), len(_TOKEN.findall(ex.target.lower()))
    return {
        "rougeL": rouge_l(prediction, ex.target),
        "length_ratio": n_pred / n_ref if n_ref else 0.0,
    }


# ----------------------------------------------------------------------------- Math

_FINAL = re.compile(r"final answer\s*:?\s*\$?\s*(-?[\d,]*\.?\d+)", re.IGNORECASE)
_ANY_NUMBER = re.compile(r"-?[\d,]*\.?\d+")


def extract_answer(text: str) -> tuple[str | None, bool]:
    """(final numeric answer, whether an explicit 'Final answer:' line was found)."""
    found = _FINAL.findall(text)
    if found:
        return found[-1].replace(",", ""), True
    numbers = [n.replace(",", "") for n in _ANY_NUMBER.findall(text)]
    numbers = [n for n in numbers if n not in ("", "-", ".")]
    return (numbers[-1] if numbers else None), False


def _same_number(a: str | None, b: str) -> bool:
    try:
        return a is not None and abs(float(a) - float(b)) < 1e-6
    except ValueError:
        return False


def score_math(ex: Example, prediction: str) -> dict[str, float]:
    answer, explicit = extract_answer(prediction)
    return {
        "answer_accuracy": float(_same_number(answer, ex.meta["final_answer"])),
        "has_final_answer": float(explicit),
    }


# --------------------------------------------------------------------------- Common

SCORERS = {"sql": score_sql, "summarize": score_summary, "math": score_math}


def score(ex: Example, prediction: str) -> dict[str, float]:
    return SCORERS[ex.task](ex, prediction)


def aggregate(per_example: Sequence[dict[str, float]]) -> dict[str, float]:
    """Mean of every metric across examples."""
    if not per_example:
        return {}
    keys = per_example[0].keys()
    return {k: sum(s[k] for s in per_example) / len(per_example) for k in keys}
