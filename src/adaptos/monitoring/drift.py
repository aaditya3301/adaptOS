"""Input (feature) drift: are recent requests unlike the data the live adapter was trained on?

Each task has a few interpretable request features. A *reference* (mean and std of each feature
over the adapter's training inputs) is saved with every adapter version. The drift score of a
window of requests is the largest standardized shift of a feature mean:

    drift = max_f |mean_window(f) - mean_train(f)| / std_train(f)

Calibrated on held-out data (see README): normal 50-request windows stay below ~0.5, while the
drift slices reach 0.9 (math) to 2.8 (sql). Sentence-embedding distance was also tested and did
not separate drifted from normal requests, so it is not used.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from adaptos.monitoring.quality import AGG_INTENT

REFERENCE_FILE = "drift_reference.json"
_SPEAKER = re.compile(r"#Person\d+#")
_NUMBER = re.compile(r"\d+(?:\.\d+)?")


def _sql(text: str) -> dict[str, float]:
    schema, _, question = text.partition("Question:")
    return {
        "question_words": len(question.split()),
        "asks_aggregate": float(bool(AGG_INTENT.search(question))),
        "schema_columns": schema.count(",") + 1,
    }


def _summarize(text: str) -> dict[str, float]:
    dialogue = text.split("\n\n", 1)[-1]
    return {
        "words": len(dialogue.split()),
        "turns": dialogue.count("\n") + 1,
        "speakers": len(set(_SPEAKER.findall(dialogue))),
    }


def _math(text: str) -> dict[str, float]:
    return {"words": len(text.split()), "numbers": len(_NUMBER.findall(text))}


FEATURES: dict[str, Callable[[str], dict[str, float]]] = {
    "sql": _sql,
    "summarize": _summarize,
    "math": _math,
}


def features(task: str, text: str) -> dict[str, float]:
    return FEATURES[task](text)


def build_reference(task: str, inputs: Sequence[str]) -> dict[str, Any]:
    rows = [features(task, t) for t in inputs]
    ref: dict[str, Any] = {"task": task, "n": len(rows), "features": {}}
    for name in rows[0]:
        values = [r[name] for r in rows]
        mean = sum(values) / len(values)
        std = (sum((v - mean) ** 2 for v in values) / len(values)) ** 0.5
        ref["features"][name] = {"mean": mean, "std": std}
    return ref


def save_reference(directory: Path, reference: dict[str, Any]) -> Path:
    path = directory / REFERENCE_FILE
    path.write_text(json.dumps(reference, indent=2), encoding="utf-8")
    return path


def load_reference(directory: Path) -> dict[str, Any] | None:
    path = Path(directory) / REFERENCE_FILE
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def drift_score(
    task: str, inputs: Sequence[str], reference: dict[str, Any]
) -> tuple[float, dict[str, float]]:
    """(max |standardized shift|, shift per feature) for a window of request inputs."""
    if not inputs:
        return 0.0, {}
    rows = [features(task, t) for t in inputs]
    shifts = {}
    for name, stats in reference["features"].items():
        mean = sum(r[name] for r in rows) / len(rows)
        std = stats["std"] if stats["std"] > 1e-9 else 1.0
        shifts[name] = (mean - stats["mean"]) / std
    return max(abs(v) for v in shifts.values()), shifts
