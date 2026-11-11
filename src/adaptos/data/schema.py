"""The single record format every task is converted to."""

from __future__ import annotations

import hashlib
from typing import Any

from pydantic import BaseModel, Field

# Split names written by the pipeline.
#   train / eval   : fit the adapter and measure it offline
#   golden         : fixed canary set; never changes between retrains (promotion gate)
#   drift_train    : "new kind" of labeled data the retraining pipeline receives
#   drift_eval     : measures how the adapter does on that new kind of request
SPLITS = ("train", "eval", "golden", "drift_train", "drift_eval")


class Example(BaseModel):
    id: str
    task: str
    input: str  # the user message, as a user would send it
    target: str  # the reference answer
    meta: dict[str, Any] = Field(default_factory=dict)


def example_id(task: str, text: str) -> str:
    """Stable id from content, so identical inputs collapse to one example."""
    digest = hashlib.sha1(f"{task}\x00{_normalize(text)}".encode()).hexdigest()
    return f"{task}-{digest[:12]}"


def _normalize(text: str) -> str:
    return " ".join(text.lower().split())
