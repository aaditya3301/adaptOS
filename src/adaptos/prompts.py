"""Prompt construction shared by training, evaluation and serving.

Every component builds chat messages through this module, so the model sees exactly the
same format in production as it did in training (no training/serving skew).
Changing a prompt changes PROMPT_VERSION, which is logged with every training run.
"""

from __future__ import annotations

PROMPT_VERSION = "1"

SYSTEM_PROMPTS: dict[str, str] = {
    "sql": (
        "You are a text-to-SQL assistant. Given CREATE TABLE statements and a question, "
        "reply with a single SQLite query that answers it. Reply with SQL only."
    ),
    "summarize": (
        "You summarize conversations. Reply with a short summary of one or two sentences "
        "that keeps the speaker labels such as #Person1#."
    ),
    "math": (
        "You solve grade-school math word problems. Reason step by step, one step per line, "
        "and finish with a line of the form 'Final answer: <number>'."
    ),
}

BASE_SYSTEM_PROMPT = "You are a helpful assistant."


def build_messages(task: str | None, user_input: str) -> list[dict[str, str]]:
    """Chat messages for a request. ``task=None`` means the plain base model (router fallback)."""
    system = SYSTEM_PROMPTS[task] if task is not None else BASE_SYSTEM_PROMPT
    return [{"role": "system", "content": system}, {"role": "user", "content": user_input}]
