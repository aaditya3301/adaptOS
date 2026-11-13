"""Turn examples into token ids with loss only on the answer tokens."""

from __future__ import annotations

import random
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Any

from adaptos.data.schema import Example
from adaptos.prompts import build_messages

IGNORE_INDEX = -100  # label value ignored by the cross-entropy loss


@dataclass
class TokenizedExample:
    input_ids: list[int]
    labels: list[int]


def tokenize_example(tokenizer: Any, ex: Example, max_seq_len: int) -> TokenizedExample:
    """Prompt tokens get IGNORE_INDEX labels; the model only learns to produce the answer."""
    messages = build_messages(ex.task, ex.input)
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    full = tokenizer.apply_chat_template(
        [*messages, {"role": "assistant", "content": ex.target}],
        tokenize=False,
        add_generation_prompt=False,
    )
    if not full.startswith(prompt):
        raise ValueError("chat template does not extend the prompt; cannot mask prompt tokens")

    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    answer_ids = tokenizer(full[len(prompt) :], add_special_tokens=False)["input_ids"]
    input_ids = (prompt_ids + answer_ids)[:max_seq_len]
    labels = ([IGNORE_INDEX] * len(prompt_ids) + answer_ids)[:max_seq_len]
    return TokenizedExample(input_ids=input_ids, labels=labels)


def collate(batch: Sequence[TokenizedExample], pad_id: int) -> dict[str, Any]:
    """Right-pad a batch to its longest sequence (dynamic padding)."""
    import torch

    width = max(len(b.input_ids) for b in batch)
    ids, labels, mask = [], [], []
    for b in batch:
        pad = width - len(b.input_ids)
        ids.append(b.input_ids + [pad_id] * pad)
        labels.append(b.labels + [IGNORE_INDEX] * pad)
        mask.append([1] * len(b.input_ids) + [0] * pad)
    return {
        "input_ids": torch.tensor(ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "attention_mask": torch.tensor(mask, dtype=torch.long),
    }


def length_grouped_batches(
    items: Sequence[TokenizedExample], batch_size: int, seed: int, group_factor: int = 16
) -> Iterator[list[TokenizedExample]]:
    """Shuffle, then sort inside large groups so batches hold similar lengths (less padding)."""
    order = list(range(len(items)))
    random.Random(seed).shuffle(order)
    group = batch_size * group_factor
    batches = []
    for start in range(0, len(order), group):
        chunk = sorted(order[start : start + group], key=lambda i: len(items[i].input_ids))
        batches += [chunk[i : i + batch_size] for i in range(0, len(chunk), batch_size)]
    random.Random(seed + 1).shuffle(batches)
    for idxs in batches:
        yield [items[i] for i in idxs]
