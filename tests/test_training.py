"""Adapter training (tiny offline model) and the router (stand-in embeddings)."""

import hashlib
import json
import math

import numpy as np
import pytest
from tests.conftest import FakeTokenizer, sql_rows, tiny_model_loader

from adaptos.data.pipeline import build_splits, write_dataset
from adaptos.data.schema import Example
from adaptos.prompts import build_messages
from adaptos.routing import OTHER, RouterModel
from adaptos.training.tokenization import IGNORE_INDEX, collate, tokenize_example

EX = Example(id="1", task="sql", input="CREATE TABLE t (a INT)\n\nQuestion: q?", target="SELECT a")


# ---------------------------------------------------------------------- adapters
def test_only_answer_tokens_are_trained():
    tok = FakeTokenizer()
    item = tokenize_example(tok, EX, max_seq_len=10_000)
    n_prompt = len(
        tok.apply_chat_template(build_messages("sql", EX.input), add_generation_prompt=True)
    )
    assert all(label == IGNORE_INDEX for label in item.labels[:n_prompt])
    assert item.labels[n_prompt:] == item.input_ids[n_prompt:] != []
    assert len(tokenize_example(tok, EX, max_seq_len=20).input_ids) == 20


def test_collate_pads_and_masks():
    pytest.importorskip("torch")
    tok = FakeTokenizer()
    a = tokenize_example(tok, EX, 10_000)
    b = tokenize_example(tok, EX.model_copy(update={"target": "SELECT a FROM t"}), 10_000)
    batch = collate([a, b], pad_id=0)
    assert int(batch["attention_mask"][0].sum()) == len(a.input_ids)
    assert (batch["labels"][0, len(a.input_ids) :] == IGNORE_INDEX).all()


def test_train_adapter_end_to_end(small_sql_cfg, app_config, tmp_settings):
    pytest.importorskip("peft")
    from adaptos.training.train_adapter import train_adapter

    splits, stats = build_splits(small_sql_cfg, sql_rows(200))
    write_dataset(small_sql_cfg, splits, stats, tmp_settings.data_dir / "processed" / "sql")
    app = app_config.model_copy(update={"adapters": {**app_config.adapters, "sql": small_sql_cfg}})
    result = train_adapter(
        "sql", settings=tmp_settings, app=app, model_loader=tiny_model_loader,
        max_steps=2, limit=20, use_mlflow=False,
    )  # fmt: skip
    assert (result.adapter_dir / "adapter_config.json").exists()
    assert (result.adapter_dir / "drift_reference.json").exists()
    info = json.loads((result.adapter_dir / "adaptos_info.json").read_text(encoding="utf-8"))
    assert info["total_steps"] == 2 and math.isfinite(result.metrics["eval_loss"])


# ------------------------------------------------------------------------ router
LABELS = ["math", OTHER, "sql"]


def _router(threshold=0.6):
    return RouterModel(LABELS, np.eye(3) * 10, np.zeros(3), "m", "r", threshold, lambda t: None)


@pytest.mark.parametrize(
    ("embedding", "threshold", "adapter", "reason"),
    [
        ([0, 0, 1.0], 0.6, "sql", "adapter"),
        ([0, 1.0, 0], 0.6, None, "general_request"),
        ([0.1, 0, 0.12], 0.9, None, "low_confidence"),
    ],
)
def test_router_decisions(embedding, threshold, adapter, reason):
    r = _router(threshold)
    d = r.decide(r.probabilities_from_embeddings(np.array([embedding]))[0])
    assert (d.adapter, d.reason) == (adapter, reason)


def test_router_save_and_load(tmp_path):
    _router().save(tmp_path)
    loaded = RouterModel.load(tmp_path, threshold=0.8, embed_fn=lambda t: None)
    assert loaded.labels == LABELS and loaded.threshold == 0.8


def hashing_embed(texts, dim=128):
    """Deterministic stand-in for the sentence-embedding model (character trigrams)."""
    out = np.zeros((len(texts), dim), dtype=np.float32)
    for i, text in enumerate(texts):
        for j in range(len(text) - 2):
            out[i, int(hashlib.md5(text[j : j + 3].lower().encode()).hexdigest(), 16) % dim] += 1
    return out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-9)


def test_train_router_end_to_end(app_config, tmp_settings):
    pytest.importorskip("sklearn")
    from adaptos.training.train_router import train_router

    math_rows = [
        {"question": f"Tom has {i} apples and buys {i % 7} more. How many apples?",
         "answer": "\n".join(f"{i} + {k} = {i + k}" for k in range(6 if i % 4 == 0 else 2))
         + f"\n#### {i}"}
        for i in range(200)
    ]  # fmt: skip
    small = {}
    for task, rows in (("sql", sql_rows(200)), ("math", math_rows)):
        cfg = app_config.adapters[task]
        ds = cfg.dataset.model_copy(update={"train_size": 30, "eval_size": 10, "golden_size": 10})
        ds.drift = ds.drift.model_copy(update={"train_size": 5, "eval_size": 5})
        small[task] = cfg = cfg.model_copy(update={"dataset": ds})
        splits, stats = build_splits(cfg, rows)
        write_dataset(cfg, splits, stats, tmp_settings.data_dir / "processed" / task)
    other = app_config.router.other_dataset.model_copy(
        update={"train_size": 30, "eval_size": 10, "test_size": 10}
    )
    app = app_config.model_copy(
        update={
            "adapters": small,
            "router": app_config.router.model_copy(update={"other_dataset": other}),
        }
    )
    other_rows = [
        {"instruction": f"Write a short poem about colour {i}", "context": "", "category": "x"}
        for i in range(60)
    ]
    summary = train_router(
        settings=tmp_settings, app=app, other_rows=other_rows, embed_fn=hashing_embed,
        use_mlflow=False,
    )  # fmt: skip
    assert summary["metrics"]["test"]["accuracy"] >= 0.9
    assert summary["metrics"]["drift"]["accuracy"] >= 0.9  # drifted requests keep their task
