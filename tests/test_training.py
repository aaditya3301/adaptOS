import json
import math

import pytest
from tests.conftest import FakeTokenizer, sql_rows, tiny_model_loader

from adaptos.data.pipeline import build_splits, write_dataset
from adaptos.data.schema import Example
from adaptos.prompts import SYSTEM_PROMPTS, build_messages
from adaptos.training.tokenization import (
    IGNORE_INDEX,
    collate,
    length_grouped_batches,
    tokenize_example,
)

EX = Example(
    id="sql-1", task="sql", input="CREATE TABLE t (a INT)\n\nQuestion: q?", target="SELECT a FROM t"
)


def test_every_adapter_has_a_system_prompt(app_config):
    assert set(app_config.adapters) <= set(SYSTEM_PROMPTS)


def test_build_messages_base_fallback():
    assert build_messages(None, "hi")[0]["content"] == "You are a helpful assistant."
    assert build_messages("sql", "hi")[1] == {"role": "user", "content": "hi"}


def test_only_answer_tokens_are_trained():
    tok = FakeTokenizer()
    item = tokenize_example(tok, EX, max_seq_len=10_000)
    prompt = tok.apply_chat_template(build_messages("sql", EX.input), add_generation_prompt=True)
    n_prompt = len(prompt)  # character-level tokenizer: one token per character
    assert len(item.input_ids) == len(item.labels)
    assert all(label == IGNORE_INDEX for label in item.labels[:n_prompt])
    assert item.labels[n_prompt:] == item.input_ids[n_prompt:]
    assert len(item.labels[n_prompt:]) == len(f"{EX.target}</assistant>")


def test_truncation_respects_max_seq_len():
    item = tokenize_example(FakeTokenizer(), EX, max_seq_len=20)
    assert len(item.input_ids) == len(item.labels) == 20


def test_collate_pads_and_masks():
    pytest.importorskip("torch")
    tok = FakeTokenizer()
    a = tokenize_example(tok, EX, 10_000)
    b = tokenize_example(
        tok, EX.model_copy(update={"target": "SELECT a FROM t WHERE a > 1"}), 10_000
    )
    batch = collate([a, b], pad_id=0)
    width = max(len(a.input_ids), len(b.input_ids))
    assert batch["input_ids"].shape == (2, width)
    assert int(batch["attention_mask"][0].sum()) == len(a.input_ids)
    assert (batch["labels"][0, len(a.input_ids) :] == IGNORE_INDEX).all()


def test_length_grouped_batches_cover_every_item_once():
    tok = FakeTokenizer()
    items = [
        tokenize_example(tok, EX.model_copy(update={"target": "SELECT a" + " " * i}), 10_000)
        for i in range(23)
    ]
    batches = list(length_grouped_batches(items, batch_size=4, seed=1))
    flat = [id(x) for b in batches for x in b]
    assert sorted(flat) == sorted(id(x) for x in items)
    assert all(len(b) <= 4 for b in batches)


def test_train_adapter_end_to_end_with_tiny_model(small_sql_cfg, app_config, tmp_settings):
    pytest.importorskip("peft")
    from adaptos.training.train_adapter import train_adapter

    splits, stats = build_splits(small_sql_cfg, sql_rows(200))
    write_dataset(small_sql_cfg, splits, stats, tmp_settings.data_dir / "processed" / "sql")
    app = app_config.model_copy(update={"adapters": {**app_config.adapters, "sql": small_sql_cfg}})

    result = train_adapter(
        "sql",
        settings=tmp_settings,
        app=app,
        model_loader=tiny_model_loader,
        max_steps=2,
        limit=20,  # 20 examples / batch 4 = 5 micro-batches; accum 4 -> 2 optimizer steps
        use_mlflow=False,
    )

    assert (result.adapter_dir / "adapter_config.json").exists()
    assert any(result.adapter_dir.glob("adapter_model.*"))
    info = json.loads((result.adapter_dir / "adaptos_info.json").read_text(encoding="utf-8"))
    assert info["dataset_version"] == result.params["dataset_version"]
    assert info["total_steps"] == 2
    assert math.isfinite(result.metrics["eval_loss"])
    assert result.run_id is None and result.model_version is None
