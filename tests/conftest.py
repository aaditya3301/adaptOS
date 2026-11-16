"""Shared fixtures: an offline tiny model + tokenizer and a small processed dataset."""

from __future__ import annotations

import pytest

from adaptos.config import REPO_ROOT, Settings, load_config


class FakeTokenizer:
    """Character-level tokenizer with a simple chat template (no download needed)."""

    pad_token_id = 0
    eos_token_id = 1
    vocab_size = 64

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        text = "".join(f"<{m['role']}>{m['content']}</{m['role']}>" for m in messages)
        return text + ("<assistant>" if add_generation_prompt else "")

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [2 + (ord(c) % 62) for c in text]}


def tiny_model_loader():
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    torch.manual_seed(0)
    config = transformers.Qwen2Config(
        vocab_size=FakeTokenizer.vocab_size,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        max_position_embeddings=1024,
    )
    return FakeTokenizer(), transformers.Qwen2ForCausalLM(config)


def sql_rows(n: int) -> list[dict]:
    schema = "CREATE TABLE t (id INTEGER, name VARCHAR, age INTEGER)"
    return [
        {
            "question": f"question number {i}",
            "context": schema,
            "answer": "SELECT COUNT(*) FROM t"
            if i % 4 == 0
            else f"SELECT name FROM t WHERE age > {i}",
        }
        for i in range(n)
    ]


@pytest.fixture
def app_config():
    return load_config(REPO_ROOT / "configs")


@pytest.fixture
def small_sql_cfg(app_config):
    cfg = app_config.adapters["sql"]
    ds = cfg.dataset.model_copy(update={"train_size": 20, "eval_size": 5, "golden_size": 5})
    ds.drift = ds.drift.model_copy(update={"train_size": 6, "eval_size": 4})
    return cfg.model_copy(update={"dataset": ds})


@pytest.fixture
def tmp_settings(tmp_path):
    return Settings(
        config_dir=REPO_ROOT / "configs",
        data_dir=tmp_path / "data",
        artifacts_dir=tmp_path / "artifacts",
        mlflow_tracking_uri="http://127.0.0.1:9",
    )
