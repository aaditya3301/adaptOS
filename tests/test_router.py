import hashlib

import numpy as np
import pytest
from tests.conftest import sql_rows

from adaptos.config import REPO_ROOT, load_config
from adaptos.data.pipeline import build_splits, write_dataset
from adaptos.routing import OTHER, RouterModel

LABELS = ["math", OTHER, "sql"]


def hashing_embed(texts, dim=128):
    """Deterministic stand-in for the sentence-embedding model (character trigram hashing)."""
    out = np.zeros((len(texts), dim), dtype=np.float32)
    for i, text in enumerate(texts):
        t = text.lower()
        for j in range(len(t) - 2):
            out[i, int(hashlib.md5(t[j : j + 3].encode()).hexdigest(), 16) % dim] += 1
    norms = np.linalg.norm(out, axis=1, keepdims=True)
    return out / np.maximum(norms, 1e-9)


def _router(threshold=0.6):
    # One-hot "embeddings": dimension i strongly indicates label i.
    coef = np.eye(3) * 10
    return RouterModel(LABELS, coef, np.zeros(3), "m", "rev", threshold, embed_fn=lambda t: None)


def test_routes_confident_request_to_adapter():
    d = _router().decide(_router().probabilities_from_embeddings(np.array([[0, 0, 1.0]]))[0])
    assert (d.adapter, d.label, d.reason) == ("sql", "sql", "adapter")
    assert d.confidence > 0.99


def test_general_request_goes_to_base_model():
    r = _router()
    d = r.decide(r.probabilities_from_embeddings(np.array([[0, 1.0, 0]]))[0])
    assert d.adapter is None and d.reason == "general_request"


def test_low_confidence_goes_to_base_model():
    r = _router(threshold=0.9)
    d = r.decide(r.probabilities_from_embeddings(np.array([[0.1, 0, 0.12]]))[0])
    assert d.adapter is None and d.reason == "low_confidence" and d.label == "sql"


def test_probabilities_sum_to_one():
    p = _router().probabilities_from_embeddings(np.random.default_rng(0).normal(size=(5, 3)))
    assert np.allclose(p.sum(axis=1), 1.0)


def test_save_and_load_roundtrip(tmp_path):
    r = _router()
    r.save(tmp_path)
    loaded = RouterModel.load(tmp_path, embed_fn=lambda t: None)
    assert loaded.labels == LABELS and loaded.threshold == 0.6
    assert np.allclose(loaded.coef, r.coef)
    assert RouterModel.load(tmp_path, threshold=0.8, embed_fn=lambda t: None).threshold == 0.8


def test_rejects_mismatched_shapes():
    with pytest.raises(ValueError, match="shapes"):
        RouterModel(LABELS, np.eye(2), np.zeros(3), "m", "r", 0.5, embed_fn=lambda t: None)


def _math_rows(n):
    rows = []
    for i in range(n):
        steps = 6 if i % 4 == 0 else 2  # every 4th problem is "multi-step" (drift)
        reasoning = "\n".join(f"step {k}: {i} + {k} = {i + k}" for k in range(steps))
        rows.append({"question": f"Tom has {i} apples and buys {i % 7} more. How many apples?",
                     "answer": f"{reasoning}\n#### {i}"})  # fmt: skip
    return rows


def test_train_router_end_to_end(tmp_settings):
    pytest.importorskip("sklearn")
    from adaptos.training.train_router import train_router

    app = load_config(REPO_ROOT / "configs")
    small = {}
    for task, rows in (("sql", sql_rows(200)), ("math", _math_rows(200))):
        cfg = app.adapters[task]
        ds = cfg.dataset.model_copy(update={"train_size": 30, "eval_size": 10, "golden_size": 10})
        ds.drift = ds.drift.model_copy(update={"train_size": 5, "eval_size": 5})
        cfg = cfg.model_copy(update={"dataset": ds})
        splits, stats = build_splits(cfg, rows)
        write_dataset(cfg, splits, stats, tmp_settings.data_dir / "processed" / task)
        small[task] = cfg
    other = app.router.other_dataset.model_copy(
        update={"train_size": 30, "eval_size": 10, "test_size": 10}
    )
    app = app.model_copy(
        update={"adapters": small, "router": app.router.model_copy(update={"other_dataset": other})}
    )
    poem = "Write a short poem about the colour number {}"
    other_rows = [
        {"instruction": poem.format(i), "context": "", "category": "creative_writing"}
        for i in range(60)
    ]
    other_rows.append({"instruction": "Summarize it", "context": "x", "category": "summarization"})

    summary = train_router(
        settings=tmp_settings, app=app, other_rows=other_rows, embed_fn=hashing_embed,
        use_mlflow=False,
    )  # fmt: skip

    assert summary["metrics"]["test"]["accuracy"] >= 0.9
    assert summary["metrics"]["drift"]["accuracy"] >= 0.9  # drifted requests keep their task
    saved = RouterModel.load(
        tmp_settings.artifacts_dir / "router" / summary["dataset_versions"]["router"],
        embed_fn=hashing_embed,
    )
    assert set(saved.labels) == {"math", "sql", OTHER}
    probe = "CREATE TABLE t (id INTEGER, name VARCHAR, age INTEGER)\n\nQuestion: question number 7"
    assert saved.route(probe).label == "sql"
