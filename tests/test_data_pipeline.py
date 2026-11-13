"""Pipeline tests use synthetic rows, so they run offline and fast (no Hugging Face download)."""

import pytest

from adaptos.config import REPO_ROOT, load_config
from adaptos.data.formatters import FORMATTERS
from adaptos.data.io import load_split, read_manifest, verify_dataset
from adaptos.data.pipeline import build_splits, write_dataset
from adaptos.data.schema import SPLITS


@pytest.fixture
def sql_cfg():
    cfg = load_config(REPO_ROOT / "configs").adapters["sql"]
    ds = cfg.dataset.model_copy(update={"train_size": 20, "eval_size": 5, "golden_size": 5})
    ds.drift = ds.drift.model_copy(update={"train_size": 6, "eval_size": 4})
    return cfg.model_copy(update={"dataset": ds})


def _rows(n: int) -> list[dict]:
    schema = "CREATE TABLE t (id INTEGER, name VARCHAR, age INTEGER)"
    rows = []
    for i in range(n):
        query = "SELECT COUNT(*) FROM t" if i % 4 == 0 else f"SELECT name FROM t WHERE age > {i}"
        rows.append({"question": f"question number {i}", "context": schema, "answer": query})
    rows.append(rows[1])  # duplicate
    rows.append({"question": "bad", "context": schema, "answer": "SELECT nope FROM t"})
    return rows


def test_every_configured_adapter_has_a_formatter_and_valid_drift_rule():
    for name, cfg in load_config(REPO_ROOT / "configs").adapters.items():
        assert name in FORMATTERS
        assert cfg.dataset.drift.rule in FORMATTERS[name].drift_rules


def test_split_sizes_disjoint_and_drift_separated(sql_cfg):
    splits, stats = build_splits(sql_cfg, _rows(200))
    sizes = {k: len(v) for k, v in splits.items()}
    assert sizes == {"train": 20, "eval": 5, "golden": 5, "drift_train": 6, "drift_eval": 4}
    ids = [e.id for split in splits.values() for e in split]
    assert len(ids) == len(set(ids)), "an example appears in more than one split"
    assert all("COUNT" in e.target for e in splits["drift_train"] + splits["drift_eval"])
    assert not any("COUNT" in e.target for e in splits["train"])
    assert stats["rows_scanned"] <= stats["rows_available"]


def test_build_is_deterministic(sql_cfg):
    a, _ = build_splits(sql_cfg, _rows(200))
    b, _ = build_splits(sql_cfg, _rows(200))
    assert {k: [e.id for e in v] for k, v in a.items()} == {
        k: [e.id for e in v] for k, v in b.items()
    }


def test_not_enough_clean_data_fails_loudly(sql_cfg):
    rows = _rows(8)  # too few to fill quotas, so every row is scanned
    with pytest.raises(ValueError, match="not enough clean data"):
        build_splits(sql_cfg, rows)


def test_manifest_version_is_content_hash(sql_cfg, tmp_path):
    splits, stats = build_splits(sql_cfg, _rows(200))
    m1 = write_dataset(sql_cfg, splits, stats, tmp_path / "a")
    m2 = write_dataset(sql_cfg, splits, stats, tmp_path / "b")
    assert m1["dataset_version"] == m2["dataset_version"]
    assert set(m1["splits"]) == set(SPLITS)
    assert read_manifest(tmp_path / "a")["dataset_version"] == m1["dataset_version"]


def test_tampered_split_is_detected(sql_cfg, tmp_path):
    splits, stats = build_splits(sql_cfg, _rows(200))
    write_dataset(sql_cfg, splits, stats, tmp_path)
    assert len(load_split(tmp_path, "train")) == 20
    with (tmp_path / "train.jsonl").open("a", encoding="utf-8") as f:
        f.write('{"id": "x", "task": "sql", "input": "i", "target": "t", "meta": {}}\n')
    with pytest.raises(ValueError, match="does not match its manifest"):
        verify_dataset(tmp_path)
