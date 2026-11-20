"""Config validation, task formatters and the dataset pipeline (offline, synthetic rows)."""

import pytest
import yaml
from tests.conftest import sql_rows

from adaptos.config import REPO_ROOT, load_config
from adaptos.data.formatters import FORMATTERS, Rejected, get_formatter, sql_executes
from adaptos.data.io import load_split, verify_dataset
from adaptos.data.pipeline import build_splits, write_dataset

SCHEMA = "CREATE TABLE head (age INTEGER, name VARCHAR)"


# ------------------------------------------------------------------------- config
def test_repo_configs_are_consistent():
    app = load_config(REPO_ROOT / "configs")
    assert set(app.adapters) == {"sql", "summarize", "math"} == set(app.monitor.thresholds)
    for name, cfg in app.adapters.items():
        assert cfg.dataset.drift.rule in FORMATTERS[name].drift_rules


def test_invalid_config_is_rejected(tmp_path):
    dst = tmp_path / "configs"
    for src in (REPO_ROOT / "configs").rglob("*.yaml"):
        target = dst / src.relative_to(REPO_ROOT / "configs")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
    path = dst / "monitor.yaml"
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    del data["thresholds"]["math"]
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    with pytest.raises(ValueError, match="thresholds missing"):
        load_config(dst)


# --------------------------------------------------------------------- formatters
def test_sql_formatter_validates_and_flags_aggregation_as_drift():
    f = get_formatter("sql")
    ex = f.format({"question": "Names?", "context": SCHEMA, "answer": "SELECT name FROM head"})
    assert ex.input == f"{SCHEMA}\n\nQuestion: Names?" and not f.is_drift(ex, "aggregation", {})
    agg = f.format({"question": "q", "context": SCHEMA, "answer": "SELECT COUNT(*) FROM head"})
    assert f.is_drift(agg, "aggregation", {})
    with pytest.raises(Rejected, match="sql_does_not_execute"):
        f.format({"question": "q", "context": SCHEMA, "answer": "SELECT salary FROM head"})


@pytest.mark.parametrize(
    ("schema", "query"),
    [
        ("ATTACH DATABASE 'x.db' AS x", "SELECT 1"),
        (SCHEMA, "SELECT 1; DROP TABLE head"),
        (f"{SCHEMA}; PRAGMA writable_schema=1", "SELECT name FROM head"),
    ],
)
def test_untrusted_sql_never_runs_anything_but_select(schema, query):
    assert sql_executes(schema, query) is False


def test_summarize_and_math_formatters():
    s = get_formatter("summarize")
    dialogue = "#Person1#: Hi, how are you?\n#Person2#: Fine thanks.\n#Person1#: Good."
    ex = s.format({"dialogue": dialogue, "summary": "They greet.", "topic": "greeting"})
    assert ex.meta == {"topic": "greeting", "turns": 3, "speakers": 2}
    assert s.is_drift(ex, "long_dialogue", {"min_turns": 3})

    m = get_formatter("math")
    ex = m.format({"question": "q", "answer": "12 x 2 = <<12*2=24>>24\n#### 1,024"})
    assert ex.target == "12 x 2 = 24\nFinal answer: 1024"  # calculator tags stripped
    with pytest.raises(Rejected, match="no_final_answer"):
        m.format({"question": "q", "answer": "no marker"})


# ----------------------------------------------------------------------- pipeline
def test_splits_are_disjoint_deterministic_and_separate_drift(small_sql_cfg):
    splits, _ = build_splits(small_sql_cfg, sql_rows(200))
    again, _ = build_splits(small_sql_cfg, sql_rows(200))
    assert {k: len(v) for k, v in splits.items()} == {
        "train": 20, "eval": 5, "golden": 5, "drift_train": 6, "drift_eval": 4
    }  # fmt: skip
    ids = [e.id for split in splits.values() for e in split]
    assert len(ids) == len(set(ids))
    assert [e.id for e in splits["train"]] == [e.id for e in again["train"]]
    assert all("COUNT" in e.target for e in splits["drift_train"] + splits["drift_eval"])
    assert not any("COUNT" in e.target for e in splits["train"])


def test_dataset_version_is_content_hash_and_tampering_is_detected(small_sql_cfg, tmp_path):
    splits, stats = build_splits(small_sql_cfg, sql_rows(200))
    m1 = write_dataset(small_sql_cfg, splits, stats, tmp_path / "a")
    m2 = write_dataset(small_sql_cfg, splits, stats, tmp_path / "b")
    assert m1["dataset_version"] == m2["dataset_version"]
    assert len(load_split(tmp_path / "a", "train")) == 20
    with (tmp_path / "a" / "train.jsonl").open("a", encoding="utf-8") as f:
        f.write('{"id": "x", "task": "sql", "input": "i", "target": "t", "meta": {}}\n')
    with pytest.raises(ValueError, match="does not match its manifest"):
        verify_dataset(tmp_path / "a")
