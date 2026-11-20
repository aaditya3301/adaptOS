import pytest
from sqlalchemy import create_engine

from adaptos.config import AdapterThresholds
from adaptos.monitoring.drift import (
    build_reference,
    drift_score,
    features,
    load_reference,
    save_reference,
)
from adaptos.monitoring.monitor import effective_quality, evaluate_window
from adaptos.monitoring.quality import score_math, score_online, score_sql, score_summary
from adaptos.serving.store import RequestStore

SCHEMA = "CREATE TABLE players (name VARCHAR, team VARCHAR, goals INTEGER)"


# ---------------------------------------------------------------- online quality
@pytest.mark.parametrize(
    ("question", "sql", "quality"),
    [
        ("Who plays for rovers?", 'SELECT name FROM players WHERE team = "rovers"', 1.0),
        ("How many players are there?", "SELECT COUNT(*) FROM players", 1.0),
        ("How many players are there?", "SELECT name FROM players", 0.4),  # missed aggregate
        ("Who plays for rovers?", "SELECT nope FROM players", 0.0),  # invalid
        ("Who plays for rovers?", "DROP TABLE players", 0.0),  # unsafe
    ],
)
def test_sql_quality(question, sql, quality):
    assert score_sql(f"{SCHEMA}\n\nQuestion: {question}", sql)["quality"] == quality


DIALOGUE = (
    "Summarize this conversation:\n\n#Person1#: Did you finish the quarterly sales report?\n"
    "#Person2#: Not yet, the regional numbers arrived late this morning.\n"
    "#Person1#: Please send me a first draft by Friday afternoon then.\n"
    "#Person2#: Sure, I will have it ready before the weekly meeting."
)


def test_summary_quality():
    good = score_summary(DIALOGUE, "#Person2# will send #Person1# the report draft by Friday.")
    assert good["quality"] == pytest.approx(1.0) and good["speakers_ok"]
    too_short = score_summary(DIALOGUE, "#Person1# report.")
    assert too_short["quality"] < good["quality"]
    unknown_speaker = score_summary(DIALOGUE, "#Person3# will send the report draft by Friday.")
    assert not unknown_speaker["speakers_ok"]


def test_math_quality():
    right = score_math("q", "12 * 7 = 84\n84 - 9 = 75\nFinal answer: 75")
    assert right["quality"] == pytest.approx(1.0) and right["equations_correct"] == 2
    wrong = score_math("q", "50 * 0.95 = 47.5\n47.5 * 4 = 185\nFinal answer: 185")
    assert wrong["equations_correct"] == 1 and wrong["quality"] == pytest.approx(0.3 + 0.35)
    assert score_math("q", "I don't know")["quality"] == pytest.approx(0.35)


def test_base_model_answers_are_not_scored():
    assert score_online(None, "hi", "hello") is None


# ------------------------------------------------------------------------- drift
def _sql_inputs(questions):
    return [f"{SCHEMA}\n\nQuestion: {q}" for q in questions]


NORMAL = _sql_inputs([f"Who plays for team {i}?" for i in range(40)])
DRIFTED = _sql_inputs([f"How many goals did team {i} score in total?" for i in range(40)])


def test_features():
    f = features("sql", DRIFTED[0])
    assert f["asks_aggregate"] == 1.0 and f["schema_columns"] == 3
    assert features("summarize", DIALOGUE)["turns"] == 4
    assert features("math", "Tom has 3 apples and 4 pears")["numbers"] == 2


def test_drift_score_detects_new_kind_of_request():
    ref = build_reference("sql", NORMAL)
    normal, _ = drift_score("sql", NORMAL[:20], ref)
    drifted, shifts = drift_score("sql", DRIFTED[:20], ref)
    assert normal < 0.5 < drifted
    assert max(shifts, key=lambda k: abs(shifts[k])) in ("asks_aggregate", "question_words")


def test_reference_roundtrip(tmp_path):
    ref = build_reference("sql", NORMAL)
    save_reference(tmp_path, ref)
    assert load_reference(tmp_path) == ref
    assert load_reference(tmp_path / "missing") is None


# ------------------------------------------------------------------------ window
T = AdapterThresholds(quality_min=0.8, drift_max=1.0)


def _rows(inputs, quality, feedback=None):
    return [{"input": t, "quality_score": quality, "feedback": feedback} for t in inputs]


def test_feedback_overrides_quality():
    assert effective_quality({"quality_score": 0.9, "feedback": "down"}) == 0.0
    assert effective_quality({"quality_score": 0.2, "feedback": "up"}) == 1.0
    assert effective_quality({"quality_score": 0.7, "feedback": None}) == 0.7


def test_healthy_window():
    s = evaluate_window("sql", "1", _rows(NORMAL[:35], 0.95), build_reference("sql", NORMAL), T, 30)
    assert s.status == "ok" and s.reasons == []


def test_low_quality_alerts():
    s = evaluate_window("sql", "1", _rows(NORMAL[:35], 0.5), build_reference("sql", NORMAL), T, 30)
    assert s.status == "alert" and "quality" in s.reasons[0]


def test_drift_alerts_even_when_quality_looks_fine():
    s = evaluate_window(
        "sql", "1", _rows(DRIFTED[:35], 0.95), build_reference("sql", NORMAL), T, 30
    )
    assert s.status == "alert" and "drift" in s.reasons[0]


def test_too_few_requests_never_alert():
    s = evaluate_window("sql", "1", _rows(DRIFTED[:10], 0.1), build_reference("sql", NORMAL), T, 30)
    assert s.status == "insufficient_data"


def test_missing_reference_is_reported():
    s = evaluate_window("sql", "1", _rows(NORMAL[:35], 0.95), None, T, 30)
    assert s.status == "no_reference" and s.drift is None


# ------------------------------------------------------------------ store schema
def test_store_adds_new_columns_to_an_existing_table(tmp_path):
    url = f"sqlite:///{(tmp_path / 'old.db').as_posix()}"
    with create_engine(url).begin() as conn:  # a table created before quality_checks existed
        conn.exec_driver_sql(
            "CREATE TABLE requests (request_id VARCHAR(36) PRIMARY KEY, created_at TIMESTAMP, "
            "input TEXT, output TEXT, adapter VARCHAR(32))"
        )
    store = RequestStore(url)
    store.init_schema()
    store.log_request(
        request_id="r1", input="i", output="o", adapter="sql", adapter_version="2",
        quality_score=0.5, quality_checks="{}",
    )  # fmt: skip
    assert store.recent("sql", "2", 10)[0]["quality_checks"] == "{}"
    assert store.recent("sql", "1", 10) == []
