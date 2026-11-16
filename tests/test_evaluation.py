import pytest

from adaptos.config import EvaluationConfig
from adaptos.data.schema import Example
from adaptos.evaluation.evaluate import score_predictions
from adaptos.evaluation.gate import decide
from adaptos.evaluation.metrics import (
    execution_match,
    extract_answer,
    extract_sql,
    rouge_l,
    score_math,
    score_sql,
)

SCHEMA = "CREATE TABLE players (name VARCHAR, team VARCHAR, goals INTEGER)"
GOLD = 'SELECT name FROM players WHERE team = "rovers" AND goals > 10'


# ------------------------------------------------------------------ SQL execution match
@pytest.mark.parametrize(
    "pred",
    [
        GOLD,
        'SELECT name FROM players WHERE goals > 10 AND team = "rovers"',  # reordered conditions
        "select name from players where team = 'rovers' and goals >= 11",  # equivalent filter
    ],
)
def test_equivalent_sql_matches(pred):
    assert execution_match(SCHEMA, GOLD, pred) == (True, True)


@pytest.mark.parametrize(
    "pred",
    [
        'SELECT team FROM players WHERE team = "rovers" AND goals > 10',  # wrong column
        'SELECT name FROM players WHERE team = "united" AND goals > 10',  # wrong value
        'SELECT name FROM players WHERE team = "rovers"',  # missing condition
    ],
)
def test_wrong_sql_does_not_match(pred):
    match, valid = execution_match(SCHEMA, GOLD, pred)
    assert match is False and valid is True


@pytest.mark.parametrize("pred", ["SELECT nope FROM players", "DROP TABLE players", "hello"])
def test_invalid_or_unsafe_sql(pred):
    assert execution_match(SCHEMA, GOLD, pred) == (False, False)


def test_extract_sql_from_chatty_reply():
    reply = "Here you go:\n```sql\nSELECT name FROM players;\n```\nHope it helps"
    assert extract_sql(reply) == "SELECT name FROM players"


def test_score_sql_keys():
    ex = Example(id="1", task="sql", input="", target=GOLD, meta={"schema": SCHEMA})
    assert score_sql(ex, GOLD) == {"execution_match": 1.0, "exact_match": 1.0, "valid_sql": 1.0}


# ------------------------------------------------------------------------- ROUGE-L / math
def test_rouge_l():
    assert rouge_l("the cat sat", "the cat sat") == pytest.approx(1.0)
    assert rouge_l("dog runs", "the cat sat") == 0.0
    # LCS("a b c d", "a c d e") = 3 -> P = R = 3/4
    assert rouge_l("a b c d", "a c d e") == pytest.approx(0.75)
    assert rouge_l("#Person1# leaves", "#person1# leaves early") > 0.75


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("2 + 2 = 4\nFinal answer: 4", ("4", True)),
        ("so it costs $1,250.\nFinal Answer: $1,250", ("1250", True)),
        ("the total is 18 apples", ("18", False)),
        ("no numbers here", (None, False)),
    ],
)
def test_extract_answer(text, expected):
    assert extract_answer(text) == expected


def test_score_math():
    ex = Example(id="1", task="math", input="", target="", meta={"final_answer": "190"})
    assert score_math(ex, "47.5 * 4 = 190\nFinal answer: 190")["answer_accuracy"] == 1.0
    assert score_math(ex, "Final answer: 185")["answer_accuracy"] == 0.0


def test_score_predictions_rows():
    ex = Example(id="m1", task="math", input="q", target="t", meta={"final_answer": "3"})
    agg, rows = score_predictions([ex, ex], ["Final answer: 3", "Final answer: 4"])
    assert agg["answer_accuracy"] == 0.5
    assert rows[0]["prediction"] == "Final answer: 3" and rows[1]["answer_accuracy"] == 0.0


# ------------------------------------------------------------------------------- gate
CFG = EvaluationConfig(
    primary_metric="m", min_score=0.3, promotion_min_delta=0.01, max_regression=0.02
)


@pytest.mark.parametrize(
    ("candidate", "champion", "promote", "reason"),
    [
        ({"golden": 0.2, "drift_eval": 0.9}, None, False, "below the minimum"),
        ({"golden": 0.5, "drift_eval": 0.1}, None, True, "first version"),
        ({"golden": 0.60, "drift_eval": 0.3}, {"golden": 0.50, "drift_eval": 0.3}, True, "golden"),
        # drift retraining: golden flat, drift much better -> promote
        ({"golden": 0.49, "drift_eval": 0.6}, {"golden": 0.50, "drift_eval": 0.3}, True, "drift"),
        # drift better but golden regressed too much -> reject
        (
            {"golden": 0.45, "drift_eval": 0.6},
            {"golden": 0.50, "drift_eval": 0.3},
            False,
            "regressed",
        ),
        (
            {"golden": 0.505, "drift_eval": 0.3},
            {"golden": 0.50, "drift_eval": 0.3},
            False,
            "no m improvement",
        ),
    ],
)
def test_gate(candidate, champion, promote, reason):
    d = decide(CFG, candidate, champion)
    assert d.promote is promote
    assert reason in d.reason


# --------------------------------------------------------------- promote / rollback
class _Version:
    def __init__(self, version):
        self.version = version


class FakeRegistryClient:
    def __init__(self):
        self.aliases: dict[str, str] = {}
        self.tags: dict[tuple[str, str], str] = {}

    def get_model_version_by_alias(self, name, alias):
        from mlflow.exceptions import MlflowException

        if alias not in self.aliases:
            raise MlflowException("alias not found")
        return _Version(self.aliases[alias])

    def set_registered_model_alias(self, name, alias, version):
        self.aliases[alias] = version

    def set_model_version_tag(self, name, version, key, value):
        self.tags[(version, key)] = value


@pytest.fixture
def fake_registry(monkeypatch):
    from adaptos.registry import adapters

    client = FakeRegistryClient()
    monkeypatch.setattr(adapters, "_client", lambda uri=None: client)
    return client


def test_promote_keeps_previous_for_rollback(fake_registry):
    from adaptos.registry import adapters as reg

    assert reg.promote("m", "1") is None
    assert reg.promote("m", "2") == "1"
    assert fake_registry.aliases == {"champion": "2", "previous": "1"}

    assert reg.rollback("m") == ("1", "2")
    assert fake_registry.aliases == {"champion": "1", "previous": "2"}
    assert fake_registry.tags[("2", "rolled_back")] == "true"


def test_rollback_needs_a_previous_version(fake_registry):
    from adaptos.registry import adapters as reg

    reg.promote("m", "1")
    with pytest.raises(LookupError, match="previous"):
        reg.rollback("m")
