import pytest

from adaptos.data.formatters import (
    Rejected,
    get_formatter,
    parse_gsm8k_answer,
    sql_executes,
)

SCHEMA = "CREATE TABLE head (age INTEGER, name VARCHAR)"


class TestSql:
    f = get_formatter("sql")

    def test_formats_valid_row(self):
        ex = self.f.format(
            {
                "question": "Names older than 56?",
                "context": SCHEMA,
                "answer": "SELECT name FROM head WHERE age > 56",
            }
        )
        assert ex.task == "sql"
        assert ex.input.startswith(SCHEMA)
        assert "Question: Names older than 56?" in ex.input
        assert ex.meta["schema"] == SCHEMA

    def test_rejects_sql_that_does_not_run(self):
        with pytest.raises(Rejected, match="sql_does_not_execute"):
            self.f.format({"question": "q", "context": SCHEMA, "answer": "SELECT salary FROM head"})

    def test_rejects_empty(self):
        with pytest.raises(Rejected, match="empty_field"):
            self.f.format({"question": "", "context": SCHEMA, "answer": "SELECT 1"})

    @pytest.mark.parametrize(
        ("query", "drift"),
        [
            ("SELECT COUNT(*) FROM head", True),
            ("SELECT name, max(age) FROM head GROUP BY name", True),
            ("SELECT name FROM head WHERE age > 3", False),
        ],
    )
    def test_aggregation_drift_rule(self, query, drift):
        ex = self.f.format({"question": "q", "context": SCHEMA, "answer": query})
        assert self.f.is_drift(ex, "aggregation", {}) is drift

    def test_unknown_drift_rule(self):
        ex = self.f.format({"question": "q", "context": SCHEMA, "answer": "SELECT name FROM head"})
        with pytest.raises(ValueError, match="unknown drift rule"):
            self.f.is_drift(ex, "nope", {})


@pytest.mark.parametrize(
    ("schema", "query"),
    [
        ("ATTACH DATABASE 'x.db' AS x", "SELECT 1"),
        (SCHEMA, "DROP TABLE head"),
        (SCHEMA, "SELECT 1; DROP TABLE head"),
        (f"{SCHEMA}; PRAGMA writable_schema=1", "SELECT name FROM head"),
    ],
)
def test_sql_executes_refuses_non_select_statements(schema, query):
    assert sql_executes(schema, query) is False


class TestSummarize:
    f = get_formatter("summarize")
    dialogue = "#Person1#: Hi, how are you?\n#Person2#: Fine thanks, and you?\n#Person1#: Good."

    def test_formats_and_counts_turns(self):
        ex = self.f.format(
            {"dialogue": self.dialogue, "summary": "They greet.", "topic": "greeting"}
        )
        assert ex.input.startswith("Summarize this conversation:")
        assert ex.meta == {"topic": "greeting", "turns": 3, "speakers": 2}

    def test_rejects_summary_longer_than_dialogue(self):
        with pytest.raises(Rejected, match="summary_not_shorter"):
            self.f.format({"dialogue": "#Person1#: Hi", "summary": "A very long summary indeed."})

    def test_long_dialogue_drift_rule(self):
        ex = self.f.format({"dialogue": self.dialogue, "summary": "They greet."})
        assert self.f.is_drift(ex, "long_dialogue", {"min_turns": 3}) is True
        assert self.f.is_drift(ex, "long_dialogue", {"min_turns": 4}) is False


class TestMath:
    f = get_formatter("math")

    def test_strips_calculator_annotations(self):
        reasoning, final = parse_gsm8k_answer("She read 12 x 2 = <<12*2=24>>24 pages.\n#### 1,024")
        assert reasoning == "She read 12 x 2 = 24 pages."
        assert final == "1024"

    def test_formats_with_final_answer_line(self):
        ex = self.f.format({"question": "2+2?", "answer": "2 + 2 = 4\n#### 4"})
        assert ex.target == "2 + 2 = 4\nFinal answer: 4"
        assert ex.meta == {"final_answer": "4", "steps": 1}

    @pytest.mark.parametrize(
        ("answer", "reason"),
        [("no marker here", "no_final_answer"), ("x\n#### four", "final_answer_not_numeric")],
    )
    def test_rejects_bad_answers(self, answer, reason):
        with pytest.raises(Rejected, match=reason):
            self.f.format({"question": "q", "answer": answer})

    def test_multi_step_drift_rule(self):
        ex = self.f.format({"question": "q", "answer": "a\nb\nc\nd\ne\n#### 5"})
        assert self.f.is_drift(ex, "multi_step", {"min_steps": 5}) is True


def test_ids_ignore_case_and_whitespace():
    f = get_formatter("math")
    a = f.format({"question": "What is  2+2?", "answer": "4\n#### 4"})
    b = f.format({"question": "what is 2+2?", "answer": "4\n#### 4"})
    assert a.id == b.id


def test_unknown_task():
    with pytest.raises(ValueError, match="no formatter"):
        get_formatter("poetry")
