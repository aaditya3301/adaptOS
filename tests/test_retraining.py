from types import SimpleNamespace

import pytest
from tests.conftest import sql_rows, tiny_model_loader

from adaptos.config import REPO_ROOT, Settings, load_config
from adaptos.data.pipeline import build_splits, write_dataset
from adaptos.registry import adapters as registry
from adaptos.retraining import controller as ctl
from adaptos.retraining.pipeline import collect_corrections, valid_correction

SCHEMA = "CREATE TABLE t (id INTEGER, name VARCHAR, age INTEGER)"
SQL_INPUT = f"{SCHEMA}\n\nQuestion: how many rows?"


# ------------------------------------------------------------ user corrections
@pytest.mark.parametrize(
    ("task", "user_input", "target", "ok"),
    [
        ("sql", SQL_INPUT, "SELECT COUNT(*) FROM t", True),
        ("sql", SQL_INPUT, "SELECT nope FROM t", False),  # does not run on the schema
        ("sql", SQL_INPUT, "DROP TABLE t", False),
        ("math", "q", "3 + 4 = 7", True),
        ("math", "q", "no idea", False),
        ("summarize", "a long conversation " * 5, "short", True),
        ("summarize", "short", "a much longer summary than the input", False),
        ("sql", SQL_INPUT, "   ", False),
    ],
)
def test_valid_correction(task, user_input, target, ok):
    assert valid_correction(task, user_input, target) is ok


def test_collect_corrections_keeps_newest_valid_one():
    rows = [  # newest first
        {"request_id": "r3", "input": SQL_INPUT, "correction": "SELECT COUNT(*) FROM t"},
        {"request_id": "r2", "input": SQL_INPUT, "correction": "SELECT COUNT(id) FROM t"},
        {"request_id": "r1", "input": SQL_INPUT, "correction": "garbage"},
    ]
    examples = collect_corrections(rows, "sql")
    assert len(examples) == 1
    assert examples[0].target == "SELECT COUNT(*) FROM t"
    assert examples[0].meta == {"source": "feedback", "request_id": "r3"}


def test_math_corrections_get_a_final_answer_line():
    ex = collect_corrections([{"request_id": "r", "input": "q", "correction": "3 + 4 = 7"}], "math")
    assert ex[0].target.endswith("Final answer: 7")


# ------------------------------------------- training on several sources (rehearsal)
def test_train_adapter_mixes_new_data_replay_and_feedback(small_sql_cfg, app_config, tmp_settings):
    pytest.importorskip("peft")
    from adaptos.training.train_adapter import train_adapter

    splits, stats = build_splits(small_sql_cfg, sql_rows(200))
    write_dataset(small_sql_cfg, splits, stats, tmp_settings.data_dir / "processed" / "sql")
    app = app_config.model_copy(update={"adapters": {**app_config.adapters, "sql": small_sql_cfg}})
    feedback = collect_corrections(
        [{"request_id": "r", "input": SQL_INPUT, "correction": "SELECT COUNT(*) FROM t"}], "sql"
    )
    result = train_adapter(
        "sql", splits=["train", "drift_train"], split_limits={"train": 6}, extra_examples=feedback,
        extra_tags={"parent_run_id": "p1"}, epochs=1, max_steps=1, settings=tmp_settings, app=app,
        model_loader=tiny_model_loader, use_mlflow=False,
    )  # fmt: skip
    assert result.params["train_sources"] == "train:6,drift_train:6,feedback:1"
    assert result.params["n_train"] == 13
    assert result.params["epochs"] == 1


# ------------------------------------------------------------------ controller
class FakeClient:
    def __init__(self):
        self.runs: list[SimpleNamespace] = []

    def search_runs(self, experiment_ids, filter_string):
        task = filter_string.split("tags.task = '")[1].split("'")[0]
        since = int(filter_string.split("start_time > ")[1])
        return [r for r in self.runs if r.task == task and r.start_ms > since]


@pytest.fixture
def controller(monkeypatch, tmp_path):
    champions = {"adaptos-adapter-sql": "1"}
    monkeypatch.setattr(
        registry,
        "get_alias_version",
        lambda name, alias, uri=None: SimpleNamespace(version=champions[name])
        if name in champions else None,
    )  # fmt: skip
    calls = {"http": [], "commands": []}
    status = {"sql": {"status": "alert", "reasons": ["quality 0.7 < 0.8"], "quality": 0.7,
                      "drift": 1.5, "samples": 50},
              "math": {"status": "ok", "reasons": []}}  # fmt: skip

    def http(method, url, body=None, headers=None):
        calls["http"].append((method, url, body))
        if url.endswith("/monitoring"):
            return {"adapters": status}
        return {"changes": [{"model": "sql", "from": "1", "to": champions["adaptos-adapter-sql"]}]}

    def run_command(cmd, log_path):
        calls["commands"].append(cmd)
        champions["adaptos-adapter-sql"] = calls.get("next_version", "2")  # what the gate did
        return calls.get("exit_code", 0)

    c = ctl.Controller.__new__(ctl.Controller)
    c.app = load_config(REPO_ROOT / "configs")
    c.settings = Settings(artifacts_dir=tmp_path, mlflow_tracking_uri="http://x", admin_token="t")
    c.http, c.run_command, c.clock = http, run_command, lambda: 1_000_000.0
    c.ignore_cooldown = False
    c.client = FakeClient()
    c.experiment = SimpleNamespace(experiment_id="1")
    run = SimpleNamespace(info=SimpleNamespace(run_id="run123"))
    c.mlflow = SimpleNamespace(
        start_run=lambda **kw: SimpleNamespace(__enter__=lambda s: run, __exit__=lambda *a: None),
        set_tags=lambda *a: None, set_tag=lambda *a: None, log_metrics=lambda *a: None,
        log_metric=lambda *a: None, log_artifact=lambda *a: None,
    )  # fmt: skip

    class Ctx:
        def __enter__(self):
            return run

        def __exit__(self, *a):
            return False

    c.mlflow.start_run = lambda **kw: Ctx()
    return SimpleNamespace(c=c, calls=calls, champions=champions)


def test_alert_triggers_retraining_and_hot_swap(controller):
    actions = controller.c.check()
    swap = [{"model": "sql", "from": "1", "to": "2"}]
    assert actions == [
        {"task": "sql", "run_id": "run123", "exit_code": 0, "action": "promoted",
         "new_version": "2", "swap": swap}
    ]  # fmt: skip
    cmd = controller.calls["commands"][0]
    assert cmd[-4:] == ["--task", "sql", "--parent-run-id", "run123"]
    reload_call = controller.calls["http"][-1]
    assert reload_call[1].endswith("/admin/reload") and reload_call[2] == {"model": "sql"}


def test_rejected_candidate_is_not_swapped(controller):
    controller.calls["next_version"] = "1"  # gate kept the old champion
    actions = controller.c.check()
    assert actions[0]["action"] == "rejected"
    assert not any(u.endswith("/admin/reload") for _, u, _ in controller.calls["http"])


def test_failed_pipeline_raises(controller):
    controller.calls["exit_code"] = 1
    with pytest.raises(RuntimeError, match="failed"):
        controller.c.check()


def test_cooldown_and_daily_cap(controller):
    now_ms = 1_000_000_000
    controller.c.client.runs = [SimpleNamespace(task="sql", start_ms=now_ms - 60_000)]  # 1 min ago
    actions = controller.c.check()
    assert actions == [{"task": "sql", "action": "skipped",
                        "reason": "retrained less than 30 min ago"}]  # fmt: skip
    controller.c.client.runs = [
        SimpleNamespace(task="sql", start_ms=now_ms - h * 3_600_000) for h in (2, 3, 4)
    ]
    assert "runs in 24h" in controller.c.check()[0]["reason"]
    assert controller.calls["commands"] == []


def test_ignore_cooldown_still_respects_daily_cap(controller):
    now_ms = 1_000_000_000
    controller.c.ignore_cooldown = True
    controller.c.client.runs = [SimpleNamespace(task="sql", start_ms=now_ms - 60_000)]
    assert controller.c.check()[0]["action"] == "promoted"  # cooldown skipped
    controller.c.client.runs = [SimpleNamespace(task="sql", start_ms=now_ms - 60_000)] * 3
    assert "runs in 24h" in controller.c.check()[0]["reason"]
