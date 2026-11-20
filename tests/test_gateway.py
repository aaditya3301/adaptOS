"""Gateway tests with a fake engine, fake router embeddings, fake registry and SQLite."""

from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient

from adaptos.config import REPO_ROOT, Settings, load_config
from adaptos.registry import adapters as registry
from adaptos.routing import RouterModel
from adaptos.serving.app import create_app
from adaptos.serving.models import ModelManager
from adaptos.serving.store import RequestStore

LABELS = ["math", "other", "sql", "summarize"]


def keyword_embed(texts):
    """One-hot 'embedding' by keyword, matching LABELS order."""
    out = np.zeros((len(texts), 4))
    for i, t in enumerate(texts):
        if "CREATE TABLE" in t:
            out[i, 2] = 1
        elif "#Person1#" in t:
            out[i, 3] = 1
        elif any(c.isdigit() for c in t):
            out[i, 0] = 1
        else:
            out[i, 1] = 1
    return out


class FakeEngine:
    device = "cpu"

    def __init__(self):
        self.adapters: dict[str, str] = {}

    def load_adapter(self, name, path):
        self.adapters[name] = str(path)

    def unload_adapter(self, name):
        self.adapters.pop(name, None)

    def generate(self, inputs, adapter, prompt_task, max_new_tokens, batch_size=1):
        if adapter is not None and adapter not in self.adapters:
            raise KeyError(adapter)
        return [f"[{adapter or 'base'}|{prompt_task}|{max_new_tokens}] answer" for _ in inputs]

    def count_tokens(self, text):
        return len(text.split())


class FakeRegistry:
    def __init__(self, root):
        self.root = root
        self.champion: dict[str, str] = {}
        self.previous: dict[str, str] = {}
        self.downloads: list[tuple[str, str]] = []

    def get_alias_version(self, name, alias, uri=None):
        v = (self.champion if alias == "champion" else self.previous).get(name)
        return SimpleNamespace(version=v) if v else None

    def cached_download(self, cache_root, name, version, uri):
        self.downloads.append((name, version))
        path = self.root / name / f"v{version}"
        path.mkdir(parents=True, exist_ok=True)
        if name == "adaptos-router":
            RouterModel(LABELS, np.eye(4) * 10, np.zeros(4), "m", "r", 0.6,
                        embed_fn=keyword_embed).save(path)  # fmt: skip
        return path

    def rollback(self, name, uri=None):
        if name not in self.previous:
            raise LookupError("no previous version")
        self.champion[name], self.previous[name] = self.previous[name], self.champion[name]


@pytest.fixture
def env(tmp_path, monkeypatch):
    fake = FakeRegistry(tmp_path / "registry")
    for fn in ("get_alias_version", "cached_download", "rollback"):
        monkeypatch.setattr(registry, fn, getattr(fake, fn))
    fake.champion = {
        "adaptos-router": "1",
        "adaptos-adapter-sql": "1",
        "adaptos-adapter-summarize": "1",
        "adaptos-adapter-math": "1",
    }
    settings = Settings(
        config_dir=REPO_ROOT / "configs",
        artifacts_dir=tmp_path / "artifacts",
        mlflow_tracking_uri="http://127.0.0.1:9",
        database_url=f"sqlite:///{(tmp_path / 'requests.db').as_posix()}",
        admin_token="secret",
    )
    app_cfg = load_config(settings.config_dir)
    engine = FakeEngine()
    manager = ModelManager(
        app_cfg, settings, engine_factory=lambda: engine, embed_factory=lambda m, r: keyword_embed
    )
    store = RequestStore(settings.database_url)
    app = create_app(settings, manager=manager, store=store, load_models=False, monitor_loop=False)
    with TestClient(app) as client:
        yield SimpleNamespace(
            client=client, manager=manager, engine=engine, registry=fake, store=store,
            settings=settings,
        )  # fmt: skip


@pytest.fixture
def ready(env):
    env.manager.load()
    assert env.manager.ready, env.manager.error
    return env


SQL = "CREATE TABLE t (a INT)\n\nQuestion: list a"
ADMIN = {"X-Admin-Token": "secret"}


# ------------------------------------------------------------------------ status
def test_health(env):
    assert env.client.get("/health").json() == {"status": "ok"}


def test_chat_is_unavailable_until_models_load(env):
    r = env.client.post("/chat", json={"message": SQL})
    assert r.status_code == 503 and "loading" in r.json()["detail"]


def test_ready_reports_each_dependency(ready):
    body = ready.client.get("/ready").json()
    assert body["checks"] == {"models": True, "mlflow": False, "database": True}


# -------------------------------------------------------------------------- chat
def test_sql_request_is_routed_to_sql_adapter_and_logged(ready):
    r = ready.client.post("/chat", json={"message": SQL})
    assert r.status_code == 200
    body = r.json()
    assert body["adapter"] == "sql" and body["adapter_version"] == "1"
    assert body["route"] == {"label": "sql", "confidence": pytest.approx(1.0, abs=1e-3),
                             "reason": "adapter", "router_version": "1"}  # fmt: skip
    assert body["answer"].startswith("[sql@v1|sql|96]")  # adapter, prompt, sql max_new_tokens
    row = ready.store.get(body["request_id"])
    assert row["adapter"] == "sql" and row["route_reason"] == "adapter" and row["output"]


def test_general_request_goes_to_base_model(ready):
    body = ready.client.post("/chat", json={"message": "Tell me a joke"}).json()
    assert body["adapter"] == "base" and body["adapter_version"] is None
    assert body["route"]["reason"] == "general_request"
    assert body["answer"].startswith("[base|None|256]")


def test_override_adapter(ready):
    body = ready.client.post("/chat", json={"message": SQL, "adapter": "base"}).json()
    assert body["adapter"] == "base" and body["route"]["reason"] == "override"
    body = ready.client.post("/chat", json={"message": "hi", "adapter": "math"}).json()
    assert body["adapter"] == "math"


@pytest.mark.parametrize(
    ("payload", "status"),
    [
        ({"message": SQL, "adapter": "poetry"}, 422),
        ({"message": ""}, 422),
        ({"message": "x" * 9000}, 413),
        ({"message": SQL, "max_new_tokens": 5000}, 422),
    ],
)
def test_invalid_chat_requests(ready, payload, status):
    assert ready.client.post("/chat", json=payload).status_code == status


def test_missing_champion_falls_back_to_base(env):
    del env.registry.champion["adaptos-adapter-sql"]
    env.manager.load()
    body = env.client.post("/chat", json={"message": SQL}).json()
    assert body["adapter"] == "base" and body["route"]["reason"] == "adapter_unavailable"


# ---------------------------------------------------------------------- feedback
def test_feedback_is_stored(ready):
    rid = ready.client.post("/chat", json={"message": SQL}).json()["request_id"]
    r = ready.client.post(
        "/feedback", json={"request_id": rid, "rating": "down", "correction": "SELECT a FROM t"}
    )
    assert r.status_code == 200
    row = ready.store.get(rid)
    assert row["feedback"] == "down" and row["correction"] == "SELECT a FROM t"


def test_feedback_for_unknown_request(ready):
    r = ready.client.post("/feedback", json={"request_id": "nope", "rating": "up"})
    assert r.status_code == 404


# ------------------------------------------------------------------------- admin
def test_admin_requires_token(ready):
    assert ready.client.post("/admin/reload", json={}).status_code == 401
    assert (
        ready.client.post("/admin/reload", json={}, headers={"X-Admin-Token": "x"}).status_code
        == 401
    )


def test_admin_disabled_without_configured_token(ready):
    ready.settings.admin_token = ""
    assert ready.client.post("/admin/reload", json={}, headers=ADMIN).status_code == 403


def test_reload_hot_swaps_to_new_champion(ready):
    ready.registry.champion["adaptos-adapter-sql"] = "2"
    r = ready.client.post("/admin/reload", json={}, headers=ADMIN)
    assert r.json()["changes"] == [{"model": "sql", "from": "1", "to": "2"}]
    assert "sql@v2" in ready.engine.adapters and "sql@v1" not in ready.engine.adapters
    body = ready.client.post("/chat", json={"message": SQL}).json()
    assert body["adapter_version"] == "2" and body["answer"].startswith("[sql@v2")
    # nothing changed -> nothing to do
    assert ready.client.post("/admin/reload", json={}, headers=ADMIN).json()["changes"] == []


def test_rollback_returns_to_previous_version(ready):
    ready.registry.champion["adaptos-adapter-sql"] = "2"
    ready.registry.previous["adaptos-adapter-sql"] = "1"
    ready.client.post("/admin/reload", json={"model": "sql"}, headers=ADMIN)
    r = ready.client.post("/admin/rollback", json={"model": "sql"}, headers=ADMIN)
    assert r.status_code == 200
    assert r.json()["live"]["sql"] == "1"
    assert ready.client.post("/chat", json={"message": SQL}).json()["adapter_version"] == "1"


def test_rollback_without_previous_version(ready):
    r = ready.client.post("/admin/rollback", json={"model": "math"}, headers=ADMIN)
    assert r.status_code == 409


def test_unknown_model_in_admin(ready):
    assert (
        ready.client.post("/admin/rollback", json={"model": "x"}, headers=ADMIN).status_code == 422
    )


# ----------------------------------------------------------------------- metrics
def test_metrics(ready):
    ready.client.post("/chat", json={"message": SQL})
    text = ready.client.get("/metrics").text
    assert 'adaptos_routed_requests_total{adapter="sql",reason="adapter"}' in text
    assert 'adaptos_live_model_version{model="sql"}' in text
    assert 'adaptos_http_requests_total{method="POST",path="/chat",status="200"}' in text


# -------------------------------------------------------------------- monitoring
def test_chat_returns_and_stores_quality(ready):
    body = ready.client.post("/chat", json={"message": SQL}).json()
    # FakeEngine answers are not SQL -> the online check scores them 0.0
    assert body["quality"] == 0.0
    row = ready.store.get(body["request_id"])
    assert row["quality_score"] == 0.0 and '"valid_sql": false' in row["quality_checks"]
    assert ready.client.post("/chat", json={"message": "hi"}).json()["quality"] is None


def test_monitoring_endpoint(ready):
    for _ in range(3):
        ready.client.post("/chat", json={"message": SQL})
    body = ready.client.get("/monitoring").json()
    sql = body["adapters"]["sql"]
    assert sql["version"] == "1" and sql["samples"] == 3
    assert sql["status"] in ("insufficient_data", "no_reference")
    assert sql["thresholds"] == {"quality_min": 0.8, "drift_max": 1.0}
    assert "adaptos_window_samples" in ready.client.get("/metrics").text


def test_redeployed_version_is_judged_only_on_new_traffic(ready):
    for _ in range(3):
        ready.client.post("/chat", json={"message": SQL})
    assert ready.client.get("/monitoring").json()["adapters"]["sql"]["samples"] == 3
    # v2 goes live, then we roll back to v1: v1's earlier requests no longer count
    ready.registry.champion["adaptos-adapter-sql"] = "2"
    ready.registry.previous["adaptos-adapter-sql"] = "1"
    ready.client.post("/admin/reload", json={"model": "sql"}, headers=ADMIN)
    ready.client.post("/admin/rollback", json={"model": "sql"}, headers=ADMIN)
    sql = ready.client.get("/monitoring").json()["adapters"]["sql"]
    assert sql["version"] == "1" and sql["samples"] == 0
    ready.client.post("/chat", json={"message": SQL})
    assert ready.client.get("/monitoring").json()["adapters"]["sql"]["samples"] == 1


def test_forced_reload_restarts_the_window_of_an_unchanged_version(ready):
    for _ in range(2):
        ready.client.post("/chat", json={"message": SQL})
    assert (
        ready.client.post("/admin/reload", json={"model": "sql"}, headers=ADMIN).json()["changes"]
        == []
    )
    r = ready.client.post("/admin/reload", json={"model": "sql", "force": True}, headers=ADMIN)
    assert r.json()["changes"] == [{"model": "sql", "from": "1", "to": "1", "window_reset": True}]
    assert "sql@v1" in ready.engine.adapters  # weights untouched
    assert ready.client.get("/monitoring").json()["adapters"]["sql"]["samples"] == 0
