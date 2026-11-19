"""AdaptOS gateway.

Run with:  uvicorn adaptos.serving.app:create_app --factory --port 8000

    POST /chat              route a request to the right adapter and answer it
    POST /feedback          thumbs up/down (+ optional correction) for an answer
    GET  /models            live router/adapter versions
    GET  /monitoring        rolling-window quality, drift and alert status per adapter
    POST /admin/reload      hot-swap to the registry's current champions   (X-Admin-Token)
    POST /admin/rollback    move a model back to its previous version      (X-Admin-Token)
    GET  /health /ready /info /metrics
"""

from __future__ import annotations

import json
import secrets
import threading
import time
import urllib.error
import urllib.request
import uuid
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import Depends, FastAPI, Header, HTTPException, Request, Response
from fastapi.responses import JSONResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel, Field

from adaptos import __version__
from adaptos.config import Settings, get_settings, load_config
from adaptos.log import configure_logging, get_logger
from adaptos.monitoring.metrics import (
    ADAPTER_ALERT,
    DRIFT_THRESHOLD,
    FEEDBACK,
    GENERATION_LATENCY,
    HTTP_LATENCY,
    HTTP_REQUESTS,
    OUTPUT_TOKENS,
    QUALITY_SCORE,
    QUALITY_THRESHOLD,
    ROUTED_REQUESTS,
    ROUTER_CONFIDENCE,
    WINDOW_DRIFT,
    WINDOW_DRIFT_FEATURE,
    WINDOW_QUALITY,
    WINDOW_SAMPLES,
)
from adaptos.monitoring.monitor import Monitor, WindowStats
from adaptos.monitoring.quality import score_online
from adaptos.serving.models import BASE, ROUTER, ModelManager
from adaptos.serving.store import RequestStore

logger = get_logger(__name__)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1)
    adapter: str | None = Field(
        default=None, description="force an adapter (task name) or 'base'; default: router"
    )
    max_new_tokens: int | None = Field(default=None, gt=0, le=1024)


class RouteInfo(BaseModel):
    label: str
    confidence: float
    reason: str
    router_version: str | None


class ChatResponse(BaseModel):
    request_id: str
    answer: str
    adapter: str
    adapter_version: str | None
    route: RouteInfo
    quality: float | None = Field(description="reference-free quality score (None for base)")
    latency_ms: float


class FeedbackRequest(BaseModel):
    request_id: str
    rating: Literal["up", "down"]
    correction: str | None = Field(default=None, description="optional better answer")


class ModelRequest(BaseModel):
    model: str | None = Field(default=None, description="'router', a task name, or all")


def _mlflow_reachable(tracking_uri: str, timeout: float = 2.0) -> bool:
    try:
        with urllib.request.urlopen(f"{tracking_uri.rstrip('/')}/health", timeout=timeout) as r:
            return r.status == 200
    except (urllib.error.URLError, TimeoutError, ValueError):
        return False


def publish_window_metrics(stats: dict[str, WindowStats]) -> None:
    for adapter, s in stats.items():
        WINDOW_SAMPLES.labels(adapter).set(s.samples)
        if s.quality is not None:
            WINDOW_QUALITY.labels(adapter).set(s.quality)
        if s.drift is not None:
            WINDOW_DRIFT.labels(adapter).set(s.drift)
        for feature, shift in s.drift_features.items():
            WINDOW_DRIFT_FEATURE.labels(adapter, feature).set(shift)
        ADAPTER_ALERT.labels(adapter).set(1 if s.status == "alert" else 0)


def create_app(
    settings: Settings | None = None,
    manager: ModelManager | None = None,
    store: RequestStore | None = None,
    load_models: bool = True,
    monitor_loop: bool = True,
) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(settings.log_level)
    config = load_config(settings.config_dir)
    manager = manager or ModelManager(config, settings)
    store = store or RequestStore(settings.database_url)
    known_models = {ROUTER, *config.adapters}
    monitor = Monitor(
        config.monitor,
        recent=lambda adapter, version, limit: store.recent(adapter, version, limit),
        live=lambda: {m: v for m, v in manager.live.items() if m != ROUTER},
        reference=lambda adapter: manager.references.get(adapter),
    )
    for adapter, t in config.monitor.thresholds.items():
        QUALITY_THRESHOLD.labels(adapter).set(t.quality_min)
        DRIFT_THRESHOLD.labels(adapter).set(t.drift_max)
    stop = threading.Event()

    def monitor_forever() -> None:
        while not stop.wait(config.monitor.check_interval_seconds):
            if manager.ready:
                try:
                    publish_window_metrics(monitor.check())
                except Exception:
                    logger.exception("monitor check failed")

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        try:
            store.init_schema()
        except Exception:
            logger.exception("request store unavailable; requests will not be logged")
        if load_models:  # loads in the background; /ready reports progress
            threading.Thread(target=manager.load, name="model-loader", daemon=True).start()
        if monitor_loop:
            threading.Thread(target=monitor_forever, name="monitor", daemon=True).start()
        yield
        stop.set()

    app = FastAPI(title="AdaptOS Gateway", version=__version__, lifespan=lifespan)
    app.state.settings, app.state.config = settings, config
    app.state.manager, app.state.store = manager, store

    def require_admin(x_admin_token: str = Header(default="")) -> None:
        if not settings.admin_token:
            raise HTTPException(403, "admin API disabled: set ADAPTOS_ADMIN_TOKEN")
        if not secrets.compare_digest(x_admin_token, settings.admin_token):
            raise HTTPException(401, "invalid admin token")

    def require_ready() -> None:
        if not manager.ready:
            raise HTTPException(503, manager.error or "models are still loading")

    @app.middleware("http")
    async def record_metrics(request: Request, call_next):
        start = time.perf_counter()
        status = 500
        try:
            response = await call_next(request)
            status = response.status_code
            return response
        finally:
            route = request.scope.get("route")
            # Use the route template (not the raw URL) to keep label cardinality bounded.
            path = getattr(route, "path", "unmatched")
            if path != "/metrics":
                HTTP_REQUESTS.labels(request.method, path, str(status)).inc()
                HTTP_LATENCY.labels(request.method, path).observe(time.perf_counter() - start)

    # ----------------------------------------------------------------- serving
    @app.post("/chat", response_model=ChatResponse, dependencies=[Depends(require_ready)])
    def chat(req: ChatRequest) -> ChatResponse:
        if len(req.message) > config.serving.max_input_chars:
            raise HTTPException(413, f"message longer than {config.serving.max_input_chars} chars")
        if req.adapter is not None and req.adapter not in {BASE, *config.adapters}:
            raise HTTPException(422, f"unknown adapter '{req.adapter}'")
        start = time.perf_counter()
        result = manager.chat(req.message, adapter=req.adapter, max_new_tokens=req.max_new_tokens)
        latency_ms = (time.perf_counter() - start) * 1000
        request_id = str(uuid.uuid4())
        d = result.decision

        ROUTED_REQUESTS.labels(result.adapter, d.reason).inc()
        ROUTER_CONFIDENCE.labels(d.label).observe(d.confidence)
        GENERATION_LATENCY.labels(result.adapter).observe(result.generate_ms / 1000)
        OUTPUT_TOKENS.labels(result.adapter).inc(result.output_tokens)
        checks = score_online(d.adapter, req.message, result.answer)
        quality = checks["quality"] if checks else None
        if quality is not None:
            QUALITY_SCORE.labels(result.adapter).observe(quality)
        try:
            store.log_request(
                request_id=request_id,
                input=req.message,
                output=result.answer,
                adapter=result.adapter,
                adapter_version=result.adapter_version,
                router_version=result.router_version,
                route_label=d.label,
                route_confidence=d.confidence,
                route_reason=d.reason,
                latency_ms=latency_ms,
                output_tokens=result.output_tokens,
                quality_score=quality,
                quality_checks=json.dumps(checks) if checks else None,
            )
        except Exception:
            logger.exception("could not log request %s", request_id)

        return ChatResponse(
            request_id=request_id,
            answer=result.answer,
            adapter=result.adapter,
            adapter_version=result.adapter_version,
            route=RouteInfo(
                label=d.label,
                confidence=round(d.confidence, 4),
                reason=d.reason,
                router_version=result.router_version,
            ),
            quality=None if quality is None else round(quality, 3),
            latency_ms=round(latency_ms, 1),
        )

    @app.post("/feedback")
    def feedback(req: FeedbackRequest) -> dict[str, str]:
        row = store.get(req.request_id)
        if row is None or not store.add_feedback(req.request_id, req.rating, req.correction):
            raise HTTPException(404, "unknown request_id")
        FEEDBACK.labels(row["adapter"], req.rating).inc()
        return {"status": "recorded"}

    @app.get("/monitoring")
    def monitoring() -> dict:
        """Rolling-window quality, drift and alert status of each live adapter."""
        stats = monitor.check()
        publish_window_metrics(stats)
        return {
            "window_size": config.monitor.window_size,
            "min_samples": config.monitor.min_samples,
            "adapters": {
                a: {**s.to_dict(), "thresholds": config.monitor.thresholds[a].model_dump()}
                for a, s in stats.items()
            },
        }

    @app.get("/models")
    def models() -> dict:
        return {
            "ready": manager.ready,
            "error": manager.error,
            "live": manager.live,
            "router_threshold": config.router.confidence_threshold,
        }

    # ------------------------------------------------------------------- admin
    @app.post("/admin/reload", dependencies=[Depends(require_admin), Depends(require_ready)])
    def admin_reload(req: ModelRequest) -> dict:
        if req.model is not None and req.model not in known_models:
            raise HTTPException(422, f"unknown model '{req.model}'")
        return {"changes": manager.reload(req.model), "live": manager.live}

    @app.post("/admin/rollback", dependencies=[Depends(require_admin), Depends(require_ready)])
    def admin_rollback(req: ModelRequest) -> dict:
        if req.model not in known_models:
            raise HTTPException(422, f"model must be one of {sorted(known_models)}")
        try:
            changes = manager.rollback(req.model)
        except LookupError as e:
            raise HTTPException(409, str(e)) from None
        return {"changes": changes, "live": manager.live}

    # ------------------------------------------------------------------ status
    @app.get("/health")
    def health() -> dict[str, str]:
        """Liveness: the process is up."""
        return {"status": "ok"}

    @app.get("/ready")
    def ready() -> JSONResponse:
        """Readiness: models loaded and dependencies reachable."""
        checks = {
            "models": manager.ready,
            "mlflow": _mlflow_reachable(settings.mlflow_tracking_uri),
            "database": store.ping(),
        }
        ok = all(checks.values())
        body = {"status": "ready" if ok else "degraded", "checks": checks}
        if manager.error:
            body["error"] = manager.error
        return JSONResponse(status_code=200 if ok else 503, content=body)

    @app.get("/info")
    def info() -> dict:
        return {
            "version": __version__,
            "base_model": config.base_model.name,
            "backends": config.backends.model_dump(),
            "adapters": {
                name: {"description": a.description, "registered_model": a.registered_model}
                for name, a in config.adapters.items()
            },
        }

    @app.get("/metrics")
    def metrics() -> Response:
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    return app
