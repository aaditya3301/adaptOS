"""Live models of the gateway: the router and one champion adapter per task.

The manager follows the ``champion`` alias in the MLflow registry. ``reload`` hot-swaps any
model whose champion changed; ``rollback`` moves the alias back to ``previous`` and reloads.
A swap loads the new adapter next to the old one, switches traffic, then drops the old one,
so requests never hit a half-loaded model.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from adaptos.config import AppConfig, Settings
from adaptos.log import get_logger
from adaptos.monitoring.drift import REFERENCE_FILE, load_reference
from adaptos.monitoring.metrics import LIVE_VERSION, MODEL_SWAPS
from adaptos.registry import adapters as registry
from adaptos.routing import RouteDecision, RouterModel, sentence_embedder

logger = get_logger(__name__)

ROUTER = "router"
BASE = "base"


@dataclass(frozen=True)
class ChatResult:
    answer: str
    adapter: str  # task name or "base"
    adapter_version: str | None
    decision: RouteDecision
    router_version: str | None
    route_ms: float
    generate_ms: float
    output_tokens: int


class ModelManager:
    def __init__(
        self,
        app: AppConfig,
        settings: Settings,
        engine_factory: Callable[[], Any] | None = None,
        embed_factory: Callable[[str, str], Any] = sentence_embedder,
    ):
        self.app = app
        self.settings = settings
        self.engine_factory = engine_factory or self._default_engine
        self.embed_factory = embed_factory
        self.engine: Any = None
        self.router: RouterModel | None = None
        self.live: dict[str, str] = {}  # model ("router" or task) -> registry version
        self.engine_names: dict[str, str] = {}  # task -> adapter name inside the engine
        self.references: dict[str, dict] = {}  # task -> drift reference of the live version
        self.ready = False
        self.error: str | None = None
        self._lock = threading.Lock()  # one reload/rollback at a time
        self._embedders: dict[tuple[str, str], Any] = {}

    # ----------------------------------------------------------------- loading
    def _default_engine(self):
        from adaptos.inference import InferenceEngine

        b, s = self.app.base_model, self.app.serving
        return InferenceEngine(
            b.name, b.revision, device="cpu", num_threads=s.num_threads, dtype=s.dtype
        )

    def _cache(self) -> Path:
        return self.settings.artifacts_dir / "registry_cache"

    def _registered_name(self, model: str) -> str:
        if model == ROUTER:
            return self.app.router.registered_model
        return self.app.adapters[model].registered_model

    def load(self) -> None:
        """Load the base model and every champion. Safe to call from a background thread."""
        try:
            t0 = time.perf_counter()
            self.engine = self.engine_factory()
            self.reload()
            self.ready = True
            logger.info("models ready in %.1fs: %s", time.perf_counter() - t0, self.live)
        except Exception as e:  # surfaced through /ready
            self.error = f"{type(e).__name__}: {e}"
            logger.exception("model loading failed")

    def _sync(self, model: str) -> dict[str, Any] | None:
        """Bring one model in line with its registry champion. Returns the change, if any."""
        uri = self.settings.mlflow_tracking_uri
        name = self._registered_name(model)
        champ = registry.get_alias_version(name, registry.CHAMPION, uri)
        old = self.live.get(model)
        if champ is None:
            if model != ROUTER and model in self.engine_names:
                self.engine.unload_adapter(self.engine_names.pop(model))
                self.references.pop(model, None)
            elif model == ROUTER:
                self.router = None
            self.live.pop(model, None)
            return None if old is None else {"model": model, "from": old, "to": None}
        if old == champ.version:
            return None

        path = registry.cached_download(self._cache(), name, champ.version, uri)
        if model == ROUTER:
            rc = self.app.router
            key = (rc.embedding_model, rc.embedding_revision)
            if key not in self._embedders:
                self._embedders[key] = self.embed_factory(*key)
            self.router = RouterModel.load(
                path, threshold=rc.confidence_threshold, embed_fn=self._embedders[key]
            )
        else:
            new_name = f"{model}@v{champ.version}"
            self.engine.load_adapter(new_name, path)  # load next to the old version
            previous = self.engine_names.get(model)
            self.engine_names[model] = new_name  # switch traffic
            if previous and previous != new_name:
                self.engine.unload_adapter(previous)  # then drop the old version
            reference = self._drift_reference(path, getattr(champ, "source", None))
            if reference is None:
                self.references.pop(model, None)
                logger.warning("%s v%s has no drift reference", model, champ.version)
            else:
                self.references[model] = reference
        self.live[model] = champ.version
        LIVE_VERSION.labels(model).set(int(champ.version))
        MODEL_SWAPS.labels(model, "load" if old is None else "swap").inc()
        logger.info("%s: serving v%s (was %s)", model, champ.version, old)
        return {"model": model, "from": old, "to": champ.version}

    def _drift_reference(self, path: Path, source: str | None) -> dict | None:
        """The version's drift reference; fetched separately if the cached copy predates it."""
        reference = load_reference(path)
        if reference is None and source:
            try:
                import mlflow

                mlflow.artifacts.download_artifacts(
                    artifact_uri=f"{source}/{REFERENCE_FILE}",
                    dst_path=str(path),
                    tracking_uri=self.settings.mlflow_tracking_uri,
                )
                reference = load_reference(path)
            except Exception:
                return None
        return reference

    def reload(self, model: str | None = None) -> list[dict[str, Any]]:
        """Follow the registry: hot-swap every model (or one) whose champion changed."""
        models = [model] if model else [ROUTER, *sorted(self.app.adapters)]
        with self._lock:
            return [c for m in models if (c := self._sync(m)) is not None]

    def rollback(self, model: str) -> list[dict[str, Any]]:
        """Point the champion alias back to the previous version, then hot-swap to it."""
        registry.rollback(self._registered_name(model), self.settings.mlflow_tracking_uri)
        MODEL_SWAPS.labels(model, "rollback").inc()
        return self.reload(model)

    # ---------------------------------------------------------------- serving
    def chat(
        self, message: str, adapter: str | None = None, max_new_tokens: int | None = None
    ) -> ChatResult:
        """Route (unless ``adapter`` forces a choice) and generate an answer."""
        t0 = time.perf_counter()
        if adapter == BASE:
            decision = RouteDecision(None, BASE, 1.0, {}, "override")
        elif adapter is not None:
            decision = RouteDecision(adapter, adapter, 1.0, {}, "override")
        elif self.router is None:
            decision = RouteDecision(None, BASE, 0.0, {}, "router_unavailable")
        else:
            decision = self.router.route(message)
        if decision.adapter is not None and decision.adapter not in self.engine_names:
            decision = replace(decision, adapter=None, reason="adapter_unavailable")
        route_ms = (time.perf_counter() - t0) * 1000

        task = decision.adapter
        if max_new_tokens is None:
            max_new_tokens = (
                self.app.adapters[task].generation.max_new_tokens
                if task
                else self.app.serving.base_max_new_tokens
            )
        t1 = time.perf_counter()
        for attempt in range(2):
            name = self.engine_names[task] if task else None
            try:
                answer = self.engine.generate(
                    [message],
                    adapter=name,
                    prompt_task=task,
                    max_new_tokens=max_new_tokens,
                    batch_size=1,
                )[0]
                break
            except KeyError:  # a hot-swap removed that version between lookup and generate
                if attempt == 1:
                    raise
        generate_ms = (time.perf_counter() - t1) * 1000
        return ChatResult(
            answer=answer,
            adapter=task or BASE,
            adapter_version=self.live.get(task) if task else None,
            decision=decision,
            router_version=self.live.get(ROUTER),
            route_ms=route_ms,
            generate_ms=generate_ms,
            output_tokens=self.engine.count_tokens(answer),
        )
