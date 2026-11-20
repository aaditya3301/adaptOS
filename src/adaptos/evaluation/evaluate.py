"""Phase 3: evaluate a registered adapter version and run the promotion gate.

Usage:
    python -m adaptos.evaluation.evaluate --adapter all --with-baseline
    python -m adaptos.evaluation.evaluate --adapter sql --limit 5 --no-gate   # quick check

For each task it scores, on the same examples and with the same engine:
    candidate   the newest (or a chosen) registered version
    champion    the version currently served, if any
    baseline    the base model without an adapter (optional, shows the adapter's lift)
on the fixed golden set and the drift_eval set, logs everything to MLflow, and moves the
``champion`` alias only if the gate passes.
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from adaptos.config import AppConfig, Settings, get_settings, load_config
from adaptos.data.io import load_split, verify_dataset
from adaptos.data.schema import Example
from adaptos.evaluation.gate import GateDecision, decide
from adaptos.evaluation.metrics import aggregate, score
from adaptos.log import configure_logging, get_logger
from adaptos.provenance import git_commit
from adaptos.registry import adapters as registry

logger = get_logger(__name__)

EVAL_SPLITS = ("golden", "drift_eval")


def score_predictions(
    examples: Sequence[Example], predictions: Sequence[str]
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    """Aggregate metrics plus one row per example (for the predictions artifact)."""
    rows, per_example = [], []
    for ex, pred in zip(examples, predictions, strict=True):
        s = score(ex, pred)
        per_example.append(s)
        rows.append(
            {"id": ex.id, "input": ex.input[-600:], "target": ex.target, "prediction": pred, **s}
        )
    return aggregate(per_example), rows


def cached_adapter(settings: Settings, model_name: str, version: str) -> Path:
    return registry.cached_download(
        settings.artifacts_dir / "registry_cache", model_name, version, settings.mlflow_tracking_uri
    )


def evaluate_task(
    task: str,
    engine: Any,
    *,
    candidate_ref: str = "latest",
    with_baseline: bool = False,
    limit: int | None = None,
    gate: bool = True,
    use_mlflow: bool = True,
    settings: Settings,
    app: AppConfig,
) -> dict[str, Any]:
    cfg = app.adapters[task]
    name, uri = cfg.registered_model, settings.mlflow_tracking_uri
    metric = cfg.evaluation.primary_metric
    if limit and gate:
        logger.warning("%s: --limit given, so the gate is skipped (too few examples)", task)
        gate = False

    task_dir = settings.data_dir / "processed" / task
    manifest = verify_dataset(task_dir)
    data = {
        s: load_split(task_dir, s)[:limit] if limit else load_split(task_dir, s)
        for s in EVAL_SPLITS
    }

    cand = (
        registry.latest_version(name, uri)
        if candidate_ref == "latest"
        else registry.get_version(name, candidate_ref, uri)
    )
    champ = registry.get_alias_version(name, registry.CHAMPION, uri)
    if champ is not None and champ.version == cand.version:
        champ = None  # re-evaluating the current champion: nothing to compare against
        gate = False

    models: dict[str, str | None] = {"candidate": f"{task}@v{cand.version}"}
    engine.load_adapter(models["candidate"], cached_adapter(settings, name, cand.version))
    if champ is not None:
        models["champion"] = f"{task}@v{champ.version}"
        engine.load_adapter(models["champion"], cached_adapter(settings, name, champ.version))
    if with_baseline:
        models["baseline"] = None

    if gate:
        registry.set_alias(name, registry.CHALLENGER, cand.version, uri)

    batch_size = 16 if engine.device == "cuda" else 4
    results: dict[str, dict[str, dict[str, float]]] = {}
    rows: dict[str, list[dict[str, Any]]] = {}
    t0 = time.perf_counter()
    for label, adapter in models.items():
        results[label] = {}
        for split, examples in data.items():
            preds = engine.generate(
                [e.input for e in examples],
                adapter=adapter,
                prompt_task=task,
                max_new_tokens=cfg.generation.max_new_tokens,
                batch_size=batch_size,
            )
            results[label][split], rows[f"{label}_{split}"] = score_predictions(examples, preds)
            logger.info(
                "%s %-9s %-10s %s=%.3f", task, label, split, metric, results[label][split][metric]
            )
    eval_time = time.perf_counter() - t0

    decision: GateDecision | None = None
    if gate:
        primary = {label: {s: r[metric] for s, r in res.items()} for label, res in results.items()}
        decision = decide(cfg.evaluation, primary["candidate"], primary.get("champion"))
        logger.info(
            "%s: gate %s v%s - %s",
            task, "PROMOTES" if decision.promote else "REJECTS", cand.version, decision.reason,
        )  # fmt: skip

    summary = {
        "task": task,
        "primary_metric": metric,
        "candidate_version": cand.version,
        "champion_version": champ.version if champ else None,
        "dataset_version": manifest["dataset_version"],
        "results": results,
        "gate": None
        if decision is None
        else {"promote": decision.promote, "reason": decision.reason},
        "eval_time_sec": eval_time,
    }

    if use_mlflow:
        import mlflow

        mlflow.set_tracking_uri(uri)
        mlflow.set_experiment(f"{app.mlflow.experiment_prefix}-evaluation")
        with mlflow.start_run(run_name=f"{task}-eval-v{cand.version}") as run:
            mlflow.set_tags(
                {
                    "task": task,
                    "stage": "evaluation",
                    "candidate_version": cand.version,
                    "champion_version": champ.version if champ else "none",
                    "dataset_version": manifest["dataset_version"],
                    "compute": engine.device,
                    "gate_decision": (
                        "skipped" if decision is None
                        else "promoted" if decision.promote else "rejected"
                    ),
                    "git_commit": git_commit(),
                }
            )  # fmt: skip
            mlflow.log_params(
                {
                    "primary_metric": metric,
                    "splits": ",".join(EVAL_SPLITS),
                    "limit": limit or "none",
                    "max_new_tokens": cfg.generation.max_new_tokens,
                    "min_score": cfg.evaluation.min_score,
                    "promotion_min_delta": cfg.evaluation.promotion_min_delta,
                    "max_regression": cfg.evaluation.max_regression,
                    **{f"n_{s}": len(e) for s, e in data.items()},
                }
            )
            mlflow.log_metrics(
                {
                    f"{label}/{split}/{k}": v
                    for label, res in results.items()
                    for split, metrics in res.items()
                    for k, v in metrics.items()
                }
                | {"eval_time_sec": eval_time}
            )
            for key, r in rows.items():
                mlflow.log_text(
                    "\n".join(json.dumps(x, ensure_ascii=False) for x in r),
                    f"predictions/{key}.jsonl",
                )
            mlflow.log_dict(summary, "summary.json")
            summary["run_id"] = run.info.run_id

        version_tags = {
            f"eval_{s}_{metric}": f"{results['candidate'][s][metric]:.4f}" for s in EVAL_SPLITS
        } | {"eval_run_id": summary["run_id"]}
        if decision is not None:
            version_tags |= {
                "gate_decision": "promoted" if decision.promote else "rejected",
                "gate_reason": decision.reason,
            }
        registry.tag_version(name, cand.version, version_tags, uri)

    if decision is not None:
        if decision.promote:
            old = registry.promote(name, cand.version, uri)
            logger.info("%s: champion is now v%s (previous: %s)", task, cand.version, old)
        registry.delete_alias(name, registry.CHALLENGER, uri)

    for adapter in models.values():
        if adapter is not None:
            engine.unload_adapter(adapter)
    return summary


def build_engine(app: AppConfig, device: str | None = None):
    from adaptos.inference import InferenceEngine

    base = app.base_model
    return InferenceEngine(
        base.name, base.revision, device=device or base.device, num_threads=base.num_threads
    )


def main(argv: Sequence[str] | None = None) -> None:
    settings = get_settings()
    configure_logging(settings.log_level)
    app = load_config(settings.config_dir)
    p = argparse.ArgumentParser(description="Evaluate adapter versions and run the promotion gate")
    p.add_argument("--adapter", choices=[*sorted(app.adapters), "all"], required=True)
    p.add_argument("--candidate", default="latest", help="'latest' or a version number")
    p.add_argument("--with-baseline", action="store_true", help="also score the plain base model")
    p.add_argument("--limit", type=int, help="score only the first N examples (skips the gate)")
    p.add_argument("--no-gate", action="store_true", help="score only; never move aliases")
    p.add_argument("--no-mlflow", action="store_true")
    p.add_argument("--device", choices=["cpu", "cuda"])
    args = p.parse_args(argv)

    engine = build_engine(app, args.device)
    tasks = sorted(app.adapters) if args.adapter == "all" else [args.adapter]
    for task in tasks:
        evaluate_task(
            task,
            engine,
            candidate_ref=args.candidate,
            with_baseline=args.with_baseline,
            limit=args.limit,
            gate=not args.no_gate,
            use_mlflow=not args.no_mlflow,
            settings=settings,
            app=app,
        )


if __name__ == "__main__":
    main()
