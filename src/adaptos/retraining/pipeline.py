"""Phase 7 retraining pipeline: new data -> train -> evaluate -> gate.

    python -m adaptos.retraining.pipeline --task sql
    python -m adaptos.retraining.pipeline --task sql --device cuda --backend qlora \
        --parent-run-id <id>

Training data for a retraining run:
    new examples   the task's ``retraining.new_data_splits`` (freshly labeled data for the new
                   kind of request) plus validated user corrections from the request log
    rehearsal      ``replay_ratio`` x as many examples from the original training split, so the
                   adapter keeps what it already does well (avoids catastrophic forgetting)
The run warm-starts from the champion adapter, then the new version goes through the same
evaluation and promotion gate as any other candidate. The controller hot-swaps it if promoted.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from typing import Any

from adaptos.config import AppConfig, Settings, get_settings, load_config
from adaptos.data.formatters import sql_executes
from adaptos.data.io import load_split
from adaptos.data.schema import Example, example_id
from adaptos.evaluation.evaluate import build_engine, evaluate_task
from adaptos.evaluation.metrics import extract_answer, extract_sql
from adaptos.log import configure_logging, get_logger
from adaptos.monitoring.quality import SCHEMA_STATEMENT
from adaptos.registry import adapters as registry

logger = get_logger(__name__)


def valid_correction(task: str, user_input: str, target: str) -> bool:
    """User corrections are untrusted labels: keep only ones that pass the task's checks."""
    target = target.strip()
    if not target:
        return False
    if task == "sql":
        schema = "; ".join(SCHEMA_STATEMENT.findall(user_input))
        return bool(schema) and sql_executes(schema, extract_sql(target))
    if task == "math":
        return extract_answer(target)[0] is not None
    if task == "summarize":
        return len(target) < len(user_input)
    return True


def collect_corrections(rows: Sequence[dict[str, Any]], task: str) -> list[Example]:
    """Turn corrected requests into training examples (newest correction wins)."""
    examples: dict[str, Example] = {}
    rejected = 0
    for row in rows:  # newest first
        target = (row.get("correction") or "").strip()
        if not valid_correction(task, row["input"], target):
            rejected += 1
            continue
        ex_id = example_id(task, row["input"])
        if ex_id not in examples:
            target = target if task != "math" or "Final answer" in target else (
                f"{target}\nFinal answer: {extract_answer(target)[0]}"
            )  # fmt: skip
            examples[ex_id] = Example(
                id=ex_id,
                task=task,
                input=row["input"],
                target=target,
                meta={"source": "feedback", "request_id": row["request_id"]},
            )
    if rejected:
        logger.info("%s: %d user corrections failed validation and were skipped", task, rejected)
    return list(examples.values())


def run_retraining(
    task: str,
    *,
    parent_run_id: str | None = None,
    device: str | None = None,
    backend: str | None = None,
    settings: Settings | None = None,
    app: AppConfig | None = None,
    correction_rows: Sequence[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    from adaptos.training.train_adapter import train_adapter

    settings = settings or get_settings()
    app = app or load_config(settings.config_dir)
    rc, cfg, uri = app.retraining, app.adapters[task], settings.mlflow_tracking_uri

    if correction_rows is None:
        try:
            from adaptos.serving.store import RequestStore

            correction_rows = RequestStore(settings.database_url).corrections(task)
        except Exception as e:  # the request log is optional for retraining
            logger.warning("request log unavailable (%s); training without user corrections", e)
            correction_rows = []
    corrections = collect_corrections(correction_rows, task)

    task_dir = settings.data_dir / "processed" / task
    n_new = sum(len(load_split(task_dir, s)) for s in rc.new_data_splits) + len(corrections)
    replay = round(rc.replay_ratio * n_new)

    champion = registry.get_alias_version(cfg.registered_model, registry.CHAMPION, uri)
    init_dir = None
    if rc.warm_start and champion is not None:
        init_dir = registry.cached_download(
            settings.artifacts_dir / "registry_cache", cfg.registered_model, champion.version, uri
        )
    logger.info(
        "%s: retraining on %d new + %d replayed examples (%d from feedback), warm start from %s",
        task, n_new, replay, len(corrections), f"v{champion.version}" if init_dir else "scratch",
    )  # fmt: skip

    trigger = f"retraining:{parent_run_id}" if parent_run_id else "retraining:manual"
    result = train_adapter(
        task,
        splits=["train", *rc.new_data_splits],
        split_limits={"train": replay},
        extra_examples=corrections,
        extra_tags={"parent_run_id": parent_run_id} if parent_run_id else None,
        epochs=rc.epochs,
        init_adapter_dir=init_dir,
        trigger=trigger,
        device=device,
        backend=backend,
        settings=settings,
        app=app,
    )
    if device == "cuda":  # release the training model's GPU memory before evaluation
        import gc

        import torch

        gc.collect()
        torch.cuda.empty_cache()

    summary = evaluate_task(
        task,
        build_engine(app, device),
        candidate_ref=str(result.model_version),
        gate=True,
        settings=settings,
        app=app,
    )
    outcome = {
        "task": task,
        "candidate_version": result.model_version,
        "previous_champion": champion.version if champion else None,
        "promoted": bool(summary["gate"] and summary["gate"]["promote"]),
        "gate_reason": summary["gate"]["reason"] if summary["gate"] else None,
        "results": summary["results"],
        "n_new": n_new,
        "n_replay": replay,
        "n_feedback": len(corrections),
        "training_run_id": result.run_id,
        "evaluation_run_id": summary.get("run_id"),
    }
    if parent_run_id:
        import mlflow

        client = mlflow.MlflowClient(uri)
        metric = cfg.evaluation.primary_metric
        client.set_tag(parent_run_id, "candidate_version", str(result.model_version))
        client.set_tag(parent_run_id, "gate_reason", outcome["gate_reason"])
        client.set_tag(parent_run_id, "training_run_id", str(result.run_id))
        client.set_tag(parent_run_id, "evaluation_run_id", str(summary.get("run_id")))
        for label, res in summary["results"].items():
            for split, metrics in res.items():
                client.log_metric(parent_run_id, f"{label}/{split}/{metric}", metrics[metric])
        for k in ("n_new", "n_replay", "n_feedback"):
            client.log_metric(parent_run_id, k, outcome[k])
    logger.info(
        "%s: candidate v%s %s (%s)",
        task, result.model_version, "PROMOTED" if outcome["promoted"] else "REJECTED",
        outcome["gate_reason"],
    )  # fmt: skip
    return outcome


def main(argv: Sequence[str] | None = None) -> None:
    settings = get_settings()
    configure_logging(settings.log_level)
    app = load_config(settings.config_dir)
    p = argparse.ArgumentParser(description="Retrain one adapter and run the promotion gate")
    p.add_argument("--task", choices=sorted(app.adapters), required=True)
    p.add_argument("--parent-run-id", help="retraining run (controller) to report into")
    p.add_argument("--device", choices=["cpu", "cuda"])
    p.add_argument("--backend", choices=["lora", "qlora"])
    args = p.parse_args(argv)
    run_retraining(
        args.task,
        parent_run_id=args.parent_run_id,
        device=args.device,
        backend=args.backend,
        settings=settings,
        app=app,
    )


if __name__ == "__main__":
    main()
