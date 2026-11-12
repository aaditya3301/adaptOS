"""Phase 1 data pipeline: download → clean → split → version → log to MLflow.

Usage:
    python -m adaptos.data.pipeline --task all
    python -m adaptos.data.pipeline --task sql --no-mlflow

Output per task (data/processed/<task>/):
    train.jsonl, eval.jsonl, golden.jsonl, drift_train.jsonl, drift_eval.jsonl, manifest.json

The dataset version is a hash of the split files' content only, so rebuilding with the same
config and source revision always gives the same version, and MLflow skips re-registering it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
from collections import Counter
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from adaptos.config import AdapterConfig, AppConfig, Settings, get_settings, load_config
from adaptos.data.formatters import Rejected, get_formatter
from adaptos.data.io import MANIFEST, sha256_file, write_jsonl
from adaptos.data.schema import Example
from adaptos.log import configure_logging, get_logger

logger = get_logger(__name__)

PIPELINE_VERSION = "1"
_CHARS_PER_TOKEN = 3.2  # conservative estimate for English text and SQL
_PROMPT_OVERHEAD_TOKENS = 48  # chat template + system prompt added at training time


def load_raw(cfg: AdapterConfig) -> list[dict[str, Any]]:
    """Download the pinned dataset revision from Hugging Face (cached locally)."""
    from datasets import load_dataset  # heavy import, only needed here

    ds = cfg.dataset
    rows = load_dataset(ds.hf_id, ds.hf_config, split=ds.source_split, revision=ds.revision)
    return list(rows)


def _fits_context(ex: Example, max_seq_len: int) -> bool:
    approx_tokens = (len(ex.input) + len(ex.target)) / _CHARS_PER_TOKEN
    return approx_tokens <= max_seq_len - _PROMPT_OVERHEAD_TOKENS


def build_splits(
    cfg: AdapterConfig, rows: Sequence[dict[str, Any]]
) -> tuple[dict[str, list[Example]], dict[str, Any]]:
    """Clean, dedupe and split raw rows. Deterministic for a given seed and input."""
    ds = cfg.dataset
    formatter = get_formatter(cfg.name)
    need_base = ds.train_size + ds.eval_size + ds.golden_size
    need_drift = ds.drift.train_size + ds.drift.eval_size

    order = list(range(len(rows)))
    random.Random(ds.seed).shuffle(order)

    base: list[Example] = []
    drift: list[Example] = []
    seen: set[str] = set()
    dropped: Counter[str] = Counter()
    scanned = 0
    for idx in order:
        if len(base) >= need_base and len(drift) >= need_drift:
            break
        scanned += 1
        try:
            ex = formatter.format(rows[idx])
        except Rejected as r:
            dropped[r.reason] += 1
            continue
        if ex.id in seen:
            dropped["duplicate"] += 1
            continue
        if not _fits_context(ex, cfg.training.max_seq_len):
            dropped["too_long"] += 1
            continue
        seen.add(ex.id)
        bucket = drift if formatter.is_drift(ex, ds.drift.rule, ds.drift.params) else base
        if len(bucket) < (need_drift if bucket is drift else need_base):
            bucket.append(ex)

    if len(base) < need_base or len(drift) < need_drift:
        raise ValueError(
            f"{cfg.name}: not enough clean data (base {len(base)}/{need_base}, "
            f"drift {len(drift)}/{need_drift}); lower the sizes in configs/adapters/{cfg.name}.yaml"
        )

    a, b = ds.train_size, ds.train_size + ds.eval_size
    splits = {
        "train": base[:a],
        "eval": base[a:b],
        "golden": base[b:need_base],
        "drift_train": drift[: ds.drift.train_size],
        "drift_eval": drift[ds.drift.train_size : need_drift],
    }
    stats = {"rows_available": len(rows), "rows_scanned": scanned, "dropped": dict(dropped)}
    return splits, stats


def _git_commit() -> str:
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True, check=True
        ).stdout.strip()
        return f"{sha}{'-dirty' if dirty else ''}"
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _avg(values: Iterable[int]) -> float:
    values = list(values)
    return round(sum(values) / len(values), 1) if values else 0.0


def write_dataset(
    cfg: AdapterConfig, splits: dict[str, list[Example]], stats: dict[str, Any], out_dir: Path
) -> dict[str, Any]:
    """Write split files and a manifest. Returns the manifest."""
    out_dir.mkdir(parents=True, exist_ok=True)
    split_info: dict[str, Any] = {}
    for name, examples in splits.items():
        path = out_dir / f"{name}.jsonl"
        write_jsonl(path, examples)
        split_info[name] = {
            "file": path.name,
            "count": len(examples),
            "sha256": sha256_file(path),
            "avg_input_chars": _avg(len(e.input) for e in examples),
            "avg_target_chars": _avg(len(e.target) for e in examples),
        }

    content_hash = hashlib.sha256(
        "".join(f"{n}:{i['sha256']}" for n, i in sorted(split_info.items())).encode()
    ).hexdigest()
    ds = cfg.dataset
    manifest = {
        "task": cfg.name,
        "dataset_version": content_hash[:12],
        "pipeline_version": PIPELINE_VERSION,
        "source": {
            "hf_id": ds.hf_id,
            "hf_config": ds.hf_config,
            "revision": ds.revision,
            "split": ds.source_split,
        },
        "seed": ds.seed,
        "drift": {"rule": ds.drift.rule, "params": ds.drift.params},
        "splits": split_info,
        "filtering": stats,
        "git_commit": _git_commit(),
    }
    (out_dir / MANIFEST).write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )
    return manifest


def log_to_mlflow(
    manifest: dict[str, Any], task_dir: Path, app: AppConfig, settings: Settings, force: bool
) -> str | None:
    """Register the dataset version as an MLflow run. Skips versions already registered."""
    import mlflow

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    experiment = mlflow.set_experiment(f"{app.mlflow.experiment_prefix}-data")
    task, version = manifest["task"], manifest["dataset_version"]

    existing = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"tags.task = '{task}' AND tags.dataset_version = '{version}'",
        output_format="list",
    )
    if existing and not force:
        run_id = existing[0].info.run_id
        logger.info("%s dataset %s already registered (run %s); skipping", task, version, run_id)
        return run_id

    with mlflow.start_run(run_name=f"{task}-data-{version}") as run:
        mlflow.set_tags(
            {
                "task": task,
                "dataset_version": version,
                "stage": "data",
                "git_commit": manifest["git_commit"],
            }
        )
        src = manifest["source"]
        mlflow.log_params(
            {
                "hf_id": src["hf_id"],
                "hf_config": src["hf_config"],
                "revision": src["revision"],
                "seed": manifest["seed"],
                "drift_rule": manifest["drift"]["rule"],
                "pipeline_version": manifest["pipeline_version"],
                **{f"size_{n}": i["count"] for n, i in manifest["splits"].items()},
            }
        )
        filtering = manifest["filtering"]
        metrics = {
            "rows_scanned": filtering["rows_scanned"],
            "rows_dropped": sum(filtering["dropped"].values()),
            **{f"dropped_{k}": v for k, v in filtering["dropped"].items()},
        }
        for name, info in manifest["splits"].items():
            metrics[f"{name}_avg_input_chars"] = info["avg_input_chars"]
            metrics[f"{name}_avg_target_chars"] = info["avg_target_chars"]
        mlflow.log_metrics(metrics)
        mlflow.log_artifacts(str(task_dir), artifact_path="dataset")
        logger.info("%s dataset %s registered in MLflow (run %s)", task, version, run.info.run_id)
        return run.info.run_id


def run(tasks: Sequence[str], use_mlflow: bool = True, force: bool = False) -> list[dict]:
    settings = get_settings()
    app = load_config(settings.config_dir)
    manifests = []
    for task in tasks:
        cfg = app.adapters[task]
        logger.info("%s: loading %s@%s", task, cfg.dataset.hf_id, cfg.dataset.revision[:8])
        splits, stats = build_splits(cfg, load_raw(cfg))
        task_dir = settings.data_dir / "processed" / task
        manifest = write_dataset(cfg, splits, stats, task_dir)
        counts = ", ".join(f"{n}={i['count']}" for n, i in manifest["splits"].items())
        logger.info("%s: dataset %s written (%s)", task, manifest["dataset_version"], counts)
        if use_mlflow:
            log_to_mlflow(manifest, task_dir, app, settings, force)
        manifests.append(manifest)
    return manifests


def main(argv: Sequence[str] | None = None) -> None:
    settings = get_settings()
    configure_logging(settings.log_level)
    adapters = sorted(load_config(settings.config_dir).adapters)
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--task", choices=[*adapters, "all"], default="all")
    parser.add_argument("--no-mlflow", action="store_true", help="skip MLflow registration")
    parser.add_argument("--force", action="store_true", help="re-register an existing version")
    args = parser.parse_args(argv)
    tasks = adapters if args.task == "all" else [args.task]
    run(tasks, use_mlflow=not args.no_mlflow, force=args.force)


if __name__ == "__main__":
    main()
