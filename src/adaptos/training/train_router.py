"""Phase 4: train the router, evaluate it, register it and run its promotion gate.

Usage:
    python -m adaptos.training.train_router

Data: adapter requests come from each task's processed splits (train / eval / golden), general
requests ("other") from a pinned slice of Dolly-15k. The drift splits are scored separately:
a drifted request is still the same task, so the router must keep routing it correctly even
when the adapter itself struggles.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from collections import Counter
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from adaptos.config import AppConfig, Settings, get_settings, load_config
from adaptos.data.io import load_split, verify_dataset
from adaptos.log import configure_logging, get_logger
from adaptos.provenance import git_commit
from adaptos.registry import adapters as registry
from adaptos.routing import OTHER, RouterModel, sentence_embedder

logger = get_logger(__name__)

SUMMARIZE_PREFIX = "Summarize this conversation:\n\n"
# router split <- task split
TASK_SPLITS = {"train": "train", "eval": "eval", "test": "golden", "drift": "drift_eval"}
Dataset = dict[str, list[tuple[str, str]]]  # split -> [(text, label)]


def load_other_rows(app: AppConfig) -> list[dict[str, Any]]:
    from datasets import load_dataset

    o = app.router.other_dataset
    return list(load_dataset(o.hf_id, split="train", revision=o.revision))


def build_dataset(
    app: AppConfig, settings: Settings, other_rows: Sequence[dict[str, Any]]
) -> tuple[Dataset, dict[str, str]]:
    """Router examples per split, plus the dataset version of every source."""
    data: Dataset = {split: [] for split in (*TASK_SPLITS, "test_no_prefix")}
    versions = {}
    for task in sorted(app.adapters):
        task_dir = settings.data_dir / "processed" / task
        versions[task] = verify_dataset(task_dir)["dataset_version"]
        for split, src in TASK_SPLITS.items():
            data[split] += [(ex.input, task) for ex in load_split(task_dir, src)]

    # Robustness check: summaries requested without the fixed instruction prefix.
    data["test_no_prefix"] = [
        (t.removeprefix(SUMMARIZE_PREFIX), lab) for t, lab in data["test"] if lab == "summarize"
    ]

    o = app.router.other_dataset
    excluded = set(o.exclude_categories)
    texts = []
    seen = set()
    for row in other_rows:
        if row.get("category") in excluded:
            continue
        text = row["instruction"].strip()
        if row.get("context"):
            text += "\n\n" + row["context"].strip()
        if text and text not in seen:
            seen.add(text)
            texts.append(text)
    random.Random(o.seed).shuffle(texts)
    a, b, c = o.train_size, o.train_size + o.eval_size, o.train_size + o.eval_size + o.test_size
    if len(texts) < c:
        raise ValueError(f"only {len(texts)} 'other' examples, need {c}")
    data["train"] += [(t, OTHER) for t in texts[:a]]
    data["eval"] += [(t, OTHER) for t in texts[a:b]]
    data["test"] += [(t, OTHER) for t in texts[b:c]]

    digest = hashlib.sha256(
        json.dumps({k: sorted(v) for k, v in data.items()}, sort_keys=True).encode()
    ).hexdigest()
    versions["router"] = digest[:12]
    versions[OTHER] = o.revision[:12]
    return data, versions


def augment(train: list[tuple[str, str]], prob: float, seed: int) -> list[tuple[str, str]]:
    """Drop the fixed summarize prefix from some training requests."""
    rng = random.Random(seed)
    return [
        (t.removeprefix(SUMMARIZE_PREFIX), lab)
        if lab == "summarize" and rng.random() < prob
        else (t, lab)
        for t, lab in train
    ]


def fit_classifier(
    x_train: np.ndarray, y_train: Sequence[str], x_eval: np.ndarray, y_eval: Sequence[str]
) -> tuple[Any, float, dict[float, float]]:
    """Multinomial logistic regression; picks C on the eval split."""
    from sklearn.linear_model import LogisticRegression

    scores, best = {}, None
    for c in (0.1, 1.0, 10.0):
        clf = LogisticRegression(C=c, max_iter=2000, class_weight="balanced")
        clf.fit(x_train, y_train)
        scores[c] = float((clf.predict(x_eval) == np.asarray(y_eval)).mean())
        if best is None or scores[c] > scores[best[0]]:
            best = (c, clf)
    assert best is not None
    return best[1], best[0], scores


def split_metrics(router: RouterModel, x: np.ndarray, y: Sequence[str]) -> dict[str, Any]:
    probs = router.probabilities_from_embeddings(x)
    decisions = [router.decide(p) for p in probs]
    pred = [d.label for d in decisions]
    y = list(y)
    acc = sum(p == t for p, t in zip(pred, y, strict=True)) / len(y)
    labels = router.labels
    confusion = {t: Counter(p for p, tt in zip(pred, y, strict=True) if tt == t) for t in labels}
    recall = {t: confusion[t][t] / max(1, sum(confusion[t].values())) for t in labels if t in y}
    # Served outcome: correct adapter, or base model for 'other'; low confidence -> base model.
    served = [d.adapter or OTHER for d in decisions]
    covered = [d.reason == "adapter" for d in decisions]
    return {
        "accuracy": acc,
        "served_accuracy": sum(s == t for s, t in zip(served, y, strict=True)) / len(y),
        "fallback_rate": sum(d.reason == "low_confidence" for d in decisions) / len(y),
        "adapter_coverage": sum(covered) / len(y),
        "recall": recall,
        "confusion": {t: dict(c) for t, c in confusion.items()},
        "predictions": pred,
        "confidence": [d.confidence for d in decisions],
    }


def train_router(
    *,
    settings: Settings | None = None,
    app: AppConfig | None = None,
    other_rows: Sequence[dict[str, Any]] | None = None,
    embed_fn: Callable[[Sequence[str]], np.ndarray] | None = None,
    use_mlflow: bool = True,
    gate: bool = True,
) -> dict[str, Any]:
    settings = settings or get_settings()
    app = app or load_config(settings.config_dir)
    rc = app.router
    data, versions = build_dataset(
        app, settings, other_rows if other_rows is not None else load_other_rows(app)
    )
    train = augment(data["train"], rc.strip_instruction_prob, rc.other_dataset.seed)
    embed = embed_fn or sentence_embedder(rc.embedding_model, rc.embedding_revision)

    t0 = time.perf_counter()
    splits = {s: rows for s, rows in {**data, "train": train}.items() if rows}  # skip empty
    x = {split: embed([t for t, _ in rows]) for split, rows in splits.items()}
    y = {split: [lab for _, lab in rows] for split, rows in splits.items()}
    embed_time = time.perf_counter() - t0
    logger.info("embedded %d texts in %.1fs", sum(len(v) for v in y.values()), embed_time)

    clf, best_c, c_scores = fit_classifier(x["train"], y["train"], x["eval"], y["eval"])
    router = RouterModel(
        labels=[str(c) for c in clf.classes_],
        coef=clf.coef_,
        intercept=clf.intercept_,
        embedding_model=rc.embedding_model,
        embedding_revision=rc.embedding_revision,
        threshold=rc.confidence_threshold,
        embed_fn=embed,
    )
    results = {
        s: split_metrics(router, x[s], y[s])
        for s in ("eval", "test", "drift", "test_no_prefix")
        if s in x
    }
    for s, r in results.items():
        logger.info(
            "router %-14s accuracy %.3f | served %.3f | fallback %.3f",
            s, r["accuracy"], r["served_accuracy"], r["fallback_rate"],
        )  # fmt: skip

    # Latency of a single request (what the gateway pays per call).
    probe = [t for t, _ in data["test"][:50]]
    t1 = time.perf_counter()
    for text in probe:
        router.route(text)
    latency_ms = (time.perf_counter() - t1) / len(probe) * 1000

    out_dir = settings.artifacts_dir / "router" / versions["router"]
    router.save(out_dir)
    test_acc = results["test"]["accuracy"]
    summary: dict[str, Any] = {
        "dataset_versions": versions,
        "best_C": best_c,
        "c_scores": c_scores,
        "latency_ms": latency_ms,
        "metrics": {
            s: {k: v for k, v in r.items() if k not in ("predictions", "confidence")}
            for s, r in results.items()
        },
    }

    if not use_mlflow:
        return summary

    import mlflow

    uri = settings.mlflow_tracking_uri
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(f"{app.mlflow.experiment_prefix}-router")
    with mlflow.start_run(run_name=f"router-{versions['router']}") as run:
        mlflow.set_tags(
            {
                "stage": "router",
                "router_dataset_version": versions["router"],
                "git_commit": git_commit(),
                **{f"dataset_version_{k}": v for k, v in versions.items() if k != "router"},
            }
        )
        mlflow.log_params(
            {
                "embedding_model": rc.embedding_model,
                "embedding_revision": rc.embedding_revision,
                "classifier": "logistic_regression",
                "C": best_c,
                "labels": ",".join(router.labels),
                "confidence_threshold": rc.confidence_threshold,
                "strip_instruction_prob": rc.strip_instruction_prob,
                **{f"n_{s}": len(v) for s, v in y.items()},
            }
        )
        metrics = {"latency_ms": latency_ms, "embed_time_sec": embed_time}
        for s, r in results.items():
            for k in ("accuracy", "served_accuracy", "fallback_rate", "adapter_coverage"):
                metrics[f"{s}/{k}"] = r[k]
            metrics |= {f"{s}/recall/{lab}": v for lab, v in r["recall"].items()}
        metrics |= {f"eval/accuracy_C_{c}": v for c, v in c_scores.items()}
        mlflow.log_metrics(metrics)
        mlflow.log_artifacts(str(out_dir), artifact_path="router")
        mlflow.log_dict(summary, "summary.json")
        misrouted = [
            {"text": t[:400], "label": lab, "predicted": p, "confidence": round(c, 4)}
            for (t, lab), p, c in zip(
                data["test"],
                results["test"]["predictions"],
                results["test"]["confidence"],
                strict=True,
            )
            if p != lab
        ]
        mlflow.log_text("\n".join(json.dumps(m) for m in misrouted), "misrouted_test.jsonl")

        registry.ensure_registered_model(
            rc.registered_model,
            description="Request router: sentence embedding + logistic regression",
            tracking_uri=uri,
        )
        mv = registry.register_adapter_version(
            rc.registered_model,
            run_id=run.info.run_id,
            source_uri=mlflow.get_artifact_uri("router"),
            tags={
                "router_dataset_version": versions["router"],
                "test_accuracy": f"{test_acc:.4f}",
                "drift_accuracy": f"{results['drift']['accuracy']:.4f}",
                "embedding_model": rc.embedding_model,
            },
            description=f"test accuracy {test_acc:.3f}, dataset {versions['router']}",
            tracking_uri=uri,
        )
        summary["run_id"], summary["version"] = run.info.run_id, mv.version
        mlflow.set_tag("registered_version", mv.version)

    if gate:
        champ = registry.get_alias_version(rc.registered_model, registry.CHAMPION, uri)
        champ_acc = (
            float(champ.tags["test_accuracy"])
            if champ is not None and champ.tags.get("router_dataset_version") == versions["router"]
            else None
        )
        if test_acc < rc.min_accuracy:
            promote, reason = False, f"test accuracy {test_acc:.3f} < minimum {rc.min_accuracy}"
        elif champ_acc is not None and test_acc < champ_acc:
            promote, reason = False, f"test accuracy {test_acc:.3f} < champion {champ_acc:.3f}"
        else:
            promote, reason = True, f"test accuracy {test_acc:.3f} >= {rc.min_accuracy}"
        registry.tag_version(
            rc.registered_model,
            mv.version,
            {"gate_decision": "promoted" if promote else "rejected", "gate_reason": reason},
            uri,
        )
        if promote:
            registry.promote(rc.registered_model, mv.version, uri)
        summary["gate"] = {"promote": promote, "reason": reason}
        logger.info(
            "router v%s: %s (%s)", mv.version, "PROMOTED" if promote else "REJECTED", reason
        )
    return summary


def main(argv: Sequence[str] | None = None) -> None:
    settings = get_settings()
    configure_logging(settings.log_level)
    p = argparse.ArgumentParser(description="Train, evaluate and register the request router")
    p.add_argument("--no-mlflow", action="store_true")
    p.add_argument("--no-gate", action="store_true")
    args = p.parse_args(argv)
    train_router(settings=settings, use_mlflow=not args.no_mlflow, gate=not args.no_gate)


if __name__ == "__main__":
    main()
