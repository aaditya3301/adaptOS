"""Phase 2: train a LoRA adapter on CPU, log everything to MLflow, register a new version.

Usage:
    python -m adaptos.training.train_adapter --adapter sql
    python -m adaptos.training.train_adapter --adapter sql --test-model --max-steps 5

The same function is reused by the retraining pipeline (Phase 7), which passes extra data
splits, a starting adapter and a trigger reason.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import random
import shutil
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from adaptos.config import AppConfig, Settings, get_settings, load_config
from adaptos.data.io import load_split, verify_dataset
from adaptos.log import configure_logging, get_logger
from adaptos.prompts import PROMPT_VERSION
from adaptos.provenance import git_commit
from adaptos.registry.adapters import (
    ADAPTER_ARTIFACT_PATH,
    ensure_registered_model,
    register_adapter_version,
)
from adaptos.training.tokenization import (
    IGNORE_INDEX,
    TokenizedExample,
    collate,
    length_grouped_batches,
    tokenize_example,
)

logger = get_logger(__name__)


@dataclass
class TrainResult:
    task: str
    adapter_dir: Path
    metrics: dict[str, float]
    run_id: str | None = None
    model_version: str | None = None
    params: dict[str, Any] = field(default_factory=dict)


class _Tracker:
    """Thin MLflow wrapper that becomes a no-op when tracking is disabled."""

    def __init__(self, enabled: bool):
        self.enabled = enabled
        if enabled:
            import mlflow

            self.mlflow = mlflow

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        if self.enabled:
            self.mlflow.log_metrics(metrics, step=step)


def _rss_mb() -> float:
    import psutil

    return psutil.Process().memory_info().rss / 2**20


def _autocast(device: str):
    """fp16 mixed precision on GPU (T4-friendly); full fp32 on CPU."""
    import torch

    if device == "cuda":
        return torch.autocast("cuda", dtype=torch.float16)
    return contextlib.nullcontext()


def _to(batch: dict[str, Any], device: str) -> dict[str, Any]:
    return {k: v.to(device) for k, v in batch.items()}


def eval_loss(
    model: Any,
    items: Sequence[TokenizedExample],
    batch_size: int,
    pad_id: int,
    device: str = "cpu",
) -> float:
    """Token-weighted mean loss over answer tokens."""
    import torch

    model.eval()
    total, tokens = 0.0, 0
    with torch.no_grad():
        for i in range(0, len(items), batch_size):
            batch = _to(collate(items[i : i + batch_size], pad_id), device)
            n = int((batch["labels"][:, 1:] != IGNORE_INDEX).sum())
            with _autocast(device):
                total += float(model(**batch).loss) * n
            tokens += n
    model.train()
    return total / max(tokens, 1)


def load_base_model(model_id: str, revision: str, backend: str, device: str, dtype: str):
    """Load tokenizer + base model. ``qlora`` loads the base in 4-bit NF4 (GPU only)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    if backend == "qlora":
        from peft import prepare_model_for_kbit_training
        from transformers import BitsAndBytesConfig

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            revision=revision,
            dtype=torch.float16,
            device_map={"": 0},
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.float16,
            ),
        )
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_id, revision=revision, dtype=getattr(torch, dtype)
        ).to(device)
    return tokenizer, model


def _lr_lambda(warmup: int, total: int):
    def fn(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        return max(0.0, (total - step) / max(1, total - warmup))

    return fn


def train_adapter(
    task: str,
    *,
    use_test_model: bool = False,
    extra_splits: Sequence[str] = (),
    init_adapter_dir: Path | None = None,
    trigger: str = "manual",
    max_steps: int | None = None,
    limit: int | None = None,
    use_mlflow: bool = True,
    register: bool = True,
    device: str | None = None,
    backend: str | None = None,
    settings: Settings | None = None,
    app: AppConfig | None = None,
    model_loader: Any = None,
) -> TrainResult:
    """Train one adapter. ``model_loader`` lets tests inject a tiny model and tokenizer.

    ``device`` / ``backend`` override the config (e.g. ``cuda`` + ``qlora`` on a GPU machine).
    """
    import torch

    settings = settings or get_settings()
    app = app or load_config(settings.config_dir)
    cfg = app.adapters[task]
    tcfg = cfg.training
    base = app.base_model
    model_id, revision = base.resolve(use_test_model)
    device = device or base.device
    backend = backend or app.backends.training
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("device 'cuda' requested but no GPU is available")
    if backend == "qlora" and device != "cuda":
        raise ValueError("qlora needs a CUDA GPU; use backend 'lora' on CPU")
    compute = f"cuda:{torch.cuda.get_device_name(0)}" if device == "cuda" else "cpu"

    # ---- data (verified against the manifest before use)
    task_dir = settings.data_dir / "processed" / task
    manifest = verify_dataset(task_dir)
    splits = ["train", *extra_splits]
    train_examples = [ex for s in splits for ex in load_split(task_dir, s)]
    if limit:
        train_examples = train_examples[:limit]
    eval_examples = load_split(task_dir, "eval")[: tcfg.eval_loss_samples]

    seed = cfg.dataset.seed
    random.seed(seed)
    torch.manual_seed(seed)
    if base.num_threads and device == "cpu":
        torch.set_num_threads(base.num_threads)

    # ---- model
    if model_loader is not None:
        tokenizer, model = model_loader()
        model = model.to(device)
    else:
        tokenizer, model = load_base_model(model_id, revision, backend, device, base.dtype)
    pad_id = (
        tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    )

    from peft import LoraConfig, PeftModel, get_peft_model

    if init_adapter_dir is not None:
        model = PeftModel.from_pretrained(model, str(init_adapter_dir), is_trainable=True)
    else:
        model = get_peft_model(
            model,
            LoraConfig(
                r=cfg.lora.r,
                lora_alpha=cfg.lora.alpha,
                lora_dropout=cfg.lora.dropout,
                target_modules=list(cfg.lora.target_modules),
                task_type="CAUSAL_LM",
            ),
        )
    trainable, total_params = model.get_nb_trainable_parameters()
    model.train()

    train_items = [tokenize_example(tokenizer, ex, tcfg.max_seq_len) for ex in train_examples]
    eval_items = [tokenize_example(tokenizer, ex, tcfg.max_seq_len) for ex in eval_examples]

    batches_per_epoch = math.ceil(len(train_items) / tcfg.batch_size)
    total_micro = math.ceil(tcfg.epochs * batches_per_epoch)
    total_steps = math.ceil(total_micro / tcfg.grad_accum_steps)
    if max_steps is not None:
        total_steps = min(total_steps, max_steps)
        total_micro = min(total_micro, total_steps * tcfg.grad_accum_steps)
    warmup = max(1, int(tcfg.warmup_ratio * total_steps))

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=tcfg.learning_rate, weight_decay=tcfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda(warmup, total_steps))
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")

    started = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    run_name = f"{task}-{backend}-{started}"
    out_dir = settings.artifacts_dir / "adapters" / task / run_name
    run_params = {
        "task": task,
        "base_model": model_id,
        "base_revision": revision,
        "dataset_version": manifest["dataset_version"],
        "train_splits": ",".join(splits),
        "n_train": len(train_items),
        "n_eval_loss": len(eval_items),
        "prompt_version": PROMPT_VERSION,
        "backend": backend,
        "device": device,
        "precision": "fp16-mixed" if device == "cuda" else base.dtype,
        "quantization": "nf4-4bit" if backend == "qlora" else "none",
        "lora_r": cfg.lora.r,
        "lora_alpha": cfg.lora.alpha,
        "lora_dropout": cfg.lora.dropout,
        "lora_target_modules": ",".join(cfg.lora.target_modules),
        "learning_rate": tcfg.learning_rate,
        "epochs": tcfg.epochs,
        "batch_size": tcfg.batch_size,
        "grad_accum_steps": tcfg.grad_accum_steps,
        "effective_batch_size": tcfg.batch_size * tcfg.grad_accum_steps,
        "max_seq_len": tcfg.max_seq_len,
        "warmup_steps": warmup,
        "total_steps": total_steps,
        "weight_decay": tcfg.weight_decay,
        "max_grad_norm": tcfg.max_grad_norm,
        "seed": seed,
        "torch_threads": torch.get_num_threads(),
        "init_adapter": str(init_adapter_dir) if init_adapter_dir else "none",
        "trainable_params": trainable,
        "total_params": total_params,
    }
    tags = {
        "task": task,
        "stage": "training",
        "trigger": trigger,
        "dataset_version": manifest["dataset_version"],
        "base_model": model_id,
        "backend": backend,
        "compute": compute,
        "git_commit": git_commit(),
    }

    tracker = _Tracker(use_mlflow)
    run_ctx: Any = contextlib.nullcontext()
    if use_mlflow:
        tracker.mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        tracker.mlflow.set_experiment(f"{app.mlflow.experiment_prefix}-training")
        run_ctx = tracker.mlflow.start_run(run_name=run_name, log_system_metrics=True)

    with run_ctx as run:
        if use_mlflow:
            tracker.mlflow.set_tags(tags)
            tracker.mlflow.log_params(run_params)
        logger.info(
            "%s: %d train examples, %d steps, %.2f%% params trainable (%s@%s, %s on %s)",
            task, len(train_items), total_steps, 100 * trainable / total_params,
            model_id, revision[:8], backend, compute,
        )  # fmt: skip

        initial_eval = eval_loss(model, eval_items, tcfg.batch_size, pad_id, device)
        tracker.log_metrics({"eval_loss": initial_eval}, step=0)
        logger.info("%s: eval loss before training %.4f", task, initial_eval)

        t0 = time.perf_counter()
        step, micro, tokens_seen, peak_rss = 0, 0, 0, _rss_mb()
        window_loss, window_tokens, window_start = 0.0, 0, time.perf_counter()
        epoch = 0
        optimizer.zero_grad()
        while micro < total_micro:
            for batch_items in length_grouped_batches(train_items, tcfg.batch_size, seed + epoch):
                batch = _to(collate(batch_items, pad_id), device)
                with _autocast(device):
                    loss = model(**batch).loss
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"non-finite loss at step {step}")
                scaler.scale(loss / tcfg.grad_accum_steps).backward()
                window_loss += loss.item()
                n_tok = int(batch["attention_mask"].sum())
                tokens_seen += n_tok
                window_tokens += n_tok
                micro += 1

                if micro % tcfg.grad_accum_steps == 0 or micro == total_micro:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(params, tcfg.max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    optimizer.zero_grad()
                    step += 1
                    elapsed = time.perf_counter() - window_start
                    n_micro = (
                        tcfg.grad_accum_steps
                        if micro % tcfg.grad_accum_steps == 0
                        else (micro % tcfg.grad_accum_steps)
                    )
                    peak_rss = max(peak_rss, _rss_mb())
                    tracker.log_metrics(
                        {
                            "train_loss": window_loss / n_micro,
                            "learning_rate": scheduler.get_last_lr()[0],
                            "tokens_per_sec": window_tokens / max(elapsed, 1e-9),
                        },
                        step=step,
                    )
                    if step == 1 or step % 10 == 0 or step == total_steps:
                        spent = time.perf_counter() - t0
                        eta = spent / step * (total_steps - step)
                        logger.info(
                            "%s: step %d/%d loss %.4f | %.1f s/step | ETA %.1f min",
                            task, step, total_steps, window_loss / n_micro,
                            spent / step, eta / 60,
                        )  # fmt: skip
                    window_loss, window_tokens, window_start = 0.0, 0, time.perf_counter()

                if micro >= total_micro:
                    break
            epoch += 1

        train_time = time.perf_counter() - t0
        final_eval = eval_loss(model, eval_items, tcfg.batch_size, pad_id, device)
        metrics = {
            "eval_loss": final_eval,
            "eval_loss_initial": initial_eval,
            "eval_loss_improvement": initial_eval - final_eval,
            "train_time_sec": train_time,
            "train_tokens": tokens_seen,
            "peak_rss_mb": peak_rss,
        }
        if device == "cuda":
            metrics["peak_gpu_mem_mb"] = torch.cuda.max_memory_allocated() / 2**20
        tracker.log_metrics(metrics, step=step)
        logger.info(
            "%s: done in %.1f min, eval loss %.4f -> %.4f",
            task, train_time / 60, initial_eval, final_eval,
        )  # fmt: skip

        # ---- save adapter + metadata needed to reproduce / serve it
        adapter_dir = out_dir / ADAPTER_ARTIFACT_PATH
        if adapter_dir.exists():
            shutil.rmtree(adapter_dir)
        model.save_pretrained(str(adapter_dir))
        info = {**run_params, "trigger": trigger, "metrics": metrics}
        (adapter_dir / "adaptos_info.json").write_text(
            json.dumps(info, indent=2, default=str), encoding="utf-8"
        )

        result = TrainResult(task=task, adapter_dir=adapter_dir, metrics=metrics, params=run_params)
        if use_mlflow:
            mlflow = tracker.mlflow
            mlflow.log_artifacts(str(adapter_dir), artifact_path=ADAPTER_ARTIFACT_PATH)
            mlflow.log_artifact(str(task_dir / "manifest.json"), artifact_path="dataset")
            mlflow.log_dict(cfg.model_dump(), "config/adapter.json")
            mlflow.log_dict(app.base_model.model_dump(), "config/base_model.json")
            result.run_id = run.info.run_id
            if register:
                ensure_registered_model(
                    cfg.registered_model,
                    description=f"LoRA adapter for '{task}': {cfg.description}",
                    tracking_uri=settings.mlflow_tracking_uri,
                )
                mv = register_adapter_version(
                    cfg.registered_model,
                    run_id=run.info.run_id,
                    source_uri=mlflow.get_artifact_uri(ADAPTER_ARTIFACT_PATH),
                    tags={
                        "task": task,
                        "dataset_version": manifest["dataset_version"],
                        "base_model": model_id,
                        "base_revision": revision,
                        "prompt_version": PROMPT_VERSION,
                        "eval_loss": f"{final_eval:.4f}",
                        "trigger": trigger,
                        "backend": backend,
                        "compute": compute,
                    },
                    description=(
                        f"Trained on {len(train_items)} examples ({'+'.join(splits)}), "
                        f"dataset {manifest['dataset_version']}, trigger: {trigger}"
                    ),
                    tracking_uri=settings.mlflow_tracking_uri,
                )
                result.model_version = mv.version
                mlflow.set_tag("registered_version", mv.version)
                logger.info("%s: registered %s version %s", task, cfg.registered_model, mv.version)
    return result


def main(argv: Sequence[str] | None = None) -> None:
    settings = get_settings()
    configure_logging(settings.log_level)
    adapters = sorted(load_config(settings.config_dir).adapters)
    p = argparse.ArgumentParser(description="Train a LoRA adapter and register it in MLflow")
    p.add_argument("--adapter", choices=[*adapters, "all"], required=True)
    p.add_argument("--test-model", action="store_true", help="use the small test base model")
    p.add_argument("--max-steps", type=int, help="stop after N optimizer steps (smoke runs)")
    p.add_argument("--limit", type=int, help="use only the first N training examples")
    p.add_argument("--no-mlflow", action="store_true")
    p.add_argument("--no-register", action="store_true")
    p.add_argument("--device", choices=["cpu", "cuda"], help="override base_model.device")
    p.add_argument("--backend", choices=["lora", "qlora"], help="override backends.training")
    args = p.parse_args(argv)
    for task in adapters if args.adapter == "all" else [args.adapter]:
        train_adapter(
            task,
            use_test_model=args.test_model,
            max_steps=args.max_steps,
            limit=args.limit,
            use_mlflow=not args.no_mlflow,
            register=not args.no_register,
            device=args.device,
            backend=args.backend,
            settings=settings,
        )


if __name__ == "__main__":
    main()
