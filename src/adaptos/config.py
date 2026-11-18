"""Typed configuration for AdaptOS.

Two sources:
- ``Settings``: runtime/environment values (service URLs, secrets) read from ``ADAPTOS_*`` env vars.
- ``AppConfig``: versioned YAML files in ``configs/`` (models, adapters, thresholds).

Keeping them apart means the YAML can be committed and logged to MLflow with every run,
while environment-specific values never end up in git.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="ADAPTOS_", env_file=".env", extra="ignore")

    config_dir: Path = REPO_ROOT / "configs"
    data_dir: Path = REPO_ROOT / "data"
    artifacts_dir: Path = REPO_ROOT / "artifacts"
    mlflow_tracking_uri: str = "http://localhost:5000"
    database_url: str = "postgresql://adaptos:adaptos@localhost:5432/adaptos"
    log_level: str = "INFO"
    admin_token: str = ""  # required in the X-Admin-Token header for /admin/*; empty = disabled


class LLMConfig(BaseModel):
    name: str
    revision: str = Field(min_length=7)
    test_name: str
    test_revision: str = Field(min_length=7)
    device: Literal["cpu", "cuda"] = "cpu"
    dtype: Literal["float32", "bfloat16", "float16"] = "float32"
    num_threads: int | None = Field(default=None, gt=0)

    def resolve(self, use_test_model: bool = False) -> tuple[str, str]:
        """(model id, pinned revision) for the main or the small test model."""
        return (
            (self.test_name, self.test_revision) if use_test_model else (self.name, self.revision)
        )


class BackendConfig(BaseModel):
    training: Literal["lora", "qlora"] = "lora"
    inference: Literal["peft", "vllm"] = "peft"


class MLflowConfig(BaseModel):
    experiment_prefix: str = "adaptos"


class ServingConfig(BaseModel):
    dtype: Literal["float32", "bfloat16"] = "bfloat16"
    num_threads: int | None = Field(default=None, gt=0)
    base_max_new_tokens: int = Field(default=256, gt=0)
    max_input_chars: int = Field(default=8000, gt=0)


class OtherDatasetConfig(BaseModel):
    hf_id: str
    revision: str = Field(min_length=7)
    exclude_categories: list[str] = Field(default_factory=list)
    train_size: int = Field(gt=0)
    eval_size: int = Field(gt=0)
    test_size: int = Field(gt=0)
    seed: int = 42


class RouterConfig(BaseModel):
    embedding_model: str
    embedding_revision: str = Field(min_length=7)
    registered_model: str
    confidence_threshold: float = Field(ge=0.0, le=1.0)
    min_accuracy: float = Field(default=0.95, ge=0.0, le=1.0)
    strip_instruction_prob: float = Field(default=0.5, ge=0.0, le=1.0)
    other_dataset: OtherDatasetConfig


class DriftConfig(BaseModel):
    """A held-out slice of the data that the adapter never trains on at first.

    It plays the role of a "new kind" of user request in the retraining demo:
    ``eval_size`` examples measure the degradation, ``train_size`` examples are the
    fresh labeled data the retraining pipeline gets.
    """

    rule: str
    params: dict[str, Any] = Field(default_factory=dict)
    train_size: int = Field(gt=0)
    eval_size: int = Field(gt=0)


class DatasetConfig(BaseModel):
    hf_id: str
    hf_config: str | None = None
    revision: str = Field(min_length=7)  # pinned dataset commit, for reproducibility
    source_split: str = "train"
    train_size: int = Field(gt=0)
    eval_size: int = Field(gt=0)
    golden_size: int = Field(gt=0)
    seed: int = 42
    drift: DriftConfig


class LoraConfig(BaseModel):
    r: int = Field(gt=0)
    alpha: int = Field(gt=0)
    dropout: float = Field(ge=0.0, lt=1.0)
    target_modules: list[str] = Field(min_length=1)


class TrainingConfig(BaseModel):
    epochs: float = Field(gt=0)
    learning_rate: float = Field(gt=0)
    batch_size: int = Field(gt=0)
    grad_accum_steps: int = Field(gt=0)
    max_seq_len: int = Field(gt=0)
    warmup_ratio: float = Field(default=0.05, ge=0.0, lt=1.0)
    weight_decay: float = Field(default=0.0, ge=0.0)
    max_grad_norm: float = Field(default=1.0, gt=0.0)
    eval_loss_samples: int = Field(default=50, gt=0)  # eval-loss subset (full metrics: Phase 3)


class EvaluationConfig(BaseModel):
    """Promotion gate. A candidate becomes champion only if it:
    - reaches ``min_score`` on the golden set,
    - does not drop more than ``max_regression`` below the champion on the golden set, and
    - beats the champion by ``promotion_min_delta`` on the golden set OR on the drift set.
    """

    primary_metric: str
    min_score: float = Field(ge=0.0, le=1.0)
    promotion_min_delta: float = Field(ge=0.0)
    max_regression: float = Field(default=0.02, ge=0.0)


class GenerationConfig(BaseModel):
    max_new_tokens: int = Field(default=128, gt=0)


class AdapterConfig(BaseModel):
    name: str
    description: str
    registered_model: str
    dataset: DatasetConfig
    lora: LoraConfig
    training: TrainingConfig
    evaluation: EvaluationConfig
    generation: GenerationConfig = Field(default_factory=GenerationConfig)


class AdapterThresholds(BaseModel):
    quality_min: float = Field(ge=0.0, le=1.0)
    drift_max: float = Field(ge=0.0)


class MonitorConfig(BaseModel):
    check_interval_seconds: int = Field(gt=0)
    window_size: int = Field(gt=0)
    min_samples: int = Field(gt=0)
    retrain_cooldown_minutes: int = Field(ge=0)
    thresholds: dict[str, AdapterThresholds]

    @model_validator(mode="after")
    def _min_samples_fit_window(self) -> MonitorConfig:
        if self.min_samples > self.window_size:
            raise ValueError("monitor.min_samples cannot exceed monitor.window_size")
        return self


class AppConfig(BaseModel):
    project: str
    base_model: LLMConfig
    backends: BackendConfig
    mlflow: MLflowConfig
    serving: ServingConfig = Field(default_factory=ServingConfig)
    router: RouterConfig
    adapters: dict[str, AdapterConfig]
    monitor: MonitorConfig

    @model_validator(mode="after")
    def _cross_checks(self) -> AppConfig:
        if not self.adapters:
            raise ValueError("at least one adapter config is required in configs/adapters/")
        missing = set(self.adapters) - set(self.monitor.thresholds)
        unknown = set(self.monitor.thresholds) - set(self.adapters)
        if missing:
            raise ValueError(f"monitor thresholds missing for adapters: {sorted(missing)}")
        if unknown:
            raise ValueError(f"monitor thresholds defined for unknown adapters: {sorted(unknown)}")
        names = [a.registered_model for a in self.adapters.values()] + [
            self.router.registered_model
        ]
        if len(names) != len(set(names)):
            raise ValueError("registered model names must be unique across adapters and router")
        return self


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return data


def load_config(config_dir: Path | None = None) -> AppConfig:
    """Load and validate base.yaml + adapters/*.yaml + monitor.yaml."""
    config_dir = Path(config_dir or get_settings().config_dir)
    adapters: dict[str, AdapterConfig] = {}
    for path in sorted((config_dir / "adapters").glob("*.yaml")):
        adapter = AdapterConfig.model_validate(_read_yaml(path))
        if adapter.name != path.stem:
            raise ValueError(f"{path.name}: adapter name '{adapter.name}' must match file name")
        adapters[adapter.name] = adapter
    return AppConfig.model_validate(
        {
            **_read_yaml(config_dir / "base.yaml"),
            "adapters": adapters,
            "monitor": _read_yaml(config_dir / "monitor.yaml"),
        }
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
