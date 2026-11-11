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
    mlflow_tracking_uri: str = "http://localhost:5000"
    database_url: str = "postgresql://adaptos:adaptos@localhost:5432/adaptos"
    log_level: str = "INFO"


class LLMConfig(BaseModel):
    name: str
    test_name: str
    device: Literal["cpu", "cuda"] = "cpu"
    dtype: Literal["float32", "bfloat16", "float16"] = "float32"


class BackendConfig(BaseModel):
    training: Literal["lora", "qlora"] = "lora"
    inference: Literal["peft", "vllm"] = "peft"


class MLflowConfig(BaseModel):
    experiment_prefix: str = "adaptos"


class RouterConfig(BaseModel):
    embedding_model: str
    registered_model: str
    confidence_threshold: float = Field(ge=0.0, le=1.0)


class DatasetConfig(BaseModel):
    hf_id: str
    hf_config: str | None = None
    train_size: int = Field(gt=0)
    eval_size: int = Field(gt=0)
    golden_size: int = Field(gt=0)
    seed: int = 42


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


class EvaluationConfig(BaseModel):
    primary_metric: str
    min_score: float = Field(ge=0.0, le=1.0)
    promotion_min_delta: float = Field(ge=0.0)


class AdapterConfig(BaseModel):
    name: str
    description: str
    registered_model: str
    dataset: DatasetConfig
    lora: LoraConfig
    training: TrainingConfig
    evaluation: EvaluationConfig


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
