"""MLflow Model Registry helpers for LoRA adapters.

Each adapter task is one registered model (e.g. ``adaptos-adapter-sql``). Every training run
adds a new version whose source is the run's ``adapter/`` artifact directory. Aliases
(``champion`` / ``challenger``) are managed by the evaluation gate in Phase 3.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

ADAPTER_ARTIFACT_PATH = "adapter"
CHAMPION = "champion"
CHALLENGER = "challenger"


def _client(tracking_uri: str | None = None):
    import mlflow

    return mlflow.MlflowClient(tracking_uri=tracking_uri)


def ensure_registered_model(name: str, description: str, tracking_uri: str | None = None) -> None:
    from mlflow.exceptions import MlflowException

    client = _client(tracking_uri)
    try:
        client.get_registered_model(name)
    except MlflowException:
        client.create_registered_model(name, description=description)


def register_adapter_version(
    name: str,
    run_id: str,
    source_uri: str,
    tags: dict[str, Any],
    description: str,
    tracking_uri: str | None = None,
):
    """Create a new model version pointing at a run's adapter artifacts."""
    client = _client(tracking_uri)
    return client.create_model_version(
        name=name,
        source=source_uri,
        run_id=run_id,
        tags={k: str(v) for k, v in tags.items()},
        description=description,
    )


def get_version(name: str, ref: str, tracking_uri: str | None = None):
    """Resolve ``ref`` (a version number or an alias such as 'champion') to a model version."""
    client = _client(tracking_uri)
    if ref.isdigit():
        return client.get_model_version(name, ref)
    return client.get_model_version_by_alias(name, ref)


def download_adapter(name: str, ref: str, dst: Path, tracking_uri: str | None = None) -> Path:
    """Download an adapter version's files into ``dst``; returns the adapter directory."""
    import mlflow

    version = get_version(name, ref, tracking_uri)
    dst.mkdir(parents=True, exist_ok=True)
    path = mlflow.artifacts.download_artifacts(
        artifact_uri=version.source, dst_path=str(dst), tracking_uri=tracking_uri
    )
    return Path(path)
