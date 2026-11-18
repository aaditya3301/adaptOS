"""MLflow Model Registry helpers for LoRA adapters.

Each adapter task is one registered model (e.g. ``adaptos-adapter-sql``). Every training run
adds a new version whose source is the run's ``adapter/`` artifact directory. Aliases
(``champion`` / ``challenger``) are managed by the evaluation gate in Phase 3.
"""

from __future__ import annotations

import contextlib
from pathlib import Path
from typing import Any

ADAPTER_ARTIFACT_PATH = "adapter"
CHAMPION = "champion"  # version served in production
CHALLENGER = "challenger"  # version currently being evaluated
PREVIOUS = "previous"  # last champion, kept for one-step rollback


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


def latest_version(name: str, tracking_uri: str | None = None):
    versions = _client(tracking_uri).search_model_versions(f"name = '{name}'")
    if not versions:
        raise LookupError(f"no versions registered for '{name}'")
    return max(versions, key=lambda v: int(v.version))


def get_alias_version(name: str, alias: str, tracking_uri: str | None = None):
    """The version an alias points to, or None if the alias is not set."""
    from mlflow.exceptions import MlflowException

    try:
        return _client(tracking_uri).get_model_version_by_alias(name, alias)
    except MlflowException:
        return None


def set_alias(name: str, alias: str, version: str, tracking_uri: str | None = None) -> None:
    _client(tracking_uri).set_registered_model_alias(name, alias, str(version))


def delete_alias(name: str, alias: str, tracking_uri: str | None = None) -> None:
    from mlflow.exceptions import MlflowException

    with contextlib.suppress(MlflowException):
        _client(tracking_uri).delete_registered_model_alias(name, alias)


def tag_version(
    name: str, version: str, tags: dict[str, Any], tracking_uri: str | None = None
) -> None:
    client = _client(tracking_uri)
    for key, value in tags.items():
        client.set_model_version_tag(name, str(version), key, str(value))


def promote(name: str, version: str, tracking_uri: str | None = None) -> str | None:
    """Make ``version`` the champion; the old champion becomes 'previous'. Returns old version."""
    old = get_alias_version(name, CHAMPION, tracking_uri)
    if old is not None and old.version != str(version):
        set_alias(name, PREVIOUS, old.version, tracking_uri)
    set_alias(name, CHAMPION, version, tracking_uri)
    return old.version if old is not None else None


def rollback(name: str, tracking_uri: str | None = None) -> tuple[str, str]:
    """Swap champion and previous. Returns (new champion, demoted version)."""
    champion = get_alias_version(name, CHAMPION, tracking_uri)
    previous = get_alias_version(name, PREVIOUS, tracking_uri)
    if champion is None or previous is None:
        raise LookupError(f"'{name}' needs both a champion and a previous version to roll back")
    set_alias(name, CHAMPION, previous.version, tracking_uri)
    set_alias(name, PREVIOUS, champion.version, tracking_uri)
    tag_version(name, champion.version, {"rolled_back": "true"}, tracking_uri)
    return previous.version, champion.version


def cached_download(cache_root: Path, name: str, version: str, tracking_uri: str) -> Path:
    """Download a registered version once into ``cache_root`` and reuse it afterwards.

    Returns the directory that holds the model files (adapter_config.json or router.json).
    """
    dst = cache_root / name / f"v{version}"
    for marker in ("adapter_config.json", "router.json"):
        found = list(dst.rglob(marker))
        if found:
            return found[0].parent
    return download_adapter(name, str(version), dst, tracking_uri)


def download_adapter(name: str, ref: str, dst: Path, tracking_uri: str | None = None) -> Path:
    """Download an adapter version's files into ``dst``; returns the adapter directory."""
    import mlflow

    version = get_version(name, ref, tracking_uri)
    dst.mkdir(parents=True, exist_ok=True)
    path = mlflow.artifacts.download_artifacts(
        artifact_uri=version.source, dst_path=str(dst), tracking_uri=tracking_uri
    )
    return Path(path)
