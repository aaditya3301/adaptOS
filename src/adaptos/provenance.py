"""Provenance helpers recorded with every dataset version and training run."""

from __future__ import annotations

import os
import subprocess


def git_commit() -> str:
    """Current commit SHA, suffixed with '-dirty' when there are uncommitted changes.

    ``ADAPTOS_GIT_COMMIT`` overrides it when code is shipped without its .git folder
    (e.g. copied to a remote GPU machine).
    """
    if override := os.environ.get("ADAPTOS_GIT_COMMIT"):
        return override
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
