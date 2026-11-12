"""Cross-platform task runner (a Makefile replacement that also works on Windows).

Usage:  python tasks.py <task>
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = sys.executable
# UTF-8 mode: MLflow prints emoji that crash the default Windows console encoding.
ENV = {**os.environ, "PYTHONUTF8": "1", "MLFLOW_DISABLE_AGENT_HINT": "1"}

TASKS: dict[str, tuple[str, list[list[str]]]] = {
    "install": (
        "Install the project with dev tools",
        [[PY, "-m", "pip", "install", "-e", ".[dev,data]"]],
    ),
    "data": (
        "Build all datasets and register them in MLflow",
        [[PY, "-m", "adaptos.data.pipeline", "--task", "all"]],
    ),
    "lint": (
        "Run ruff lint + format check",
        [
            [PY, "-m", "ruff", "check", "."],
            [PY, "-m", "ruff", "format", "--check", "."],
        ],
    ),
    "format": (
        "Auto-fix lint issues and format code",
        [
            [PY, "-m", "ruff", "check", "--fix", "."],
            [PY, "-m", "ruff", "format", "."],
        ],
    ),
    "test": ("Run the test suite", [[PY, "-m", "pytest"]]),
    "up": ("Build and start the Docker stack", [["docker", "compose", "up", "-d", "--build"]]),
    "down": ("Stop the Docker stack (keeps data volumes)", [["docker", "compose", "down"]]),
    "reset": ("Stop the stack and DELETE all data volumes", [["docker", "compose", "down", "-v"]]),
    "ps": ("Show stack status", [["docker", "compose", "ps"]]),
    "logs": ("Follow logs of all services", [["docker", "compose", "logs", "-f", "--tail=100"]]),
    "gateway": (
        "Run the gateway locally with auto-reload",
        [
            [
                PY,
                "-m",
                "uvicorn",
                "adaptos.serving.app:create_app",
                "--factory",
                "--reload",
                "--port",
                "8000",
            ]
        ],
    ),
}


def main() -> int:
    if len(sys.argv) != 2 or sys.argv[1] not in TASKS:
        print("Usage: python tasks.py <task>\n")
        for name, (desc, _) in TASKS.items():
            print(f"  {name:<10} {desc}")
        return 1
    for cmd in TASKS[sys.argv[1]][1]:
        print("$", " ".join(cmd), flush=True)
        code = subprocess.call(cmd, cwd=ROOT, env=ENV)
        if code != 0:
            return code
    return 0


if __name__ == "__main__":
    sys.exit(main())
