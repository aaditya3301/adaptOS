"""Reading/writing processed datasets and checking their integrity."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from pathlib import Path

from adaptos.data.schema import Example

MANIFEST = "manifest.json"


def write_jsonl(path: Path, examples: Iterable[Example]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # newline="\n" keeps files byte-identical on Windows and Linux, so hashes match.
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for ex in examples:
            f.write(json.dumps(ex.model_dump(), ensure_ascii=False, sort_keys=True) + "\n")


def read_jsonl(path: Path) -> list[Example]:
    with path.open(encoding="utf-8") as f:
        return [Example.model_validate_json(line) for line in f if line.strip()]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def read_manifest(task_dir: Path) -> dict:
    return json.loads((task_dir / MANIFEST).read_text(encoding="utf-8"))


def verify_dataset(task_dir: Path) -> dict:
    """Check every split file against the manifest. Returns the manifest if all match."""
    manifest = read_manifest(task_dir)
    for name, info in manifest["splits"].items():
        actual = sha256_file(task_dir / info["file"])
        if actual != info["sha256"]:
            raise ValueError(
                f"{task_dir.name}/{info['file']} does not match its manifest "
                f"(dataset {manifest['dataset_version']}); split '{name}' was modified"
            )
    return manifest


def load_split(task_dir: Path, split: str) -> list[Example]:
    """Load one split after verifying the dataset has not been modified."""
    manifest = verify_dataset(task_dir)
    if split not in manifest["splits"]:
        raise KeyError(f"unknown split '{split}'; available: {sorted(manifest['splits'])}")
    return read_jsonl(task_dir / manifest["splits"][split]["file"])
