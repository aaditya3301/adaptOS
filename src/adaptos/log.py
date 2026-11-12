"""Logging setup shared by every AdaptOS service and pipeline."""

from __future__ import annotations

import logging

_FORMAT = "%(asctime)s %(levelname)s [%(name)s] %(message)s"


# Third-party libraries that log every HTTP request at INFO level.
_NOISY = ("httpx", "httpx2", "httpcore", "urllib3", "huggingface_hub", "datasets", "filelock")


def configure_logging(level: str = "INFO") -> None:
    logging.basicConfig(level=level.upper(), format=_FORMAT, force=True)
    for name in _NOISY:
        logging.getLogger(name).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
