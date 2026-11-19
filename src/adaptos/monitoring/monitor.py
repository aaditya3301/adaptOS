"""Rolling-window health of each live adapter: quality, drift and alert status.

For every adapter the monitor looks at the last ``window_size`` requests served by the
*currently live* version (older versions' traffic does not count against a new version).

    quality = mean over the window of: 1/0 for thumbs up/down, else the online quality score
    drift   = feature-drift score of the window's inputs vs the version's training reference
    alert   = window has >= min_samples requests and (quality < quality_min or drift > drift_max)

The retraining controller (Phase 7) acts on these alerts.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

from adaptos.config import AdapterThresholds, MonitorConfig
from adaptos.monitoring.drift import drift_score


@dataclass
class WindowStats:
    adapter: str
    version: str | None
    samples: int
    quality: float | None
    drift: float | None
    drift_features: dict[str, float] = field(default_factory=dict)
    feedback_up: int = 0
    feedback_down: int = 0
    status: str = "ok"  # "ok" | "alert" | "insufficient_data" | "no_reference"
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def effective_quality(row: dict[str, Any]) -> float | None:
    """User feedback overrides the automatic score."""
    if row.get("feedback") == "up":
        return 1.0
    if row.get("feedback") == "down":
        return 0.0
    return row.get("quality_score")


def evaluate_window(
    adapter: str,
    version: str | None,
    rows: Sequence[dict[str, Any]],
    reference: dict[str, Any] | None,
    thresholds: AdapterThresholds,
    min_samples: int,
) -> WindowStats:
    scores = [q for r in rows if (q := effective_quality(r)) is not None]
    stats = WindowStats(
        adapter=adapter,
        version=version,
        samples=len(rows),
        quality=sum(scores) / len(scores) if scores else None,
        drift=None,
        feedback_up=sum(r.get("feedback") == "up" for r in rows),
        feedback_down=sum(r.get("feedback") == "down" for r in rows),
    )
    if reference is not None and rows:
        stats.drift, stats.drift_features = drift_score(
            adapter, [r["input"] for r in rows], reference
        )
    if len(rows) < min_samples:
        stats.status = "insufficient_data"
        return stats
    if stats.quality is not None and stats.quality < thresholds.quality_min:
        stats.reasons.append(f"quality {stats.quality:.3f} < {thresholds.quality_min}")
    if stats.drift is not None and stats.drift > thresholds.drift_max:
        worst = max(stats.drift_features, key=lambda k: abs(stats.drift_features[k]))
        stats.reasons.append(f"input drift {stats.drift:.2f} > {thresholds.drift_max} ({worst})")
    if stats.reasons:
        stats.status = "alert"
    elif reference is None:
        stats.status = "no_reference"
    return stats


class Monitor:
    """Computes window stats for every live adapter from the request log."""

    def __init__(
        self,
        config: MonitorConfig,
        recent: Callable[[str, str | None, int], list[dict[str, Any]]],
        live: Callable[[], dict[str, str]],
        reference: Callable[[str], dict[str, Any] | None],
    ):
        self.config = config
        self.recent = recent  # (adapter, version, limit) -> newest rows
        self.live = live  # adapter -> live version
        self.reference = reference  # adapter -> drift reference of the live version

    def check(self) -> dict[str, WindowStats]:
        live = self.live()
        out = {}
        for adapter, thresholds in self.config.thresholds.items():
            version = live.get(adapter)
            rows = self.recent(adapter, version, self.config.window_size) if version else []
            out[adapter] = evaluate_window(
                adapter,
                version,
                rows,
                self.reference(adapter) if version else None,
                thresholds,
                self.config.min_samples,
            )
        return out
