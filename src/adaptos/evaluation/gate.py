"""Champion/challenger promotion gate (pure logic, no I/O)."""

from __future__ import annotations

from dataclasses import dataclass

from adaptos.config import EvaluationConfig


@dataclass(frozen=True)
class GateDecision:
    promote: bool
    reason: str


def decide(
    cfg: EvaluationConfig,
    candidate: dict[str, float],
    champion: dict[str, float] | None,
) -> GateDecision:
    """Decide whether a candidate replaces the champion.

    ``candidate`` / ``champion`` map split name ('golden', 'drift_eval') to the primary metric.
    Rules, in order:
      1. golden score must reach the absolute floor ``min_score``
      2. with no champion yet, passing the floor is enough
      3. golden score may not regress by more than ``max_regression``
      4. it must improve golden OR drift_eval by at least ``promotion_min_delta``
    """
    m = cfg.primary_metric
    golden = candidate["golden"]
    if golden < cfg.min_score:
        return GateDecision(False, f"golden {m} {golden:.3f} is below the minimum {cfg.min_score}")
    if champion is None:
        return GateDecision(True, f"first version: golden {m} {golden:.3f} >= {cfg.min_score}")

    champ_golden = champion["golden"]
    if golden < champ_golden - cfg.max_regression:
        return GateDecision(
            False,
            f"golden {m} regressed {champ_golden:.3f} -> {golden:.3f} "
            f"(more than the allowed {cfg.max_regression})",
        )
    gains = []
    for split in ("golden", "drift_eval"):
        if split in candidate and split in champion:
            delta = candidate[split] - champion[split]
            if delta >= cfg.promotion_min_delta:
                gains.append(f"{split} {champion[split]:.3f} -> {candidate[split]:.3f}")
    if gains:
        return GateDecision(True, "improves " + "; ".join(gains))
    return GateDecision(
        False, f"no {m} improvement of at least {cfg.promotion_min_delta} over the champion"
    )
