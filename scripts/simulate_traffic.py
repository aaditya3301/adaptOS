"""Send realistic traffic to the gateway, taken from a processed dataset split.

    python scripts/simulate_traffic.py --task sql --split golden --n 30      # normal traffic
    python scripts/simulate_traffic.py --task sql --split drift_eval --n 30  # the new kind

``golden`` requests look like the adapter's training data; ``drift_eval`` requests are the
held-out new kind (e.g. aggregation SQL) that the live adapter has never seen.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from adaptos.data.io import load_split  # noqa: E402


def post(url: str, body: dict) -> dict:
    req = urllib.request.Request(
        url, json.dumps(body).encode(), {"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.load(r)


def get(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=60) as r:
        return json.load(r)


def simulate(
    task: str,
    split: str,
    n: int,
    gateway: str = "http://localhost:8000",
    seed: int = 0,
    verbose: bool = True,
) -> dict[str, Any]:
    """Send ``n`` requests from ``split``; return a summary plus the monitor's view."""
    examples = load_split(ROOT / "data" / "processed" / task, split)
    random.Random(seed).shuffle(examples)
    qualities, routed = [], {}
    start = time.time()
    for i, ex in enumerate(examples[:n], 1):
        r = post(f"{gateway}/chat", {"message": ex.input})
        routed[r["adapter"]] = routed.get(r["adapter"], 0) + 1
        if r["quality"] is not None:
            qualities.append(r["quality"])
        if verbose:
            quality = "-" if r["quality"] is None else f"{r['quality']:.2f}"
            print(
                f"  [{i:3d}/{n}] {r['adapter']:>9} v{r['adapter_version'] or '-'} "
                f"quality {quality:<5} {r['latency_ms'] / 1000:5.1f}s  {r['answer'][:60]!r}",
                flush=True,
            )
    monitor = get(f"{gateway}/monitoring")["adapters"][task]
    return {
        "requests": n,
        "seconds": round(time.time() - start),
        "routed": routed,
        "mean_quality": statistics.mean(qualities) if qualities else None,
        "monitor": monitor,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--task", required=True, choices=["sql", "summarize", "math"])
    p.add_argument("--split", default="golden", choices=["golden", "eval", "drift_eval"])
    p.add_argument("--n", type=int, default=30)
    p.add_argument("--gateway", default="http://localhost:8000")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    # Model answers can contain any character; don't crash on consoles like Windows cp1252.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    s = simulate(args.task, args.split, args.n, args.gateway, args.seed)
    m = s["monitor"]
    print(
        f"\n{s['requests']} {args.split} requests in {s['seconds']}s | routed: {s['routed']} "
        f"| mean quality {s['mean_quality']}"
    )
    print(
        f"monitor[{args.task}]: status={m['status']} samples={m['samples']} "
        f"quality={m['quality']} drift={m['drift']} reasons={m['reasons']}"
    )


if __name__ == "__main__":
    main()
