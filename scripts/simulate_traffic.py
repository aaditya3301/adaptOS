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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from adaptos.data.io import load_split  # noqa: E402


def post(url: str, body: dict) -> dict:
    req = urllib.request.Request(
        url, json.dumps(body).encode(), {"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.load(r)


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

    examples = load_split(ROOT / "data" / "processed" / args.task, args.split)
    random.Random(args.seed).shuffle(examples)
    qualities, routed = [], {}
    start = time.time()
    for i, ex in enumerate(examples[: args.n], 1):
        r = post(f"{args.gateway}/chat", {"message": ex.input})
        routed[r["adapter"]] = routed.get(r["adapter"], 0) + 1
        if r["quality"] is not None:
            qualities.append(r["quality"])
        print(
            f"[{i:3d}/{args.n}] {r['adapter']:>9} v{r['adapter_version'] or '-'} "
            f"quality {r['quality'] if r['quality'] is not None else '-':<5} "
            f"{r['latency_ms'] / 1000:5.1f}s  {r['answer'][:70]!r}",
            flush=True,
        )
    mean_q = statistics.mean(qualities) if qualities else float("nan")
    print(
        f"\n{args.n} {args.split} requests in {time.time() - start:.0f}s | routed: {routed} "
        f"| mean quality {mean_q:.3f}"
    )
    status = json.load(urllib.request.urlopen(f"{args.gateway}/monitoring", timeout=60))
    s = status["adapters"][args.task]
    print(
        f"monitor[{args.task}]: status={s['status']} samples={s['samples']} "
        f"quality={s['quality']} drift={s['drift']} reasons={s['reasons']}"
    )


if __name__ == "__main__":
    main()
