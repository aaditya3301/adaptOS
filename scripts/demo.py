"""Guided end-to-end demo of AdaptOS on the running stack.

    python scripts/demo.py                    # full demo, pausing between steps
    python scripts/demo.py --yes              # no pauses between steps
    python scripts/demo.py --no-retrain       # stop at the drift alert

Story: the switchboard routes requests to the right adapter -> normal traffic is healthy ->
a new kind of request arrives (drift) -> the monitor raises an alert -> the controller
retrains, evaluates and gates a new version -> the gateway hot-swaps it -> the same traffic is
healthy again. Needs the stack running (python tasks.py up) and the models from Phases 1-4.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import textwrap
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from simulate_traffic import get, post, simulate  # noqa: E402

from adaptos.config import get_settings, load_config  # noqa: E402
from adaptos.registry import adapters as registry  # noqa: E402

SAMPLES = [
    "CREATE TABLE employees (name VARCHAR, dept VARCHAR, salary INTEGER)\n\n"
    "Question: Which employees in the sales department earn more than 50000?",
    "Summarize this conversation:\n\n#Person1#: Did you finish the report?\n"
    "#Person2#: Not yet, the data came in late.\n#Person1#: Please send a draft by Friday.",
    "A bakery sells 12 muffins per tray. They bake 7 trays and sell all but 9 muffins. "
    "How many muffins did they sell?",
    "What is the capital of Australia?",
]


class Demo:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.settings = get_settings()
        self.app = load_config(self.settings.config_dir)
        self.gw = self.app.retraining.gateway_url
        self.uri = self.settings.mlflow_tracking_uri
        self.model_name = self.app.adapters[args.task].registered_model
        self.step_no = 0

    # ---------------------------------------------------------------- helpers
    def step(self, title: str, explain: str) -> None:
        self.step_no += 1
        print(f"\n{'=' * 78}\nSTEP {self.step_no}: {title}\n{'=' * 78}")
        print(textwrap.fill(explain, 78))
        if not self.args.yes:
            input("\n  [Enter] to run this step ")

    def monitor_line(self, m: dict) -> str:
        q = "-" if m["quality"] is None else f"{m['quality']:.3f}"
        d = "-" if m["drift"] is None else f"{m['drift']:.2f}"
        reasons = f"  <- {'; '.join(m['reasons'])}" if m["reasons"] else ""
        return (
            f"  monitor[{self.args.task}] v{m['version']}: {m['status'].upper()} | "
            f"window {m['samples']} requests | quality {q} | drift {d}{reasons}"
        )

    def admin_reload(self, model: str, force: bool = False) -> dict:
        token = self.settings.admin_token or "dev-admin-token"  # docker-compose default
        req = urllib.request.Request(
            f"{self.gw}/admin/reload",
            json.dumps({"model": model, "force": force}).encode(),
            {"Content-Type": "application/json", "X-Admin-Token": token},
        )
        with urllib.request.urlopen(req, timeout=300) as resp:
            return json.load(resp)

    def aliases(self) -> str:
        champ = registry.get_alias_version(self.model_name, registry.CHAMPION, self.uri)
        prev = registry.get_alias_version(self.model_name, registry.PREVIOUS, self.uri)
        return (
            f"champion=v{champ.version if champ else '-'} previous=v{prev.version if prev else '-'}"
        )

    # ------------------------------------------------------------------ steps
    def run(self) -> None:
        task = self.args.task
        ready = get(f"{self.gw}/ready")
        if ready.get("status") != "ready":
            sys.exit(f"Gateway is not ready ({ready}). Start the stack: python tasks.py up")

        self.step(
            "What is deployed",
            "One base LLM (Qwen2.5-0.5B) with a LoRA adapter per task on top, plus a router. "
            "Each one is a versioned model in the MLflow registry; the gateway serves the "
            "version with the 'champion' alias.",
        )
        print(f"  live versions: {get(f'{self.gw}/models')['live']}")
        print(f"  registry ({self.model_name}): {self.aliases()}")

        self.step(
            "The switchboard routes each request",
            "The router embeds the request and picks an adapter, or the plain base model "
            "for general questions. Every answer gets a reference-free quality score.",
        )
        for message in SAMPLES:
            r = post(f"{self.gw}/chat", {"message": message, "max_new_tokens": 120})
            route = r["route"]
            served = r["adapter"] + (f" v{r['adapter_version']}" if r["adapter_version"] else "")
            print(f"\n  Q: {message.splitlines()[-1][:90]}")
            print(f"  -> {served} (router: {route['label']} {route['confidence']:.2f}, "
                  f"{route['reason']}) quality {r['quality']}")  # fmt: skip
            print(f"  A: {r['answer'][:150]!r}")

        start = self.args.start_version
        live = get(f"{self.gw}/models")["live"].get(task)
        self.step(
            f"Start from {task} v{start}",
            f"For a repeatable demo, {task} v{start} (trained without the new kind of request) "
            f"is made champion and deployed with a fresh monitoring window. Live now: v{live}.",
        )
        if live != start:
            registry.promote(self.model_name, start, self.uri)
        # force: even if v{start} is already live, restart its window so earlier runs don't count
        print(f"  deploy: {self.admin_reload(task, force=True)['changes']}")
        print(f"  registry: {self.aliases()}")

        n_normal, n_drift = self.args.n_normal, self.args.n_drift
        self.step(
            "Normal traffic",
            f"{n_normal} requests that look like the adapter's training data. The monitor "
            "judges the live version on its recent requests: quality and input drift.",
        )
        s = simulate(task, "golden", n_normal, self.gw, seed=11, verbose=self.args.verbose)
        print(
            f"  sent {n_normal} requests in {s['seconds']}s, mean quality {s['mean_quality']:.3f}"
        )
        print(self.monitor_line(s["monitor"]))

        self.step(
            "Drift: a new kind of request arrives",
            f"{n_drift} requests of a kind v{start} never saw (for sql: questions that need "
            "COUNT/SUM/AVG). Watch quality fall and input drift rise past their thresholds. "
            "Grafana: http://localhost:3000",
        )
        s = simulate(task, "drift_eval", n_drift, self.gw, seed=12, verbose=self.args.verbose)
        print(f"  sent {n_drift} requests in {s['seconds']}s, mean quality {s['mean_quality']:.3f}")
        print(self.monitor_line(s["monitor"]))
        if s["monitor"]["status"] != "alert":
            print("  (no alert yet: send more drifted traffic or lower --n-normal)")
        if self.args.no_retrain:
            print("\n  --no-retrain: stopping here. Run the controller to heal it:")
            print("  python -m adaptos.retraining.controller --once")
            return

        self.step(
            "Self-healing: the controller retrains",
            "The controller sees the alert and runs the retraining pipeline: warm start from "
            "the champion, new labeled data plus rehearsal of old data, evaluation against the "
            "champion, promotion gate, and a hot-swap of the gateway if the new version wins.",
        )
        before = get(f"{self.gw}/models")["live"].get(task)
        cmd = [sys.executable, "-m", "adaptos.retraining.controller", "--once", "--ignore-cooldown"]
        env = {**os.environ, "PYTHONUTF8": "1", "MLFLOW_DISABLE_AGENT_HINT": "1"}
        code = subprocess.call(cmd, cwd=ROOT, env=env)
        after = get(f"{self.gw}/models")["live"].get(task)
        print(f"\n  controller exit code {code} | live {task}: v{before} -> v{after}")
        print(f"  registry: {self.aliases()}")
        if code != 0:
            print("  The retraining pipeline FAILED; its log is attached to the run in the "
                  "'adaptos-retraining' experiment in MLflow. Nothing was deployed.")  # fmt: skip
            return
        if after == before:
            print("  No new version was deployed: the gate rejected the candidate, or the "
                  "daily retraining cap was reached ('adaptos-retraining' in MLflow).")  # fmt: skip
            return

        self.step(
            "The same kind of traffic, after retraining",
            f"The new version v{after} learned the new kind of request, and its drift "
            "reference includes it. The monitor only counts traffic served by the live "
            "version, so the alert clears.",
        )
        s = simulate(task, "drift_eval", n_drift, self.gw, seed=13, verbose=self.args.verbose)
        print(f"  sent {n_drift} requests in {s['seconds']}s, mean quality {s['mean_quality']:.3f}")
        print(self.monitor_line(s["monitor"]))

        print(f"\n{'=' * 78}\nDone. Explore the evidence:")
        print("  MLflow  http://localhost:5000  experiments adaptos-retraining / -training / "
              "-evaluation, models adaptos-adapter-*")  # fmt: skip
        print("  Grafana http://localhost:3000  dashboard 'AdaptOS Overview'")
        print("  Gateway http://localhost:8000/docs")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--task", default="sql", choices=["sql", "summarize", "math"])
    p.add_argument("--start-version", default="1", help="version to start the demo from")
    p.add_argument("--n-normal", type=int, default=30)
    p.add_argument("--n-drift", type=int, default=35)
    p.add_argument("--yes", action="store_true", help="do not pause between steps")
    p.add_argument("--verbose", action="store_true", help="print every request")
    p.add_argument("--no-retrain", action="store_true", help="stop after the drift alert")
    args = p.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    Demo(args).run()


if __name__ == "__main__":
    main()
