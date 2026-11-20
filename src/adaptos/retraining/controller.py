"""Phase 7 controller: closes the loop from monitoring alert to a new champion in production.

    python -m adaptos.retraining.controller --once          # one check (demo)
    python -m adaptos.retraining.controller                 # keep watching

Each cycle:
    1. read GET /monitoring from the gateway
    2. for every adapter in "alert": skip if it was retrained within the cooldown or hit the
       daily cap; otherwise open a retraining run in MLflow (experiment ``adaptos-retraining``)
       that records why it was triggered
    3. run the retraining pipeline
    4. if the gate promoted the new version, tell the gateway to hot-swap to it
       (POST /admin/reload); the alert then clears because the window only counts the live
       version, and its drift reference now includes the new kind of request
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import urllib.request
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from adaptos.config import REPO_ROOT, AppConfig, Settings, get_settings, load_config
from adaptos.log import configure_logging, get_logger
from adaptos.registry import adapters as registry

logger = get_logger(__name__)

EXPERIMENT_SUFFIX = "retraining"
DAY_SECONDS = 24 * 3600


def http_json(method: str, url: str, body: dict | None = None, headers: dict | None = None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(
        url,
        data=data,
        method=method,
        headers={"Content-Type": "application/json", **(headers or {})},
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.load(r)


def pipeline_command(task: str, run_id: str) -> list[str]:
    return [sys.executable, "-m", "adaptos.retraining.pipeline", "--task", task,
            "--parent-run-id", run_id]  # fmt: skip


class Controller:
    def __init__(
        self,
        app: AppConfig,
        settings: Settings,
        http: Callable[..., Any] = http_json,
        run_command: Callable[[list[str], Path], int] | None = None,
        clock: Callable[[], float] = time.time,
        ignore_cooldown: bool = False,
    ):
        import mlflow

        self.app, self.settings = app, settings
        self.ignore_cooldown = ignore_cooldown
        self.http = http
        self.run_command = run_command or self._run_logged
        self.clock = clock
        self.mlflow = mlflow
        self.client = mlflow.MlflowClient(settings.mlflow_tracking_uri)
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        self.experiment = mlflow.set_experiment(
            f"{app.mlflow.experiment_prefix}-{EXPERIMENT_SUFFIX}"
        )

    # ----------------------------------------------------------------- policy
    def recent_runs(self, task: str, within_sec: float) -> list[Any]:
        since_ms = int((self.clock() - within_sec) * 1000)
        return self.client.search_runs(
            [self.experiment.experiment_id],
            filter_string=f"tags.task = '{task}' AND attributes.start_time > {since_ms}",
        )

    def allowed(self, task: str) -> tuple[bool, str]:
        """Cooldown and daily cap stop a persistent alert from retraining in a loop.

        ``ignore_cooldown`` (demos) skips the cooldown; the daily cap always applies.
        """
        cooldown = self.app.monitor.retrain_cooldown_minutes * 60
        if not self.ignore_cooldown and self.recent_runs(task, cooldown):
            return False, f"retrained less than {self.app.monitor.retrain_cooldown_minutes} min ago"
        if len(self.recent_runs(task, DAY_SECONDS)) >= self.app.retraining.max_runs_per_day:
            return False, f"reached {self.app.retraining.max_runs_per_day} runs in 24h"
        return True, "ok"

    # ------------------------------------------------------------------ cycle
    def check(self) -> list[dict[str, Any]]:
        gateway = self.app.retraining.gateway_url
        status = self.http("GET", f"{gateway}/monitoring")["adapters"]
        actions = []
        for task, stats in status.items():
            if stats["status"] != "alert":
                continue
            ok, why = self.allowed(task)
            if not ok:
                logger.info("%s is in alert but not retrained: %s", task, why)
                actions.append({"task": task, "action": "skipped", "reason": why})
                continue
            actions.append(self.retrain(task, stats))
        if not actions:
            logger.info("all adapters healthy: %s", {t: s["status"] for t, s in status.items()})
        return actions

    def retrain(self, task: str, stats: dict[str, Any]) -> dict[str, Any]:
        cfg = self.app.adapters[task]
        uri = self.settings.mlflow_tracking_uri
        before = registry.get_alias_version(cfg.registered_model, registry.CHAMPION, uri)
        reasons = "; ".join(stats["reasons"])
        logger.info("%s: ALERT (%s) -> retraining", task, reasons)

        started = self.clock()
        stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
        with self.mlflow.start_run(run_name=f"{task}-retrain-{stamp}") as run:
            run_id = run.info.run_id
            self.mlflow.set_tags(
                {
                    "task": task,
                    "stage": "retraining",
                    "trigger_reason": reasons,
                    "champion_before": before.version if before else "none",
                }
            )
            self.mlflow.log_metrics(
                {
                    k: v
                    for k, v in {
                        "trigger/window_quality": stats.get("quality"),
                        "trigger/window_drift": stats.get("drift"),
                        "trigger/window_samples": stats.get("samples"),
                    }.items()
                    if v is not None
                }
            )
            log_path = self.settings.artifacts_dir / "retraining" / f"{run_id}.log"
            code = self.run_command(pipeline_command(task, run_id), log_path)
            if log_path.exists():
                self.mlflow.log_artifact(str(log_path))

            after = registry.get_alias_version(cfg.registered_model, registry.CHAMPION, uri)
            promoted = after is not None and (before is None or after.version != before.version)
            outcome: dict[str, Any] = {"task": task, "run_id": run_id, "exit_code": code}
            if code != 0:
                outcome["action"] = "failed"
            elif promoted:
                swap = self.http(
                    "POST",
                    f"{self.app.retraining.gateway_url}/admin/reload",
                    {"model": task},
                    {"X-Admin-Token": self.settings.admin_token or "dev-admin-token"},
                )
                outcome |= {
                    "action": "promoted",
                    "new_version": after.version,
                    "swap": swap["changes"],
                }
                self.mlflow.set_tag("champion_after", after.version)
            else:
                outcome["action"] = "rejected"
            self.mlflow.set_tag("outcome", outcome["action"])
            self.mlflow.log_metric("duration_sec", self.clock() - started)
        logger.info("%s: retraining %s", task, outcome)
        if code != 0:
            raise RuntimeError(f"retraining pipeline for {task} failed; see {log_path}")
        return outcome

    def _run_logged(self, cmd: list[str], log_path: Path) -> int:
        """Run the pipeline, streaming its output to the console and a log file."""
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w", encoding="utf-8") as log:
            proc = subprocess.Popen(
                cmd, cwd=REPO_ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding="utf-8", errors="replace",
            )  # fmt: skip
            assert proc.stdout is not None
            for line in proc.stdout:
                sys.stdout.write(line)
                log.write(line)
            return proc.wait()

    def loop(self) -> None:
        interval = self.app.monitor.check_interval_seconds
        logger.info("controller watching %s every %ss", self.app.retraining.gateway_url, interval)
        while True:
            try:
                self.check()
            except Exception:
                logger.exception("controller cycle failed")
            time.sleep(interval)


def main(argv: Sequence[str] | None = None) -> None:
    settings = get_settings()
    configure_logging(settings.log_level)
    p = argparse.ArgumentParser(description="Watch adapter health and retrain on alerts")
    p.add_argument("--once", action="store_true", help="run a single check and exit")
    p.add_argument(
        "--ignore-cooldown", action="store_true", help="skip the cooldown (daily cap still applies)"
    )
    args = p.parse_args(argv)
    controller = Controller(
        load_config(settings.config_dir),
        settings,
        ignore_cooldown=args.ignore_cooldown,
    )
    if args.once:
        print(json.dumps(controller.check(), indent=2, default=str))
    else:
        controller.loop()


if __name__ == "__main__":
    main()
