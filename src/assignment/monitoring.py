
"""
Checkpoint 3 — Monitoring and Alerts
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


def default_metrics_path() -> str:
    repo_root = Path(__file__).resolve().parents[2]
    return str(repo_root / "outputs" / "metrics.json")


@dataclass
class Alert:
    metric: str
    value: float
    threshold: float
    message: str


@dataclass
class MonitoringAlert:
    """Track security metrics and generate alerts."""

    block_rate_threshold: float = 0.5
    rate_limit_hit_threshold: int = 5
    judge_fail_rate_threshold: float = 0.3

    alerts: list[Alert] = field(default_factory=list)

    total_requests: int = 0
    blocked_requests: int = 0
    rate_limit_hits: int = 0
    judge_checks: int = 0
    judge_fails: int = 0

    def check_metrics(self) -> list[Alert]:
        self.alerts.clear()

        metrics = self.snapshot()

        block_rate = metrics["block_rate"]
        rate_limit_hits = metrics["rate_limit_hits"]
        judge_fail_rate = metrics["judge_fail_rate"]

        if block_rate > self.block_rate_threshold:
            self.alerts.append(Alert(
                metric="block_rate",
                value=block_rate,
                threshold=self.block_rate_threshold,
                message="Block rate exceeded configured threshold",
            ))

        if rate_limit_hits >= self.rate_limit_hit_threshold:
            self.alerts.append(Alert(
                metric="rate_limit_hits",
                value=rate_limit_hits,
                threshold=self.rate_limit_hit_threshold,
                message="Rate limit hit threshold reached",
            ))

        if (
            self.judge_checks > 0
            and judge_fail_rate > self.judge_fail_rate_threshold
        ):
            self.alerts.append(Alert(
                metric="judge_fail_rate",
                value=judge_fail_rate,
                threshold=self.judge_fail_rate_threshold,
                message="LLM judge failure rate exceeded threshold",
            ))

        return self.alerts

    def snapshot(self) -> dict:
        block_rate = (
            self.blocked_requests / self.total_requests
            if self.total_requests else 0.0
        )

        judge_fail_rate = (
            self.judge_fails / self.judge_checks
            if self.judge_checks else 0.0
        )

        return {
            "total_requests": self.total_requests,
            "blocked_requests": self.blocked_requests,
            "block_rate": block_rate,
            "rate_limit_hits": self.rate_limit_hits,
            "judge_checks": self.judge_checks,
            "judge_fails": self.judge_fails,
            "judge_fail_rate": judge_fail_rate,
            "alerts": [
                {
                    "metric": alert.metric,
                    "value": alert.value,
                    "threshold": alert.threshold,
                    "message": alert.message,
                }
                for alert in self.alerts
            ],
        }

    def export_json(self, filepath: str | None = None):
        self.check_metrics()

        path = Path(filepath or default_metrics_path())
        path.parent.mkdir(parents=True, exist_ok=True)

        path.write_text(
            json.dumps(
                self.snapshot(),
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        return str(path)
