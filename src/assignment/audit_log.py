
"""
Checkpoint 3 — Audit Logging
"""
from __future__ import annotations

import json
import uuid

from datetime import datetime, timezone
from pathlib import Path


def default_audit_log_path() -> str:
    repo_root = Path(__file__).resolve().parents[2]
    return str(repo_root / "outputs" / "audit_log.json")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class AuditLogPlugin:
    """Record request inputs, outputs, decisions and latency."""

    def __init__(self):
        self.name = "audit_log"
        self.logs: list[dict] = []
        self._open: dict[str, float] = {}

    def record_input(
        self,
        *,
        user_id: str,
        text: str,
        request_id: str | None = None,
    ):
        request_id = request_id or str(uuid.uuid4())

        started_at = datetime.now(timezone.utc)

        self._open[request_id] = started_at.timestamp()

        self.logs.append({
            "event": "input",
            "request_id": request_id,
            "user_id": user_id,
            "input": text,
            "timestamp": started_at.isoformat(),
        })

        return request_id

    def record_output(
        self,
        *,
        user_id: str,
        text: str,
        blocked: bool = False,
        layer: str | None = None,
        request_id: str | None = None,
    ):
        now = datetime.now(timezone.utc)

        started = self._open.pop(request_id, None) if request_id else None

        latency_ms = (
            round((now.timestamp() - started) * 1000, 2)
            if started is not None
            else None
        )

        self.logs.append({
            "event": "output",
            "request_id": request_id,
            "user_id": user_id,
            "output": text,
            "blocked": blocked,
            "layer": layer,
            "latency_ms": latency_ms,
            "timestamp": now.isoformat(),
        })

    def export_json(self, filepath: str | None = None):
        path = Path(filepath or default_audit_log_path())

        path.parent.mkdir(parents=True, exist_ok=True)

        path.write_text(
            json.dumps(
                self.logs,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        return str(path)
