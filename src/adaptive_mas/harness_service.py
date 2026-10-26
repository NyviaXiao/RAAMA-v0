"""Loopback-only API and independently run SQLite worker for cached harness decisions."""

from __future__ import annotations

import argparse
import hashlib
import hmac
import ipaddress
import json
import math
import os
import re
import signal
import socket
import sqlite3
import subprocess
import sys
import threading
import time
import tomllib
import uuid
import shutil
from contextlib import closing
from datetime import date, datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Callable, Mapping
from urllib.parse import parse_qs, urlparse

from adaptive_mas.observability import decision_summary, export_span_if_configured, safe_attributes, span_record, utc_now
from adaptive_mas.execution_ledger import simulate_execution_ledger
from adaptive_mas.release_governance import HarnessReleaseStore
from adaptive_mas.research_provenance import stable_hash
from adaptive_mas.risk_profile import resolve_risk_profile, validate_canonical_risk_profile
from adaptive_mas.runtime_binding import build_workflow_runtime_binding
from adaptive_mas.workflow_snapshot import materialize_workflow_snapshot, verify_workflow_materialization


class ServiceConflict(RuntimeError):
    """A stable API idempotency key was reused with changed input."""


class WorkflowControlRequested(RuntimeError):
    def __init__(self, action: str) -> None:
        super().__init__(f"workflow {action} requested")
        self.action = action


class WorkerLeaseLost(ServiceConflict):
    """The worker attempt no longer owns the durable job lease."""


def _json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _is_loopback_host_header(value: str) -> bool:
    try:
        parsed = urlparse(f"//{value}")
        host = parsed.hostname
        # Reject malformed ports and user-info syntax in the authority.
        parsed.port
    except ValueError:
        return False
    return (
        host is not None
        and parsed.username is None
        and parsed.password is None
        and host.lower() in {"127.0.0.1", "localhost", "::1"}
    )


class HarnessServiceStore:
    """Durable queue, leases, spans and results in the same SQLite file as task state."""

    def __init__(self, database_path: str | Path) -> None:
        self.database_path = Path(database_path).resolve()
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS service_jobs (
                    task_id TEXT PRIMARY KEY,
                    idempotency_key TEXT NOT NULL UNIQUE,
                    request_hash TEXT NOT NULL,
                    request_json TEXT NOT NULL,
                    status TEXT NOT NULL,
                    stage TEXT NOT NULL,
                    attempt_count INTEGER NOT NULL DEFAULT 0,
                    worker_id TEXT,
                    lease_token TEXT,
                    lease_until REAL,
                    result_json TEXT,
                    error_json TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS service_jobs_claim_idx
                    ON service_jobs(status, lease_until, created_at);
                CREATE TABLE IF NOT EXISTS service_job_events (
                    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    task_id TEXT NOT NULL,
                    event_key TEXT NOT NULL,
                    stage TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    UNIQUE(task_id, event_key)
                );
                CREATE TABLE IF NOT EXISTS service_workers (
                    worker_id TEXT PRIMARY KEY,
                    state TEXT NOT NULL,
                    last_seen_at REAL NOT NULL,
                    current_task_id TEXT
                );
                CREATE TABLE IF NOT EXISTS service_execution_ledgers (
                    ledger_id TEXT PRIMARY KEY,
                    request_hash TEXT NOT NULL,
                    result_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                """
            )
            job_columns = {
                str(row["name"]) for row in connection.execute("PRAGMA table_info(service_jobs)").fetchall()
            }
            if "lease_token" not in job_columns:
                connection.execute("ALTER TABLE service_jobs ADD COLUMN lease_token TEXT")

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, timeout=30, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA busy_timeout=30000")
        return connection

    @staticmethod
    def _job(row: sqlite3.Row | None) -> dict[str, object] | None:
        if row is None:
            return None
        result = dict(row)
        for column in ("request_json", "result_json", "error_json"):
            value = result.pop(column, None)
            if value is not None:
                result[column.removesuffix("_json")] = json.loads(value)
        return result

    def submit(
        self, task_id: str, idempotency_key: str, request: Mapping[str, object]
    ) -> dict[str, object]:
        request_json = _json(dict(request))
        idempotency_request = dict(request)
        idempotency_request.pop("runtime_binding", None)
        request_hash = _sha256(_json(idempotency_request))
        now = _now()
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            prior = connection.execute(
                "SELECT * FROM service_jobs WHERE idempotency_key=?", (idempotency_key,)
            ).fetchone()
            if prior is not None:
                if prior["request_hash"] != request_hash:
                    connection.rollback()
                    raise ServiceConflict("idempotency key was reused with changed request data")
                connection.commit()
                return self._job(prior)  # type: ignore[return-value]
            prior_id = connection.execute(
                "SELECT * FROM service_jobs WHERE task_id=?", (task_id,)
            ).fetchone()
            if prior_id is not None:
                connection.rollback()
                raise ServiceConflict("task_id already belongs to another job")
            connection.execute(
                "INSERT INTO service_jobs(task_id,idempotency_key,request_hash,request_json,status,stage,created_at,updated_at) "
                "VALUES(?,?,?,?,?,?,?,?)",
                (task_id, idempotency_key, request_hash, request_json, "queued", "queued", now, now),
            )
            self._insert_event(
                connection, task_id, "job_queued", "queued",
                span_record(
                    trace_id=task_id,
                    name="job.queued",
                    attributes={"request_hash": request_hash, "mode": request.get("mode")},
                    started_at=utc_now(),
                    status="ok",
                ),
            )
            row = connection.execute(
                "SELECT * FROM service_jobs WHERE task_id=?", (task_id,)
            ).fetchone()
            connection.commit()
            return self._job(row)  # type: ignore[return-value]

    @staticmethod
    def _insert_event(
        connection: sqlite3.Connection,
        task_id: str,
        event_key: str,
        stage: str,
        payload: Mapping[str, object],
    ) -> None:
        connection.execute(
            "INSERT OR IGNORE INTO service_job_events(task_id,event_key,stage,payload_json,created_at) VALUES(?,?,?,?,?)",
            (task_id, event_key, stage, _json(dict(payload)), _now()),
        )
        if "trace_id" in payload and "span_id" in payload:
            export_span_if_configured(payload)

    def append_event(
        self, task_id: str, event_key: str, stage: str, payload: Mapping[str, object]
    ) -> None:
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            self._insert_event(
                connection, task_id, event_key, stage,
                safe_attributes(payload) | {
                    key: value for key, value in payload.items()
                    if key in {"trace_id", "span_id", "parent_span_id", "name", "started_at", "ended_at", "duration_ms", "status", "attributes"}
                },
            )
            connection.commit()

    def claim_next(self, worker_id: str, *, lease_seconds: float = 900.0) -> dict[str, object] | None:
        if lease_seconds <= 0:
            raise ValueError("lease_seconds must be positive")
        now_epoch = time.time()
        now = _now()
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            expired = connection.execute(
                "SELECT task_id,attempt_count,worker_id,stage FROM service_jobs "
                "WHERE status IN ('running','pause_requested','cancel_requested') "
                "AND lease_until IS NOT NULL AND lease_until < ?",
                (now_epoch,),
            ).fetchall()
            for stale in expired:
                task_id = str(stale["task_id"])
                attempt = int(stale["attempt_count"])
                error = {
                    "reason": "worker lease expired before a durable completion record",
                    "previous_worker_id": stale["worker_id"],
                    "previous_stage": str(stale["stage"]),
                    "attempt_count": attempt,
                }
                connection.execute(
                    "UPDATE service_jobs SET status='unknown',stage='unknown',error_json=?,worker_id=NULL,lease_token=NULL,lease_until=NULL,updated_at=? "
                    "WHERE task_id=? AND status IN ('running','pause_requested','cancel_requested') AND lease_until < ?",
                    (_json(error), now, task_id, now_epoch),
                )
                self._insert_event(
                    connection, task_id, f"job_unknown:expired_lease:{attempt}", "unknown",
                    span_record(
                        trace_id=task_id,
                        name="job.unknown",
                        attributes=error,
                        status="unknown",
                    ),
                )
            row = connection.execute(
                "SELECT * FROM service_jobs WHERE status='queued' ORDER BY created_at LIMIT 1",
            ).fetchone()
            if row is None:
                connection.execute(
                    "INSERT INTO service_workers(worker_id,state,last_seen_at,current_task_id) VALUES(?,?,?,NULL) "
                    "ON CONFLICT(worker_id) DO UPDATE SET state=excluded.state,last_seen_at=excluded.last_seen_at,current_task_id=NULL",
                    (worker_id, "idle", now_epoch),
                )
                connection.commit()
                return None
            task_id = str(row["task_id"])
            attempt = int(row["attempt_count"]) + 1
            lease_token = uuid.uuid4().hex
            lease_until = now_epoch + lease_seconds
            connection.execute(
                "UPDATE service_jobs SET status='running',stage='preflight',attempt_count=?,worker_id=?,lease_token=?,lease_until=?,updated_at=? WHERE task_id=?",
                (attempt, worker_id, lease_token, lease_until, now, task_id),
            )
            connection.execute(
                "INSERT INTO service_workers(worker_id,state,last_seen_at,current_task_id) VALUES(?,?,?,?) "
                "ON CONFLICT(worker_id) DO UPDATE SET state=excluded.state,last_seen_at=excluded.last_seen_at,current_task_id=excluded.current_task_id",
                (worker_id, "busy", now_epoch, task_id),
            )
            span = span_record(
                trace_id=task_id,
                name="job.claimed",
                attributes={"attempt": attempt},
                status="ok",
            )
            self._insert_event(connection, task_id, f"job_claimed:{attempt}", "preflight", span)
            claimed = connection.execute(
                "SELECT * FROM service_jobs WHERE task_id=?", (task_id,)
            ).fetchone()
            connection.commit()
            return self._job(claimed)

    def update_stage(
        self,
        task_id: str,
        stage: str,
        *,
        worker_id: str,
        lease_token: str,
        payload: Mapping[str, object] | None = None,
    ) -> None:
        now = _now()
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            lease = connection.execute(
                "SELECT status FROM service_jobs WHERE task_id=? AND worker_id=? AND lease_token=?",
                (task_id, worker_id, lease_token),
            ).fetchone()
            if lease is not None and lease["status"] in {"pause_requested", "cancel_requested"}:
                connection.rollback()
                raise WorkflowControlRequested(str(lease["status"]).removesuffix("_requested"))
            connection.execute(
                "UPDATE service_jobs SET stage=?,lease_until=?,updated_at=? WHERE task_id=? AND worker_id=? AND lease_token=? AND status='running'",
                (stage, time.time() + 900, now, task_id, worker_id, lease_token),
            )
            if connection.execute("SELECT changes()").fetchone()[0] != 1:
                connection.rollback()
                raise WorkerLeaseLost("worker no longer owns the job lease")
            self._insert_event(
                connection, task_id, f"stage:{stage}:{hashlib.sha256(_json(payload or {}).encode()).hexdigest()[:12]}",
                stage, span_record(trace_id=task_id, name=f"job.{stage}", attributes=payload or {}, status="ok"),
            )
            connection.execute(
                "UPDATE service_workers SET state='busy',last_seen_at=?,current_task_id=? WHERE worker_id=?",
                (time.time(), task_id, worker_id),
            )
            connection.commit()

    def renew_lease(
        self, task_id: str, worker_id: str, *, lease_token: str, lease_seconds: float = 900.0
    ) -> None:
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                "UPDATE service_jobs SET lease_until=?,updated_at=? WHERE task_id=? AND worker_id=? "
                "AND lease_token=? AND status IN ('running','pause_requested','cancel_requested')",
                (time.time() + lease_seconds, _now(), task_id, worker_id, lease_token),
            )
            if connection.execute("SELECT changes()").fetchone()[0] != 1:
                connection.rollback()
                raise WorkerLeaseLost("worker no longer owns the job lease")
            connection.execute(
                "UPDATE service_workers SET last_seen_at=? WHERE worker_id=? AND current_task_id=?",
                (time.time(), worker_id, task_id),
            )
            connection.commit()

    def request_control(
        self,
        task_id: str,
        *,
        action: str,
        operator_id: str,
        reason: str,
    ) -> dict[str, object]:
        if action not in {"pause", "resume", "cancel"}:
            raise ValueError("action must be pause, resume, or cancel")
        if not operator_id.strip() or not reason.strip():
            raise ValueError("operator identity and control reason are required")
        now = _now()
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT status,attempt_count FROM service_jobs WHERE task_id=?", (task_id,)
            ).fetchone()
            if row is None:
                raise KeyError(task_id)
            previous = str(row["status"])
            no_op = (
                (action == "pause" and previous in {"paused", "pause_requested"})
                or (action == "resume" and previous == "queued")
                or (action == "cancel" and previous in {"cancelled", "cancel_requested"})
            )
            if no_op:
                connection.commit()
                return {"task_id": task_id, "status": previous, "changed": False}
            transitions = {
                ("pause", "queued"): "paused",
                ("pause", "running"): "pause_requested",
                ("resume", "paused"): "queued",
                ("cancel", "queued"): "cancelled",
                ("cancel", "paused"): "cancelled",
                ("cancel", "running"): "cancel_requested",
                ("cancel", "pause_requested"): "cancel_requested",
            }
            next_status = transitions.get((action, previous))
            if next_status is None:
                raise ServiceConflict(f"cannot {action} a task in {previous} state")
            if next_status in {"paused", "cancelled", "queued"}:
                connection.execute(
                    "UPDATE service_jobs SET status=?,stage=?,worker_id=NULL,lease_token=NULL,lease_until=NULL,updated_at=? WHERE task_id=?",
                    (next_status, next_status, now, task_id),
                )
                if previous == "running":
                    connection.execute(
                        "UPDATE service_workers SET state='idle',last_seen_at=?,current_task_id=NULL WHERE current_task_id=?",
                        (time.time(), task_id),
                    )
            else:
                connection.execute(
                    "UPDATE service_jobs SET status=?,stage=?,updated_at=? WHERE task_id=?",
                    (next_status, next_status, now, task_id),
                )
            attempt = int(row["attempt_count"])
            self._insert_event(
                connection,
                task_id,
                f"job_control:{attempt}:{uuid.uuid4().hex}",
                f"control_{action}",
                span_record(
                    trace_id=task_id,
                    name=f"job.control.{action}",
                    attributes={
                        "operator_id": operator_id,
                        "reason": reason.strip()[:1000],
                        "previous_status": previous,
                        "next_status": next_status,
                        "attempt_count": attempt,
                    },
                    status="ok",
                ),
            )
            connection.commit()
        return {"task_id": task_id, "status": next_status, "changed": True}

    def current_control(self, task_id: str, worker_id: str, lease_token: str) -> str | None:
        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT status,worker_id,lease_token FROM service_jobs WHERE task_id=?", (task_id,)
            ).fetchone()
        if row is None or row["worker_id"] != worker_id or row["lease_token"] != lease_token:
            return "lease_lost"
        status = str(row["status"])
        if status in {"pause_requested", "cancel_requested"}:
            return status.removesuffix("_requested")
        return None

    def acknowledge_control(self, task_id: str, worker_id: str, lease_token: str) -> str | None:
        now = _now()
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT status,attempt_count FROM service_jobs WHERE task_id=? AND worker_id=? AND lease_token=?",
                (task_id, worker_id, lease_token),
            ).fetchone()
            if row is None:
                connection.commit()
                return None
            previous = str(row["status"])
            next_status = {"pause_requested": "paused", "cancel_requested": "cancelled"}.get(previous)
            if next_status is None:
                connection.commit()
                return None
            attempt = int(row["attempt_count"])
            connection.execute(
                "UPDATE service_jobs SET status=?,stage=?,worker_id=NULL,lease_token=NULL,lease_until=NULL,updated_at=? "
                "WHERE task_id=? AND worker_id=? AND lease_token=? AND status=?",
                (next_status, next_status, now, task_id, worker_id, lease_token, previous),
            )
            if connection.execute("SELECT changes()").fetchone()[0] != 1:
                connection.rollback()
                return None
            connection.execute(
                "UPDATE service_workers SET state='idle',last_seen_at=?,current_task_id=NULL WHERE worker_id=? AND current_task_id=?",
                (time.time(), worker_id, task_id),
            )
            self._insert_event(
                connection,
                task_id,
                f"job_control_ack:{attempt}:{next_status}",
                next_status,
                span_record(
                    trace_id=task_id,
                    name=f"job.control.{next_status}",
                    attributes={"worker_id": worker_id, "attempt_count": attempt},
                    status="ok",
                ),
            )
            connection.commit()
            return next_status

    def complete(
        self,
        task_id: str,
        worker_id: str,
        lease_token: str,
        result: Mapping[str, object],
        *,
        elapsed_ms: float,
    ) -> None:
        now = _now()
        result_json = _json(dict(result))
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT status,result_json,worker_id,lease_token FROM service_jobs WHERE task_id=?", (task_id,)
            ).fetchone()
            if row is None:
                raise KeyError(task_id)
            if row["status"] == "completed":
                if row["worker_id"] != worker_id or row["lease_token"] != lease_token:
                    raise WorkerLeaseLost("worker attempt no longer owns the completed job")
                if row["result_json"] != result_json:
                    raise ServiceConflict("completed job result is immutable")
                connection.commit()
                return
            if row["worker_id"] != worker_id or row["lease_token"] != lease_token:
                raise WorkerLeaseLost("worker attempt no longer owns the job lease")
            if row["status"] in {"pause_requested", "cancel_requested"}:
                connection.rollback()
                raise WorkflowControlRequested(str(row["status"]).removesuffix("_requested"))
            connection.execute(
                "UPDATE service_jobs SET status='completed',stage='completed',result_json=?,error_json=NULL,lease_until=NULL,updated_at=? "
                "WHERE task_id=? AND worker_id=? AND lease_token=? AND status='running'",
                (result_json, now, task_id, worker_id, lease_token),
            )
            if connection.execute("SELECT changes()").fetchone()[0] != 1:
                connection.rollback()
                raise WorkerLeaseLost("worker attempt no longer owns the job lease")
            self._insert_event(
                connection, task_id, "job_completed", "completed",
                span_record(
                    trace_id=task_id,
                    name="job.completed",
                    attributes=decision_summary(result),
                    started_at=datetime.fromtimestamp(time.time() - elapsed_ms / 1000, timezone.utc),
                    ended_at=utc_now(),
                    status="ok",
                ),
            )
            connection.execute(
                "UPDATE service_workers SET state='idle',last_seen_at=?,current_task_id=NULL WHERE worker_id=?",
                (time.time(), worker_id),
            )
            connection.commit()

    def fail(
        self,
        task_id: str,
        worker_id: str,
        lease_token: str,
        error: Mapping[str, object],
        *,
        unknown: bool = False,
    ) -> None:
        now = _now()
        status = "unknown" if unknown else "failed"
        error_json = _json(safe_attributes(error))
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT status FROM service_jobs WHERE task_id=? AND worker_id=? AND lease_token=?",
                (task_id, worker_id, lease_token),
            ).fetchone()
            if row is not None and row["status"] in {"pause_requested", "cancel_requested"}:
                connection.rollback()
                raise WorkflowControlRequested(str(row["status"]).removesuffix("_requested"))
            connection.execute(
                "UPDATE service_jobs SET status=?,stage=?,error_json=?,lease_token=NULL,lease_until=NULL,updated_at=? "
                "WHERE task_id=? AND worker_id=? AND lease_token=? AND status='running'",
                (status, status, error_json, now, task_id, worker_id, lease_token),
            )
            if connection.execute("SELECT changes()").fetchone()[0] != 1:
                connection.rollback()
                raise WorkerLeaseLost("worker attempt no longer owns the job lease")
            self._insert_event(
                connection, task_id, f"job_{status}", status,
                span_record(trace_id=task_id, name=f"job.{status}", attributes=error, status=status),
            )
            connection.execute(
                "UPDATE service_workers SET state='idle',last_seen_at=?,current_task_id=NULL WHERE worker_id=?",
                (time.time(), worker_id),
            )
            connection.commit()

    def reconcile_unknown_job(
        self,
        task_id: str,
        *,
        expected_attempt: int,
        operator_id: str,
        reason: str,
    ) -> dict[str, object]:
        """Requeue an expired worker job only after an operator confirms no effect was applied."""
        if isinstance(expected_attempt, bool) or expected_attempt < 1:
            raise ValueError("expected_attempt must be positive")
        if not operator_id.strip() or not reason.strip():
            raise ValueError("operator identity and reconciliation reason are required")
        now = _now()
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT status,attempt_count,error_json FROM service_jobs WHERE task_id=?",
                (task_id,),
            ).fetchone()
            if row is None:
                raise KeyError(task_id)
            if row["status"] != "unknown":
                raise ServiceConflict("only an unknown job can be reconciled")
            if int(row["attempt_count"]) != expected_attempt:
                raise ServiceConflict("job attempt changed; refresh the task before reconciling")
            prior_error = json.loads(row["error_json"]) if row["error_json"] else {}
            connection.execute(
                "UPDATE service_jobs SET status='queued',stage='reconciled_retry',error_json=NULL,worker_id=NULL,lease_token=NULL,lease_until=NULL,updated_at=? "
                "WHERE task_id=? AND status='unknown' AND attempt_count=?",
                (now, task_id, expected_attempt),
            )
            if connection.execute("SELECT changes()").fetchone()[0] != 1:
                connection.rollback()
                raise ServiceConflict("job attempt changed; refresh the task before reconciling")
            event = span_record(
                trace_id=task_id,
                name="job.reconciled",
                attributes={
                    "action": "retry_after_confirmed_no_external_effect",
                    "operator_id": operator_id,
                    "reason": reason.strip()[:1000],
                    "reconciled_attempt": expected_attempt,
                    "prior_error": prior_error,
                },
                status="ok",
            )
            self._insert_event(
                connection, task_id,
                f"job_reconciled_no_effect:{expected_attempt}", "reconciled_retry", event,
            )
            connection.commit()
        return {
            "task_id": task_id,
            "status": "queued",
            "stage": "reconciled_retry",
            "reconciled_attempt": expected_attempt,
            "operator_id": operator_id,
        }

    def dead_letter_unknown_job(
        self,
        task_id: str,
        *,
        expected_attempt: int,
        operator_id: str,
        reason: str,
    ) -> dict[str, object]:
        if isinstance(expected_attempt, bool) or expected_attempt < 1:
            raise ValueError("expected_attempt must be positive")
        if not operator_id.strip() or not reason.strip():
            raise ValueError("operator identity and dead-letter reason are required")
        now = _now()
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT status,attempt_count,error_json FROM service_jobs WHERE task_id=?", (task_id,)
            ).fetchone()
            if row is None:
                raise KeyError(task_id)
            if row["status"] != "unknown":
                raise ServiceConflict("only an unknown job can be closed as a dead letter")
            if int(row["attempt_count"]) != expected_attempt:
                raise ServiceConflict("job attempt changed; refresh the task before resolving it")
            prior_error = json.loads(row["error_json"]) if row["error_json"] else {}
            connection.execute(
                "UPDATE service_jobs SET status='dead_letter',stage='dead_letter',worker_id=NULL,lease_token=NULL,lease_until=NULL,updated_at=? "
                "WHERE task_id=? AND status='unknown' AND attempt_count=?",
                (now, task_id, expected_attempt),
            )
            if connection.execute("SELECT changes()").fetchone()[0] != 1:
                connection.rollback()
                raise ServiceConflict("job attempt changed; refresh the task before resolving it")
            self._insert_event(
                connection,
                task_id,
                f"job_dead_letter:{expected_attempt}",
                "dead_letter",
                span_record(
                    trace_id=task_id,
                    name="job.dead_letter",
                    attributes={
                        "operator_id": operator_id,
                        "reason": reason.strip()[:1000],
                        "resolved_attempt": expected_attempt,
                        "prior_error": prior_error,
                        "action": "close_without_retry",
                    },
                    status="ok",
                ),
            )
            connection.commit()
        return {
            "task_id": task_id,
            "status": "dead_letter",
            "stage": "dead_letter",
            "resolved_attempt": expected_attempt,
            "operator_id": operator_id,
        }

    def get_job(self, task_id: str) -> dict[str, object] | None:
        with closing(self._connect()) as connection:
            row = connection.execute("SELECT * FROM service_jobs WHERE task_id=?", (task_id,)).fetchone()
            return self._job(row)

    def get_job_by_idempotency_key(self, idempotency_key: str) -> dict[str, object] | None:
        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT * FROM service_jobs WHERE idempotency_key=?", (idempotency_key,)
            ).fetchone()
            return self._job(row)

    def list_jobs(self, limit: int = 50) -> list[dict[str, object]]:
        with closing(self._connect()) as connection:
            rows = connection.execute(
                "SELECT * FROM service_jobs ORDER BY created_at DESC LIMIT ?", (max(1, min(limit, 200)),)
            ).fetchall()
            return [self._job(row) for row in rows]  # type: ignore[misc]

    def events(self, task_id: str) -> list[dict[str, object]]:
        with closing(self._connect()) as connection:
            service_rows = connection.execute(
                "SELECT event_id,event_key,stage,payload_json,created_at FROM service_job_events WHERE task_id=? ORDER BY event_id",
                (task_id,),
            ).fetchall()
            try:
                task_rows = connection.execute(
                    "SELECT event_id,event_key,stage,payload_json,created_at FROM task_events WHERE task_id=? ORDER BY event_id",
                    (task_id,),
                ).fetchall()
            except sqlite3.OperationalError:
                task_rows = []
        combined = [
            {
                "source": source,
                "event_id": int(row["event_id"]),
                "event_key": str(row["event_key"]),
                "stage": str(row["stage"]),
                "payload": safe_attributes(json.loads(row["payload_json"])),
                "created_at": str(row["created_at"]),
            }
            for source, rows in (("service", service_rows), ("harness", task_rows))
            for row in rows
        ]
        return sorted(combined, key=lambda row: (str(row["created_at"]), str(row["source"]), int(row["event_id"])))

    def worker_status(self, *, stale_after_seconds: float = 20.0) -> list[dict[str, object]]:
        with closing(self._connect()) as connection:
            rows = connection.execute(
                "SELECT worker_id,state,last_seen_at,current_task_id FROM service_workers ORDER BY worker_id"
            ).fetchall()
        now = time.time()
        return [
            {
                "worker_id": str(row["worker_id"]),
                "state": str(row["state"]) if now - float(row["last_seen_at"]) <= stale_after_seconds else "offline",
                "last_seen_at": datetime.fromtimestamp(float(row["last_seen_at"]), timezone.utc).isoformat(),
                "current_task_id": row["current_task_id"],
            }
            for row in rows
        ]

    def save_execution_ledger(
        self, ledger_id: str, request_hash: str, result: Mapping[str, object]
    ) -> dict[str, object]:
        result_json = _json(dict(result))
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            prior = connection.execute(
                "SELECT request_hash,result_json,created_at FROM service_execution_ledgers WHERE ledger_id=?",
                (ledger_id,),
            ).fetchone()
            if prior is not None:
                if prior["request_hash"] != request_hash:
                    connection.rollback()
                    raise ServiceConflict("ledger_id is immutable and already belongs to different input")
                connection.commit()
                return {
                    "ledger_id": ledger_id,
                    "request_hash": request_hash,
                    "result": json.loads(prior["result_json"]),
                    "created_at": prior["created_at"],
                    "reused": True,
                }
            created_at = _now()
            connection.execute(
                "INSERT INTO service_execution_ledgers VALUES (?,?,?,?)",
                (ledger_id, request_hash, result_json, created_at),
            )
            connection.commit()
        return {
            "ledger_id": ledger_id,
            "request_hash": request_hash,
            "result": dict(result),
            "created_at": created_at,
            "reused": False,
        }

    def get_execution_ledger(self, ledger_id: str) -> dict[str, object] | None:
        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT ledger_id,request_hash,result_json,created_at FROM service_execution_ledgers WHERE ledger_id=?",
                (ledger_id,),
            ).fetchone()
        if row is None:
            return None
        return {
            "ledger_id": str(row["ledger_id"]),
            "request_hash": str(row["request_hash"]),
            "result": json.loads(row["result_json"]),
            "created_at": str(row["created_at"]),
        }

    def list_execution_ledgers(self, limit: int = 50) -> list[dict[str, object]]:
        with closing(self._connect()) as connection:
            rows = connection.execute(
                "SELECT ledger_id,request_hash,result_json,created_at FROM service_execution_ledgers ORDER BY created_at DESC LIMIT ?",
                (max(1, min(limit, 200)),),
            ).fetchall()
        output = []
        for row in rows:
            result = json.loads(row["result_json"])
            summary = result.get("summary", {}) if isinstance(result, Mapping) else {}
            output.append({
                "ledger_id": str(row["ledger_id"]),
                "request_hash": str(row["request_hash"]),
                "created_at": str(row["created_at"]),
                "execution_mode": result.get("execution_mode") if isinstance(result, Mapping) else None,
                "summary": summary,
                "ruleset": result.get("ruleset") if isinstance(result, Mapping) else None,
            })
        return output


def _within_root(value: object, root: Path, *, must_exist: bool, want_dir: bool | None = None) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("file and directory inputs must be nonempty project paths")
    candidate = Path(value)
    resolved = (root / candidate).resolve() if not candidate.is_absolute() else candidate.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError("service file inputs must stay inside the configured project root") from exc
    if must_exist and not resolved.exists():
        raise ValueError(f"configured input does not exist: {resolved.name}")
    if want_dir is True and resolved.exists() and not resolved.is_dir():
        raise ValueError(f"configured input must be a directory: {resolved.name}")
    if want_dir is False and resolved.exists() and not resolved.is_file():
        raise ValueError(f"configured input must be a file: {resolved.name}")
    if resolved.stem.lower() == "test" and resolved.suffix.lower() == ".csv":
        raise ValueError("the local scoring CSV is prohibited by the service")
    return str(resolved)


def parse_risk_profile_request(
    request: Mapping[str, object], project_root: str | Path
) -> dict[str, object]:
    """Parse a preview request with the same versioned rules used at task submission."""
    allowed = {"risk_template", "risk_expression", "risk_constraints", "risk_budget_annualized", "config"}
    extra = set(request) - allowed
    if extra:
        raise ValueError(f"unsupported risk profile fields: {sorted(extra)}")
    for key in ("risk_template", "risk_expression"):
        if key in request and not isinstance(request[key], str):
            raise ValueError(f"{key} must be text")
    if "risk_constraints" in request and not isinstance(request["risk_constraints"], Mapping):
        raise ValueError("risk_constraints must be an object")
    if "risk_budget_annualized" in request and request["risk_budget_annualized"] is None:
        raise ValueError("risk_budget_annualized must be a finite number when supplied")
    root = Path(project_root).resolve()
    config_path = _within_root(
        request.get("config", "configs/daily_three_session.toml"), root,
        must_exist=True, want_dir=False,
    )
    config_values = tomllib.loads(Path(config_path).read_text(encoding="utf-8"))
    return resolve_risk_profile(
        risk_template=request.get("risk_template"),
        risk_expression=request.get("risk_expression"),
        risk_constraints=request.get("risk_constraints"),
        risk_budget_annualized=request.get("risk_budget_annualized"),
        base_defaults={
            "risk_budget_annualized": config_values.get("risk_budget_annualized", 0.20),
            "maximum_symbol_weight": config_values.get("maximum_symbol_weight", 0.05),
            "maximum_one_way_turnover": config_values.get("maximum_one_way_turnover", 0.50),
            "minimum_cash_weight": config_values.get("minimum_cash_weight", 0.0),
        },
    )


def validate_decision_job(request: Mapping[str, object], project_root: str | Path) -> dict[str, object]:
    """Accept only an offline artifact-reuse decision; the worker cannot make provider calls."""
    root = Path(project_root).resolve()
    allowed = {
        "mode", "as_of", "horizon_sessions", "risk_budget_annualized", "risk_template",
        "risk_expression", "risk_constraints", "baostock_run_dir",
        "portfolio", "training_seed", "training_seed_trained_through", "training_seed_horizon_sessions",
        "reuse_research_json", "config", "public_cache_dir", "harness_version_dir",
    }
    extra = set(request) - allowed
    if extra:
        raise ValueError(f"unsupported request fields: {sorted(extra)}")
    if request.get("mode") != "decision_reuse":
        raise ValueError("local service accepts decision_reuse jobs only; provider execution is disabled")
    try:
        as_of = date.fromisoformat(str(request["as_of"]))
        trained_through = date.fromisoformat(str(request["training_seed_trained_through"]))
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("as_of and training_seed_trained_through must be ISO dates") from exc
    horizon = request.get("horizon_sessions")
    seed_horizon = request.get("training_seed_horizon_sessions")
    if not isinstance(horizon, int) or isinstance(horizon, bool) or horizon < 1:
        raise ValueError("horizon_sessions must be a positive integer")
    if seed_horizon != horizon:
        raise ValueError("training seed horizon must match horizon_sessions")
    if trained_through >= as_of:
        raise ValueError("training seed cutoff must precede as_of")
    portfolio = request.get("portfolio")
    if not isinstance(portfolio, Mapping) or set(portfolio) != {"cash_cny", "holdings_cny"}:
        raise ValueError("portfolio must contain only cash_cny and holdings_cny")
    holdings = portfolio.get("holdings_cny")
    if not isinstance(holdings, Mapping):
        raise ValueError("portfolio holdings_cny must be an object")
    normalized_holdings = {}
    for symbol, amount in holdings.items():
        if not re.fullmatch(r"[0-9]{6}", str(symbol)):
            raise ValueError("portfolio symbol must be a six-digit string")
        number = float(amount)
        if not math.isfinite(number) or number < 0:
            raise ValueError("portfolio holdings must be finite nonnegative CNY values")
        normalized_holdings[str(symbol)] = number
    cash = float(portfolio.get("cash_cny"))
    if not math.isfinite(cash) or cash < 0:
        raise ValueError("portfolio cash must be finite and nonnegative")

    required_paths = {
        "baostock_run_dir": True,
        "training_seed": False,
        "reuse_research_json": False,
    }
    normalized: dict[str, object] = {
        "mode": "decision_reuse",
        "as_of": as_of.isoformat(),
        "horizon_sessions": horizon,
        "portfolio": {"cash_cny": cash, "holdings_cny": normalized_holdings},
        "training_seed_trained_through": trained_through.isoformat(),
        "training_seed_horizon_sessions": horizon,
    }
    for key, want_dir in required_paths.items():
        normalized[key] = _within_root(request.get(key), root, must_exist=True, want_dir=want_dir)
    for key, default, want_dir in (
        ("config", "configs/daily_three_session.toml", False),
        ("public_cache_dir", "data/raw/akshare", True),
        ("harness_version_dir", "data/processed/harness_versions", True),
    ):
        raw = request.get(key, default)
        if raw is not None:
            normalized[key] = _within_root(raw, root, must_exist=True, want_dir=want_dir)
    profile = parse_risk_profile_request(
        {
            key: request[key]
            for key in ("risk_template", "risk_expression", "risk_constraints", "risk_budget_annualized")
            if key in request
        } | {"config": normalized["config"]},
        root,
    )
    if not profile["execution_allowed"]:
        raise ValueError("risk profile needs clarification: " + " ".join(profile["unresolved"]))
    normalized["risk_profile"] = profile
    normalized["risk_budget_annualized"] = profile["constraints"]["risk_budget_annualized"]
    normalized["minimum_cash_weight"] = profile["constraints"]["minimum_cash_weight"]
    return normalized


def validate_execution_ledger_request(request: Mapping[str, object]) -> dict[str, object]:
    """Check the local replay contract before it reaches the deterministic ledger."""
    allowed = {
        "execution_mode", "trading_dates", "market_bars", "orders", "rule_pack",
        "initial_cash_cny", "initial_lots", "initial_holdings_cny", "corporate_actions",
        "source_artifact_hashes", "prediction_records", "prediction_linkage_mode", "risk_limits",
        "research_cost_cny_by_date",
    }
    extra = set(request) - allowed
    required = {
        "execution_mode", "trading_dates", "market_bars", "orders", "rule_pack",
        "initial_cash_cny", "prediction_linkage_mode",
    }
    missing = required - set(request)
    if extra or missing:
        raise ValueError(f"execution ledger fields invalid; missing={sorted(missing)}, unsupported={sorted(extra)}")
    if request.get("prediction_linkage_mode") not in {"linked", "independent"}:
        raise ValueError("prediction_linkage_mode must be linked or independent")
    if not isinstance(request.get("trading_dates"), list) or not isinstance(request.get("market_bars"), list) or not isinstance(request.get("orders"), list):
        raise ValueError("trading_dates, market_bars, and orders must be arrays")
    for key in (
        "initial_lots", "initial_holdings_cny", "rule_pack", "source_artifact_hashes",
        "risk_limits", "research_cost_cny_by_date",
    ):
        value = request.get(key)
        if value is not None and not isinstance(value, Mapping):
            raise ValueError(f"{key} must be an object")
    for key in ("corporate_actions", "prediction_records"):
        value = request.get(key, [])
        if not isinstance(value, list):
            raise ValueError(f"{key} must be an array")
    return dict(request)


def _replay_runner(
    job: Mapping[str, object],
    *,
    project_root: Path,
    database_path: Path,
    control_check: Callable[[], str | None] | None = None,
) -> dict[str, object]:
    request = job["request"]
    assert isinstance(request, Mapping)
    root = project_root.resolve()
    run_root = database_path.parent
    artifact_dir = run_root / "service_artifacts" / str(job["task_id"])
    artifact_dir.mkdir(parents=True, exist_ok=True)
    portfolio_path = artifact_dir / "portfolio_state.json"
    risk_profile_path = artifact_dir / "risk_profile.json"
    runtime_binding_path = artifact_dir / "runtime_binding.json"
    output_path = artifact_dir / "decision.json"
    portfolio_path.write_text(
        json.dumps(request["portfolio"], ensure_ascii=False, indent=2), encoding="utf-8"
    )
    risk_profile_path.write_text(
        json.dumps(request["risk_profile"], ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )
    runtime_binding = request.get("runtime_binding")
    materialization: Mapping[str, object] | None = None
    source_root = root
    request_paths: dict[str, str | Path] = {
        key: str(request[key])
        for key in (
            "baostock_run_dir", "config", "public_cache_dir", "reuse_research_json", "training_seed",
        )
    }
    if isinstance(runtime_binding, Mapping):
        runtime_binding_path.write_text(
            json.dumps(runtime_binding, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
        )
        raw_materialization = runtime_binding.get("materialization")
        if raw_materialization is not None:
            if not isinstance(raw_materialization, Mapping):
                raise ValueError("queued workflow materialization descriptor is invalid")
            materialization = raw_materialization
            verify_workflow_materialization(materialization, project_root=root)
            source_root = Path(str(materialization["source_root"]))
            materialized_inputs = materialization.get("inputs")
            if not isinstance(materialized_inputs, Mapping):
                raise ValueError("queued workflow materialization has no input map")
            if set(materialized_inputs) != set(request_paths):
                raise ValueError("queued workflow materialization is missing required inputs")
            request_paths = {
                key: str(materialized_inputs[key]["path"])
                for key in request_paths
                if isinstance(materialized_inputs[key], Mapping)
            }
            if len(request_paths) != 5:
                raise ValueError("queued workflow materialized input paths are invalid")
    harness_version_dir = (
        str(materialization["runtime_package_store_root"])
        if materialization is not None else str(request["harness_version_dir"])
    )
    command = [
        sys.executable,
        str(source_root / "scripts" / "run_harness_decision.py"),
        "--baostock-run-dir", str(request_paths["baostock_run_dir"]),
        "--portfolio-state-json", str(portfolio_path),
        "--as-of", str(request["as_of"]),
        "--horizon-sessions", str(request["horizon_sessions"]),
        "--risk-profile-json", str(risk_profile_path),
        "--config", str(request_paths["config"]),
        "--training-seed", str(request_paths["training_seed"]),
        "--training-seed-trained-through", str(request["training_seed_trained_through"]),
        "--training-seed-horizon-sessions", str(request["training_seed_horizon_sessions"]),
        "--public-cache-dir", str(request_paths["public_cache_dir"]),
        "--harness-version-dir", harness_version_dir,
        "--run-dir", str(run_root),
        "--task-id", str(job["task_id"]),
        "--reuse-research-json", str(request_paths["reuse_research_json"]),
        "--output", str(output_path),
    ]
    if isinstance(runtime_binding, Mapping):
        package = runtime_binding["runtime_package"]
        assert isinstance(package, Mapping)
        command.extend((
            "--workflow-binding-json", str(runtime_binding_path),
            "--pinned-runtime-package-id", str(package["package_id"]),
        ))
        if package.get("package_sha256") is not None:
            command.extend(("--pinned-runtime-package-sha256", str(package["package_sha256"])))
        if materialization is not None:
            command.extend(("--workflow-project-root", str(root)))
    environment = dict(os.environ)
    for name in tuple(environment):
        if name.upper() in {
            "QWEN_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "POPAGENT_LLM_KEY"
        } or name.upper().endswith("_API_KEY"):
            environment.pop(name, None)
    if materialization is not None:
        snapshot_pythonpath = str(source_root / "src")
        existing_pythonpath = environment.get("PYTHONPATH")
        environment["PYTHONPATH"] = os.pathsep.join(
            [snapshot_pythonpath, *([existing_pythonpath] if existing_pythonpath else [])]
        )
    process = subprocess.Popen(
        command,
        cwd=source_root,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    started_at = time.monotonic()
    while True:
        try:
            stdout, stderr = process.communicate(timeout=0.25)
            break
        except subprocess.TimeoutExpired:
            if control_check is not None:
                control = control_check()
                if control is not None:
                    process.terminate()
                    try:
                        process.communicate(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.communicate()
                    if control == "lease_lost":
                        raise WorkflowControlRequested("lease_lost")
                    raise WorkflowControlRequested(control)
            if time.monotonic() - started_at >= 1800:
                process.kill()
                stdout, stderr = process.communicate()
                raise RuntimeError(f"offline decision replay exceeded 1800 seconds: {(stderr or stdout)[-4000:]}")
    if process.returncode != 0:
        error = (stderr or stdout or "decision replay failed")[-4000:]
        raise RuntimeError(f"offline decision replay exited {process.returncode}: {error}")
    try:
        result = json.loads(output_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("offline worker completed without a valid decision artifact") from exc
    if not isinstance(result, dict):
        raise RuntimeError("offline worker result must be a JSON object")
    return result


class HarnessJobWorker:
    def __init__(
        self,
        store: HarnessServiceStore,
        *,
        project_root: str | Path,
        worker_id: str | None = None,
        runner: Callable[[Mapping[str, object]], Mapping[str, object]] | None = None,
    ) -> None:
        self.store = store
        self.project_root = Path(project_root).resolve()
        self.worker_id = worker_id or f"worker-{uuid.uuid4().hex[:10]}"
        self.runner = runner

    def run_once(self) -> bool:
        job = self.store.claim_next(self.worker_id)
        if job is None:
            return False
        task_id = str(job["task_id"])
        lease_token = str(job["lease_token"])
        started = time.monotonic()
        lease_stop = threading.Event()
        lease_thread: threading.Thread | None = None
        try:
            self.store.update_stage(
                task_id, "running_replay", worker_id=self.worker_id, lease_token=lease_token,
            )

            def keep_lease() -> None:
                while not lease_stop.wait(30.0):
                    try:
                        self.store.renew_lease(task_id, self.worker_id, lease_token=lease_token)
                    except WorkerLeaseLost:
                        return

            lease_thread = threading.Thread(target=keep_lease, daemon=True)
            lease_thread.start()
            if self.store.acknowledge_control(task_id, self.worker_id, lease_token) is not None:
                return True
            if self.runner is None:
                result = _replay_runner(
                    job,
                    project_root=self.project_root,
                    database_path=self.store.database_path,
                    control_check=lambda: self.store.current_control(task_id, self.worker_id, lease_token),
                )
            else:
                result = dict(self.runner(job))
            if self.store.acknowledge_control(task_id, self.worker_id, lease_token) is not None:
                return True
            self.store.update_stage(
                task_id, "persisting_result", worker_id=self.worker_id, lease_token=lease_token,
            )
            self.store.complete(
                task_id, self.worker_id, lease_token, result,
                elapsed_ms=(time.monotonic() - started) * 1000,
            )
        except WorkflowControlRequested as exc:
            if exc.action != "lease_lost":
                self.store.acknowledge_control(task_id, self.worker_id, lease_token)
        except WorkerLeaseLost:
            pass
        except Exception as exc:
            try:
                self.store.fail(
                    task_id,
                    self.worker_id,
                    lease_token,
                    {"error_type": type(exc).__name__, "message": str(exc)[:1000]},
                )
            except WorkflowControlRequested:
                self.store.acknowledge_control(task_id, self.worker_id, lease_token)
            except WorkerLeaseLost:
                pass
        finally:
            lease_stop.set()
            if lease_thread is not None:
                lease_thread.join(timeout=1)
        return True

    def run_forever(self, *, poll_seconds: float = 1.0, stop_event: threading.Event | None = None) -> None:
        stop_event = stop_event or threading.Event()
        while not stop_event.is_set():
            if not self.run_once():
                stop_event.wait(poll_seconds)


def _redact_request_for_ui(request: Mapping[str, object]) -> dict[str, object]:
    result = {
        key: value for key, value in request.items()
        if key not in {"portfolio", "baostock_run_dir", "training_seed", "reuse_research_json", "config", "public_cache_dir", "harness_version_dir", "runtime_binding"}
    }
    for key in ("baostock_run_dir", "training_seed", "reuse_research_json", "config"):
        value = request.get(key)
        if isinstance(value, str):
            result[f"{key}_name"] = Path(value).name
    return result


class HarnessRequestHandler(BaseHTTPRequestHandler):
    server_version = "RAAMAHarness/1"
    protocol_version = "HTTP/1.1"

    def _context(self) -> dict[str, object]:
        return self.server.context  # type: ignore[attr-defined]

    def _local_only(self) -> bool:
        try:
            if not ipaddress.ip_address(self.client_address[0]).is_loopback:
                self._send_json(403, {"error": "loopback clients only"})
                return False
        except ValueError:
            self._send_json(403, {"error": "loopback clients only"})
            return False
        if not _is_loopback_host_header(self.headers.get("Host", "")):
            self._send_json(403, {"error": "invalid local host"})
            return False
        return True

    def _send_json(self, status: int, payload: Mapping[str, object]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict[str, object]:
        length = int(self.headers.get("Content-Length", "0"))
        if length < 1 or length > 1_000_000:
            raise ValueError("request body must be between 1 byte and 1 MB")
        value = json.loads(self.rfile.read(length).decode("utf-8"))
        if not isinstance(value, dict):
            raise ValueError("request JSON must be an object")
        return value

    def _job_payload(self, job: Mapping[str, object], *, detail: bool = False) -> dict[str, object]:
        request = job.get("request", {})
        result = job.get("result")
        response: dict[str, object] = {
            "task_id": job["task_id"],
            "status": job["status"],
            "stage": job["stage"],
            "attempt_count": job["attempt_count"],
            "created_at": job["created_at"],
            "updated_at": job["updated_at"],
            "request": _redact_request_for_ui(request) if isinstance(request, Mapping) else {},
            "result_summary": decision_summary(result) if isinstance(result, Mapping) else None,
            "error": job.get("error"),
        }
        if detail:
            response["result"] = result
            response["events"] = self._context()["store"].events(str(job["task_id"]))  # type: ignore[union-attr]
        return response

    def do_GET(self) -> None:
        if not self._local_only():
            return
        parsed = urlparse(self.path)
        path = parsed.path
        context = self._context()
        store: HarnessServiceStore = context["store"]  # type: ignore[assignment]
        if path == "/":
            body = (Path(__file__).parent / "web" / "index.html").read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)
            return
        if path == "/api/health":
            workers = store.worker_status()
            self._send_json(200, {
                "ok": True,
                "provider_execution": "disabled; cached research replay only",
                "workers": workers,
                "worker_online": any(worker["state"] in {"idle", "busy"} for worker in workers),
                "workflow_reconciliation_enabled": bool(
                    context.get("workflow_operator_id") and context.get("workflow_operator_token")
                ),
                "timestamp": _now(),
            })
            return
        if path == "/api/tasks":
            jobs = store.list_jobs()
            self._send_json(200, {"tasks": [self._job_payload(job) for job in jobs]})
            return
        if path == "/api/execution-ledgers":
            self._send_json(200, {"ledgers": store.list_execution_ledgers()})
            return
        if path.startswith("/api/execution-ledgers/"):
            ledger_id = path.removeprefix("/api/execution-ledgers/")
            if not ledger_id or "/" in ledger_id:
                self._send_json(404, {"error": "execution ledger not found"})
                return
            ledger = store.get_execution_ledger(ledger_id)
            if ledger is None:
                self._send_json(404, {"error": "execution ledger not found"})
                return
            self._send_json(200, ledger)
            return
        if path.startswith("/api/tasks/"):
            task_id = path.removeprefix("/api/tasks/")
            if not task_id or "/" in task_id:
                self._send_json(404, {"error": "task not found"})
                return
            job = store.get_job(task_id)
            if job is None:
                self._send_json(404, {"error": "task not found"})
                return
            query = parse_qs(parsed.query)
            self._send_json(200, self._job_payload(job, detail=query.get("detail") == ["1"]))
            return
        if path == "/api/releases":
            release_store: HarnessReleaseStore = context["release_store"]  # type: ignore[assignment]
            self._send_json(200, {"active": release_store.status_summary(), "history": release_store.audit_history()})
            return
        if path == "/api/evaluations":
            release_store: HarnessReleaseStore = context["release_store"]  # type: ignore[assignment]
            self._send_json(200, release_store.evaluation_status())
            return
        self._send_json(404, {"error": "not found"})

    def do_POST(self) -> None:
        if not self._local_only():
            return
        context = self._context()
        path = urlparse(self.path).path
        try:
            payload = self._read_json()
            if path == "/api/risk-profile/parse":
                profile_request = payload.get("request", payload)
                if not isinstance(profile_request, Mapping):
                    raise ValueError("request must contain a risk profile object")
                profile = parse_risk_profile_request(profile_request, context["project_root"])
                self._send_json(200, profile)
                return
            if path == "/api/tasks":
                request = validate_decision_job(payload.get("request", payload), context["project_root"])
                idempotency_key = str(payload.get("idempotency_key", "")).strip()
                if not re.fullmatch(r"[A-Za-z0-9._:-]{1,128}", idempotency_key):
                    raise ValueError("idempotency_key must use letters, digits, dot, underscore, colon, or hyphen")
                task_id = str(payload.get("task_id") or uuid.uuid4().hex)
                if not re.fullmatch(r"[A-Za-z0-9._-]{1,96}", task_id):
                    raise ValueError("task_id is not a safe identifier")
                service_store: HarnessServiceStore = context["store"]  # type: ignore[assignment]
                if service_store.get_job_by_idempotency_key(idempotency_key) is not None:
                    job = service_store.submit(task_id, idempotency_key, request)
                    self._send_json(202, self._job_payload(job))
                    return
                runtime_package = HarnessReleaseStore(
                    str(request["harness_version_dir"]),
                    project_root=context["project_root"],
                ).pin_active_runtime_package()
                root = Path(context["project_root"]).resolve()
                snapshot_root = (
                    Path(service_store.database_path).parent
                    / "service_artifacts" / f"ws-{uuid.uuid4().hex[:12]}"
                )
                try:
                    materialization = materialize_workflow_snapshot(
                        request,
                        project_root=root,
                        snapshot_root=snapshot_root,
                        runtime_package=runtime_package,
                        runtime_package_store_root=request["harness_version_dir"],
                    )
                    request["runtime_binding"] = build_workflow_runtime_binding(
                        request,
                        project_root=root,
                        runtime_package=runtime_package,
                        materialization=materialization,
                    )
                    job = service_store.submit(task_id, idempotency_key, request)
                except Exception:
                    if snapshot_root.exists():
                        shutil.rmtree(snapshot_root, ignore_errors=True)
                    raise
                stored_binding = job.get("request", {}).get("runtime_binding", {})
                stored_materialization = (
                    stored_binding.get("materialization")
                    if isinstance(stored_binding, Mapping) else None
                )
                if (
                    isinstance(stored_materialization, Mapping)
                    and stored_materialization.get("snapshot_root") != str(snapshot_root.resolve())
                    and snapshot_root.exists()
                ):
                    shutil.rmtree(snapshot_root, ignore_errors=True)
                self._send_json(202, self._job_payload(job))
                return
            if path.startswith("/api/tasks/") and path.endswith("/control"):
                task_id = path.removeprefix("/api/tasks/").removesuffix("/control").strip("/")
                if not re.fullmatch(r"[A-Za-z0-9._-]{1,96}", task_id):
                    raise ValueError("task_id is not a safe identifier")
                if set(payload) != {"action", "reason"}:
                    raise ValueError("task control requires action and reason")
                action = payload["action"]
                reason = payload["reason"]
                if not isinstance(action, str) or action not in {"pause", "resume", "cancel"}:
                    raise ValueError("action must be pause, resume, or cancel")
                if not isinstance(reason, str) or not reason.strip():
                    raise ValueError("reason must be a nonempty string")
                operator_id = self._require_workflow_operator(context)
                result = context["store"].request_control(  # type: ignore[union-attr]
                    task_id, action=action, operator_id=operator_id, reason=reason,
                )
                self._send_json(200, result)
                return
            if path.startswith("/api/tasks/") and path.endswith("/dead-letter"):
                task_id = path.removeprefix("/api/tasks/").removesuffix("/dead-letter").strip("/")
                if not re.fullmatch(r"[A-Za-z0-9._-]{1,96}", task_id):
                    raise ValueError("task_id is not a safe identifier")
                if set(payload) != {"expected_attempt", "reason"}:
                    raise ValueError("dead-letter resolution requires expected_attempt and reason")
                expected_attempt = payload["expected_attempt"]
                if isinstance(expected_attempt, bool) or not isinstance(expected_attempt, int):
                    raise ValueError("expected_attempt must be an integer")
                reason = payload["reason"]
                if not isinstance(reason, str) or not reason.strip():
                    raise ValueError("reason must be a nonempty string")
                operator_id = self._require_workflow_operator(context)
                result = context["store"].dead_letter_unknown_job(  # type: ignore[union-attr]
                    task_id,
                    expected_attempt=expected_attempt,
                    operator_id=operator_id,
                    reason=reason,
                )
                self._send_json(200, result)
                return
            if path.startswith("/api/tasks/") and path.endswith("/reconcile"):
                task_id = path.removeprefix("/api/tasks/").removesuffix("/reconcile").strip("/")
                if not re.fullmatch(r"[A-Za-z0-9._-]{1,96}", task_id):
                    raise ValueError("task_id is not a safe identifier")
                if set(payload) != {"action", "expected_attempt", "reason"}:
                    raise ValueError("reconciliation requires action, expected_attempt, and reason")
                if payload["action"] != "retry_after_confirmed_no_external_effect":
                    raise ValueError("unsupported reconciliation action")
                expected_attempt = payload["expected_attempt"]
                if isinstance(expected_attempt, bool) or not isinstance(expected_attempt, int):
                    raise ValueError("expected_attempt must be an integer")
                reason = payload["reason"]
                if not isinstance(reason, str) or not reason.strip():
                    raise ValueError("reason must be a nonempty string")
                operator_id = self._require_workflow_operator(context)
                result = context["store"].reconcile_unknown_job(  # type: ignore[union-attr]
                    task_id,
                    expected_attempt=expected_attempt,
                    operator_id=operator_id,
                    reason=reason,
                )
                self._send_json(200, result)
                return
            if path == "/api/execution-ledgers":
                ledger_id = str(payload.get("ledger_id", "")).strip()
                if not re.fullmatch(r"[A-Za-z0-9._-]{1,96}", ledger_id):
                    raise ValueError("ledger_id must be a safe immutable identifier")
                request = payload.get("request")
                if not isinstance(request, Mapping):
                    raise ValueError("request must contain the execution-ledger input object")
                normalized = validate_execution_ledger_request(request)
                result = simulate_execution_ledger(**normalized)
                saved = context["store"].save_execution_ledger(  # type: ignore[union-attr]
                    ledger_id, stable_hash(normalized), result
                )
                self._send_json(200 if saved["reused"] else 201, saved)
                return
            if path == "/api/evaluations":
                release_store: HarnessReleaseStore = context["release_store"]  # type: ignore[assignment]
                action = payload.get("action")
                if action == "predeclare":
                    if set(payload) != {"action", "protocol_id", "protocol"} or not isinstance(payload.get("protocol"), Mapping):
                        raise ValueError("predeclare requires only protocol_id and a protocol object")
                    result = release_store.predeclare_protocol(str(payload["protocol_id"]), payload["protocol"])
                elif action == "register_package":
                    if set(payload) == {"action", "package"} and isinstance(payload.get("package"), Mapping):
                        result = release_store.register_package(payload["package"])
                    elif set(payload) == {"action", "package_path"}:
                        result = release_store.register_package_file(str(payload["package_path"]))
                    else:
                        raise ValueError("register_package requires a complete package object or project-local package_path")
                elif action == "evaluate":
                    expected = {"action", "protocol_id", "partition", "run_id", "package_id", "input_artifact_path"}
                    if set(payload) != expected:
                        raise ValueError("evaluate requires protocol_id, partition, immutable run_id, package_id, and input_artifact_path")
                    result = release_store.evaluate_partition(
                        protocol_id=str(payload["protocol_id"]),
                        partition=str(payload["partition"]),
                        run_id=str(payload["run_id"]),
                        package_id=str(payload["package_id"]),
                        input_artifact_path=str(payload["input_artifact_path"]),
                    )
                else:
                    raise ValueError("action must be predeclare, register_package, or evaluate")
                self._send_json(200, result)
                return
            if path == "/api/releases/publish":
                self._require_release_manager(context)
                release_store: HarnessReleaseStore = context["release_store"]  # type: ignore[assignment]
                bundle = release_store.publish_release(
                    package_id=str(payload["package_id"]),
                    evidence_ids=payload["evidence_ids"],
                    actor_role="release_manager",
                    reason=str(payload["reason"]),
                    release_id=payload.get("release_id"),
                )
                self._send_json(201, {
                    "release_id": bundle["release_id"],
                    "bundle_sha256": bundle["bundle_sha256"],
                    "evidence_status": bundle["evidence_status"],
                })
                return
            if path == "/api/releases/rollback":
                self._require_release_manager(context)
                release_store: HarnessReleaseStore = context["release_store"]  # type: ignore[assignment]
                result = release_store.rollback(
                    str(payload["release_id"]),
                    config=payload.get("config", {}),
                    actor_role="release_manager",
                    reason=str(payload["reason"]),
                    training_state=payload.get("training_state"),
                    training_seed_provenance=payload.get("training_seed_provenance"),
                    versions=payload.get("versions"),
                )
                self._send_json(200, result)
                return
        except ServiceConflict as exc:
            self._send_json(409, {"error": str(exc)})
            return
        except PermissionError as exc:
            self._send_json(403, {"error": str(exc)})
            return
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            self._send_json(400, {"error": str(exc)})
            return
        except (RuntimeError, FileNotFoundError) as exc:
            self._send_json(422, {"error": str(exc)[:1000]})
            return
        self._send_json(404, {"error": "not found"})

    def _require_release_manager(self, context: Mapping[str, object]) -> None:
        expected = context.get("release_manager_token")
        if not isinstance(expected, str) or not expected:
            raise PermissionError("release manager is unavailable until RAAMA_RELEASE_MANAGER_TOKEN is configured")
        supplied = self.headers.get("Authorization", "")
        prefix = "Bearer "
        token = supplied[len(prefix):] if supplied.startswith(prefix) else ""
        if not hmac.compare_digest(token, expected):
            raise PermissionError("release manager authorization failed")

    def _require_workflow_operator(self, context: Mapping[str, object]) -> str:
        expected = context.get("workflow_operator_token")
        operator_id = context.get("workflow_operator_id")
        if not isinstance(expected, str) or not expected or not isinstance(operator_id, str) or not operator_id:
            raise PermissionError("workflow reconciliation is not enabled")
        supplied = self.headers.get("Authorization", "")
        prefix = "Bearer "
        token = supplied[len(prefix):] if supplied.startswith(prefix) else ""
        if not hmac.compare_digest(token, expected):
            raise PermissionError("workflow operator authorization failed")
        return operator_id

    def log_message(self, format: str, *args: object) -> None:
        # Keep the local console useful without reflecting query contents or account data.
        print(f"[raama-api] {self.client_address[0]} {format % args}")


class LocalHarnessHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, server_address, RequestHandlerClass, bind_and_activate=True):
        if isinstance(server_address, tuple) and server_address[0] == "::1":
            self.address_family = socket.AF_INET6
        super().__init__(server_address, RequestHandlerClass, bind_and_activate)


def serve(
    *,
    database_path: str | Path,
    project_root: str | Path,
    version_dir: str | Path,
    host: str = "127.0.0.1",
    port: int = 8765,
    release_manager_token: str | None = None,
    workflow_operator_id: str | None = None,
    workflow_operator_token: str | None = None,
) -> None:
    if host not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("the API server must bind to a loopback interface")
    store = HarnessServiceStore(database_path)
    release_store = HarnessReleaseStore(version_dir, project_root=project_root, task_store_path=database_path)
    server = LocalHarnessHTTPServer((host, port), HarnessRequestHandler)
    server.context = {
        "store": store,
        "release_store": release_store,
        "project_root": Path(project_root).resolve(),
        "release_manager_token": release_manager_token,
        "workflow_operator_id": workflow_operator_id,
        "workflow_operator_token": workflow_operator_token,
    }
    print(f"RAAMA 本地工作台已启动：http://{host}:{port}")
    print("离线研究复用已启用；provider 请求与 broker 执行均关闭。")
    try:
        server.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def run_worker(
    *,
    database_path: str | Path,
    project_root: str | Path,
    poll_seconds: float = 1.0,
    worker_id: str | None = None,
) -> None:
    worker = HarnessJobWorker(
        HarnessServiceStore(database_path),
        project_root=project_root,
        worker_id=worker_id,
    )
    print(f"RAAMA worker {worker.worker_id} 已就绪。")
    stop = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_args: stop.set())
    worker.run_forever(poll_seconds=poll_seconds, stop_event=stop)


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    run_root = project_root / "data" / "processed" / "harness_service"
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    serve_parser = subparsers.add_parser("serve", help="start the Chinese loopback workbench API")
    serve_parser.add_argument("--database", type=Path, default=run_root / "harness_state.sqlite3")
    serve_parser.add_argument("--project-root", type=Path, default=project_root)
    serve_parser.add_argument("--version-dir", type=Path, default=project_root / "data" / "processed" / "harness_versions")
    serve_parser.add_argument("--host", default="127.0.0.1")
    serve_parser.add_argument("--port", type=int, default=8765)
    worker_parser = subparsers.add_parser("worker", help="run the independent offline decision worker")
    worker_parser.add_argument("--database", type=Path, default=run_root / "harness_state.sqlite3")
    worker_parser.add_argument("--project-root", type=Path, default=project_root)
    worker_parser.add_argument("--poll-seconds", type=float, default=1.0)
    worker_parser.add_argument("--worker-id")
    args = parser.parse_args()
    if args.command == "serve":
        serve(
            database_path=args.database,
            project_root=args.project_root,
            version_dir=args.version_dir,
            host=args.host,
            port=args.port,
            release_manager_token=os.environ.get("RAAMA_RELEASE_MANAGER_TOKEN"),
            workflow_operator_id=os.environ.get("RAAMA_WORKFLOW_OPERATOR_ID"),
            workflow_operator_token=os.environ.get("RAAMA_WORKFLOW_OPERATOR_TOKEN"),
        )
    else:
        run_worker(
            database_path=args.database,
            project_root=args.project_root,
            poll_seconds=max(0.1, args.poll_seconds),
            worker_id=args.worker_id,
        )


if __name__ == "__main__":
    main()
