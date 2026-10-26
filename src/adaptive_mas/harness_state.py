"""Transactional local state for resumable public-evidence harness tasks."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Mapping


class StateConflict(RuntimeError):
    """An idempotency key was reused with different immutable input."""


class StateBudgetExhausted(RuntimeError):
    """A task or shared experiment has no reserved provider calls remaining."""


class UnknownOutboundOutcome(RuntimeError):
    """A request may have reached the provider and requires explicit reconciliation."""


def _json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _digest(value: object) -> str:
    return hashlib.sha256(_json(value).encode("utf-8")).hexdigest()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class HarnessStateStore:
    """SQLite-backed task, request, feedback and execution state with idempotent writes."""

    def __init__(self, path: str | Path, *, timeout_seconds: float = 30.0) -> None:
        self.path = Path(path)
        self.timeout_seconds = timeout_seconds
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connection() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS tasks (
                    task_id TEXT PRIMARY KEY,
                    idempotency_key TEXT NOT NULL UNIQUE,
                    request_hash TEXT NOT NULL,
                    status TEXT NOT NULL,
                    stage TEXT NOT NULL,
                    request_json TEXT NOT NULL,
                    result_json TEXT,
                    error_json TEXT,
                    max_provider_calls INTEGER NOT NULL,
                    provider_calls_used INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS budget_scopes (
                    scope_id TEXT PRIMARY KEY,
                    maximum_calls INTEGER NOT NULL,
                    calls_used INTEGER NOT NULL
                );
                CREATE TABLE IF NOT EXISTS provider_requests (
                    task_id TEXT NOT NULL REFERENCES tasks(task_id) ON DELETE CASCADE,
                    request_hash TEXT NOT NULL,
                    status TEXT NOT NULL,
                    attempt_count INTEGER NOT NULL DEFAULT 1,
                    response_json TEXT,
                    response_hash TEXT,
                    usage_json TEXT,
                    logical_counted INTEGER NOT NULL DEFAULT 0,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY(task_id, request_hash)
                );
                CREATE TABLE IF NOT EXISTS provider_attempts (
                    attempt_id TEXT PRIMARY KEY,
                    task_id TEXT NOT NULL REFERENCES tasks(task_id) ON DELETE CASCADE,
                    scope_id TEXT NOT NULL,
                    request_hash TEXT NOT NULL,
                    attempt_number INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    UNIQUE(task_id, request_hash, attempt_number)
                );
                CREATE TABLE IF NOT EXISTS task_artifacts (
                    task_id TEXT NOT NULL REFERENCES tasks(task_id) ON DELETE CASCADE,
                    name TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    payload_hash TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY(task_id, name)
                );
                CREATE TABLE IF NOT EXISTS task_events (
                    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    task_id TEXT NOT NULL REFERENCES tasks(task_id) ON DELETE CASCADE,
                    event_key TEXT NOT NULL,
                    stage TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    UNIQUE(task_id, event_key)
                );
                CREATE TABLE IF NOT EXISTS feedback (
                    task_id TEXT NOT NULL REFERENCES tasks(task_id) ON DELETE CASCADE,
                    feedback_id TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    payload_hash TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY(task_id, feedback_id)
                );
                CREATE TABLE IF NOT EXISTS execution_results (
                    task_id TEXT NOT NULL REFERENCES tasks(task_id) ON DELETE CASCADE,
                    execution_id TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    payload_hash TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY(task_id, execution_id)
                );
                """
            )
            request_columns = {
                str(row["name"])
                for row in connection.execute("PRAGMA table_info(provider_requests)").fetchall()
            }
            if "logical_counted" not in request_columns:
                connection.execute(
                    "ALTER TABLE provider_requests ADD COLUMN logical_counted INTEGER NOT NULL DEFAULT 0"
                )

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(
            self.path,
            timeout=self.timeout_seconds,
            isolation_level=None,
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute(f"PRAGMA busy_timeout={int(self.timeout_seconds * 1000)}")
        try:
            yield connection
        finally:
            connection.close()

    @staticmethod
    def _begin(connection: sqlite3.Connection) -> None:
        connection.execute("BEGIN IMMEDIATE")

    @staticmethod
    def _task_dict(row: sqlite3.Row | None) -> dict[str, object] | None:
        if row is None:
            return None
        result = dict(row)
        for key in ("request_json", "result_json", "error_json"):
            if result.get(key) is not None:
                result[key.removesuffix("_json")] = json.loads(result[key])
                del result[key]
        return result

    def open_task(
        self,
        task_id: str,
        idempotency_key: str,
        request: Mapping[str, object],
        *,
        max_provider_calls: int,
    ) -> dict[str, object]:
        if not task_id.strip() or not idempotency_key.strip():
            raise ValueError("task_id and idempotency_key must be nonempty")
        if max_provider_calls < 0:
            raise ValueError("max_provider_calls must be nonnegative")
        request_json = _json(dict(request))
        request_hash = hashlib.sha256(request_json.encode("utf-8")).hexdigest()
        now = _now()
        with self._connection() as connection:
            self._begin(connection)
            by_key = connection.execute(
                "SELECT * FROM tasks WHERE idempotency_key=?", (idempotency_key,)
            ).fetchone()
            if by_key is not None:
                if by_key["request_hash"] != request_hash:
                    raise StateConflict("idempotency key was reused with different task input")
                connection.commit()
                return self._task_dict(by_key)  # type: ignore[return-value]
            by_id = connection.execute(
                "SELECT * FROM tasks WHERE task_id=?", (task_id,)
            ).fetchone()
            if by_id is not None:
                if by_id["request_hash"] != request_hash or by_id["idempotency_key"] != idempotency_key:
                    raise StateConflict("task_id already belongs to a different request")
                connection.commit()
                return self._task_dict(by_id)  # type: ignore[return-value]
            connection.execute(
                """INSERT INTO tasks(
                    task_id,idempotency_key,request_hash,status,stage,request_json,
                    max_provider_calls,created_at,updated_at
                ) VALUES(?,?,?,?,?,?,?,?,?)""",
                (task_id, idempotency_key, request_hash, "running", "created", request_json,
                 max_provider_calls, now, now),
            )
            connection.execute(
                "INSERT INTO task_events(task_id,event_key,stage,payload_json,created_at) VALUES(?,?,?,?,?)",
                (task_id, "task_created", "created", "{}", now),
            )
            row = connection.execute("SELECT * FROM tasks WHERE task_id=?", (task_id,)).fetchone()
            connection.commit()
            return self._task_dict(row)  # type: ignore[return-value]

    def get_task(self, task_id: str) -> dict[str, object] | None:
        with self._connection() as connection:
            row = connection.execute("SELECT * FROM tasks WHERE task_id=?", (task_id,)).fetchone()
            return self._task_dict(row)

    def checkpoint_task(
        self, task_id: str, stage: str, payload: Mapping[str, object] | None = None,
        *, event_key: str | None = None,
    ) -> None:
        event_key = event_key or f"{stage}:{_digest(payload or {})}"
        payload_json = _json(dict(payload or {}))
        now = _now()
        with self._connection() as connection:
            self._begin(connection)
            row = connection.execute("SELECT status FROM tasks WHERE task_id=?", (task_id,)).fetchone()
            if row is None:
                raise KeyError(f"unknown task_id {task_id}")
            if row["status"] == "completed":
                connection.commit()
                return
            connection.execute(
                "UPDATE tasks SET stage=?,updated_at=? WHERE task_id=?",
                (stage, now, task_id),
            )
            connection.execute(
                """INSERT OR IGNORE INTO task_events(task_id,event_key,stage,payload_json,created_at)
                   VALUES(?,?,?,?,?)""",
                (task_id, event_key, stage, payload_json, now),
            )
            connection.commit()

    def transition_task_status(
        self,
        task_id: str,
        *,
        expected_statuses: set[str] | frozenset[str],
        status: str,
        stage: str,
        event_key: str,
        payload: Mapping[str, object] | None = None,
    ) -> bool:
        allowed = {"running", "paused", "cancelled", "failed", "unknown", "dead_letter"}
        if status not in allowed or not expected_statuses or not expected_statuses <= allowed:
            raise ValueError("task lifecycle transition contains an unsupported status")
        if not event_key.strip() or not stage.strip():
            raise ValueError("event_key and stage must be nonempty")
        payload_json = _json(dict(payload or {}))
        now = _now()
        with self._connection() as connection:
            self._begin(connection)
            row = connection.execute("SELECT status FROM tasks WHERE task_id=?", (task_id,)).fetchone()
            if row is None:
                raise KeyError(f"unknown task_id {task_id}")
            prior = connection.execute(
                "SELECT stage,payload_json FROM task_events WHERE task_id=? AND event_key=?",
                (task_id, event_key),
            ).fetchone()
            if prior is not None:
                if prior["payload_json"] != payload_json:
                    raise StateConflict("task lifecycle event key was reused with different content")
                if prior["stage"] not in {stage, f"{stage}_rejected"}:
                    raise StateConflict("task lifecycle event key was reused for another transition")
                connection.commit()
                return prior["stage"] == stage

            accepted = str(row["status"]) in expected_statuses
            event_stage = stage if accepted else f"{stage}_rejected"
            if accepted:
                connection.execute(
                    "UPDATE tasks SET status=?,stage=?,updated_at=? WHERE task_id=?",
                    (status, stage, now, task_id),
                )
            connection.execute(
                "INSERT INTO task_events(task_id,event_key,stage,payload_json,created_at) VALUES(?,?,?,?,?)",
                (task_id, event_key, event_stage, payload_json, now),
            )
            connection.commit()
            return accepted

    def dead_letter_task(
        self,
        task_id: str,
        *,
        expected_status: str,
        operator_id: str,
        reason: str,
        event_key: str,
    ) -> bool:
        if expected_status not in {"failed", "unknown"}:
            raise ValueError("only failed or unknown tasks can be dead-lettered")
        if not operator_id.strip() or not reason.strip():
            raise ValueError("operator identity and dead-letter reason are required")
        return self.transition_task_status(
            task_id,
            expected_statuses={expected_status},
            status="dead_letter",
            stage="dead_letter",
            event_key=event_key,
            payload={
                "expected_status": expected_status,
                "operator_id": operator_id,
                "reason": reason,
            },
        )

    def complete_task(
        self,
        task_id: str,
        result: Mapping[str, object],
        *,
        feedback_id: str | None = None,
        feedback: Mapping[str, object] | None = None,
    ) -> None:
        payload_json = _json(dict(result))
        feedback_json = _json(dict(feedback)) if feedback is not None else None
        feedback_hash = hashlib.sha256(feedback_json.encode("utf-8")).hexdigest() if feedback_json else None
        now = _now()
        with self._connection() as connection:
            self._begin(connection)
            row = connection.execute("SELECT status,result_json FROM tasks WHERE task_id=?", (task_id,)).fetchone()
            if row is None:
                raise KeyError(f"unknown task_id {task_id}")
            if row["status"] == "completed":
                if row["result_json"] != payload_json:
                    raise StateConflict("completed task result is immutable")
            if feedback_json is not None:
                if not feedback_id:
                    raise ValueError("feedback_id is required with feedback")
                prior_feedback = connection.execute(
                    "SELECT payload_hash FROM feedback WHERE task_id=? AND feedback_id=?",
                    (task_id, feedback_id),
                ).fetchone()
                if prior_feedback is not None and prior_feedback["payload_hash"] != feedback_hash:
                    raise StateConflict("feedback record id was reused with different content")
                connection.execute(
                    "INSERT OR IGNORE INTO feedback(task_id,feedback_id,payload_json,payload_hash,created_at) VALUES(?,?,?,?,?)",
                    (task_id, feedback_id, feedback_json, feedback_hash, now),
                )
            if row["status"] == "completed":
                connection.commit()
                return
            connection.execute(
                "UPDATE tasks SET status='completed',stage='completed',result_json=?,error_json=NULL,updated_at=? WHERE task_id=?",
                (payload_json, now, task_id),
            )
            connection.execute(
                "INSERT OR IGNORE INTO task_events(task_id,event_key,stage,payload_json,created_at) VALUES(?,?,?,?,?)",
                (task_id, "task_completed", "completed", "{}", now),
            )
            connection.commit()

    def fail_task(self, task_id: str, error: Mapping[str, object], *, unknown_outcome: bool = False) -> None:
        status = "unknown" if unknown_outcome else "failed"
        payload_json = _json(dict(error))
        now = _now()
        with self._connection() as connection:
            connection.execute(
                "UPDATE tasks SET status=?,error_json=?,updated_at=? WHERE task_id=? AND status!='completed'",
                (status, payload_json, now, task_id),
            )

    def get_artifact(self, task_id: str, name: str) -> dict[str, object] | None:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT payload_json FROM task_artifacts WHERE task_id=? AND name=?",
                (task_id, name),
            ).fetchone()
            return json.loads(row["payload_json"]) if row else None

    def list_task_events(self, task_id: str) -> list[dict[str, object]]:
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT event_id,event_key,stage,payload_json,created_at FROM task_events WHERE task_id=? ORDER BY event_id",
                (task_id,),
            ).fetchall()
            return [
                {
                    "event_id": int(row["event_id"]),
                    "event_key": row["event_key"],
                    "stage": row["stage"],
                    "payload": json.loads(row["payload_json"]),
                    "created_at": row["created_at"],
                }
                for row in rows
            ]

    def get_feedback(self, task_id: str, feedback_id: str) -> dict[str, object] | None:
        return self._get_immutable_record("feedback", task_id, "feedback_id", feedback_id)

    def get_execution_result(self, task_id: str, execution_id: str) -> dict[str, object] | None:
        return self._get_immutable_record("execution_results", task_id, "execution_id", execution_id)

    def _get_immutable_record(
        self, table: str, task_id: str, id_column: str, record_id: str
    ) -> dict[str, object] | None:
        if table not in {"feedback", "execution_results"} or id_column not in {"feedback_id", "execution_id"}:
            raise ValueError("unsupported immutable state record")
        with self._connection() as connection:
            row = connection.execute(
                f"SELECT payload_json FROM {table} WHERE task_id=? AND {id_column}=?",
                (task_id, record_id),
            ).fetchone()
            return json.loads(row["payload_json"]) if row else None

    def put_artifact(self, task_id: str, name: str, payload: Mapping[str, object]) -> None:
        payload_json = _json(dict(payload))
        payload_hash = hashlib.sha256(payload_json.encode("utf-8")).hexdigest()
        now = _now()
        with self._connection() as connection:
            self._begin(connection)
            row = connection.execute(
                "SELECT payload_hash FROM task_artifacts WHERE task_id=? AND name=?",
                (task_id, name),
            ).fetchone()
            if row is not None and row["payload_hash"] != payload_hash:
                raise StateConflict(f"immutable task artifact {name!r} changed on replay")
            connection.execute(
                "INSERT OR IGNORE INTO task_artifacts(task_id,name,payload_json,payload_hash,created_at) VALUES(?,?,?,?,?)",
                (task_id, name, payload_json, payload_hash, now),
            )
            connection.commit()

    def ensure_budget_scope(
        self, scope_id: str, maximum_calls: int, *, imported_calls: int = 0
    ) -> int:
        if maximum_calls < 1 or not 0 <= imported_calls <= maximum_calls:
            raise ValueError("invalid shared budget scope limits")
        with self._connection() as connection:
            self._begin(connection)
            row = connection.execute(
                "SELECT maximum_calls,calls_used FROM budget_scopes WHERE scope_id=?", (scope_id,)
            ).fetchone()
            if row is None:
                connection.execute(
                    "INSERT INTO budget_scopes(scope_id,maximum_calls,calls_used) VALUES(?,?,?)",
                    (scope_id, maximum_calls, imported_calls),
                )
                used = imported_calls
            else:
                if int(row["maximum_calls"]) != maximum_calls:
                    raise StateConflict("shared budget scope maximum cannot change")
                used = max(int(row["calls_used"]), imported_calls)
                if used != int(row["calls_used"]):
                    connection.execute(
                        "UPDATE budget_scopes SET calls_used=? WHERE scope_id=?",
                        (used, scope_id),
                    )
            connection.commit()
            return used

    def reserve_provider_request(
        self,
        task_id: str,
        request_hash: str,
        *,
        scope_id: str,
        scope_maximum_calls: int,
        task_maximum_calls: int,
        imported_scope_calls: int = 0,
    ) -> dict[str, object] | None:
        """Reserve a provider call; return saved response for a completed duplicate."""
        now = _now()
        with self._connection() as connection:
            self._begin(connection)
            scope = connection.execute(
                "SELECT maximum_calls,calls_used FROM budget_scopes WHERE scope_id=?", (scope_id,)
            ).fetchone()
            if scope is None:
                if not 1 <= scope_maximum_calls or not 0 <= imported_scope_calls <= scope_maximum_calls:
                    raise ValueError("invalid shared budget scope limits")
                connection.execute(
                    "INSERT INTO budget_scopes(scope_id,maximum_calls,calls_used) VALUES(?,?,?)",
                    (scope_id, scope_maximum_calls, imported_scope_calls),
                )
                scope_used = imported_scope_calls
            else:
                if int(scope["maximum_calls"]) != scope_maximum_calls:
                    raise StateConflict("shared budget scope maximum cannot change")
                scope_used = max(int(scope["calls_used"]), imported_scope_calls)
                if scope_used != int(scope["calls_used"]):
                    connection.execute(
                        "UPDATE budget_scopes SET calls_used=? WHERE scope_id=?",
                        (scope_used, scope_id),
                    )
            task = connection.execute(
                "SELECT status,max_provider_calls,provider_calls_used FROM tasks WHERE task_id=?",
                (task_id,),
            ).fetchone()
            if task is None:
                raise KeyError(f"unknown task_id {task_id}")
            prior = connection.execute(
                "SELECT status,response_json FROM provider_requests WHERE task_id=? AND request_hash=?",
                (task_id, request_hash),
            ).fetchone()
            if prior is not None and prior["status"] == "completed":
                connection.commit()
                return {"kind": "cached_response", "response": json.loads(prior["response_json"])}
            unresolved = connection.execute(
                "SELECT request_hash FROM provider_requests WHERE task_id=? AND status IN ('started','unknown') LIMIT 1",
                (task_id,),
            ).fetchone()
            if unresolved is not None:
                raise UnknownOutboundOutcome(
                    "task has a request with an unknown delivery outcome; reconcile it before sending another request"
                )
            if task["status"] == "completed":
                raise StateConflict("completed task cannot start a provider request")
            if int(task["provider_calls_used"]) >= min(task_maximum_calls, int(task["max_provider_calls"])):
                raise StateBudgetExhausted("task provider-call budget exhausted")
            if scope_used >= scope_maximum_calls:
                raise StateBudgetExhausted("shared provider-call budget exhausted")
            if prior is None:
                attempt_number = 1
                connection.execute(
                    "INSERT INTO provider_requests(task_id,request_hash,status,attempt_count,updated_at) VALUES(?,?,?,?,?)",
                    (task_id, request_hash, "started", 1, now),
                )
            else:
                attempt_number = int(prior["attempt_count"]) + 1
                connection.execute(
                    "UPDATE provider_requests SET status='started',attempt_count=attempt_count+1,response_json=NULL,response_hash=NULL,usage_json=NULL,updated_at=? WHERE task_id=? AND request_hash=?",
                    (now, task_id, request_hash),
                )
            attempt_id = uuid.uuid4().hex
            connection.execute(
                "INSERT INTO provider_attempts(attempt_id,task_id,scope_id,request_hash,attempt_number,status,created_at,updated_at) VALUES(?,?,?,?,?,?,?,?)",
                (attempt_id, task_id, scope_id, request_hash, attempt_number, "started", now, now),
            )
            connection.execute(
                "UPDATE budget_scopes SET calls_used=calls_used+1 WHERE scope_id=?", (scope_id,)
            )
            connection.execute(
                "UPDATE tasks SET provider_calls_used=provider_calls_used+1,status='running',updated_at=? WHERE task_id=?",
                (now, task_id),
            )
            connection.commit()
            return {"kind": "reserved", "attempt_id": attempt_id}

    def get_cached_provider_response(
        self, task_id: str, request_hash: str
    ) -> dict[str, object] | None:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT status,response_json FROM provider_requests WHERE task_id=? AND request_hash=?",
                (task_id, request_hash),
            ).fetchone()
            if row is None or row["status"] != "completed":
                return None
            return json.loads(row["response_json"])

    def has_provider_request(self, task_id: str, request_hash: str) -> bool:
        with self._connection() as connection:
            return connection.execute(
                "SELECT 1 FROM provider_requests WHERE task_id=? AND request_hash=?",
                (task_id, request_hash),
            ).fetchone() is not None

    def task_logical_turn_count(self, task_id: str) -> int:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT COUNT(*) AS turn_count FROM provider_requests WHERE task_id=?",
                (task_id,),
            ).fetchone()
            return int(row["turn_count"])

    def claim_logical_turn(self, task_id: str, request_hash: str) -> bool:
        """Count one exact inference request once, including across process restarts."""
        with self._connection() as connection:
            self._begin(connection)
            row = connection.execute(
                "SELECT logical_counted FROM provider_requests WHERE task_id=? AND request_hash=?",
                (task_id, request_hash),
            ).fetchone()
            if row is None:
                raise KeyError("logical turn has no durable request record")
            if bool(row["logical_counted"]):
                connection.commit()
                return False
            connection.execute(
                "UPDATE provider_requests SET logical_counted=1 WHERE task_id=? AND request_hash=?",
                (task_id, request_hash),
            )
            connection.commit()
            return True

    def complete_provider_request(
        self,
        task_id: str,
        request_hash: str,
        response: Mapping[str, object],
        usage: Mapping[str, object] | None = None,
        *,
        attempt_id: str | None = None,
    ) -> None:
        response_json = _json(dict(response))
        response_hash = hashlib.sha256(response_json.encode("utf-8")).hexdigest()
        usage_json = _json(dict(usage or response.get("usage", {})))
        now = _now()
        with self._connection() as connection:
            self._begin(connection)
            prior = connection.execute(
                "SELECT status,response_hash FROM provider_requests WHERE task_id=? AND request_hash=?",
                (task_id, request_hash),
            ).fetchone()
            if prior is not None and prior["status"] == "completed" and prior["response_hash"] != response_hash:
                raise StateConflict("provider request response is immutable")
            if prior is None:
                connection.execute(
                    "INSERT INTO provider_requests(task_id,request_hash,status,attempt_count,response_json,response_hash,usage_json,updated_at) VALUES(?,?,?,?,?,?,?,?)",
                    (task_id, request_hash, "completed", 0, response_json, response_hash, usage_json, now),
                )
            else:
                connection.execute(
                    "UPDATE provider_requests SET status='completed',response_json=?,response_hash=?,usage_json=?,updated_at=? WHERE task_id=? AND request_hash=?",
                    (response_json, response_hash, usage_json, now, task_id, request_hash),
                )
            if attempt_id is None:
                connection.execute(
                    "UPDATE provider_attempts SET status='completed',updated_at=? WHERE task_id=? AND request_hash=? AND status='started'",
                    (now, task_id, request_hash),
                )
            else:
                connection.execute(
                    "UPDATE provider_attempts SET status='completed',updated_at=? WHERE attempt_id=? AND task_id=? AND request_hash=?",
                    (now, attempt_id, task_id, request_hash),
                )
            connection.commit()

    def fail_provider_request_before_connect(
        self, task_id: str, request_hash: str, *, scope_id: str, attempt_id: str | None = None
    ) -> None:
        """Release a reservation only when the caller proves no socket connected."""
        now = _now()
        with self._connection() as connection:
            self._begin(connection)
            row = connection.execute(
                "SELECT status FROM provider_requests WHERE task_id=? AND request_hash=?",
                (task_id, request_hash),
            ).fetchone()
            if row is None or row["status"] != "started":
                raise StateConflict("pre-connect failure requires one pending durable reservation")
            connection.execute(
                "UPDATE provider_requests SET status='failed_before_connect',updated_at=? WHERE task_id=? AND request_hash=?",
                (now, task_id, request_hash),
            )
            if attempt_id is None:
                connection.execute(
                    "UPDATE provider_attempts SET status='failed_before_connect',updated_at=? WHERE task_id=? AND request_hash=? AND status='started'",
                    (now, task_id, request_hash),
                )
            else:
                connection.execute(
                    "UPDATE provider_attempts SET status='failed_before_connect',updated_at=? WHERE attempt_id=? AND task_id=? AND request_hash=?",
                    (now, attempt_id, task_id, request_hash),
                )
            connection.execute(
                "UPDATE tasks SET provider_calls_used=MAX(provider_calls_used-1,0),updated_at=? WHERE task_id=?",
                (now, task_id),
            )
            connection.execute(
                "UPDATE budget_scopes SET calls_used=MAX(calls_used-1,0) WHERE scope_id=?", (scope_id,)
            )
            connection.commit()

    def fail_provider_request_quota_rejected(
        self,
        task_id: str,
        request_hash: str,
        *,
        http_status: int,
        provider_error_code: str,
        provider_error_sha256: str,
        attempt_id: str | None = None,
    ) -> None:
        """Resolve a reserved request after the provider returned an explicit quota rejection."""
        now = _now()
        with self._connection() as connection:
            self._begin(connection)
            row = connection.execute(
                "SELECT status FROM provider_requests WHERE task_id=? AND request_hash=?",
                (task_id, request_hash),
            ).fetchone()
            if row is None or row["status"] != "started":
                raise StateConflict("quota rejection requires one pending provider request")
            connection.execute(
                "UPDATE provider_requests SET status='quota_rejected',updated_at=? WHERE task_id=? AND request_hash=?",
                (now, task_id, request_hash),
            )
            if attempt_id is None:
                connection.execute(
                    "UPDATE provider_attempts SET status='quota_rejected',updated_at=? WHERE task_id=? AND request_hash=? AND status='started'",
                    (now, task_id, request_hash),
                )
            else:
                connection.execute(
                    "UPDATE provider_attempts SET status='quota_rejected',updated_at=? WHERE attempt_id=? AND task_id=? AND request_hash=?",
                    (now, attempt_id, task_id, request_hash),
                )
            connection.execute(
                "INSERT INTO task_events(task_id,event_key,stage,payload_json,created_at) VALUES(?,?,?,?,?)",
                (
                    task_id,
                    f"provider-quota-rejected:{request_hash}:{attempt_id or 'latest'}",
                    "provider_quota_rejected",
                    _json({
                        "request_sha256": request_hash,
                        "http_status": http_status,
                        "provider_error_code": provider_error_code,
                        "provider_error_sha256": provider_error_sha256,
                        "provider_request_sent": True,
                    }),
                    now,
                ),
            )
            connection.commit()

    def reconcile_request_cache(self, task_id: str, response_cache_dir: str | Path) -> list[str]:
        cache_dir = Path(response_cache_dir)
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT request_hash FROM provider_requests WHERE task_id=? AND status='started'",
                (task_id,),
            ).fetchall()
        unresolved = []
        for row in rows:
            request_hash = str(row["request_hash"])
            cache_file = cache_dir / f"{request_hash}.json"
            if cache_file.is_file():
                response = json.loads(cache_file.read_text(encoding="utf-8"))
                self.complete_provider_request(task_id, request_hash, response, response.get("usage", {}))
            else:
                unresolved.append(request_hash)
        return unresolved

    def record_feedback(
        self, task_id: str, feedback_id: str, payload: Mapping[str, object]
    ) -> None:
        self._insert_immutable_record("feedback", task_id, "feedback_id", feedback_id, payload)

    def record_execution_result(
        self, task_id: str, execution_id: str, payload: Mapping[str, object]
    ) -> None:
        self._insert_immutable_record("execution_results", task_id, "execution_id", execution_id, payload)

    def _insert_immutable_record(
        self, table: str, task_id: str, id_column: str, record_id: str,
        payload: Mapping[str, object],
    ) -> None:
        if table not in {"feedback", "execution_results"} or id_column not in {"feedback_id", "execution_id"}:
            raise ValueError("unsupported immutable state record")
        payload_json = _json(dict(payload))
        payload_hash = hashlib.sha256(payload_json.encode("utf-8")).hexdigest()
        now = _now()
        with self._connection() as connection:
            self._begin(connection)
            row = connection.execute(
                f"SELECT payload_hash FROM {table} WHERE task_id=? AND {id_column}=?",
                (task_id, record_id),
            ).fetchone()
            if row is not None and row["payload_hash"] != payload_hash:
                raise StateConflict(f"{table} record id was reused with different content")
            connection.execute(
                f"INSERT OR IGNORE INTO {table}(task_id,{id_column},payload_json,payload_hash,created_at) VALUES(?,?,?,?,?)",
                (task_id, record_id, payload_json, payload_hash, now),
            )
            connection.commit()
