from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

pytest.importorskip("temporalio")

from adaptive_mas.harness_state import HarnessStateStore
from adaptive_mas.industrial_rag import cli
from adaptive_mas.industrial_rag import temporal_workflow
from adaptive_mas.industrial_rag.temporal_workflow import (
    IndustrialRagIngestActivityInput,
    IndustrialRagIngestRequest,
    capture_ingest_request,
    ingest_document_activity,
    read_captured_input_bytes,
)


def test_temporal_ingest_captures_inputs_and_reuses_first_persisted_identity(tmp_path, monkeypatch) -> None:
    source_path = tmp_path / "source.txt"
    metadata_path = tmp_path / "metadata.json"
    original_source = (
        b"Synthetic annual report. Revenue was 100 million yuan for the year. "
        b"Operating income and assets are reported in a plain text table.\n"
    )
    metadata = {
        "document_id": "temporal-capture-document",
        "version_id": "temporal-capture-version",
        "source_id": "temporal-capture-source",
        "title": "Synthetic capture fixture",
        "document_type": "synthetic_report",
        "security_code": "000001",
        "entity_name": "Synthetic issuer",
        "report_period": "2026",
        "accounting_basis": "consolidated",
        "source_url": "https://example.invalid/temporal-capture",
        "rights_note": "Synthetic fixture only",
        "published_at": "2026-01-01T00:00:00+00:00",
        "first_available_at": "2026-01-01T00:00:00+00:00",
        "time_basis": "strict_forward",
        "read_principals": ["local-public"],
    }
    original_metadata = json.dumps(metadata, ensure_ascii=False, sort_keys=True).encode("utf-8")
    source_path.write_bytes(original_source)
    metadata_path.write_bytes(original_metadata)
    request = IndustrialRagIngestRequest(
        task_id="temporal-capture-task",
        idempotency_key="temporal-capture-idempotency",
        source_path=str(source_path),
        metadata_path=str(metadata_path),
        database_path=str(tmp_path / "rag.sqlite3"),
        object_root=str(tmp_path / "objects"),
        scratch_root=str(tmp_path / "scratch"),
        state_database_path=str(tmp_path / "task-state.sqlite3"),
    )

    captured = capture_ingest_request(request)
    assert captured.is_captured
    assert captured.source_sha256 == hashlib.sha256(original_source).hexdigest()
    assert captured.metadata_sha256 == hashlib.sha256(original_metadata).hexdigest()
    assert captured.captured_at is not None

    source_path.write_bytes(b"Changed after capture: 900 million yuan.\n")
    changed_metadata = {**metadata, "document_id": "changed-document", "version_id": "changed-version"}
    metadata_path.write_text(json.dumps(changed_metadata), encoding="utf-8")
    assert read_captured_input_bytes(captured) == (original_source, original_metadata)

    monkeypatch.setattr(temporal_workflow.activity, "heartbeat", lambda *args, **kwargs: None)
    activity_input = IndustrialRagIngestActivityInput(captured)
    first_result = ingest_document_activity(activity_input)
    assert first_result["content_sha256"] == captured.source_sha256
    assert first_result["document_id"] == metadata["document_id"]
    assert first_result["version_id"] == metadata["version_id"]

    state = HarnessStateStore(captured.state_database_path)
    assert state.get_artifact(captured.task_id, "industrial_rag_ingest_result") == first_result
    assert state.get_artifact(captured.task_id, "industrial_rag_ingest_identity") is None
    assert datetime.fromisoformat(str(first_result["ingested_at"])) >= datetime.fromisoformat(captured.captured_at)
    with sqlite3.connect(captured.database_path) as connection:
        persisted = connection.execute(
            "SELECT content_sha256,captured_at,ingested_at FROM rag_document_versions WHERE version_id=?",
            (metadata["version_id"],),
        ).fetchone()
    assert persisted == (captured.source_sha256, captured.captured_at, first_result["ingested_at"])

    assert ingest_document_activity(activity_input) == first_result
    captured_source = Path(captured.source_path)
    captured_metadata = Path(captured.metadata_path)
    try:
        captured_source.write_bytes(original_source + b"tampered")
        with pytest.raises(ValueError, match="captured source bytes"):
            read_captured_input_bytes(captured)
    finally:
        captured_source.write_bytes(original_source)
    try:
        captured_metadata.write_bytes(original_metadata + b" ")
        with pytest.raises(ValueError, match="captured metadata bytes"):
            read_captured_input_bytes(captured)
    finally:
        captured_metadata.write_bytes(original_metadata)


def test_temporal_failed_transaction_does_not_freeze_ingested_at(tmp_path, monkeypatch) -> None:
    source_path = tmp_path / "transaction-source.txt"
    metadata_path = tmp_path / "transaction-metadata.json"
    source_path.write_text(
        "Synthetic annual report. Revenue was 100 million yuan. This line supports parsing.\n",
        encoding="utf-8",
    )
    metadata = {
        "document_id": "temporal-transaction-document",
        "version_id": "temporal-transaction-version",
        "source_id": "temporal-transaction-source",
        "title": "Synthetic transaction fixture",
        "document_type": "synthetic_report",
        "security_code": "000001",
        "entity_name": "Synthetic issuer",
        "report_period": "2026",
        "accounting_basis": "consolidated",
        "source_url": "https://example.invalid/temporal-transaction",
        "rights_note": "Synthetic fixture only",
        "published_at": "2026-01-01T00:00:00+00:00",
        "first_available_at": "2026-01-01T00:00:00+00:00",
        "time_basis": "strict_forward",
        "read_principals": ["local-public"],
    }
    metadata_path.write_text(json.dumps(metadata, sort_keys=True), encoding="utf-8")
    request = IndustrialRagIngestRequest(
        task_id="temporal-transaction-task",
        idempotency_key="temporal-transaction-idempotency",
        source_path=str(source_path),
        metadata_path=str(metadata_path),
        database_path=str(tmp_path / "transaction-rag.sqlite3"),
        object_root=str(tmp_path / "transaction-objects"),
        scratch_root=str(tmp_path / "transaction-scratch"),
        state_database_path=str(tmp_path / "transaction-state.sqlite3"),
    )
    captured = capture_ingest_request(request)
    captured_at = datetime.fromisoformat(captured.captured_at)
    failed_attempt_time = captured_at + timedelta(minutes=1)
    successful_attempt_time = captured_at + timedelta(hours=2)

    class ControlledClock:
        times = iter((failed_attempt_time, successful_attempt_time))

        @classmethod
        def now(cls, _timezone=None):
            return next(cls.times)

    monkeypatch.setattr(temporal_workflow, "datetime", ControlledClock)
    monkeypatch.setattr(temporal_workflow.activity, "heartbeat", lambda *args, **kwargs: None)
    real_store = cli._store(captured.database_path, captured.object_root)

    class FaultInjectingStore:
        attempts = 0

        def ingest_bundle(self, bundle, *, ingested_at_factory=None):
            self.attempts += 1
            if self.attempts == 1:
                def fail_after_timestamp(*_args, **_kwargs):
                    raise RuntimeError("injected rollback after ingest timestamp selection")

                with monkeypatch.context() as patcher:
                    patcher.setattr(real_store, "_insert_children", fail_after_timestamp)
                    return real_store.ingest_bundle(bundle, ingested_at_factory=ingested_at_factory)
            if self.attempts == 2:
                real_store.ingest_bundle(bundle, ingested_at_factory=ingested_at_factory)
                raise RuntimeError("injected activity failure after committed RAG ingest")
            return real_store.ingest_bundle(bundle, ingested_at_factory=ingested_at_factory)

    fault_store = FaultInjectingStore()
    monkeypatch.setattr(cli, "_store", lambda *_args: fault_store)
    activity_input = IndustrialRagIngestActivityInput(captured)

    with pytest.raises(RuntimeError, match="rollback after ingest timestamp selection"):
        ingest_document_activity(activity_input)
    with sqlite3.connect(captured.database_path) as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM rag_document_versions WHERE version_id=?",
            (metadata["version_id"],),
        ).fetchone()[0] == 0
    state = HarnessStateStore(captured.state_database_path)
    assert state.get_artifact(captured.task_id, "industrial_rag_ingest_identity") is None
    assert state.get_artifact(captured.task_id, "industrial_rag_ingest_result") is None

    with pytest.raises(RuntimeError, match="after committed RAG ingest"):
        ingest_document_activity(activity_input)
    with sqlite3.connect(captured.database_path) as connection:
        persisted_after_commit = connection.execute(
            "SELECT ingested_at FROM rag_document_versions WHERE version_id=?",
            (metadata["version_id"],),
        ).fetchone()
    assert persisted_after_commit == (successful_attempt_time.isoformat(),)
    assert state.get_artifact(captured.task_id, "industrial_rag_ingest_result") is None

    recovered = ingest_document_activity(activity_input)
    assert recovered["status"] == "reused"
    assert recovered["ingested_at"] == successful_attempt_time.isoformat()
    assert fault_store.attempts == 3
    assert state.get_artifact(captured.task_id, "industrial_rag_ingest_result") == recovered
