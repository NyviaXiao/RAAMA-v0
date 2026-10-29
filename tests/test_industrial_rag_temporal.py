from __future__ import annotations

import asyncio
import hashlib
import json
import sqlite3
from dataclasses import asdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, get_type_hints

import pytest

pytest.importorskip("temporalio")
from temporalio.converter import DataConverter

from adaptive_mas.harness_state import HarnessStateStore
from adaptive_mas.agents.harness_prompts import seed_candidate
import adaptive_mas.agents.harness as harness_module
from adaptive_mas.industrial_rag import cli
from adaptive_mas.industrial_rag import temporal_workflow
from adaptive_mas.industrial_rag.temporal_workflow import (
    IndustrialRagEvaluationActivityInput,
    IndustrialRagEvaluationRequest,
    IndustrialRagIndexActivityInput,
    IndustrialRagIndexRequest,
    IndustrialRagOperationRequest,
    IndustrialRagOperationWorkflow,
    IndustrialRagResearchActivityInput,
    IndustrialRagResearchRequest,
    IndustrialRagIngestActivityInput,
    IndustrialRagIngestRequest,
    TemporalTaskInitialization,
    TemporalWorkflowControl,
    build_index_activity,
    capture_ingest_request,
    capture_evaluation_request,
    capture_research_request,
    dead_letter_temporal_task,
    evaluate_manifest_activity,
    initialize_temporal_task_activity,
    ingest_document_activity,
    read_captured_input_bytes,
    read_captured_evaluation_bytes,
    read_captured_research_bytes,
    run_research_activity,
    verify_research_inputs_activity,
    verify_evaluation_inputs_activity,
    verify_index_inputs_activity,
)


def test_temporal_sdk_converter_roundtrips_json_payloads_and_rejects_object_hints(tmp_path) -> None:
    converter = DataConverter.default
    initialization = TemporalTaskInitialization(
        "converter-task", "converter-key", "state.sqlite3",
        {
            "operation": "ingest",
            "captured_at": "2026-10-27T00:00:00+00:00",
            "nested": {"attempt": 1, "captured": True},
        },
    )
    operation = IndustrialRagOperationRequest("index", {"nested": {"attempt": 1}})
    state_path = tmp_path / "task-state.sqlite3"
    request = TemporalTaskInitialization(
        "converter-task", "converter-key", str(state_path), {"nested": {"captured": True}},
    )
    running_result = initialize_temporal_task_activity(request)
    HarnessStateStore(state_path).complete_task(
        request.task_id,
        {"status": "ingested", "version_id": "synthetic-version", "nested": {"saved": True}},
    )
    completed_result = initialize_temporal_task_activity(request)

    async def roundtrip() -> None:
        initialization_payloads = await converter.encode([initialization])
        assert await converter.decode(
            initialization_payloads,
            type_hints=[TemporalTaskInitialization],
        ) == [initialization]

        operation_payloads = await converter.encode([operation])
        assert await converter.decode(
            operation_payloads,
            type_hints=[IndustrialRagOperationRequest],
        ) == [operation]

        for activity_result in (running_result, completed_result):
            result_payloads = await converter.encode([activity_result])
            assert await converter.decode(
                result_payloads,
                type_hints=[dict[str, Any]],
            ) == [activity_result]
            with pytest.raises(TypeError, match="Failed converting value for key 'result'"):
                await converter.decode(result_payloads, type_hints=[dict[str, object]])

    asyncio.run(roundtrip())
    assert get_type_hints(TemporalTaskInitialization)["request"] == dict[str, Any]
    assert get_type_hints(initialize_temporal_task_activity)["return"] == dict[str, Any]
    assert get_type_hints(IndustrialRagOperationRequest)["payload"] == dict[str, Any]
    activity_results = (
        temporal_workflow.ingest_document_activity,
        temporal_workflow.initialize_temporal_task_activity,
        temporal_workflow.verify_index_inputs_activity,
        temporal_workflow.build_index_activity,
        temporal_workflow.verify_evaluation_inputs_activity,
        temporal_workflow.evaluate_manifest_activity,
        temporal_workflow.verify_research_inputs_activity,
        temporal_workflow.run_research_activity,
        temporal_workflow.apply_temporal_control_activity,
        temporal_workflow.fail_temporal_task_activity,
    )
    assert all(get_type_hints(activity_fn)["return"] == dict[str, Any] for activity_fn in activity_results)


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


def test_temporal_ingest_cancel_is_durable_and_prevents_activity_side_effect(tmp_path, monkeypatch) -> None:
    source_path = tmp_path / "cancel-source.txt"
    metadata_path = tmp_path / "cancel-metadata.json"
    source_path.write_text("Synthetic report source for a cancelled workflow.\n", encoding="utf-8")
    metadata = {
        "document_id": "temporal-cancel-document",
        "version_id": "temporal-cancel-version",
        "source_id": "temporal-cancel-source",
        "title": "Synthetic cancel fixture",
        "document_type": "synthetic_report",
        "security_code": "000001",
        "entity_name": "Synthetic issuer",
        "report_period": "2026",
        "accounting_basis": "consolidated",
        "source_url": "https://example.invalid/temporal-cancel",
        "rights_note": "Synthetic fixture only",
        "published_at": "2026-01-01T00:00:00+00:00",
        "first_available_at": "2026-01-01T00:00:00+00:00",
        "time_basis": "strict_forward",
        "read_principals": ["local-public"],
    }
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    database_path = tmp_path / "cancel-rag.sqlite3"
    state_database_path = tmp_path / "cancel-state.sqlite3"
    request = capture_ingest_request(IndustrialRagIngestRequest(
        task_id="temporal-cancel-task",
        idempotency_key="temporal-cancel-key",
        source_path=str(source_path),
        metadata_path=str(metadata_path),
        database_path=str(database_path),
        object_root=str(tmp_path / "cancel-objects"),
        scratch_root=str(tmp_path / "cancel-scratch"),
        state_database_path=str(state_database_path),
    ))
    instance = temporal_workflow.IndustrialRagIngestWorkflow()
    instance.control(TemporalWorkflowControl("cancel-ingest", "cancel", "operator-i", "Stop before parsing."))
    monkeypatch.setattr(temporal_workflow.activity, "heartbeat", lambda *args, **kwargs: None)
    calls = []

    async def fake_execute_activity(fn, value, **_kwargs):
        calls.append(fn.__name__)
        return fn(value)

    monkeypatch.setattr(temporal_workflow.workflow, "execute_activity", fake_execute_activity)
    result = asyncio.run(instance.run(request))
    assert result == {"status": "cancelled", "task_id": request.task_id}
    assert calls == ["initialize_temporal_task_activity", "apply_temporal_control_activity"]
    assert not database_path.exists()
    state = HarnessStateStore(state_database_path)
    assert state.get_task(request.task_id)["status"] == "cancelled"
    assert state.list_task_events(request.task_id)[-1]["stage"] == "cancelled"


def test_temporal_ingest_dead_letter_with_result_artifact_requires_reconciliation(tmp_path, monkeypatch) -> None:
    source_path = tmp_path / "dead-letter-ingest-source.txt"
    metadata_path = tmp_path / "dead-letter-ingest-metadata.json"
    source_path.write_text("Synthetic source for dead-letter recovery.", encoding="utf-8")
    metadata_path.write_text(json.dumps({
        "document_id": "dead-letter-ingest-document",
        "version_id": "dead-letter-ingest-version",
        "source_id": "dead-letter-ingest-source",
        "title": "Synthetic dead-letter fixture",
        "document_type": "synthetic_report",
        "security_code": "000001",
        "entity_name": "Synthetic issuer",
        "report_period": "2026",
        "accounting_basis": "consolidated",
        "source_url": "https://example.invalid/dead-letter-ingest",
        "rights_note": "Synthetic fixture only",
        "published_at": "2026-01-01T00:00:00+00:00",
        "first_available_at": "2026-01-01T00:00:00+00:00",
        "time_basis": "strict_forward",
        "read_principals": ["local-public"],
    }, sort_keys=True), encoding="utf-8")
    request = capture_ingest_request(IndustrialRagIngestRequest(
        task_id="dead-letter-ingest-task",
        idempotency_key="dead-letter-ingest-key",
        source_path=str(source_path),
        metadata_path=str(metadata_path),
        database_path=str(tmp_path / "dead-letter-ingest.sqlite3"),
        object_root=str(tmp_path / "dead-letter-ingest-objects"),
        scratch_root=str(tmp_path / "dead-letter-ingest-scratch"),
        state_database_path=str(tmp_path / "dead-letter-ingest-state.sqlite3"),
    ))
    state = HarnessStateStore(request.state_database_path)
    state.open_task(
        request.task_id,
        request.idempotency_key,
        temporal_workflow._ingest_task_request(request),
        max_provider_calls=0,
    )
    artifact = {"status": "ingested", "version_id": "dead-letter-ingest-version"}
    state.put_artifact(request.task_id, "industrial_rag_ingest_result", artifact)
    assert state.transition_task_status(
        request.task_id, expected_statuses={"running"}, status="failed", stage="failed",
        event_key="ingest-failed-before-dead-letter", payload={"error": "synthetic failure"},
    )
    assert state.dead_letter_task(
        request.task_id, expected_status="failed", operator_id="operator-ingest",
        reason="Preserve the artifact for explicit reconciliation.", event_key="ingest-dead-letter",
    )

    initialized = initialize_temporal_task_activity(TemporalTaskInitialization(
        request.task_id,
        request.idempotency_key,
        request.state_database_path,
        temporal_workflow._ingest_task_request(request),
        result_artifact="industrial_rag_ingest_result",
    ))
    assert initialized == {
        "status": "dead_letter",
        "result": None,
        "reconciliation_required": True,
        "result_artifact_name": "industrial_rag_ingest_result",
    }
    with pytest.raises(RuntimeError, match="committed ingest artifact requires explicit reconciliation"):
        ingest_document_activity(IndustrialRagIngestActivityInput(request))
    assert state.get_task(request.task_id)["status"] == "dead_letter"
    assert state.get_task(request.task_id)["stage"] == "dead_letter"
    assert state.get_artifact(request.task_id, "industrial_rag_ingest_result") == artifact
    assert not Path(request.database_path).exists()


def _evaluation_inputs(tmp_path: Path) -> tuple[Path, Path, dict[str, object], dict[str, object]]:
    manifest_path = tmp_path / "manifest.json"
    predictions_path = tmp_path / "predictions.json"
    manifest: dict[str, object] = {
        "benchmark_id": "synthetic-temporal-evaluation",
        "cases": [{
            "case_id": "synthetic-unanswerable",
            "data_partition": "development",
            "view_status": "unseen",
            "answerability": "unanswerable",
            "required_evidence": [],
        }],
    }
    predictions: dict[str, object] = {
        "run_id": "synthetic-temporal-run",
        "cases": [{
            "case_id": "synthetic-unanswerable",
            "predicted_answerability": "unanswerable",
            "retrieved": [],
        }],
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    predictions_path.write_text(json.dumps(predictions), encoding="utf-8")
    return manifest_path, predictions_path, manifest, predictions


def test_temporal_evaluation_captures_bytes_and_persists_offline_result(tmp_path, monkeypatch) -> None:
    manifest_path, predictions_path, manifest, predictions = _evaluation_inputs(tmp_path)
    raw_request = IndustrialRagEvaluationRequest(
        task_id="temporal-evaluation-task",
        idempotency_key="temporal-evaluation-key",
        manifest_path=str(manifest_path),
        predictions_path=str(predictions_path),
        scratch_root=str(tmp_path / "scratch"),
        state_database_path=str(tmp_path / "state.sqlite3"),
    )
    captured = capture_evaluation_request(raw_request)
    repeated_capture = capture_evaluation_request(raw_request)
    assert repeated_capture == captured
    assert captured.manifest_sha256 == hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    assert captured.predictions_sha256 == hashlib.sha256(predictions_path.read_bytes()).hexdigest()

    manifest_path.write_text('{"cases":[]}', encoding="utf-8")
    predictions_path.write_text('{"cases":[]}', encoding="utf-8")
    manifest_bytes, prediction_bytes = read_captured_evaluation_bytes(captured)
    assert json.loads(manifest_bytes) == manifest
    assert json.loads(prediction_bytes) == predictions

    activity_input = IndustrialRagEvaluationActivityInput(captured)
    task_request = temporal_workflow._operation_identity("industrial_rag.evaluate.v1", captured)
    monkeypatch.setattr(temporal_workflow.activity, "heartbeat", lambda *args, **kwargs: None)
    initialized = initialize_temporal_task_activity(TemporalTaskInitialization(
        captured.task_id, captured.idempotency_key, captured.state_database_path, task_request,
    ))
    assert initialized == {"status": "running", "result": None}
    verified = verify_evaluation_inputs_activity(activity_input)
    assert verified == {
        "manifest_sha256": captured.manifest_sha256,
        "predictions_sha256": captured.predictions_sha256,
        "case_count": 1,
    }
    first = evaluate_manifest_activity(activity_input)
    retry = evaluate_manifest_activity(activity_input)
    assert first == retry
    assert first["benchmark_sha256"] == captured.manifest_sha256
    assert first["prediction_sha256"] == captured.predictions_sha256
    assert first["sample_count"] == 1
    state = HarnessStateStore(captured.state_database_path)
    assert state.get_task(captured.task_id)["status"] == "completed"
    assert state.get_artifact(captured.task_id, "industrial_rag_operation_result") == first

    captured_predictions = Path(captured.predictions_path)
    good_bytes = captured_predictions.read_bytes()
    captured_predictions.write_bytes(good_bytes + b" ")
    try:
        with pytest.raises(ValueError, match="captured evaluation predictions"):
            verify_evaluation_inputs_activity(activity_input)
    finally:
        captured_predictions.write_bytes(good_bytes)


def _research_bundle_fixture() -> dict[str, object]:
    trading_day = "2026-01-02"
    return {
        "schema_version": "adaptive-mas.temporal-research-input.v1",
        "as_of": trading_day,
        "research_data": {
            "trading_dates": [trading_day],
            "adjusted_bars": [],
            "execution_bars": [],
            "index_bars": [],
            "membership_snapshots": [{
                "event_time": trading_day,
                "available_at": trading_day,
                "members": ["000001"],
                "source_ref": "synthetic-membership-v1",
            }],
        },
        "public_snapshot": None,
        "candidate": seed_candidate(),
        "config": {"prediction_horizon_sessions": 1},
        "candidate_version_id": "synthetic-seed-v1",
        "required_observations": [],
        "procedural_memory": None,
        "model_risk_context": None,
        "endpoint": "https://synthetic-cache.invalid/v1/chat/completions",
        "model": "synthetic-cache-model",
        "package_identity_sha256": "a" * 64,
    }


def test_temporal_research_recovers_saved_manager_result_after_task_ack_loss_offline(tmp_path, monkeypatch) -> None:
    bundle_path = tmp_path / "research-bundle.json"
    original_bundle = _research_bundle_fixture()
    original_bytes = json.dumps(original_bundle, sort_keys=True).encode("utf-8")
    bundle_path.write_bytes(original_bytes)
    request = IndustrialRagResearchRequest(
        task_id="temporal-research-task",
        idempotency_key="temporal-research-key",
        bundle_path=str(bundle_path),
        scratch_root=str(tmp_path / "scratch"),
        state_database_path=str(tmp_path / "research-state.sqlite3"),
        response_cache_dir=str(tmp_path / "response-cache"),
        call_ledger_path=str(tmp_path / "calls.jsonl"),
        request_archive_dir=str(tmp_path / "request-archive"),
    )
    captured = capture_research_request(request)
    assert capture_research_request(request) == captured
    assert read_captured_research_bytes(captured) == original_bytes

    bundle_path.write_text('{"schema_version":"changed"}', encoding="utf-8")
    activity_input = IndustrialRagResearchActivityInput(captured)
    task_request = temporal_workflow._operation_identity("financial_research.cached_replay.v1", captured)
    state = HarnessStateStore(captured.state_database_path)
    state.open_task(captured.task_id, captured.idempotency_key, task_request, max_provider_calls=0)
    monkeypatch.setattr(temporal_workflow.activity, "heartbeat", lambda *args, **kwargs: None)
    verified = verify_research_inputs_activity(activity_input)
    assert verified["bundle_sha256"] == captured.bundle_sha256
    assert verified["offline_only"] is True

    runner_calls = []

    def fake_research_runner(research, as_of, public_snapshot, candidate, client, budget, **kwargs):
        runner_calls.append((research.trading_dates, as_of, client.offline_only, budget.provider_call_count))
        assert public_snapshot is None
        assert candidate["manager_prompt"]
        assert kwargs["candidate_version_id"] == "synthetic-seed-v1"
        assert budget.task_id == captured.task_id
        assert budget.budget_scope_id == f"temporal-offline:{captured.task_id}"
        assert isinstance(budget.package_identity_sha256, str) and len(budget.package_identity_sha256) == 64
        assert budget.persistent_tool_names == frozenset({"plan_research", "query_specialist", "finalize_research"})
        return {"status": "completed", "tool_trace": [{"role": "manager", "status": "completed"}]}

    monkeypatch.setattr(temporal_workflow, "run_research_harness", fake_research_runner)
    real_complete_task = HarnessStateStore.complete_task

    def lose_task_ack_after_manager_artifact(self, task_id, result, **kwargs):
        if task_id == captured.task_id and self.get_artifact(task_id, "industrial_rag_operation_result") is not None:
            raise RuntimeError("injected worker loss after manager artifact persistence")
        return real_complete_task(self, task_id, result, **kwargs)

    monkeypatch.setattr(HarnessStateStore, "complete_task", lose_task_ack_after_manager_artifact)
    with pytest.raises(RuntimeError, match="after manager artifact persistence"):
        run_research_activity(activity_input)
    saved = state.get_artifact(captured.task_id, "industrial_rag_operation_result")
    assert saved is not None
    assert state.get_task(captured.task_id)["status"] == "running"

    monkeypatch.setattr(HarnessStateStore, "complete_task", real_complete_task)
    retry = run_research_activity(activity_input)
    assert retry == saved
    assert retry["provider_calls"] == 0
    assert retry["research_result"]["status"] == "completed"
    assert runner_calls == [((date(2026, 1, 2),), date(2026, 1, 2), True, 0)]
    assert state.get_task(captured.task_id)["status"] == "completed"
    assert state.get_task(captured.task_id)["result"] == retry
    assert [event["stage"] for event in state.list_task_events(captured.task_id)] == [
        "created", "research_inputs_verified", "research_started", "research_persisted", "completed",
    ]

    capture_path = Path(captured.bundle_path)
    good_bytes = capture_path.read_bytes()
    capture_path.write_bytes(good_bytes + b" ")
    try:
        with pytest.raises(ValueError, match="captured research bundle"):
            verify_research_inputs_activity(activity_input)
    finally:
        capture_path.write_bytes(good_bytes)


def test_temporal_research_operation_workflow_applies_checkpoint_controls(tmp_path, monkeypatch) -> None:
    bundle_path = tmp_path / "research-bundle.json"
    bundle_path.write_text(json.dumps(_research_bundle_fixture(), sort_keys=True), encoding="utf-8")
    request = capture_research_request(IndustrialRagResearchRequest(
        task_id="temporal-research-workflow-task",
        idempotency_key="temporal-research-workflow-key",
        bundle_path=str(bundle_path),
        scratch_root=str(tmp_path / "scratch"),
        state_database_path=str(tmp_path / "state.sqlite3"),
        response_cache_dir=str(tmp_path / "response-cache"),
        call_ledger_path=str(tmp_path / "calls.jsonl"),
        request_archive_dir=str(tmp_path / "request-archive"),
    ))
    operation = IndustrialRagOperationRequest("research", asdict(request))
    instance = IndustrialRagOperationWorkflow()
    instance.control(TemporalWorkflowControl("pause-research", "pause", "operator-r", "Review research bundle."))
    monkeypatch.setattr(temporal_workflow.activity, "heartbeat", lambda *args, **kwargs: None)

    def fake_research_runner(*_args, **_kwargs):
        return {"status": "completed", "tool_trace": [{"role": "manager", "status": "completed"}]}

    monkeypatch.setattr(temporal_workflow, "run_research_harness", fake_research_runner)

    async def fake_execute_activity(fn, value, **_kwargs):
        return fn(value)

    async def resume_at_wait(predicate, **_kwargs):
        instance.control(TemporalWorkflowControl(
            "resume-research", "resume", "operator-r", "Review completed.",
        ))
        assert predicate()

    monkeypatch.setattr(temporal_workflow.workflow, "execute_activity", fake_execute_activity)
    monkeypatch.setattr(temporal_workflow.workflow, "wait_condition", resume_at_wait)
    result = asyncio.run(instance.run(operation))
    assert result["status"] == "completed"
    assert result["provider_calls"] == 0
    assert [item["action"] for item in instance.state()["control_history"]] == ["pause", "resume"]
    state = HarnessStateStore(request.state_database_path)
    assert state.get_task(request.task_id)["status"] == "completed"
    assert [item["stage"] for item in state.list_task_events(request.task_id)] == [
        "created", "paused", "resumed", "research_inputs_verified", "research_started",
        "research_persisted", "completed",
    ]


def test_temporal_research_cache_miss_fails_before_provider_network(tmp_path, monkeypatch) -> None:
    bundle = _research_bundle_fixture()
    bundle_path = tmp_path / "research-bundle.json"
    bundle_path.write_text(json.dumps(bundle, sort_keys=True), encoding="utf-8")
    request = capture_research_request(IndustrialRagResearchRequest(
        task_id="temporal-research-miss-task",
        idempotency_key="temporal-research-miss-key",
        bundle_path=str(bundle_path),
        scratch_root=str(tmp_path / "scratch"),
        state_database_path=str(tmp_path / "state.sqlite3"),
        response_cache_dir=str(tmp_path / "empty-cache"),
        call_ledger_path=str(tmp_path / "calls.jsonl"),
        request_archive_dir=str(tmp_path / "request-archive"),
    ))
    network_calls = []
    monkeypatch.setattr(temporal_workflow.activity, "heartbeat", lambda *args, **kwargs: None)
    monkeypatch.setattr(harness_module, "urlopen", lambda *args, **kwargs: network_calls.append(True))
    with pytest.raises(RuntimeError, match="offline-only Qwen client"):
        run_research_activity(IndustrialRagResearchActivityInput(request))
    assert network_calls == []
    state = HarnessStateStore(request.state_database_path)
    task = state.get_task(request.task_id)
    assert task["provider_calls_used"] == 0
    assert task["status"] == "running"


def _synthetic_index_request(tmp_path: Path) -> IndustrialRagIndexRequest:
    source_path = tmp_path / "index-source.txt"
    metadata_path = tmp_path / "index-metadata.json"
    source_path.write_text(
        "Synthetic annual report. Revenue was 100 million yuan for the year. "
        "Operating income and assets are reported in a plain text table.\n",
        encoding="utf-8",
    )
    metadata = {
        "document_id": "temporal-index-document",
        "version_id": "temporal-index-version",
        "source_id": "temporal-index-source",
        "title": "Synthetic temporal index fixture",
        "document_type": "synthetic_report",
        "security_code": "000001",
        "entity_name": "Synthetic issuer",
        "report_period": "2026",
        "accounting_basis": "consolidated",
        "source_url": "https://example.invalid/temporal-index",
        "rights_note": "Synthetic fixture only",
        "published_at": "2026-01-01T00:00:00+00:00",
        "first_available_at": "2026-01-01T00:00:00+00:00",
        "time_basis": "strict_forward",
        "read_principals": ["local-public"],
    }
    metadata_path.write_text(json.dumps(metadata, sort_keys=True), encoding="utf-8")
    database_path = tmp_path / "index.sqlite3"
    object_root = tmp_path / "objects"
    cli.ingest_file(
        source_path=source_path,
        metadata_path=metadata_path,
        database_path=database_path,
        object_root=object_root,
        scratch_root=tmp_path / "parser-scratch",
    )
    return IndustrialRagIndexRequest(
        task_id="temporal-index-task",
        idempotency_key="temporal-index-key",
        release_id="temporal-index-release",
        version_ids=("temporal-index-version",),
        database_path=str(database_path),
        object_root=str(object_root),
        state_database_path=str(tmp_path / "index-state.sqlite3"),
        actor_id="synthetic-operator",
        reason="Exercise local Temporal index operation.",
    )


def test_temporal_index_workflow_pauses_resumes_and_publishes_release(tmp_path, monkeypatch) -> None:
    request = _synthetic_index_request(tmp_path)
    operation = IndustrialRagOperationRequest("index", asdict(request))
    instance = IndustrialRagOperationWorkflow()
    instance.control(TemporalWorkflowControl("pause-1", "pause", "operator-a", "Review the index input."))
    monkeypatch.setattr(temporal_workflow.activity, "heartbeat", lambda *args, **kwargs: None)

    async def fake_execute_activity(fn, value, **_kwargs):
        return fn(value)

    async def resume_at_wait(predicate, **_kwargs):
        instance.control(TemporalWorkflowControl("resume-1", "resume", "operator-a", "Review completed."))
        assert predicate()

    monkeypatch.setattr(temporal_workflow.workflow, "execute_activity", fake_execute_activity)
    monkeypatch.setattr(temporal_workflow.workflow, "wait_condition", resume_at_wait)
    result = asyncio.run(instance.run(operation))
    assert result["status"] == "published"
    state = instance.state()
    assert state["status"] == "completed"
    assert [item["action"] for item in state["control_history"]] == ["pause", "resume"]
    assert all(item["accepted"] for item in state["control_history"])
    task_state = HarnessStateStore(request.state_database_path)
    assert task_state.get_task(request.task_id)["status"] == "completed"
    assert [event["stage"] for event in task_state.list_task_events(request.task_id)] == [
        "created", "paused", "resumed", "index_inputs_verified", "index_started",
        "index_persisted", "completed",
    ]


def test_temporal_resume_during_paused_task_restore_is_not_overwritten(tmp_path, monkeypatch) -> None:
    request = _synthetic_index_request(tmp_path)
    operation = IndustrialRagOperationRequest("index", asdict(request))
    task_state = HarnessStateStore(request.state_database_path)
    task_state.open_task(
        request.task_id,
        request.idempotency_key,
        {
            "operation": "industrial_rag.index.v1",
            "request": asdict(request),
        },
        max_provider_calls=0,
    )
    assert task_state.transition_task_status(
        request.task_id,
        expected_statuses={"running"},
        status="paused",
        stage="paused",
        event_key="seed-restored-pause",
    )

    instance = IndustrialRagOperationWorkflow()
    instance.control(TemporalWorkflowControl(
        "resume-before-restore", "resume", "operator-a", "Continue the paused index task.",
    ))
    monkeypatch.setattr(temporal_workflow.activity, "heartbeat", lambda *args, **kwargs: None)

    async def fake_execute_activity(fn, value, **_kwargs):
        return fn(value)

    async def unexpected_wait(*_args, **_kwargs):
        raise AssertionError("a pending resume command must prevent the restored pause wait")

    monkeypatch.setattr(temporal_workflow.workflow, "execute_activity", fake_execute_activity)
    monkeypatch.setattr(temporal_workflow.workflow, "wait_condition", unexpected_wait)
    result = asyncio.run(instance.run(operation))

    assert result["status"] == "published"
    assert task_state.get_task(request.task_id)["status"] == "completed"
    assert [event["stage"] for event in task_state.list_task_events(request.task_id)] == [
        "created", "paused", "resumed", "index_inputs_verified", "index_started",
        "index_persisted", "completed",
    ]


def test_temporal_operation_cancel_before_index_has_no_release_side_effect(tmp_path, monkeypatch) -> None:
    request = _synthetic_index_request(tmp_path)
    operation = IndustrialRagOperationRequest("index", asdict(request))
    instance = IndustrialRagOperationWorkflow()
    instance.control(TemporalWorkflowControl("cancel-1", "cancel", "operator-b", "Stop before indexing."))
    monkeypatch.setattr(temporal_workflow.activity, "heartbeat", lambda *args, **kwargs: None)
    calls = []

    async def fake_execute_activity(fn, value, **_kwargs):
        calls.append(fn.__name__)
        return fn(value)

    monkeypatch.setattr(temporal_workflow.workflow, "execute_activity", fake_execute_activity)
    result = asyncio.run(instance.run(operation))
    assert result == {"status": "cancelled", "task_id": request.task_id}
    assert calls == ["initialize_temporal_task_activity", "apply_temporal_control_activity"]
    state = HarnessStateStore(request.state_database_path)
    assert state.get_task(request.task_id)["status"] == "cancelled"
    assert state.list_task_events(request.task_id)[-1]["stage"] == "cancelled"
    assert cli._store(request.database_path, request.object_root).release_summaries()["releases"] == []


def test_temporal_index_failure_is_durable_and_can_be_dead_lettered(tmp_path, monkeypatch) -> None:
    request = IndustrialRagIndexRequest(
        task_id="temporal-index-failure-task",
        idempotency_key="temporal-index-failure-key",
        release_id="temporal-index-failure-release",
        version_ids=("missing-version",),
        database_path=str(tmp_path / "missing.sqlite3"),
        object_root=str(tmp_path / "missing-objects"),
        state_database_path=str(tmp_path / "failure-state.sqlite3"),
        actor_id="synthetic-operator",
        reason="Exercise operation failure and dead letter.",
    )
    operation = IndustrialRagOperationRequest("index", asdict(request))
    instance = IndustrialRagOperationWorkflow()
    monkeypatch.setattr(temporal_workflow.activity, "heartbeat", lambda *args, **kwargs: None)

    async def fail_after_activity_retry(fn, value, **_kwargs):
        if fn is verify_index_inputs_activity:
            try:
                fn(value)
            except Exception as error:
                raise temporal_workflow.ActivityError(
                    str(error), scheduled_event_id=1, started_event_id=2,
                    identity="synthetic-worker", activity_type=fn.__name__,
                    activity_id="verify-index", retry_state=None,
                ) from error
        return fn(value)

    monkeypatch.setattr(temporal_workflow.workflow, "execute_activity", fail_after_activity_retry)
    with pytest.raises(temporal_workflow.ActivityError):
        asyncio.run(instance.run(operation))
    state = HarnessStateStore(request.state_database_path)
    assert state.get_task(request.task_id)["status"] == "failed"
    with pytest.raises(ValueError, match="command ID"):
        dead_letter_temporal_task(
            request.state_database_path,
            request.task_id,
            expected_status="failed",
            operator_id="operator-c",
            reason="Missing source version has been investigated.",
            command_id=" ",
        )
    dead_letter = dead_letter_temporal_task(
        request.state_database_path,
        request.task_id,
        expected_status="failed",
        operator_id="operator-c",
        reason="Missing source version has been investigated.",
        command_id="dead-letter-1",
    )
    assert dead_letter == {
        "changed": True,
        "status": "dead_letter",
        "reconciliation_required": False,
        "result_artifact_name": None,
    }
    assert state.list_task_events(request.task_id)[-1]["payload"]["operator_id"] == "operator-c"
    assert dead_letter_temporal_task(
        request.state_database_path,
        request.task_id,
        expected_status="failed",
        operator_id="operator-c",
        reason="Missing source version has been investigated.",
        command_id="dead-letter-1",
    ) == {
        "changed": True,
        "status": "dead_letter",
        "reconciliation_required": False,
        "result_artifact_name": None,
    }


def test_temporal_index_reuses_completed_result_after_commit_ack_loss(tmp_path, monkeypatch) -> None:
    request = _synthetic_index_request(tmp_path)
    activity_input = IndustrialRagIndexActivityInput(request)
    real_complete_task = HarnessStateStore.complete_task
    real_build_release = temporal_workflow.build_release
    build_calls = []
    monkeypatch.setattr(temporal_workflow.activity, "heartbeat", lambda *args, **kwargs: None)

    def count_build_release(**kwargs):
        build_calls.append(True)
        return real_build_release(**kwargs)

    def commit_then_lose_ack(self, *args, **kwargs):
        real_complete_task(self, *args, **kwargs)
        raise RuntimeError("injected activity acknowledgement loss after task commit")

    monkeypatch.setattr(temporal_workflow, "build_release", count_build_release)
    monkeypatch.setattr(HarnessStateStore, "complete_task", commit_then_lose_ack)
    with pytest.raises(RuntimeError, match="acknowledgement loss"):
        build_index_activity(activity_input)

    monkeypatch.setattr(HarnessStateStore, "complete_task", real_complete_task)
    replayed = build_index_activity(activity_input)
    state = HarnessStateStore(request.state_database_path)
    assert replayed["status"] == "published"
    assert state.get_task(request.task_id)["status"] == "completed"
    assert state.get_task(request.task_id)["result"] == replayed
    assert state.get_artifact(request.task_id, "industrial_rag_operation_result") == replayed
    assert build_calls == [True]
    assert [event["stage"] for event in state.list_task_events(request.task_id)] == [
        "created", "index_started", "index_persisted", "completed",
    ]
    assert cli._store(request.database_path, request.object_root).active_release_id() == request.release_id


def test_temporal_dead_letter_preserves_committed_index_for_operator_reconciliation(tmp_path, monkeypatch) -> None:
    request = _synthetic_index_request(tmp_path)
    activity_input = IndustrialRagIndexActivityInput(request)
    real_complete_task = HarnessStateStore.complete_task
    real_build_release = temporal_workflow.build_release
    build_calls = []
    monkeypatch.setattr(temporal_workflow.activity, "heartbeat", lambda *args, **kwargs: None)

    def count_build_release(**kwargs):
        build_calls.append(True)
        return real_build_release(**kwargs)

    def fail_task_result_persistence(self, *_args, **_kwargs):
        raise RuntimeError("injected task-result persistence failure")

    monkeypatch.setattr(temporal_workflow, "build_release", count_build_release)
    monkeypatch.setattr(HarnessStateStore, "complete_task", fail_task_result_persistence)
    with pytest.raises(RuntimeError, match="task-result persistence failure"):
        build_index_activity(activity_input)

    monkeypatch.setattr(HarnessStateStore, "complete_task", real_complete_task)
    state = HarnessStateStore(request.state_database_path)
    artifact = state.get_artifact(request.task_id, "industrial_rag_operation_result")
    assert artifact is not None and artifact["status"] == "published"
    assert state.get_task(request.task_id)["status"] == "running"

    failure = temporal_workflow.fail_temporal_task_activity(temporal_workflow.TemporalFailureActivityInput(
        request.task_id,
        request.state_database_path,
        "RuntimeError",
        "task-result persistence failed after release publication",
    ))
    assert failure == {
        "accepted": True,
        "status": "failed",
        "reconciliation_required": True,
        "result_artifact_name": "industrial_rag_operation_result",
    }
    dead_letter = dead_letter_temporal_task(
        request.state_database_path,
        request.task_id,
        expected_status="failed",
        operator_id="operator-reconcile",
        reason="Release publication committed, but task-result completion failed.",
        command_id="dead-letter-after-publication",
    )
    assert dead_letter == {
        "changed": True,
        "status": "dead_letter",
        "reconciliation_required": True,
        "result_artifact_name": "industrial_rag_operation_result",
    }

    task_request = temporal_workflow._operation_identity("industrial_rag.index.v1", request)
    initialized = initialize_temporal_task_activity(TemporalTaskInitialization(
        request.task_id, request.idempotency_key, request.state_database_path, task_request,
    ))
    assert initialized == {
        "status": "dead_letter",
        "result": None,
        "reconciliation_required": True,
        "result_artifact_name": "industrial_rag_operation_result",
    }

    operation = IndustrialRagOperationRequest("index", asdict(request))
    instance = IndustrialRagOperationWorkflow()
    calls = []

    async def fake_execute_activity(fn, value, **_kwargs):
        calls.append(fn.__name__)
        return fn(value)

    monkeypatch.setattr(temporal_workflow.workflow, "execute_activity", fake_execute_activity)
    workflow_result = asyncio.run(instance.run(operation))
    assert workflow_result == {
        "status": "dead_letter",
        "task_id": request.task_id,
        "reconciliation_required": True,
        "result_artifact_name": "industrial_rag_operation_result",
    }
    assert calls == ["initialize_temporal_task_activity"]
    assert instance.state()["pending_reconciliation"] == ["industrial_rag_operation_result"]

    with pytest.raises(RuntimeError, match="explicit reconciliation"):
        build_index_activity(activity_input)
    assert state.get_task(request.task_id)["status"] == "dead_letter"
    assert state.get_task(request.task_id)["stage"] == "dead_letter"
    assert state.get_artifact(request.task_id, "industrial_rag_operation_result") == artifact
    assert [event["stage"] for event in state.list_task_events(request.task_id)] == [
        "created", "index_started", "index_persisted", "failed", "dead_letter",
    ]
    assert cli._store(request.database_path, request.object_root).active_release_id() == request.release_id
    assert build_calls == [True]


def test_temporal_cancelled_task_with_committed_artifact_is_not_automatically_completed(tmp_path, monkeypatch) -> None:
    request = _synthetic_index_request(tmp_path)
    task_request = temporal_workflow._operation_identity("industrial_rag.index.v1", request)
    state = HarnessStateStore(request.state_database_path)
    state.open_task(request.task_id, request.idempotency_key, task_request, max_provider_calls=0)
    artifact = {"status": "published", "release_id": request.release_id}
    state.put_artifact(request.task_id, "industrial_rag_operation_result", artifact)
    assert state.transition_task_status(
        request.task_id,
        expected_statuses={"running"},
        status="cancelled",
        stage="cancelled",
        event_key="operator-cancel-after-commit",
        payload={"operator_id": "operator-cancel", "reason": "Stop automatic replay."},
    )

    initialized = initialize_temporal_task_activity(TemporalTaskInitialization(
        request.task_id, request.idempotency_key, request.state_database_path, task_request,
    ))
    assert initialized == {
        "status": "cancelled",
        "result": None,
        "reconciliation_required": True,
        "result_artifact_name": "industrial_rag_operation_result",
    }
    with pytest.raises(RuntimeError, match="task is cancelled"):
        build_index_activity(IndustrialRagIndexActivityInput(request))
    assert state.get_task(request.task_id)["status"] == "cancelled"
    assert state.get_task(request.task_id)["stage"] == "cancelled"
    assert state.get_task(request.task_id).get("result") is None
    assert state.get_artifact(request.task_id, "industrial_rag_operation_result") == artifact
