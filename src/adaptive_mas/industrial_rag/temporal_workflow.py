"""Temporal worker for durable local ingestion, release indexing, research replay, and offline evaluation."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import re
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, replace
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from temporalio import activity, workflow
from temporalio.client import Client
from temporalio.common import RetryPolicy
from temporalio.exceptions import ActivityError
from temporalio.worker import Worker

with workflow.unsafe.imports_passed_through():
    from adaptive_mas.agents.harness import MAX_CALLS_PER_EXPERIMENT, CallBudget, QwenClient, run_research_harness
    from adaptive_mas.data.research import MembershipSnapshot, ResearchData
    from adaptive_mas.harness_state import HarnessStateStore
    from adaptive_mas.industrial_rag import cli
    from adaptive_mas.industrial_rag.cli import build_release, ingest_file
    from adaptive_mas.industrial_rag.evaluation import evaluate_manifest_bytes


TASK_QUEUE = "raama-industrial-rag-v1"


@dataclass(frozen=True)
class IndustrialRagIngestRequest:
    task_id: str
    idempotency_key: str
    source_path: str
    metadata_path: str
    database_path: str
    object_root: str
    scratch_root: str
    state_database_path: str
    pdf_backend: str = "pypdf"
    docling_artifacts_path: str | None = None
    activity_heartbeat_timeout_seconds: int = 60
    source_sha256: str | None = None
    metadata_sha256: str | None = None
    captured_at: str | None = None

    def __post_init__(self) -> None:
        for name in (
            "task_id", "idempotency_key", "source_path", "metadata_path", "database_path",
            "object_root", "scratch_root", "state_database_path",
        ):
            if not getattr(self, name).strip():
                raise ValueError(f"{name} must be nonempty")
        if self.pdf_backend not in {"pypdf", "docling"}:
            raise ValueError("pdf_backend must be pypdf or docling")
        if not 3 <= self.activity_heartbeat_timeout_seconds <= 900:
            raise ValueError("activity heartbeat timeout must be within 3..900 seconds")
        captured_fields = (self.source_sha256, self.metadata_sha256, self.captured_at)
        if any(value is not None for value in captured_fields):
            if not all(value is not None for value in captured_fields):
                raise ValueError("captured source hashes and captured_at must be provided together")
            if not re.fullmatch(r"[0-9a-f]{64}", str(self.source_sha256)):
                raise ValueError("source_sha256 must be a lowercase SHA-256 digest")
            if not re.fullmatch(r"[0-9a-f]{64}", str(self.metadata_sha256)):
                raise ValueError("metadata_sha256 must be a lowercase SHA-256 digest")
            captured = datetime.fromisoformat(str(self.captured_at))
            if captured.tzinfo is None:
                raise ValueError("captured_at must include a timezone")

    @property
    def is_captured(self) -> bool:
        return self.source_sha256 is not None and self.metadata_sha256 is not None and self.captured_at is not None


@dataclass(frozen=True)
class IndustrialRagIndexRequest:
    task_id: str
    idempotency_key: str
    release_id: str
    version_ids: tuple[str, ...]
    database_path: str
    object_root: str
    state_database_path: str
    actor_id: str
    reason: str

    def __post_init__(self) -> None:
        for name in (
            "task_id", "idempotency_key", "release_id", "database_path", "object_root",
            "state_database_path", "actor_id", "reason",
        ):
            if not getattr(self, name).strip():
                raise ValueError(f"{name} must be nonempty")
        if not self.version_ids or any(not value.strip() for value in self.version_ids):
            raise ValueError("version_ids must contain at least one nonempty ID")
        if len(set(self.version_ids)) != len(self.version_ids):
            raise ValueError("version_ids must be unique")


@dataclass(frozen=True)
class IndustrialRagEvaluationRequest:
    task_id: str
    idempotency_key: str
    manifest_path: str
    predictions_path: str
    scratch_root: str
    state_database_path: str
    manifest_sha256: str | None = None
    predictions_sha256: str | None = None
    captured_at: str | None = None

    def __post_init__(self) -> None:
        for name in (
            "task_id", "idempotency_key", "manifest_path", "predictions_path",
            "scratch_root", "state_database_path",
        ):
            if not getattr(self, name).strip():
                raise ValueError(f"{name} must be nonempty")
        captured_fields = (self.manifest_sha256, self.predictions_sha256, self.captured_at)
        if any(value is not None for value in captured_fields):
            if not all(value is not None for value in captured_fields):
                raise ValueError("captured evaluation hashes and captured_at must be provided together")
            if not re.fullmatch(r"[0-9a-f]{64}", str(self.manifest_sha256)):
                raise ValueError("manifest_sha256 must be a lowercase SHA-256 digest")
            if not re.fullmatch(r"[0-9a-f]{64}", str(self.predictions_sha256)):
                raise ValueError("predictions_sha256 must be a lowercase SHA-256 digest")
            captured = datetime.fromisoformat(str(self.captured_at))
            if captured.tzinfo is None:
                raise ValueError("captured_at must include a timezone")

    @property
    def is_captured(self) -> bool:
        return (
            self.manifest_sha256 is not None
            and self.predictions_sha256 is not None
            and self.captured_at is not None
        )


@dataclass(frozen=True)
class IndustrialRagResearchRequest:
    task_id: str
    idempotency_key: str
    bundle_path: str
    scratch_root: str
    state_database_path: str
    response_cache_dir: str
    call_ledger_path: str
    request_archive_dir: str
    bundle_sha256: str | None = None
    captured_at: str | None = None

    def __post_init__(self) -> None:
        for name in (
            "task_id", "idempotency_key", "bundle_path", "scratch_root", "state_database_path",
            "response_cache_dir", "call_ledger_path", "request_archive_dir",
        ):
            if not getattr(self, name).strip():
                raise ValueError(f"{name} must be nonempty")
        if self.bundle_sha256 is not None or self.captured_at is not None:
            if self.bundle_sha256 is None or self.captured_at is None:
                raise ValueError("captured research bundle hash and captured_at must be provided together")
            if not re.fullmatch(r"[0-9a-f]{64}", self.bundle_sha256):
                raise ValueError("bundle_sha256 must be a lowercase SHA-256 digest")
            captured = datetime.fromisoformat(self.captured_at)
            if captured.tzinfo is None:
                raise ValueError("captured_at must include a timezone")

    @property
    def is_captured(self) -> bool:
        return self.bundle_sha256 is not None and self.captured_at is not None


@dataclass(frozen=True)
class IndustrialRagOperationRequest:
    operation: str
    payload: dict[str, object]

    def __post_init__(self) -> None:
        if self.operation not in {"index", "evaluate", "research"}:
            raise ValueError("Temporal RAG operation must be index, evaluate, or research")
        if not self.payload:
            raise ValueError("Temporal RAG operation payload must be nonempty")


@dataclass(frozen=True)
class TemporalWorkflowControl:
    command_id: str
    action: str
    operator_id: str
    reason: str

    def __post_init__(self) -> None:
        if not self.command_id.strip() or not self.operator_id.strip() or not self.reason.strip():
            raise ValueError("control command ID, operator identity, and reason are required")
        if self.action not in {"pause", "resume", "cancel"}:
            raise ValueError("action must be pause, resume, or cancel")


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _write_capture_file(path: Path, payload: bytes, expected_sha256: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError:
        existing = path.read_bytes()
        if _sha256(existing) != expected_sha256 or existing != payload:
            raise ValueError("immutable Temporal capture file already contains different bytes") from None


def _capture_metadata(capture_dir: Path, input_hashes: dict[str, str]) -> str:
    metadata_path = capture_dir / "capture-metadata.json"
    capture_dir.mkdir(parents=True, exist_ok=True)
    try:
        captured_at = datetime.now(timezone.utc).isoformat()
        payload = json.dumps(
            {"captured_at": captured_at, "input_sha256": input_hashes},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        with metadata_path.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        return captured_at
    except FileExistsError:
        try:
            saved = json.loads(metadata_path.read_text(encoding="utf-8"))
            saved_time = str(saved["captured_at"])
            saved_hashes = saved["input_sha256"]
            captured = datetime.fromisoformat(saved_time)
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise ValueError("immutable Temporal capture metadata is invalid") from error
        if saved_hashes != input_hashes or captured.tzinfo is None:
            raise ValueError("Temporal idempotency key already captured different input bytes") from None
        return saved_time


def capture_ingest_request(request: IndustrialRagIngestRequest) -> IndustrialRagIngestRequest:
    """Copy local inputs into an idempotent immutable capture before Temporal scheduling."""
    if request.is_captured:
        read_captured_input_bytes(request)
        return request

    source = Path(request.source_path).resolve(strict=True)
    metadata_source = Path(request.metadata_path).resolve(strict=True)
    if not source.is_file() or not metadata_source.is_file():
        raise FileNotFoundError("Temporal ingest source and metadata must be regular files")
    source_bytes = source.read_bytes()
    metadata_bytes = metadata_source.read_bytes()
    source_sha256 = _sha256(source_bytes)
    metadata_sha256 = _sha256(metadata_bytes)
    try:
        metadata = json.loads(metadata_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("Temporal ingest metadata must be valid UTF-8 JSON") from error
    if not isinstance(metadata, dict):
        raise ValueError("Temporal ingest metadata must contain one JSON object")
    expected_sha256 = metadata.get("expected_sha256")
    if expected_sha256 and expected_sha256 != source_sha256:
        raise ValueError("local source SHA-256 does not match the supplied manifest")

    capture_id = hashlib.sha256(
        f"{request.task_id}\0{request.idempotency_key}".encode("utf-8")
    ).hexdigest()[:32]
    capture_dir = Path(request.scratch_root).expanduser().resolve() / "temporal-captures" / capture_id
    source_suffix = source.suffix if re.fullmatch(r"\.[A-Za-z0-9]{1,10}", source.suffix) else ".bin"
    source_capture = capture_dir / f"source-{source_sha256}{source_suffix}"
    metadata_capture = capture_dir / f"metadata-{metadata_sha256}.json"
    captured_at = _capture_metadata(capture_dir, {
        "source_sha256": source_sha256,
        "metadata_sha256": metadata_sha256,
    })
    _write_capture_file(source_capture, source_bytes, source_sha256)
    _write_capture_file(metadata_capture, metadata_bytes, metadata_sha256)
    return replace(
        request,
        source_path=str(source_capture),
        metadata_path=str(metadata_capture),
        source_sha256=source_sha256,
        metadata_sha256=metadata_sha256,
        captured_at=captured_at,
    )


def read_captured_input_bytes(request: IndustrialRagIngestRequest) -> tuple[bytes, bytes]:
    if not request.is_captured:
        raise ValueError("Temporal ingest request must be captured before scheduling")
    try:
        source_bytes = Path(request.source_path).read_bytes()
        metadata_bytes = Path(request.metadata_path).read_bytes()
    except OSError as error:
        raise ValueError("captured Temporal ingest input is unavailable") from error
    if _sha256(source_bytes) != request.source_sha256:
        raise ValueError("captured source bytes do not match the scheduled SHA-256")
    if _sha256(metadata_bytes) != request.metadata_sha256:
        raise ValueError("captured metadata bytes do not match the scheduled SHA-256")
    return source_bytes, metadata_bytes


def capture_evaluation_request(
    request: IndustrialRagEvaluationRequest,
) -> IndustrialRagEvaluationRequest:
    """Freeze exact manifest and prediction bytes before an evaluation workflow is scheduled."""
    if request.is_captured:
        read_captured_evaluation_bytes(request)
        return request
    manifest = Path(request.manifest_path).resolve(strict=True)
    predictions = Path(request.predictions_path).resolve(strict=True)
    if not manifest.is_file() or not predictions.is_file():
        raise FileNotFoundError("Temporal evaluation manifest and predictions must be regular files")
    manifest_bytes = manifest.read_bytes()
    prediction_bytes = predictions.read_bytes()
    manifest_sha256 = _sha256(manifest_bytes)
    predictions_sha256 = _sha256(prediction_bytes)
    capture_id = hashlib.sha256(
        f"{request.task_id}\0{request.idempotency_key}".encode("utf-8")
    ).hexdigest()[:32]
    capture_dir = Path(request.scratch_root).expanduser().resolve() / "temporal-captures" / capture_id
    manifest_capture = capture_dir / f"evaluation-manifest-{manifest_sha256}.json"
    predictions_capture = capture_dir / f"evaluation-predictions-{predictions_sha256}.json"
    captured_at = _capture_metadata(capture_dir, {
        "manifest_sha256": manifest_sha256,
        "predictions_sha256": predictions_sha256,
    })
    _write_capture_file(manifest_capture, manifest_bytes, manifest_sha256)
    _write_capture_file(predictions_capture, prediction_bytes, predictions_sha256)
    return replace(
        request,
        manifest_path=str(manifest_capture),
        predictions_path=str(predictions_capture),
        manifest_sha256=manifest_sha256,
        predictions_sha256=predictions_sha256,
        captured_at=captured_at,
    )


def read_captured_evaluation_bytes(request: IndustrialRagEvaluationRequest) -> tuple[bytes, bytes]:
    if not request.is_captured:
        raise ValueError("Temporal evaluation inputs must be captured before scheduling")
    try:
        manifest_bytes = Path(request.manifest_path).read_bytes()
        prediction_bytes = Path(request.predictions_path).read_bytes()
    except OSError as error:
        raise ValueError("captured Temporal evaluation input is unavailable") from error
    if _sha256(manifest_bytes) != request.manifest_sha256:
        raise ValueError("captured evaluation manifest does not match the scheduled SHA-256")
    if _sha256(prediction_bytes) != request.predictions_sha256:
        raise ValueError("captured evaluation predictions do not match the scheduled SHA-256")
    return manifest_bytes, prediction_bytes


def capture_research_request(request: IndustrialRagResearchRequest) -> IndustrialRagResearchRequest:
    """Capture a complete local research bundle before scheduling its offline replay."""
    if request.is_captured:
        read_captured_research_bytes(request)
        return request
    source = Path(request.bundle_path).resolve(strict=True)
    if not source.is_file():
        raise FileNotFoundError("Temporal research bundle must be a regular file")
    payload = source.read_bytes()
    digest = _sha256(payload)
    capture_id = hashlib.sha256(
        f"{request.task_id}\0{request.idempotency_key}".encode("utf-8")
    ).hexdigest()[:32]
    capture_dir = Path(request.scratch_root).expanduser().resolve() / "temporal-captures" / capture_id
    capture_path = capture_dir / f"research-bundle-{digest}.json"
    captured_at = _capture_metadata(capture_dir, {"research_bundle_sha256": digest})
    _write_capture_file(capture_path, payload, digest)
    return replace(
        request,
        bundle_path=str(capture_path),
        bundle_sha256=digest,
        captured_at=captured_at,
    )


def read_captured_research_bytes(request: IndustrialRagResearchRequest) -> bytes:
    if not request.is_captured:
        raise ValueError("Temporal research bundle must be captured before scheduling")
    try:
        payload = Path(request.bundle_path).read_bytes()
    except OSError as error:
        raise ValueError("captured Temporal research bundle is unavailable") from error
    if _sha256(payload) != request.bundle_sha256:
        raise ValueError("captured research bundle does not match the scheduled SHA-256")
    return payload


@dataclass(frozen=True)
class TemporalTaskInitialization:
    task_id: str
    idempotency_key: str
    state_database_path: str
    request: dict[str, Any]
    result_artifact: str = "industrial_rag_operation_result"


@dataclass(frozen=True)
class TemporalControlActivityInput:
    task_id: str
    state_database_path: str
    control: TemporalWorkflowControl


@dataclass(frozen=True)
class TemporalFailureActivityInput:
    task_id: str
    state_database_path: str
    error_type: str
    message: str
    result_artifact: str = "industrial_rag_operation_result"


@dataclass(frozen=True)
class IndustrialRagIndexActivityInput:
    request: IndustrialRagIndexRequest


@dataclass(frozen=True)
class IndustrialRagEvaluationActivityInput:
    request: IndustrialRagEvaluationRequest


@dataclass(frozen=True)
class IndustrialRagResearchActivityInput:
    request: IndustrialRagResearchRequest


@dataclass(frozen=True)
class IndustrialRagIngestActivityInput:
    request: IndustrialRagIngestRequest


def _ingest_task_request(request: IndustrialRagIngestRequest) -> dict[str, object]:
    return {
        "operation": "industrial_rag.ingest.v1",
        "request": asdict(request),
        "captured_at": request.captured_at,
        "source_sha256": request.source_sha256,
        "metadata_sha256": request.metadata_sha256,
    }


@activity.defn
def ingest_document_activity(value: IndustrialRagIngestActivityInput) -> dict[str, Any]:
    request = value.request
    if not request.is_captured:
        raise ValueError("Temporal ingest activity requires source and metadata captured before scheduling")
    state_store = HarnessStateStore(request.state_database_path)
    task_request = _ingest_task_request(request)
    task = state_store.open_task(
        request.task_id, request.idempotency_key, task_request, max_provider_calls=0,
    )
    if task["status"] == "completed":
        result = task["result"]
        if not isinstance(result, dict):
            raise RuntimeError("completed Temporal ingest task has no persisted result")
        return result
    if task["status"] != "running":
        saved_result = state_store.get_artifact(request.task_id, "industrial_rag_ingest_result")
        reconciliation = "; committed ingest artifact requires explicit reconciliation" if saved_result is not None else ""
        raise RuntimeError(f"Temporal ingest task is {task['status']}{reconciliation}")

    saved_result = state_store.get_artifact(request.task_id, "industrial_rag_ingest_result")
    if saved_result is not None:
        state_store.complete_task(
            request.task_id, saved_result, expected_statuses={"running"},
        )
        return saved_result

    source_bytes, metadata_bytes = read_captured_input_bytes(request)

    state_store.checkpoint_task(
        request.task_id, "ingest_started", {"operation": "industrial_rag.ingest.v1"},
        event_key="industrial_rag_ingest_started",
    )
    activity.heartbeat({"stage": "ingesting"})

    result = ingest_file(
        source_path=request.source_path,
        metadata_path=request.metadata_path,
        database_path=request.database_path,
        object_root=request.object_root,
        scratch_root=request.scratch_root,
        pdf_backend=request.pdf_backend,
        docling_artifacts_path=request.docling_artifacts_path,
        source_bytes=source_bytes,
        metadata_bytes=metadata_bytes,
        captured_at=request.captured_at,
        ingested_at_factory=lambda: datetime.now(timezone.utc),
    )
    state_store.put_artifact(request.task_id, "industrial_rag_ingest_result", result)
    state_store.checkpoint_task(
        request.task_id, "ingest_persisted",
        {"content_sha256": result["content_sha256"], "status": result["status"]},
        event_key="industrial_rag_ingest_persisted",
    )
    state_store.complete_task(request.task_id, result, expected_statuses={"running"})
    activity.heartbeat({"stage": "completed", "content_sha256": result["content_sha256"]})
    return result


def _operation_identity(operation: str, request: object) -> dict[str, object]:
    return {"operation": operation, "request": asdict(request)}


def _load_operation_result(store: HarnessStateStore, task_id: str) -> dict[str, object] | None:
    task = store.get_task(task_id)
    if task is None:
        raise KeyError(f"unknown task_id {task_id}")
    if task["status"] == "completed":
        result = task.get("result")
        if isinstance(result, dict):
            return result
        raise RuntimeError("completed Temporal operation task has no persisted result")
    if task["status"] != "running":
        saved = store.get_artifact(task_id, "industrial_rag_operation_result")
        reconciliation = "; committed operation artifact requires explicit reconciliation" if saved is not None else ""
        raise RuntimeError(f"Temporal operation task is {task['status']}{reconciliation}")
    saved = store.get_artifact(task_id, "industrial_rag_operation_result")
    if saved is not None:
        store.complete_task(task_id, saved, expected_statuses={"running"})
        return saved
    return None


@activity.defn
def initialize_temporal_task_activity(value: TemporalTaskInitialization) -> dict[str, object]:
    store = HarnessStateStore(value.state_database_path)
    task = store.open_task(
        value.task_id, value.idempotency_key, value.request, max_provider_calls=0,
    )
    if task["status"] == "completed":
        result = task.get("result")
        if isinstance(result, dict):
            return {"status": "completed", "result": result}
        raise RuntimeError("completed Temporal task has no persisted result")
    if task["status"] != "running":
        artifact = store.get_artifact(value.task_id, value.result_artifact)
        result: dict[str, object] = {
            "status": str(task["status"]),
            "result": None,
            "reconciliation_required": artifact is not None,
        }
        if artifact is not None:
            result["result_artifact_name"] = value.result_artifact
        return result
    return {"status": "running", "result": None}


@activity.defn
def verify_index_inputs_activity(value: IndustrialRagIndexActivityInput) -> dict[str, object]:
    request = value.request
    store = cli._store(request.database_path, request.object_root)
    manifest = store.source_manifest(list(request.version_ids))
    found = {str(row.get("version_id")) for row in manifest}
    missing = sorted(set(request.version_ids) - found)
    if missing:
        raise FileNotFoundError(f"document versions are not registered: {', '.join(missing)}")
    task_store = HarnessStateStore(request.state_database_path)
    task_store.checkpoint_task(
        request.task_id,
        "index_inputs_verified",
        {"release_id": request.release_id, "version_ids": list(request.version_ids)},
        event_key="industrial_rag_index_inputs_verified",
    )
    activity.heartbeat({"stage": "index_inputs_verified", "version_count": len(found)})
    return {"release_id": request.release_id, "version_ids": list(request.version_ids), "source_count": len(found)}


@activity.defn
def build_index_activity(value: IndustrialRagIndexActivityInput) -> dict[str, object]:
    request = value.request
    store = HarnessStateStore(request.state_database_path)
    task_request = _operation_identity("industrial_rag.index.v1", request)
    store.open_task(request.task_id, request.idempotency_key, task_request, max_provider_calls=0)
    saved = _load_operation_result(store, request.task_id)
    if saved is not None:
        return saved
    task = store.get_task(request.task_id)
    if task is not None and task["status"] in {"cancelled", "dead_letter"}:
        raise RuntimeError(f"Temporal index task is {task['status']}")
    store.checkpoint_task(
        request.task_id,
        "index_started",
        {"release_id": request.release_id, "version_ids": list(request.version_ids)},
        event_key="industrial_rag_index_started",
    )
    activity.heartbeat({"stage": "building_index", "release_id": request.release_id})
    result = build_release(
        release_id=request.release_id,
        version_ids=list(request.version_ids),
        database_path=request.database_path,
        object_root=request.object_root,
        actor_id=request.actor_id,
        reason=request.reason,
    )
    store.put_artifact(request.task_id, "industrial_rag_operation_result", result)
    store.checkpoint_task(
        request.task_id,
        "index_persisted",
        {"release_id": request.release_id, "status": result.get("status")},
        event_key="industrial_rag_index_persisted",
    )
    store.complete_task(request.task_id, result, expected_statuses={"running"})
    return result


def _read_evaluation_objects(request: IndustrialRagEvaluationRequest) -> tuple[dict[str, object], dict[str, object]]:
    manifest_bytes, predictions_bytes = read_captured_evaluation_bytes(request)
    try:
        manifest = json.loads(manifest_bytes.decode("utf-8"))
        predictions = json.loads(predictions_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("captured evaluation inputs must be valid UTF-8 JSON") from error
    if not isinstance(manifest, dict) or not isinstance(predictions, dict):
        raise ValueError("captured evaluation inputs must each contain one JSON object")
    return manifest, predictions


@activity.defn
def verify_evaluation_inputs_activity(value: IndustrialRagEvaluationActivityInput) -> dict[str, object]:
    request = value.request
    manifest, predictions = _read_evaluation_objects(request)
    if not isinstance(manifest.get("cases"), list) or not isinstance(predictions.get("cases"), list):
        raise ValueError("evaluation manifest and predictions must each contain a cases array")
    task_store = HarnessStateStore(request.state_database_path)
    task_store.checkpoint_task(
        request.task_id,
        "evaluation_inputs_verified",
        {
            "manifest_sha256": request.manifest_sha256,
            "predictions_sha256": request.predictions_sha256,
            "case_count": len(manifest["cases"]),
            "prediction_count": len(predictions["cases"]),
        },
        event_key="industrial_rag_evaluation_inputs_verified",
    )
    activity.heartbeat({"stage": "evaluation_inputs_verified"})
    return {
        "manifest_sha256": request.manifest_sha256,
        "predictions_sha256": request.predictions_sha256,
        "case_count": len(manifest["cases"]),
    }


@activity.defn
def evaluate_manifest_activity(value: IndustrialRagEvaluationActivityInput) -> dict[str, object]:
    request = value.request
    manifest_bytes, prediction_bytes = read_captured_evaluation_bytes(request)
    store = HarnessStateStore(request.state_database_path)
    task_request = _operation_identity("industrial_rag.evaluate.v1", request)
    store.open_task(request.task_id, request.idempotency_key, task_request, max_provider_calls=0)
    saved = _load_operation_result(store, request.task_id)
    if saved is not None:
        return saved
    task = store.get_task(request.task_id)
    if task is not None and task["status"] in {"cancelled", "dead_letter"}:
        raise RuntimeError(f"Temporal evaluation task is {task['status']}")
    store.checkpoint_task(
        request.task_id,
        "evaluation_started",
        {"manifest_sha256": request.manifest_sha256, "predictions_sha256": request.predictions_sha256},
        event_key="industrial_rag_evaluation_started",
    )
    result = evaluate_manifest_bytes(manifest_bytes, prediction_bytes)
    store.put_artifact(request.task_id, "industrial_rag_operation_result", result)
    store.checkpoint_task(
        request.task_id,
        "evaluation_persisted",
        {"benchmark_sha256": result["benchmark_sha256"], "prediction_sha256": result["prediction_sha256"]},
        event_key="industrial_rag_evaluation_persisted",
    )
    store.complete_task(request.task_id, result, expected_statuses={"running"})
    return result


def _bundle_date(value: object, name: str) -> date:
    if not isinstance(value, str):
        raise ValueError(f"research bundle {name} must be an ISO date string")
    try:
        return date.fromisoformat(value)
    except ValueError as error:
        raise ValueError(f"research bundle {name} must be an ISO date string") from error


def _research_bundle(payload: bytes) -> dict[str, Any]:
    try:
        bundle = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("captured research bundle must be valid UTF-8 JSON") from error
    if not isinstance(bundle, dict) or bundle.get("schema_version") != "adaptive-mas.temporal-research-input.v1":
        raise ValueError("research bundle has an unsupported schema_version")
    raw_research = bundle.get("research_data")
    if not isinstance(raw_research, dict):
        raise ValueError("research bundle requires research_data")
    trading_dates = raw_research.get("trading_dates")
    if not isinstance(trading_dates, list) or not trading_dates:
        raise ValueError("research_data.trading_dates must be a nonempty array")

    def bars(name: str) -> list[dict[str, object]]:
        rows = raw_research.get(name)
        if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
            raise ValueError(f"research_data.{name} must be an array of objects")
        result: list[dict[str, object]] = []
        for row in rows:
            normalized = dict(row)
            for key in ("trade_date", "available_at"):
                if key not in normalized:
                    raise ValueError(f"research_data.{name} rows require {key}")
                normalized[key] = _bundle_date(normalized[key], f"{name}.{key}")
            result.append(normalized)
        return result

    raw_memberships = raw_research.get("membership_snapshots")
    if not isinstance(raw_memberships, list) or not raw_memberships:
        raise ValueError("research_data.membership_snapshots must be a nonempty array")
    memberships: list[MembershipSnapshot] = []
    for row in raw_memberships:
        if not isinstance(row, dict) or not isinstance(row.get("members"), list):
            raise ValueError("membership snapshot rows require a members array")
        members = row["members"]
        if any(not isinstance(member, str) or not member for member in members):
            raise ValueError("membership snapshot member IDs must be nonempty strings")
        source_ref = row.get("source_ref")
        if not isinstance(source_ref, str) or not source_ref.strip():
            raise ValueError("membership snapshot source_ref must be nonempty")
        memberships.append(MembershipSnapshot(
            event_time=_bundle_date(row.get("event_time"), "membership_snapshot.event_time"),
            available_at=_bundle_date(row.get("available_at"), "membership_snapshot.available_at"),
            members=frozenset(members),
            source_ref=source_ref,
        ))
    research = ResearchData(
        trading_dates=[_bundle_date(value, "trading_dates") for value in trading_dates],
        adjusted_bars=bars("adjusted_bars"),
        execution_bars=bars("execution_bars"),
        index_bars=bars("index_bars"),
        membership_snapshots=memberships,
    )
    as_of = _bundle_date(bundle.get("as_of"), "as_of")
    if as_of not in research.trading_dates:
        raise ValueError("research bundle as_of is absent from its trading calendar")
    candidate = bundle.get("candidate")
    required_candidate = (
        "context_policy", "fundamental_prompt", "handoff_schema", "manager_prompt",
        "market_prompt", "procedural_experience", "research_policy_json",
    )
    if not isinstance(candidate, dict) or any(
        not isinstance(candidate.get(key), str) or not candidate[key]
        for key in required_candidate
    ):
        raise ValueError("research bundle candidate is missing a required manager or expert module")
    config = bundle.get("config")
    if not isinstance(config, dict):
        raise ValueError("research bundle config must be an object")
    public_snapshot = bundle.get("public_snapshot")
    if public_snapshot is not None and not isinstance(public_snapshot, dict):
        raise ValueError("research bundle public_snapshot must be an object or null")
    endpoint = bundle.get("endpoint")
    model = bundle.get("model")
    candidate_version_id = bundle.get("candidate_version_id")
    package_identity_sha256 = bundle.get("package_identity_sha256")
    if not all(isinstance(item, str) and item.strip() for item in (endpoint, model, candidate_version_id)):
        raise ValueError("research bundle endpoint, model, and candidate_version_id must be nonempty")
    if not isinstance(package_identity_sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", package_identity_sha256):
        raise ValueError("research bundle package_identity_sha256 must be a lowercase SHA-256 digest")
    required_observations = bundle.get("required_observations", [])
    if not isinstance(required_observations, list) or any(not isinstance(item, dict) for item in required_observations):
        raise ValueError("research bundle required_observations must be an array of objects")
    procedural_memory = bundle.get("procedural_memory")
    if procedural_memory is not None and not isinstance(procedural_memory, dict):
        raise ValueError("research bundle procedural_memory must be an object or null")
    model_risk_context = bundle.get("model_risk_context")
    if model_risk_context is not None and not isinstance(model_risk_context, dict):
        raise ValueError("research bundle model_risk_context must be an object or null")
    return {
        "as_of": as_of,
        "research": research,
        "public_snapshot": public_snapshot,
        "candidate": candidate,
        "config": config,
        "candidate_version_id": candidate_version_id,
        "required_observations": required_observations,
        "procedural_memory": procedural_memory,
        "model_risk_context": model_risk_context,
        "endpoint": endpoint,
        "model": model,
        "package_identity_sha256": package_identity_sha256,
    }


@activity.defn
def verify_research_inputs_activity(value: IndustrialRagResearchActivityInput) -> dict[str, object]:
    request = value.request
    bundle = _research_bundle(read_captured_research_bytes(request))
    store = HarnessStateStore(request.state_database_path)
    store.checkpoint_task(
        request.task_id,
        "research_inputs_verified",
        {
            "bundle_sha256": request.bundle_sha256,
            "as_of": bundle["as_of"].isoformat(),
            "candidate_version_id": bundle["candidate_version_id"],
            "offline_only": True,
        },
        event_key="financial_research_inputs_verified",
    )
    activity.heartbeat({"stage": "research_inputs_verified", "offline_only": True})
    return {
        "bundle_sha256": request.bundle_sha256,
        "as_of": bundle["as_of"].isoformat(),
        "offline_only": True,
    }


@activity.defn
def run_research_activity(value: IndustrialRagResearchActivityInput) -> dict[str, object]:
    request = value.request
    bundle = _research_bundle(read_captured_research_bytes(request))
    store = HarnessStateStore(request.state_database_path)
    task_request = _operation_identity("financial_research.cached_replay.v1", request)
    store.open_task(request.task_id, request.idempotency_key, task_request, max_provider_calls=0)
    saved = _load_operation_result(store, request.task_id)
    if saved is not None:
        return saved
    task = store.get_task(request.task_id)
    if task is not None and task["status"] in {"cancelled", "dead_letter"}:
        raise RuntimeError(f"Temporal research task is {task['status']}")
    store.checkpoint_task(
        request.task_id,
        "research_started",
        {"bundle_sha256": request.bundle_sha256, "provider_calls_max": 0},
        event_key="financial_research_started",
    )
    budget = CallBudget(
        max_calls=MAX_CALLS_PER_EXPERIMENT,
        ledger_path=Path(request.call_ledger_path),
        response_cache_dir=Path(request.response_cache_dir),
        request_archive_dir=Path(request.request_archive_dir),
        state_store=store,
        task_id=request.task_id,
        budget_scope_id=f"temporal-offline:{request.task_id}",
        task_provider_call_limit=0,
        package_identity_sha256=str(bundle["package_identity_sha256"]),
    )
    budget.limit_next_task(10)
    client = QwenClient(
        str(bundle["endpoint"]), "offline-cache-only", str(bundle["model"]), offline_only=True,
    )
    activity.heartbeat({"stage": "running_multi_expert_manager_harness", "offline_only": True})
    result = run_research_harness(
        bundle["research"],
        bundle["as_of"],
        bundle["public_snapshot"],
        bundle["candidate"],
        client,
        budget,
        config=bundle["config"],
        candidate_version_id=str(bundle["candidate_version_id"]),
        required_observations=bundle["required_observations"],
        procedural_memory=bundle["procedural_memory"],
        model_risk_context=bundle["model_risk_context"],
    )
    if result.get("status") != "completed" or budget.provider_call_count != 0:
        raise RuntimeError("offline Temporal research did not complete within its zero-provider-call contract")
    persisted: dict[str, object] = {
        "status": "completed",
        "as_of": bundle["as_of"].isoformat(),
        "candidate_version_id": bundle["candidate_version_id"],
        "package_identity_sha256": bundle["package_identity_sha256"],
        "bundle_sha256": request.bundle_sha256,
        "provider_calls": 0,
        "research_result": result,
    }
    store.put_artifact(request.task_id, "industrial_rag_operation_result", persisted)
    store.checkpoint_task(
        request.task_id,
        "research_persisted",
        {"bundle_sha256": request.bundle_sha256, "provider_calls": 0},
        event_key="financial_research_persisted",
    )
    store.complete_task(request.task_id, persisted, expected_statuses={"running"})
    return persisted


@activity.defn
def apply_temporal_control_activity(value: TemporalControlActivityInput) -> dict[str, object]:
    store = HarnessStateStore(value.state_database_path)
    control = value.control
    if control.action == "pause":
        expected, target, stage = {"running", "paused"}, "paused", "paused"
    elif control.action == "resume":
        expected, target, stage = {"running", "paused"}, "running", "resumed"
    else:
        expected, target, stage = {"running", "paused"}, "cancelled", "cancelled"
    accepted = store.transition_task_status(
        value.task_id,
        expected_statuses=expected,
        status=target,
        stage=stage,
        event_key=f"temporal-control:{control.command_id}",
        payload=asdict(control),
    )
    task = store.get_task(value.task_id)
    return {"accepted": accepted, "status": task["status"] if task else "missing", "action": control.action}


@activity.defn
def fail_temporal_task_activity(value: TemporalFailureActivityInput) -> dict[str, object]:
    store = HarnessStateStore(value.state_database_path)
    accepted = store.transition_task_status(
        value.task_id,
        expected_statuses={"running", "paused"},
        status="failed",
        stage="failed",
        event_key="temporal-workflow-failed",
        payload={"error_type": value.error_type, "message": value.message[:2000]},
    )
    task = store.get_task(value.task_id)
    artifact = store.get_artifact(value.task_id, value.result_artifact)
    outcome: dict[str, object] = {
        "accepted": accepted,
        "status": task["status"] if task else "missing",
        "reconciliation_required": artifact is not None,
    }
    if artifact is not None:
        outcome["result_artifact_name"] = value.result_artifact
    if task is not None and task["status"] == "completed" and isinstance(task.get("result"), dict):
        outcome["result"] = task["result"]
    return outcome


def dead_letter_temporal_task(
    state_database_path: str | Path,
    task_id: str,
    *,
    expected_status: str,
    operator_id: str,
    reason: str,
    command_id: str,
) -> dict[str, object]:
    if not command_id.strip():
        raise ValueError("dead-letter command ID is required")
    store = HarnessStateStore(state_database_path)
    changed = store.dead_letter_task(
        task_id,
        expected_status=expected_status,
        operator_id=operator_id,
        reason=reason,
        event_key=f"temporal-dead-letter:{command_id}",
    )
    task = store.get_task(task_id)
    artifact_name = next((
        name for name in ("industrial_rag_operation_result", "industrial_rag_ingest_result")
        if store.get_artifact(task_id, name) is not None
    ), None)
    return {
        "changed": changed,
        "status": task["status"] if task else "missing",
        "reconciliation_required": artifact_name is not None,
        "result_artifact_name": artifact_name,
    }


@workflow.defn(name="industrial-rag-ingest-v1")
class IndustrialRagIngestWorkflow:
    def __init__(self) -> None:
        self._state: dict[str, Any] = {
            "status": "pending", "stage": "created", "known": [], "unknown": [],
            "action": None, "evidence": [], "budget": {"provider_calls_max": 0, "provider_calls_used": 0},
            "pending_reconciliation": [], "checkpoint": "workflow_created", "next_step": "ingest_source",
        }
        self._pending_controls: list[TemporalWorkflowControl] = []
        self._control_commands: dict[str, dict[str, str]] = {}
        self._pause_requested = False
        self._cancel_requested = False

    @workflow.signal
    def control(self, command: TemporalWorkflowControl) -> None:
        payload = asdict(command)
        previous = self._control_commands.get(command.command_id)
        if previous is not None:
            if previous != payload:
                self._state = {
                    **self._state,
                    "last_control_error": "command_id was reused with different content",
                }
            return
        self._control_commands[command.command_id] = payload
        self._pending_controls.append(command)
        if command.action == "pause":
            self._pause_requested = True
        elif command.action == "resume":
            self._pause_requested = False
        else:
            self._cancel_requested = True
            self._pause_requested = False
        self._state = {
            **self._state,
            "control_requested": command.action,
            "control_operator_id": command.operator_id,
        }

    async def _flush_controls(self, task_id: str, state_database_path: str) -> None:
        while self._pending_controls:
            command = self._pending_controls.pop(0)
            outcome = await workflow.execute_activity(
                apply_temporal_control_activity,
                TemporalControlActivityInput(task_id, state_database_path, command),
                start_to_close_timeout=timedelta(seconds=30),
                retry_policy=RetryPolicy(
                    initial_interval=timedelta(seconds=1), maximum_interval=timedelta(seconds=5),
                    maximum_attempts=3,
                ),
            )
            history = list(self._state.get("control_history", []))
            history.append({**asdict(command), "accepted": bool(outcome["accepted"]), "status": outcome["status"]})
            self._state = {**self._state, "control_history": history}
            if not outcome["accepted"]:
                if command.action in {"pause", "resume"}:
                    self._pause_requested = False
                if command.action == "cancel":
                    self._cancel_requested = False

    async def _safe_point(self, task_id: str, state_database_path: str) -> bool:
        await self._flush_controls(task_id, state_database_path)
        if self._cancel_requested:
            self._state = {
                **self._state,
                "status": "cancelled",
                "stage": "cancelled_before_ingest",
                "checkpoint": "cancelled_before_activity",
                "next_step": None,
            }
            return False
        if self._pause_requested:
            self._state = {
                **self._state,
                "status": "paused",
                "stage": "paused_before_ingest",
                "checkpoint": "paused_before_activity",
            }
            await workflow.wait_condition(lambda: not self._pause_requested or self._cancel_requested)
            await self._flush_controls(task_id, state_database_path)
            if self._cancel_requested:
                self._state = {
                    **self._state,
                    "status": "cancelled",
                    "stage": "cancelled_before_ingest",
                    "checkpoint": "cancelled_before_activity",
                    "next_step": None,
                }
                return False
        self._state = {**self._state, "status": "running"}
        return True

    @workflow.run
    async def run(self, request: IndustrialRagIngestRequest) -> dict[str, Any]:
        self._state = {
            "task_id": request.task_id,
            "status": "running",
            "stage": "ingesting",
            "known": ["request frozen", "source parsing runs in an activity", "provider calls are disabled"],
            "unknown": [],
            "action": "ingest one local source with immutable provenance",
            "evidence": [],
            "budget": {"provider_calls_max": 0, "provider_calls_used": 0},
            "pending_reconciliation": [],
            "checkpoint": "activity_scheduled",
            "next_step": "persist_ingest_result",
        }
        if not request.is_captured:
            raise ValueError("Temporal ingest request must be captured before workflow scheduling")
        task_request = _ingest_task_request(request)
        activity_input = IndustrialRagIngestActivityInput(request=request)
        heartbeat_seconds = request.activity_heartbeat_timeout_seconds
        if request.pdf_backend == "docling":
            heartbeat_seconds = max(heartbeat_seconds, 360)
        try:
            initialized = await workflow.execute_activity(
                initialize_temporal_task_activity,
                TemporalTaskInitialization(
                    request.task_id,
                    request.idempotency_key,
                    request.state_database_path,
                    task_request,
                    result_artifact="industrial_rag_ingest_result",
                ),
                start_to_close_timeout=timedelta(seconds=30),
                retry_policy=RetryPolicy(
                    initial_interval=timedelta(seconds=1), maximum_interval=timedelta(seconds=5),
                    maximum_attempts=3,
                ),
            )
            if initialized["status"] == "completed":
                self._state = {
                    **self._state,
                    "status": "completed",
                    "stage": "ingest_result_reused",
                    "checkpoint": "stored_result_reused",
                    "pending_reconciliation": [],
                    "next_step": None,
                }
                return initialized["result"]
            if initialized["status"] == "paused":
                if not self._pending_controls:
                    self._pause_requested = True
                artifact_name = initialized.get("result_artifact_name")
                self._state = {
                    **self._state,
                    "status": "paused",
                    "pending_reconciliation": [artifact_name] if isinstance(artifact_name, str) else [],
                }
            elif initialized["status"] != "running":
                artifact_name = initialized.get("result_artifact_name")
                reconciliation_required = bool(initialized.get("reconciliation_required"))
                self._state = {
                    **self._state,
                    "status": initialized["status"],
                    "stage": f"task_{initialized['status']}",
                    "checkpoint": "terminal_task_state_reused",
                    "reconciliation_required": reconciliation_required,
                    "pending_reconciliation": [artifact_name] if isinstance(artifact_name, str) else [],
                    "next_step": "operator_reconcile_committed_side_effect" if reconciliation_required else None,
                }
                result: dict[str, object] = {"status": initialized["status"], "task_id": request.task_id}
                if reconciliation_required:
                    result["reconciliation_required"] = True
                    result["result_artifact_name"] = artifact_name
                return result
            if not await self._safe_point(request.task_id, request.state_database_path):
                return {"status": "cancelled", "task_id": request.task_id}
            result = await workflow.execute_activity(
                ingest_document_activity,
                activity_input,
                start_to_close_timeout=timedelta(minutes=6),
                heartbeat_timeout=timedelta(seconds=heartbeat_seconds),
                retry_policy=RetryPolicy(
                    initial_interval=timedelta(seconds=1),
                    maximum_interval=timedelta(seconds=10),
                    maximum_attempts=3,
                ),
            )
            await self._flush_controls(request.task_id, request.state_database_path)
        except ActivityError as error:
            try:
                failure = await workflow.execute_activity(
                    fail_temporal_task_activity,
                    TemporalFailureActivityInput(
                        request.task_id,
                        request.state_database_path,
                        type(error).__name__,
                        str(error),
                        result_artifact="industrial_rag_ingest_result",
                    ),
                    start_to_close_timeout=timedelta(seconds=30),
                    retry_policy=RetryPolicy(
                        initial_interval=timedelta(seconds=1), maximum_interval=timedelta(seconds=5),
                        maximum_attempts=3,
                    ),
                )
            except ActivityError:
                failure = {"accepted": False, "status": "failure_recording_failed"}
            durable_status = str(failure["status"])
            reconciliation_required = bool(failure.get("reconciliation_required"))
            artifact_name = failure.get("result_artifact_name")
            if durable_status == "completed" and isinstance(failure.get("result"), dict):
                self._state = {
                    **self._state,
                    "status": "completed",
                    "stage": "ingest_committed_before_workflow_failure",
                    "checkpoint": "stored_result_reused",
                    "pending_reconciliation": [],
                    "next_step": None,
                }
                return failure["result"]
            if durable_status in {"cancelled", "dead_letter", "unknown"}:
                self._state = {
                    **self._state,
                    "status": durable_status,
                    "stage": f"ingest_{durable_status}_preserved",
                    "reconciliation_required": reconciliation_required,
                    "pending_reconciliation": [artifact_name] if isinstance(artifact_name, str) else [],
                    "checkpoint": "terminal_task_state_preserved",
                    "next_step": "operator_reconcile_committed_side_effect" if reconciliation_required else None,
                }
                result = {"status": durable_status, "task_id": request.task_id}
                if reconciliation_required:
                    result["reconciliation_required"] = True
                    result["result_artifact_name"] = artifact_name
                return result
            self._state = {
                **self._state,
                "status": "failed",
                "stage": "ingest_failed",
                "unknown": ["activity outcome may require worker-side inspection"],
                "reconciliation_required": reconciliation_required,
                "pending_reconciliation": [artifact_name] if isinstance(artifact_name, str) else [],
                "checkpoint": "failure_recorded" if failure["accepted"] else "activity_failed",
                "next_step": "operator_reconcile_committed_side_effect" if reconciliation_required else "inspect_temporal_history_and_task_state",
            }
            raise
        self._state = {
            **self._state,
            "status": "completed",
            "stage": "ingest_completed",
            "evidence": [
                {"content_sha256": result["content_sha256"], "version_id": result["version_id"]},
            ],
            "checkpoint": "ingest_result_persisted",
            "pending_reconciliation": [],
            "next_step": None,
        }
        return result

    @workflow.query
    def state(self) -> dict[str, Any]:
        return self._state


@workflow.defn(name="industrial-rag-operation-v1")
class IndustrialRagOperationWorkflow:
    """Durable local index/evaluation operations with audited checkpoint controls."""

    def __init__(self) -> None:
        self._state: dict[str, Any] = {
            "status": "pending", "stage": "created", "operation": None,
            "control_history": [], "checkpoint": "workflow_created",
        }
        self._pending_controls: list[TemporalWorkflowControl] = []
        self._control_commands: dict[str, dict[str, str]] = {}
        self._pause_requested = False
        self._cancel_requested = False

    @workflow.signal
    def control(self, command: TemporalWorkflowControl) -> None:
        payload = asdict(command)
        previous = self._control_commands.get(command.command_id)
        if previous is not None:
            if previous != payload:
                self._state = {
                    **self._state,
                    "last_control_error": "command_id was reused with different content",
                }
            return
        self._control_commands[command.command_id] = payload
        self._pending_controls.append(command)
        if command.action == "pause":
            self._pause_requested = True
        elif command.action == "resume":
            self._pause_requested = False
        else:
            self._cancel_requested = True
            self._pause_requested = False
        self._state = {
            **self._state,
            "control_requested": command.action,
            "control_operator_id": command.operator_id,
        }

    @workflow.query
    def state(self) -> dict[str, Any]:
        return dict(self._state)

    async def _flush_controls(self, task_id: str, state_database_path: str) -> None:
        while self._pending_controls:
            command = self._pending_controls.pop(0)
            outcome = await workflow.execute_activity(
                apply_temporal_control_activity,
                TemporalControlActivityInput(task_id, state_database_path, command),
                start_to_close_timeout=timedelta(seconds=30),
                retry_policy=RetryPolicy(
                    initial_interval=timedelta(seconds=1), maximum_interval=timedelta(seconds=5),
                    maximum_attempts=3,
                ),
            )
            history = list(self._state.get("control_history", []))
            history.append({**asdict(command), "accepted": bool(outcome["accepted"]), "status": outcome["status"]})
            self._state = {**self._state, "control_history": history}
            if not outcome["accepted"]:
                if command.action in {"pause", "resume"}:
                    self._pause_requested = False
                if command.action == "cancel":
                    self._cancel_requested = False

    async def _safe_point(self, task_id: str, state_database_path: str) -> bool:
        await self._flush_controls(task_id, state_database_path)
        if self._cancel_requested:
            self._state = {
                **self._state,
                "status": "cancelled",
                "stage": "cancelled_at_checkpoint",
                "checkpoint": "cancelled_before_next_activity",
                "next_step": None,
            }
            return False
        if self._pause_requested:
            self._state = {
                **self._state,
                "status": "paused",
                "stage": "paused_at_checkpoint",
                "checkpoint": "paused_before_next_activity",
            }
            await workflow.wait_condition(lambda: not self._pause_requested or self._cancel_requested)
            await self._flush_controls(task_id, state_database_path)
            if self._cancel_requested:
                self._state = {
                    **self._state,
                    "status": "cancelled",
                    "stage": "cancelled_at_checkpoint",
                    "checkpoint": "cancelled_before_next_activity",
                    "next_step": None,
                }
                return False
        self._state = {**self._state, "status": "running"}
        return True

    async def _mark_failed(
        self, task_id: str, state_database_path: str, error: BaseException,
    ) -> dict[str, object]:
        try:
            outcome = await workflow.execute_activity(
                fail_temporal_task_activity,
                TemporalFailureActivityInput(
                    task_id, state_database_path, type(error).__name__, str(error),
                ),
                start_to_close_timeout=timedelta(seconds=30),
                retry_policy=RetryPolicy(
                    initial_interval=timedelta(seconds=1), maximum_interval=timedelta(seconds=5),
                    maximum_attempts=3,
                ),
            )
        except ActivityError:
            outcome = {"accepted": False, "status": "failure_recording_failed"}
        durable_status = str(outcome["status"])
        reconciliation_required = bool(outcome.get("reconciliation_required"))
        artifact_name = outcome.get("result_artifact_name")
        visible_status = (
            durable_status
            if durable_status in {"completed", "cancelled", "dead_letter", "unknown"}
            else "failed"
        )
        self._state = {
            **self._state,
            "status": visible_status,
            "stage": (
                "operation_committed_before_workflow_failure"
                if durable_status == "completed" else
                f"operation_{durable_status}_preserved"
                if durable_status in {"cancelled", "dead_letter", "unknown"} else
                "operation_failed"
            ),
            "reconciliation_required": reconciliation_required,
            "pending_reconciliation": [artifact_name] if isinstance(artifact_name, str) else [],
            "unknown": (
                ["inspect the durable task artifact when activity outcome is ambiguous"]
                if reconciliation_required else []
            ),
            "checkpoint": "operation_failure_recorded" if outcome["accepted"] else "operation_failure_not_recorded",
            "next_step": (
                "operator_reconcile_committed_side_effect"
                if reconciliation_required else
                "inspect_temporal_history_and_task_state"
                if visible_status == "failed" else None
            ),
        }
        return outcome

    @workflow.run
    async def run(self, operation: IndustrialRagOperationRequest) -> dict[str, object]:
        if operation.operation == "index":
            request = IndustrialRagIndexRequest(**{
                **operation.payload,
                "version_ids": tuple(operation.payload.get("version_ids", ())),
            })
            task_request = _operation_identity("industrial_rag.index.v1", request)
            task_id, state_database_path = request.task_id, request.state_database_path
            activity_input: (
                IndustrialRagIndexActivityInput
                | IndustrialRagEvaluationActivityInput
                | IndustrialRagResearchActivityInput
            ) = (
                IndustrialRagIndexActivityInput(request)
            )
        elif operation.operation == "evaluate":
            request = IndustrialRagEvaluationRequest(**operation.payload)
            if not request.is_captured:
                raise ValueError("Temporal evaluation request must be captured before scheduling")
            task_request = _operation_identity("industrial_rag.evaluate.v1", request)
            task_id, state_database_path = request.task_id, request.state_database_path
            activity_input = IndustrialRagEvaluationActivityInput(request)
        elif operation.operation == "research":
            request = IndustrialRagResearchRequest(**operation.payload)
            if not request.is_captured:
                raise ValueError("Temporal research bundle must be captured before scheduling")
            task_request = _operation_identity("financial_research.cached_replay.v1", request)
            task_id, state_database_path = request.task_id, request.state_database_path
            activity_input = IndustrialRagResearchActivityInput(request)
        else:
            raise ValueError("unsupported Temporal RAG operation")

        self._state = {
            **self._state,
            "task_id": task_id,
            "operation": operation.operation,
            "status": "running",
            "stage": "initializing",
            "known": ["operation request frozen", "provider calls are disabled"],
            "unknown": [],
            "budget": {"provider_calls_max": 0, "provider_calls_used": 0},
            "checkpoint": "task_state_opening",
            "next_step": "verify_operation_inputs",
        }
        try:
            initialized = await workflow.execute_activity(
                initialize_temporal_task_activity,
                TemporalTaskInitialization(
                    task_id, request.idempotency_key, state_database_path, task_request,
                ),
                start_to_close_timeout=timedelta(seconds=30),
                retry_policy=RetryPolicy(
                    initial_interval=timedelta(seconds=1), maximum_interval=timedelta(seconds=5),
                    maximum_attempts=3,
                ),
            )
            if initialized["status"] == "completed":
                self._state = {
                    **self._state,
                    "status": "completed",
                    "stage": "operation_reused",
                    "checkpoint": "stored_result_reused",
                    "pending_reconciliation": [],
                    "next_step": None,
                }
                return initialized["result"]
            if initialized["status"] == "paused":
                if not self._pending_controls:
                    self._pause_requested = True
                artifact_name = initialized.get("result_artifact_name")
                self._state = {
                    **self._state,
                    "status": "paused",
                    "stage": "restored_paused_checkpoint",
                    "reconciliation_required": bool(initialized.get("reconciliation_required")),
                    "pending_reconciliation": [artifact_name] if isinstance(artifact_name, str) else [],
                }
            elif initialized["status"] != "running":
                artifact_name = initialized.get("result_artifact_name")
                reconciliation_required = bool(initialized.get("reconciliation_required"))
                self._state = {
                    **self._state,
                    "status": initialized["status"],
                    "stage": f"task_{initialized['status']}",
                    "checkpoint": "terminal_task_state_reused",
                    "reconciliation_required": reconciliation_required,
                    "pending_reconciliation": [artifact_name] if isinstance(artifact_name, str) else [],
                    "next_step": "operator_reconcile_committed_side_effect" if reconciliation_required else None,
                }
                result: dict[str, object] = {"status": initialized["status"], "task_id": task_id}
                if reconciliation_required:
                    result["reconciliation_required"] = True
                    result["result_artifact_name"] = artifact_name
                return result

            if not await self._safe_point(task_id, state_database_path):
                return {"status": "cancelled", "task_id": task_id}

            if operation.operation == "index":
                self._state = {**self._state, "stage": "verifying_index_inputs", "checkpoint": "index_preflight_started"}
                prepared = await workflow.execute_activity(
                    verify_index_inputs_activity,
                    activity_input,
                    start_to_close_timeout=timedelta(minutes=2),
                    heartbeat_timeout=timedelta(seconds=60),
                    retry_policy=RetryPolicy(
                        initial_interval=timedelta(seconds=1), maximum_interval=timedelta(seconds=10),
                        maximum_attempts=3,
                    ),
                )
                self._state = {
                    **self._state,
                    "stage": "index_inputs_verified",
                    "evidence": [prepared],
                    "checkpoint": "index_preflight_completed",
                    "next_step": "build_release_index",
                }
            elif operation.operation == "evaluate":
                self._state = {**self._state, "stage": "verifying_evaluation_inputs", "checkpoint": "evaluation_preflight_started"}
                prepared = await workflow.execute_activity(
                    verify_evaluation_inputs_activity,
                    activity_input,
                    start_to_close_timeout=timedelta(minutes=2),
                    heartbeat_timeout=timedelta(seconds=60),
                    retry_policy=RetryPolicy(
                        initial_interval=timedelta(seconds=1), maximum_interval=timedelta(seconds=10),
                        maximum_attempts=3,
                    ),
                )
                self._state = {
                    **self._state,
                    "stage": "evaluation_inputs_verified",
                    "evidence": [prepared],
                    "checkpoint": "evaluation_preflight_completed",
                    "next_step": "score_frozen_predictions",
                }
            else:
                self._state = {**self._state, "stage": "verifying_research_inputs", "checkpoint": "research_preflight_started"}
                prepared = await workflow.execute_activity(
                    verify_research_inputs_activity,
                    activity_input,
                    start_to_close_timeout=timedelta(minutes=2),
                    heartbeat_timeout=timedelta(seconds=60),
                    retry_policy=RetryPolicy(
                        initial_interval=timedelta(seconds=1), maximum_interval=timedelta(seconds=10),
                        maximum_attempts=3,
                    ),
                )
                self._state = {
                    **self._state,
                    "stage": "research_inputs_verified",
                    "evidence": [prepared],
                    "checkpoint": "research_preflight_completed",
                    "next_step": "run_offline_multi_expert_harness",
                }
            if not await self._safe_point(task_id, state_database_path):
                return {"status": "cancelled", "task_id": task_id, "preflight": prepared}

            self._state = {
                **self._state,
                "stage": "running_operation",
                "checkpoint": "operation_activity_started",
            }
            activity_fn = {
                "index": build_index_activity,
                "evaluate": evaluate_manifest_activity,
                "research": run_research_activity,
            }[operation.operation]
            result = await workflow.execute_activity(
                activity_fn,
                activity_input,
                start_to_close_timeout=timedelta(minutes=6),
                heartbeat_timeout=timedelta(seconds=60),
                retry_policy=RetryPolicy(
                    initial_interval=timedelta(seconds=1), maximum_interval=timedelta(seconds=10),
                    maximum_attempts=3,
                ),
            )
            await self._flush_controls(task_id, state_database_path)
            self._state = {
                **self._state,
                "status": "completed",
                "stage": "operation_completed",
                "evidence": [{"result_status": result.get("status"), "task_id": task_id}],
                "checkpoint": "operation_result_persisted",
                "pending_reconciliation": [],
                "next_step": None,
            }
            return result
        except ActivityError as error:
            outcome = await self._mark_failed(task_id, state_database_path, error)
            durable_status = str(outcome["status"])
            if durable_status == "completed" and isinstance(outcome.get("result"), dict):
                return outcome["result"]
            if durable_status in {"cancelled", "dead_letter", "unknown"}:
                result = {"status": durable_status, "task_id": task_id}
                if outcome.get("reconciliation_required"):
                    result["reconciliation_required"] = True
                    result["result_artifact_name"] = outcome.get("result_artifact_name")
                return result
            raise


async def _run_worker(address: str, task_queue: str) -> None:
    client = await Client.connect(address)
    activity_executor = ThreadPoolExecutor(max_workers=4)
    worker = Worker(
        client,
        task_queue=task_queue,
        workflows=[IndustrialRagIngestWorkflow, IndustrialRagOperationWorkflow],
        activities=[
            ingest_document_activity,
            initialize_temporal_task_activity,
            verify_index_inputs_activity,
            build_index_activity,
            verify_evaluation_inputs_activity,
            evaluate_manifest_activity,
            verify_research_inputs_activity,
            run_research_activity,
            apply_temporal_control_activity,
            fail_temporal_task_activity,
        ],
        activity_executor=activity_executor,
    )
    try:
        await worker.run()
    finally:
        activity_executor.shutdown(wait=True)


def _load_request(path: str | Path) -> IndustrialRagIngestRequest:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Temporal ingest request file must contain one JSON object")
    return IndustrialRagIngestRequest(**value)


def _load_index_request(path: str | Path) -> IndustrialRagIndexRequest:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Temporal index request file must contain one JSON object")
    version_ids = value.get("version_ids")
    if not isinstance(version_ids, (list, tuple)):
        raise ValueError("Temporal index request version_ids must be an array")
    return IndustrialRagIndexRequest(**{**value, "version_ids": tuple(version_ids)})


def _load_evaluation_request(path: str | Path) -> IndustrialRagEvaluationRequest:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Temporal evaluation request file must contain one JSON object")
    return IndustrialRagEvaluationRequest(**value)


def _load_research_request(path: str | Path) -> IndustrialRagResearchRequest:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Temporal research request file must contain one JSON object")
    return IndustrialRagResearchRequest(**value)


async def _submit(address: str, task_queue: str, request: IndustrialRagIngestRequest) -> dict[str, object]:
    request = capture_ingest_request(request)
    client = await Client.connect(address)
    handle = await client.start_workflow(
        IndustrialRagIngestWorkflow.run,
        request,
        id=request.task_id,
        task_queue=task_queue,
    )
    return {"workflow_id": handle.id, "run_id": handle.result_run_id}


async def _submit_operation(
    address: str,
    task_queue: str,
    request: IndustrialRagIndexRequest | IndustrialRagEvaluationRequest | IndustrialRagResearchRequest,
) -> dict[str, object]:
    if isinstance(request, IndustrialRagEvaluationRequest):
        request = capture_evaluation_request(request)
        operation = IndustrialRagOperationRequest("evaluate", asdict(request))
    elif isinstance(request, IndustrialRagResearchRequest):
        request = capture_research_request(request)
        operation = IndustrialRagOperationRequest("research", asdict(request))
    else:
        operation = IndustrialRagOperationRequest("index", asdict(request))
    client = await Client.connect(address)
    handle = await client.start_workflow(
        IndustrialRagOperationWorkflow.run,
        operation,
        id=str(operation.payload["task_id"]),
        task_queue=task_queue,
    )
    result: dict[str, object] = {"workflow_id": handle.id, "run_id": handle.result_run_id}
    if operation.operation == "evaluate":
        result["manifest_sha256"] = operation.payload["manifest_sha256"]
        result["predictions_sha256"] = operation.payload["predictions_sha256"]
    elif operation.operation == "research":
        result["bundle_sha256"] = operation.payload["bundle_sha256"]
        result["provider_calls_max"] = 0
    return result


async def _query(address: str, workflow_id: str, *, workflow_kind: str = "ingest") -> dict[str, object]:
    client = await Client.connect(address)
    handle = client.get_workflow_handle(workflow_id)
    query = (
        IndustrialRagOperationWorkflow.state
        if workflow_kind == "operation" else IndustrialRagIngestWorkflow.state
    )
    return await handle.query(query)


async def _control_operation(
    address: str,
    workflow_id: str,
    command: TemporalWorkflowControl,
    *,
    workflow_kind: str = "operation",
) -> dict[str, object]:
    client = await Client.connect(address)
    handle = client.get_workflow_handle(workflow_id)
    signal = IndustrialRagIngestWorkflow.control if workflow_kind == "ingest" else IndustrialRagOperationWorkflow.control
    await handle.signal(signal, command)
    return {"workflow_id": workflow_id, "command_id": command.command_id, "action": command.action}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    worker_parser = subparsers.add_parser("worker", help="run the Industrial RAG Temporal worker")
    worker_parser.add_argument("--address", default=os.environ.get("TEMPORAL_ADDRESS", "localhost:7233"))
    worker_parser.add_argument("--task-queue", default=TASK_QUEUE)
    submit_parser = subparsers.add_parser("submit", help="start a durable local-source ingest workflow")
    submit_parser.add_argument("--address", default=os.environ.get("TEMPORAL_ADDRESS", "localhost:7233"))
    submit_parser.add_argument("--task-queue", default=TASK_QUEUE)
    submit_parser.add_argument("--request", type=Path, required=True)
    index_parser = subparsers.add_parser("submit-index", help="start an immutable local corpus index and release workflow")
    index_parser.add_argument("--address", default=os.environ.get("TEMPORAL_ADDRESS", "localhost:7233"))
    index_parser.add_argument("--task-queue", default=TASK_QUEUE)
    index_parser.add_argument("--request", type=Path, required=True)
    evaluation_parser = subparsers.add_parser("submit-evaluate", help="capture and score frozen evaluation inputs offline")
    evaluation_parser.add_argument("--address", default=os.environ.get("TEMPORAL_ADDRESS", "localhost:7233"))
    evaluation_parser.add_argument("--task-queue", default=TASK_QUEUE)
    evaluation_parser.add_argument("--request", type=Path, required=True)
    research_parser = subparsers.add_parser("submit-research", help="run the multi-expert manager harness from exact local response caches only")
    research_parser.add_argument("--address", default=os.environ.get("TEMPORAL_ADDRESS", "localhost:7233"))
    research_parser.add_argument("--task-queue", default=TASK_QUEUE)
    research_parser.add_argument("--request", type=Path, required=True)
    query_parser = subparsers.add_parser("query", help="read the current workflow state")
    query_parser.add_argument("--address", default=os.environ.get("TEMPORAL_ADDRESS", "localhost:7233"))
    query_parser.add_argument("--workflow-id", required=True)
    query_parser.add_argument("--workflow-kind", choices=["ingest", "operation"], default="ingest")
    control_parser = subparsers.add_parser("control", help="pause, resume, or cancel a workflow at an activity checkpoint")
    control_parser.add_argument("--address", default=os.environ.get("TEMPORAL_ADDRESS", "localhost:7233"))
    control_parser.add_argument("--workflow-id", required=True)
    control_parser.add_argument("--workflow-kind", choices=["ingest", "operation"], default="operation")
    control_parser.add_argument("--action", choices=["pause", "resume", "cancel"], required=True)
    control_parser.add_argument("--operator-id", required=True)
    control_parser.add_argument("--reason", required=True)
    control_parser.add_argument("--command-id", default=None)
    dead_letter_parser = subparsers.add_parser("dead-letter", help="auditedly close a failed or unknown durable operation")
    dead_letter_parser.add_argument("--state-database", type=Path, required=True)
    dead_letter_parser.add_argument("--task-id", required=True)
    dead_letter_parser.add_argument("--expected-status", choices=["failed", "unknown"], required=True)
    dead_letter_parser.add_argument("--operator-id", required=True)
    dead_letter_parser.add_argument("--reason", required=True)
    dead_letter_parser.add_argument("--command-id", default=None)
    args = parser.parse_args()
    if args.command == "worker":
        asyncio.run(_run_worker(args.address, args.task_queue))
    elif args.command == "submit":
        print(json.dumps(asyncio.run(_submit(args.address, args.task_queue, _load_request(args.request))), indent=2))
    elif args.command == "submit-index":
        print(json.dumps(asyncio.run(_submit_operation(args.address, args.task_queue, _load_index_request(args.request))), indent=2))
    elif args.command == "submit-evaluate":
        print(json.dumps(asyncio.run(_submit_operation(args.address, args.task_queue, _load_evaluation_request(args.request))), indent=2))
    elif args.command == "submit-research":
        print(json.dumps(asyncio.run(_submit_operation(args.address, args.task_queue, _load_research_request(args.request))), indent=2))
    elif args.command == "control":
        command = TemporalWorkflowControl(
            command_id=args.command_id or uuid.uuid4().hex,
            action=args.action,
            operator_id=args.operator_id,
            reason=args.reason,
        )
        result = asyncio.run(_control_operation(
            args.address, args.workflow_id, command, workflow_kind=args.workflow_kind,
        ))
        print(json.dumps(result, ensure_ascii=False, indent=2))
    elif args.command == "dead-letter":
        result = dead_letter_temporal_task(
            args.state_database,
            args.task_id,
            expected_status=args.expected_status,
            operator_id=args.operator_id,
            reason=args.reason,
            command_id=args.command_id or uuid.uuid4().hex,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(json.dumps(
            asyncio.run(_query(args.address, args.workflow_id, workflow_kind=args.workflow_kind)),
            ensure_ascii=False, indent=2,
        ))


if __name__ == "__main__":
    main()
