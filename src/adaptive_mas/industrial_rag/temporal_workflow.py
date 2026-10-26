"""Temporal worker and submitter for durable Industrial RAG source ingestion."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from temporalio import activity, workflow
from temporalio.client import Client
from temporalio.common import RetryPolicy
from temporalio.exceptions import ActivityError
from temporalio.worker import Worker

with workflow.unsafe.imports_passed_through():
    from adaptive_mas.harness_state import HarnessStateStore, StateConflict
    from adaptive_mas.industrial_rag.cli import ingest_file


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
    _write_capture_file(source_capture, source_bytes, source_sha256)
    _write_capture_file(metadata_capture, metadata_bytes, metadata_sha256)
    return replace(
        request,
        source_path=str(source_capture),
        metadata_path=str(metadata_capture),
        source_sha256=source_sha256,
        metadata_sha256=metadata_sha256,
        captured_at=datetime.now(timezone.utc).isoformat(),
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


@dataclass(frozen=True)
class IndustrialRagIngestActivityInput:
    request: IndustrialRagIngestRequest


@activity.defn
def ingest_document_activity(value: IndustrialRagIngestActivityInput) -> dict[str, Any]:
    request = value.request
    if not request.is_captured:
        raise ValueError("Temporal ingest activity requires source and metadata captured before scheduling")
    state_store = HarnessStateStore(request.state_database_path)
    task_request = {
        "operation": "industrial_rag.ingest.v1",
        "request": asdict(request),
        "captured_at": request.captured_at,
        "source_sha256": request.source_sha256,
        "metadata_sha256": request.metadata_sha256,
    }
    task = state_store.open_task(
        request.task_id, request.idempotency_key, task_request, max_provider_calls=0,
    )
    if task["status"] == "completed":
        result = task["result"]
        if not isinstance(result, dict):
            raise RuntimeError("completed Temporal ingest task has no persisted result")
        return result

    saved_result = state_store.get_artifact(request.task_id, "industrial_rag_ingest_result")
    if saved_result is not None:
        state_store.complete_task(request.task_id, saved_result)
        return saved_result

    source_bytes, metadata_bytes = read_captured_input_bytes(request)

    state_store.checkpoint_task(
        request.task_id, "ingest_started", {"operation": "industrial_rag.ingest.v1"},
        event_key="industrial_rag_ingest_started",
    )
    activity.heartbeat({"stage": "ingesting"})

    def freeze_ingested_at() -> str:
        artifact_name = "industrial_rag_ingest_identity"
        identity = state_store.get_artifact(request.task_id, artifact_name)
        if identity is None:
            proposed = {
                "source_sha256": request.source_sha256,
                "metadata_sha256": request.metadata_sha256,
                "captured_at": request.captured_at,
                "ingested_at": datetime.now(timezone.utc).isoformat(),
            }
            try:
                state_store.put_artifact(request.task_id, artifact_name, proposed)
            except StateConflict:
                # Concurrent activity retries may race to establish the same immutable identity.
                pass
            identity = state_store.get_artifact(request.task_id, artifact_name)
        if (
            identity is None
            or identity.get("source_sha256") != request.source_sha256
            or identity.get("metadata_sha256") != request.metadata_sha256
            or identity.get("captured_at") != request.captured_at
            or not isinstance(identity.get("ingested_at"), str)
        ):
            raise RuntimeError("Temporal ingest retry found a conflicting durable ingest identity")
        return str(identity["ingested_at"])

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
        ingested_at_factory=freeze_ingested_at,
    )
    state_store.put_artifact(request.task_id, "industrial_rag_ingest_result", result)
    state_store.checkpoint_task(
        request.task_id, "ingest_persisted",
        {"content_sha256": result["content_sha256"], "status": result["status"]},
        event_key="industrial_rag_ingest_persisted",
    )
    state_store.complete_task(request.task_id, result)
    activity.heartbeat({"stage": "completed", "content_sha256": result["content_sha256"]})
    return result


@workflow.defn(name="industrial-rag-ingest-v1")
class IndustrialRagIngestWorkflow:
    def __init__(self) -> None:
        self._state: dict[str, Any] = {
            "status": "pending", "stage": "created", "known": [], "unknown": [],
            "action": None, "evidence": [], "budget": {"provider_calls_max": 0, "provider_calls_used": 0},
            "pending_reconciliation": [], "checkpoint": "workflow_created", "next_step": "ingest_source",
        }

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
        activity_input = IndustrialRagIngestActivityInput(request=request)
        heartbeat_seconds = request.activity_heartbeat_timeout_seconds
        if request.pdf_backend == "docling":
            heartbeat_seconds = max(heartbeat_seconds, 360)
        try:
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
        except ActivityError:
            self._state = {
                **self._state,
                "status": "failed",
                "stage": "ingest_failed",
                "unknown": ["activity outcome may require worker-side inspection"],
                "checkpoint": "activity_failed",
                "next_step": "inspect_temporal_history_and_task_state",
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
            "next_step": None,
        }
        return result

    @workflow.query
    def state(self) -> dict[str, Any]:
        return self._state


async def _run_worker(address: str, task_queue: str) -> None:
    client = await Client.connect(address)
    activity_executor = ThreadPoolExecutor(max_workers=4)
    worker = Worker(
        client,
        task_queue=task_queue,
        workflows=[IndustrialRagIngestWorkflow],
        activities=[ingest_document_activity],
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


async def _query(address: str, workflow_id: str) -> dict[str, object]:
    client = await Client.connect(address)
    handle = client.get_workflow_handle(workflow_id)
    return await handle.query(IndustrialRagIngestWorkflow.state)


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
    query_parser = subparsers.add_parser("query", help="read the current workflow state")
    query_parser.add_argument("--address", default=os.environ.get("TEMPORAL_ADDRESS", "localhost:7233"))
    query_parser.add_argument("--workflow-id", required=True)
    args = parser.parse_args()
    if args.command == "worker":
        asyncio.run(_run_worker(args.address, args.task_queue))
    elif args.command == "submit":
        print(json.dumps(asyncio.run(_submit(args.address, args.task_queue, _load_request(args.request))), indent=2))
    else:
        print(json.dumps(asyncio.run(_query(args.address, args.workflow_id)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
