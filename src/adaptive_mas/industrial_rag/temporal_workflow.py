"""Temporal worker and submitter for durable Industrial RAG source ingestion."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any

from temporalio import activity, workflow
from temporalio.client import Client
from temporalio.common import RetryPolicy
from temporalio.exceptions import ActivityError
from temporalio.worker import Worker

with workflow.unsafe.imports_passed_through():
    from adaptive_mas.harness_state import HarnessStateStore
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


@dataclass(frozen=True)
class IndustrialRagIngestActivityInput:
    request: IndustrialRagIngestRequest
    captured_at: str
    ingested_at: str


@activity.defn
def ingest_document_activity(value: IndustrialRagIngestActivityInput) -> dict[str, Any]:
    request = value.request
    state_store = HarnessStateStore(request.state_database_path)
    task_request = {
        "operation": "industrial_rag.ingest.v1",
        "request": asdict(request),
        "captured_at": value.captured_at,
        "ingested_at": value.ingested_at,
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
        captured_at=value.captured_at,
        ingested_at=value.ingested_at,
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
        now = workflow.now().isoformat()
        activity_input = IndustrialRagIngestActivityInput(
            request=request, captured_at=now, ingested_at=now,
        )
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
