"""Prove a completed Industrial RAG ingest survives a Temporal worker crash before ack."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sqlite3
import subprocess
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from adaptive_mas.harness_state import HarnessStateStore  # noqa: E402
from adaptive_mas.industrial_rag.temporal_workflow import (  # noqa: E402
    IndustrialRagIngestRequest,
    IndustrialRagIngestActivityInput,
    IndustrialRagIngestWorkflow,
    capture_ingest_request,
    ingest_document_activity,
    read_captured_input_bytes,
)


async def _worker(args: argparse.Namespace) -> None:
    from temporalio import activity
    from temporalio.client import Client
    from temporalio.worker import Worker

    @activity.defn(name="ingest_document_activity")
    def tracked_ingest(value: IndustrialRagIngestActivityInput) -> dict[str, Any]:
        with args.attempt_log.open("a", encoding="utf-8") as stream:
            stream.write(f"{os.getpid()}\n")
        result = ingest_document_activity(value)
        if args.crash_after_completion:
            os._exit(86)
        return result

    client = await Client.connect(args.address)
    activity_executor = ThreadPoolExecutor(max_workers=2)
    worker = Worker(
        client,
        task_queue=args.task_queue,
        workflows=[IndustrialRagIngestWorkflow],
        activities=[tracked_ingest],
        activity_executor=activity_executor,
    )
    args.ready_file.write_text("ready\n", encoding="utf-8")
    await worker.run()


def _start_worker(
    address: str,
    task_queue: str,
    state_dir: Path,
    attempt_log: Path,
    label: str,
    *,
    crash_after_completion: bool,
) -> subprocess.Popen[bytes]:
    ready_file = state_dir / f"{label}.ready"
    log_path = state_dir / f"{label}.log"
    command = [
        sys.executable, str(Path(__file__).resolve()),
        "--worker", "--address", address, "--task-queue", task_queue,
        "--ready-file", str(ready_file), "--attempt-log", str(attempt_log),
    ]
    if crash_after_completion:
        command.append("--crash-after-completion")
    log_file = log_path.open("wb")
    process = subprocess.Popen(
        command, cwd=ROOT, stdout=log_file, stderr=subprocess.STDOUT,
    )
    process._raama_log_file = log_file  # type: ignore[attr-defined]
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        if ready_file.is_file():
            return process
        if process.poll() is not None:
            log_file.flush()
            detail = log_path.read_text(encoding="utf-8", errors="replace")[-1200:]
            raise RuntimeError(f"Temporal smoke worker {label} exited before polling: {detail}")
        time.sleep(0.1)
    process.terminate()
    process.wait(timeout=5)
    raise RuntimeError(f"Temporal smoke worker {label} did not become ready")


def _stop_worker(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is None:
        process.terminate()
        process.wait(timeout=10)
    log_file = getattr(process, "_raama_log_file", None)
    if log_file is not None:
        log_file.close()


async def _run_smoke(address: str, task_queue: str, state_dir: Path) -> dict[str, object]:
    from temporalio.client import Client

    state_dir.mkdir(parents=True, exist_ok=False)
    run_id = uuid.uuid4().hex[:12]
    source_path = state_dir / "synthetic-source.txt"
    metadata_path = state_dir / "synthetic-metadata.json"
    state_database_path = state_dir / "task-state.sqlite3"
    attempt_log = state_dir / "activity-attempts.txt"
    original_source_bytes = (
        "Synthetic local report. Revenue was 100 million yuan in 2026 Q3. "
        "This text is only for the Temporal recovery smoke test.\n"
    ).encode("utf-8")
    source_path.write_bytes(original_source_bytes)
    original_metadata = {
        "document_id": f"temporal-smoke-document-{run_id}",
        "version_id": f"temporal-smoke-version-{run_id}",
        "source_id": f"temporal-smoke-source-{run_id}",
        "title": "Synthetic Temporal workflow record",
        "document_type": "synthetic_report",
        "security_code": "000001",
        "entity_name": "Synthetic issuer",
        "report_period": "2026Q3",
        "accounting_basis": "consolidated",
        "source_url": "https://example.invalid/temporal-smoke",
        "rights_note": "Synthetic fixture only",
        "published_at": "2026-01-01T00:00:00+00:00",
        "first_available_at": "2026-01-01T00:00:00+00:00",
        "time_basis": "strict_forward",
        "read_principals": ["local-public"],
    }
    original_metadata_bytes = json.dumps(
        original_metadata, ensure_ascii=False, sort_keys=True, indent=2,
    ).encode("utf-8")
    metadata_path.write_bytes(original_metadata_bytes)

    request = IndustrialRagIngestRequest(
        task_id=f"temporal-ingest-{run_id}",
        idempotency_key=f"temporal-ingest-{run_id}",
        source_path=str(source_path),
        metadata_path=str(metadata_path),
        database_path=str(state_dir / "rag.sqlite3"),
        object_root=str(state_dir / "objects"),
        scratch_root=str(state_dir / "parser-scratch"),
        state_database_path=str(state_database_path),
        activity_heartbeat_timeout_seconds=3,
    )
    request = capture_ingest_request(request)
    # Prove the scheduled workflow consumes the captured bytes after the original paths drift.
    source_path.write_text(
        "Synthetic local report. Revenue was 900 million yuan in 2026 Q3. Changed after scheduling.\n",
        encoding="utf-8",
    )
    changed_metadata = dict(original_metadata)
    changed_metadata["document_id"] = f"changed-after-capture-document-{run_id}"
    changed_metadata["version_id"] = f"changed-after-capture-version-{run_id}"
    metadata_path.write_text(
        json.dumps(changed_metadata, ensure_ascii=False, sort_keys=True, indent=2),
        encoding="utf-8",
    )
    client = await Client.connect(address)
    first_worker: subprocess.Popen[bytes] | None = None
    second_worker: subprocess.Popen[bytes] | None = None
    try:
        first_worker = _start_worker(
            address, task_queue, state_dir, attempt_log, "worker-before-crash",
            crash_after_completion=True,
        )
        workflow_handle = await client.start_workflow(
            IndustrialRagIngestWorkflow.run,
            request,
            id=request.task_id,
            task_queue=task_queue,
        )
        exit_code = await asyncio.to_thread(first_worker.wait, 25)
        if exit_code != 86:
            raise AssertionError(f"first Temporal worker did not exit after durable ingest (exit={exit_code})")

        durable = HarnessStateStore(state_database_path)
        task = durable.get_task(request.task_id)
        if task is None or task["status"] != "completed":
            raise AssertionError("ingest activity did not durably complete before the simulated lost ack")

        second_worker = _start_worker(
            address, task_queue, state_dir, attempt_log, "worker-after-restart",
            crash_after_completion=False,
        )
        result = await asyncio.wait_for(workflow_handle.result(), timeout=35)
        state = await workflow_handle.query(IndustrialRagIngestWorkflow.state)
        attempts = attempt_log.read_text(encoding="utf-8").splitlines()
        events = durable.list_task_events(request.task_id)
        artifact = durable.get_artifact(request.task_id, "industrial_rag_ingest_result")
        identity = durable.get_artifact(request.task_id, "industrial_rag_ingest_identity")
        if (
            len(attempts) < 2
            or not isinstance(result, dict)
            or result.get("status") not in {"ingested", "reused"}
            or artifact != result
            or identity is None
            or result.get("content_sha256") != request.source_sha256
            or result.get("version_id") != original_metadata["version_id"]
            or result.get("document_id") != original_metadata["document_id"]
            or identity.get("ingested_at", "") < str(request.captured_at)
            or identity.get("source_sha256") != request.source_sha256
            or identity.get("metadata_sha256") != request.metadata_sha256
            or state.get("status") != "completed"
            or state.get("budget") != {"provider_calls_max": 0, "provider_calls_used": 0}
            or len([event for event in events if event["event_key"] == "industrial_rag_ingest_persisted"]) != 1
        ):
            raise AssertionError("Temporal retry did not resume from the durable ingest artifact")
        with sqlite3.connect(request.database_path) as connection:
            persisted_version = connection.execute(
                "SELECT content_sha256,captured_at,ingested_at FROM rag_document_versions WHERE version_id=?",
                (original_metadata["version_id"],),
            ).fetchone()
        if (
            persisted_version is None
            or persisted_version[0] != request.source_sha256
            or persisted_version[1] != request.captured_at
            or persisted_version[2] != identity["ingested_at"]
        ):
            raise AssertionError("Temporal ingest did not persist the captured source and durable ingest identity")
        captured_source_bytes, captured_metadata_bytes = read_captured_input_bytes(request)
        if captured_source_bytes != original_source_bytes or captured_metadata_bytes != original_metadata_bytes:
            raise AssertionError("source or metadata path drift changed the immutable Temporal capture")

        source_capture = Path(request.source_path)
        source_capture.write_bytes(captured_source_bytes + b"tampered")
        try:
            read_captured_input_bytes(request)
        except ValueError as error:
            if "source bytes" not in str(error):
                raise
        else:
            raise AssertionError("Temporal activity input accepted a changed captured source file")
        finally:
            source_capture.write_bytes(captured_source_bytes)

        metadata_capture = Path(request.metadata_path)
        metadata_capture.write_bytes(captured_metadata_bytes + b" ")
        try:
            read_captured_input_bytes(request)
        except ValueError as error:
            if "metadata bytes" not in str(error):
                raise
        else:
            raise AssertionError("Temporal activity input accepted changed captured metadata")
        finally:
            metadata_capture.write_bytes(captured_metadata_bytes)
        return {
            "schema_version": "raama.financial-rag.temporal-ingest-recovery-result.v1",
            "workflow_id": request.task_id,
            "worker_crashed_after_activity_side_effect": True,
            "activity_attempt_count": len(attempts),
            "durable_artifact_reused": True,
            "captured_source_and_metadata_hashes_verified": True,
            "original_paths_changed_after_capture_without_changing_ingest": True,
            "tampered_captured_source_rejected": True,
            "tampered_captured_metadata_rejected": True,
            "ingested_at_frozen_after_source_capture": True,
            "task_event_count": len(events),
            "workflow_state_completed": True,
            "provider_calls": 0,
            "source_text_is_synthetic": True,
        }
    finally:
        if second_worker is not None:
            _stop_worker(second_worker)
        if first_worker is not None:
            _stop_worker(first_worker)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--address", default="localhost:7233")
    parser.add_argument("--task-queue", default="raama-industrial-rag-v1")
    parser.add_argument("--state-dir", type=Path)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--ready-file", type=Path)
    parser.add_argument("--attempt-log", type=Path)
    parser.add_argument("--crash-after-completion", action="store_true")
    args = parser.parse_args()
    if args.worker:
        if args.ready_file is None or args.attempt_log is None:
            raise ValueError("smoke worker requires ready-file and attempt-log paths")
        asyncio.run(_worker(args))
    else:
        if args.state_dir is None:
            raise ValueError("smoke run requires a unique state-dir path")
        print(json.dumps(
            asyncio.run(_run_smoke(args.address, args.task_queue, args.state_dir.resolve())),
            ensure_ascii=False, sort_keys=True, indent=2,
        ))


if __name__ == "__main__":
    main()
