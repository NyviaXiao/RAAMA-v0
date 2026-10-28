"""Run real Temporal index, evaluation, and offline-cache research operations on synthetic inputs."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import subprocess
import sys
import time
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from adaptive_mas.harness_evolution import seed_candidate  # noqa: E402
from adaptive_mas.harness_state import HarnessStateStore  # noqa: E402
from adaptive_mas.industrial_rag.models import (  # noqa: E402
    Block,
    Document,
    DocumentBundle,
    DocumentVersion,
    FinancialFact,
    Section,
    Span,
)
from adaptive_mas.industrial_rag.object_store import LocalContentAddressedStore  # noqa: E402
from adaptive_mas.industrial_rag.repository import IndustrialRagStore  # noqa: E402
from adaptive_mas.industrial_rag.temporal_workflow import (  # noqa: E402
    TASK_QUEUE,
    IndustrialRagEvaluationRequest,
    IndustrialRagIndexRequest,
    IndustrialRagOperationRequest,
    IndustrialRagOperationWorkflow,
    IndustrialRagResearchRequest,
    capture_evaluation_request,
    capture_research_request,
)


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, default=str) + "\n", encoding="utf-8")


def _seed_synthetic_index(run_dir: Path, run_id: str) -> tuple[str, Path, Path]:
    payload = b"Synthetic Temporal operation fixture. Revenue was 120 million CNY in 2025.\n"
    content_hash = hashlib.sha256(payload).hexdigest()
    version_id = f"temporal-operation-version-{run_id}"
    document_id = f"temporal-operation-document-{run_id}"
    source_id = f"temporal-operation-source-{run_id}"
    captured_at = datetime(2025, 3, 1, 10, tzinfo=timezone.utc)
    text = payload.decode("utf-8")
    section_id = f"{version_id}-section"
    span_id = f"{version_id}-span"
    document = Document(
        document_id=document_id,
        title="Synthetic Temporal operation fixture",
        document_type="synthetic_report",
        security_code="000001",
        entity_name="Synthetic issuer",
        report_period="2025",
        accounting_basis="consolidated",
    )
    version = DocumentVersion(
        version_id=version_id,
        source_id=source_id,
        content_sha256=content_hash,
        source_url="https://example.invalid/temporal-operation-fixture",
        rights_note="Synthetic fixture only; no external source content.",
        published_at=captured_at,
        first_available_at=captured_at,
        captured_at=captured_at,
        ingested_at=captured_at,
        time_basis="strict_forward",
        parser_name="synthetic-fixture",
        parser_version="1",
        parser_confidence=1.0,
    )
    bundle = DocumentBundle(
        document=document,
        version=version,
        payload=payload,
        sections=(Section(section_id, version_id, "Synthetic", 1, start_page=1, end_page=1),),
        spans=(Span(span_id, version_id, 1, 0, len(text), text),),
        blocks=(Block(
            f"{version_id}-block", version_id, text,
            section_id=section_id, page_number=1, char_start=0, char_end=len(text), span_id=span_id,
        ),),
        facts=(FinancialFact(
            fact_id=f"{version_id}-fact",
            version_id=version_id,
            security_code="000001",
            entity_name="Synthetic issuer",
            metric="营业收入",
            raw_value="120",
            normalized_value="120",
            unit="人民币百万元",
            multiplier="1",
            currency="CNY",
            report_period="2025",
            consolidation="consolidated",
            span_ids=(span_id,),
            parser_version="1",
            confidence=1.0,
            review_status="reviewed",
        ),),
        read_principals=("local-public",),
    )
    database_path = run_dir / "rag.sqlite3"
    object_root = run_dir / "objects"
    IndustrialRagStore(database_path, LocalContentAddressedStore(object_root)).ingest_bundle(bundle)
    return version_id, database_path, object_root


def _research_bundle() -> dict[str, Any]:
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


def _start_worker(address: str, task_queue: str, run_dir: Path) -> subprocess.Popen[bytes]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    log_stream = (run_dir / "worker.log").open("wb")
    process = subprocess.Popen(
        [
            sys.executable, "-m", "adaptive_mas.industrial_rag.temporal_workflow",
            "worker", "--address", address, "--task-queue", task_queue,
        ],
        cwd=ROOT,
        env=env,
        stdout=log_stream,
        stderr=subprocess.STDOUT,
    )
    process._raama_log_stream = log_stream  # type: ignore[attr-defined]
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        if process.poll() is not None:
            log_stream.flush()
            detail = (run_dir / "worker.log").read_text(encoding="utf-8", errors="replace")[-2000:]
            raise RuntimeError(f"Temporal operation worker exited before polling: {detail}")
        time.sleep(0.2)
    return process


def _stop_worker(process: subprocess.Popen[bytes]) -> int:
    if process.poll() is None:
        process.terminate()
        process.wait(timeout=10)
    log_stream = getattr(process, "_raama_log_stream", None)
    if log_stream is not None:
        log_stream.close()
    return int(process.returncode or 0)


async def _run_operation(client: Any, task_queue: str, operation: str, payload: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    request = IndustrialRagOperationRequest(operation, payload)
    workflow_id = str(payload["task_id"])
    handle = await client.start_workflow(
        IndustrialRagOperationWorkflow.run,
        request,
        id=workflow_id,
        task_queue=task_queue,
    )
    result: dict[str, Any] | None = None
    failure: dict[str, str] | None = None
    try:
        raw_result = await asyncio.wait_for(handle.result(), timeout=120)
        if not isinstance(raw_result, dict):
            raise TypeError("Temporal operation result must be an object")
        result = raw_result
    except Exception as error:
        failure = {"error_type": type(error).__name__, "message": str(error)[:1000]}

    state = await handle.query(IndustrialRagOperationWorkflow.state)
    description = await handle.describe()
    history = await handle.fetch_history()
    history_json = history.to_json()
    history_path = run_dir / f"{operation}-history.json"
    history_path.write_text(history_json + "\n", encoding="utf-8")
    receipt = {
        "workflow_id": handle.id,
        "run_id": history.run_id,
        "temporal_status": str(description.status),
        "result": result,
        "failure": failure,
        "workflow_state": state,
        "history_file": history_path.name,
        "history_sha256": hashlib.sha256(history_json.encode("utf-8")).hexdigest(),
        "history_event_count": len(history.events),
    }
    _write_json(run_dir / f"{operation}-receipt.json", receipt)
    return receipt


async def run(address: str, output_dir: Path, task_queue: str) -> dict[str, Any]:
    if output_dir.exists():
        raise ValueError("output directory already exists; refusing to overwrite workflow evidence")
    output_dir.mkdir(parents=True, exist_ok=False)
    run_id = uuid.uuid4().hex[:12]
    version_id, database_path, object_root = _seed_synthetic_index(output_dir, run_id)
    state_path = output_dir / "workflow-state.sqlite3"

    manifest_path = output_dir / "evaluation-manifest.json"
    predictions_path = output_dir / "evaluation-predictions.json"
    manifest_path.write_text(json.dumps({
        "benchmark_id": "synthetic-temporal-evaluation",
        "cases": [{
            "case_id": "synthetic-unanswerable",
            "data_partition": "development",
            "view_status": "unseen",
            "answerability": "unanswerable",
            "required_evidence": [],
        }],
    }, sort_keys=True), encoding="utf-8")
    predictions_path.write_text(json.dumps({
        "run_id": f"synthetic-temporal-run-{run_id}",
        "cases": [{
            "case_id": "synthetic-unanswerable",
            "predicted_answerability": "unanswerable",
            "retrieved": [],
        }],
    }, sort_keys=True), encoding="utf-8")
    research_bundle_path = output_dir / "research-bundle.json"
    _write_json(research_bundle_path, _research_bundle())

    from temporalio.client import Client

    client = await Client.connect(address)
    worker = _start_worker(address, task_queue, output_dir)
    try:
        index_request = IndustrialRagIndexRequest(
            task_id=f"temporal-index-{run_id}",
            idempotency_key=f"temporal-index-{run_id}",
            release_id=f"temporal-release-{run_id}",
            version_ids=(version_id,),
            database_path=str(database_path),
            object_root=str(object_root),
            state_database_path=str(state_path),
            actor_id="synthetic-operator",
            reason="bounded Temporal service operation smoke",
        )
        index_receipt = await _run_operation(client, task_queue, "index", asdict(index_request), output_dir)

        evaluation_request = capture_evaluation_request(IndustrialRagEvaluationRequest(
            task_id=f"temporal-evaluate-{run_id}",
            idempotency_key=f"temporal-evaluate-{run_id}",
            manifest_path=str(manifest_path),
            predictions_path=str(predictions_path),
            scratch_root=str(output_dir / "capture-scratch"),
            state_database_path=str(state_path),
        ))
        evaluation_receipt = await _run_operation(
            client, task_queue, "evaluate", asdict(evaluation_request), output_dir,
        )

        research_request = capture_research_request(IndustrialRagResearchRequest(
            task_id=f"temporal-research-{run_id}",
            idempotency_key=f"temporal-research-{run_id}",
            bundle_path=str(research_bundle_path),
            scratch_root=str(output_dir / "capture-scratch"),
            state_database_path=str(state_path),
            response_cache_dir=str(output_dir / "empty-response-cache"),
            call_ledger_path=str(output_dir / "research-call-ledger.jsonl"),
            request_archive_dir=str(output_dir / "research-request-archive"),
        ))
        research_receipt = await _run_operation(
            client, task_queue, "research", asdict(research_request), output_dir,
        )
    finally:
        worker_exit = _stop_worker(worker)

    durable = HarnessStateStore(state_path)
    research_task = durable.get_task(f"temporal-research-{run_id}")
    if (
        not isinstance(index_receipt["result"], dict)
        or index_receipt["result"].get("status") != "published"
        or not isinstance(evaluation_receipt["result"], dict)
        or evaluation_receipt["result"].get("sample_count") != 1
        or research_receipt["result"] is not None
        or research_receipt["workflow_state"].get("status") != "failed"
        or not isinstance(research_task, dict)
        or research_task.get("status") != "failed"
        or int(research_task.get("provider_calls_used", -1)) != 0
        or worker.poll() is None
    ):
        raise AssertionError("Temporal operation service smoke did not meet its success/fail-closed checks")

    summary = {
        "run_id": run_id,
        "server_address": address,
        "task_queue": task_queue,
        "provider_calls": 0,
        "worker_exit_code": worker_exit,
        "worker_stopped": worker.poll() is not None,
        "operations": {
            "index": index_receipt,
            "evaluate": evaluation_receipt,
            "research": {
                **research_receipt,
                "durable_task_status": research_task["status"],
                "durable_provider_calls_used": research_task["provider_calls_used"],
                "expected_outcome": "offline_cache_miss_failed_closed_without_provider_call",
            },
        },
        "limitations": [
            "Synthetic index and evaluation only; no public source or production release was used.",
            "Research exercised the real Temporal workflow and worker with an empty response cache and failed closed before provider access; successful cached research replay was not exercised through the live service.",
            "This does not validate Temporal HA, production scaling, or PostgreSQL-backed Temporal state.",
        ],
    }
    _write_json(output_dir / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--address", required=True, help="loopback address for a dedicated Temporal dev server")
    parser.add_argument("--output-dir", type=Path, required=True, help="new directory for synthetic inputs and receipts")
    parser.add_argument("--task-queue", default=TASK_QUEUE)
    args = parser.parse_args()
    print(json.dumps(asyncio.run(run(args.address, args.output_dir, args.task_queue)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
