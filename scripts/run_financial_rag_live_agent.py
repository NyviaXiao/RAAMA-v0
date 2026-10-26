"""Run one generation-driven Qwen financial agent against a bound local RAG release."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from adaptive_mas.agents import harness as agent_harness  # noqa: E402
from adaptive_mas.agents.harness import (  # noqa: E402
    CallBudget,
    ProviderQuotaRejected,
    QwenClient,
    _fundamental_handlers,
    _run_tool_agent,
)
from adaptive_mas.industrial_rag.qwen_models import (  # noqa: E402
    LocalQwenDenseAdapter,
    LocalQwenReranker,
    Qwen3EmbeddingRuntime,
    Qwen3RerankerRuntime,
)
from adaptive_mas.industrial_rag.retrieval import IndustrialRagService  # noqa: E402
from adaptive_mas.industrial_rag.object_store import LocalContentAddressedStore  # noqa: E402
from adaptive_mas.industrial_rag.repository import IndustrialRagStore  # noqa: E402
from scripts import run_financial_rag_current_capture as capture  # noqa: E402


SOURCE_RUN_ID = "current-capture-qwen-cmb-traced-20261026"
SOURCE_RUN_DIR = ROOT / ".tmp" / "industrial_rag_current_capture_20261026" / SOURCE_RUN_ID
LIVE_OUTPUT_ROOT = ROOT / ".tmp" / "industrial_rag_live_agent_20261027"
FALLBACK_MODEL = "qwen3.7-max-preview"
MAX_PROVIDER_CALLS = 5
MAX_LOGICAL_TURNS = 4


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".pending")
    payload = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with temporary.open("x", encoding="utf-8", newline="\n") as output:
        output.write(payload)
        output.flush()
        os.fsync(output.fileno())
    temporary.replace(path)


def _append_jsonl(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False).encode("utf-8") + b"\n"
    with path.open("ab") as output:
        output.write(payload)
        output.flush()
        os.fsync(output.fileno())


def _implementation_hashes() -> dict[str, str]:
    paths = {
        "live_agent_runner.py": Path(__file__).resolve(),
        "current_capture_runner.py": Path(capture.__file__).resolve(),
        "harness.py": Path(agent_harness.__file__).resolve(),
        "industrial_rag_retrieval.py": ROOT / "src" / "adaptive_mas" / "industrial_rag" / "retrieval.py",
        "industrial_rag_qwen_models.py": ROOT / "src" / "adaptive_mas" / "industrial_rag" / "qwen_models.py",
    }
    return {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in paths.items()}


def _load_provider_client() -> QwenClient:
    api_key = os.environ.get("POPAGENT_LLM_KEY")
    if not api_key:
        key_path = ROOT / "data" / "raw" / "qwen_key.txt"
        if not key_path.is_file():
            raise RuntimeError("the existing Qwen client has no configured credential")
        api_key = key_path.read_text(encoding="utf-8").strip()
    if not api_key:
        raise RuntimeError("the existing Qwen client credential is empty")
    return QwenClient.from_environment(api_key=api_key)


class QuotaFallbackClient:
    """Switch to the approved preview model only after an explicit quota rejection."""

    def __init__(self, primary: QwenClient, event_path: Path) -> None:
        self.primary = primary
        self.fallback = QwenClient(primary.endpoint, primary.api_key, FALLBACK_MODEL)
        self.active = primary
        self.event_path = event_path
        self.fallback_used = False
        self.model_events: list[dict[str, object]] = []

    @property
    def model(self) -> str:
        return self.active.model

    def complete(self, *args: Any, **kwargs: Any) -> dict[str, object]:
        if self.fallback_used:
            return self.active.complete(*args, **kwargs)
        try:
            return self.primary.complete(*args, **kwargs)
        except ProviderQuotaRejected as error:
            if self.primary.model == FALLBACK_MODEL:
                raise
            event: dict[str, object] = {
                "event": "fallback_model_selected_after_provider_quota_rejection",
                "primary_model": self.primary.model,
                "fallback_model": FALLBACK_MODEL,
                "http_status": error.status_code,
                "provider_error_code": error.error_code,
                "provider_error_sha256": error.response_sha256,
                "recorded_at": datetime.now().astimezone().isoformat(),
            }
            _append_jsonl(self.event_path, event)
            self.model_events.append(event)
            self.fallback_used = True
            self.active = self.fallback
            return self.fallback.complete(*args, **kwargs)


def _verify_capture_run() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    if not SOURCE_RUN_DIR.is_dir():
        raise FileNotFoundError(f"bound source capture is missing: {SOURCE_RUN_ID}")
    envelope = capture._read_run_identity(SOURCE_RUN_DIR / "run_identity.json")
    run_identity = dict(envelope["run_identity"])
    if run_identity.get("run_id") != SOURCE_RUN_ID:
        raise capture.RunIdentityError("source capture run identity has a different run id")
    if run_identity.get("evaluation_scope", {}).get("case_ids") != [
        "current_capture_cmb_2024_revenue"
    ]:
        raise capture.RunIdentityError("source capture is not the reviewed single-case CMB run")
    baseline_path = SOURCE_RUN_DIR / "agent_progress_600036.json"
    if not baseline_path.is_file():
        raise FileNotFoundError("completed local Qwen baseline progress is missing")
    baseline_progress = json.loads(baseline_path.read_text(encoding="utf-8"))
    if not isinstance(baseline_progress, dict) or not isinstance(baseline_progress.get("completed_result"), dict):
        raise capture.RunIdentityError("local Qwen baseline has no completed agent result")
    completed = baseline_progress["completed_result"]
    if baseline_progress.get("attempts_sha256") != capture._canonical_hash(baseline_progress.get("attempts")):
        raise capture.RunIdentityError("local Qwen baseline progress failed its attempt hash")
    if baseline_progress.get("completed_result_sha256") != capture._canonical_hash(completed):
        raise capture.RunIdentityError("local Qwen baseline result failed its completion hash")
    journal = capture.RetrievalTraceJournal(
        SOURCE_RUN_DIR / "retrieval_trace.jsonl",
        run_identity_sha256=str(envelope["run_identity_sha256"]),
        runtime_model_identity_sha256=str(envelope["runtime_model_identity_sha256"]),
    )
    attempts = baseline_progress.get("attempts")
    if not isinstance(attempts, list) or not attempts or not isinstance(attempts[-1], dict):
        raise capture.RunIdentityError("local Qwen baseline has no completed agent attempt")
    journal.verify_references(attempts[-1].get("retrieval_trace_records"), require_nonempty=True)
    journal.verify_references(completed.get("retrieval_trace_records"), require_nonempty=True)
    paired_path = SOURCE_RUN_DIR / "agent_progress_600036_ablation12.json"
    paired_progress = json.loads(paired_path.read_text(encoding="utf-8"))
    if paired_progress.get("attempts_sha256") != capture._canonical_hash(paired_progress.get("attempts")):
        raise capture.RunIdentityError("failed paired reranker progress failed its attempt hash")
    paired_attempts = paired_progress.get("attempts")
    if not isinstance(paired_attempts, list) or not paired_attempts or not isinstance(paired_attempts[-1], dict):
        raise capture.RunIdentityError("failed paired reranker attempt is missing")
    journal.verify_references(paired_attempts[-1].get("retrieval_trace_records"), require_nonempty=True)
    case_id = str(baseline_progress.get("case_id"))
    expected_case = next((spec for spec in capture.CASE_SPECS if str(spec["case_id"]) == case_id), None)
    if expected_case is None:
        raise capture.RunIdentityError("source capture case is absent from the reviewed case registry")
    return envelope, baseline_progress, dict(expected_case)


def _load_service(
    source_envelope: dict[str, Any],
) -> tuple[IndustrialRagStore, IndustrialRagService, dict[str, object]]:
    store = IndustrialRagStore(
        SOURCE_RUN_DIR / "industrial_rag.sqlite3",
        LocalContentAddressedStore(SOURCE_RUN_DIR / "objects"),
    )
    if store.active_release_id() != SOURCE_RUN_ID:
        raise capture.RunIdentityError("the bound source release is no longer active in its local store")
    embedding = Qwen3EmbeddingRuntime(capture.QWEN_ASSETS)
    reranker_runtime = Qwen3RerankerRuntime(capture.QWEN_ASSETS)
    dense = LocalQwenDenseAdapter(
        store,
        embedding,
        cache_path=ROOT / ".tmp" / "industrial_rag_dense_vector_cache.sqlite3",
    )
    reranker = LocalQwenReranker(reranker_runtime)
    runtime_identity = capture._runtime_model_identity(
        use_qwen=True,
        dense_adapter=dense,
        reranker=reranker,
    )
    saved_identity = source_envelope.get("runtime_model_identity")
    saved_identity_hash = source_envelope.get("runtime_model_identity_sha256")
    if saved_identity != runtime_identity or saved_identity_hash != capture._canonical_hash(runtime_identity):
        raise capture.RunIdentityError("loaded local Qwen runtime identity differs from the source capture")
    service = IndustrialRagService(
        store,
        channel_adapters={"dense": dense},
        reranker=reranker,
        allow_retrospective_diagnostic=False,
    )
    return store, service, runtime_identity


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="current-capture-qwen-cmb-live-agent-20261027")
    args = parser.parse_args()
    if not capture.re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}", args.run_id):
        raise ValueError("run-id must be a safe immutable identifier")

    source_envelope, baseline_progress, spec = _verify_capture_run()
    output_dir = LIVE_OUTPUT_ROOT / args.run_id
    output_dir.mkdir(parents=True, exist_ok=False)
    store, service, runtime_identity = _load_service(source_envelope)
    provider_client = _load_provider_client()
    runtime_identity_sha256 = capture._canonical_hash(runtime_identity)
    baseline_identity = baseline_progress
    case_identity = {
        key: baseline_identity[key]
        for key in (
            "run_identity_sha256", "case_id", "security_code", "source_id", "source_sha256",
            "version_id", "report_periods", "query", "as_of", "time_mode",
            "retrieval_policy_sha256", "principal_id", "permission_revision", "continuation_revision",
            "case_identity_sha256",
        )
    }
    implementation_hashes = _implementation_hashes()
    live_identity: dict[str, object] = {
        "schema_version": "raama.financial-rag.live-agent-identity.v1",
        "run_id": args.run_id,
        "source_capture_run_id": SOURCE_RUN_ID,
        "source_run_identity_sha256": source_envelope["run_identity_sha256"],
        "source_runtime_model_identity_sha256": source_envelope["runtime_model_identity_sha256"],
        "case_identity_sha256": case_identity["case_identity_sha256"],
        "runtime_model_identity_sha256": runtime_identity_sha256,
        "agent_driver": "qwen_generation_tool_calls",
        "primary_model": provider_client.model,
        "quota_fallback_model": FALLBACK_MODEL,
        "quota_fallback_policy": "only on structured provider error explicitly identifying quota exhaustion",
        "max_provider_calls": MAX_PROVIDER_CALLS,
        "max_logical_turns": MAX_LOGICAL_TURNS,
        "implementation_sha256": implementation_hashes,
    }
    live_identity_sha256 = capture._canonical_hash(live_identity)
    _write_json(output_dir / "live_agent_identity.json", {
        "live_agent_identity": live_identity,
        "live_agent_identity_sha256": live_identity_sha256,
    })

    if provider_client.model != live_identity["primary_model"]:
        raise capture.RunIdentityError("configured provider model changed after live identity was bound")
    event_path = output_dir / "provider_model_events.jsonl"
    client = QuotaFallbackClient(provider_client, event_path)
    budget = CallBudget(
        max_calls=MAX_PROVIDER_CALLS,
        ledger_path=output_dir / "provider_request_ledger.jsonl",
        response_cache_dir=output_dir / "provider_response_cache",
        request_archive_dir=output_dir / "provider_request_archive",
        package_identity_sha256=live_identity_sha256,
    )
    budget.limit_next_task(MAX_LOGICAL_TURNS)

    journal = capture.RetrievalTraceJournal(
        output_dir / "retrieval_trace.jsonl",
        run_identity_sha256=live_identity_sha256,
        runtime_model_identity_sha256=runtime_identity_sha256,
    )
    permission_revision = store.permission_revision()
    continuation_revision = store.continuation_revision()
    release_id = SOURCE_RUN_ID
    started_at = datetime.now().astimezone().isoformat()
    progress: dict[str, Any] = {
        **live_identity,
        "live_agent_identity_sha256": live_identity_sha256,
        "runtime_model_identity": runtime_identity,
        "case_identity": case_identity,
        "status": "running",
        "started_at": started_at,
        "tool_events": [],
        "retrieval_trace_records": [],
        "model_events": [],
    }
    progress_path = output_dir / "agent_progress.json"
    _write_json(progress_path, progress)

    def identity_check() -> None:
        if store.active_release_id() != release_id:
            raise capture.RunIdentityError("active source release changed during live agent run")
        if store.permission_revision() != permission_revision:
            raise capture.RunIdentityError("source permission revision changed during live agent run")
        if store.continuation_revision() != continuation_revision:
            raise capture.RunIdentityError("source continuation revision changed during live agent run")
        if capture._runtime_model_identity(
            use_qwen=True,
            dense_adapter=service.channel_adapters["dense"],
            reranker=service.reranker,
        ) != runtime_identity:
            raise capture.RunIdentityError("local Qwen runtime identity changed during live agent run")
        if _implementation_hashes() != implementation_hashes:
            raise capture.RunIdentityError("live agent implementation changed during live run")

    current_tool_event: dict[str, Any] | None = None

    def trace_callback(event: dict[str, object]) -> None:
        if current_tool_event is None:
            raise RuntimeError("retrieval trace event has no active agent tool call")
        record = journal.append({
            "live_agent_identity_sha256": live_identity_sha256,
            "case_identity_sha256": case_identity["case_identity_sha256"],
            "tool": current_tool_event.get("tool"),
            "tool_event_index": len(progress["tool_events"]) - 1,
        }, event)
        progress["retrieval_trace_records"].append({
            "sequence": record["sequence"],
            "event_sha256": record["event_sha256"],
        })
        progress["updated_at"] = datetime.now().astimezone().isoformat()
        _write_json(progress_path, progress)

    decision_time = datetime.fromisoformat(str(case_identity["as_of"])).astimezone(capture.SHANGHAI)
    visible_evidence_ids: set[str] = set()
    tools, handlers, catalog = _fundamental_handlers(
        None,
        decision_time,
        industrial_rag_service=service,
        industrial_rag_principal_id=str(case_identity["principal_id"]),
        industrial_rag_release_id=release_id,
        research_policy_json=str(capture._read_run_identity(SOURCE_RUN_DIR / "run_identity.json")["run_identity"]["research_policy_json"]),
        retrieval_trace_callback=trace_callback,
    )
    handlers["submit_report"] = lambda arguments: capture._accept_report(
        arguments,
        catalog,
        "fundamental",
        visible_evidence_ids=visible_evidence_ids,
        decision_time=decision_time,
    )
    for name, handler in tuple(handlers.items()):
        def traced_handler(arguments: dict[str, object], *, tool_name=name, tool_handler=handler):
            nonlocal current_tool_event
            identity_check()
            event: dict[str, Any] = {
                "tool": tool_name,
                "arguments": dict(arguments),
                "status": "started",
                "started_at": datetime.now().astimezone().isoformat(),
            }
            progress["tool_events"].append(event)
            current_tool_event = event
            _write_json(progress_path, progress)
            try:
                result = tool_handler(arguments)
            except Exception as error:
                event.update({
                    "status": "error",
                    "error": {"type": type(error).__name__, "message": str(error)},
                    "ended_at": datetime.now().astimezone().isoformat(),
                })
                _write_json(progress_path, progress)
                raise
            event.update({
                "status": "returned",
                "result": result,
                "ended_at": datetime.now().astimezone().isoformat(),
            })
            _write_json(progress_path, progress)
            identity_check()
            return result
        handlers[name] = traced_handler

    case = {
        "as_of": decision_time.isoformat(),
        "question": capture.QUERY_TEXT,
        "symbol": str(spec["security_code"]),
        "allowed_data": "the pinned local strict-forward industrial RAG release only",
    }
    final_result: dict[str, object] | None = None
    failure: dict[str, object] | None = None
    try:
        identity_check()
        final_result = _run_tool_agent(
            role_id="fundamental",
            system_prompt="Use the indexed local financial filing and cite every source fact.",
            task=case,
            tools=tools,
            handlers=handlers,
            client=client,  # type: ignore[arg-type]
            budget=budget,
            max_turns=MAX_LOGICAL_TURNS,
            visible_evidence_ids=visible_evidence_ids,
        )
        identity_check()
        if final_result.get("status") != "completed":
            raise RuntimeError("generation-driven financial agent did not complete")
        tool_trace = list(final_result.get("tool_trace", []))
        reads = [item.get("result", {}) for item in tool_trace if item.get("tool") == "read_indexed_financial_evidence"]
        if not reads or not all(str(item.get("content_scope", "")).startswith("reopened_source") for item in reads):
            raise RuntimeError("generation-driven agent did not reopen its cited source evidence")
        finding = final_result["report"]["findings"][0]
        expected_validation = (
            "calculation_recomputed_decimal_50_half_even_semantics_unverified"
            if capture._case_calculation_eligible(spec)
            else "source_observations_validated_semantics_unverified"
        )
        if finding.get("claim_validation", {}).get("status") != expected_validation:
            raise RuntimeError("generation-driven report did not pass the expected source or Decimal validation")
    except Exception as error:
        failure = {
            "type": type(error).__name__,
            "message": str(error),
        }
        if isinstance(error, ProviderQuotaRejected):
            failure.update({
                "provider_http_status": error.status_code,
                "provider_error_code": error.error_code,
                "provider_error_sha256": error.response_sha256,
            })

    if failure is None:
        progress["status"] = "completed"
        progress["completed_at"] = datetime.now().astimezone().isoformat()
        progress["agent_tool_trace"] = final_result.get("tool_trace") if final_result else None
        progress["validated_report"] = final_result.get("report") if final_result else None
        progress["actual_model"] = final_result.get("model") if final_result else client.model
    else:
        progress["status"] = "failed"
        progress["completed_at"] = datetime.now().astimezone().isoformat()
        progress["failure"] = failure
    progress["provider_budget"] = budget.summary()
    progress["model_events"] = list(client.model_events)
    progress["tool_count"] = len(progress["tool_events"])
    progress["retrieval_trace_event_count"] = journal.sequence
    progress["retrieval_trace_sha256"] = hashlib.sha256((output_dir / "retrieval_trace.jsonl").read_bytes()).hexdigest()
    progress["provider_request_ledger_sha256"] = hashlib.sha256((output_dir / "provider_request_ledger.jsonl").read_bytes()).hexdigest() if (output_dir / "provider_request_ledger.jsonl").is_file() else None
    _write_json(progress_path, progress)
    output_record: dict[str, object] = {
        "schema_version": "raama.financial-rag.live-agent-result.v1",
        "live_agent_identity_sha256": live_identity_sha256,
        "source_capture_run_id": SOURCE_RUN_ID,
        "source_run_identity_sha256": source_envelope["run_identity_sha256"],
        "case_identity_sha256": case_identity["case_identity_sha256"],
        "runtime_model_identity_sha256": runtime_identity_sha256,
        "agent_driver": "qwen_generation_tool_calls",
        "generation_model_called": budget.provider_call_count > 0 or budget.cache_hits > 0,
        "actual_model": final_result.get("model") if final_result else client.model,
        "fallback_used": client.fallback_used,
        "model_events": client.model_events,
        "provider_budget": budget.summary(),
        "status": "completed" if failure is None else "failed",
        "failure": failure,
        "agent_result": final_result,
        "tool_events": progress["tool_events"],
        "retrieval_trace_event_count": journal.sequence,
        "retrieval_trace_sha256": progress["retrieval_trace_sha256"],
        "retrieval_trace_references": progress["retrieval_trace_records"],
    }
    _write_json(output_dir / "live_agent_result.json", output_record)
    summary = {
        "run_id": args.run_id,
        "status": output_record["status"],
        "agent_driver": output_record["agent_driver"],
        "generation_model_called": output_record["generation_model_called"],
        "actual_model": output_record["actual_model"],
        "fallback_used": client.fallback_used,
        "provider_call_count": budget.provider_call_count,
        "logical_inference_count": len(budget.logical_calls),
        "tool_count": len(progress["tool_events"]),
        "retrieval_trace_event_count": journal.sequence,
        "claim_validation_status": (
            final_result.get("report", {}).get("findings", [{}])[0].get("claim_validation", {}).get("status")
            if final_result else None
        ),
        "output_dir": str(output_dir),
    }
    _write_json(output_dir / "live_agent_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False))
    if failure is not None:
        raise RuntimeError(f"generation-driven financial agent failed: {failure['type']}: {failure['message']}")


if __name__ == "__main__":
    main()
