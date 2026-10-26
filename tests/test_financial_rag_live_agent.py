from __future__ import annotations

import io
import json
import hashlib
import shutil
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from urllib.error import HTTPError

import pytest

from adaptive_mas.agents import harness as harness_module
from adaptive_mas.agents.harness import CallBudget, QwenClient
from adaptive_mas.harness_state import HarnessStateStore
from adaptive_mas.industrial_rag.models import (
    Block,
    CorpusReleaseSpec,
    Document,
    DocumentBundle,
    DocumentVersion,
    FinancialFact,
    Section,
    Span,
)
from adaptive_mas.industrial_rag.object_store import LocalContentAddressedStore
from adaptive_mas.industrial_rag.repository import IndustrialRagStore
from adaptive_mas.research_provenance import stable_hash
from scripts import run_financial_rag_current_capture as capture
from scripts import run_financial_rag_live_agent as live_agent
from scripts.run_financial_rag_live_agent import FALLBACK_MODEL, QuotaFallbackClient


class _Response:
    def __init__(self, body: dict[str, object]) -> None:
        self.body = body

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self) -> bytes:
        return json.dumps(self.body).encode("utf-8")


def _http_error(code: int, body: dict[str, object]) -> HTTPError:
    return HTTPError(
        "https://fixture.invalid/chat/completions",
        code,
        "provider rejection",
        None,
        io.BytesIO(json.dumps(body).encode("utf-8")),
    )


def _synthetic_bound_case(tmp_path, monkeypatch):
    case_id = "synthetic_case"
    source_id = "synthetic-source"
    version_id = "synthetic-version"
    run_id = "synthetic-live-source"
    query = "Synthetic filing question"
    principal_id = "synthetic-analyst"
    spec = {
        "case_id": case_id,
        "source_id": source_id,
        "filename": "synthetic.pdf",
        "security_code": "123456",
        "accounting_basis": "Synthetic consolidated",
        "expected_raw_values": {"2023": "100", "2024": "110"},
    }
    monkeypatch.setattr(capture, "CASE_SPECS", (spec,))
    monkeypatch.setattr(capture, "QUERY_TEXT", query)
    monkeypatch.setattr(live_agent, "SOURCE_RUN_ID", run_id)

    now = datetime.now(timezone.utc).replace(microsecond=0)
    first_available_at = now - timedelta(days=2)
    captured_at = now - timedelta(days=1)
    ingested_at = now - timedelta(hours=12)
    payload = b"synthetic source payload for backend identity validation"
    source_sha256 = hashlib.sha256(payload).hexdigest()
    document = Document(
        document_id="synthetic-document",
        title="Synthetic filing fixture",
        document_type="annual_report_summary",
        security_code="123456",
        entity_name="Synthetic Corp",
        report_period="2024",
        accounting_basis="Synthetic consolidated",
    )
    version = DocumentVersion(
        version_id=version_id,
        source_id=source_id,
        content_sha256=source_sha256,
        source_url="https://example.invalid/synthetic.pdf",
        rights_note="Synthetic test data only",
        published_at=first_available_at,
        first_available_at=first_available_at,
        captured_at=captured_at,
        ingested_at=ingested_at,
        time_basis="strict_forward",
        parser_name="synthetic-test-parser",
        parser_version="1",
        parser_confidence=0.99,
    )
    text = "Synthetic reported metric"
    section = Section("synthetic-section", version_id, "Synthetic", 1, start_page=1, end_page=1)
    span = Span("synthetic-span", version_id, 1, 0, len(text), text)
    block = Block(
        "synthetic-block", version_id, text, section_id="synthetic-section", page_number=1,
        char_start=0, char_end=len(text), parser_confidence=0.99, span_id="synthetic-span",
    )
    facts = tuple(
        FinancialFact(
            fact_id=f"synthetic-fact-{period}",
            version_id=version_id,
            security_code="123456",
            entity_name="Synthetic Corp",
            metric="synthetic_metric",
            raw_value=value,
            normalized_value=value,
            unit="CNY",
            multiplier="1",
            currency="CNY",
            report_period=period,
            consolidation="Synthetic consolidated",
            span_ids=("synthetic-span",),
            parser_version="1",
            confidence=0.99,
            review_status="reviewed",
        )
        for period, value in (("2023", "100"), ("2024", "110"))
    )
    store = IndustrialRagStore(
        tmp_path / "identity.sqlite3",
        LocalContentAddressedStore(tmp_path / "identity-objects"),
    )
    store.ingest_bundle(DocumentBundle(
        document, version, payload, (section,), (span,), (block,), facts=facts,
        read_principals=(principal_id,),
    ))
    text_manifest = str(store.preview_reference_index_manifest(run_id, [version_id])["index_manifest_sha256"])
    release_spec = CorpusReleaseSpec(
        release_id=run_id,
        source_manifest_sha256=stable_hash(store.source_manifest([version_id])),
        parser_version="synthetic-test-parser@1",
        chunker_version="synthetic-chunker@1",
        embedding_model="none",
        embedding_dimension=None,
        required_backends=("sqlite_reference",),
        text_index_manifest_sha256=text_manifest,
        vector_index_manifest_sha256=hashlib.sha256(b"synthetic-vector-manifest").hexdigest(),
    )
    store.create_release(release_spec, [version_id])
    receipt = store.build_reference_index_receipt(run_id)
    store.mark_backend_ready(receipt, store.verify_reference_index)
    store.publish_release(run_id, actor_id="synthetic-test", reason="synthetic identity fixture")

    source = {
        "case_id": case_id,
        "source_id": source_id,
        "source_sha256": source_sha256,
        "version_id": version_id,
        "first_available_at": first_available_at.isoformat(),
    }
    access_scope = {
        "principal_id": principal_id,
        "permission_revision": store.permission_revision(),
        "continuation_revision": store.continuation_revision(),
    }
    policy_json = '{"channels":["structured_facts"]}'
    policy_sha256 = hashlib.sha256(policy_json.encode("utf-8")).hexdigest()
    run_identity = {
        "schema_version": "raama.financial-rag.current-capture.run-identity.v1",
        "run_id": run_id,
        "as_of": now.isoformat(),
        "time_mode": "strict_forward",
        "query": query,
        "qwen_inference_enabled": True,
        "evaluation_scope": {"case_ids": [case_id], "module_matrix_enabled": False},
        "research_policy_json": policy_json,
        "research_policy_sha256": policy_sha256,
        "source_manifest_sha256": capture._sha256(capture.SOURCE_MANIFEST.read_bytes()),
        "source_records": [source],
        "release": {
            "release_id": run_id,
            "spec_sha256": store.release_spec(run_id)[1],
            "version_ids": [version_id],
            "active_release_id": run_id,
        },
        "access_scope": access_scope,
        "indexed_financial_facts": capture._indexed_fact_identity(store, [source]),
        "case_specs_sha256": capture._canonical_hash(capture.CASE_SPECS),
    }
    envelope = {
        "schema_version": "raama.financial-rag.current-capture.run-identity-envelope.v1",
        "run_identity": run_identity,
        "run_identity_sha256": capture._canonical_hash(run_identity),
        "runtime_model_identity": {"fixture": "synthetic-runtime"},
    }
    envelope["runtime_model_identity_sha256"] = capture._canonical_hash(envelope["runtime_model_identity"])
    case_identity = capture._case_identity_fields(
        envelope["run_identity_sha256"],
        spec,
        source,
        as_of=now,
        retrieval_policy_sha256=policy_sha256,
        access_scope=access_scope,
        query=query,
    )
    progress = {
        **case_identity,
        "runtime_model_identity_sha256": envelope["runtime_model_identity_sha256"],
        "case_identity_sha256": capture._canonical_hash(case_identity),
        "progress_kind": "agent",
        "research_ablation_id": None,
    }
    monkeypatch.setattr(live_agent, "SOURCE_RUN_IDENTITY_SHA256", envelope["run_identity_sha256"])
    monkeypatch.setattr(live_agent, "SOURCE_RUNTIME_MODEL_IDENTITY_SHA256", envelope["runtime_model_identity_sha256"])
    monkeypatch.setattr(live_agent, "SOURCE_CASE_IDENTITY_SHA256", progress["case_identity_sha256"])
    return envelope, progress, spec, store


def test_live_agent_falls_back_only_after_explicit_quota_and_keeps_both_attempts(tmp_path, monkeypatch) -> None:
    store = HarnessStateStore(tmp_path / "provider-state.sqlite3")
    store.open_task("task", "task", {"case": "synthetic"}, max_provider_calls=2)
    sent_models: list[str] = []

    def fake_urlopen(request, **_kwargs):
        payload = json.loads(request.data)
        sent_models.append(str(payload["model"]))
        if payload["model"] == "qwen3.7-max-2026-06-08":
            raise _http_error(429, {"error": {"code": "insufficient_quota", "message": "quota exhausted"}})
        return _Response({
            "choices": [{"message": {"tool_calls": []}}],
            "usage": {"prompt_tokens": 11, "completion_tokens": 3},
        })

    monkeypatch.setattr(harness_module, "urlopen", fake_urlopen)
    primary = QwenClient("https://fixture.invalid", "fixture-secret", "qwen3.7-max-2026-06-08")
    fallback_client = QuotaFallbackClient(primary, tmp_path / "model-events.jsonl")
    budget = CallBudget(
        max_calls=5,
        ledger_path=tmp_path / "requests.jsonl",
        response_cache_dir=tmp_path / "responses",
        request_archive_dir=tmp_path / "request-archive",
        state_store=store,
        task_id="task",
        budget_scope_id="scope",
        task_provider_call_limit=2,
    )
    budget.limit_next_task(4)

    result = fallback_client.complete(
        [{"role": "user", "content": "synthetic"}],
        budget=budget,
        role="fundamental",
    )

    assert result["usage"] == {"prompt_tokens": 11, "completion_tokens": 3}
    assert sent_models == ["qwen3.7-max-2026-06-08", FALLBACK_MODEL]
    assert fallback_client.fallback_used is True
    assert fallback_client.model == FALLBACK_MODEL
    assert budget.provider_call_count == 2
    assert budget.summary()["quota_rejection_count"] == 1
    assert budget.unresolved_call is False
    assert "fixture-secret" not in (tmp_path / "requests.jsonl").read_text(encoding="utf-8")
    assert json.loads((tmp_path / "model-events.jsonl").read_text(encoding="utf-8"))[
        "provider_error_code"
    ] == "insufficient_quota"


def test_live_agent_does_not_fallback_on_provider_rate_limit(tmp_path, monkeypatch) -> None:
    sent_models: list[str] = []

    def fake_urlopen(request, **_kwargs):
        sent_models.append(str(json.loads(request.data)["model"]))
        raise _http_error(429, {"error": {"code": "rate_limit_exceeded", "message": "rate limit reached"}})

    monkeypatch.setattr(harness_module, "urlopen", fake_urlopen)
    primary = QwenClient("https://fixture.invalid", "fixture-secret", "primary-model")
    client = QuotaFallbackClient(primary, tmp_path / "model-events.jsonl")
    budget = CallBudget(
        max_calls=5,
        ledger_path=tmp_path / "requests.jsonl",
        response_cache_dir=tmp_path / "responses",
    )
    budget.limit_next_task(4)

    with pytest.raises(HTTPError):
        client.complete(
            [{"role": "user", "content": "synthetic"}],
            budget=budget,
            role="fundamental",
        )

    assert sent_models == ["primary-model"]
    assert client.fallback_used is False
    assert budget.unresolved_call is True
    assert budget.summary()["quota_rejection_count"] == 0
    assert not (tmp_path / "model-events.jsonl").exists()


def test_live_capture_identity_is_reconstructed_from_source_release_and_progress(tmp_path, monkeypatch) -> None:
    envelope, progress, spec, store = _synthetic_bound_case(tmp_path, monkeypatch)

    verified = live_agent._verify_capture_case_identity(envelope, progress, spec, store)

    assert verified["case_identity_sha256"] == progress["case_identity_sha256"]
    assert verified["source_sha256"] == envelope["run_identity"]["source_records"][0]["source_sha256"]
    assert verified["principal_id"] == envelope["run_identity"]["access_scope"]["principal_id"]
    assert verified["security_code"] == spec["security_code"]


def test_versioned_run_bundle_binds_new_issuer_and_exact_manifest_bytes(tmp_path, monkeypatch) -> None:
    envelope, progress, spec, store = _synthetic_bound_case(tmp_path, monkeypatch)
    bundle_dir = tmp_path / "portable-bundle"
    source_run_dir = bundle_dir / "source-run"
    source_run_dir.mkdir(parents=True)
    manifest_path = bundle_dir / "seed_manifest.json"
    manifest_bytes = b'{"synthetic_manifest":true}\r\n'
    manifest_path.write_bytes(manifest_bytes)
    manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    envelope["run_identity"]["source_manifest_sha256"] = manifest_sha256
    envelope["run_identity_sha256"] = capture._canonical_hash(envelope["run_identity"])
    source = envelope["run_identity"]["source_records"][0]
    access_scope = envelope["run_identity"]["access_scope"]
    case_identity = capture._case_identity_fields(
        envelope["run_identity_sha256"],
        spec,
        source,
        as_of=datetime.fromisoformat(envelope["run_identity"]["as_of"]),
        retrieval_policy_sha256=envelope["run_identity"]["research_policy_sha256"],
        access_scope=access_scope,
        query=envelope["run_identity"]["query"],
        report_periods=("2023", "2024"),
    )
    case_identity_sha256 = capture._canonical_hash(case_identity)
    progress.clear()
    progress.update({
        **case_identity,
        "runtime_model_identity_sha256": envelope["runtime_model_identity_sha256"],
        "case_identity_sha256": case_identity_sha256,
        "progress_kind": "agent",
        "research_ablation_id": None,
    })
    shutil.copy2(store.database_path, source_run_dir / "industrial_rag.sqlite3")
    shutil.copytree(store.object_store.root, source_run_dir / "objects")
    live_agent._write_json(source_run_dir / "run_identity.json", envelope)
    journal = capture.RetrievalTraceJournal(
        source_run_dir / "retrieval_trace.jsonl",
        run_identity_sha256=envelope["run_identity_sha256"],
        runtime_model_identity_sha256=envelope["runtime_model_identity_sha256"],
    )
    trace_record = journal.append({"case_id": spec["case_id"]}, {"event": "synthetic-source-reopen"})
    trace_reference = {"sequence": trace_record["sequence"], "event_sha256": trace_record["event_sha256"]}
    completed_result = {"retrieval_trace_records": [trace_reference]}
    attempt = {"retrieval_trace_records": [trace_reference]}
    progress.update({
        "attempts": [attempt],
        "attempts_sha256": capture._canonical_hash([attempt]),
        "completed_result": completed_result,
        "completed_result_sha256": capture._canonical_hash(completed_result),
    })
    live_agent._write_json(source_run_dir / "agent_progress_123456.json", progress)
    paired_attempt = {"retrieval_trace_records": [trace_reference]}
    live_agent._write_json(source_run_dir / "agent_progress_123456_ablation12.json", {
        "attempts": [paired_attempt],
        "attempts_sha256": capture._canonical_hash([paired_attempt]),
    })
    value = {
        "schema_version": live_agent.RUN_BUNDLE_SCHEMA,
        "source_run_id": envelope["run_identity"]["run_id"],
        "source_run_dir": "source-run",
        "source_manifest_path": "seed_manifest.json",
        "source_manifest_sha256": manifest_sha256,
        "case_id": spec["case_id"],
        "case_specs": list(capture.CASE_SPECS),
        "report_periods": ["2023", "2024"],
        "expected_run_identity_sha256": envelope["run_identity_sha256"],
        "expected_runtime_model_identity_sha256": envelope["runtime_model_identity_sha256"],
        "expected_case_identity_sha256": case_identity_sha256,
    }
    bundle_path = bundle_dir / "run-bundle.json"
    bundle_path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    expected_bundle_sha256 = hashlib.sha256(bundle_path.read_bytes()).hexdigest()

    bundle = live_agent._load_run_bundle(bundle_path, expected_bundle_sha256)
    loaded_envelope, loaded_progress, loaded_spec, verified = live_agent._verify_capture_run(bundle)

    assert bundle.source_run_id == "synthetic-live-source"
    assert bundle.source_run_dir == source_run_dir.resolve()
    assert verified["case_identity_sha256"] == case_identity_sha256
    assert verified["query"] == "Synthetic filing question"
    assert verified["report_periods"] == ["2023", "2024"]
    assert loaded_envelope["run_identity_sha256"] == envelope["run_identity_sha256"]
    assert loaded_progress["completed_result_sha256"] == capture._canonical_hash(loaded_progress["completed_result"])
    assert loaded_spec["security_code"] == "123456"

    manifest_path.write_bytes(manifest_bytes.replace(b"\r\n", b"\n"))
    with pytest.raises(capture.RunIdentityError, match="raw bytes"):
        bundle.verify_immutable_inputs()


def test_versioned_run_bundle_requires_exact_caller_supplied_bundle_hash(tmp_path, monkeypatch) -> None:
    envelope, _progress, spec, _store = _synthetic_bound_case(tmp_path, monkeypatch)
    bundle_dir = tmp_path / "portable-bundle"
    bundle_dir.mkdir()
    source_run_dir = bundle_dir / "source-run"
    source_run_dir.mkdir()
    manifest_path = bundle_dir / "seed_manifest.json"
    manifest_bytes = capture.SOURCE_MANIFEST.read_bytes()
    manifest_path.write_bytes(manifest_bytes)
    value = {
        "schema_version": live_agent.RUN_BUNDLE_SCHEMA,
        "source_run_id": envelope["run_identity"]["run_id"],
        "source_run_dir": "source-run",
        "source_manifest_path": "seed_manifest.json",
        "source_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "case_id": spec["case_id"],
        "case_specs": list(capture.CASE_SPECS),
        "report_periods": ["2023", "2024"],
        "expected_run_identity_sha256": envelope["run_identity_sha256"],
        "expected_runtime_model_identity_sha256": envelope["runtime_model_identity_sha256"],
        "expected_case_identity_sha256": "0" * 64,
    }
    bundle_path = bundle_dir / "run-bundle.json"
    bundle_path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    expected_bundle_sha256 = hashlib.sha256(bundle_path.read_bytes()).hexdigest()
    bundle_path.write_bytes(bundle_path.read_bytes() + b" ")

    with pytest.raises(capture.RunIdentityError, match="caller-supplied expected hash"):
        live_agent._load_run_bundle(bundle_path, expected_bundle_sha256)


@pytest.mark.parametrize(
    ("field", "changed_value"),
    [
        ("source_sha256", "0" * 64),
        ("principal_id", "parent-different-principal"),
        ("security_code", "000001"),
    ],
)
def test_live_capture_identity_rejects_tampered_progress_fields(tmp_path, monkeypatch, field, changed_value) -> None:
    envelope, progress, spec, store = _synthetic_bound_case(tmp_path, monkeypatch)
    progress[field] = changed_value

    with pytest.raises(capture.RunIdentityError, match="case identity"):
        live_agent._verify_capture_case_identity(envelope, progress, spec, store)


def test_live_capture_identity_rejects_missing_progress_identity_field(tmp_path, monkeypatch) -> None:
    envelope, progress, spec, store = _synthetic_bound_case(tmp_path, monkeypatch)
    del progress["case_identity_sha256"]

    with pytest.raises(capture.RunIdentityError, match="case identity"):
        live_agent._verify_capture_case_identity(envelope, progress, spec, store)


def test_live_capture_identity_rejects_rewritten_historical_run_envelope(tmp_path, monkeypatch) -> None:
    envelope, progress, spec, store = _synthetic_bound_case(tmp_path, monkeypatch)
    envelope["run_identity"]["as_of"] = (datetime.now(timezone.utc) + timedelta(days=1)).isoformat()
    envelope["run_identity_sha256"] = capture._canonical_hash(envelope["run_identity"])

    with pytest.raises(capture.RunIdentityError, match="envelope"):
        live_agent._verify_capture_case_identity(envelope, progress, spec, store)


@pytest.mark.parametrize("revision", ["permission", "continuation"])
def test_live_capture_identity_rejects_revision_drift_at_startup(tmp_path, monkeypatch, revision) -> None:
    envelope, progress, spec, store = _synthetic_bound_case(tmp_path, monkeypatch)
    if revision == "permission":
        store.set_access("synthetic-analyst", "synthetic-document", can_read=False)
    else:
        recorded = envelope["run_identity"]["access_scope"]["continuation_revision"]
        monkeypatch.setattr(store, "continuation_revision", lambda: recorded + 1)

    with pytest.raises(capture.RunIdentityError, match="revision has changed"):
        live_agent._verify_capture_case_identity(envelope, progress, spec, store)


@pytest.mark.parametrize("revision", ["permission", "continuation"])
def test_live_agent_rechecks_pinned_revisions_after_model_loading(tmp_path, monkeypatch, revision) -> None:
    envelope, progress, spec, store = _synthetic_bound_case(tmp_path, monkeypatch)
    case_identity = live_agent._verify_capture_case_identity(envelope, progress, spec, store)
    if revision == "permission":
        store.set_access("synthetic-analyst", "synthetic-document", can_read=False)
    else:
        recorded = case_identity["continuation_revision"]
        monkeypatch.setattr(store, "continuation_revision", lambda: recorded + 1)

    with pytest.raises(capture.RunIdentityError, match="before the live agent request"):
        live_agent._verify_runtime_access_revisions(store, case_identity)
