from __future__ import annotations

import json
import hashlib
import sys
from datetime import datetime, timezone

import pytest

from scripts import run_financial_rag_current_capture as capture
from scripts import run_financial_rag_live_agent as live_agent


def _case_identity() -> dict[str, object]:
    return {
        "run_identity_sha256": "run-hash",
        "case_id": "case-1",
        "security_code": "600036",
        "source_id": "source-1",
        "source_sha256": "source-hash",
        "version_id": "version-1",
        "report_periods": ["2023", "2024"],
        "query": capture.QUERY_TEXT,
        "as_of": "2026-10-26T00:00:00+00:00",
        "time_mode": "strict_forward",
        "retrieval_policy_sha256": "policy-hash",
        "principal_id": "local-public",
        "permission_revision": 3,
        "continuation_revision": 7,
    }


def _query_progress(case: dict[str, object], runtime_hash: str = "runtime-hash") -> dict[str, object]:
    row: dict[str, object] = {
        **case,
        "runtime_model_identity_sha256": runtime_hash,
        "case_identity_sha256": capture._canonical_hash(case),
        "result": {"source_values": ["339123", "337488"]},
    }
    row["case_result_sha256"] = capture._case_result_sha256(row)
    return {
        "schema_version": "raama.financial-rag.current-capture.query-progress.v2",
        "run_identity_sha256": "run-hash",
        "runtime_model_identity_sha256": runtime_hash,
        "cases": [row],
    }


def test_capture_emits_bundle_loadable_from_raw_manifest_bytes(tmp_path) -> None:
    output_dir = tmp_path / "new-run"
    output_dir.mkdir()
    manifest_path = output_dir / "manifest.json"
    manifest_bytes = b'{"fixture":"preserve-newlines"}\r\n'
    manifest_path.write_bytes(manifest_bytes)
    case_specs = ({"case_id": "ISSUER-2026", "source_id": "filing-2026", "security_code": "123456"},)

    emitted = capture._write_live_agent_run_bundle(
        output_dir,
        source_run_id="capture-new-issuer-2026",
        source_manifest_path=manifest_path,
        case_id="ISSUER-2026",
        case_specs=case_specs,
        report_periods=("2023", "2024"),
        expected_run_identity_sha256="1" * 64,
        expected_runtime_model_identity_sha256="2" * 64,
        expected_case_identity_sha256="3" * 64,
    )

    bundle_path = output_dir / emitted["path"]
    assert emitted["expected_bundle_sha256"] == hashlib.sha256(bundle_path.read_bytes()).hexdigest()
    bundle = live_agent._load_run_bundle(bundle_path, emitted["expected_bundle_sha256"])
    assert bundle.source_run_dir == output_dir.resolve()
    assert bundle.source_manifest_path == manifest_path.resolve()
    assert bundle.source_manifest_sha256 == hashlib.sha256(manifest_bytes).hexdigest()


def test_run_identity_rejects_changes_to_mode_policy_time_code_source_or_release() -> None:
    saved = {
        "run_id": "capture-1",
        "qwen_inference_enabled": False,
        "evaluation_scope": {"case_ids": ["case-a", "case-b"], "module_matrix_enabled": True},
        "as_of": "2026-10-26T00:00:00+00:00",
        "research_policy_sha256": "policy-a",
        "implementation_sha256": {"harness.py": "code-a"},
        "source_records": [{"source_id": "source-a", "source_sha256": "source-hash-a"}],
        "release": {"release_id": "capture-1", "spec_sha256": "release-a"},
        "access_scope": {"principal_id": "local-public", "permission_revision": 3, "continuation_revision": 7},
    }

    for key, changed_value in (
        ("qwen_inference_enabled", True),
        ("evaluation_scope", {"case_ids": ["case-a"], "module_matrix_enabled": False}),
        ("as_of", "2026-10-27T00:00:00+00:00"),
        ("research_policy_sha256", "policy-b"),
        ("implementation_sha256", {"harness.py": "code-b"}),
        ("source_records", [{"source_id": "source-a", "source_sha256": "source-hash-b"}]),
        ("release", {"release_id": "capture-1", "spec_sha256": "release-b"}),
        ("access_scope", {"principal_id": "local-public", "permission_revision": 4, "continuation_revision": 7}),
    ):
        current = dict(saved)
        current[key] = changed_value
        with pytest.raises(capture.RunIdentityError, match="run identity mismatch"):
            capture._verify_run_identity(saved, current)


def test_run_identity_and_each_case_bind_live_permission_and_continuation_revisions(tmp_path) -> None:
    class Store:
        permission = 3
        continuation = 7

        def release_summaries(self):
            return {"active_release_id": "capture-1", "releases": [{"release_id": "capture-1", "status": "published"}]}

        def release_spec(self, _release_id):
            return {}, "release-spec-hash"

        def release_versions(self, _release_id):
            return ["version-1"]

        def permission_revision(self):
            return self.permission

        def continuation_revision(self):
            return self.continuation

        def read_facts(self, _version_ids):
            return []

    source_path = tmp_path / "source.pdf"
    source_path.write_bytes(b"reviewed source")
    manifest_path = tmp_path / "source-manifest.json"
    manifest_path.write_text("{}", encoding="utf-8")
    docling_path = tmp_path / "docling-assets.json"
    docling_path.write_text("{}", encoding="utf-8")
    qwen_path = tmp_path / "qwen-assets.json"
    qwen_path.write_text("{}", encoding="utf-8")
    sources = [{
        "case_id": "case-1",
        "source_id": "source-1",
        "source_sha256": capture._sha256(source_path.read_bytes()),
        "version_id": "version-1",
        "first_available_at_used_for_current_capture": "2026-10-26T00:00:00+00:00",
        "source_path": str(source_path),
    }]
    store = Store()

    def build_identity():
        return capture._build_run_identity(
            run_id="capture-1",
            as_of=datetime(2026, 10, 26, tzinfo=timezone.utc),
            use_qwen=False,
            source_manifest_path=manifest_path,
            docling_manifest_path=docling_path,
            qwen_manifest_path=qwen_path,
            source_records=sources,
            store=store,
        )

    saved = build_identity()
    assert saved["evaluation_scope"] == {
        "case_ids": sorted(str(spec["case_id"]) for spec in capture.CASE_SPECS),
        "module_matrix_enabled": True,
    }
    assert saved["access_scope"] == {
        "principal_id": "local-public",
        "permission_revision": 3,
        "continuation_revision": 7,
    }
    case = capture._case_identity_fields(
        "run-hash",
        {"case_id": "case-1", "security_code": "600036"},
        sources[0],
        as_of=datetime(2026, 10, 26, tzinfo=timezone.utc),
        retrieval_policy_sha256="policy-hash",
        access_scope=saved["access_scope"],
    )
    assert case["principal_id"] == "local-public"
    assert case["permission_revision"] == 3
    assert case["continuation_revision"] == 7

    store.permission = 4
    with pytest.raises(capture.RunIdentityError, match="access_scope"):
        capture._verify_run_identity(saved, build_identity())
    store.permission = 3
    store.continuation = 8
    with pytest.raises(capture.RunIdentityError, match="access_scope"):
        capture._verify_run_identity(saved, build_identity())


def test_narrow_evaluation_scope_is_bound_into_run_identity() -> None:
    saved = {
        "run_id": "capture-1",
        "evaluation_scope": {"case_ids": ["case-a", "case-b"], "module_matrix_enabled": True},
    }
    changed = {
        **saved,
        "evaluation_scope": {"case_ids": ["case-a"], "module_matrix_enabled": False},
    }
    with pytest.raises(capture.RunIdentityError, match="run identity mismatch: evaluation_scope"):
        capture._verify_run_identity(saved, changed)


def test_query_progress_accepts_same_mode_results_and_rejects_tampering_or_model_change() -> None:
    case = _case_identity()
    progress = _query_progress(case)
    accepted = capture._validate_query_progress(
        progress,
        run_identity_sha256="run-hash",
        runtime_model_identity_sha256="runtime-hash",
        case_identity_by_id={"case-1": case},
    )
    assert accepted[0]["source_id"] == "source-1"

    tampered_source = json.loads(json.dumps(progress))
    tampered_source["cases"][0]["source_id"] = "source-other"
    with pytest.raises(ValueError, match="case identity mismatch"):
        capture._validate_query_progress(
            tampered_source,
            run_identity_sha256="run-hash",
            runtime_model_identity_sha256="runtime-hash",
            case_identity_by_id={"case-1": case},
        )

    tampered_value = json.loads(json.dumps(progress))
    tampered_value["cases"][0]["result"]["source_values"][1] = "146695"
    with pytest.raises(ValueError, match="result was modified"):
        capture._validate_query_progress(
            tampered_value,
            run_identity_sha256="run-hash",
            runtime_model_identity_sha256="runtime-hash",
            case_identity_by_id={"case-1": case},
        )

    with pytest.raises(ValueError, match="runtime model identity"):
        capture._validate_query_progress(
            progress,
            run_identity_sha256="run-hash",
            runtime_model_identity_sha256="different-runtime",
            case_identity_by_id={"case-1": case},
        )


def test_per_case_progress_requires_matching_identity_and_an_intact_tool_trace(tmp_path) -> None:
    identity_fields = {
        "case_id": "case-1",
        "run_identity_sha256": "run-hash",
        "runtime_model_identity_sha256": "runtime-hash",
        "case_identity_sha256": "case-hash",
        "progress_kind": "agent",
        "research_ablation_id": None,
    }
    attempts = [{"status": "completed", "tool_events": [{"tool": "search", "result": {"retrieved": ["fact-1"]}}]}]
    progress = {
        **identity_fields,
        "attempts": attempts,
        "attempts_sha256": capture._canonical_hash(attempts),
        "completed_result": {"source_reads": [{"source_id": "source-1", "raw_value": "337488"}]},
    }
    progress["completed_result_sha256"] = capture._canonical_hash(progress["completed_result"])
    path = tmp_path / "agent-progress.json"
    path.write_text(json.dumps(progress), encoding="utf-8")
    assert capture._existing_progress_record(path, identity_fields) == progress

    progress["attempts"][0]["tool_events"][0]["result"]["retrieved"][0] = "tampered-fact"
    path.write_text(json.dumps(progress), encoding="utf-8")
    with pytest.raises(ValueError, match="integrity check"):
        capture._existing_progress_record(path, identity_fields)

    with pytest.raises(ValueError, match="identity mismatch"):
        capture._existing_progress_record(path, {**identity_fields, "runtime_model_identity_sha256": "other-runtime"})


def test_retrieval_trace_journal_is_append_only_identity_bound_and_integrity_checked(tmp_path) -> None:
    path = tmp_path / "retrieval_trace.jsonl"
    journal = capture.RetrievalTraceJournal(
        path,
        run_identity_sha256="run-hash",
        runtime_model_identity_sha256="runtime-hash",
    )
    first = journal.append(
        {"case_id": "case-1", "tool_event_index": 0},
        {"stage": "channel_completed", "candidates": [{"block_id": "block-1", "rank": 1}]},
    )
    second = journal.append(
        {"case_id": "case-1", "tool_event_index": 0},
        {"stage": "search_failed", "error": {"type": "TimeoutError", "message": "stopped"}},
    )
    assert first["sequence"] == 1
    assert second["sequence"] == 2

    resumed = capture.RetrievalTraceJournal(
        path,
        run_identity_sha256="run-hash",
        runtime_model_identity_sha256="runtime-hash",
    )
    assert resumed.sequence == 2
    assert resumed.previous_event_sha256 == second["event_sha256"]
    resumed.verify_references(
        [
            {"sequence": first["sequence"], "event_sha256": first["event_sha256"]},
            {"sequence": second["sequence"], "event_sha256": second["event_sha256"]},
        ],
        require_nonempty=True,
    )
    with pytest.raises(capture.RunIdentityError, match="model identity"):
        capture.RetrievalTraceJournal(
            path,
            run_identity_sha256="run-hash",
            runtime_model_identity_sha256="other-runtime",
        )

    lines = path.read_text(encoding="utf-8").splitlines()
    tampered = json.loads(lines[1])
    tampered["event"]["candidates"][0]["rank"] = 3
    lines[1] = json.dumps(tampered, ensure_ascii=False)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(capture.RunIdentityError, match="event chain"):
        capture.RetrievalTraceJournal(
            path,
            run_identity_sha256="run-hash",
            runtime_model_identity_sha256="runtime-hash",
        )


def test_truncated_trace_tail_cannot_back_a_completed_case_or_module_cache(tmp_path) -> None:
    path = tmp_path / "retrieval_trace.jsonl"
    journal = capture.RetrievalTraceJournal(
        path,
        run_identity_sha256="run-hash",
        runtime_model_identity_sha256="runtime-hash",
    )
    first = journal.append({"case_id": "case-1"}, {"stage": "search_started"})
    second = journal.append({"case_id": "case-1"}, {"stage": "search_completed"})
    case_progress = {
        "retrieval_trace_records": [
            {"sequence": first["sequence"], "event_sha256": first["event_sha256"]},
            {"sequence": second["sequence"], "event_sha256": second["event_sha256"]},
        ]
    }
    module_progress = {
        "results": {
            "12": {
                "retrieval_trace_records": [
                    {"sequence": second["sequence"], "event_sha256": second["event_sha256"]}
                ]
            }
        }
    }
    lines = path.read_bytes().splitlines(keepends=True)
    path.write_bytes(b"".join(lines[:-1]))
    truncated = capture.RetrievalTraceJournal(
        path,
        run_identity_sha256="run-hash",
        runtime_model_identity_sha256="runtime-hash",
    )
    with pytest.raises(capture.RunIdentityError, match="missing, changed, or reordered"):
        truncated.verify_references_in(case_progress)
    with pytest.raises(capture.RunIdentityError, match="missing, changed, or reordered"):
        truncated.verify_references_in(module_progress)


def test_progress_header_rejects_legacy_and_cross_mode_progress_before_runtime_setup() -> None:
    with pytest.raises(ValueError, match="legacy schema"):
        capture._validate_query_progress_header(
            [{"case_id": "case-1"}],
            run_identity_sha256="run-hash",
            runtime_model_identity_sha256=None,
        )

    progress = _query_progress(_case_identity())
    with pytest.raises(ValueError, match="run identity"):
        capture._validate_query_progress_header(
            progress,
            run_identity_sha256="different-mode-run-hash",
            runtime_model_identity_sha256="runtime-hash",
        )


def test_legacy_resume_fails_before_model_construction(tmp_path, monkeypatch) -> None:
    run_dir = tmp_path / "legacy"
    run_dir.mkdir()
    (run_dir / "industrial_rag.sqlite3").touch()
    monkeypatch.setattr(sys, "argv", [
        "run_financial_rag_current_capture.py",
        "--run-id", "legacy",
        "--output-root", str(tmp_path),
        "--resume-existing",
    ])
    monkeypatch.setattr(capture, "Qwen3EmbeddingRuntime", lambda *_args, **_kwargs: pytest.fail("model runtime constructed"))
    monkeypatch.setattr(capture, "Qwen3RerankerRuntime", lambda *_args, **_kwargs: pytest.fail("model runtime constructed"))

    with pytest.raises(ValueError, match="run_identity.json"):
        capture.main()


def test_cross_mode_resume_fails_before_model_construction(tmp_path, monkeypatch) -> None:
    run_dir = tmp_path / "capture-1"
    run_dir.mkdir()
    (run_dir / "industrial_rag.sqlite3").touch()
    run_identity = {
        "run_id": "capture-1",
        "as_of": datetime(2026, 10, 26, tzinfo=timezone.utc).isoformat(),
        "qwen_inference_enabled": True,
    }
    (run_dir / "run_identity.json").write_text(json.dumps({
        "schema_version": "raama.financial-rag.current-capture.run-identity-envelope.v1",
        "run_identity": run_identity,
        "run_identity_sha256": capture._canonical_hash(run_identity),
        "runtime_model_identity": None,
        "runtime_model_identity_sha256": None,
    }), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", [
        "run_financial_rag_current_capture.py",
        "--run-id", "capture-1",
        "--output-root", str(tmp_path),
        "--resume-existing",
        "--skip-qwen",
    ])
    monkeypatch.setattr(capture, "Qwen3EmbeddingRuntime", lambda *_args, **_kwargs: pytest.fail("model runtime constructed"))
    monkeypatch.setattr(capture, "Qwen3RerankerRuntime", lambda *_args, **_kwargs: pytest.fail("model runtime constructed"))

    with pytest.raises(capture.RunIdentityError, match="mode differs"):
        capture.main()


def test_resume_scope_mismatch_fails_before_model_construction(tmp_path, monkeypatch) -> None:
    run_dir = tmp_path / "capture-1"
    run_dir.mkdir()
    (run_dir / "industrial_rag.sqlite3").touch()
    run_identity = {
        "run_id": "capture-1",
        "as_of": datetime(2026, 10, 26, tzinfo=timezone.utc).isoformat(),
        "qwen_inference_enabled": False,
        "evaluation_scope": {"case_ids": ["another-case"], "module_matrix_enabled": False},
    }
    (run_dir / "run_identity.json").write_text(json.dumps({
        "schema_version": "raama.financial-rag.current-capture.run-identity-envelope.v1",
        "run_identity": run_identity,
        "run_identity_sha256": capture._canonical_hash(run_identity),
        "runtime_model_identity": None,
        "runtime_model_identity_sha256": None,
    }), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", [
        "run_financial_rag_current_capture.py",
        "--run-id", "capture-1",
        "--output-root", str(tmp_path),
        "--resume-existing",
        "--skip-qwen",
        "--case-id", str(capture.CASE_SPECS[0]["case_id"]),
        "--skip-module-matrix",
    ])
    monkeypatch.setattr(capture, "Qwen3EmbeddingRuntime", lambda *_args, **_kwargs: pytest.fail("model runtime constructed"))
    with pytest.raises(capture.RunIdentityError, match="evaluation scope differs"):
        capture.main()
