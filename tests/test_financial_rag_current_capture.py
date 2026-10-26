from __future__ import annotations

import json
import sys
from datetime import datetime, timezone

import pytest

from scripts import run_financial_rag_current_capture as capture


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


def test_run_identity_rejects_changes_to_mode_policy_time_code_source_or_release() -> None:
    saved = {
        "run_id": "capture-1",
        "qwen_inference_enabled": False,
        "as_of": "2026-10-26T00:00:00+00:00",
        "research_policy_sha256": "policy-a",
        "implementation_sha256": {"harness.py": "code-a"},
        "source_records": [{"source_id": "source-a", "source_sha256": "source-hash-a"}],
        "release": {"release_id": "capture-1", "spec_sha256": "release-a"},
    }

    for key, changed_value in (
        ("qwen_inference_enabled", True),
        ("as_of", "2026-10-27T00:00:00+00:00"),
        ("research_policy_sha256", "policy-b"),
        ("implementation_sha256", {"harness.py": "code-b"}),
        ("source_records", [{"source_id": "source-a", "source_sha256": "source-hash-b"}]),
        ("release", {"release_id": "capture-1", "spec_sha256": "release-b"}),
    ):
        current = dict(saved)
        current[key] = changed_value
        with pytest.raises(capture.RunIdentityError, match="run identity mismatch"):
            capture._verify_run_identity(saved, current)


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
