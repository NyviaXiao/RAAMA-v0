"""Run one generation-driven Qwen financial agent against a bound local RAG release."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from adaptive_mas.agents import harness as agent_harness  # noqa: E402
from adaptive_mas.agents.harness import (  # noqa: E402
    CallBudget,
    DEFAULT_MODEL,
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
from adaptive_mas.industrial_rag.object_store import LocalContentAddressedStore, ObjectStoreError  # noqa: E402
from adaptive_mas.industrial_rag.repository import IndustrialRagStore  # noqa: E402
from scripts import run_financial_rag_current_capture as capture  # noqa: E402


SOURCE_RUN_ID = "current-capture-qwen-cmb-traced-20261026"
SOURCE_RUN_IDENTITY_SHA256 = "8f14cdc0760513c125b35fb057b7ab0caa661e03f11fedc3a19ef3bd8ad89b9d"
SOURCE_RUNTIME_MODEL_IDENTITY_SHA256 = "1a9250a6bbdbed5140e6c99868c2954bbf47d2b5390146fc85d54aeda2e6125a"
SOURCE_CASE_IDENTITY_SHA256 = "b0e3da8fbbe19b9a90eca7969b0b18d0bb6cb46db553edd2f77778dcc65a9899"
SOURCE_RUN_DIR = ROOT / ".tmp" / "industrial_rag_current_capture_20261026" / SOURCE_RUN_ID
LIVE_OUTPUT_ROOT = ROOT / ".tmp" / "industrial_rag_live_agent_20261027"
MAX_PROVIDER_CALLS = 5
MAX_LOGICAL_TURNS = 5
RUN_BUNDLE_SCHEMA = "raama.financial-rag.live-agent-run-bundle.v1"
DEFAULT_RUN_ID = "current-capture-qwen38-cmb-live-agent-20261027"


@dataclass(frozen=True, slots=True)
class LiveAgentRunBundle:
    source_run_id: str
    source_run_dir: Path
    source_manifest_path: Path
    source_manifest_sha256: str
    case_id: str
    case_specs: tuple[dict[str, Any], ...]
    report_periods: tuple[str, ...]
    expected_run_identity_sha256: str
    expected_runtime_model_identity_sha256: str
    expected_case_identity_sha256: str
    bundle_sha256: str
    bundle_path: Path | None = None
    dense_cache_path: Path | None = None

    def verify_immutable_inputs(self) -> None:
        if self.bundle_path is not None and capture._sha256(self.bundle_path.read_bytes()) != self.bundle_sha256:
            raise capture.RunIdentityError("live-agent run bundle changed after its expected hash was checked")
        if capture._sha256(self.source_manifest_path.read_bytes()) != self.source_manifest_sha256:
            raise capture.RunIdentityError("source manifest raw bytes differ from the run-bundle hash")


def _default_run_bundle() -> LiveAgentRunBundle:
    manifest_sha256 = capture._sha256(capture.SOURCE_MANIFEST.read_bytes())
    case_ids = [str(spec["case_id"]) for spec in capture.CASE_SPECS]
    default_case_id = "current_capture_cmb_2024_revenue"
    if default_case_id not in case_ids and len(case_ids) == 1:
        default_case_id = case_ids[0]
    identity_lock = {
        "schema_version": RUN_BUNDLE_SCHEMA,
        "source_run_id": SOURCE_RUN_ID,
        "source_manifest_sha256": manifest_sha256,
        "expected_run_identity_sha256": SOURCE_RUN_IDENTITY_SHA256,
        "expected_runtime_model_identity_sha256": SOURCE_RUNTIME_MODEL_IDENTITY_SHA256,
        "expected_case_identity_sha256": SOURCE_CASE_IDENTITY_SHA256,
    }
    return LiveAgentRunBundle(
        source_run_id=SOURCE_RUN_ID,
        source_run_dir=SOURCE_RUN_DIR,
        source_manifest_path=capture.SOURCE_MANIFEST,
        source_manifest_sha256=manifest_sha256,
        case_id=default_case_id,
        case_specs=tuple(dict(spec) for spec in capture.CASE_SPECS),
        report_periods=("2023", "2024"),
        expected_run_identity_sha256=SOURCE_RUN_IDENTITY_SHA256,
        expected_runtime_model_identity_sha256=SOURCE_RUNTIME_MODEL_IDENTITY_SHA256,
        expected_case_identity_sha256=SOURCE_CASE_IDENTITY_SHA256,
        bundle_sha256=capture._canonical_hash(identity_lock),
        dense_cache_path=ROOT / ".tmp" / "industrial_rag_dense_vector_cache.sqlite3",
    )


def _load_run_bundle(path: Path, expected_bundle_sha256: str) -> LiveAgentRunBundle:
    if not capture.re.fullmatch(r"[0-9a-f]{64}", expected_bundle_sha256):
        raise ValueError("expected bundle SHA-256 must be 64 lowercase hexadecimal characters")
    bundle_path = path.resolve(strict=True)
    raw_bundle = bundle_path.read_bytes()
    bundle_sha256 = capture._sha256(raw_bundle)
    if bundle_sha256 != expected_bundle_sha256:
        raise capture.RunIdentityError("live-agent run bundle does not match the caller-supplied expected hash")
    try:
        value = json.loads(raw_bundle.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise capture.RunIdentityError("live-agent run bundle is not valid UTF-8 JSON") from error
    required_keys = {
        "schema_version", "source_run_id", "source_run_dir", "source_manifest_path",
        "source_manifest_sha256", "case_id", "case_specs", "report_periods",
        "expected_run_identity_sha256", "expected_runtime_model_identity_sha256",
        "expected_case_identity_sha256",
    }
    if not isinstance(value, dict) or set(value) != required_keys or value.get("schema_version") != RUN_BUNDLE_SCHEMA:
        raise capture.RunIdentityError("live-agent run bundle has an unsupported schema or field set")
    for field in ("source_run_id", "case_id"):
        if not isinstance(value[field], str) or not capture.re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}", value[field]):
            raise capture.RunIdentityError(f"live-agent run bundle {field} is not a safe identifier")
    hashes = (
        "source_manifest_sha256", "expected_run_identity_sha256",
        "expected_runtime_model_identity_sha256", "expected_case_identity_sha256",
    )
    if any(not isinstance(value[field], str) or not capture.re.fullmatch(r"[0-9a-f]{64}", value[field]) for field in hashes):
        raise capture.RunIdentityError("live-agent run bundle contains an invalid SHA-256 identity")
    if not isinstance(value["source_run_dir"], str) or not value["source_run_dir"]:
        raise capture.RunIdentityError("live-agent run bundle source_run_dir is required")
    if not isinstance(value["source_manifest_path"], str) or not value["source_manifest_path"]:
        raise capture.RunIdentityError("live-agent run bundle source_manifest_path is required")
    if (
        not isinstance(value["case_specs"], list)
        or not value["case_specs"]
        or any(not isinstance(spec, dict) for spec in value["case_specs"])
        or len({str(spec.get("case_id")) for spec in value["case_specs"]}) != len(value["case_specs"])
        or value["case_id"] not in {str(spec.get("case_id")) for spec in value["case_specs"]}
    ):
        raise capture.RunIdentityError("live-agent run bundle case registry is missing, duplicated, or incomplete")
    periods = value["report_periods"]
    if (
        not isinstance(periods, list) or not periods
        or any(not isinstance(period, str) or not period.strip() for period in periods)
        or len(periods) != len(set(periods))
    ):
        raise capture.RunIdentityError("live-agent run bundle report periods must be unique nonempty strings")
    if tuple(periods) != tuple(capture.REPORT_PERIODS):
        raise capture.RunIdentityError("live-agent report validation currently requires periods 2023 and 2024")

    def resolve_input(raw_path: str, label: str, *, directory: bool) -> Path:
        candidate = Path(raw_path)
        if not candidate.is_absolute():
            candidate = bundle_path.parent / candidate
        resolved = candidate.resolve(strict=True)
        if directory and not resolved.is_dir():
            raise capture.RunIdentityError(f"live-agent run bundle {label} is not a directory")
        if not directory and not resolved.is_file():
            raise capture.RunIdentityError(f"live-agent run bundle {label} is not a file")
        return resolved

    source_run_dir = resolve_input(value["source_run_dir"], "source_run_dir", directory=True)
    source_manifest_path = resolve_input(value["source_manifest_path"], "source_manifest_path", directory=False)
    bundle = LiveAgentRunBundle(
        source_run_id=value["source_run_id"],
        source_run_dir=source_run_dir,
        source_manifest_path=source_manifest_path,
        source_manifest_sha256=value["source_manifest_sha256"],
        case_id=value["case_id"],
        case_specs=tuple(dict(spec) for spec in value["case_specs"]),
        report_periods=tuple(periods),
        expected_run_identity_sha256=value["expected_run_identity_sha256"],
        expected_runtime_model_identity_sha256=value["expected_runtime_model_identity_sha256"],
        expected_case_identity_sha256=value["expected_case_identity_sha256"],
        bundle_sha256=bundle_sha256,
        bundle_path=bundle_path,
    )
    bundle.verify_immutable_inputs()
    return bundle


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


def _load_provider_client(model: str = DEFAULT_MODEL) -> QwenClient:
    api_key = os.environ.get("POPAGENT_LLM_KEY")
    if not api_key:
        key_path = ROOT / "data" / "raw" / "qwen_key.txt"
        if not key_path.is_file():
            raise RuntimeError("the existing Qwen client has no configured credential")
        api_key = key_path.read_text(encoding="utf-8").strip()
    if not api_key:
        raise RuntimeError("the existing Qwen client credential is empty")
    return QwenClient.from_environment(api_key=api_key, model_override=model)


def _verify_capture_case_identity(
    envelope: dict[str, Any],
    baseline_progress: dict[str, Any],
    spec: dict[str, Any],
    store: IndustrialRagStore,
    bundle: LiveAgentRunBundle | None = None,
) -> dict[str, object]:
    bundle = bundle or _default_run_bundle()
    bundle.verify_immutable_inputs()
    run_identity = envelope.get("run_identity")
    if (
        envelope.get("schema_version") != "raama.financial-rag.current-capture.run-identity-envelope.v1"
        or not isinstance(run_identity, dict)
        or envelope.get("run_identity_sha256") != capture._canonical_hash(run_identity)
        or envelope.get("run_identity_sha256") != bundle.expected_run_identity_sha256
    ):
        raise capture.RunIdentityError("source capture run identity envelope is invalid")
    run_identity_sha256 = str(envelope["run_identity_sha256"])
    case_id = bundle.case_id
    if (
        run_identity.get("run_id") != bundle.source_run_id
        or run_identity.get("case_specs_sha256") != capture._canonical_hash(bundle.case_specs)
        or run_identity.get("source_manifest_sha256") != bundle.source_manifest_sha256
        or not isinstance(run_identity.get("query"), str)
        or not str(run_identity["query"]).strip()
        or run_identity.get("time_mode") != "strict_forward"
        or run_identity.get("evaluation_scope") != {"case_ids": [case_id], "module_matrix_enabled": False}
        or str(spec.get("case_id")) != case_id
    ):
        raise capture.RunIdentityError("source capture run binding differs from the reviewed case configuration")
    expected_raw_values = spec.get("expected_raw_values")
    if not isinstance(expected_raw_values, dict) or set(expected_raw_values) != set(bundle.report_periods):
        raise capture.RunIdentityError("run-bundle report periods differ from the reviewed source-value registry")

    policy_json = run_identity.get("research_policy_json")
    policy_sha256 = run_identity.get("research_policy_sha256")
    if not isinstance(policy_json, str) or policy_sha256 != capture._sha256(policy_json.encode("utf-8")):
        raise capture.RunIdentityError("source capture research policy binding is invalid")

    sources = run_identity.get("source_records")
    if not isinstance(sources, list) or len(sources) != 1 or not isinstance(sources[0], dict):
        raise capture.RunIdentityError("source capture must bind exactly one source record")
    source = dict(sources[0])
    if source.get("case_id") != case_id or source.get("source_id") != spec.get("source_id"):
        raise capture.RunIdentityError("source capture case is bound to a different source")
    source_sha256 = str(source.get("source_sha256", ""))
    version_id = str(source.get("version_id", ""))
    if not capture.re.fullmatch(r"[0-9a-f]{64}", source_sha256):
        raise capture.RunIdentityError("source capture original hash is invalid")
    try:
        source_payload = store.object_store.get(source_sha256)
    except (FileNotFoundError, ObjectStoreError) as error:
        raise capture.RunIdentityError("source capture original is unavailable from its content-addressed store") from error
    if capture._sha256(source_payload) != source_sha256:
        raise capture.RunIdentityError("source capture original bytes do not match the bound source hash")

    release = run_identity.get("release")
    if not isinstance(release, dict) or (
        release.get("release_id") != bundle.source_run_id
        or release.get("active_release_id") != bundle.source_run_id
        or store.active_release_id() != bundle.source_run_id
    ):
        raise capture.RunIdentityError("source capture release is not the pinned active release")
    _release_spec, release_spec_sha256 = store.release_spec(bundle.source_run_id)
    if release.get("spec_sha256") != release_spec_sha256:
        raise capture.RunIdentityError("source capture release specification hash changed")
    release_versions = store.release_versions(bundle.source_run_id)
    if release.get("version_ids") != release_versions or version_id not in release_versions:
        raise capture.RunIdentityError("source capture release versions differ from the bound source")
    release_summary = store.release_summaries()
    published_release = next(
        (item for item in release_summary["releases"] if item["release_id"] == bundle.source_run_id),
        None,
    )
    if (
        release_summary["active_release_id"] != bundle.source_run_id
        or published_release is None
        or published_release["status"] != "published"
    ):
        raise capture.RunIdentityError("source capture release is not published")

    source_rows = store.source_manifest([version_id])
    if len(source_rows) != 1:
        raise capture.RunIdentityError("source capture version is absent from the registered release")
    registered_source = source_rows[0]
    if (
        registered_source["source_id"] != source.get("source_id")
        or registered_source["content_sha256"] != source_sha256
        or registered_source["version_id"] != version_id
    ):
        raise capture.RunIdentityError("source capture identity differs from the registered original")
    if capture._indexed_fact_identity(store, [source]) != run_identity.get("indexed_financial_facts"):
        raise capture.RunIdentityError("source capture financial fact identity differs from the registered release")

    access_scope = run_identity.get("access_scope")
    if (
        not isinstance(access_scope, dict)
        or set(access_scope) != {"principal_id", "permission_revision", "continuation_revision"}
        or not isinstance(access_scope.get("principal_id"), str)
        or not access_scope["principal_id"]
    ):
        raise capture.RunIdentityError("source capture access scope is missing")
    if (
        store.permission_revision() != access_scope.get("permission_revision")
        or store.continuation_revision() != access_scope.get("continuation_revision")
    ):
        raise capture.RunIdentityError("source capture access or continuation revision has changed")

    as_of = datetime.fromisoformat(str(run_identity["as_of"]))
    case_identity = capture._case_identity_fields(
        run_identity_sha256,
        spec,
        source,
        as_of=as_of,
        retrieval_policy_sha256=str(policy_sha256),
        access_scope=access_scope,
        query=str(run_identity["query"]),
        report_periods=bundle.report_periods,
    )
    permitted = store.permitted_versions(
        str(access_scope["principal_id"]),
        str(spec["security_code"]),
        as_of,
        time_mode="strict_forward",
        report_periods=tuple(str(item) for item in case_identity["report_periods"]),
        accounting_basis=(str(spec["accounting_basis"]) if spec.get("accounting_basis") is not None else None),
        release_id=bundle.source_run_id,
    )
    if (
        len(permitted) != 1
        or permitted[0]["version_id"] != version_id
        or permitted[0]["source_id"] != source.get("source_id")
        or permitted[0]["content_sha256"] != source_sha256
        or permitted[0]["security_code"] != spec.get("security_code")
        or permitted[0]["accounting_basis"] != spec.get("accounting_basis")
        or permitted[0]["first_available_at"] != source.get("first_available_at")
    ):
        raise capture.RunIdentityError("source capture case does not resolve to its exact permitted release evidence")

    runtime_model_identity = envelope.get("runtime_model_identity")
    runtime_model_identity_sha256 = envelope.get("runtime_model_identity_sha256")
    if (
        not isinstance(runtime_model_identity, dict)
        or runtime_model_identity_sha256 != capture._canonical_hash(runtime_model_identity)
        or runtime_model_identity_sha256 != bundle.expected_runtime_model_identity_sha256
        or baseline_progress.get("runtime_model_identity_sha256") != runtime_model_identity_sha256
    ):
        raise capture.RunIdentityError("source capture runtime model identity is invalid")
    case_identity_sha256 = capture._canonical_hash(case_identity)
    if (
        case_identity_sha256 != bundle.expected_case_identity_sha256
        or baseline_progress.get("progress_kind") != "agent"
        or baseline_progress.get("research_ablation_id") is not None
        or baseline_progress.get("case_identity_sha256") != case_identity_sha256
        or any(baseline_progress.get(key) != value for key, value in case_identity.items())
    ):
        raise capture.RunIdentityError("local Qwen baseline progress does not match the bound source case identity")
    return {**case_identity, "case_identity_sha256": case_identity_sha256}


def _verify_runtime_access_revisions(
    store: IndustrialRagStore,
    case_identity: dict[str, object],
) -> tuple[int, int]:
    permission_revision = store.permission_revision()
    continuation_revision = store.continuation_revision()
    if (
        permission_revision != case_identity["permission_revision"]
        or continuation_revision != case_identity["continuation_revision"]
    ):
        raise capture.RunIdentityError("source access or continuation revision changed before the live agent request")
    return permission_revision, continuation_revision


def _verify_capture_run(
    bundle: LiveAgentRunBundle | None = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, object]]:
    bundle = bundle or _default_run_bundle()
    bundle.verify_immutable_inputs()
    if not bundle.source_run_dir.is_dir():
        raise FileNotFoundError(f"bound source capture is missing: {bundle.source_run_id}")
    envelope = capture._read_run_identity(bundle.source_run_dir / "run_identity.json")
    run_identity = dict(envelope["run_identity"])
    if run_identity.get("run_id") != bundle.source_run_id:
        raise capture.RunIdentityError("source capture run identity has a different run id")
    expected_case = next((item for item in bundle.case_specs if str(item["case_id"]) == bundle.case_id), None)
    if expected_case is None:
        raise capture.RunIdentityError("source capture case is absent from the run-bundle case registry")
    if run_identity.get("evaluation_scope", {}).get("case_ids") != [bundle.case_id]:
        raise capture.RunIdentityError("source capture is not bound to the run-bundle case")
    security_code = str(expected_case["security_code"])
    if not capture.re.fullmatch(r"\d{6}", security_code):
        raise capture.RunIdentityError("run-bundle case has an invalid security code")
    baseline_path = bundle.source_run_dir / f"agent_progress_{security_code}.json"
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
        bundle.source_run_dir / "retrieval_trace.jsonl",
        run_identity_sha256=str(envelope["run_identity_sha256"]),
        runtime_model_identity_sha256=str(envelope["runtime_model_identity_sha256"]),
    )
    attempts = baseline_progress.get("attempts")
    if not isinstance(attempts, list) or not attempts or not isinstance(attempts[-1], dict):
        raise capture.RunIdentityError("local Qwen baseline has no completed agent attempt")
    journal.verify_references(attempts[-1].get("retrieval_trace_records"), require_nonempty=True)
    journal.verify_references(completed.get("retrieval_trace_records"), require_nonempty=True)
    paired_path = bundle.source_run_dir / f"agent_progress_{security_code}_ablation12.json"
    if paired_path.is_file():
        paired_progress = json.loads(paired_path.read_text(encoding="utf-8"))
        if not isinstance(paired_progress, dict):
            raise capture.RunIdentityError("paired reranker progress is malformed")
        paired_attempts = paired_progress.get("attempts")
        if "attempts_sha256" in paired_progress and paired_progress.get("attempts_sha256") != capture._canonical_hash(paired_attempts):
            raise capture.RunIdentityError("paired reranker progress failed its attempt hash")
        if paired_attempts is not None:
            if not isinstance(paired_attempts, list):
                raise capture.RunIdentityError("paired reranker attempts are malformed")
            for paired_attempt in paired_attempts:
                if not isinstance(paired_attempt, dict):
                    raise capture.RunIdentityError("paired reranker attempt is malformed")
                if "retrieval_trace_records" in paired_attempt:
                    journal.verify_references(paired_attempt["retrieval_trace_records"])
    source_store = IndustrialRagStore(
        bundle.source_run_dir / "industrial_rag.sqlite3",
        LocalContentAddressedStore(bundle.source_run_dir / "objects"),
    )
    case_identity = _verify_capture_case_identity(envelope, baseline_progress, dict(expected_case), source_store, bundle)
    return envelope, baseline_progress, dict(expected_case), case_identity


def _load_service(
    source_envelope: dict[str, Any],
    bundle: LiveAgentRunBundle | None = None,
    *,
    qwen_assets_path: Path | None = None,
    dense_cache_path: Path,
) -> tuple[IndustrialRagStore, IndustrialRagService, dict[str, object]]:
    bundle = bundle or _default_run_bundle()
    bundle.verify_immutable_inputs()
    store = IndustrialRagStore(
        bundle.source_run_dir / "industrial_rag.sqlite3",
        LocalContentAddressedStore(bundle.source_run_dir / "objects"),
    )
    if store.active_release_id() != bundle.source_run_id:
        raise capture.RunIdentityError("the bound source release is no longer active in its local store")
    source_access_scope = source_envelope["run_identity"]["access_scope"]
    if (
        store.permission_revision() != source_access_scope["permission_revision"]
        or store.continuation_revision() != source_access_scope["continuation_revision"]
    ):
        raise capture.RunIdentityError("source access or continuation revision changed before model loading")
    assets_path = qwen_assets_path or capture.QWEN_ASSETS
    embedding = Qwen3EmbeddingRuntime(assets_path)
    reranker_runtime = Qwen3RerankerRuntime(assets_path)
    dense = LocalQwenDenseAdapter(
        store,
        embedding,
        cache_path=dense_cache_path,
        require_cached_vectors=True,
    )
    dense.verify_persistent_cache(store.release_versions(bundle.source_run_id))
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
    parser.add_argument("--run-id")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Exact configured Qwen model identifier")
    parser.add_argument("--run-bundle", type=Path, help="Versioned identity-locked JSON bundle for a source capture")
    parser.add_argument("--expected-bundle-sha256", help="Out-of-band SHA-256 pin for --run-bundle raw bytes")
    parser.add_argument("--output-root", type=Path, help="Parent directory for a new, non-overwriting live-agent run")
    parser.add_argument("--qwen-assets", type=Path, help="Pinned local Qwen asset directory; required with --run-bundle")
    parser.add_argument("--dense-cache", type=Path, help="Existing identity-checked dense-vector cache to reuse")
    args = parser.parse_args()
    if args.run_bundle is None:
        if args.expected_bundle_sha256 is not None:
            raise ValueError("expected-bundle-sha256 requires run-bundle")
        bundle = _default_run_bundle()
        run_id = args.run_id or DEFAULT_RUN_ID
    else:
        if args.expected_bundle_sha256 is None:
            raise ValueError("run-bundle requires an out-of-band expected-bundle-sha256")
        if args.run_id is None or args.output_root is None or args.qwen_assets is None:
            raise ValueError("portable run-bundle mode requires run-id, output-root, and qwen-assets")
        bundle = _load_run_bundle(args.run_bundle, args.expected_bundle_sha256)
        run_id = args.run_id
    if not capture.re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}", run_id):
        raise ValueError("run-id must be a safe immutable identifier")
    if args.model != "qwen3.8-27b":
        raise ValueError("this approved run requires the exact model identifier qwen3.8-27b")

    source_envelope, _baseline_progress, spec, case_identity = _verify_capture_run(bundle)
    dense_cache_source_path = capture._resolve_dense_cache_path(
        run_dir=bundle.source_run_dir,
        explicit_path=args.dense_cache or bundle.dense_cache_path,
    )
    output_root = args.output_root.resolve() if args.output_root is not None else LIVE_OUTPUT_ROOT
    output_dir = output_root / run_id
    if output_dir.resolve().is_relative_to(bundle.source_run_dir.resolve()):
        raise ValueError("live-agent output directory cannot be inside the immutable source run")
    output_dir.mkdir(parents=True, exist_ok=False)
    dense_cache_path = LocalQwenDenseAdapter.copy_persistent_cache(
        dense_cache_source_path,
        output_dir / "dense_vector_cache.sqlite3",
    )
    qwen_assets_path = args.qwen_assets or capture.QWEN_ASSETS
    store, service, runtime_identity = _load_service(
        source_envelope,
        bundle,
        qwen_assets_path=qwen_assets_path,
        dense_cache_path=dense_cache_path,
    )
    permission_revision, continuation_revision = _verify_runtime_access_revisions(store, case_identity)
    model_selected_at = datetime.now().astimezone().isoformat()
    provider_client = _load_provider_client(args.model)
    runtime_identity_sha256 = capture._canonical_hash(runtime_identity)
    source_run_identity = source_envelope["run_identity"]
    implementation_hashes = _implementation_hashes()
    live_identity: dict[str, object] = {
        "schema_version": "raama.financial-rag.live-agent-identity.v1",
        "run_id": run_id,
        "source_capture_run_id": bundle.source_run_id,
        "run_bundle_sha256": bundle.bundle_sha256,
        "source_manifest_sha256": bundle.source_manifest_sha256,
        "source_run_identity_sha256": source_envelope["run_identity_sha256"],
        "source_runtime_model_identity_sha256": source_envelope["runtime_model_identity_sha256"],
        "dense_vector_cache_source": str(dense_cache_source_path),
        "source_implementation_sha256": source_run_identity["implementation_sha256"],
        "case_identity_sha256": case_identity["case_identity_sha256"],
        "runtime_model_identity_sha256": runtime_identity_sha256,
        "agent_driver": "qwen_generation_tool_calls",
        "requested_model": provider_client.model,
        "model_selection": {
            "exact_model_identifier": args.model,
            "selection_source": "explicit --model argument",
            "selected_at": model_selected_at,
            "previous_preview_quota_status": "human-reported exhausted; not retried",
        },
        "fallback_policy": "disabled; preserve quota or availability failures",
        "max_provider_calls": MAX_PROVIDER_CALLS,
        "max_logical_turns": MAX_LOGICAL_TURNS,
        "implementation_sha256": implementation_hashes,
    }
    live_identity_sha256 = capture._canonical_hash(live_identity)
    _write_json(output_dir / "live_agent_identity.json", {
        "live_agent_identity": live_identity,
        "live_agent_identity_sha256": live_identity_sha256,
    })

    if provider_client.model != live_identity["requested_model"]:
        raise capture.RunIdentityError("configured provider model changed after live identity was bound")
    client = provider_client
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
    release_id = bundle.source_run_id
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
        bundle.verify_immutable_inputs()
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
        research_policy_json=str(source_run_identity["research_policy_json"]),
        retrieval_trace_callback=trace_callback,
    )
    calculator = handlers["calculate_financial_metric"]

    def submit_report(arguments: dict[str, object]) -> dict[str, object]:
        prepared = agent_harness._materialize_report_calculations(
            arguments,
            catalog,
            visible_evidence_ids,
            calculator,  # type: ignore[arg-type]
        )
        return capture._accept_report(
            prepared,
            catalog,
            "fundamental",
            visible_evidence_ids=visible_evidence_ids,
            decision_time=decision_time,
        )

    handlers["submit_report"] = submit_report
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
        "question": str(source_run_identity["query"]),
        "symbol": str(spec["security_code"]),
        "report_periods": list(bundle.report_periods),
        "period_instruction": (
            "Use these exact report_periods in searches and observations. Do not convert a year label to a calendar date or infer another period."
        ),
        "allowed_data": "the pinned local strict-forward industrial RAG release only",
    }
    final_result: dict[str, object] | None = None
    failure: dict[str, object] | None = None
    try:
        identity_check()
        final_result = _run_tool_agent(
            role_id="fundamental",
            system_prompt=(
                "Use the indexed local financial filing and cite every source fact. Use the exact report_periods "
                "provided in the task. For a calculation claim, provide calculation_spec with the operation and "
                "two cited source operands, or cite deterministic calculator evidence. Keep every numeric literal "
                "in the exact value and unit represented by source observations or calculated evidence."
            ),
            task=case,
            tools=tools,
            handlers=handlers,
            client=client,  # type: ignore[arg-type]
            budget=budget,
            max_turns=MAX_LOGICAL_TURNS,
            visible_evidence_ids=visible_evidence_ids,
            reserve_correction_turn=True,
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
        reported_models = sorted({
            str(row["provider_reported_model"])
            for row in budget.calls if row.get("provider_reported_model")
        })
        progress["provider_reported_models"] = reported_models
        progress["actual_model"] = reported_models[0] if len(reported_models) == 1 else None
    else:
        progress["status"] = "failed"
        progress["completed_at"] = datetime.now().astimezone().isoformat()
        progress["failure"] = failure
    progress["provider_budget"] = budget.summary()
    reported_models = sorted({
        str(row["provider_reported_model"])
        for row in budget.calls if row.get("provider_reported_model")
    })
    progress["provider_reported_models"] = reported_models
    progress["tool_count"] = len(progress["tool_events"])
    progress["retrieval_trace_event_count"] = journal.sequence
    progress["retrieval_trace_sha256"] = hashlib.sha256((output_dir / "retrieval_trace.jsonl").read_bytes()).hexdigest()
    progress["provider_request_ledger_sha256"] = hashlib.sha256((output_dir / "provider_request_ledger.jsonl").read_bytes()).hexdigest() if (output_dir / "provider_request_ledger.jsonl").is_file() else None
    _write_json(progress_path, progress)
    output_record: dict[str, object] = {
        "schema_version": "raama.financial-rag.live-agent-result.v1",
        "live_agent_identity_sha256": live_identity_sha256,
        "source_capture_run_id": bundle.source_run_id,
        "source_run_identity_sha256": source_envelope["run_identity_sha256"],
        "case_identity_sha256": case_identity["case_identity_sha256"],
        "runtime_model_identity_sha256": runtime_identity_sha256,
        "agent_driver": "qwen_generation_tool_calls",
        "generation_model_called": budget.provider_call_count > 0 or budget.cache_hits > 0,
        "requested_model": client.model,
        "actual_model": reported_models[0] if len(reported_models) == 1 else None,
        "provider_reported_models": reported_models,
        "fallback_used": False,
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
        "run_id": run_id,
        "status": output_record["status"],
        "agent_driver": output_record["agent_driver"],
        "generation_model_called": output_record["generation_model_called"],
        "requested_model": output_record["requested_model"],
        "actual_model": output_record["actual_model"],
        "provider_reported_models": reported_models,
        "fallback_used": False,
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
