"""Run a source-scoped, local-only current-capture RAG and tool-chain check."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Callable, Mapping

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from adaptive_mas.agents.harness import (  # noqa: E402
    SHANGHAI,
    InvalidResearchOutput,
    _accept_report,
    _fundamental_handlers,
    _run_tool_agent,
)
from adaptive_mas.industrial_rag.cli import build_release  # noqa: E402
from adaptive_mas.industrial_rag.document_ai import ASSET_MANIFEST  # noqa: E402
from adaptive_mas.industrial_rag.models import (  # noqa: E402
    Document,
    DocumentBundle,
    DocumentVersion,
    FinancialFact,
)
from adaptive_mas.harness_evolution import _ABLATION_MODULES  # noqa: E402
from adaptive_mas.industrial_rag.object_store import LocalContentAddressedStore  # noqa: E402
from adaptive_mas.industrial_rag.planning import accounting_basis_status  # noqa: E402
from adaptive_mas.industrial_rag.parser import (  # noqa: E402
    _table_row_group_blocks,
    annotate_table_continuation_candidates,
    parse_isolated,
    records_from_parse,
)
from adaptive_mas.industrial_rag.qwen_models import (  # noqa: E402
    EMBEDDING_REPO,
    EMBEDDING_REVISION,
    RERANKER_REPO,
    RERANKER_REVISION,
    LocalQwenDenseAdapter,
    LocalQwenReranker,
    Qwen3EmbeddingRuntime,
    Qwen3RerankerRuntime,
)
from adaptive_mas.industrial_rag.repository import IndustrialRagStore  # noqa: E402
from adaptive_mas.industrial_rag.retrieval import IndustrialRagService  # noqa: E402


SUPERVISION = ROOT / ".tmp" / "industrial_rag_supervision_20261025"
SOURCE_MANIFEST = ROOT / "benchmarks" / "financial_rag" / "seed_manifest.json"
DOCLING_ASSETS = SUPERVISION / "docling-models"
QWEN_ASSETS = SUPERVISION / "qwen-models"
OUTPUT_ROOT = ROOT / ".tmp" / "industrial_rag_current_capture_20261026"
QUERY_TEXT = "2024年度营业收入是多少，较2023年度增长多少？"
CAPTURE_PRINCIPAL_ID = "local-public"


class RunIdentityError(RuntimeError):
    pass


CASE_SPECS: tuple[dict[str, object], ...] = (
    {
        "case_id": "current_capture_cmb_2024_revenue",
        "source_id": "cninfo-1222895153",
        "filename": "cmb2024summary.pdf",
        "security_code": "600036",
        "physical_page": 3,
        "parser_table_id": "docling-table-1-p1",
        "row_index": 2,
        "source_metric_label": "营业收入",
        "expected_raw_values": {"2024": "337,488", "2023": "339,123"},
        "source_unit": "人民币百万元",
        "base_unit_conversion_factor": "1000000",
        "unit": "人民币百万元",
        "currency": "CNY",
        "accounting_basis": "本集团",
        "manual_review_note": "Page image labels the table unit as 人民币百万元 and the section as 本集团; row 营业收入, 2024 column is 337,488.",
    },
    {
        "case_id": "current_capture_pingan_2024_revenue",
        "source_id": "cninfo-1222806505",
        "filename": "pingan2024full.pdf",
        "security_code": "000001",
        "physical_page": 16,
        "parser_table_id": "docling-table-1-p1",
        "row_index": 1,
        "source_metric_label": "营业收入",
        "expected_raw_values": {"2024": "146,695", "2023": "164,699"},
        "source_unit": "人民币百万元",
        "base_unit_conversion_factor": "1000000",
        "unit": "人民币百万元",
        "currency": "CNY",
        "accounting_basis": None,
        "accounting_basis_status": "unknown",
        "manual_review_note": "Page image labels the table unit as 人民币百万元; row 营业收入, 2024 column is 146,695. The target table does not state its accounting basis.",
    },
    {
        "case_id": "current_capture_catl_2024_revenue",
        "source_id": "cninfo-1222806982",
        "filename": "catl2024full.pdf",
        "security_code": "300750",
        "physical_page": 9,
        "parser_table_id": "docling-table-2-p1",
        "row_index": 2,
        "source_metric_label": "营业收入（千元）",
        "expected_raw_values": {"2024": "362,012,554", "2023": "400,917,045"},
        "source_unit": "千元",
        "base_unit_conversion_factor": "1000",
        "unit": "千元",
        "currency": "",
        "accounting_basis": None,
        "accounting_basis_status": "unknown",
        "manual_review_note": "Page image labels the row unit as 千元; row 营业收入, 2024 column is 362,012,554. The target table does not state its currency or accounting basis.",
    },
)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary_path, path)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_hash(value: object) -> str:
    serialized = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return _sha256(serialized.encode("utf-8"))


class RetrievalTraceJournal:
    schema_version = "raama.financial-rag.retrieval-trace.v1"

    def __init__(self, path: Path, *, run_identity_sha256: str, runtime_model_identity_sha256: str) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.run_identity_sha256 = run_identity_sha256
        self.runtime_model_identity_sha256 = runtime_model_identity_sha256
        self.sequence = 0
        self.previous_event_sha256 = "0" * 64
        self.event_hashes: dict[int, str] = {}
        if self.path.exists():
            self._read_and_validate()
        else:
            header: dict[str, object] = {
                "record_type": "header",
                "schema_version": self.schema_version,
                "run_identity_sha256": run_identity_sha256,
                "runtime_model_identity_sha256": runtime_model_identity_sha256,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
            header["header_sha256"] = _canonical_hash(header)
            self._append_line(header)

    def _append_line(self, record: Mapping[str, object]) -> None:
        line = json.dumps(record, ensure_ascii=False, sort_keys=True, allow_nan=False).encode("utf-8") + b"\n"
        with self.path.open("ab") as output:
            output.write(line)
            output.flush()
            os.fsync(output.fileno())

    def _read_and_validate(self) -> None:
        payload = self.path.read_bytes()
        lines = payload.splitlines(keepends=True)
        if not lines or any(not line.endswith(b"\n") for line in lines):
            raise RunIdentityError("retrieval trace has an incomplete trailing record")
        header = json.loads(lines[0])
        header_body = {key: value for key, value in header.items() if key != "header_sha256"}
        if (
            header.get("record_type") != "header"
            or header.get("schema_version") != self.schema_version
            or header.get("run_identity_sha256") != self.run_identity_sha256
            or header.get("runtime_model_identity_sha256") != self.runtime_model_identity_sha256
            or header.get("header_sha256") != _canonical_hash(header_body)
        ):
            raise RunIdentityError("retrieval trace header differs from the current run or model identity")
        previous = "0" * 64
        for expected_sequence, raw_line in enumerate(lines[1:], start=1):
            record = json.loads(raw_line)
            body = {key: value for key, value in record.items() if key != "event_sha256"}
            if (
                record.get("record_type") != "event"
                or record.get("sequence") != expected_sequence
                or record.get("previous_event_sha256") != previous
                or record.get("event_sha256") != _canonical_hash(body)
            ):
                raise RunIdentityError("retrieval trace event chain failed integrity validation")
            previous = str(record["event_sha256"])
            self.event_hashes[expected_sequence] = previous
        self.sequence = len(lines) - 1
        self.previous_event_sha256 = previous

    def verify_references(
        self,
        references: object,
        *,
        require_nonempty: bool = False,
    ) -> None:
        if not isinstance(references, list) or (require_nonempty and not references):
            raise RunIdentityError("saved progress has no retrieval trace journal references")
        previous_sequence = 0
        for reference in references:
            if not isinstance(reference, dict):
                raise RunIdentityError("saved retrieval trace reference is malformed")
            sequence = reference.get("sequence")
            digest = reference.get("event_sha256")
            if (
                isinstance(sequence, bool)
                or not isinstance(sequence, int)
                or sequence < 1
                or sequence <= previous_sequence
                or not isinstance(digest, str)
                or self.event_hashes.get(sequence) != digest
            ):
                raise RunIdentityError("saved progress references a missing, changed, or reordered retrieval trace event")
            previous_sequence = sequence

    def verify_references_in(self, value: object) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                if key == "retrieval_trace_records":
                    self.verify_references(item)
                else:
                    self.verify_references_in(item)
        elif isinstance(value, list):
            for item in value:
                self.verify_references_in(item)

    def append(self, context: Mapping[str, object], event: Mapping[str, object]) -> dict[str, object]:
        body: dict[str, object] = {
            "record_type": "event",
            "sequence": self.sequence + 1,
            "previous_event_sha256": self.previous_event_sha256,
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            "context": dict(context),
            "event": dict(event),
        }
        record = {**body, "event_sha256": _canonical_hash(body)}
        self._append_line(record)
        self.sequence = int(record["sequence"])
        self.previous_event_sha256 = str(record["event_sha256"])
        self.event_hashes[self.sequence] = self.previous_event_sha256
        return record

    def references_between(self, first_sequence: int, last_sequence: int | None = None) -> list[dict[str, object]]:
        end = self.sequence if last_sequence is None else min(last_sequence, self.sequence)
        return [
            {"sequence": sequence, "event_sha256": self.event_hashes[sequence]}
            for sequence in range(first_sequence, end + 1)
        ]


def _case_calculation_reasons(spec: Mapping[str, Any]) -> list[str]:
    reasons = []
    if spec.get("currency") != "CNY":
        reasons.append("currency_not_stated_in_target_table")
    if accounting_basis_status(spec.get("accounting_basis")) != "known":
        reasons.append("accounting_basis_unknown_in_target_table")
    return reasons


def _case_calculation_eligible(spec: Mapping[str, Any]) -> bool:
    return not _case_calculation_reasons(spec)


def _implementation_hashes() -> dict[str, str]:
    paths = (
        Path(__file__).resolve(),
        ROOT / "src" / "adaptive_mas" / "agents" / "harness.py",
        ROOT / "src" / "adaptive_mas" / "industrial_rag" / "retrieval.py",
        ROOT / "src" / "adaptive_mas" / "industrial_rag" / "planning.py",
        ROOT / "src" / "adaptive_mas" / "industrial_rag" / "models.py",
        ROOT / "src" / "adaptive_mas" / "industrial_rag" / "parser.py",
        ROOT / "src" / "adaptive_mas" / "industrial_rag" / "qwen_models.py",
        ROOT / "src" / "adaptive_mas" / "harness_evolution.py",
    )
    return {path.relative_to(ROOT).as_posix(): _sha256(path.read_bytes()) for path in paths}


def _source_identity(source: Mapping[str, Any]) -> dict[str, object]:
    return {
        "case_id": source["case_id"],
        "source_id": source["source_id"],
        "source_sha256": source["source_sha256"],
        "version_id": source["version_id"],
        "first_available_at": source["first_available_at_used_for_current_capture"],
    }


def _indexed_fact_identity(store: IndustrialRagStore, sources: list[Mapping[str, Any]]) -> list[dict[str, object]]:
    version_ids = sorted({str(source["version_id"]) for source in sources})
    rows = [dict(row) for row in store.read_facts(version_ids)]
    rows.sort(key=lambda row: (str(row.get("version_id")), str(row.get("report_period")), str(row.get("fact_id"))))
    keys = (
        "fact_id", "version_id", "source_id", "security_code", "entity_name", "metric", "raw_value",
        "normalized_value", "unit", "multiplier", "currency", "report_period", "consolidation",
        "review_status", "confidence", "span_ids",
    )
    return [
        {
            key: (
                json.loads(str(row["payload_json"])).get("span_ids", [])
                if key == "span_ids" else row.get(key)
            )
            for key in keys
        }
        for row in rows
    ]


def _build_run_identity(
    *,
    run_id: str,
    as_of: datetime,
    use_qwen: bool,
    source_manifest_path: Path,
    docling_manifest_path: Path,
    qwen_manifest_path: Path,
    source_records: list[Mapping[str, Any]],
    store: IndustrialRagStore,
    case_ids: list[str] | None = None,
    module_matrix_enabled: bool = True,
) -> dict[str, object]:
    release_summary = store.release_summaries()
    release = next(
        (item for item in release_summary["releases"] if item["release_id"] == run_id),
        None,
    )
    if release is None or release_summary["active_release_id"] != run_id or release["status"] != "published":
        raise RuntimeError("run identity requires the published active source release")
    _release_spec, release_spec_sha256 = store.release_spec(run_id)
    for source in source_records:
        if _sha256(Path(str(source["source_path"])).read_bytes()) != source["source_sha256"]:
            raise RunIdentityError(f"current-capture source bytes changed: {source['source_id']}")
    policy_json = _policy_json(use_qwen=use_qwen)
    return {
        "schema_version": "raama.financial-rag.current-capture.run-identity.v1",
        "run_id": run_id,
        "as_of": as_of.astimezone(timezone.utc).isoformat(),
        "time_mode": "strict_forward",
        "query": QUERY_TEXT,
        "qwen_inference_enabled": use_qwen,
        "evaluation_scope": {
            "case_ids": sorted(case_ids) if case_ids is not None else sorted(str(spec["case_id"]) for spec in CASE_SPECS),
            "module_matrix_enabled": module_matrix_enabled,
        },
        "retrieval_channels": ["structured_facts", "bm25", "dense"] if use_qwen else ["structured_facts", "bm25"],
        "research_policy_json": policy_json,
        "research_policy_sha256": _sha256(policy_json.encode("utf-8")),
        "source_manifest_sha256": _sha256(source_manifest_path.read_bytes()),
        "source_records": [_source_identity(source) for source in source_records],
        "release": {
            "release_id": run_id,
            "spec_sha256": release_spec_sha256,
            "version_ids": sorted(store.release_versions(run_id)),
            "active_release_id": release_summary["active_release_id"],
        },
        "access_scope": {
            "principal_id": CAPTURE_PRINCIPAL_ID,
            "permission_revision": store.permission_revision(),
            "continuation_revision": store.continuation_revision(),
        },
        "indexed_financial_facts": _indexed_fact_identity(store, source_records),
        "docling_asset_manifest_sha256": _sha256(docling_manifest_path.read_bytes()),
        "model": {
            "embedding_repo": EMBEDDING_REPO,
            "embedding_revision": EMBEDDING_REVISION,
            "reranker_repo": RERANKER_REPO,
            "reranker_revision": RERANKER_REVISION,
            "asset_manifest_sha256": _sha256(qwen_manifest_path.read_bytes()),
            "local_files_only": True,
            "device": "CPU" if use_qwen else None,
        },
        "case_specs_sha256": _canonical_hash(CASE_SPECS),
        "module_catalog_sha256": _canonical_hash(_ABLATION_MODULES),
        "implementation_sha256": _implementation_hashes(),
    }


def _read_run_identity(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError("resume-existing requires a bound run_identity.json; legacy runs cannot be resumed")
    envelope = json.loads(path.read_text(encoding="utf-8"))
    identity = envelope.get("run_identity") if isinstance(envelope, dict) else None
    identity_sha256 = envelope.get("run_identity_sha256") if isinstance(envelope, dict) else None
    if (
        not isinstance(envelope, dict)
        or envelope.get("schema_version") != "raama.financial-rag.current-capture.run-identity-envelope.v1"
        or not isinstance(identity, dict)
        or identity_sha256 != _canonical_hash(identity)
    ):
        raise ValueError("saved run identity is missing or its hash is invalid")
    return envelope


def _verify_run_identity(saved: Mapping[str, Any], current: Mapping[str, Any]) -> None:
    if dict(saved) != dict(current):
        changed = sorted(key for key in set(saved) | set(current) if saved.get(key) != current.get(key))
        raise RunIdentityError("resume-existing run identity mismatch: " + ", ".join(changed))


def _case_identity_fields(
    run_identity_sha256: str,
    spec: Mapping[str, Any],
    source: Mapping[str, Any],
    *,
    as_of: datetime,
    retrieval_policy_sha256: str,
    access_scope: Mapping[str, object],
) -> dict[str, object]:
    return {
        "run_identity_sha256": run_identity_sha256,
        "case_id": str(spec["case_id"]),
        "security_code": str(spec["security_code"]),
        "source_id": str(source["source_id"]),
        "source_sha256": str(source["source_sha256"]),
        "version_id": str(source["version_id"]),
        "report_periods": ["2023", "2024"],
        "query": QUERY_TEXT,
        "as_of": as_of.astimezone(timezone.utc).isoformat(),
        "time_mode": "strict_forward",
        "retrieval_policy_sha256": retrieval_policy_sha256,
        "principal_id": access_scope["principal_id"],
        "permission_revision": access_scope["permission_revision"],
        "continuation_revision": access_scope["continuation_revision"],
    }


def _case_result_sha256(row: Mapping[str, Any]) -> str:
    return _canonical_hash({key: value for key, value in row.items() if key != "case_result_sha256"})


def _validate_query_progress(
    value: object,
    *,
    run_identity_sha256: str,
    runtime_model_identity_sha256: str,
    case_identity_by_id: Mapping[str, Mapping[str, object]],
) -> list[dict[str, Any]]:
    if not isinstance(value, dict) or value.get("schema_version") != "raama.financial-rag.current-capture.query-progress.v2":
        raise ValueError("query progress is unbound or uses a legacy schema; refusing to reuse completed cases")
    if value.get("run_identity_sha256") != run_identity_sha256:
        raise ValueError("query progress run identity does not match the active configuration")
    if value.get("runtime_model_identity_sha256") != runtime_model_identity_sha256:
        raise ValueError("query progress runtime model identity does not match the loaded model")
    rows = value.get("cases")
    if not isinstance(rows, list):
        raise ValueError("query progress cases must be an array")
    seen: set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("query progress contains a malformed case result")
        case_id = str(row.get("case_id"))
        expected_case = case_identity_by_id.get(case_id)
        if expected_case is None or case_id in seen:
            raise ValueError("query progress contains an unknown or duplicate case")
        seen.add(case_id)
        expected_case_identity = _canonical_hash(expected_case)
        if (
            row.get("run_identity_sha256") != run_identity_sha256
            or row.get("runtime_model_identity_sha256") != runtime_model_identity_sha256
            or row.get("case_identity_sha256") != expected_case_identity
            or any(row.get(key) != value for key, value in expected_case.items())
        ):
            raise ValueError(f"query progress case identity mismatch: {case_id}")
        if row.get("case_result_sha256") != _case_result_sha256(row):
            raise ValueError(f"query progress case result was modified: {case_id}")
    return rows


def _validate_query_progress_header(
    value: object,
    *,
    run_identity_sha256: str,
    runtime_model_identity_sha256: str | None,
) -> None:
    if not isinstance(value, dict) or value.get("schema_version") != "raama.financial-rag.current-capture.query-progress.v2":
        raise ValueError("query progress is unbound or uses a legacy schema; refusing to reuse completed cases")
    if value.get("run_identity_sha256") != run_identity_sha256:
        raise ValueError("query progress run identity does not match the active configuration")
    if value.get("runtime_model_identity_sha256") != runtime_model_identity_sha256:
        raise ValueError("query progress runtime model identity does not match the saved run environment")
    if not isinstance(value.get("cases"), list):
        raise ValueError("query progress cases must be an array")


def _runtime_model_identity(
    *,
    use_qwen: bool,
    dense_adapter: LocalQwenDenseAdapter | None,
    reranker: LocalQwenReranker | None,
) -> dict[str, object]:
    return {
        "qwen_inference_enabled": use_qwen,
        "embedding_cache_identity": dict(dense_adapter.cache_identity) if dense_adapter is not None else None,
        "reranker_cache_identity": dict(reranker.cache_identity) if reranker is not None else None,
    }


def _bind_runtime_model_identity(
    identity_path: Path,
    query_progress_path: Path,
    runtime_identity: Mapping[str, object],
    *,
    resuming: bool,
) -> str:
    envelope = _read_run_identity(identity_path)
    saved_runtime_identity = envelope.get("runtime_model_identity")
    progress_has_cases = False
    if query_progress_path.is_file():
        progress = json.loads(query_progress_path.read_text(encoding="utf-8"))
        if isinstance(progress, dict) and isinstance(progress.get("cases"), list):
            progress_has_cases = bool(progress["cases"])
    if saved_runtime_identity is None:
        if resuming and progress_has_cases:
            raise ValueError("completed case results exist without a bound runtime model identity")
        envelope["runtime_model_identity"] = dict(runtime_identity)
        envelope["runtime_model_identity_sha256"] = _canonical_hash(runtime_identity)
        _write_json(identity_path, envelope)
    elif (
        not isinstance(saved_runtime_identity, dict)
        or saved_runtime_identity != dict(runtime_identity)
        or envelope.get("runtime_model_identity_sha256") != _canonical_hash(runtime_identity)
    ):
        raise ValueError("loaded runtime model identity differs from the saved run environment")
    return _canonical_hash(runtime_identity)


def _resume_attempts(progress_path: Path, identity_fields: Mapping[str, object]) -> list[dict[str, Any]]:
    if not progress_path.is_file():
        return []
    saved = json.loads(progress_path.read_text(encoding="utf-8"))
    if not isinstance(saved, dict) or any(saved.get(key) != value for key, value in identity_fields.items()):
        raise ValueError("per-case progress identity mismatch; refusing to overwrite its trace")
    attempts = saved.get("attempts")
    if not isinstance(attempts, list):
        raise ValueError("per-case progress has no attempt history")
    return [dict(attempt) for attempt in attempts]


def _existing_progress_record(progress_path: Path, identity_fields: Mapping[str, object]) -> dict[str, Any] | None:
    if not progress_path.is_file():
        return None
    saved = json.loads(progress_path.read_text(encoding="utf-8"))
    if not isinstance(saved, dict) or any(saved.get(key) != value for key, value in identity_fields.items()):
        raise ValueError("progress identity mismatch; refusing to reuse or overwrite its trace")
    if "attempts" in saved and saved.get("attempts_sha256") != _canonical_hash(saved.get("attempts")):
        raise ValueError("saved tool-attempt trace failed its integrity check")
    if (
        "completed_result" in saved
        and saved.get("completed_result_sha256") != _canonical_hash(saved.get("completed_result"))
    ):
        raise ValueError("saved completed case result failed its integrity check")
    if "results" in saved and saved.get("results_sha256") != _canonical_hash(saved.get("results")):
        raise ValueError("saved module results failed their integrity check")
    if (
        "active_module_period_results" in saved
        and saved.get("active_module_period_results_sha256") != _canonical_hash(saved.get("active_module_period_results"))
    ):
        raise ValueError("saved partial module results failed their integrity check")
    return saved


def _read_source_records(path: Path = SOURCE_MANIFEST) -> dict[str, dict[str, Any]]:
    manifest_path = path if path.is_absolute() else ROOT / path
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = manifest.get("source_records")
    if not isinstance(records, list):
        raise ValueError("source manifest has no source record list")
    return {
        str(record["source_id"]): dict(record)
        for record in records
        if isinstance(record, dict) and isinstance(record.get("source_id"), str)
    }


def _page_subset(payload: bytes, page_number: int) -> bytes:
    from io import BytesIO
    from pypdf import PdfReader, PdfWriter

    reader = PdfReader(BytesIO(payload))
    if page_number < 1 or page_number > len(reader.pages):
        raise ValueError("target physical page is outside the source PDF")
    writer = PdfWriter()
    writer.add_page(reader.pages[page_number - 1])
    output = BytesIO()
    writer.write(output)
    return output.getvalue()


def _map_targeted_page(
    full_parse: Mapping[str, Any],
    targeted_parse: Mapping[str, Any],
    *,
    physical_page: int,
    parser_table_id: str,
    unit: str,
    accounting_basis: str | None,
    currency: str,
) -> dict[str, Any]:
    tables = [item for item in targeted_parse.get("tables", []) if item.get("table_id") == parser_table_id]
    if len(tables) != 1:
        raise ValueError(f"expected one manually scoped Docling table {parser_table_id!r}")
    raw_table = dict(tables[0])
    mapped_table_id = re.sub(r"-p\d+$", f"-p{physical_page}", parser_table_id)
    raw_table.update({
        "table_id": mapped_table_id,
        "page_start": physical_page,
        "page_end": physical_page,
        "unit_text": unit,
        "accounting_basis": accounting_basis,
    })
    selected_cells: list[dict[str, Any]] = []
    for raw_cell in targeted_parse.get("cells", []):
        if raw_cell.get("table_id") != parser_table_id:
            continue
        cell = dict(raw_cell)
        cell["cell_id"] = str(cell["cell_id"]).replace(parser_table_id, mapped_table_id, 1)
        cell["table_id"] = mapped_table_id
        cell["page_number"] = physical_page
        if cell.get("unit") is None and cell.get("value_kind") in {"number", "zero", "basis_points"}:
            cell["unit"] = unit
        if cell.get("currency") is None and currency:
            cell["currency"] = currency
        if cell.get("accounting_basis") is None:
            cell["accounting_basis"] = accounting_basis
        selected_cells.append(cell)
    if not selected_cells:
        raise ValueError(f"selected Docling table {parser_table_id!r} has no cells")

    selected_tables = [raw_table]
    merged = dict(full_parse)
    merged["tables"] = selected_tables
    merged["cells"] = selected_cells
    annotate_table_continuation_candidates(selected_tables, selected_cells, merged.get("pages", []))
    merged["blocks"] = list(full_parse.get("blocks", [])) + _table_row_group_blocks(selected_tables, selected_cells)
    merged["parser_name"] = "hybrid_pypdf_full_source_docling_targeted_table"
    merged["parser_version"] = (
        f"pypdf-full:{full_parse.get('parser_version')}+"
        f"docling-targeted:{targeted_parse.get('parser_version')}"
    )
    merged["targeted_parse_scope"] = {
        "full_source_backend": full_parse.get("parser_name"),
        "full_source_page_count": len(full_parse.get("pages", [])),
        "full_docling_parse": False,
        "targeted_backend": targeted_parse.get("parser_name"),
        "targeted_source_physical_page": physical_page,
        "selected_raw_parser_table_id": parser_table_id,
        "mapped_physical_parser_table_id": mapped_table_id,
        "table_selection_basis": "explicit local source-review scope; table ID is not passed to retrieval",
    }
    return merged


def _selected_revenue_cells(
    merged_parse: Mapping[str, Any],
    *,
    row_index: int,
    source_metric_label: str,
    expected_raw_values: Mapping[str, str],
) -> dict[str, dict[str, Any]]:
    table_id = str(merged_parse["tables"][0]["table_id"])
    row = [
        dict(cell)
        for cell in merged_parse.get("cells", [])
        if cell.get("table_id") == table_id and int(cell.get("row_index", -1)) == row_index
    ]
    row.sort(key=lambda item: int(item["column_index"]))
    labels = [cell for cell in row if int(cell["column_index"]) == 0]
    year_cells = {
        "2024": [cell for cell in row if int(cell["column_index"]) == 1],
        "2023": [cell for cell in row if int(cell["column_index"]) == 2],
    }
    if len(labels) != 1 or any(len(cells) != 1 for cells in year_cells.values()):
        raise ValueError("manually reviewed revenue row has an unexpected cell layout")
    if str(labels[0].get("raw_text", "")).strip() != source_metric_label:
        raise ValueError("Docling revenue row label differs from the local review record")
    selected: dict[str, dict[str, Any]] = {}
    for period, cells in year_cells.items():
        value = cells[0]
        if str(value.get("raw_text", "")).strip() != expected_raw_values[period]:
            raise ValueError(f"Docling {period} revenue value differs from the local review record")
        if value.get("value_kind") not in {"number", "zero"} or value.get("normalized_value") is None:
            raise ValueError(f"Docling did not produce a finite numeric value for the reviewed {period} cell")
        selected[period] = value
    return selected


def _page_evidence_span(records: Mapping[str, Any], *, page: int, metric: str, raw_value: str) -> Any:
    candidates = [span for span in records["spans"] if int(span.page_number) == page]
    label_normalized = re.sub(r"\s+", "", metric)
    value_digits = re.sub(r"\D", "", raw_value)
    for span in candidates:
        normalized = re.sub(r"\s+", "", span.text)
        if label_normalized in normalized and value_digits in re.sub(r"\D", "", normalized):
            return span
    raise ValueError("full-source pypdf text has no exact page span containing the reviewed metric and value")


def _capture_source(
    *,
    spec: Mapping[str, object],
    source_record: Mapping[str, Any],
    output_dir: Path,
    store: IndustrialRagStore,
) -> dict[str, Any]:
    source_path = (ROOT / str(source_record["local_path"])).resolve()
    payload = source_path.read_bytes()
    digest = _sha256(payload)
    if digest != source_record.get("content_sha256"):
        raise ValueError(f"local source bytes differ from the recorded capture: {spec['source_id']}")

    source_run_dir = output_dir / "sources" / str(spec["source_id"])
    source_run_dir.mkdir(parents=True, exist_ok=False)
    full_parse = parse_isolated(
        payload,
        source_path.name,
        scratch_root=output_dir / "parser_scratch",
        timeout_seconds=180,
        pdf_backend="pypdf",
    )
    from io import BytesIO
    from pypdf import PdfReader

    page_count = len(PdfReader(BytesIO(payload)).pages)
    if full_parse.get("status") != "parsed" or len(full_parse.get("pages", [])) != page_count:
        raise RuntimeError(f"full-source pypdf parsing did not cover {spec['source_id']}")
    _write_json(source_run_dir / "full_source_pypdf_parse.json", full_parse)

    physical_page = int(spec["physical_page"])
    targeted_parse = parse_isolated(
        _page_subset(payload, physical_page),
        f"{source_path.stem}-physical-page-{physical_page}.pdf",
        scratch_root=output_dir / "parser_scratch",
        timeout_seconds=180,
        pdf_backend="docling",
        docling_artifacts_path=DOCLING_ASSETS,
        parser_timeout_seconds=180,
    )
    if targeted_parse.get("status") != "parsed":
        raise RuntimeError(f"targeted Docling parsing did not complete for {spec['source_id']}")
    _write_json(source_run_dir / "targeted_docling_parse.json", targeted_parse)
    merged_parse = _map_targeted_page(
        full_parse,
        targeted_parse,
        physical_page=physical_page,
        parser_table_id=str(spec["parser_table_id"]),
        unit=str(spec["unit"]),
        accounting_basis=spec.get("accounting_basis"),
        currency=str(spec.get("currency", "")),
    )
    _write_json(source_run_dir / "merged_hybrid_parse.json", merged_parse)
    version_id = f"v-{digest[:24]}"
    records = records_from_parse(merged_parse, version_id=version_id)
    value_cells = _selected_revenue_cells(
        merged_parse,
        row_index=int(spec["row_index"]),
        source_metric_label=str(spec["source_metric_label"]),
        expected_raw_values=spec["expected_raw_values"],
    )
    facts = []
    page_spans = {}
    conversion_factor = Decimal(str(spec["base_unit_conversion_factor"]))
    calculator_currency = str(spec.get("currency", ""))
    fact_unit = "CNY" if calculator_currency == "CNY" else str(spec["source_unit"])
    fact_currency = "CNY" if calculator_currency == "CNY" else ""
    for period, value_cell in value_cells.items():
        page_span = _page_evidence_span(
            records,
            page=physical_page,
            metric="营业收入",
            raw_value=str(value_cell["raw_text"]),
        )
        page_spans[period] = page_span
        source_value = Decimal(str(value_cell["normalized_value"]))
        fact_value = source_value * conversion_factor if calculator_currency == "CNY" else source_value
        facts.append(FinancialFact(
            fact_id=f"fact-{version_id}-revenue-{period}",
            version_id=version_id,
            security_code=str(spec["security_code"]),
            entity_name=str(source_record["issuer"]),
            metric="营业收入",
            raw_value=f"{value_cell['raw_text']} {spec['source_unit']}",
            normalized_value=format(fact_value, "f"),
            unit=fact_unit,
            multiplier="1",
            currency=fact_currency,
            report_period=period,
            consolidation=str(spec.get("accounting_basis") or ""),
            span_ids=(page_span.span_id,),
            parser_version=str(targeted_parse["parser_version"]),
            confidence=0.99,
            review_status="reviewed",
        ))

    captured_at = str(source_record["captured_at"])
    ingest_time = datetime.now(timezone.utc).isoformat()
    document = Document(
        document_id=f"doc-{source_record['source_id']}",
        title=str(source_record["title"]),
        document_type=str(source_record["document_type"]),
        security_code=str(spec["security_code"]),
        entity_name=str(source_record["issuer"]),
        report_period=None,
        accounting_basis=spec.get("accounting_basis") or "",
    )
    version = DocumentVersion(
        version_id=version_id,
        source_id=str(source_record["source_id"]),
        content_sha256=digest,
        source_url=str(source_record["source_url"]),
        rights_note=str(source_record["rights_note"]),
        published_at=source_record.get("published_at"),
        first_available_at=captured_at,
        captured_at=captured_at,
        ingested_at=ingest_time,
        time_basis="strict_forward",
        parser_name=str(merged_parse["parser_name"]),
        parser_version=str(merged_parse["parser_version"]),
        parser_confidence=min(
            float(full_parse.get("parser_confidence") or 0.0),
            float(targeted_parse.get("parser_confidence") or 0.0),
        ),
        parser_status="parsed",
    )
    bundle = DocumentBundle(
        document=document,
        version=version,
        payload=payload,
        sections=records["sections"],
        spans=records["spans"],
        blocks=records["blocks"],
        tables=records["tables"],
        cells=records["cells"],
        facts=tuple(facts),
        read_principals=(CAPTURE_PRINCIPAL_ID, "local-supervision"),
    )
    ingest = store.ingest_bundle(bundle)
    review_record = {
        "review_status": "reviewed_local_development_fact",
        "reviewer_identity": "codex-local-source-inspection",
        "review_independent": False,
        "reviewed_at": datetime.now(timezone.utc).isoformat(),
        "source_id": str(source_record["source_id"]),
        "source_sha256": digest,
        "physical_page": physical_page,
        "table_id": str(merged_parse["tables"][0]["table_id"]),
        "row_index": int(spec["row_index"]),
        "metric_label": str(spec["source_metric_label"]),
        "reviewed_period_cells": {
            period: {
                "value_cell_id": str(value_cell["cell_id"]),
                "value_column_index": int(value_cell["column_index"]),
                "raw_value": str(value_cell["raw_text"]),
                "source_unit": str(spec["source_unit"]),
                "source_normalized_value": str(value_cell["normalized_value"]),
                "base_unit_conversion_factor": str(conversion_factor) if calculator_currency == "CNY" else None,
                "calculator_value": format(
                    Decimal(str(value_cell["normalized_value"])) * conversion_factor
                    if calculator_currency == "CNY" else Decimal(str(value_cell["normalized_value"])),
                    "f",
                ),
                "calculator_unit": fact_unit,
                "pypdf_source_span_id": page_spans[period].span_id,
            }
            for period, value_cell in value_cells.items()
        },
        "unit": fact_unit,
        "calculator_currency": fact_currency or None,
        "source_unit": str(spec["source_unit"]),
        "base_unit_conversion_factor": str(conversion_factor) if calculator_currency == "CNY" else None,
        "calculator_eligibility": (
            "verified_absolute_currency" if calculator_currency == "CNY"
            else "source_observation_only_currency_not_stated_in_target_table"
        ),
        "accounting_basis": spec.get("accounting_basis"),
        "accounting_basis_status": accounting_basis_status(spec.get("accounting_basis")),
        "pypdf_span_contains_metric_and_both_period_values": True,
        "manual_review_note": str(spec["manual_review_note"]),
    }
    _write_json(source_run_dir / "local_fact_review.json", review_record)
    return {
        "case_id": str(spec["case_id"]),
        "source_id": str(source_record["source_id"]),
        "source_path": str(source_path),
        "source_url": str(source_record["source_url"]),
        "source_sha256": digest,
        "source_bytes": len(payload),
        "source_page_count": page_count,
        "recorded_local_capture_at": captured_at,
        "historical_published_at": source_record.get("published_at"),
        "historical_first_available_proxy_not_used": source_record.get("first_available_at"),
        "first_available_at_used_for_current_capture": captured_at,
        "time_basis_used": "strict_forward",
        "ingested_at": ingest_time,
        "version_id": version_id,
        "parser": {
            "full_source": full_parse.get("parser_name"),
            "full_source_page_count": len(full_parse.get("pages", [])),
            "targeted": targeted_parse.get("parser_name"),
            "targeted_version": targeted_parse.get("parser_version"),
            "targeted_physical_page": physical_page,
            "targeted_table_id": str(merged_parse["tables"][0]["table_id"]),
            "full_docling_parse": False,
        },
        "fact_review": review_record,
        "ingest": ingest,
    }


def _resume_source(
    *,
    spec: Mapping[str, object],
    source_record: Mapping[str, Any],
    output_dir: Path,
    store: IndustrialRagStore,
    release_id: str,
) -> dict[str, Any]:
    source_path = (ROOT / str(source_record["local_path"])).resolve()
    payload = source_path.read_bytes()
    digest = _sha256(payload)
    if digest != source_record.get("content_sha256"):
        raise ValueError(f"local source bytes differ from the recorded capture: {spec['source_id']}")
    version_id = f"v-{digest[:24]}"
    if version_id not in store.release_versions(release_id):
        raise ValueError(f"saved strict-forward release does not contain {spec['source_id']}")
    source_run_dir = output_dir / "sources" / str(spec["source_id"])
    review_record = json.loads((source_run_dir / "local_fact_review.json").read_text(encoding="utf-8"))
    full_parse = json.loads((source_run_dir / "full_source_pypdf_parse.json").read_text(encoding="utf-8"))
    targeted_parse = json.loads((source_run_dir / "targeted_docling_parse.json").read_text(encoding="utf-8"))
    merged_parse = json.loads((source_run_dir / "merged_hybrid_parse.json").read_text(encoding="utf-8"))
    if (
        review_record.get("source_id") != source_record.get("source_id")
        or review_record.get("source_sha256") != digest
        or review_record.get("review_status") != "reviewed_local_development_fact"
        or full_parse.get("status") != "parsed"
        or targeted_parse.get("status") != "parsed"
        or review_record.get("accounting_basis") != spec.get("accounting_basis")
        or review_record.get("accounting_basis_status") != accounting_basis_status(spec.get("accounting_basis"))
        or review_record.get("source_unit") != spec.get("source_unit")
    ):
        raise ValueError(f"saved current-capture evidence is incomplete or inconsistent: {spec['source_id']}")
    reviewed_cells = review_record.get("reviewed_period_cells")
    if not isinstance(reviewed_cells, dict):
        raise ValueError(f"saved source-value review is missing: {spec['source_id']}")
    for period, expected_raw_value in dict(spec["expected_raw_values"]).items():
        reviewed = reviewed_cells.get(str(period))
        if not isinstance(reviewed, dict) or reviewed.get("raw_value") != str(expected_raw_value):
            raise ValueError(f"saved source-value review differs from the reviewed target: {spec['source_id']} {period}")
    stored_facts = [dict(row) for row in store.read_facts([version_id])]
    expected_unit = "CNY" if spec.get("currency") == "CNY" else str(spec["source_unit"])
    expected_basis = str(spec.get("accounting_basis") or "")
    expected_periods = {str(period) for period in dict(spec["expected_raw_values"])}
    facts_by_period = {
        str(row.get("report_period")): row
        for row in stored_facts
        if str(row.get("metric")) == "营业收入" and str(row.get("security_code")) == str(spec["security_code"])
    }
    if set(facts_by_period) != expected_periods:
        raise ValueError(f"saved indexed facts do not match reviewed periods: {spec['source_id']}")
    factor = Decimal(str(spec["base_unit_conversion_factor"])) if spec.get("currency") == "CNY" else Decimal("1")
    for period, raw_value in dict(spec["expected_raw_values"]).items():
        fact = facts_by_period[str(period)]
        expected_value = Decimal(str(raw_value).replace(",", "")) * factor
        if (
            Decimal(str(fact.get("normalized_value"))) != expected_value
            or str(fact.get("unit")) != expected_unit
            or str(fact.get("consolidation") or "") != expected_basis
            or str(fact.get("entity_name")) != str(source_record["issuer"])
            or fact.get("review_status") != "reviewed"
        ):
            raise ValueError(f"saved indexed fact differs from the reviewed source: {spec['source_id']} {period}")
    return {
        "case_id": str(spec["case_id"]),
        "source_id": str(source_record["source_id"]),
        "source_path": str(source_path),
        "source_url": str(source_record["source_url"]),
        "source_sha256": digest,
        "source_bytes": len(payload),
        "source_page_count": len(full_parse.get("pages", [])),
        "recorded_local_capture_at": str(source_record["captured_at"]),
        "historical_published_at": source_record.get("published_at"),
        "historical_first_available_proxy_not_used": source_record.get("first_available_at"),
        "first_available_at_used_for_current_capture": str(source_record["captured_at"]),
        "time_basis_used": "strict_forward",
        "version_id": version_id,
        "parser": {
            "full_source": full_parse.get("parser_name"),
            "full_source_page_count": len(full_parse.get("pages", [])),
            "targeted": targeted_parse.get("parser_name"),
            "targeted_version": targeted_parse.get("parser_version"),
            "targeted_physical_page": int(spec["physical_page"]),
            "targeted_table_id": str(merged_parse["tables"][0]["table_id"]),
            "full_docling_parse": False,
        },
        "fact_review": review_record,
        "ingest": {
            "status": "reused_from_published_current_capture_release",
            "release_id": release_id,
            "version_id": version_id,
        },
    }


def _policy_json(*, use_qwen: bool = True) -> str:
    return json.dumps({
        "channels": ["structured_facts", "bm25", "dense"] if use_qwen else ["structured_facts", "bm25"],
        "max_searches": 2,
        "candidate_limit": 2,
        "symbol_limit": 1,
        "evidence_limit": 8,
        "rrf_rank_constant": 60,
        "reranker_profile": "configured" if use_qwen else "disabled",
        "table_expansion_limit": 4,
        "parent_context_mode": "reviewed_parent_and_child",
        "context_char_limit": 800,
        "context_source_limit": 2,
        "challenge_policy": "mandatory_correction_search",
        "stop_policy": "coverage_then_bounded_search",
        "memory_refs": ["procedural_experience"],
    }, ensure_ascii=False, separators=(",", ":"))


class ScriptedToolCallClient:
    """Drive the real fundamental tool protocol without calling a model provider."""

    model = "offline-scripted-tool-protocol-v1"

    def __init__(
        self,
        *,
        query: str,
        symbol: str,
        calculator_eligible: bool,
        calculation_ineligibility_reasons: list[str],
    ) -> None:
        self.query = query
        self.symbol = symbol
        self.calculator_eligible = calculator_eligible
        self.calculation_ineligibility_reasons = list(calculation_ineligibility_reasons)
        self.turn = 0
        self.selected: list[dict[str, Any]] = []

    @staticmethod
    def _response(calls: list[tuple[str, dict[str, object]]]) -> dict[str, object]:
        return {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": f"offline-call-{index + 1}",
                            "type": "function",
                            "function": {"name": name, "arguments": json.dumps(arguments, ensure_ascii=False)},
                        }
                        for index, (name, arguments) in enumerate(calls)
                    ],
                },
            }],
        }

    def complete(self, messages, _tools, *, budget=None, role=None, tool_choice=None):
        del budget, role, tool_choice
        self.turn += 1
        if self.turn == 1:
            return self._response([
                ("search_indexed_financial_documents", {
                    "question": self.query,
                    "symbols": [self.symbol],
                    "report_periods": [period],
                    "task_type": "fact",
                    "limit": 2,
                })
                for period in ("2023", "2024")
            ])
        if self.turn == 2:
            search_messages = [
                message for message in messages
                if message.get("role") == "tool" and message.get("name") == "search_indexed_financial_documents"
            ]
            if len(search_messages) != 2:
                raise RuntimeError("scripted tool driver did not receive indexed search results")
            retrieved = [
                item
                for message in search_messages
                for item in json.loads(str(message["content"])).get("retrieved", [])
                if isinstance(item, dict)
            ]
            facts_by_period = {
                str(item["financial_fact"].get("report_period")): item
                for item in retrieved
                if isinstance(item.get("financial_fact"), dict)
                and (
                    item["financial_fact"].get("numeric_eligible") is True
                    if self.calculator_eligible
                    else item["financial_fact"].get("source_observation_eligible") is True
                )
                and item["financial_fact"].get("report_period") in {"2023", "2024"}
            }
            if not {"2023", "2024"}.issubset(facts_by_period):
                raise RuntimeError("indexed search did not return both reviewed annual source facts")
            self.selected = [facts_by_period[period] for period in ("2023", "2024")]
            return self._response([
                ("read_indexed_financial_evidence", {"evidence_id": str(item["evidence_id"])})
                for item in self.selected
            ])
        if self.turn == 3:
            if not self.calculator_eligible:
                return self._submit_source_observation(messages)
            return self._response([("calculate_financial_metric", {
                "operation": "growth_percent",
                "operands": [
                    {"evidence_id": str(item["evidence_id"]), "field": "营业收入"}
                    for item in self.selected
                ],
            })])
        if self.turn == 4:
            if not self.calculator_eligible:
                return self._submit_source_observation(messages)
            return self._submit_calculation(messages)
        raise RuntimeError("scripted tool driver exceeded its four-turn plan")

    def _source_observations(self) -> list[dict[str, str]]:
        return [
            {
                "evidence_id": str(item["evidence_id"]),
                "field": "营业收入",
                "value": str(item["financial_fact"]["normalized_value"]),
                "unit": str(item["financial_fact"]["unit"]),
                "report_period": str(item["financial_fact"]["report_period"]),
            }
            for item in self.selected
        ]

    def _submit_source_observation(self, _messages: list[dict[str, Any]]) -> dict[str, object]:
        report = {
            "summary": "The source observations were checked against the local captured filing; calculation was withheld because a required unit or accounting basis is unknown.",
            "findings": [{
                "symbol": self.symbol,
                "conclusion": (
                    "同一来源表的2023年度和2024年度营业收入单元格已核对；"
                    "目标表的币种或会计口径未明确，因此仅报告来源观察，不报告同比计算。"
                ),
                "claim_type": "source_fact",
                "evidence_ids": [str(item["evidence_id"]) for item in self.selected],
                "observations": self._source_observations(),
                "uncertainties": [
                    reason.replace("_", " ") for reason in self.calculation_ineligibility_reasons
                ],
            }],
            "unresolved_items": [
                "Growth calculation withheld: " + ", ".join(self.calculation_ineligibility_reasons)
            ],
        }
        return self._response([("submit_report", report)])

    def _submit_calculation(self, messages: list[dict[str, Any]]) -> dict[str, object]:
        calculation_message = next(
            (message for message in reversed(messages)
             if message.get("role") == "tool" and message.get("name") == "calculate_financial_metric"),
            None,
        )
        if calculation_message is None:
            raise RuntimeError("scripted tool driver did not receive the Decimal calculation result")
        calculation = json.loads(str(calculation_message["content"]))
        if calculation.get("status") != "calculated":
            raise RuntimeError("deterministic financial calculator did not return a result")
        result_id = str(calculation["evidence_id"])
        result_value = str(calculation["value"])
        conclusion = f"同一来源表中，营业收入从2023年度到2024年度变化了{result_value}%；该值由Decimal计算得出。"
        uncertainties = []
        report = {
            "summary": "The reported percentage is a deterministic calculation from two reviewed source facts; source meaning remains unverified.",
            "findings": [{
                "symbol": self.symbol,
                "conclusion": conclusion,
                "claim_type": "calculation",
                "evidence_ids": [str(item["evidence_id"]) for item in self.selected] + [result_id],
                "observations": self._source_observations() + [{
                    "evidence_id": result_id,
                    "field": str(calculation["metric"]),
                    "value": result_value,
                    "unit": str(calculation["unit"]),
                    "report_period": str(calculation["report_period"]),
                }],
                "uncertainties": uncertainties,
            }],
            "unresolved_items": [],
        }
        return self._response([("submit_report", report)])


def _merge_search_results(results: list[Mapping[str, Any]]) -> dict[str, Any]:
    retrieved: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for result in results:
        for item in result.get("retrieved", []):
            if not isinstance(item, Mapping):
                continue
            evidence_id = str(item.get("evidence_id"))
            if evidence_id not in seen_ids:
                seen_ids.add(evidence_id)
                retrieved.append(dict(item))
    return {
        "status": "available" if any(result.get("status") == "available" for result in results) else (
            results[0].get("status", "no_evidence") if results else "no_evidence"
        ),
        "research_policy": results[0].get("research_policy") if results else None,
        "channels": [channel for result in results for channel in result.get("channels", [])],
        "coverage": [row for result in results for row in result.get("coverage", [])],
        "warnings": sorted({warning for result in results for warning in result.get("warnings", [])}),
        "retrieved": retrieved,
    }


def _collect_retrieval_trace_references(*values: object) -> list[dict[str, object]]:
    references_by_sequence: dict[int, dict[str, object]] = {}

    def visit(value: object) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                if key == "retrieval_trace_records" and isinstance(item, list):
                    for reference in item:
                        if not isinstance(reference, dict):
                            raise RunIdentityError("saved retrieval trace reference is malformed")
                        sequence = reference.get("sequence")
                        if isinstance(sequence, bool) or not isinstance(sequence, int):
                            raise RunIdentityError("saved retrieval trace reference is malformed")
                        existing = references_by_sequence.get(sequence)
                        if existing is not None and existing != reference:
                            raise RunIdentityError("saved retrieval trace references conflict")
                        references_by_sequence[sequence] = dict(reference)
                else:
                    visit(item)
        elif isinstance(value, list):
            for item in value:
                visit(item)

    for value in values:
        visit(value)
    return [references_by_sequence[sequence] for sequence in sorted(references_by_sequence)]


def _run_agent_chain(
    *,
    service: IndustrialRagService,
    release_id: str,
    case: Mapping[str, Any],
    decision_time: datetime,
    research_ablation_id: int | None,
    use_qwen: bool,
    progress_path: Path,
    identity_fields: Mapping[str, object],
    runtime_model_identity: Mapping[str, object],
    identity_check: Callable[[], None],
    retrieval_trace_journal: RetrievalTraceJournal,
) -> dict[str, Any]:
    query = QUERY_TEXT
    symbol = str(case["security_code"])
    visible_ids: set[str] = set()
    saved_progress = _existing_progress_record(progress_path, identity_fields)
    if saved_progress is not None:
        saved_attempts = saved_progress.get("attempts")
        if not isinstance(saved_attempts, list):
            raise RunIdentityError("saved agent progress has no attempt list")
        retrieval_trace_journal.verify_references_in(saved_attempts)
        completed_result = saved_progress.get("completed_result")
        if (
            isinstance(saved_attempts, list)
            and saved_attempts
            and isinstance(saved_attempts[-1], dict)
            and saved_attempts[-1].get("status") == "completed"
            and isinstance(completed_result, dict)
            and saved_progress.get("completed_result_sha256") == _canonical_hash(completed_result)
        ):
            retrieval_trace_journal.verify_references(
                saved_attempts[-1].get("retrieval_trace_records"),
                require_nonempty=True,
            )
            identity_check()
            return completed_result
    attempts = _resume_attempts(progress_path, identity_fields)
    attempt: dict[str, Any] = {
        "attempt_number": len(attempts) + 1,
        "status": "running",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "tool_events": [],
        "retrieval_trace_records": [],
    }
    attempts.append(attempt)
    progress_record = {
        **identity_fields,
        "case_id": case["case_id"],
        "security_code": symbol,
        "model_identity": dict(runtime_model_identity),
        "attempts": attempts,
    }

    def persist_progress() -> None:
        attempt["elapsed_seconds"] = round(time.perf_counter() - started, 3)
        progress_record["updated_at"] = datetime.now(timezone.utc).isoformat()
        progress_record["attempts_sha256"] = _canonical_hash(attempts)
        _write_json(progress_path, progress_record)

    started = time.perf_counter()
    persist_progress()

    def retrieval_trace_callback(trace_event: dict[str, object]) -> None:
        tool_events = attempt["tool_events"]
        active_index = len(tool_events) - 1
        active_tool = tool_events[active_index] if active_index >= 0 else None
        if not isinstance(active_tool, dict):
            raise RuntimeError("retrieval trace event has no active scripted tool call")
        record = retrieval_trace_journal.append(
            {
                **identity_fields,
                "attempt_number": attempt["attempt_number"],
                "tool_event_index": active_index,
                "tool": active_tool.get("tool"),
            },
            trace_event,
        )
        attempt["retrieval_trace_records"].append({
            "sequence": record["sequence"],
            "event_sha256": record["event_sha256"],
        })
        persist_progress()

    tools, handlers, catalog = _fundamental_handlers(
        None,
        decision_time,
        industrial_rag_service=service,
        industrial_rag_principal_id=CAPTURE_PRINCIPAL_ID,
        industrial_rag_release_id=release_id,
        research_policy_json=_policy_json(use_qwen=use_qwen),
        research_ablation_id=research_ablation_id,
        retrieval_trace_callback=retrieval_trace_callback,
    )
    handlers["submit_report"] = lambda arguments: _accept_report(
        arguments,
        catalog,
        "fundamental",
        visible_evidence_ids=visible_ids,
        decision_time=decision_time,
    )
    for name, handler in tuple(handlers.items()):
        def traced_handler(arguments: dict[str, object], *, tool_name=name, tool_handler=handler):
            identity_check()
            event = {
                "tool": tool_name,
                "arguments": dict(arguments),
                "status": "started",
                "started_at": datetime.now(timezone.utc).isoformat(),
            }
            attempt["tool_events"].append(event)
            persist_progress()
            tool_started = time.perf_counter()
            try:
                result = tool_handler(arguments)
            except Exception as error:
                event.update({
                    "status": "error",
                    "error": {"type": type(error).__name__, "message": str(error)},
                    "elapsed_seconds": round(time.perf_counter() - tool_started, 3),
                    "ended_at": datetime.now(timezone.utc).isoformat(),
                })
                persist_progress()
                raise
            event.update({
                "status": "returned",
                "result": result,
                "elapsed_seconds": round(time.perf_counter() - tool_started, 3),
                "ended_at": datetime.now(timezone.utc).isoformat(),
            })
            persist_progress()
            identity_check()
            return result
        handlers[name] = traced_handler
    client = ScriptedToolCallClient(
        query=query,
        symbol=symbol,
        calculator_eligible=bool(case["calculator_eligible"]),
        calculation_ineligibility_reasons=list(case.get("calculation_ineligibility_reasons", [])),
    )

    def persist_failure(error: Exception) -> None:
        failure: dict[str, Any] = {"type": type(error).__name__, "message": str(error)}
        partial_trace = getattr(error, "partial_tool_trace", None)
        if partial_trace is not None:
            failure["partial_tool_trace"] = partial_trace
        attempt.update({
            "status": "failed",
            "failure": failure,
            "selected_candidates": list(client.selected),
            "ended_at": datetime.now(timezone.utc).isoformat(),
        })
        persist_progress()

    try:
        result = _run_tool_agent(
            role_id="fundamental",
            system_prompt="Use the indexed local financial filing and cite every source fact.",
            task={
                "as_of": decision_time.isoformat(),
                "question": query,
                "symbol": symbol,
                "allowed_data": "the pinned local strict-forward industrial RAG release only",
            },
            tools=tools,
            handlers=handlers,
            client=client,
            budget=None,
            max_turns=4,
            visible_evidence_ids=visible_ids,
        )
    except Exception as exc:
        persist_failure(exc)
        if isinstance(exc, InvalidResearchOutput):
            raise RuntimeError(f"fundamental report validation failed: {exc.category}: {exc.correction_detail}") from None
        raise
    attempt["driver_candidate"] = {
        "status": result.get("status"),
        "report": result.get("report"),
        "tool_trace": result.get("tool_trace"),
    }
    persist_progress()
    if result.get("status") != "completed":
        error = RuntimeError("fundamental tool protocol did not complete")
        persist_failure(error)
        raise error
    trace = list(result["tool_trace"])
    search_results = [item["result"] for item in trace if item["tool"] == "search_indexed_financial_documents"]
    search = _merge_search_results(search_results)
    reads = [item["result"] for item in trace if item["tool"] == "read_indexed_financial_evidence"]
    finding = result["report"]["findings"][0]
    if not reads or not all(item.get("content_scope", "").startswith("reopened_source") for item in reads):
        error = RuntimeError("agent tool chain did not reopen source evidence")
        persist_failure(error)
        raise error
    expected_validation = (
        "calculation_recomputed_decimal_50_half_even_semantics_unverified"
        if case["calculator_eligible"]
        else "source_observations_validated_semantics_unverified"
    )
    if finding.get("claim_validation", {}).get("status") != expected_validation:
        error = RuntimeError("agent report did not pass the expected exact source or Decimal validation")
        persist_failure(error)
        raise error
    try:
        identity_check()
    except Exception as exc:
        persist_failure(exc)
        raise
    attempt.update({
        "status": "completed",
        "claim_validation": finding.get("claim_validation"),
        "ended_at": datetime.now(timezone.utc).isoformat(),
    })
    persist_progress()
    completed_result = {
        "agent_driver": client.model,
        "agent_model_called": False,
        "retrieval_trace_records": list(attempt["retrieval_trace_records"]),
        "query": query,
        "security_code": symbol,
        "research_ablation_id": research_ablation_id,
        "module_intervention": "none" if research_ablation_id is None else "Qwen reranker disabled (ablation 12)",
        "search_status": search.get("status"),
        "search_result": search,
        "research_policy": search.get("research_policy"),
        "channels": search.get("channels"),
        "coverage": search.get("coverage"),
        "retrieval_order": [
            {
                "rank": index + 1,
                "evidence_id": item.get("evidence_id"),
                "research_role": item.get("research_role"),
                "citation": {
                    key: (item.get("citation") or {}).get(key)
                    for key in ("source_id", "content_sha256", "page_number", "span_id", "table_id", "row_start", "row_end", "numeric_eligible")
                    if (item.get("citation") or {}).get(key) is not None
                },
                "financial_fact": item.get("financial_fact"),
            }
            for index, item in enumerate(search.get("retrieved", []))
        ],
        "source_reads": reads,
        "validated_report": result["report"],
        "tool_trace": trace,
    }
    progress_record["completed_result"] = completed_result
    progress_record["completed_result_sha256"] = _canonical_hash(completed_result)
    persist_progress()
    return completed_result


def _run_rag_intervention(
    *,
    service: IndustrialRagService,
    release_id: str,
    symbol: str,
    decision_time: datetime,
    research_ablation_id: int,
    report_period: str,
    use_qwen: bool,
    retrieval_trace_callback: Callable[[dict[str, object]], None] | None = None,
) -> dict[str, Any]:
    _tools, handlers, _catalog = _fundamental_handlers(
        None,
        decision_time,
        industrial_rag_service=service,
        industrial_rag_principal_id=CAPTURE_PRINCIPAL_ID,
        industrial_rag_release_id=release_id,
        research_policy_json=_policy_json(use_qwen=use_qwen),
        research_ablation_id=research_ablation_id,
        retrieval_trace_callback=retrieval_trace_callback,
    )
    return handlers["search_indexed_financial_documents"]({
        "question": QUERY_TEXT,
        "symbols": [symbol],
        "report_periods": [report_period],
        "task_type": "fact",
        "limit": 2,
    })


def _module_matrix(
    *,
    service: IndustrialRagService,
    release_id: str,
    symbol: str,
    decision_time: datetime,
    baseline: Mapping[str, Any],
    reranker_disabled: Mapping[str, Any],
    use_qwen: bool,
    progress_path: Path,
    identity_fields: Mapping[str, object],
    runtime_model_identity: Mapping[str, object],
    identity_check: Callable[[], None],
    retrieval_trace_journal: RetrievalTraceJournal,
) -> list[dict[str, Any]]:
    baseline_items = list(baseline.get("retrieval_order", []))
    baseline_ids = [str(item.get("evidence_id")) for item in baseline_items]
    result_by_module: dict[int, dict[str, Any]] = {}
    saved_progress = _existing_progress_record(progress_path, identity_fields)
    active_module_id: int | None = None
    active_period_results: list[dict[str, Any]] = []
    elapsed_by_module: dict[int, float] = {}
    if saved_progress is not None:
        retrieval_trace_journal.verify_references_in(saved_progress)
        saved_results = saved_progress.get("results", {})
        if isinstance(saved_results, dict):
            result_by_module = {
                int(module_id): dict(result)
                for module_id, result in saved_results.items()
                if str(module_id).isdigit() and isinstance(result, dict)
            }
        saved_elapsed = saved_progress.get("elapsed_seconds_by_module", {})
        if isinstance(saved_elapsed, dict):
            elapsed_by_module = {
                int(module_id): float(seconds)
                for module_id, seconds in saved_elapsed.items()
                if str(module_id).isdigit()
            }
        raw_active_id = saved_progress.get("active_module_id")
        if isinstance(raw_active_id, int) and raw_active_id not in result_by_module:
            active_module_id = raw_active_id
            raw_period_results = saved_progress.get("active_module_period_results", [])
            if isinstance(raw_period_results, list):
                active_period_results = [dict(row) for row in raw_period_results if isinstance(row, dict)]
    runnable_ids = {2, 3, 7, 8, 9, 11, 13, 14, 15, 16}
    if use_qwen:
        runnable_ids.update({10, 12})
        result_by_module[12] = {
            **dict(reranker_disabled.get("search_result", {})),
            "retrieval_trace_records": list(reranker_disabled.get("retrieval_trace_records", [])),
        }
    descriptions = {
        2: "Disable reviewed cross-page continuation context.",
        3: "Remove semantic row-group chunks from the retrieval candidate set.",
        7: "Disable constrained query rewriting.",
        8: "Disable multi-hop decomposition.",
        9: "Disable BM25 support retrieval while retaining its correction-search guard.",
        10: "Disable the dense embedding channel.",
        11: "Replace reciprocal-rank fusion with the first available channel.",
        12: "Disable the configured local Qwen3 reranker.",
        13: "Disable reviewed table-row expansion.",
        14: "Disable normalized structured-fact retrieval while retaining report validation.",
        15: "Remove the candidate context cap under the fixed tool ceiling.",
        16: "Remove the candidate evidence cap under the service ceiling.",
    }

    def persist_module_progress(progress_record: dict[str, Any]) -> None:
        progress_record["results_sha256"] = _canonical_hash(progress_record.get("results", {}))
        if "active_module_period_results" in progress_record:
            progress_record["active_module_period_results_sha256"] = _canonical_hash(
                progress_record["active_module_period_results"]
            )
        _write_json(progress_path, progress_record)

    module_ids_to_run = sorted((runnable_ids - ({12} if use_qwen else set())) - set(result_by_module))
    for module_id in module_ids_to_run:
        identity_check()
        module_started = time.perf_counter()
        period_results = active_period_results if module_id == active_module_id else []
        completed_periods = {
            str(row.get("report_period"))
            for row in period_results
            if isinstance(row.get("result"), dict) and row.get("status", "completed") == "completed"
        }
        active_period_record: dict[str, Any] | None = None
        try:
            for period in ("2023", "2024"):
                if period in completed_periods:
                    continue
                identity_check()
                search_started = time.perf_counter()
                active_period_record = next(
                    (
                        row for row in period_results
                        if str(row.get("report_period")) == period and row.get("status") == "running"
                    ),
                    None,
                )
                if active_period_record is None:
                    active_period_record = {
                        "report_period": period,
                        "status": "running",
                        "retrieval_trace_records": [],
                    }
                    period_results.append(active_period_record)
                period_trace_records = active_period_record["retrieval_trace_records"]
                persist_module_progress({
                    **identity_fields,
                    "security_code": symbol,
                    "model_identity": dict(runtime_model_identity),
                    "phase": "running",
                    "active_module_id": module_id,
                    "completed_module_ids": sorted(result_by_module),
                    "elapsed_seconds_by_module": elapsed_by_module,
                    "results": result_by_module,
                    "active_module_period_results": period_results,
                })

                def trace_intervention(event: dict[str, object]) -> None:
                    record = retrieval_trace_journal.append(
                        {
                            **identity_fields,
                            "progress_kind": "module_matrix",
                            "module_id": module_id,
                            "report_period": period,
                        },
                        event,
                    )
                    period_trace_records.append({
                        "sequence": record["sequence"],
                        "event_sha256": record["event_sha256"],
                    })
                    persist_module_progress({
                        **identity_fields,
                        "security_code": symbol,
                        "model_identity": dict(runtime_model_identity),
                        "phase": "running",
                        "active_module_id": module_id,
                        "completed_module_ids": sorted(result_by_module),
                        "elapsed_seconds_by_module": elapsed_by_module,
                        "results": result_by_module,
                        "active_module_period_results": period_results,
                    })

                result = _run_rag_intervention(
                    service=service,
                    release_id=release_id,
                    symbol=symbol,
                    decision_time=decision_time,
                    research_ablation_id=module_id,
                    report_period=period,
                    use_qwen=use_qwen,
                    retrieval_trace_callback=trace_intervention,
                )
                active_period_record.update({
                    "status": "completed",
                    "result": result,
                    "elapsed_seconds": round(time.perf_counter() - search_started, 3),
                })
                persist_module_progress({
                    **identity_fields,
                    "security_code": symbol,
                    "model_identity": dict(runtime_model_identity),
                    "phase": "running",
                    "active_module_id": module_id,
                    "completed_module_ids": sorted(result_by_module),
                    "elapsed_seconds_by_module": elapsed_by_module,
                    "results": result_by_module,
                    "active_module_period_results": period_results,
                })
                identity_check()
            merged_period_results = [
                row["result"] for row in period_results
                if isinstance(row.get("result"), dict)
            ]
            result_by_module[module_id] = {
                **_merge_search_results(merged_period_results),
                "retrieval_trace_records": [
                    reference
                    for row in period_results
                    for reference in row.get("retrieval_trace_records", [])
                ],
            }
        except Exception as exc:
            if isinstance(exc, RunIdentityError):
                raise
            if active_period_record is not None and active_period_record.get("status") == "running":
                active_period_record.update({
                    "status": "failed",
                    "failure": {"type": type(exc).__name__, "message": str(exc)},
                    "elapsed_seconds": round(time.perf_counter() - module_started, 3),
                })
            result_by_module[module_id] = {
                "status": "execution_error",
                "execution_error": {"type": type(exc).__name__, "message": str(exc)},
                "partial_period_results": period_results,
                "retrieval_trace_records": [
                    reference
                    for row in period_results
                    for reference in row.get("retrieval_trace_records", [])
                ],
                "retrieved": [],
            }
        elapsed_by_module[module_id] = round(time.perf_counter() - module_started, 3)
        active_module_id = None
        active_period_results = []
        persist_module_progress({
            **identity_fields,
            "security_code": symbol,
            "model_identity": dict(runtime_model_identity),
            "phase": "running",
            "completed_module_ids": sorted(result_by_module),
            "elapsed_seconds_by_module": elapsed_by_module,
            "results": result_by_module,
        })
        identity_check()
    rows = []
    for module_id, module_name, mode in _ABLATION_MODULES:
        if module_id in runnable_ids:
            result = result_by_module[module_id]
            retrieved = [item for item in result.get("retrieved", []) if isinstance(item, Mapping)]
            after_ids = [str(item.get("evidence_id")) for item in retrieved]
            rows.append({
                "module_id": module_id,
                "module": module_name,
                "catalog_mode": mode,
                "execution_status": "error" if result.get("status") == "execution_error" else "executed",
                "retrieval_status": result.get("status"),
                "intervention_applied": True,
                "same_question_release_source_as_baseline": True,
                "research_ablation_id": module_id,
                "intervention_description": descriptions[module_id],
                "baseline_evidence_ids": baseline_ids,
                "intervention_evidence_ids": after_ids,
                "evidence_order_changed": baseline_ids != after_ids,
                "channels": result.get("channels"),
                "coverage": result.get("coverage"),
                "retrieval_trace_records": list(result.get("retrieval_trace_records", [])),
                "numeric_fact_periods_returned": sorted({
                    str((item.get("financial_fact") or {}).get("report_period"))
                    for item in retrieved
                    if isinstance(item.get("financial_fact"), Mapping)
                    and (item.get("financial_fact") or {}).get("numeric_eligible") is True
                }),
                "warnings": result.get("warnings", []),
                "execution_error": result.get("execution_error"),
            })
        else:
            reasons = {
                1: "Offline parser and index rebuild required; not safely switchable in this run.",
                4: "Immutable page and cell citation linkage contract; preservation observed, no destructive ablation attempted.",
                5: "Immutable issuer and metric alias contract; no alias mutation attempted.",
                6: "Immutable time, period, and basis extraction contract; no temporal guard mutation attempted.",
                17: "Immutable challenge and stop safety policy; no safety guard mutation attempted.",
                18: "Claim and citation validator exercised by the numeric report chain; validator not ablated.",
                19: "Decimal calculator and recomputation validator exercised where unit eligibility allowed; arithmetic safety not ablated.",
                20: "Conflict and unsupported-claim safety checks remain enabled; no conflict guard mutation attempted.",
            }
            if module_id == 10 and not use_qwen:
                reasons[module_id] = "Dense retrieval was absent from this supplemental baseline; dense-off would be a no-op."
            if module_id == 12 and not use_qwen:
                reasons[module_id] = "Reranking was disabled in this supplemental baseline; reranker-off would be a no-op."
            rows.append({
                "module_id": module_id,
                "module": module_name,
                "catalog_mode": mode,
                "execution_status": "not_ablated",
                "intervention_applied": False,
                "same_question_release_source_as_baseline": True,
                "reason": reasons[module_id],
                "baseline_evidence_ids": baseline_ids,
            })
    persist_module_progress({
        **identity_fields,
        "security_code": symbol,
        "model_identity": dict(runtime_model_identity),
        "phase": "completed",
        "completed_module_ids": sorted(result_by_module),
        "elapsed_seconds_by_module": elapsed_by_module,
        "results": result_by_module,
        "module_rows": rows,
    })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, default=SOURCE_MANIFEST)
    parser.add_argument("--docling-assets", type=Path, default=DOCLING_ASSETS)
    parser.add_argument("--qwen-assets", type=Path, default=QWEN_ASSETS)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--run-id", help="Unique local run directory")
    parser.add_argument("--resume-existing", action="store_true", help="Reuse parsed documents and a published release in the named run directory")
    parser.add_argument("--skip-qwen", action="store_true", help="Run retrieval and claim validation without local Qwen inference")
    parser.add_argument("--case-id", choices=[str(spec["case_id"]) for spec in CASE_SPECS], help="Restrict this diagnostic run to one reviewed issuer case")
    parser.add_argument("--skip-module-matrix", action="store_true", help="Run the primary tool chain and paired reranker check without the 20-module matrix")
    args = parser.parse_args()
    selected_specs = [
        spec for spec in CASE_SPECS
        if args.case_id is None or str(spec["case_id"]) == args.case_id
    ]

    run_id = args.run_id or datetime.now(timezone.utc).strftime("current-capture-%Y%m%dT%H%M%SZ")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}", run_id):
        raise ValueError("run-id must be a safe immutable identifier")
    output_dir = args.output_root.resolve() / run_id
    use_qwen = not args.skip_qwen
    as_of: datetime | None = None
    saved_identity_envelope: dict[str, Any] | None = None
    saved_run_identity: dict[str, Any] | None = None
    if args.resume_existing:
        if not output_dir.is_dir() or not (output_dir / "industrial_rag.sqlite3").is_file():
            raise FileNotFoundError("resume-existing requires an existing materialized current-capture run")
        saved_identity_envelope = _read_run_identity(output_dir / "run_identity.json")
        saved_run_identity = dict(saved_identity_envelope["run_identity"])
        if saved_run_identity.get("run_id") != run_id:
            raise RunIdentityError("saved run identity names a different run id")
        if saved_run_identity.get("qwen_inference_enabled") is not use_qwen:
            raise RunIdentityError("resume-existing Qwen mode differs from the saved run; refusing before model initialization")
        requested_scope = {
            "case_ids": sorted(str(spec["case_id"]) for spec in selected_specs),
            "module_matrix_enabled": not args.skip_module_matrix,
        }
        if saved_run_identity.get("evaluation_scope") != requested_scope:
            raise RunIdentityError("resume-existing evaluation scope differs from the saved run; refusing before model initialization")
        as_of = datetime.fromisoformat(str(saved_run_identity["as_of"])).astimezone(timezone.utc)
    else:
        as_of = None
    if not args.resume_existing:
        output_dir.mkdir(parents=True, exist_ok=False)

    manifest_records = _read_source_records(args.source_manifest)
    store = IndustrialRagStore(
        output_dir / "industrial_rag.sqlite3",
        LocalContentAddressedStore(output_dir / "objects"),
    )
    source_records: list[dict[str, Any]] = []
    version_ids: list[str] = []
    for spec in selected_specs:
        source_record = manifest_records.get(str(spec["source_id"]))
        if source_record is None:
            raise ValueError(f"source record is missing: {spec['source_id']}")
        if Path(str(source_record.get("local_path", ""))).name != str(spec["filename"]):
            raise ValueError(f"source path does not match the reviewed current-capture case: {spec['source_id']}")
        capture = _resume_source if args.resume_existing else _capture_source
        capture_args = {
            "spec": spec,
            "source_record": source_record,
            "output_dir": output_dir,
            "store": store,
        }
        if args.resume_existing:
            captured = capture(**capture_args, release_id=run_id)
        else:
            captured = capture(**capture_args)
        source_records.append(captured)
        version_ids.append(str(captured["version_id"]))

    if args.resume_existing:
        release_summary = store.release_summaries()
        matching_release = next((item for item in release_summary["releases"] if item["release_id"] == run_id), None)
        if release_summary["active_release_id"] != run_id or matching_release is None or matching_release["status"] != "published":
            raise RuntimeError("resume-existing release is not the published active current-capture release")
        if set(version_ids) != set(matching_release["versions"]):
            raise RuntimeError("resume-existing source versions differ from the published release manifest")
        release = {"status": matching_release["status"], "release": matching_release, "reused": True}
    else:
        release = build_release(
            release_id=run_id,
            version_ids=version_ids,
            database_path=output_dir / "industrial_rag.sqlite3",
            object_root=output_dir / "objects",
            actor_id="local-supervision",
            reason="Current-capture multi-issuer strict-forward development evidence",
        )
    release_id = store.active_release_id()
    if release_id != run_id:
        raise RuntimeError("new current-capture release is not the active release")
    if as_of is None:
        as_of = datetime.now(timezone.utc)
    decision_time = as_of.astimezone(SHANGHAI)

    source_manifest_path = (ROOT / args.source_manifest).resolve() if not args.source_manifest.is_absolute() else args.source_manifest.resolve()
    docling_manifest_path = args.docling_assets / ASSET_MANIFEST
    qwen_manifest_path = args.qwen_assets / "raama_qwen_models.json"
    run_identity = _build_run_identity(
        run_id=run_id,
        as_of=as_of,
        use_qwen=use_qwen,
        source_manifest_path=source_manifest_path,
        docling_manifest_path=docling_manifest_path,
        qwen_manifest_path=qwen_manifest_path,
        source_records=source_records,
        store=store,
        case_ids=[str(spec["case_id"]) for spec in selected_specs],
        module_matrix_enabled=not args.skip_module_matrix,
    )
    run_identity_sha256 = _canonical_hash(run_identity)
    identity_path = output_dir / "run_identity.json"
    if args.resume_existing:
        assert saved_identity_envelope is not None and saved_run_identity is not None
        _verify_run_identity(saved_run_identity, run_identity)
        saved_runtime_identity_sha256 = saved_identity_envelope.get("runtime_model_identity_sha256")
    else:
        saved_runtime_identity_sha256 = None
        saved_identity_envelope = {
            "schema_version": "raama.financial-rag.current-capture.run-identity-envelope.v1",
            "run_identity": run_identity,
            "run_identity_sha256": run_identity_sha256,
            "runtime_model_identity": None,
            "runtime_model_identity_sha256": None,
        }
        _write_json(identity_path, saved_identity_envelope)

    query_progress_path = output_dir / "query_progress.json"
    query_progress_exists = query_progress_path.is_file()
    raw_query_progress: object | None = None
    if query_progress_exists:
        raw_query_progress = json.loads(query_progress_path.read_text(encoding="utf-8"))
        _validate_query_progress_header(
            raw_query_progress,
            run_identity_sha256=run_identity_sha256,
            runtime_model_identity_sha256=(
                str(saved_runtime_identity_sha256) if saved_runtime_identity_sha256 is not None else None
            ),
        )
    spec_by_case_id = {str(spec["case_id"]): spec for spec in selected_specs}
    source_by_case_id = {str(source["case_id"]): source for source in source_records}
    retrieval_policy_sha256 = str(run_identity["research_policy_sha256"])
    case_identity_by_id: dict[str, dict[str, object]] = {
        case_id: _case_identity_fields(
            run_identity_sha256,
            spec,
            source_by_case_id[case_id],
            as_of=as_of,
            retrieval_policy_sha256=retrieval_policy_sha256,
            access_scope=run_identity["access_scope"],
        )
        for case_id, spec in spec_by_case_id.items()
    }
    saved_runtime_identity = saved_identity_envelope.get("runtime_model_identity") if saved_identity_envelope else None
    if saved_runtime_identity is not None and (
        not isinstance(saved_runtime_identity, dict)
        or saved_runtime_identity_sha256 != _canonical_hash(saved_runtime_identity)
    ):
        raise RunIdentityError("saved runtime model identity or its hash is invalid")
    if query_progress_exists:
        if isinstance(raw_query_progress, dict) and raw_query_progress.get("cases"):
            if not isinstance(saved_runtime_identity_sha256, str):
                raise RunIdentityError("completed cases have no saved runtime model identity")
            _validate_query_progress(
                raw_query_progress,
                run_identity_sha256=run_identity_sha256,
                runtime_model_identity_sha256=saved_runtime_identity_sha256,
                case_identity_by_id=case_identity_by_id,
            )

    dense_adapter = None
    reranker = None
    if use_qwen:
        embedding = Qwen3EmbeddingRuntime(args.qwen_assets)
        reranker_runtime = Qwen3RerankerRuntime(args.qwen_assets)
        dense_adapter = LocalQwenDenseAdapter(
            store,
            embedding,
            cache_path=ROOT / ".tmp" / "industrial_rag_dense_vector_cache.sqlite3",
        )
        reranker = LocalQwenReranker(reranker_runtime)
    runtime_model_identity = _runtime_model_identity(
        use_qwen=use_qwen,
        dense_adapter=dense_adapter,
        reranker=reranker,
    )
    runtime_model_identity_sha256 = _bind_runtime_model_identity(
        identity_path,
        query_progress_path,
        runtime_model_identity,
        resuming=args.resume_existing,
    )
    saved_identity_envelope = _read_run_identity(identity_path)

    def identity_check() -> None:
        current_identity = _build_run_identity(
            run_id=run_id,
            as_of=as_of,
            use_qwen=use_qwen,
            source_manifest_path=source_manifest_path,
            docling_manifest_path=docling_manifest_path,
            qwen_manifest_path=qwen_manifest_path,
            source_records=source_records,
            store=store,
            case_ids=[str(spec["case_id"]) for spec in selected_specs],
            module_matrix_enabled=not args.skip_module_matrix,
        )
        _verify_run_identity(run_identity, current_identity)
        current_runtime_identity = _runtime_model_identity(
            use_qwen=use_qwen,
            dense_adapter=dense_adapter,
            reranker=reranker,
        )
        if current_runtime_identity != runtime_model_identity:
            raise RunIdentityError("runtime model identity changed during the run")

    identity_check()
    if not query_progress_exists:
        query_progress: dict[str, Any] = {
            "schema_version": "raama.financial-rag.current-capture.query-progress.v2",
            "run_identity_sha256": run_identity_sha256,
            "runtime_model_identity_sha256": runtime_model_identity_sha256,
            "cases": [],
        }
    else:
        validated_rows = _validate_query_progress(
            raw_query_progress,
            run_identity_sha256=run_identity_sha256,
            runtime_model_identity_sha256=runtime_model_identity_sha256,
            case_identity_by_id=case_identity_by_id,
        )
        query_progress = dict(raw_query_progress)
        query_progress["cases"] = validated_rows
    query_runs = list(query_progress["cases"])
    _write_json(query_progress_path, query_progress)
    retrieval_trace_journal = RetrievalTraceJournal(
        output_dir / "retrieval_trace.jsonl",
        run_identity_sha256=run_identity_sha256,
        runtime_model_identity_sha256=runtime_model_identity_sha256,
    )
    for query_record in query_runs:
        retrieval_trace_journal.verify_references(
            query_record.get("retrieval_trace_records"),
            require_nonempty=True,
        )

    service = IndustrialRagService(
        store,
        channel_adapters={"dense": dense_adapter} if dense_adapter is not None else {},
        reranker=reranker,
        allow_retrospective_diagnostic=False,
    )
    model_identity = {
        "embedding_repo": EMBEDDING_REPO,
        "embedding_revision": EMBEDDING_REVISION,
        "embedding_cache_identity": dense_adapter.cache_identity if dense_adapter is not None else None,
        "reranker_repo": RERANKER_REPO,
        "reranker_revision": RERANKER_REVISION,
        "reranker_cache_identity": reranker.cache_identity if reranker is not None else None,
        "qwen_inference_enabled_this_run": use_qwen,
        "local_files_only": True,
        "asset_manifest_sha256": _sha256(qwen_manifest_path.read_bytes()),
    }
    completed_case_ids = {str(item["case_id"]) for item in query_runs}
    for source, spec in zip(source_records, selected_specs):
        identity_check()
        if str(source["case_id"]) in completed_case_ids:
            identity_check()
            continue
        case_identity = case_identity_by_id[str(source["case_id"])]
        case_identity_sha256 = _canonical_hash(case_identity)
        case_progress_identity = {
            **case_identity,
            "runtime_model_identity_sha256": runtime_model_identity_sha256,
            "case_identity_sha256": case_identity_sha256,
        }
        case = {
            "case_id": source["case_id"],
            "question": QUERY_TEXT,
            "security_code": spec["security_code"],
            "report_periods": ["2023", "2024"],
            "calculator_eligible": _case_calculation_eligible(spec),
            "calculation_ineligibility_reasons": _case_calculation_reasons(spec),
            "source_id": source["source_id"],
        }
        baseline = _run_agent_chain(
            service=service,
            release_id=run_id,
            case=case,
            decision_time=decision_time,
            research_ablation_id=None,
            use_qwen=use_qwen,
            progress_path=output_dir / f"agent_progress_{spec['security_code']}.json",
            identity_fields={**case_progress_identity, "progress_kind": "agent", "research_ablation_id": None},
            runtime_model_identity=runtime_model_identity,
            identity_check=identity_check,
            retrieval_trace_journal=retrieval_trace_journal,
        )
        no_reranker = (
            _run_agent_chain(
                service=service,
                release_id=run_id,
                case=case,
                decision_time=decision_time,
                research_ablation_id=12,
                use_qwen=True,
                progress_path=output_dir / f"agent_progress_{spec['security_code']}_ablation12.json",
                identity_fields={**case_progress_identity, "progress_kind": "agent", "research_ablation_id": 12},
                runtime_model_identity=runtime_model_identity,
                identity_check=identity_check,
                retrieval_trace_journal=retrieval_trace_journal,
            )
            if use_qwen
            else {"status": "not_run_qwen_disabled_in_supplemental_baseline"}
        )
        baseline_order = [item["evidence_id"] for item in baseline["retrieval_order"]]
        no_reranker_order = [item["evidence_id"] for item in no_reranker.get("retrieval_order", [])]
        module_rows = (
            _module_matrix(
                service=service,
                release_id=run_id,
                symbol=str(spec["security_code"]),
                decision_time=decision_time,
                baseline=baseline,
                reranker_disabled=no_reranker,
                use_qwen=use_qwen,
                progress_path=output_dir / f"module_progress_{spec['security_code']}.json",
                identity_fields={**case_progress_identity, "progress_kind": "module_matrix"},
                runtime_model_identity=runtime_model_identity,
                identity_check=identity_check,
                retrieval_trace_journal=retrieval_trace_journal,
            )
            if not args.skip_module_matrix else []
        )
        identity_check()
        query_record = {
            **case_identity,
            "retrieval_trace_records": _collect_retrieval_trace_references(
                baseline,
                no_reranker,
                module_rows,
            ),
            "runtime_model_identity_sha256": runtime_model_identity_sha256,
            "case_identity_sha256": case_identity_sha256,
            "numeric_calculation_eligible": bool(case["calculator_eligible"]),
            "calculation_ineligibility_reasons": list(case["calculation_ineligibility_reasons"]),
            "baseline_retrieval": baseline,
            "paired_ablation_reranker_disabled": no_reranker,
            "module_ablation_matrix": module_rows,
            "module_comparison": {
                "same_question_release_and_source": True,
                "changed_module": (
                    "configured Qwen3 reranker disabled for ablation 12"
                    if use_qwen else
                    "no Qwen intervention; supplemental baseline has dense retrieval and reranking disabled"
                ),
                "paired_ablation_status": no_reranker.get("status", no_reranker.get("search_status")),
                "baseline_reranker_identity": model_identity["reranker_cache_identity"],
                "baseline_dense_identity": model_identity["embedding_cache_identity"],
                "retrieval_order_changed": baseline_order != no_reranker_order if use_qwen else None,
                "baseline_order": baseline_order,
                "reranker_disabled_order": no_reranker_order,
            },
        }
        query_record["case_result_sha256"] = _case_result_sha256(query_record)
        query_runs.append(query_record)
        completed_case_ids.add(str(source["case_id"]))
        query_progress["cases"] = query_runs
        _write_json(query_progress_path, query_progress)

    record = {
        "schema_version": "raama.financial-rag.current-capture.v1",
        "run_id": run_id,
        "capture_classification": "current-capture",
        "evaluation_scope": run_identity["evaluation_scope"],
        "capture_resume": {
            "resumed_existing_capture": args.resume_existing,
            "prior_attempt_reused_same_local_sources_and_release": args.resume_existing,
            "prior_attempt_note": "Saved source and release artifacts passed run-identity and indexed-fact checks before continuation." if args.resume_existing else None,
        },
        "as_of": as_of.isoformat(),
        "available_from_rule": "recorded local source capture time is used as first_available_at",
        "historical_2025_first_available_proxies_used": False,
        "future_returns_evaluated": False,
        "provider_calls": 0,
        "agent_model_called": False,
        "agent_tool_protocol": "real fundamental tool handlers, reopened evidence, deterministic Decimal calculator, and report validator driven by deterministic offline scripted tool calls",
        "query_text_shared_by_selected_cases_and_ablations": QUERY_TEXT,
        "release": release,
        "release_id": run_id,
        "time_mode": "strict_forward",
        "source_records": source_records,
        "run_identity_sha256": run_identity_sha256,
        "runtime_model_identity_sha256": runtime_model_identity_sha256,
        "model_identity": model_identity,
        "retrieval_trace_journal": {
            "path": "retrieval_trace.jsonl",
            "schema_version": RetrievalTraceJournal.schema_version,
            "event_count": retrieval_trace_journal.sequence,
            "terminal_event_sha256": retrieval_trace_journal.previous_event_sha256,
        },
        "dense_vector_cache": {
            "path": str((ROOT / ".tmp" / "industrial_rag_dense_vector_cache.sqlite3").relative_to(ROOT))
            if use_qwen else None,
            "reuse_identity": "source_id+original_sha256+block_id+block_text_sha256+model_revision+asset_manifest+instruction+dimension"
            if use_qwen else None,
        },
        "docling_asset_manifest_sha256": _sha256(docling_manifest_path.read_bytes()) if docling_manifest_path.is_file() else None,
        "local_inference_device": "CPU" if use_qwen else None,
        "retrieval_channel_scope": ["structured_facts", "bm25", "dense"] if use_qwen else ["structured_facts", "bm25"],
        "historical_qwen_attempt": {
            "status": "retrieval_returned_without_both_annual_source_facts_captured",
            "embedding_revision": EMBEDDING_REVISION,
            "reranker_revision": RERANKER_REVISION,
            "provider_calls": 0,
            "retrieval_trace_persisted": False,
            "candidates_or_ranks_available": False,
        },
        "module_ablation_scope": (
            "Not run in this targeted single-case scope."
            if args.skip_module_matrix else
            "Actual bounded retrieval handler interventions for modules 2, 3, and 7-16; parser rebuild, immutable citation/alias/time contracts, and safety guards are recorded as not ablated."
        ),
        "queries": query_runs,
        "limitations": [
            f"{len(source_records)} viewed development document(s) and one manually scoped table page per document are not an independent or blind benchmark.",
            "Full-source page text uses pypdf; Docling parses only the reviewed revenue-table page, not every page.",
            "Numeric facts were locally reviewed by Codex against the page images and parser output; this review is not independent.",
            "The tool protocol is real, but the tool-call sequence is scripted; no language model or provider was called.",
            "A separate historical local Qwen attempt returned without both annual facts exposed to the scripted driver; its exact retrieval response was not persisted. This run's retrieval results and failure traces are checkpointed under the bound run and runtime identities.",
            "CATL's reviewed table labels the values in 千元 but does not state currency; source observations are validated, but no currency calculation is attempted for that case.",
            "Ping An and CATL target tables do not state accounting basis; their values remain source observations and no growth calculation is eligible.",
            "The module matrix records single-query retrieval outcomes only when enabled; it is not a causal ablation study, independent acceptance result, or quality score.",
            "Retrieval ranks and exact claim validation do not establish semantic entailment or investment quality.",
        ],
    }
    _write_json(output_dir / "current_capture_run.json", record)
    print(json.dumps({
        "run_id": run_id,
        "output_dir": str(output_dir),
        "release_status": release.get("status"),
        "as_of": as_of.isoformat(),
        "time_mode": "strict_forward",
        "source_count": len(source_records),
        "query_count": len(query_runs),
        "provider_calls": 0,
        "model_identity": model_identity,
        "agent_claims": [
            {
                "case_id": item["case_id"],
                "baseline_status": item["baseline_retrieval"]["validated_report"]["findings"][0]["claim_validation"]["status"],
                "reranker_disabled_status": item["paired_ablation_reranker_disabled"].get("validated_report", {}).get("findings", [{}])[0].get("claim_validation", {}).get("status"),
                "calculation_evidence": item["baseline_retrieval"]["validated_report"]["findings"][0]["claim_validation"]["calculation_evidence_ids"],
                "retrieval_order_changed": item["module_comparison"]["retrieval_order_changed"],
                "module_rows": len(item["module_ablation_matrix"]),
                "module_interventions_executed": sum(row["intervention_applied"] for row in item["module_ablation_matrix"]),
            }
            for item in query_runs
        ],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
