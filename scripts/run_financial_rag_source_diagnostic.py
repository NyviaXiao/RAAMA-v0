"""Run one local public-filing parse, index, retrieval, and exact-location diagnostic."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from adaptive_mas.industrial_rag.cli import build_release  # noqa: E402
from adaptive_mas.industrial_rag.document_ai import ASSET_MANIFEST  # noqa: E402
from adaptive_mas.industrial_rag.evaluation import evaluate_manifest  # noqa: E402
from adaptive_mas.industrial_rag.models import Document, DocumentBundle, DocumentVersion, QueryPlan  # noqa: E402
from adaptive_mas.industrial_rag.parser import (  # noqa: E402
    _table_row_group_blocks,
    parse_isolated,
    records_from_parse,
)
from adaptive_mas.industrial_rag.planning import normalize_period, requested_metric_specs, requested_periods  # noqa: E402
from adaptive_mas.industrial_rag.repository import IndustrialRagStore  # noqa: E402
from adaptive_mas.industrial_rag.retrieval import IndustrialRagService, _financial_observations  # noqa: E402


SUPERVISION = Path(".tmp/industrial_rag_supervision_20261025")
DEFAULT_MANIFEST = ROOT / "benchmarks/financial_rag/seed_manifest.json"


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _source_case(manifest_path: Path, case_id: str) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    case = next((item for item in manifest.get("cases", []) if item.get("case_id") == case_id), None)
    if not isinstance(case, dict):
        raise ValueError(f"case_id {case_id!r} is absent from the manifest")
    case_sources = case.get("source_records")
    if not isinstance(case_sources, list) or len(case_sources) != 1 or not isinstance(case_sources[0], dict):
        raise ValueError("this diagnostic requires exactly one source record")
    source_id = case_sources[0].get("source_id")
    manifest_sources = manifest.get("source_records", [])
    source_metadata = next(
        (item for item in manifest_sources if isinstance(item, dict) and item.get("source_id") == source_id),
        None,
    )
    if source_metadata is None:
        raise ValueError(f"source_id {source_id!r} has no top-level manifest source record")
    return manifest, case, dict(source_metadata) | dict(case_sources[0])


def _parse_page_range(value: str) -> tuple[int, int]:
    match = re.fullmatch(r"(\d+)(?:-(\d+))?", value.strip())
    if match is None:
        raise ValueError("targeted-page-range must be N or N-M using 1-based physical page numbers")
    start = int(match.group(1))
    end = int(match.group(2) or start)
    if start < 1 or end < start:
        raise ValueError("targeted-page-range must be an ordered positive page range")
    return start, end


def _make_page_subset(payload: bytes, first_page: int, last_page: int) -> bytes:
    from io import BytesIO
    from pypdf import PdfReader, PdfWriter

    reader = PdfReader(BytesIO(payload))
    if first_page < 1 or last_page < first_page or last_page > len(reader.pages):
        raise ValueError("targeted page range must be within the source PDF")
    writer = PdfWriter()
    for page in reader.pages[first_page - 1:last_page]:
        writer.add_page(page)
    output = BytesIO()
    writer.write(output)
    return output.getvalue()


def _case_eval_artifacts(
    manifest: Mapping[str, Any],
    case: Mapping[str, Any],
    retrieved: list[dict[str, Any]],
    output_dir: Path,
    query_binding: Mapping[str, Any],
) -> dict[str, Any]:
    exact_metrics = {
        item.metric for item in requested_metric_specs(str(case["question"]))
        if item.match_status == "exact"
    }
    requested_period_keys = {
        normalize_period(period)
        for period in requested_periods(str(case["question"]), tuple(case.get("report_periods", [])))
    }
    requested_period_keys.discard(None)
    answerable_from_retrieval = any(
        observation["metric"] in exact_metrics
        and (not requested_period_keys or observation["period"] in requested_period_keys)
        for item in retrieved
        for observation in _financial_observations(item)
    )
    case_manifest = {key: value for key, value in manifest.items() if key != "cases"}
    case_manifest["cases"] = [dict(case)]
    case_manifest["synthetic_adversarial_cases"] = []
    manifest_path = output_dir / "single_case_manifest.json"
    predictions_path = output_dir / "predictions.json"
    _write_json(manifest_path, case_manifest)
    _write_json(predictions_path, {
        "schema_version": "adaptive-mas.financial-rag-predictions.v1",
        "run_id": output_dir.name,
        "model_revisions": {
            "retrieval": "local_bm25_reference_chinese_bigram_analyzer+exact-metric-period-table-row-priority",
        },
        "query_binding": dict(query_binding),
        "cases": [{
            "case_id": case["case_id"],
            "predicted_answerability": "answerable" if answerable_from_retrieval else "unanswerable",
            "retrieved": retrieved,
        }],
    })
    report = evaluate_manifest(manifest_path, predictions_path, k_values=(1, 3, 5, 10))
    _write_json(output_dir / "exact_location_score.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--case-id", default="catl2024_annual_revenue")
    parser.add_argument("--source", type=Path, help="Override the source record's local_path")
    parser.add_argument("--targeted-page-range", default="1-9", help="Bounded physical source pages for Docling parsing, as N or N-M")
    parser.add_argument("--parser-table-id", default="docling-table-10-p9", help="Manually reviewed raw ID from targeted Docling output")
    parser.add_argument(
        "--docling-assets",
        type=Path,
        default=ROOT / SUPERVISION / "docling-models",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / SUPERVISION / "catl_source_to_score_runs",
    )
    parser.add_argument("--run-id", help="Unique run directory and immutable local release ID")
    parser.add_argument("--parse-timeout-seconds", type=float, default=300)
    args = parser.parse_args()

    run_id = args.run_id or datetime.now(timezone.utc).strftime("catl-public-e2e-%Y%m%dT%H%M%SZ")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}", run_id):
        raise ValueError("run-id must be a safe immutable identifier")
    output_dir = args.output_root / run_id
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")
    output_dir.mkdir(parents=True)

    manifest, case, source_record = _source_case(args.manifest.resolve(), args.case_id)
    target_first_page, target_last_page = _parse_page_range(args.targeted_page_range)
    source_path = args.source.resolve() if args.source else (ROOT / source_record["local_path"]).resolve()
    payload = source_path.read_bytes()
    content_sha256 = hashlib.sha256(payload).hexdigest()
    if content_sha256 != source_record.get("content_sha256"):
        raise ValueError("local source SHA-256 does not match the manifest source record")

    from io import BytesIO
    from pypdf import PdfReader

    source_page_count = len(PdfReader(BytesIO(payload)).pages)
    if target_last_page > source_page_count:
        raise ValueError("targeted page range exceeds the original source page count")

    full_parse = parse_isolated(
        payload,
        source_path.name,
        scratch_root=output_dir / "parser_scratch",
        timeout_seconds=args.parse_timeout_seconds,
        pdf_backend="pypdf",
    )
    if full_parse.get("status") != "parsed" or len(full_parse.get("pages", [])) != source_page_count:
        raise RuntimeError("full-source pypdf text extraction did not cover every physical page")
    _write_json(output_dir / "full_source_pypdf_parse.json", full_parse)

    targeted_parse = parse_isolated(
        _make_page_subset(payload, target_first_page, target_last_page),
        f"{source_path.stem}-pages{target_first_page:03d}-{target_last_page:03d}.pdf",
        scratch_root=output_dir / "parser_scratch",
        timeout_seconds=args.parse_timeout_seconds,
        pdf_backend="docling",
        docling_artifacts_path=args.docling_assets,
        parser_timeout_seconds=args.parse_timeout_seconds,
    )
    if targeted_parse.get("status") != "parsed":
        raise RuntimeError("targeted Docling parse did not complete")
    _write_json(output_dir / "targeted_docling_parse.json", targeted_parse)

    parser_tables = [
        table for table in targeted_parse.get("tables", [])
        if table.get("table_id") == args.parser_table_id
    ]
    if len(parser_tables) != 1:
        raise ValueError(f"expected one manually selected parser table {args.parser_table_id!r}")
    parser_table = parser_tables[0]
    parser_page = int(parser_table["page_start"])
    original_page = target_first_page + parser_page - 1
    if original_page > target_last_page:
        raise ValueError("selected parser table falls outside the explicitly targeted physical page range")
    selected_table = dict(parser_table)
    canonical_raw_table_id = re.sub(r"-p\d+$", f"-p{original_page}", args.parser_table_id)
    selected_table.update({
        "table_id": canonical_raw_table_id,
        "page_start": original_page,
        "page_end": original_page,
    })
    selected_cells = [
        cell for cell in targeted_parse.get("cells", [])
        if cell.get("table_id") == args.parser_table_id
        and cell.get("page_number") == parser_page
    ]
    if not selected_cells:
        raise ValueError("selected Docling table has no cells on the requested source page")
    mapped_cells = []
    for raw_cell in selected_cells:
        cell = dict(raw_cell)
        cell["cell_id"] = str(cell["cell_id"]).replace(args.parser_table_id, canonical_raw_table_id, 1)
        cell["table_id"] = canonical_raw_table_id
        cell["page_number"] = original_page
        mapped_cells.append(cell)
    selected_cells = mapped_cells
    selected_tables = [selected_table]

    version_id = f"v-{content_sha256[:24]}"
    canonical_table_id = f"tbl-{version_id}-{canonical_raw_table_id}"
    merged_parse = dict(full_parse)
    merged_parse["tables"] = [selected_table]
    merged_parse["cells"] = selected_cells
    merged_parse["blocks"] = list(full_parse.get("blocks", [])) + _table_row_group_blocks(selected_tables, selected_cells)
    merged_parse["parser_name"] = "hybrid_pypdf_full_source_docling_targeted_table"
    merged_parse["parser_version"] = (
        f"pypdf-full:{full_parse.get('parser_version')}+"
        f"docling-targeted:{targeted_parse.get('parser_version')}"
    )
    merged_parse["source_sha256"] = content_sha256
    merged_parse["targeted_parse_scope"] = {
        "original_source_sha256": content_sha256,
        "original_source_page_count": source_page_count,
        "full_source_backend": full_parse.get("parser_name"),
        "full_docling_parse": False,
        "targeted_backend": targeted_parse.get("parser_name"),
        "targeted_parser_version": targeted_parse.get("parser_version"),
        "targeted_physical_pages": list(range(target_first_page, target_last_page + 1)),
        "selected_parser_page_index": parser_page,
        "selected_original_physical_page": original_page,
        "selected_raw_parser_table_id": args.parser_table_id,
        "mapped_physical_parser_table_id": canonical_raw_table_id,
        "canonical_version_scoped_table_id": canonical_table_id,
        "table_header_rows": selected_table.get("header_rows"),
    }
    _write_json(output_dir / "merged_hybrid_parse.json", merged_parse)

    records = records_from_parse(merged_parse, version_id=version_id)
    report_period = next(iter(case.get("report_periods", [])), None)
    document = Document(
        document_id=f"doc-{source_record['source_id']}",
        title=str(source_record.get("title") or f"CNINFO annual report {source_record['source_id']}"),
        document_type=str(source_record.get("document_type") or "annual_report"),
        security_code=str(case["security_code"]),
        entity_name=source_record.get("issuer"),
        report_period=report_period,
    )
    now = datetime.now(timezone.utc)
    version = DocumentVersion(
        version_id=version_id,
        source_id=str(source_record["source_id"]),
        content_sha256=content_sha256,
        source_url=str(source_record["source_url"]),
        rights_note=str(source_record["rights_note"]),
        first_available_at=source_record.get("first_available_at"),
        published_at=source_record.get("published_at"),
        captured_at=source_record["captured_at"],
        ingested_at=now,
        time_basis=str(source_record.get("time_basis", "retrospective_proxy")),  # type: ignore[arg-type]
        parser_name=str(merged_parse["parser_name"]),
        parser_version=str(merged_parse["parser_version"]),
        parser_confidence=min(
            float(full_parse.get("parser_confidence") or 0.86),
            float(targeted_parse.get("parser_confidence") or 0.86),
        ),
        parser_status="parsed",
    )
    database_path = output_dir / "industrial_rag.sqlite3"
    store = IndustrialRagStore(database_path)
    bundle = DocumentBundle(
        document=document,
        version=version,
        payload=payload,
        sections=records["sections"],
        spans=records["spans"],
        blocks=records["blocks"],
        tables=records["tables"],
        cells=records["cells"],
        facts=(),
        read_principals=("local-public",),
    )
    ingest = store.ingest_bundle(bundle)
    release = build_release(
        release_id=run_id,
        version_ids=[version_id],
        database_path=database_path,
        object_root=output_dir / "objects",
        actor_id="local-supervision",
        reason="Local full-source text plus targeted Docling table diagnostic",
    )

    service = IndustrialRagService(store, allow_retrospective_diagnostic=True)
    query_plan = QueryPlan(
        plan_id="public-filing-source-e2e",
        query=str(case["question"]),
        security_code=str(case["security_code"]),
        as_of=str(case["as_of"]),
        principal_id="local-public",
        task_type="table",
        report_periods=tuple(str(value) for value in case.get("report_periods", [])),
        release_id=run_id,
        channels=("bm25",),
        top_k=10,
        time_mode="retrospective_diagnostic",
    )
    search = service.search(query_plan, use_cache=False)
    retrieved: list[dict[str, Any]] = []
    reopened_rows: list[dict[str, Any]] = []
    for raw_item in search.get("retrieved", []):
        item = dict(raw_item)
        citation = dict(item.get("citation") or {})
        table_id, row_start, row_end = citation.get("table_id"), citation.get("row_start"), citation.get("row_end")
        if isinstance(table_id, str) and isinstance(row_start, int) and isinstance(row_end, int):
            try:
                expanded = service.read_table_row(query_plan, table_id, row_start, row_end)
                item["citation"] = {**citation, **expanded["citation"]}
                item["reopened_table_row"] = expanded
                reopened_rows.append({
                    "evidence_id": item.get("evidence_id"),
                    "table_id": table_id,
                    "row_start": row_start,
                    "row_end": row_end,
                    "supporting_cell_count": len(expanded["citation"].get("supporting_cells", [])),
                })
            except (PermissionError, FileNotFoundError, ValueError) as error:
                item["table_reopen_status"] = str(error)
        retrieved.append(item)

    _write_json(output_dir / "search.json", search)
    query_binding = dict(search.get("query_binding") or {})
    evaluation = _case_eval_artifacts(manifest, case, retrieved, output_dir, query_binding)
    asset_manifest = args.docling_assets / ASSET_MANIFEST
    run_record = {
        "run_id": run_id,
        "case_id": case["case_id"],
        "query_text": case["question"],
        "case_as_of": case["as_of"],
        "query_binding": query_binding,
        "ingested_at": version.ingested_at.isoformat(),
        "gold_locator_used_for_query": False,
        "source_url": source_record["source_url"],
        "source_sha256": content_sha256,
        "source_bytes": len(payload),
        "source_page_count": source_page_count,
        "full_source_parser": {
            "name": full_parse.get("parser_name"),
            "version": full_parse.get("parser_version"),
            "status": full_parse.get("status"),
            "extracted_pages": len(full_parse.get("pages", [])),
        },
        "targeted_docling_parser": {
            "name": targeted_parse.get("parser_name"),
            "version": targeted_parse.get("parser_version"),
            "status": targeted_parse.get("status"),
            "configured_artifact_manifest_sha256": (
                hashlib.sha256(asset_manifest.read_bytes()).hexdigest() if asset_manifest.is_file() else None
            ),
            "targeted_source_page_range": [target_first_page, target_last_page],
            "selected_parser_page_index": parser_page,
            "selected_original_physical_page": original_page,
            "selected_raw_table_id": args.parser_table_id,
            "mapped_physical_table_id": canonical_raw_table_id,
            "oracle_page_selection": {
                "physical_page": original_page,
                "targeted_source_page_range": [target_first_page, target_last_page],
                "selection_basis": "explicit CLI diagnostic scope; not passed to retrieval",
            },
            "oracle_table_selection": {
                "raw_parser_table_id": args.parser_table_id,
                "mapped_physical_table_id": canonical_raw_table_id,
                "selection_basis": "explicit manually reviewed CLI mapping; not passed to retrieval",
            },
            "table_header_rows": selected_table.get("header_rows"),
            "timeout_seconds": args.parse_timeout_seconds,
            "asset_directory": str(args.docling_assets.resolve()),
        },
        "canonical_table_id": canonical_table_id,
        "ingested_block_count": len(records["blocks"]),
        "ingested_table_count": len(records["tables"]),
        "ingested_cell_count": len(records["cells"]),
        "ingest": ingest,
        "release": release,
        "search_status": search.get("status"),
        "query_binding": query_binding,
        "retrieved_count": len(retrieved),
        "reopened_rows": reopened_rows,
        "exact_location_score": evaluation.get("cases", []),
        "diagnostic_scope": "one viewed public development case; no independent acceptance inference",
    }
    _write_json(output_dir / "run_record.json", run_record)
    print(json.dumps({
        "run_id": run_id,
        "output_dir": str(output_dir),
        "source_sha256": content_sha256,
        "source_page_count": source_page_count,
        "full_source_parser": full_parse.get("parser_name"),
        "targeted_parser_table_id": args.parser_table_id,
        "targeted_original_physical_page": original_page,
        "canonical_table_id": canonical_table_id,
        "search_status": search.get("status"),
        "retrieved_count": len(retrieved),
        "exact_location_score": evaluation.get("cases", []),
        "full_docling_parse": False,
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
