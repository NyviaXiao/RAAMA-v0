"""Pinned, offline-first Docling OCR/layout/table extraction for local PDFs."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any

from adaptive_mas.runtime_limits import apply_thread_limits, process_thread_limit


DOCLING_VERSION = "2.129.0"
RAPIDOCR_VERSION = "3.9.2"
LAYOUT_REPO = "docling-project/docling-layout-heron"
LAYOUT_REVISION = "8f39ad3c0b4c58e9c2d2c84a38465abf757272d8"
TABLEFORMER_REPO = "docling-project/docling-models"
TABLEFORMER_REVISION = "fc0f2d45e2218ea24bce5045f58a389aed16dc23"
TABLEFORMER_TAG = "v2.3.0"
ASSET_MANIFEST = "raama_docling_assets.json"


def _file_manifest(root: Path) -> list[dict[str, object]]:
    entries = []
    for path in sorted(item for item in root.rglob("*") if item.is_file() and item.name != ASSET_MANIFEST):
        digest = hashlib.sha256()
        with path.open("rb") as source:
            for part in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(part)
        entries.append({"path": path.relative_to(root).as_posix(), "size": path.stat().st_size, "sha256": digest.hexdigest()})
    return entries


def download_docling_assets(asset_dir: str | Path, *, progress: bool = True) -> dict[str, object]:
    """Fetch only the pinned layout, TableFormer, and Chinese RapidOCR assets."""
    apply_thread_limits()
    try:
        import docling
        from docling.models.stages.ocr.rapid_ocr_model import RapidOcrModel
        from docling.models.stages.table_structure.table_structure_model import TableStructureModel
        from docling.utils.model_downloader import download_hf_model
    except ImportError as error:
        raise RuntimeError("install the document-ai optional dependencies before fetching Docling assets") from error
    if getattr(docling, "__version__", None) != DOCLING_VERSION:
        raise RuntimeError(f"Docling {DOCLING_VERSION} is required for the pinned asset layout")
    from importlib.metadata import version as distribution_version
    if distribution_version("rapidocr") != RAPIDOCR_VERSION:
        raise RuntimeError(f"RapidOCR {RAPIDOCR_VERSION} is required for the pinned asset layout")

    root = Path(asset_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    download_hf_model(
        repo_id=LAYOUT_REPO,
        revision=LAYOUT_REVISION,
        local_dir=root / LAYOUT_REPO.replace("/", "--"),
        progress=progress,
    )
    download_hf_model(
        repo_id=TABLEFORMER_REPO,
        revision=TABLEFORMER_REVISION,
        local_dir=root / TableStructureModel._model_repo_folder,
        progress=progress,
    )
    RapidOcrModel.download_models(
        backend="onnxruntime",
        lang="ch",
        local_dir=root / RapidOcrModel._model_repo_folder,
        progress=progress,
    )

    manifest: dict[str, object] = {
        "schema_version": "raama.docling-assets.v1",
        "docling_version": DOCLING_VERSION,
        "layout": {"repo_id": LAYOUT_REPO, "revision": LAYOUT_REVISION},
        "tableformer": {"repo_id": TABLEFORMER_REPO, "revision": TABLEFORMER_REVISION, "tag": TABLEFORMER_TAG},
        "rapidocr": {"package": "rapidocr", "version": RAPIDOCR_VERSION, "backend": "onnxruntime", "language": "ch"},
        "files": _file_manifest(root),
    }
    path = root / ASSET_MANIFEST
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"asset_dir": str(root), "manifest_path": str(path), "file_count": len(manifest["files"])}


def _require_pinned_assets(asset_dir: Path) -> dict[str, Any]:
    manifest_path = asset_dir / ASSET_MANIFEST
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        raise RuntimeError("pinned Docling assets are missing; run the explicit model-fetch command first") from None
    if (
        manifest.get("schema_version") != "raama.docling-assets.v1"
        or manifest.get("docling_version") != DOCLING_VERSION
        or manifest.get("layout") != {"repo_id": LAYOUT_REPO, "revision": LAYOUT_REVISION}
        or manifest.get("tableformer") != {"repo_id": TABLEFORMER_REPO, "revision": TABLEFORMER_REVISION, "tag": TABLEFORMER_TAG}
        or manifest.get("rapidocr") != {"package": "rapidocr", "version": RAPIDOCR_VERSION, "backend": "onnxruntime", "language": "ch"}
        or not manifest.get("files")
    ):
        raise RuntimeError("Docling asset manifest does not match the pinned model contract")
    actual = _file_manifest(asset_dir)
    if actual != manifest["files"]:
        raise RuntimeError("Docling assets failed the recorded path, size, or SHA-256 integrity check")
    return manifest


def _bbox(item: Any) -> list[float] | None:
    provenance = getattr(item, "prov", None) or []
    if not provenance:
        return None
    box = getattr(provenance[0], "bbox", None)
    if box is None:
        return None
    try:
        left, top, right, bottom = float(box.l), float(box.t), float(box.r), float(box.b)
    except (AttributeError, TypeError, ValueError):
        return None
    return [left, min(top, bottom), right, max(top, bottom)]


def _docling_header_contract(cells: list[Any]) -> tuple[int, str | None]:
    """Count only column-heading rows; row-header stubs remain data rows."""
    column_rows = {
        int(cell.start_row_offset_idx)
        for cell in cells
        if bool(getattr(cell, "column_header", False))
    }
    if not column_rows:
        return 1, "column_headers_not_detected"
    last = max(column_rows)
    if column_rows != set(range(last + 1)):
        return 1, "non_contiguous_column_header_rows"
    return max(1, last + 1), None


def _explicit_unit_text(parts: list[str]) -> str | None:
    """Read a unit only when the source explicitly labels it as a unit."""
    lines = [line.strip() for part in parts for line in str(part).splitlines()]
    for index, line in enumerate(lines):
        match = re.fullmatch(r"(?:单位|unit)\s*[:：]\s*(.*)", line, flags=re.IGNORECASE)
        if match is None:
            continue
        value = match.group(1).strip()
        if value:
            return value
        for following in lines[index + 1 :]:
            if not following:
                continue
            if following.startswith("|") or set(following) <= {"-", "|", " ", ":"}:
                return None
            return following
        return None
    return None


def _period_header(value: str) -> str | None:
    """Keep the source's period label when a table heading states one clearly."""
    text = value.strip()
    period_marker = r"年|FY|Q[1-4]|季度|截至|期末|(?:1[-—~至]3|1[-—~至]6|1[-—~至]9|4[-—~至]6|7[-—~至]9|10[-—~至]12)月|12月31日"
    if re.search(r"(?:19|20)\d{2}", text) and re.search(period_marker, text, flags=re.IGNORECASE):
        return text
    return None


_STATEMENT_TITLE = re.compile(
    r"(?:资产负债表|利润表|现金流量表|所有者权益变动表|主要会计数据和财务指标|财务指标)",
)
_TABLE_HEADER_WORDS = (
    "项目", "科目", "指标", "本期金额", "上期金额", "期末余额", "期初余额",
    "期末数", "期初数", "报告期",
)


def _docling_table_segments(cells: list[Any]) -> list[dict[str, Any]]:
    """Separate a merged Docling table only at explicit headers or statement titles."""
    rows: dict[int, list[Any]] = {}
    for cell in cells:
        rows.setdefault(int(cell.start_row_offset_idx), []).append(cell)
    if not rows:
        return []
    for row_cells in rows.values():
        row_cells.sort(key=lambda cell: int(cell.start_col_offset_idx))

    column_header_rows = sorted({
        int(cell.start_row_offset_idx)
        for cell in cells
        if bool(getattr(cell, "column_header", False))
    })
    header_runs: list[tuple[int, int]] = []
    for row in column_header_rows:
        if not header_runs or row != header_runs[-1][1] + 1:
            header_runs.append((row, row))
        else:
            header_runs[-1] = (header_runs[-1][0], row)

    def row_text(row_index: int) -> str:
        return " ".join(str(getattr(cell, "text", "")).strip() for cell in rows[row_index] if str(getattr(cell, "text", "")).strip())

    def is_statement_title(row_index: int) -> bool:
        nonempty = [str(getattr(cell, "text", "")).strip() for cell in rows[row_index] if str(getattr(cell, "text", "")).strip()]
        return len(nonempty) <= 2 and any(_STATEMENT_TITLE.search(value) for value in nonempty)

    def is_header_like(row_index: int) -> bool:
        texts = [str(getattr(cell, "text", "")).strip() for cell in rows[row_index] if str(getattr(cell, "text", "")).strip()]
        return (
            len(texts) >= 2
            and (
                any(re.fullmatch(r"(?:项目|科目|指标|项目名称|会计科目)", text) for text in texts)
                or any(word in " ".join(texts) for word in _TABLE_HEADER_WORDS[3:])
            )
        )

    def is_text_preamble(row_index: int) -> bool:
        return (
            is_statement_title(row_index)
            or _period_header(row_text(row_index)) is not None
            or _explicit_unit_text([row_text(row_index)]) is not None
            or not any(re.search(r"[-+]?\d", str(getattr(cell, "text", ""))) for cell in rows[row_index])
        )

    first_header_end = min(rows) - 1
    if header_runs:
        first_start, first_end = header_runs[0]
        preamble_rows = [row for row in rows if row < first_start]
        if first_start <= min(rows) + 1 or all(is_text_preamble(row) for row in preamble_rows):
            first_header_end = first_end
    boundaries: set[int] = set()
    for run_start, _run_end in header_runs[1:]:
        if run_start > first_header_end:
            title_row = run_start - 1 if run_start - 1 in rows and is_statement_title(run_start - 1) else run_start
            boundaries.add(title_row)
    for row_index in sorted(rows):
        if row_index <= first_header_end:
            continue
        if is_statement_title(row_index):
            boundaries.add(row_index)
        elif is_header_like(row_index) and not (
            row_index - 1 in rows and is_statement_title(row_index - 1)
        ):
            boundaries.add(row_index)

    row_indexes = sorted(rows)
    starts = sorted({row_indexes[0], *boundaries})
    segments: list[dict[str, Any]] = []
    for segment_index, start in enumerate(starts):
        end = starts[segment_index + 1] - 1 if segment_index + 1 < len(starts) else row_indexes[-1]
        segment_rows = [row for row in row_indexes if start <= row <= end]
        segment_cells = [cell for row in segment_rows for cell in rows[row]]
        segment_header_rows = [row for row in column_header_rows if start <= row <= end]
        quarantine = None
        if not segment_header_rows:
            header_rows = 1
            quarantine = "column_headers_not_detected"
        else:
            header_start = segment_header_rows[0]
            contiguous_end = header_start
            for row in segment_header_rows[1:]:
                if row != contiguous_end + 1:
                    break
                contiguous_end = row
            preamble_rows = [row for row in segment_rows if row < header_start]
            preamble_is_text = all(is_text_preamble(row) for row in preamble_rows)
            if not preamble_is_text:
                header_rows = max(1, contiguous_end - start + 1)
                quarantine = "data_precedes_column_headers"
            else:
                header_rows = contiguous_end - start + 1
            if contiguous_end < max(segment_header_rows):
                quarantine = quarantine or "non_contiguous_column_header_rows"
        title_values = [
            str(getattr(cell, "text", "")).strip()
            for row in segment_rows
            if is_statement_title(row)
            for cell in rows[row]
            if str(getattr(cell, "text", "")).strip()
        ]
        segments.append({
            "start_row": start,
            "end_row": end,
            "cells": segment_cells,
            "column_header_rows": tuple(segment_header_rows),
            "header_rows": max(1, header_rows),
            "quarantine_reason": quarantine,
            "title": " ".join(dict.fromkeys(title_values)) or None,
        })
    return segments


def _docling_segment_records(
    segment: dict[str, Any],
    *,
    table_id: str,
    page_number: int,
    title_parts: list[str],
    unit_context: list[str],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    from adaptive_mas.industrial_rag.parser import normalize_financial_cell

    source_cells = list(segment["cells"])
    row_offset = int(segment["start_row"])
    header_rows = int(segment["header_rows"])
    title_values = [value for value in (*title_parts, segment.get("title")) if value]
    title = " · ".join(dict.fromkeys(title_values)) or None
    unit_parts = unit_context + title_values + [str(getattr(cell, "text", "")) for cell in source_cells]
    unit_text = _explicit_unit_text(unit_parts)
    table_cells: list[dict[str, Any]] = []
    for cell in source_cells:
        row_index = int(cell.start_row_offset_idx) - row_offset
        column_index = int(cell.start_col_offset_idx)
        raw = str(getattr(cell, "text", ""))
        normalized = normalize_financial_cell(raw)
        kind = str(normalized["value_kind"])
        if kind == "unknown":
            kind = "text"
        table_cells.append({
            "cell_id": f"{table_id}-r{row_index}-c{column_index}",
            "table_id": table_id,
            "row_index": row_index,
            "column_index": column_index,
            "raw_text": raw,
            "normalized_value": normalized.get("normalized_value"),
            "value_kind": kind,
            "page_number": page_number,
            "unit": unit_text,
            "multiplier": normalized.get("multiplier"),
            "currency": None,
            "accounting_basis": None,
            "report_period": None,
            "footnote_refs": [],
        })

    header_row_indexes = {
        int(row_index) - row_offset for row_index in segment.get("column_header_rows", ())
    }
    headers_by_column: dict[int, list[str]] = {}
    for cell in table_cells:
        if int(cell["row_index"]) in header_row_indexes:
            headers_by_column.setdefault(int(cell["column_index"]), []).append(str(cell["raw_text"]).strip())
    period_by_column: dict[int, str] = {}
    for column, values in headers_by_column.items():
        periods = tuple(dict.fromkeys(
            period for value in values
            if (period := _period_header(value)) is not None
        ))
        if len(periods) == 1:
            period_by_column[column] = periods[0]
    for cell in table_cells:
        if int(cell["row_index"]) >= header_rows:
            cell["report_period"] = period_by_column.get(int(cell["column_index"]))

    table = {
        "table_id": table_id,
        "title": title,
        "header_rows": header_rows,
        "page_start": page_number,
        "page_end": page_number,
        "unit_text": unit_text,
        "accounting_basis": None,
        "report_period": None,
        "continued_from_table_id": None,
        "continuation_confidence": None,
        "row_group_quarantine_reason": segment.get("quarantine_reason"),
    }
    return table, table_cells


def parse_pdf_with_docling(payload: bytes, *, asset_dir: str | Path, max_pages: int, timeout_seconds: float) -> dict[str, Any]:
    """Parse a PDF on CPU using local model files and return source-bound geometry."""
    apply_thread_limits()
    artifacts = Path(asset_dir).resolve()
    _require_pinned_assets(artifacts)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["DOCLING_ARTIFACTS_PATH"] = str(artifacts)
    try:
        import docling
        from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.layout_model_specs import LayoutModelConfig
        from docling.datamodel.pipeline_options import (
            LayoutOptions,
            PdfPipelineOptions,
            RapidOcrOptions,
            TableFormerMode,
            TableStructureOptions,
        )
        from docling.document_converter import DocumentConverter, PdfFormatOption
        from docling_core.types.doc.document import TableItem
    except ImportError as error:
        raise RuntimeError("install the document-ai optional dependencies to use the Docling parser") from error
    if getattr(docling, "__version__", None) != DOCLING_VERSION:
        raise RuntimeError(f"Docling {DOCLING_VERSION} is required by the pinned asset manifest")
    options = PdfPipelineOptions(
        document_timeout=timeout_seconds,
        accelerator_options=AcceleratorOptions(num_threads=process_thread_limit(), device=AcceleratorDevice.CPU),
        enable_remote_services=False,
        allow_external_plugins=False,
        artifacts_path=artifacts,
        do_ocr=True,
        ocr_options=RapidOcrOptions(lang=["ch"], backend="onnxruntime"),
        do_table_structure=True,
        table_structure_options=TableStructureOptions(do_cell_matching=True, mode=TableFormerMode.ACCURATE),
        layout_options=LayoutOptions(model_spec=LayoutModelConfig(
            name="docling_layout_heron", repo_id=LAYOUT_REPO, revision=LAYOUT_REVISION,
        )),
        ocr_batch_size=1,
        layout_batch_size=1,
        table_batch_size=1,
        queue_max_size=2,
    )
    converter = DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)})
    with tempfile.TemporaryDirectory(prefix="raama-docling-") as temporary_dir:
        source = Path(temporary_dir) / "source.pdf"
        source.write_bytes(payload)
        conversion = converter.convert(source, max_num_pages=max_pages)

    document = conversion.document
    page_numbers = sorted(int(value) for value in document.pages)
    if len(page_numbers) > max_pages:
        raise RuntimeError("Docling returned more pages than the configured page limit")
    page_text: dict[int, list[str]] = {number: [] for number in page_numbers}
    table_context_by_page: dict[int, list[str]] = {number: [] for number in page_numbers}
    raw_tables: list[dict[str, Any]] = []
    raw_cells: list[dict[str, Any]] = []
    content_blocks: dict[int, list[dict[str, Any]]] = {number: [] for number in page_numbers}
    table_index = 0
    block_index: dict[int, int] = {number: 0 for number in page_numbers}
    block_total = 0
    table_row_total = 0

    from adaptive_mas.industrial_rag.parser import _table_row_group_blocks

    for item, _level in document.iterate_items():
        provenance = getattr(item, "prov", None) or []
        if not provenance:
            continue
        page_number = int(getattr(provenance[0], "page_no", 0) or 0)
        if page_number not in page_text:
            continue
        if isinstance(item, TableItem):
            captions = [str(getattr(value, "text", "")).strip() for value in getattr(item, "captions", [])]
            caption_title = " ".join(value for value in captions if value) or None
            data_cells = list(getattr(getattr(item, "data", None), "table_cells", []) or [])
            if len(raw_cells) + len(data_cells) > 1_000_000:
                raise RuntimeError("Docling table exceeds the configured cell limit")
            segments = _docling_table_segments(data_cells)
            prior_context = list(table_context_by_page[page_number])
            item_table_cells: list[dict[str, Any]] = []
            for segment_index, segment in enumerate(segments):
                table_index += 1
                table_id = f"docling-table-{table_index}-p{page_number}"
                title_parts = [value for value in (caption_title, segment.get("title")) if value]
                table_record, table_cells = _docling_segment_records(
                    segment,
                    table_id=table_id,
                    page_number=page_number,
                    title_parts=title_parts,
                    unit_context=prior_context if segment_index == 0 else [],
                )
                table_row_total += max((int(cell["row_index"]) for cell in table_cells), default=-1) + 1
                if table_row_total > 100_000:
                    raise RuntimeError("Docling table exceeds the configured row limit")
                raw_cells.extend(table_cells)
                item_table_cells.extend(table_cells)
                raw_tables.append(table_record)
            try:
                table_text = item.export_to_markdown(doc=document).strip()
            except Exception:
                table_text = ""
            if table_text:
                page_text[page_number].append(table_text)
            elif item_table_cells:
                page_text[page_number].append("\n".join(
                    " | ".join(
                        str(cell["raw_text"]) for cell in sorted(
                            (value for value in item_table_cells if int(value["row_index"]) == row),
                            key=lambda value: int(value["column_index"]),
                        )
                    )
                    for row in sorted({int(value["row_index"]) for value in item_table_cells})
                ))
            table_context_by_page[page_number] = []
            continue

        text = str(getattr(item, "text", "") or "").strip()
        if not text:
            continue
        page_text[page_number].append(text)
        table_context_by_page[page_number].append(text)
        block_index[page_number] += 1
        block_total += 1
        if block_total > 50_000:
            raise RuntimeError("Docling output exceeds the configured block limit")
        label = getattr(getattr(item, "label", None), "value", None) or str(getattr(item, "label", "text"))
        label = str(label).casefold().replace(" ", "_")
        content_blocks[page_number].append({
            "block_id": f"docling-p{page_number}-b{block_index[page_number]}",
            "page_number": page_number,
            "block_type": label,
            "text": text[:4800],
            "char_start": None,
            "char_end": None,
            "parent_block_id": f"page-{page_number}",
            "previous_block_id": None,
            "next_block_id": None,
            "parser_confidence": 0.8,
            "quarantine_reason": None,
            "page_kind": "pdf_docling",
            "citation_text_exact": False,
            "bbox": _bbox(item),
        })

    pages = [(number, "\n".join(value).strip(), "pdf_docling") for number, value in sorted(page_text.items()) if any(part.strip() for part in value)]
    if not pages:
        return {
            "status": "quarantined", "error_code": "no_extractable_text",
            "error": "Docling OCR/layout pipeline produced no readable text or table cells",
            "pages": [], "blocks": [], "tables": [], "cells": [],
            "parser_name": "docling_rapidocr_tableformer",
            "parser_version": f"docling-{DOCLING_VERSION}+rapidocr-{RAPIDOCR_VERSION}+layout-{LAYOUT_REVISION}+tableformer-{TABLEFORMER_REVISION}",
            "parser_confidence": 0.0,
            "quality": {"source_page_count": len(document.pages), "extracted_page_count": 0, "signal": "structural_page_coverage_heuristic_not_accuracy"},
        }

    blocks: list[dict[str, Any]] = []
    for number, text, _kind in pages:
        blocks.append({
            "block_id": f"page-{number}", "page_number": number, "block_type": "page",
            "text": text[:4800], "char_start": None, "char_end": None,
            "parent_block_id": None, "previous_block_id": None, "next_block_id": None,
            "parser_confidence": 0.8, "quarantine_reason": None, "page_kind": "pdf_docling",
            "citation_text_exact": False, "bbox": None,
        })
        blocks.extend(content_blocks[number])
    blocks.extend(_table_row_group_blocks(raw_tables, raw_cells))
    source_page_count = len(document.pages)
    confidence = len(pages) / source_page_count if source_page_count else 0.0
    if confidence < 0.6:
        return {
            "status": "quarantined", "error_code": "low_page_coverage",
            "error": "fewer than 60 percent of source pages yielded readable text or table cells",
            "pages": [], "blocks": [], "tables": [], "cells": [],
            "parser_name": "docling_rapidocr_tableformer",
            "parser_version": f"docling-{DOCLING_VERSION}+rapidocr-{RAPIDOCR_VERSION}+layout-{LAYOUT_REVISION}+tableformer-{TABLEFORMER_REVISION}",
            "parser_confidence": confidence,
            "quality": {"source_page_count": source_page_count, "extracted_page_count": len(pages), "signal": "structural_page_coverage_heuristic_not_accuracy"},
        }
    return {
        "status": "parsed", "error_code": None, "error": None,
        "pages": [{"page_number": number, "text": text, "kind": kind} for number, text, kind in pages],
        "blocks": blocks, "tables": raw_tables, "cells": raw_cells,
        "parser_name": "docling_rapidocr_tableformer",
        "parser_version": f"docling-{DOCLING_VERSION}+rapidocr-{RAPIDOCR_VERSION}+layout-{LAYOUT_REVISION}+tableformer-{TABLEFORMER_REVISION}",
        "parser_confidence": confidence,
        "quality": {"source_page_count": source_page_count, "extracted_page_count": len(pages), "signal": "structural_page_coverage_heuristic_not_accuracy"},
    }
