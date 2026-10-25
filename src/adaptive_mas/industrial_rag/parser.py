"""Bounded, isolated format adapters and provenance-preserving document chunking."""

from __future__ import annotations

import argparse
import csv
import hashlib
import html.parser
import json
import os
import re
import subprocess
import sys
import tempfile
import zipfile
from decimal import Decimal
from pathlib import Path
from typing import Any, Mapping, Sequence
from xml.etree import ElementTree

from adaptive_mas.industrial_rag.models import Block, Section, Span, Table, TableCell


MAX_INPUT_BYTES = 100 * 1024 * 1024
MAX_UNCOMPRESSED_BYTES = 250 * 1024 * 1024
MAX_PDF_PAGES = 500
MAX_BLOCK_CHARS = 1200


class ParseError(RuntimeError):
    """Parser failed or the requested format is not supported in this runtime."""


class UnsupportedFormat(ParseError):
    pass


class _TextOnlyHTML(html.parser.HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        if data.strip():
            self.parts.append(data.strip())


def _safe_xml(payload: bytes, source_name: str) -> ElementTree.Element:
    if b"<!DOCTYPE" in payload.upper() or b"<!ENTITY" in payload.upper():
        raise ParseError(f"{source_name} contains a prohibited DTD/entity declaration")
    try:
        return ElementTree.fromstring(payload)
    except ElementTree.ParseError as error:
        raise ParseError(f"{source_name} XML is malformed") from error


def _decode(payload: bytes) -> str:
    try:
        return payload.decode("utf-8-sig")
    except UnicodeDecodeError:
        return payload.decode("gb18030", errors="strict")


def _split_long_text(text: str, max_chars: int) -> list[tuple[str, int, int]]:
    if not text:
        return []
    paragraphs: list[tuple[str, int, int]] = []
    cursor = 0
    for raw in re.split(r"(?<=\n)\s*(?=\S)|(?<=[。！？；])", text):
        if not raw:
            continue
        start = text.find(raw, cursor)
        if start < 0:
            start = cursor
        end = start + len(raw)
        cursor = end
        clean = raw.strip()
        if not clean:
            continue
        if len(clean) <= max_chars:
            paragraphs.append((clean, start, end))
            continue
        left = start
        while left < end:
            right = min(end, left + max_chars)
            if right < end:
                boundary = text.rfind("。", left, right)
                boundary = max(boundary, text.rfind("；", left, right), text.rfind("\n", left, right))
                if boundary > left + max_chars // 2:
                    right = boundary + 1
            part = text[left:right].strip()
            if part:
                actual_start = text.find(part, left, right)
                paragraphs.append((part, actual_start if actual_start >= 0 else left, right))
            left = max(right, left + 1)
    return paragraphs


def _blocks_for_pages(pages: Sequence[tuple[int, str, str]], max_chars: int = MAX_BLOCK_CHARS) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    previous_child: str | None = None
    for page_number, text, page_kind in pages:
        page_id = f"page-{page_number}"
        blocks.append({
            "block_id": page_id, "page_number": page_number, "block_type": "page",
            "text": text[: MAX_BLOCK_CHARS * 4], "char_start": 0,
            "char_end": min(len(text), MAX_BLOCK_CHARS * 4), "parent_block_id": None,
            "previous_block_id": None, "next_block_id": None, "parser_confidence": 0.92,
            "quarantine_reason": None, "page_kind": page_kind,
        })
        pieces = _split_long_text(text, max_chars)
        child_ids: list[str] = []
        for index, (part, start, end) in enumerate(pieces):
            block_id = f"p{page_number}-b{index + 1}"
            child_ids.append(block_id)
            blocks.append({
                "block_id": block_id, "page_number": page_number,
                "block_type": "table_row_group" if page_kind in {"csv", "spreadsheet", "markdown_table"} else "paragraph",
                "text": part, "char_start": start, "char_end": end, "parent_block_id": page_id,
                "previous_block_id": previous_child, "next_block_id": None,
                "parser_confidence": 0.86, "quarantine_reason": None, "page_kind": page_kind,
            })
            if previous_child is not None:
                prior = next(row for row in reversed(blocks) if row["block_id"] == previous_child)
                prior["next_block_id"] = block_id
            previous_child = block_id
    return blocks


def _table_from_rows(rows: Sequence[Sequence[str]], *, version_hint: str = "parsed") -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not rows:
        return [], []
    table_id = f"table-{version_hint}"
    tables = [{"table_id": table_id, "title": None, "header_rows": 1, "page_start": 1, "page_end": 1,
               "unit_text": None, "accounting_basis": None, "report_period": None,
               "continued_from_table_id": None, "continuation_confidence": None}]
    cells: list[dict[str, Any]] = []
    for row_index, row in enumerate(rows):
        for column_index, value in enumerate(row):
            trimmed = value.strip()
            if not trimmed:
                kind, normalized = "blank", None
            elif trimmed in {"-", "—", "–", "－", "--", "——"}:
                kind, normalized = "dash", None
            else:
                normalized = trimmed.replace(",", "").replace("，", "")
                is_percent = normalized.endswith("%")
                if is_percent:
                    normalized = normalized[:-1]
                    kind = "percent"
                else:
                    kind = "text"
                if re.fullmatch(r"\(\s*[-+]?\d+(?:\.\d+)?\s*\)", normalized):
                    inner = re.sub(r"[()\s]", "", normalized)
                    normalized = str(-abs(Decimal(inner)))
                    kind = "percent" if is_percent else "number"
                elif re.fullmatch(r"[-+]?\d+(?:\.\d+)?", normalized):
                    kind = "zero" if normalized in {"0", "0.0", "0.00"} and not is_percent else kind if is_percent else "number"
                else:
                    normalized = None
            cells.append({
                "cell_id": f"{table_id}-r{row_index}-c{column_index}", "table_id": table_id,
                "row_index": row_index, "column_index": column_index, "raw_text": value,
                "normalized_value": normalized, "value_kind": kind, "page_number": 1,
                "unit": None, "multiplier": None, "currency": None, "accounting_basis": None,
                "report_period": None, "footnote_refs": [],
            })
    return tables, cells


def _xlsx_rows(zipped: zipfile.ZipFile) -> list[list[list[str]]]:
    names = set(zipped.namelist())
    shared: list[str] = []
    if "xl/sharedStrings.xml" in names:
        root = _safe_xml(zipped.read("xl/sharedStrings.xml"), "XLSX sharedStrings.xml")
        ns = {"x": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
        for item in root.findall("x:si", ns):
            shared.append("".join(node.text or "" for node in item.findall(".//x:t", ns)))
    sheet_paths = sorted(name for name in names if re.fullmatch(r"xl/worksheets/sheet\d+\.xml", name))
    output = []
    ns = {"x": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    for sheet_path in sheet_paths:
        root = _safe_xml(zipped.read(sheet_path), sheet_path)
        rows: list[list[str]] = []
        for row in root.findall(".//x:sheetData/x:row", ns):
            values: list[str] = []
            for cell in row.findall("x:c", ns):
                cell_ref = cell.attrib.get("r", "")
                col_letters = re.match(r"[A-Z]+", cell_ref)
                col = 0
                if col_letters:
                    for letter in col_letters.group(0):
                        col = col * 26 + ord(letter) - 64
                    col -= 1
                while len(values) <= col:
                    values.append("")
                cell_type = cell.attrib.get("t")
                value_node = cell.find("x:v", ns)
                inline_node = cell.find("x:is", ns)
                value = ""
                if cell_type == "s" and value_node is not None:
                    index = int(value_node.text or "-1")
                    value = shared[index] if 0 <= index < len(shared) else ""
                elif cell_type == "inlineStr" and inline_node is not None:
                    value = "".join(node.text or "" for node in inline_node.findall(".//x:t", ns))
                elif value_node is not None:
                    value = value_node.text or ""
                formula = cell.find("x:f", ns)
                if formula is not None and value:
                    value = f"{value} [formula:{formula.text or ''}]"
                values[col] = value
            rows.append(values)
        output.append(rows)
    return output


def _parse_bytes(payload: bytes, filename: str) -> dict[str, Any]:
    if not payload:
        raise ParseError("source document is empty")
    if len(payload) > MAX_INPUT_BYTES:
        raise ParseError("source document exceeds the configured 100 MiB input limit")
    suffix = Path(filename).suffix.lower()
    pages: list[tuple[int, str, str]] = []
    tables: list[dict[str, Any]] = []
    cells: list[dict[str, Any]] = []
    parser_name = "stdlib"
    parser_version = f"python-stdlib-{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    status = "parsed"
    source_page_count = 1
    parser_confidence = 1.0

    if suffix == ".pdf":
        try:
            import logging
            logging.getLogger("pypdf").setLevel(logging.ERROR)
            import io
            from pypdf import PdfReader
            reader = PdfReader(io.BytesIO(payload), strict=True)
        except Exception:
            raise ParseError("PDF structure could not be parsed") from None
        if reader.is_encrypted:
            raise ParseError("encrypted PDF is unsupported")
        if len(reader.pages) > MAX_PDF_PAGES:
            raise ParseError("PDF exceeds the configured 500 page limit")
        try:
            import pypdf
            parser_version = f"pypdf-{pypdf.__version__}"
        except Exception:
            parser_version = "pypdf-version-unavailable"
        try:
            import fontTools
            parser_version += f"+fonttools-{fontTools.__version__}"
        except Exception:
            parser_version += "+fonttools-unavailable"
        parser_name = "pypdf_text_layer"
        source_page_count = len(reader.pages)
        for number, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()
            if text:
                pages.append((number, text, "pdf_text"))
        if not pages:
            return {"status": "quarantined", "error_code": "ocr_unavailable", "error": "PDF has no readable text layer; OCR adapter is not configured", "pages": [], "blocks": [], "tables": [], "cells": [], "parser_name": parser_name, "parser_version": parser_version, "parser_confidence": 0.0, "quality": {"source_page_count": source_page_count, "extracted_page_count": 0, "signal": "structural_text_coverage_heuristic_not_accuracy"}}
        parser_confidence = len(pages) / source_page_count if source_page_count else 0.0
    elif suffix in {".png", ".jpg", ".jpeg"}:
        return {"status": "quarantined", "error_code": "ocr_unavailable", "error": "image-only source requires an installed OCR adapter", "pages": [], "blocks": [], "tables": [], "cells": [], "parser_name": "unavailable", "parser_version": "unavailable"}
    elif suffix in {".txt", ".md", ".markdown"}:
        pages.append((1, _decode(payload), "markdown" if suffix != ".txt" else "text"))
    elif suffix in {".html", ".htm"}:
        parser = _TextOnlyHTML()
        parser.feed(_decode(payload))
        pages.append((1, "\n".join(parser.parts), "html"))
    elif suffix == ".csv":
        text = _decode(payload)
        rows = list(csv.reader(text.splitlines()))
        tables, cells = _table_from_rows(rows, version_hint=hashlib.sha256(payload).hexdigest()[:12])
        pages.append((1, "\n".join(" | ".join(row) for row in rows), "csv"))
        parser_name = "stdlib_csv"
    elif suffix == ".json":
        try:
            value = json.loads(_decode(payload))
        except json.JSONDecodeError:
            raise ParseError("JSON document is malformed") from None
        pages.append((1, json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2), "json"))
    elif suffix in {".xbrl", ".xml"}:
        root = _safe_xml(payload, "XBRL/XML")
        facts = []
        for node in root.iter():
            value = "".join(node.itertext()).strip()
            if value and len(node) == 0:
                facts.append(f"{node.tag.rsplit('}', 1)[-1]}={value}")
        pages.append((1, "\n".join(facts), "xbrl"))
        parser_name = "stdlib_xml_raw_facts"
    elif suffix in {".docx", ".pptx", ".xlsx"}:
        if suffix == ".xlsm":
            raise UnsupportedFormat("macro-enabled Office files are rejected")
        try:
            with zipfile.ZipFile(__import__("io").BytesIO(payload)) as zipped:
                infos = zipped.infolist()
                if sum(info.file_size for info in infos) > MAX_UNCOMPRESSED_BYTES:
                    raise ParseError("Office archive exceeds the configured uncompressed size limit")
                if any(info.filename.lower().endswith("vbaProject.bin") for info in infos):
                    raise ParseError("Office archive containing VBA macros is rejected")
                if suffix == ".docx":
                    root = _safe_xml(zipped.read("word/document.xml"), "DOCX document.xml")
                    ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
                    paragraphs = ["".join(node.text or "" for node in paragraph.findall(".//w:t", ns)) for paragraph in root.findall(".//w:p", ns)]
                    pages.append((1, "\n".join(text for text in paragraphs if text), "docx"))
                elif suffix == ".pptx":
                    slide_names = sorted((name for name in zipped.namelist() if re.fullmatch(r"ppt/slides/slide\d+\.xml", name)), key=lambda name: int(re.search(r"slide(\d+)", name).group(1)))
                    for page_number, name in enumerate(slide_names, start=1):
                        root = _safe_xml(zipped.read(name), name)
                        ns = {"a": "http://schemas.openxmlformats.org/drawingml/2006/main"}
                        text = "\n".join(node.text or "" for node in root.findall(".//a:t", ns))
                        pages.append((page_number, text, "pptx_slide"))
                else:
                    sheet_rows = _xlsx_rows(zipped)
                    for page_number, rows in enumerate(sheet_rows, start=1):
                        tables_found, cells_found = _table_from_rows(rows, version_hint=f"{hashlib.sha256(payload).hexdigest()[:8]}-{page_number}")
                        for table in tables_found:
                            table["page_start"] = page_number
                            table["page_end"] = page_number
                        for cell in cells_found:
                            cell["page_number"] = page_number
                        tables.extend(tables_found)
                        cells.extend(cells_found)
                        pages.append((page_number, "\n".join(" | ".join(row) for row in rows), "spreadsheet"))
        except (KeyError, zipfile.BadZipFile, ElementTree.ParseError):
            raise ParseError("Office document package is malformed or incomplete") from None
        parser_name = {".docx": "stdlib_docx_xml", ".pptx": "stdlib_pptx_xml", ".xlsx": "stdlib_xlsx_xml"}[suffix]
    else:
        raise UnsupportedFormat(f"file type {suffix or '(no extension)'} is not supported")

    pages = [(number, text.strip(), kind) for number, text, kind in pages if text.strip()]
    if not pages:
        return {"status": "quarantined", "error_code": "no_extractable_text", "error": "parser produced no readable text", "pages": [], "blocks": [], "tables": tables, "cells": cells, "parser_name": parser_name, "parser_version": parser_version, "parser_confidence": 0.0, "quality": {"source_page_count": source_page_count, "extracted_page_count": 0, "signal": "structural_text_coverage_heuristic_not_accuracy"}}
    if parser_confidence < 0.6:
        return {"status": "quarantined", "error_code": "low_page_coverage", "error": "fewer than 60 percent of source pages yielded readable text", "pages": [], "blocks": [], "tables": [], "cells": [], "parser_name": parser_name, "parser_version": parser_version, "parser_confidence": parser_confidence, "quality": {"source_page_count": source_page_count, "extracted_page_count": len(pages), "signal": "structural_text_coverage_heuristic_not_accuracy"}}
    return {
        "status": status, "error_code": None, "error": None,
        "pages": [{"page_number": number, "text": text, "kind": kind} for number, text, kind in pages],
        "blocks": _blocks_for_pages(pages), "tables": tables, "cells": cells,
        "parser_name": parser_name, "parser_version": parser_version,
        "parser_confidence": parser_confidence,
        "quality": {"source_page_count": source_page_count, "extracted_page_count": len(pages), "signal": "structural_text_coverage_heuristic_not_accuracy"},
    }


def _worker_main(input_path: Path, output_path: Path, filename: str) -> int:
    try:
        result = _parse_bytes(input_path.read_bytes(), filename)
        output_path.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
        return 0
    except Exception as error:
        value = {"status": "quarantined", "error_code": type(error).__name__, "error": str(error)[:500], "parser_confidence": 0.0}
        output_path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
        return 0


def parse_isolated(
    payload: bytes,
    filename: str,
    *,
    scratch_root: str | Path,
    timeout_seconds: float = 45,
    max_input_bytes: int = MAX_INPUT_BYTES,
) -> dict[str, Any]:
    """Run untrusted source parsing in a time-bounded child process with no network API."""
    if not payload or len(payload) > max_input_bytes:
        raise ParseError("source is empty or exceeds the configured byte limit")
    suffix = Path(filename).suffix.lower()
    allowed = {".pdf", ".png", ".jpg", ".jpeg", ".txt", ".md", ".markdown", ".html", ".htm", ".csv", ".json", ".xbrl", ".xml", ".docx", ".pptx", ".xlsx"}
    if suffix not in allowed:
        raise UnsupportedFormat(f"file type {suffix or '(no extension)'} is not supported")
    scratch = Path(scratch_root).resolve()
    scratch.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="rag-parse-", dir=scratch) as temporary_dir:
        work = Path(temporary_dir)
        input_path, output_path = work / "source.bin", work / "result.json"
        input_path.write_bytes(payload)
        module_root = Path(__file__).resolve().parents[3]
        environment = {
            key: os.environ[key]
            for key in ("PATH", "SYSTEMROOT", "WINDIR", "TEMP", "TMP")
            if key in os.environ
        }
        environment["PYTHONIOENCODING"] = "utf-8"
        existing_pythonpath = os.environ.get("PYTHONPATH")
        environment["PYTHONPATH"] = str(module_root) if not existing_pythonpath else str(module_root) + os.pathsep + existing_pythonpath
        command = [sys.executable, "-m", "adaptive_mas.industrial_rag.parser", "--worker", str(input_path), str(output_path), Path(filename).name]
        try:
            subprocess.run(command, cwd=module_root, env=environment, timeout=timeout_seconds, check=True, capture_output=True)
        except subprocess.TimeoutExpired:
            raise ParseError(f"isolated parser exceeded its {timeout_seconds:g}s time limit") from None
        except subprocess.CalledProcessError as error:
            raise ParseError(f"isolated parser process failed (exit code {error.returncode})") from None
        try:
            return json.loads(output_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            raise ParseError("isolated parser returned no valid result") from None


def normalize_financial_cell(raw: str) -> dict[str, object]:
    """Keep source spelling while distinguishing blank, dash, zero, negative, %, and bps."""
    text = raw.strip()
    if not text:
        return {"raw_text": raw, "value_kind": "blank", "normalized_value": None}
    if text in {"-", "—", "–", "－", "--", "——"}:
        return {"raw_text": raw, "value_kind": "dash", "normalized_value": None}
    multiplier = 1
    parenthesized_negative = text.startswith("(") and text.endswith(")")
    if parenthesized_negative:
        text = text[1:-1].strip()
    if text.lower().endswith("bp") or text.endswith("基点"):
        number = re.sub(r"(?:bp|基点)$", "", text, flags=re.IGNORECASE)
        kind = "basis_points"
    elif text.endswith("%"):
        number = text[:-1]
        kind = "percent"
    else:
        number = text
        kind = "number"
    number = number.replace(",", "").replace("，", "").replace(" ", "")
    if parenthesized_negative:
        number = "-" + number.lstrip("+-")
    if not re.fullmatch(r"[-+]?\d+(?:\.\d+)?", number):
        return {"raw_text": raw, "value_kind": "unknown", "normalized_value": None}
    if Decimal(number) == 0 and kind == "number":
        kind = "zero"
    return {"raw_text": raw, "value_kind": kind, "normalized_value": number, "multiplier": str(multiplier)}


def validate_cross_page_table_continuation(previous: Mapping[str, object], current: Mapping[str, object]) -> dict[str, object]:
    """Return a continuation candidate; never join tables on weak evidence alone."""
    checks = {
        "same_title": previous.get("title") == current.get("title") and bool(previous.get("title")),
        "same_header": previous.get("header") == current.get("header") and bool(previous.get("header")),
        "same_security": previous.get("security_code") == current.get("security_code"),
        "same_report_period": previous.get("report_period") == current.get("report_period"),
        "same_unit": previous.get("unit") == current.get("unit"),
        "adjacent_pages": isinstance(previous.get("page_end"), int) and current.get("page_start") == previous.get("page_end") + 1,
        "explicit_continued_marker": bool(current.get("continued_marker")),
    }
    score = sum(checks.values()) / len(checks)
    verified = checks["same_header"] and checks["same_security"] and checks["same_report_period"] and checks["same_unit"] and checks["adjacent_pages"] and (checks["same_title"] or checks["explicit_continued_marker"])
    return {"decision": "verified_continuation" if verified else "possible_continuation" if score >= 0.5 else "separate_tables", "confidence": round(score, 4), "checks": checks, "automatic_merge_allowed": verified}


def records_from_parse(result: Mapping[str, Any], version_id: str) -> dict[str, tuple[object, ...]]:
    """Bind parser locations to version-scoped canonical records without guessing facts."""
    safe = re.sub(r"[^A-Za-z0-9._:-]", "_", version_id)
    pages = {int(page["page_number"]): str(page["text"]) for page in result.get("pages", [])}
    sections = tuple(
        Section(
            section_id=f"sec-{safe}-p{page_number}", version_id=version_id,
            title=f"Page {page_number}", level=1, start_page=page_number, end_page=page_number,
        )
        for page_number in sorted(pages)
    )
    section_by_page = {int(item.start_page): item.section_id for item in sections}
    block_map: dict[str, str] = {}
    for raw in result.get("blocks", []):
        block_map[str(raw["block_id"])] = f"blk-{safe}-{raw['block_id']}"
    blocks: list[Block] = []
    spans: list[Span] = []
    for raw in result.get("blocks", []):
        page_number = int(raw["page_number"])
        block_id = block_map[str(raw["block_id"])]
        text = str(raw.get("text", ""))
        char_start = int(raw["char_start"]) if raw.get("char_start") is not None else 0
        char_end = int(raw["char_end"]) if raw.get("char_end") is not None else char_start + len(text)
        parent_raw = raw.get("parent_block_id")
        previous_raw = raw.get("previous_block_id")
        next_raw = raw.get("next_block_id")
        blocks.append(Block(
            block_id=block_id, version_id=version_id, text=text,
            block_type=str(raw.get("block_type", "paragraph")),
            section_id=section_by_page.get(page_number), page_number=page_number,
            char_start=char_start, char_end=char_end,
            parent_block_id=block_map.get(str(parent_raw)) if parent_raw is not None else None,
            previous_block_id=block_map.get(str(previous_raw)) if previous_raw is not None else None,
            next_block_id=block_map.get(str(next_raw)) if next_raw is not None else None,
            parser_confidence=float(raw["parser_confidence"]) if raw.get("parser_confidence") is not None else None,
            quarantine_reason=raw.get("quarantine_reason"),
            span_id=f"span-{safe}-{raw['block_id']}" if text else None,
        ))
        if text:
            spans.append(Span(
                span_id=f"span-{safe}-{raw['block_id']}", version_id=version_id,
                page_number=page_number, char_start=char_start, char_end=char_end, text=text,
            ))
    tables: list[Table] = []
    table_ids: dict[str, str] = {}
    for raw in result.get("tables", []):
        old_id = str(raw["table_id"])
        table_id = f"tbl-{safe}-{old_id}"
        table_ids[old_id] = table_id
        tables.append(Table(
            table_id=table_id, version_id=version_id, title=raw.get("title"),
            header_rows=int(raw.get("header_rows", 1)),
            continued_from_table_id=None,
            continuation_confidence=raw.get("continuation_confidence"),
            unit_text=raw.get("unit_text"), accounting_basis=raw.get("accounting_basis"),
            report_period=raw.get("report_period"), page_start=raw.get("page_start"), page_end=raw.get("page_end"),
        ))
    cells: list[TableCell] = []
    for raw in result.get("cells", []):
        old_table_id = str(raw["table_id"])
        if old_table_id not in table_ids:
            raise ParseError("parser table cell references a missing table")
        cells.append(TableCell(
            cell_id=f"cell-{safe}-{raw['cell_id']}", table_id=table_ids[old_table_id],
            version_id=version_id, row_index=int(raw["row_index"]), column_index=int(raw["column_index"]),
            raw_text=str(raw.get("raw_text", "")), normalized_value=raw.get("normalized_value"),
            value_kind=str(raw.get("value_kind", "text")), page_number=raw.get("page_number"),
            unit=raw.get("unit"), multiplier=raw.get("multiplier"), currency=raw.get("currency"),
            accounting_basis=raw.get("accounting_basis"), report_period=raw.get("report_period"),
            footnote_refs=tuple(raw.get("footnote_refs", [])),
        ))
    return {"sections": sections, "spans": tuple(spans), "blocks": tuple(blocks), "tables": tuple(tables), "cells": tuple(cells)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("input_path", nargs="?")
    parser.add_argument("output_path", nargs="?")
    parser.add_argument("filename", nargs="?")
    arguments = parser.parse_args()
    if not arguments.worker or not arguments.input_path or not arguments.output_path or not arguments.filename:
        raise SystemExit("parser worker arguments are incomplete")
    raise SystemExit(_worker_main(Path(arguments.input_path), Path(arguments.output_path), arguments.filename))
