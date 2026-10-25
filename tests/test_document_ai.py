from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from adaptive_mas.industrial_rag import document_ai
from adaptive_mas.industrial_rag.parser import _table_row_group_blocks


def test_docling_asset_manifest_rejects_modified_files(tmp_path) -> None:
    asset = tmp_path / "model.bin"
    asset.write_bytes(b"pinned model bytes")
    manifest = {
        "schema_version": "raama.docling-assets.v1",
        "docling_version": document_ai.DOCLING_VERSION,
        "layout": {"repo_id": document_ai.LAYOUT_REPO, "revision": document_ai.LAYOUT_REVISION},
        "tableformer": {
            "repo_id": document_ai.TABLEFORMER_REPO,
            "revision": document_ai.TABLEFORMER_REVISION,
            "tag": document_ai.TABLEFORMER_TAG,
        },
        "rapidocr": {
            "package": "rapidocr", "version": document_ai.RAPIDOCR_VERSION,
            "backend": "onnxruntime", "language": "ch",
        },
        "files": document_ai._file_manifest(tmp_path),
    }
    (tmp_path / document_ai.ASSET_MANIFEST).write_text(json.dumps(manifest), encoding="utf-8")
    document_ai._require_pinned_assets(tmp_path)

    asset.write_bytes(b"replaced model bytes")
    with pytest.raises(RuntimeError, match="SHA-256 integrity check"):
        document_ai._require_pinned_assets(tmp_path)


def test_docling_row_header_stubs_do_not_expand_column_header_rows() -> None:
    cells = [
        SimpleNamespace(start_row_offset_idx=0, column_header=True, row_header=False),
        SimpleNamespace(start_row_offset_idx=1, column_header=False, row_header=True),
        SimpleNamespace(start_row_offset_idx=2, column_header=False, row_header=True),
    ]
    assert document_ai._docling_header_contract(cells) == (1, None)


def test_docling_noncontiguous_column_header_rows_are_quarantined() -> None:
    cells = [
        SimpleNamespace(start_row_offset_idx=0, column_header=True, row_header=False),
        SimpleNamespace(start_row_offset_idx=1, column_header=False, row_header=True),
        SimpleNamespace(start_row_offset_idx=2, column_header=True, row_header=False),
    ]
    assert document_ai._docling_header_contract(cells) == (1, "non_contiguous_column_header_rows")


def test_docling_missing_column_header_structure_is_quarantined() -> None:
    cells = [
        SimpleNamespace(start_row_offset_idx=0, column_header=False, row_header=True),
        SimpleNamespace(start_row_offset_idx=1, column_header=False, row_header=True),
    ]
    assert document_ai._docling_header_contract(cells) == (1, "column_headers_not_detected")


def test_docling_ambiguous_table_rows_are_marked_unsearchable() -> None:
    tables = [{"table_id": "table-1", "header_rows": 1, "page_start": 1,
               "row_group_quarantine_reason": "non_contiguous_column_header_rows"}]
    cells = [
        {"table_id": "table-1", "row_index": 0, "column_index": 0, "raw_text": "Metric", "page_number": 1},
        {"table_id": "table-1", "row_index": 1, "column_index": 0, "raw_text": "Revenue", "page_number": 1},
        {"table_id": "table-1", "row_index": 2, "column_index": 0, "raw_text": "Profit", "page_number": 1},
    ]
    groups = _table_row_group_blocks(tables, cells)
    assert [group["row_start"] for group in groups] == [1, 2]
    assert all(group["quarantine_reason"] == "non_contiguous_column_header_rows" for group in groups)


def test_docling_unit_and_period_are_kept_as_explicit_source_labels() -> None:
    assert document_ai._explicit_unit_text(["2024年度财务报告", "单位：", "人民币百万元"]) == "人民币百万元"
    assert document_ai._explicit_unit_text(["2024年度财务报告", "人民币百万元"]) is None
    assert document_ai._period_header("2024年度") == "2024年度"
    assert document_ai._period_header("2024年") == "2024年"
    assert document_ai._period_header("累计收入") is None
