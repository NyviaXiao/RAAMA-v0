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
    assert document_ai._explicit_unit_text(["（人民币百万元，特别注明除外）"]) == "人民币百万元"
    assert document_ai._period_header("2024年度") == "2024年度"
    assert document_ai._period_header("2024年") == "2024年"
    assert document_ai._period_header("2024年1-3月") == "2024年1-3月"
    assert document_ai._combined_period_header(["2024年", "第一季度"]) == "2024年第一季度"
    assert document_ai._period_header("累计收入") is None


def test_docling_splits_a_merged_table_at_a_repeated_statement_heading() -> None:
    cells = [
        SimpleNamespace(start_row_offset_idx=0, start_col_offset_idx=0, text="项目", column_header=True),
        SimpleNamespace(start_row_offset_idx=0, start_col_offset_idx=1, text="2024年12月31日", column_header=True),
        SimpleNamespace(start_row_offset_idx=1, start_col_offset_idx=0, text="货币资金", column_header=False),
        SimpleNamespace(start_row_offset_idx=1, start_col_offset_idx=1, text="1,000", column_header=False),
        SimpleNamespace(start_row_offset_idx=2, start_col_offset_idx=0, text="合并现金流量表", column_header=False),
        SimpleNamespace(start_row_offset_idx=3, start_col_offset_idx=0, text="项目", column_header=True),
        SimpleNamespace(start_row_offset_idx=3, start_col_offset_idx=1, text="2024年1-3月", column_header=True),
        SimpleNamespace(start_row_offset_idx=4, start_col_offset_idx=0, text="经营活动产生的现金流量净额", column_header=False),
        SimpleNamespace(start_row_offset_idx=4, start_col_offset_idx=1, text="(1,208)", column_header=False),
    ]
    segments = document_ai._docling_table_segments(cells)
    assert [segment["start_row"] for segment in segments] == [0, 2]
    assert [segment["title"] for segment in segments] == [None, "合并现金流量表"]
    assert [segment["header_rows"] for segment in segments] == [1, 2]
    assert all(segment["quarantine_reason"] is None for segment in segments)
    assert segments[1]["column_header_rows"] == (3,)

    balance_table, balance_cells = document_ai._docling_segment_records(
        segments[0],
        table_id="table-balance",
        page_number=3,
        title_parts=[],
        unit_context=["单位：", "人民币百万元"],
    )
    cashflow_table, cashflow_cells = document_ai._docling_segment_records(
        segments[1],
        table_id="table-cashflow",
        page_number=3,
        title_parts=[],
        unit_context=[],
    )
    assert balance_table["unit_text"] == "人民币百万元"
    assert cashflow_table["unit_text"] is None
    q1_amount = next(cell for cell in cashflow_cells if cell["raw_text"] == "(1,208)")
    assert q1_amount["normalized_value"] == "-1208"
    assert q1_amount["report_period"] == "2024年1-3月"
    assert q1_amount["row_index"] == 2


def test_docling_keeps_a_subtable_without_column_headers_unsearchable() -> None:
    cells = [
        SimpleNamespace(start_row_offset_idx=0, start_col_offset_idx=0, text="项目", column_header=True),
        SimpleNamespace(start_row_offset_idx=0, start_col_offset_idx=1, text="2024年12月31日", column_header=True),
        SimpleNamespace(start_row_offset_idx=1, start_col_offset_idx=0, text="资产总额", column_header=False),
        SimpleNamespace(start_row_offset_idx=1, start_col_offset_idx=1, text="1,000", column_header=False),
        SimpleNamespace(start_row_offset_idx=2, start_col_offset_idx=0, text="合并现金流量表", column_header=False),
        SimpleNamespace(start_row_offset_idx=3, start_col_offset_idx=0, text="经营活动现金流", column_header=False),
        SimpleNamespace(start_row_offset_idx=3, start_col_offset_idx=1, text="(1,208)", column_header=False),
    ]
    segments = document_ai._docling_table_segments(cells)
    assert [segment["start_row"] for segment in segments] == [0, 2]
    assert segments[1]["quarantine_reason"] == "column_headers_not_detected"


def test_docling_splits_repeated_year_and_quarter_headers_without_column_header_flags() -> None:
    cells = [
        SimpleNamespace(start_row_offset_idx=0, start_col_offset_idx=0, text="项目", column_header=True),
        SimpleNamespace(start_row_offset_idx=0, start_col_offset_idx=1, text="2024年12月31日", column_header=True),
        SimpleNamespace(start_row_offset_idx=1, start_col_offset_idx=0, text="经营活动产生的现金流量净额", column_header=False),
        SimpleNamespace(start_row_offset_idx=1, start_col_offset_idx=1, text="447,023", column_header=False),
        *[
            SimpleNamespace(start_row_offset_idx=2, start_col_offset_idx=column, text="2024年", column_header=False)
            for column in range(1, 5)
        ],
        SimpleNamespace(start_row_offset_idx=3, start_col_offset_idx=0, text="（人民币百万元）", column_header=False),
        *[
            SimpleNamespace(start_row_offset_idx=3, start_col_offset_idx=column, text=quarter, column_header=False)
            for column, quarter in enumerate(("第一季度", "第二季度", "第三季度", "第四季度"), start=1)
        ],
        SimpleNamespace(start_row_offset_idx=4, start_col_offset_idx=0, text="按季度披露的经营业绩指标", column_header=False),
        SimpleNamespace(start_row_offset_idx=5, start_col_offset_idx=0, text="经营活动产生的现金流量净额", column_header=False),
        SimpleNamespace(start_row_offset_idx=5, start_col_offset_idx=1, text="(1,208)", column_header=False),
    ]
    segments = document_ai._docling_table_segments(cells)
    assert [segment["start_row"] for segment in segments] == [0, 2]
    assert [segment["header_rows"] for segment in segments] == [1, 3]
    assert segments[0]["end_row"] == 1
    assert segments[1]["title"] == "按季度披露的经营业绩指标"
    assert segments[1]["column_header_rows"] == (2, 3)
    assert all(segment["quarantine_reason"] is None for segment in segments)

    annual_table, annual_cells = document_ai._docling_segment_records(
        segments[0], table_id="annual-table", page_number=4, title_parts=[], unit_context=[],
    )
    quarterly_table, quarterly_cells = document_ai._docling_segment_records(
        segments[1], table_id="quarterly-table", page_number=4, title_parts=[], unit_context=[],
    )
    annual_value = next(cell for cell in annual_cells if cell["raw_text"] == "447,023")
    q1_value = next(cell for cell in quarterly_cells if cell["raw_text"] == "(1,208)")
    assert annual_table["unit_text"] is None
    assert annual_value["report_period"] == "2024年12月31日"
    assert "(1,208)" not in {cell["raw_text"] for cell in annual_cells}
    assert quarterly_table["unit_text"] == "人民币百万元"
    assert quarterly_table["title"] == "按季度披露的经营业绩指标"
    assert q1_value["normalized_value"] == "-1208"
    assert q1_value["report_period"] == "2024年第一季度"
