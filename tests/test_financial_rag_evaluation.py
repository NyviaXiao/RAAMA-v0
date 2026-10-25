from __future__ import annotations

import json

import pytest

from adaptive_mas.industrial_rag.evaluation import evaluate_manifest


def _case() -> dict[str, object]:
    return {
        "case_id": "synthetic-revenue",
        "data_partition": "development",
        "view_status": "public_viewed_diagnostic",
        "answerability": "answerable",
        "expected_refusal_reason": None,
        "required_evidence": [{
            "source_id": "synthetic-source",
            "content_sha256": "a" * 64,
            "page_number": 2,
            "evidence_type": "cell",
            "table_id": "table-1",
            "row_index": 1,
            "column_index": 1,
            "metric": "营业收入",
            "raw_value": "10",
            "normalized_value": "10.00",
            "unit": "人民币百万元",
            "multiplier": "1",
            "report_period": "2024年度",
            "accounting_basis": "合并",
        }],
    }


def _candidate(*, raw_value: str = "10.00", normalized_value: str = "10.00", row_index: int = 1,
               unit: str = "人民币百万元", multiplier: str = "1", report_period: str = "2024年度") -> dict[str, object]:
    return {
        "evidence_id": "ev-table-row-1",
        "block_id": "block-1",
        "text": "列头：指标 | 报告期 | 数值 | 单位\n数据行：营业收入 | 2024年度 | 10.00 | 人民币百万元",
        "citation": {
            "source_id": "synthetic-source",
            "content_sha256": "a" * 64,
            "page_number": 2,
            "table_id": "table-1",
            "row_start": row_index,
            "row_end": row_index,
            "unit": unit,
            "accounting_basis": "合并",
            "supporting_cells": [
                {"cell_id": "table-1-r0-c1", "page_number": 2, "row_index": 0, "column_index": 1,
                 "raw_text": "2024年度"},
                {"cell_id": "table-1-r1-c0", "page_number": 2, "row_index": row_index, "column_index": 0,
                 "raw_text": "营业收入"},
                {"cell_id": "table-1-r1-c1", "page_number": 2, "row_index": row_index, "column_index": 1,
                 "raw_text": raw_value, "normalized_value": normalized_value, "unit": unit,
                 "multiplier": multiplier, "report_period": report_period, "accounting_basis": "合并"},
            ],
        },
    }


def _write_inputs(tmp_path, manifest: dict[str, object], predictions: dict[str, object]) -> tuple[str, str]:
    manifest_path = tmp_path / "manifest.json"
    predictions_path = tmp_path / "predictions.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
    predictions_path.write_text(json.dumps(predictions, ensure_ascii=False), encoding="utf-8")
    return str(manifest_path), str(predictions_path)


def _run(tmp_path, case: dict[str, object], rows: list[dict[str, object]], *, k_values=(1, 3)):
    manifest_path, predictions_path = _write_inputs(
        tmp_path,
        {"benchmark_id": "synthetic-eval", "cases": [case]},
        {"run_id": "run-1", "cases": [{
            "case_id": "synthetic-revenue", "retrieved": rows,
            "predicted_answerability": "answerable", "latency_ms": 25,
        }]},
    )
    return evaluate_manifest(manifest_path, predictions_path, k_values=k_values)


def test_offline_evaluator_scores_exact_cell_evidence_without_claiming_acceptance(tmp_path) -> None:
    report = _run(tmp_path, _case(), [_candidate()])
    assert report["metrics"]["at_k"]["1"]["evidence_recall_mean"] == 1.0
    assert report["metrics"]["at_k"]["1"]["complete_evidence_set_rate"] == 1.0
    assert report["metrics"]["at_k"]["1"]["ndcg_mean"] == 1.0
    assert report["metrics"]["mrr_mean"] == 1.0
    assert report["metrics"]["answerability_accuracy"] == 1.0
    assert report["metrics"]["by_partition"]["development"]["at_k"]["1"]["ndcg_mean"] == 1.0
    assert report["diagnostic_only"] is True
    assert report["acceptance_claim_permitted"] is False


@pytest.mark.parametrize("candidate", [
    _candidate(raw_value="100", normalized_value="100"),
    _candidate(row_index=2),
    _candidate(unit="人民币元"),
    _candidate(unit=None),
    _candidate(unit="unknown"),
    _candidate(multiplier="1000000"),
    _candidate(report_period="2023年度"),
])
def test_offline_evaluator_rejects_wrong_value_or_original_location(tmp_path, candidate) -> None:
    report = _run(tmp_path, _case(), [candidate])
    assert report["metrics"]["at_k"]["1"]["evidence_recall_mean"] == 0.0
    assert report["metrics"]["at_k"]["1"]["ndcg_mean"] == 0.0


def test_exact_cell_rejects_conflicting_top_level_numeric_metadata(tmp_path) -> None:
    candidate = _candidate(raw_value="100", normalized_value="100")
    candidate["citation"]["raw_value"] = "10"
    candidate["citation"]["normalized_value"] = "10.00"
    report = _run(tmp_path, _case(), [candidate])
    assert report["metrics"]["at_k"]["1"]["evidence_recall_mean"] == 0.0


def test_exact_cell_inherits_unit_from_table_when_cell_omits_it(tmp_path) -> None:
    candidate = _candidate()
    candidate["citation"]["supporting_cells"][2].pop("unit")
    report = _run(tmp_path, _case(), [candidate])
    assert report["metrics"]["at_k"]["1"]["evidence_recall_mean"] == 1.0


@pytest.mark.parametrize("metadata", [
    {"citation": {"unit": "unknown"}},
    {"citation": {"unit": "人民币元"}},
    {"citation": {"multiplier": "1000000"}},
    {"citation": {"report_period": "2023年度"}},
    {"candidate": {"unit": "人民币元"}},
    {"candidate": {"report_period": "2023年度"}},
])
def test_exact_cell_rejects_unknown_or_conflicting_inherited_metadata(tmp_path, metadata) -> None:
    candidate = _candidate()
    candidate["citation"]["supporting_cells"][2].pop("unit")
    target = candidate if "candidate" in metadata else candidate["citation"]
    target.update(metadata.get("candidate", metadata.get("citation", {})))
    report = _run(tmp_path, _case(), [candidate])
    assert report["metrics"]["at_k"]["1"]["evidence_recall_mean"] == 0.0


def test_exact_cell_rejects_explicit_cell_unit_conflicting_with_table_unit(tmp_path) -> None:
    candidate = _candidate()
    candidate["citation"]["supporting_cells"][2]["unit"] = "人民币元"
    report = _run(tmp_path, _case(), [candidate])
    assert report["metrics"]["at_k"]["1"]["evidence_recall_mean"] == 0.0


@pytest.mark.parametrize(("candidate_record_id", "expected_recall"), [
    ("grant-revocation-1", 1.0),
    ("grant-revocation-2", 0.0),
])
def test_permission_evidence_matches_event_id_without_document_location(
    tmp_path, candidate_record_id: str, expected_recall: float,
) -> None:
    case = _case()
    case["required_evidence"] = [{
        "source_id": None,
        "content_sha256": None,
        "page_number": None,
        "evidence_type": "permission",
        "evidence_record_id": "grant-revocation-1",
        "metric": None,
    }]
    candidate = {
        "evidence_id": "event-1",
        "citation": {"evidence_record_id": candidate_record_id},
    }

    report = _run(tmp_path, case, [candidate])

    assert report["metrics"]["at_k"]["1"]["evidence_recall_mean"] == expected_recall


def test_offline_evaluator_does_not_match_metric_substrings_in_text(tmp_path) -> None:
    candidate = _candidate()
    candidate["text"] = "营业收入 10.00"
    candidate["citation"]["supporting_cells"][1]["raw_text"] = "营业收入增长率"
    report = _run(tmp_path, _case(), [candidate])
    assert report["metrics"]["at_k"]["1"]["evidence_recall_mean"] == 0.0


def test_duplicate_candidates_do_not_increase_evidence_recall_or_ndcg(tmp_path) -> None:
    rows = [_candidate()]
    rows.extend({**_candidate(), "evidence_id": f"duplicate-{index}"} for index in range(2))
    report = _run(tmp_path, _case(), rows)
    metrics = report["metrics"]
    assert metrics["duplicate_evidence_count"] == 2
    assert metrics["at_k"]["3"]["evidence_recall_mean"] == 1.0
    assert metrics["at_k"]["3"]["ndcg_mean"] == 1.0
    assert report["cases"][0]["at_k"]["3"]["relevant_candidate_count"] == 1


def test_missing_prediction_is_scored_as_a_failure(tmp_path) -> None:
    manifest_path, predictions_path = _write_inputs(
        tmp_path,
        {"benchmark_id": "synthetic-eval", "cases": [_case()]},
        {"run_id": "run-1", "cases": []},
    )
    report = evaluate_manifest(manifest_path, predictions_path, k_values=(1,))
    assert report["prediction_count"] == 0
    assert report["metrics"]["at_k"]["1"]["evidence_recall_mean"] == 0.0
    assert report["metrics"]["at_k"]["1"]["ndcg_mean"] == 0.0
    assert report["metrics"]["mrr_mean"] == 0.0
    assert report["metrics"]["answerability_accuracy"] == 0.0


def test_offline_evaluator_rejects_unknown_prediction_cases(tmp_path) -> None:
    manifest_path, predictions_path = _write_inputs(
        tmp_path,
        {"benchmark_id": "synthetic-eval", "cases": [_case()]},
        {"cases": [{"case_id": "unknown-case", "retrieved": []}]},
    )
    with pytest.raises(ValueError, match="unknown case"):
        evaluate_manifest(manifest_path, predictions_path)


def test_offline_evaluator_rejects_duplicate_gold_locations(tmp_path) -> None:
    case = _case()
    case["required_evidence"] = case["required_evidence"] * 2
    with pytest.raises(ValueError, match="repeats a required evidence location"):
        _run(tmp_path, case, [_candidate()])


def test_acceptance_gate_requires_more_than_100_unseen_independent_predictions(tmp_path) -> None:
    cases = [{
        "case_id": f"acceptance-{index:03d}",
        "data_partition": "independent_acceptance",
        "view_status": "unseen",
        "answerability": "unanswerable",
        "expected_refusal_reason": "no_evidence",
        "required_evidence": [],
    } for index in range(101)]
    predictions = [{
        "case_id": case["case_id"], "retrieved": [], "predicted_answerability": "unanswerable",
        "refusal_reason": "no_evidence",
    } for case in cases]
    manifest_path, predictions_path = _write_inputs(
        tmp_path,
        {"benchmark_id": "synthetic-acceptance-gate", "cases": cases},
        {"run_id": "complete-run", "cases": predictions},
    )
    complete = evaluate_manifest(manifest_path, predictions_path, k_values=(1,))
    assert complete["acceptance_claim_permitted"] is True
    assert complete["diagnostic_only"] is False

    predictions.pop()
    _, missing_path = _write_inputs(
        tmp_path,
        {"benchmark_id": "synthetic-acceptance-gate", "cases": cases},
        {"run_id": "incomplete-run", "cases": predictions},
    )
    incomplete = evaluate_manifest(manifest_path, missing_path, k_values=(1,))
    assert incomplete["acceptance_claim_permitted"] is False
