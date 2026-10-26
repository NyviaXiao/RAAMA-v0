"""Offline, label-isolated evaluation for frozen financial RAG QA manifests."""

from __future__ import annotations

import hashlib
import json
import math
import re
import statistics
from collections import Counter
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Mapping, Sequence


_ANSWERABILITY = {"answerable", "partial", "unanswerable", "permission_denied", "time_ineligible"}


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _read_object(path: str | Path) -> tuple[dict[str, Any], bytes]:
    raw = Path(path).read_bytes()
    return _read_object_bytes(raw, Path(path).name)


def _read_object_bytes(raw: bytes, name: str) -> tuple[dict[str, Any], bytes]:
    value = json.loads(raw.decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{name} must contain one JSON object")
    return value, raw


def _candidate_citation(candidate: Mapping[str, Any]) -> dict[str, Any]:
    citation = candidate.get("citation", {})
    return dict(citation) if isinstance(citation, Mapping) else {}


def _supporting_cells(candidate: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    value = _candidate_citation(candidate).get("supporting_cells", [])
    return [cell for cell in value if isinstance(cell, Mapping)] if isinstance(value, list) else []


def _supporting_spans(candidate: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    value = _candidate_citation(candidate).get("supporting_spans", [])
    return [span for span in value if isinstance(span, Mapping)] if isinstance(value, list) else []


def _decimal_value(value: Any, *, allow_percent: bool = False) -> tuple[Decimal, str] | None:
    if isinstance(value, bool) or value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    negative = text.startswith("(") and text.endswith(")")
    if negative:
        text = text[1:-1].strip()
    kind = "number"
    if allow_percent and (text.endswith("%") or text.casefold().endswith("bp") or text.endswith("基点")):
        if text.endswith("%"):
            text, kind = text[:-1], "percent"
        elif text.endswith("基点"):
            text, kind = text[:-2], "basis_points"
        else:
            text, kind = text[:-2], "basis_points"
    text = re.sub(r"[,，\s]", "", text)
    if not re.fullmatch(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", text):
        return None
    if negative:
        text = "-" + text.lstrip("+-")
    try:
        number = Decimal(text)
    except InvalidOperation:
        return None
    return (number, kind) if number.is_finite() else None


def _value_matches(key: str, expected: Any, observed: Any) -> bool:
    if key in {"raw_value", "normalized_value", "multiplier"}:
        expected_number = _decimal_value(expected, allow_percent=key == "raw_value")
        observed_number = _decimal_value(observed, allow_percent=key == "raw_value")
        if expected_number is None or observed_number is None:
            return False
        return expected_number == observed_number
    return " ".join(str(expected).split()).casefold() == " ".join(str(observed).split()).casefold()


def _has_metadata(value: Any) -> bool:
    return value is not None and not (isinstance(value, str) and not value.strip())


def _normalized_gold_evidence(item: Mapping[str, Any]) -> dict[str, Any]:
    normalized = dict(item)
    reference = normalized.get("cell_reference")
    if isinstance(reference, str):
        match = re.fullmatch(r"r(\d+)c(\d+)", reference, flags=re.IGNORECASE)
        if match:
            reference_row, reference_column = int(match.group(1)), int(match.group(2))
            if normalized.get("row_index", reference_row) != reference_row or normalized.get("column_index", reference_column) != reference_column:
                raise ValueError("cell_reference conflicts with row_index/column_index")
            normalized["row_index"] = reference_row
            normalized["column_index"] = reference_column
    return normalized


def _validate_gold_evidence(case_id: Any, gold: Sequence[Mapping[str, Any]]) -> None:
    for item in gold:
        evidence_type = item.get("evidence_type")
        if evidence_type in {"permission", "time"}:
            if not isinstance(item.get("evidence_record_id"), str) or not item["evidence_record_id"].strip():
                raise ValueError(f"case {case_id!r} evidence requires an exact evidence_record_id for {evidence_type!r}")
            source_id = item.get("source_id")
            content_sha256 = item.get("content_sha256")
            page_number = item.get("page_number")
            if source_id is not None and (not isinstance(source_id, str) or not source_id.strip()):
                raise ValueError(f"case {case_id!r} evidence has an invalid optional source_id")
            if content_sha256 is not None and (
                not isinstance(content_sha256, str) or re.fullmatch(r"[0-9a-f]{64}", content_sha256) is None
            ):
                raise ValueError(f"case {case_id!r} evidence has an invalid optional content_sha256")
            if page_number is not None and (
                isinstance(page_number, bool) or not isinstance(page_number, int) or page_number < 1
            ):
                raise ValueError(f"case {case_id!r} evidence has an invalid optional page_number")
        elif (
            not isinstance(item.get("source_id"), str)
            or not str(item.get("source_id")).strip()
            or not isinstance(item.get("content_sha256"), str)
            or re.fullmatch(r"[0-9a-f]{64}", str(item.get("content_sha256"))) is None
            or isinstance(item.get("page_number"), bool)
            or not isinstance(item.get("page_number"), int)
            or int(item["page_number"]) < 1
        ):
            raise ValueError(f"case {case_id!r} evidence requires source_id, content_sha256, and page_number")
        if evidence_type == "cell":
            located = (
                isinstance(item.get("table_id"), str) and bool(item.get("table_id"))
                and isinstance(item.get("row_index"), int) and not isinstance(item.get("row_index"), bool) and item["row_index"] >= 0
                and isinstance(item.get("column_index"), int) and not isinstance(item.get("column_index"), bool) and item["column_index"] >= 0
            )
        elif evidence_type == "table":
            located = isinstance(item.get("table_id"), str) and bool(item.get("table_id")) and (
                (isinstance(item.get("row_index"), int) and not isinstance(item.get("row_index"), bool) and item["row_index"] >= 0)
                or (
                    isinstance(item.get("row_start"), int) and not isinstance(item.get("row_start"), bool) and item["row_start"] >= 0
                    and isinstance(item.get("row_end"), int) and not isinstance(item.get("row_end"), bool) and item["row_end"] >= item["row_start"]
                )
            )
        elif evidence_type == "fact":
            located = item.get("fact_id")
        elif evidence_type in {"span", "paragraph", "refutation"}:
            located = item.get("span_id") or (item.get("char_start") is not None and item.get("char_end") is not None)
        elif evidence_type in {"permission", "time"}:
            located = item.get("evidence_record_id")
        else:
            located = False
        if not located:
            raise ValueError(f"case {case_id!r} evidence lacks an exact original location for {evidence_type!r}")


def _located_cells(required: Mapping[str, Any], candidate: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    cells = _supporting_cells(candidate)
    location_keys = ("cell_id", "row_index", "column_index")
    expected_location = {key: required[key] for key in location_keys if required.get(key) is not None}
    if not expected_location:
        return []
    return [
        cell for cell in cells
        if all(cell.get(key) == value for key, value in expected_location.items())
        and (cell.get("page_number") is None or required.get("page_number") is None or cell.get("page_number") == required.get("page_number"))
    ]


def _required_match(required: Mapping[str, Any], candidate: Mapping[str, Any]) -> bool:
    citation = _candidate_citation(candidate)
    supporting_span_match = False
    if required.get("evidence_type") in {"span", "paragraph", "refutation"}:
        for span in _supporting_spans(candidate):
            if all(
                required.get(key) is None or span.get(key) == required.get(key)
                for key in ("source_id", "content_sha256", "page_number", "span_id", "char_start", "char_end")
            ) and (required.get("span_id") or (required.get("char_start") is not None and required.get("char_end") is not None)):
                supporting_span_match = True
                break
    fields = (
        ("source_id", "source_id"),
        ("content_sha256", "content_sha256"),
        ("page_number", "page_number"),
    )
    for expected_key, citation_key in fields:
        expected = required.get(expected_key)
        if expected is not None and citation.get(citation_key) != expected and not (
            citation_key == "page_number" and supporting_span_match
        ):
            return False
    evidence_type = required.get("evidence_type")
    if evidence_type in {"permission", "time"}:
        evidence_record_id = required.get("evidence_record_id")
        return (
            isinstance(evidence_record_id, str)
            and bool(evidence_record_id.strip())
            and citation.get("evidence_record_id") == evidence_record_id
        )
    if not all(required.get(key) is not None for key in ("source_id", "content_sha256", "page_number")):
        return False

    located_cells = _located_cells(required, candidate)
    location_keys = ("cell_id", "row_index", "column_index")
    has_cell_location = any(required.get(key) is not None for key in location_keys)
    if evidence_type == "fact":
        if not required.get("fact_id") or citation.get("fact_id") != required.get("fact_id"):
            return False
    elif evidence_type in {"table", "cell"}:
        expected_table = required.get("table_id")
        if not expected_table or citation.get("table_id") != expected_table:
            return False
        if evidence_type == "cell" and not has_cell_location:
            return False
        if has_cell_location and not located_cells:
            return False
        if not has_cell_location and not any(required.get(key) is not None for key in ("row_start", "row_end")):
            return False
        if required.get("row_start") is not None and citation.get("row_start") != required.get("row_start"):
            return False
        if required.get("row_end") is not None and citation.get("row_end") != required.get("row_end"):
            return False
        if not citation.get("table_id"):
            return False
    elif evidence_type in {"span", "paragraph", "refutation"}:
        if not (required.get("span_id") or (required.get("char_start") is not None and required.get("char_end") is not None)):
            return False
        direct_match = all(
            required.get(key) is None or citation.get(key) == required.get(key)
            for key in ("span_id", "char_start", "char_end")
        )
        if not direct_match and not supporting_span_match:
            return False
    else:
        return False

    row_index = required.get("row_index")
    row_cells = [
        cell for cell in _supporting_cells(candidate)
        if row_index is not None and cell.get("row_index") == row_index
    ]
    if evidence_type in {"table", "cell"} and not row_cells and located_cells:
        row_cells = located_cells
    if located_cells:
        # For an exact cell citation, source cell content is authoritative.
        # Top-level fields may inherit table metadata when a cell omits that
        # field, but may never override or contradict an explicit cell value.
        cell_field_keys = {
            "raw_value": "raw_text",
            "normalized_value": "normalized_value",
            "unit": "unit",
            "multiplier": "multiplier",
            "currency": "currency",
            "report_period": "report_period",
            "accounting_basis": "accounting_basis",
        }
        for key, cell_key in cell_field_keys.items():
            cell_values = [cell[cell_key] for cell in located_cells if _has_metadata(cell.get(cell_key))]
            cited_values = [
                value for value in (citation.get(key), candidate.get(key))
                if _has_metadata(value)
            ]
            if cell_values:
                if any(
                    not any(_value_matches(key, cell_value, cited_value) for cell_value in cell_values)
                    for cited_value in cited_values
                ):
                    return False
            elif key in {"unit", "multiplier", "currency", "report_period", "accounting_basis"}:
                source_values = [citation[key]] if _has_metadata(citation.get(key)) else []
                if any(
                    not any(_value_matches(key, source_value, cited_value) for source_value in source_values)
                    for cited_value in cited_values
                ):
                    return False
        metric_cells = [cell.get("raw_text") for cell in row_cells if _has_metadata(cell.get("raw_text"))]
        for metric_value in (citation.get("metric"), candidate.get("metric")):
            if _has_metadata(metric_value) and metric_cells and not any(
                _value_matches("metric", metric_value, cell_value) for cell_value in metric_cells
            ):
                return False

    for key in ("metric", "raw_value", "normalized_value", "unit", "multiplier", "currency", "report_period", "accounting_basis"):
        expected = required.get(key)
        if not _has_metadata(expected):
            continue
        if located_cells:
            if key == "metric":
                observed_values = [
                    cell.get("raw_text") for cell in row_cells
                    if _has_metadata(cell.get("raw_text"))
                ]
            else:
                cell_key = {
                    "raw_value": "raw_text",
                    "normalized_value": "normalized_value",
                    "unit": "unit",
                    "multiplier": "multiplier",
                    "currency": "currency",
                    "report_period": "report_period",
                    "accounting_basis": "accounting_basis",
                }[key]
                observed_values = [
                    cell.get(cell_key) for cell in located_cells
                    if _has_metadata(cell.get(cell_key))
                ]
                if (
                    not observed_values
                    and key in {"unit", "multiplier", "currency", "report_period", "accounting_basis"}
                    and _has_metadata(citation.get(key))
                ):
                    observed_values = [citation[key]]
        else:
            observed_values = []
            if _has_metadata(citation.get(key)):
                observed_values.append(citation[key])
            if _has_metadata(candidate.get(key)):
                observed_values.append(candidate[key])
            value_cells = row_cells
            cell_key = "raw_text" if key == "raw_value" else key
            observed_values.extend(
                cell[cell_key] for cell in value_cells if _has_metadata(cell.get(cell_key))
            )
            if key == "metric":
                observed_values.extend(
                    cell.get("raw_text") for cell in row_cells
                    if _has_metadata(cell.get("raw_text"))
                )
        if not any(_value_matches(key, expected, observed) for observed in observed_values):
            return False
    return True


def _gold_identity(required: Mapping[str, Any]) -> str:
    keys = (
        "source_id", "content_sha256", "page_number", "evidence_type", "table_id", "cell_id", "cell_reference",
        "row_index", "column_index", "row_start", "row_end", "fact_id", "span_id", "char_start",
        "char_end", "evidence_record_id", "metric", "raw_value", "normalized_value", "unit", "multiplier", "currency",
        "report_period", "accounting_basis",
    )
    return json.dumps({key: required.get(key) for key in keys}, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _candidate_identity(candidate: Mapping[str, Any]) -> str:
    citation = _candidate_citation(candidate)
    cells = sorted(
        json.dumps(
            {key: cell.get(key) for key in ("page_number", "row_index", "column_index", "cell_id")},
            sort_keys=True, separators=(",", ":"),
        )
        for cell in _supporting_cells(candidate)
    )
    location = {
        key: citation.get(key)
        for key in (
            "source_id", "content_sha256", "page_number", "table_id", "row_start", "row_end",
            "span_id", "char_start", "char_end", "fact_id", "evidence_record_id",
        )
    }
    location["supporting_cells"] = cells
    has_precise_location = bool(
        citation.get("fact_id")
        or citation.get("evidence_record_id")
        or citation.get("span_id")
        or (citation.get("char_start") is not None and citation.get("char_end") is not None)
        or (citation.get("table_id") and (citation.get("row_start") is not None or cells))
    )
    if not has_precise_location:
        location["evidence_id"] = candidate.get("evidence_id")
        location["block_id"] = candidate.get("block_id")
    return json.dumps(location, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _case_metrics(case: Mapping[str, Any], prediction: Mapping[str, Any], k_values: Sequence[int]) -> dict[str, Any]:
    gold = [_normalized_gold_evidence(item) for item in case.get("required_evidence", []) if isinstance(item, Mapping)]
    _validate_gold_evidence(case.get("case_id"), gold)
    gold_ids = [_gold_identity(item) for item in gold]
    if len(gold_ids) != len(set(gold_ids)):
        raise ValueError(f"case {case.get('case_id')!r} repeats a required evidence location")
    retrieved = [item for item in prediction.get("retrieved", []) if isinstance(item, Mapping)]
    expected_ids = [str(item) for item in prediction.get("evidence_ids", []) if item is not None]
    if expected_ids and retrieved:
        retrieved = [dict(item) for item in retrieved]
        for index, item in enumerate(retrieved):
            item.setdefault("evidence_id", expected_ids[index] if index < len(expected_ids) else None)
    identities = [_candidate_identity(item) for item in retrieved]
    duplicate_count = len(identities) - len(set(identities))
    matches = [[_required_match(required, candidate) for required in gold] for candidate in retrieved]
    seen_candidate_identities: set[str] = set()
    recovered_gold: set[int] = set()
    recovered_by_rank: list[set[int]] = []
    relevant_at_rank: list[bool] = []
    for identity, row in zip(identities, matches):
        new_gold = set() if identity in seen_candidate_identities else {
            index for index, matched in enumerate(row) if matched and index not in recovered_gold
        }
        seen_candidate_identities.add(identity)
        recovered_gold.update(new_gold)
        recovered_by_rank.append(set(recovered_gold))
        relevant_at_rank.append(bool(new_gold))
    first_relevant = next((rank for rank, relevant in enumerate(relevant_at_rank, start=1) if relevant), None)
    actual_answerability = prediction.get("predicted_answerability")
    expected_answerability = str(case.get("answerability"))
    expected_refusal = case.get("expected_refusal_reason")
    actual_refusal = prediction.get("refusal_reason")

    ranked: dict[str, Any] = {}
    for k in k_values:
        if k < 1:
            raise ValueError("evaluation cutoffs must be positive")
        recovered = recovered_by_rank[min(k, len(recovered_by_rank)) - 1] if recovered_by_rank and k > 0 else set()
        relevant_count = sum(relevant_at_rank[:k])
        dcg = sum(1.0 / math.log2(rank + 1) for rank, relevant in enumerate(relevant_at_rank[:k], start=1) if relevant)
        ideal_count = min(len(gold), k)
        ideal_dcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_count + 1))
        ranked[str(k)] = {
            "evidence_recall": len(recovered) / len(gold) if gold else None,
            "complete_evidence_set": bool(gold) and len(recovered) == len(gold),
            "relevant_candidate_count": relevant_count,
            "ndcg": dcg / ideal_dcg if ideal_dcg else None,
        }

    allowed_sources = {str(item.get("source_id")) for item in gold if item.get("source_id")}
    matching_source_citations = sum(
        str(_candidate_citation(candidate).get("source_id")) in allowed_sources
        for candidate in retrieved
    )
    latency = prediction.get("latency_ms")
    latency_value = float(latency) if isinstance(latency, (int, float)) and not isinstance(latency, bool) and latency >= 0 else None
    return {
        "case_id": str(case.get("case_id", "")),
        "has_prediction": bool(prediction),
        "answerability_labeled": expected_answerability in _ANSWERABILITY,
        "partition": case.get("data_partition"),
        "view_status": case.get("view_status"),
        "answerability_correct": actual_answerability == expected_answerability if actual_answerability in _ANSWERABILITY else False,
        "expected_refusal": expected_refusal is not None,
        "refused": actual_refusal is not None,
        "refusal_reason_correct": actual_refusal == expected_refusal if expected_refusal is not None else None,
        "first_relevant_rank": first_relevant,
        "mrr": 1.0 / first_relevant if first_relevant else 0.0,
        "duplicates": duplicate_count,
        "retrieved_count": len(retrieved),
        "citation_source_match_rate": matching_source_citations / len(retrieved) if retrieved and allowed_sources else None,
        "latency_ms": latency_value,
        "at_k": ranked,
    }


def _mean(values: Sequence[float]) -> float | None:
    return statistics.fmean(values) if values else None


def _percentile(values: Sequence[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _metric_summary(scored: Sequence[Mapping[str, Any]], k_values: Sequence[int]) -> dict[str, Any]:
    answerability_labeled = [row for row in scored if row["answerability_labeled"]]
    expected_refusals = [row for row in scored if row["expected_refusal"]]
    refusals_correct = [row for row in expected_refusals if row["refusal_reason_correct"]]
    latency_values = [float(row["latency_ms"]) for row in scored if row["latency_ms"] is not None]
    at_k_summary: dict[str, Any] = {}
    for k in k_values:
        entries = [row["at_k"][str(k)] for row in scored]
        recalls = [float(item["evidence_recall"]) for item in entries if item["evidence_recall"] is not None]
        complete_set = [item["complete_evidence_set"] for item in entries]
        ndcgs = [float(item["ndcg"]) for item in entries if item["ndcg"] is not None]
        at_k_summary[str(k)] = {
            "evidence_recall_mean": _mean(recalls),
            "complete_evidence_set_rate": _mean([1.0 if value else 0.0 for value in complete_set]) if complete_set else None,
            "ndcg_mean": _mean(ndcgs),
            "scored_case_count": len(entries),
        }
    return {
        "at_k": at_k_summary,
        "mrr_mean": _mean([float(row["mrr"]) for row in scored]),
        "answerability_accuracy": _mean([1.0 if row["answerability_correct"] else 0.0 for row in answerability_labeled]),
        "answerability_scored_cases": len(answerability_labeled),
        "refusal_reason_accuracy": len(refusals_correct) / len(expected_refusals) if expected_refusals else None,
        "expected_refusal_cases": len(expected_refusals),
        "duplicate_evidence_count": sum(int(row["duplicates"]) for row in scored),
        "citation_source_match_rate": _mean([
            float(row["citation_source_match_rate"]) for row in scored
            if row["citation_source_match_rate"] is not None
        ]),
        "latency_ms": {
            "mean": _mean(latency_values), "p50": _percentile(latency_values, 0.50),
            "p95": _percentile(latency_values, 0.95), "sample_count": len(latency_values),
        },
    }


def evaluate_manifest(
    manifest_path: str | Path,
    predictions_path: str | Path,
    *,
    k_values: Sequence[int] = (1, 3, 5, 10),
) -> dict[str, Any]:
    """Score explicit stored predictions against a frozen manifest, never an active query."""
    _, manifest_bytes = _read_object(manifest_path)
    _, predictions_bytes = _read_object(predictions_path)
    return evaluate_manifest_bytes(manifest_bytes, predictions_bytes, k_values=k_values)


def evaluate_manifest_bytes(
    manifest_bytes: bytes,
    predictions_bytes: bytes,
    *,
    k_values: Sequence[int] = (1, 3, 5, 10),
) -> dict[str, Any]:
    """Score exact in-memory input bytes after callers have verified their capture hashes."""
    if not k_values or any(isinstance(k, bool) or not isinstance(k, int) or k < 1 for k in k_values) or len(set(k_values)) != len(k_values):
        raise ValueError("evaluation cutoffs must be unique positive integers")
    manifest, manifest_bytes = _read_object_bytes(manifest_bytes, "manifest")
    predictions, predictions_bytes = _read_object_bytes(predictions_bytes, "predictions")
    cases = manifest.get("cases")
    prediction_cases = predictions.get("cases")
    if not isinstance(cases, list) or not isinstance(prediction_cases, list):
        raise ValueError("manifest and prediction file must both contain a cases array")
    case_by_id: dict[str, Mapping[str, Any]] = {}
    for case in cases:
        if not isinstance(case, Mapping) or not case.get("case_id"):
            raise ValueError("benchmark cases require stable case_id values")
        case_id = str(case["case_id"])
        if case_id in case_by_id:
            raise ValueError("benchmark case IDs must be unique")
        case_by_id[case_id] = case
    prediction_by_id: dict[str, Mapping[str, Any]] = {}
    for prediction in prediction_cases:
        if not isinstance(prediction, Mapping) or not prediction.get("case_id"):
            raise ValueError("prediction rows require stable case_id values")
        case_id = str(prediction["case_id"])
        if case_id not in case_by_id:
            raise ValueError(f"prediction refers to unknown case {case_id!r}")
        if case_id in prediction_by_id:
            raise ValueError(f"duplicate prediction for case {case_id!r}")
        prediction_by_id[case_id] = prediction

    scored = [
        _case_metrics(case, prediction_by_id.get(case_id, {}), k_values)
        for case_id, case in case_by_id.items()
    ]
    partitions = Counter(str(row["partition"]) for row in scored)
    views = Counter(str(row["view_status"]) for row in scored)
    independent_cases = [case for case in cases if isinstance(case, Mapping) and case.get("data_partition") == "independent_acceptance"]
    independent_unseen = bool(independent_cases) and all(case.get("view_status") == "unseen" for case in independent_cases)
    predicted_independent = all(str(case["case_id"]) in prediction_by_id for case in independent_cases)
    metrics_by_partition = {
        partition: _metric_summary([row for row in scored if str(row["partition"]) == partition], k_values)
        for partition in sorted(partitions)
    }
    metrics_by_view = {
        view_status: _metric_summary([row for row in scored if str(row["view_status"]) == view_status], k_values)
        for view_status in sorted(views)
    }
    return {
        "schema_version": "adaptive-mas.financial-rag-evaluation.v1",
        "benchmark_id": manifest.get("benchmark_id"),
        "benchmark_sha256": _sha256(manifest_bytes),
        "prediction_sha256": _sha256(predictions_bytes),
        "run_id": predictions.get("run_id"),
        "model_revisions": predictions.get("model_revisions", {}),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "sample_count": len(cases),
        "prediction_count": len(prediction_by_id),
        "partition_counts": dict(sorted(partitions.items())),
        "view_status_counts": dict(sorted(views.items())),
        "diagnostic_only": (
            len(cases) <= 100
            or bool(partitions.get("independent_acceptance", 0) == 0)
            or bool(views.get("public_viewed_diagnostic", 0) + views.get("used_for_development", 0))
        ),
        "acceptance_claim_permitted": (
            len(independent_cases) > 100
            and independent_unseen
            and predicted_independent
        ),
        "metrics": {
            **_metric_summary(scored, k_values),
            "by_partition": metrics_by_partition,
            "by_view_status": metrics_by_view,
            "unmeasured_layers": [
                "parse_accuracy", "claim_semantic_support", "agent_quality", "investment_utility",
                "throughput", "index_latency", "failure_recovery", "production_slo",
            ],
        },
        "cases": scored,
        "limitations": [
            "A small or viewed benchmark is diagnostic only; it cannot establish the plan's 95% acceptance target.",
            "Evidence matching requires exact source/hash/page and original table-cell, fact, or text-span location; it does not judge semantic entailment.",
            "This evaluator reads labels offline and must not be called from a research agent or live query path.",
        ],
    }
