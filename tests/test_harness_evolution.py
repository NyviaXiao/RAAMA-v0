from __future__ import annotations

import json
from datetime import date, datetime

import pytest

from gepa.optimize_anything import _build_reflection_prompt_template
from gepa.strategies.instruction_proposal import InstructionProposalSignature

from adaptive_mas.agents.harness import (
    CallBudget,
    SUBMIT_REPORT,
    _accept_report,
    _compact_tool_message,
    _fundamental_handlers,
    _run_tool_agent,
    _validate_report,
    compact_agent_messages,
)
from adaptive_mas.agents.harness_prompts import seed_candidate
from adaptive_mas.harness_evolution import (
    HarnessCase,
    HarnessVersionStore,
    build_retrospective_cases,
    _public_tool_projection,
    _selection_summary,
    _reflection_safe_info,
    score_research_result,
    split_retrospective_cases,
    summarize_retrospective_cases,
)
from adaptive_mas.data.research import MembershipSnapshot, ResearchData


def _synthetic_result() -> dict[str, object]:
    source = {
        "evidence_id": "public-evidence-1",
        "source_record_id": "public-record-1",
        "symbol": "600000",
        "source_type": "financial_statement",
        "title": "Q2 public financial statement",
        "report_period": "2026Q2",
        "published_at": "2026-08-30T18:00:00+08:00",
        "available_at": "2026-08-30T18:00:00+08:00",
        "fetched_at": "2026-08-31T09:00:00+08:00",
        "facts": {"revenue_yoy_pct": 12.5},
        "facts_units": {"revenue_yoy_pct": "percent"},
    }
    report = {
        "summary": "A supported public research summary.",
        "findings": [
            {
                "symbol": "600000",
                "conclusion": "Revenue increased year over year.",
                "evidence_ids": ["public-evidence-1", "public-evidence-1"],
                "observations": [{
                    "evidence_id": "public-evidence-1",
                    "field": "revenue_yoy_pct",
                    "value": 12.5,
                    "unit": "percent",
                    "report_period": "2026Q2",
                    "period_start": None,
                    "period_end": None,
                }],
                "uncertainties": [],
            }
        ],
        "unresolved_items": [],
    }
    empty_report = {"summary": "No additional findings.", "findings": [], "unresolved_items": []}
    return {
        "as_of": "2026-09-01",
        "evidence_catalog": {"public-evidence-1": source},
        "agents": {
            "market": {
                "report": report,
                "tool_trace": [
                    {"tool": "search_financial_facts_and_announcements", "arguments": {}, "result": {
                        "evidence_refs": [{"evidence_id": "public-evidence-1"}],
                    }},
                    {"tool": "search_financial_facts_and_announcements", "arguments": {}, "result": {
                        "evidence_refs": [{"evidence_id": "public-evidence-1"}],
                    }},
                ],
            },
            "fundamental": {"report": empty_report, "tool_trace": []},
            "research_manager": {"report": empty_report, "tool_trace": []},
        },
        "required_observations": [{
            "symbol": "600000",
            "field": "revenue_yoy_pct",
            "report_period": "2026Q2",
        }],
        "task_usage": {
            "provider_call_count": 2,
            "response_cache_reuse_count": 0,
            "model_turn_count": 2,
            "prompt_tokens": 100,
            "completion_tokens": 20,
            "logical_estimated_cost_cny": 0.00384,
            "api_incremental_cost_cny": 0.00384,
        },
    }


def test_reflection_examples_expose_only_the_public_date() -> None:
    case = HarnessCase(
        date(2026, 9, 1),
        {"source_record_ids": ["private-internal-id"], "raw_refs": ["local/path"]},
        "exploration",
    )

    assert case.evaluator_input() == {"as_of": "2026-09-01"}


def test_public_tool_projection_keeps_retrospective_capture_audit_fields() -> None:
    projected = _public_tool_projection("read_prior_evidence", {
        "evidence_id": "public-evidence-1",
        "symbol": "600000",
        "fetched_at": "2026-09-27T09:00:00+08:00",
        "vintage_status": "late_backfill_unverified",
        "retrospective_backfill_proxy": True,
        "vintage_verified": False,
    })

    assert projected["fetched_at"] == "2026-09-27T09:00:00+08:00"
    assert projected["vintage_status"] == "late_backfill_unverified"
    assert projected["retrospective_backfill_proxy"] is True
    assert projected["vintage_verified"] is False


def test_accepted_report_projection_keeps_citation_timing_audit() -> None:
    projected = _public_tool_projection("submit_report", {
        "accepted": True,
        "accepted_report": {
            "summary": "Title-only announcement requires follow-up.",
            "findings": [{
                "symbol": "600000",
                "conclusion": "The title requires follow-up.",
                "evidence_ids": ["evidence-1"],
                "observations": [],
                "citation_validation": [{
                    "evidence_id": "evidence-1",
                    "available_at": "2026-08-30T18:00:00+08:00",
                    "published_at": "2026-08-30T18:00:00+08:00",
                    "fetched_at": None,
                    "vintage_status": "capture_time_unknown",
                    "retrospective_backfill_proxy": False,
                    "source_timing_status": "capture_time_unknown",
                }],
            }],
            "unresolved_items": [],
        },
    })

    timing = projected["report"]["findings"][0]["citation_validation"][0]
    assert timing["source_timing_status"] == "capture_time_unknown"
    assert timing["fetched_at"] is None


def test_retrospective_split_is_3_2_2_plus_distinct_reuse() -> None:
    dates = [date(2026, 1, day) for day in range(1, 9)]
    cases = [
        HarnessCase(day, {}, split)
        for day, split in zip(
            dates,
            ["exploration"] * 3 + ["selection"] * 2 + ["frozen_evaluation"] * 2 + ["reuse_demo"],
            strict=True,
        )
    ]

    groups = split_retrospective_cases(cases)

    assert tuple(len(groups[name]) for name in ("exploration", "selection", "frozen_evaluation", "reuse_demo")) == (3, 2, 2, 1)
    assert groups["reuse_demo"][0].as_of not in {case.as_of for case in cases[:-1]}


def test_retrospective_builder_keeps_company_period_and_announcement_events_in_one_case(monkeypatch) -> None:
    dates = [date(2026, 1, day) for day in range(1, 10)]
    research = ResearchData(
        dates,
        [],
        [],
        [],
        [MembershipSnapshot(dates[0], dates[0], frozenset({"600000", "600001", "600002"}), "fixture")],
    )

    def evidence_record(symbol: str, report_period: str, source_id: str) -> dict[str, object]:
        return {
            "record_kind": "state",
            "expert": "fundamental_rule_v1",
            "symbol": symbol,
            "event_type": "quarterly_fundamental_growth",
            "evidence": [{
                "source_record_id": source_id,
                "dataset": "financial_indicator",
                "report_period": report_period,
                "revenue_yoy_pct": 12.5,
                "published_at": "2026-01-01T18:00:00+08:00",
                "available_at": "2026-01-02T00:00:00+08:00",
            }],
        }

    def fake_snapshot(_cache, decision_time, _horizon, **_kwargs):
        index = dates.index(decision_time.date()) - 1
        report_period = f"{2025 + index // 4}Q{index % 4 + 1}"
        return {
            "p9_outputs": [
                evidence_record("600000", "2025Q1", f"shared-period-{index}"),
                evidence_record("600001", report_period, f"unique-a-{index}"),
                evidence_record("600002", report_period, f"unique-b-{index}"),
                {
                    "record_kind": "event",
                    "expert": "announcement_title_rule_v1",
                    "symbol": "600001",
                    "event_type": "announcement_headline",
                    "evidence": [{
                        "source_record_id": f"announcement-copy-{index}",
                        "dataset": "announcement",
                        "title": "Annual report notice",
                        "url": "https://example.invalid/?announcementId=shared-event",
                        "published_at": "2026-01-01",
                        "available_at": "2026-01-02",
                    }],
                },
            ],
            "p10_outputs": [],
            "macro_outputs": [],
            "attention_outputs": [],
        }

    monkeypatch.setattr("adaptive_mas.harness_evolution.build_public_expert_snapshot", fake_snapshot)
    cases = build_retrospective_cases(
        research,
        "unused-cache-path",
        horizon_sessions=3,
        case_dates=dates[1:9],
        config={
            "feature_trend_sessions": 1,
            "feature_reversal_sessions": 1,
            "feature_liquidity_return_sessions": 1,
            "feature_recent_amount_sessions": 1,
            "feature_reference_amount_sessions": 1,
        },
    )

    case_summary = summarize_retrospective_cases(cases)

    assert [len(case.required_observations) for case in cases] == [2] * 8
    assert all(len(case.public_snapshot["p9_outputs"]) >= 2 for case in cases)
    assert not case_summary["overlapping_case_pairs"]
    assert all(value == 0 for value in case_summary["event_group_overlap_by_split_pair"].values())
    assert case_summary["eligible_for_model_execution"] is True
    assert cases[0].public_snapshot["p9_outputs"][0]["evidence"][0]["source_record_id"] == "shared-period-0"
    assert all(
        "shared-period" not in str(record)
        for case in cases[1:]
        for record in case.public_snapshot["p9_outputs"]
    )
    assert all(
        "announcement-copy" not in str(record)
        for case in cases[1:]
        for record in case.public_snapshot["p9_outputs"]
    )


def test_call_budget_separates_logical_cache_cost_from_incremental_api_cost(tmp_path) -> None:
    ledger = tmp_path / "calls.jsonl"
    budget = CallBudget(max_calls=10, ledger_path=ledger, response_cache_dir=tmp_path / "cache")
    budget.begin_call(120, 1_200, "market", "request-hash")
    budget.finish_call(1, {"prompt_tokens": 100, "completion_tokens": 20}, 1.0)
    budget.record_cache_hit({"prompt_tokens": 100, "completion_tokens": 20}, "market", "request-hash")

    summary = budget.summary()

    assert summary["call_count"] == 1
    assert summary["logical_inference_count"] == 2
    assert summary["incremental_api_cost_cny"] == 0.00192
    assert summary["logical_inference_estimated_cost_cny"] == 0.00384

    reloaded = CallBudget(max_calls=10, ledger_path=ledger, response_cache_dir=tmp_path / "cache")
    assert reloaded.summary()["logical_inference_count"] == 2


def test_persisted_procedural_experience_is_reloaded_with_hash_evidence(tmp_path) -> None:
    store = HarnessVersionStore(tmp_path)
    candidate = seed_candidate()
    candidate["procedural_experience"] += " Keep every exact fact tied to its report period."

    version_id, _ = store.adopt(
        candidate,
        previous_version_id="seed",
        selection={"accepted": True},
        reason="independent selection gain",
        run_id="offline-test",
    )
    loaded, loaded_version_id, version = store.load_active()

    assert loaded_version_id == version_id
    assert loaded == candidate
    assert version["procedural_memory"]["loaded_from_persistent_store"] is True
    assert version["procedural_memory"]["updated_from_prior_task"] is True

    export_path = tmp_path / "active-prompt-export.json"
    exported = store.export_active(export_path)
    persisted_export = json.loads(export_path.read_text(encoding="utf-8"))
    assert persisted_export == exported
    assert set(exported) == {
        "schema_version", "version_id", "candidate_sha256",
        "procedural_experience_sha256", "components", "export_scope",
    }
    assert exported["components"] == candidate


def test_scorer_does_not_reward_duplicate_citations_or_repeated_evidence_reads() -> None:
    result = _synthetic_result()

    metrics = score_research_result(result)

    assert metrics["citation_precision"] == 1.0
    assert metrics["unique_citation_count"] == 1
    assert metrics["duplicate_citation_count"] == 1
    assert metrics["tool_efficiency"] == 0.5
    assert metrics["required_fact_coverage"] == 1.0


def test_scorer_keeps_missing_capture_time_unknown_in_strict_pit_status() -> None:
    result = _synthetic_result()
    result["public_evidence_mode"] = "captured_public_cache_cutoff"
    result["evidence_catalog"]["public-evidence-1"].pop("fetched_at")

    metrics = score_research_result(result)
    report = _reflection_safe_info(result, metrics)

    assert metrics["strict_capture_time_accuracy"] == 0.0
    assert metrics["strict_capture_time_check_count"] == 2
    assert metrics["capture_time_unknown_source_count"] == 1
    assert metrics["strict_historical_pit_verified"] is False
    assert metrics["strict_historical_pit_status"] == "unverified"
    assert "missing_fetched_at" in metrics["strict_historical_pit_reasons"]
    assert report["quality_metrics"]["strict_historical_pit_verified"] is False
    assert report["quality_metrics"]["strict_historical_pit_status"] == "unverified"


def test_scorer_does_not_verify_strict_pit_without_citations() -> None:
    result = _synthetic_result()
    result["public_evidence_mode"] = "captured_public_cache_cutoff"
    finding = result["agents"]["market"]["report"]["findings"][0]
    finding["evidence_ids"] = []
    finding["observations"] = []

    metrics = score_research_result(result)

    assert metrics["strict_capture_time_accuracy"] == 0.0
    assert metrics["strict_capture_time_check_count"] == 0
    assert metrics["strict_historical_pit_verified"] is False
    assert "no_citations_to_verify" in metrics["strict_historical_pit_reasons"]


def test_scorer_does_not_infer_strict_pit_from_an_unknown_evidence_mode() -> None:
    result = _synthetic_result()
    result["public_evidence_mode"] = "unknown"

    metrics = score_research_result(result)

    assert metrics["strict_capture_time_accuracy"] == 1.0
    assert metrics["strict_historical_pit_verified"] is False
    assert "public_evidence_mode_unknown" in metrics["strict_historical_pit_reasons"]


@pytest.mark.parametrize("timestamp_field", ["published_at", "available_at"])
def test_scorer_rejects_strict_pit_when_public_availability_is_after_cutoff(timestamp_field: str) -> None:
    result = _synthetic_result()
    result["public_evidence_mode"] = "captured_public_cache_cutoff"
    result["evidence_catalog"]["public-evidence-1"][timestamp_field] = "2026-09-02T09:00:00+08:00"
    result["evidence_catalog"]["public-evidence-1"]["fetched_at"] = "2026-08-31T09:00:00+08:00"

    metrics = score_research_result(result)

    assert metrics["point_in_time_accuracy"] == 0.0
    assert metrics["strict_capture_time_accuracy"] == 1.0
    assert metrics["strict_historical_pit_verified"] is False
    assert "published_or_available_after_decision_time_or_unknown" in metrics["strict_historical_pit_reasons"]


def test_scorer_maps_exact_market_metric_and_uses_the_actual_required_fact_count() -> None:
    result = _synthetic_result()
    source = result["evidence_catalog"]["public-evidence-1"]
    source["metric"] = "exact_market_metric"
    source["value"] = 12.5
    finding = result["agents"]["market"]["report"]["findings"][0]
    finding["observations"][0]["field"] = "exact_market_metric"
    finding["observations"][0]["value"] = 12.5
    result["required_observations"] = [{
        "symbol": "600000",
        "field": "exact_market_metric",
        "report_period": "2026Q2",
    }]

    metrics = score_research_result(result)

    assert metrics["fact_accuracy"] == 1.0
    assert metrics["required_fact_count"] == 1
    assert metrics["required_fact_coverage"] == 1.0
    assert metrics["evaluator_version"] == "adaptive-mas.harness-evaluator.v3"

    finding["observations"][0]["field"] = "near_match_alias"
    metrics = score_research_result(result)
    assert metrics["fact_accuracy"] == 0.0
    assert metrics["required_fact_coverage"] == 0.0


def test_empty_required_fact_set_is_not_applicable_and_summary_skips_it() -> None:
    result = _synthetic_result()
    result["required_observations"] = []

    metrics = score_research_result(result)
    summary = _selection_summary([
        {"as_of": "2026-09-01", "metrics": metrics},
        {"as_of": "2026-09-02", "metrics": {**metrics, "required_fact_coverage": 1.0}},
    ])

    assert metrics["required_fact_coverage"] is None
    assert metrics["required_fact_coverage_status"] == "not_applicable"
    assert summary["metrics"]["required_fact_coverage"] == 1.0


def test_report_text_and_collection_sizes_are_soft_but_evidence_boundary_is_hard() -> None:
    source = _synthetic_result()["evidence_catalog"]["public-evidence-1"]
    source["source_record_id"] = "public-record-test"
    catalog = {source["evidence_id"]: source}
    observation = {
        "evidence_id": source["evidence_id"],
        "field": "revenue_yoy_pct",
        "value": 12.5,
        "unit": "percent",
        "report_period": "2026Q2",
        "period_start": None,
        "period_end": None,
    }
    report = {
        "summary": "s" * 260,
        "findings": [{
            "symbol": source["symbol"],
            "conclusion": "c" * 150,
            "evidence_ids": [source["evidence_id"]] * 5,
            "observations": [observation, observation],
            "uncertainties": ["u"] * 3,
        } for _ in range(4)],
        "unresolved_items": ["unresolved"] * 5,
    }

    accepted = _validate_report(report, catalog, "market")
    assert len(accepted["findings"]) == 4
    assert len(accepted["findings"][0]["observations"]) == 2
    assert len(accepted["unresolved_items"]) == 5

    report["findings"][0]["evidence_ids"] = ["unknown-source"]
    with pytest.raises(ValueError, match="cited evidence that was not supplied"):
        _validate_report(report, catalog, "market")


@pytest.mark.parametrize(
    ("field", "value", "unit", "report_period", "visible_ids", "error"),
    [
        ("revenue_yoy_pct", 12.6, "percent", "2026Q2", {"public-evidence-1"}, "value does not match"),
        ("revenue_yoy_pct", 12.5, "CNY", "2026Q2", {"public-evidence-1"}, "unit does not match"),
        ("revenue_yoy_pct", 12.5, "percent", "2026Q1", {"public-evidence-1"}, "report period does not match"),
        ("revenue_surprise_pct", 12.5, "percent", "2026Q2", {"public-evidence-1"}, "was not exposed"),
        ("revenue_yoy_pct", 12.5, "percent", "2026Q2", set(), "not exposed to the agent"),
    ],
)
def test_runtime_structured_fact_gate_rejects_wrong_value_unit_period_field_and_hidden_citation(
    field, value, unit, report_period, visible_ids, error
) -> None:
    source = _synthetic_result()["evidence_catalog"]["public-evidence-1"]
    report = {
        "summary": "A cited report.",
        "findings": [{
            "symbol": source["symbol"],
            "conclusion": "Revenue increased.",
            "evidence_ids": [source["evidence_id"]],
            "observations": [{
                "evidence_id": source["evidence_id"],
                "field": field,
                "value": value,
                "unit": unit,
                "report_period": report_period,
                "period_start": None,
                "period_end": None,
            }],
            "uncertainties": [],
        }],
        "unresolved_items": [],
    }

    with pytest.raises(ValueError, match=error):
        _validate_report(
            report,
            {source["evidence_id"]: source},
            "fundamental",
            visible_evidence_ids=visible_ids,
        )


def test_runtime_fact_gate_labels_verified_observations_and_keeps_semantics_separate() -> None:
    source = _synthetic_result()["evidence_catalog"]["public-evidence-1"]
    report = {
        "summary": "A cited report.",
        "findings": [{
            "symbol": source["symbol"],
            "conclusion": "Revenue increased.",
            "evidence_ids": [source["evidence_id"]],
            "observations": [{
                "evidence_id": source["evidence_id"],
                "field": "revenue_yoy_pct",
                "value": 12.5,
                "unit": "percent",
                "report_period": "2026Q2",
                "period_start": None,
                "period_end": None,
            }],
            "uncertainties": [],
        }],
        "unresolved_items": [],
    }

    accepted = _validate_report(
        report,
        {source["evidence_id"]: source},
        "fundamental",
        visible_evidence_ids={source["evidence_id"]},
    )
    finding = accepted["findings"][0]
    assert finding["observations"][0]["structured_validation"] == "exact_source_value_unit_period_match"
    assert finding["numeric_candidate_eligibility"]["expected_returns_changed"] is False
    assert finding["semantic_claim_support"]["status"] == "citations_present_semantic_entailment_not_automatically_verified"


def test_runtime_fact_gate_rejects_symbol_mismatch_and_unknown_report_period() -> None:
    source = _synthetic_result()["evidence_catalog"]["public-evidence-1"]
    report = {
        "summary": "A cited report.",
        "findings": [{
            "symbol": "600001",
            "conclusion": "Revenue increased.",
            "evidence_ids": [source["evidence_id"]],
            "observations": [{
                "evidence_id": source["evidence_id"],
                "field": "revenue_yoy_pct",
                "value": 12.5,
                "unit": "percent",
                "report_period": "2026Q2",
                "period_start": None,
                "period_end": None,
            }],
            "uncertainties": [],
        }],
        "unresolved_items": [],
    }
    with pytest.raises(ValueError, match="reported symbol 600001 without supplied evidence"):
        _validate_report(report, {source["evidence_id"]: source}, "fundamental")

    report["findings"][0]["symbol"] = source["symbol"]
    source["report_period"] = None
    with pytest.raises(ValueError, match="no verified report period"):
        _validate_report(report, {source["evidence_id"]: source}, "fundamental")


def test_runtime_citation_gate_checks_title_only_sources_without_numeric_observations() -> None:
    source = _synthetic_result()["evidence_catalog"]["public-evidence-1"]
    source.update({
        "source_type": "public_announcement_title",
        "facts": {},
        "available_at": "2026-09-02T09:00:00+08:00",
        "published_at": "2026-09-02T08:00:00+08:00",
        "fetched_at": "2026-09-02T09:05:00+08:00",
    })
    report = {
        "summary": "A title-only announcement citation.",
        "findings": [{
            "symbol": source["symbol"],
            "conclusion": "The title requires follow-up.",
            "evidence_ids": [source["evidence_id"]],
            "observations": [],
            "uncertainties": ["The announcement body is unavailable."],
        }],
        "unresolved_items": [],
    }

    with pytest.raises(ValueError, match="unavailable at decision time"):
        _validate_report(
            report,
            {source["evidence_id"]: source},
            "news",
            visible_evidence_ids={source["evidence_id"]},
            decision_time=datetime.fromisoformat("2026-09-01T16:00:00+08:00"),
        )

    source.update({
        "available_at": "2026-08-30T18:00:00+08:00",
        "published_at": "2026-08-30T18:00:00+08:00",
        "fetched_at": None,
    })
    accepted = _validate_report(
        report,
        {source["evidence_id"]: source},
        "news",
        visible_evidence_ids={source["evidence_id"]},
        decision_time=datetime.fromisoformat("2026-09-01T16:00:00+08:00"),
    )
    assert accepted["findings"][0]["citation_validation"][0]["source_timing_status"] == "capture_time_unknown"


def test_runtime_fact_gate_enforces_capture_time_and_labels_retrospective_proxy() -> None:
    source = _synthetic_result()["evidence_catalog"]["public-evidence-1"]
    report = {
        "summary": "A cited report.",
        "findings": [{
            "symbol": source["symbol"],
            "conclusion": "Revenue increased.",
            "evidence_ids": [source["evidence_id"]],
            "observations": [{
                "evidence_id": source["evidence_id"],
                "field": "revenue_yoy_pct",
                "value": 12.5,
                "unit": "percent",
                "report_period": "2026Q2",
                "period_start": None,
                "period_end": None,
            }],
            "uncertainties": [],
        }],
        "unresolved_items": [],
    }
    decision_time = datetime.fromisoformat("2026-09-01T16:00:00+08:00")
    source["fetched_at"] = "2026-09-02T09:00:00+08:00"
    with pytest.raises(ValueError, match="captured after decision time"):
        _validate_report(
            report,
            {source["evidence_id"]: source},
            "fundamental",
            visible_evidence_ids={source["evidence_id"]},
            decision_time=decision_time,
        )

    source["retrospective_backfill_proxy"] = True
    accepted = _validate_report(
        report,
        {source["evidence_id"]: source},
        "fundamental",
        visible_evidence_ids={source["evidence_id"]},
        decision_time=decision_time,
    )
    assert accepted["findings"][0]["observations"][0]["source_timing_status"] == "retrospective_publication_proxy"


def test_context_compaction_preserves_dynamic_facts_identity_vintage_and_reopen_reference() -> None:
    import json

    payload = {
        "evidence_id": "public-evidence-1",
        "source_record_id": "record-1",
        "symbol": "600000",
        "source_type": "structured_financial_facts",
        "dataset": "cached_financial_statements",
        "report_period": "2026Q2",
        "published_at": "2026-08-30T18:00:00+08:00",
        "available_at": "2026-08-30T18:00:00+08:00",
        "fetched_at": "2026-09-01T09:00:00+08:00",
        "vintage": "sha256:fixture-vintage",
        "vintage_status": "late_backfill_unverified",
        "facts": {"non_allowlisted_runtime_metric": 47.25},
        "facts_units": {"non_allowlisted_runtime_metric": "percent"},
        "reopen_ref": {"tool": "read_cached_source", "arguments": {"evidence_id": "public-evidence-1"}},
        "body_text": "x" * 20_000,
    }
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "task"},
        {"role": "assistant", "tool_calls": [{"id": "old-call", "type": "function"}]},
        {"role": "tool", "name": "read_cached_source", "content": json.dumps(payload)},
        {"role": "assistant", "tool_calls": [{"id": "latest-call", "type": "function"}]},
        {"role": "tool", "name": "read_cached_source", "content": json.dumps({"summary": "latest"})},
    ]

    compacted = compact_agent_messages(messages)
    summary = json.loads(compacted[2]["content"])
    preserved = summary["steps"][0]["result"]
    fact_record = next(
        row for row in summary["steps"][0]["result"].get("structured_facts", [preserved])
        if row.get("facts")
    )

    assert fact_record["facts"]["non_allowlisted_runtime_metric"] == 47.25
    assert fact_record["symbol"] == "600000"
    assert fact_record["report_period"] == "2026Q2"
    assert fact_record["published_at"].startswith("2026-08-30")
    assert fact_record["available_at"].startswith("2026-08-30")
    assert fact_record["fetched_at"].startswith("2026-09-01")
    assert fact_record["vintage_status"] == "late_backfill_unverified"
    assert fact_record["facts_units"]["non_allowlisted_runtime_metric"] == "percent"
    assert fact_record["reopen_ref"]["tool"] == "read_cached_source"


def test_reference_only_fallback_keeps_structured_facts_instead_of_dropping_them() -> None:
    import json

    payload = {
        "summary": "s" * 20_000,
        "evidence_id": "public-evidence-2",
        "source_record_id": "record-2",
        "symbol": "600001",
        "source_type": "structured_financial_facts",
        "report_period": "2026Q2",
        "published_at": "2026-08-30T18:00:00+08:00",
        "available_at": "2026-08-30T18:00:00+08:00",
        "fetched_at": "2026-08-30T18:01:00+08:00",
        "vintage_status": "captured",
        "facts": {"dynamic_field_from_source": 5.25},
        "facts_units": {"dynamic_field_from_source": "CNY"},
        "reopen_ref": {"tool": "read_cached_source", "arguments": {"evidence_id": "public-evidence-2"}},
    }
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "task"},
        {"role": "assistant", "tool_calls": [{"id": "old-call", "type": "function"}]},
        {"role": "tool", "name": "read_cached_source", "content": json.dumps(payload)},
        {"role": "assistant", "tool_calls": [{"id": "latest-call", "type": "function"}]},
        {"role": "tool", "name": "read_cached_source", "content": json.dumps({"summary": "latest"})},
    ]

    compacted = compact_agent_messages(messages)
    step = json.loads(compacted[2]["content"])["steps"][0]
    fact_record = step["result"]["structured_facts"][0]

    assert "summary" not in step["result"]
    assert fact_record["facts"]["dynamic_field_from_source"] == 5.25
    assert fact_record["facts_units"]["dynamic_field_from_source"] == "CNY"
    assert fact_record["report_period"] == "2026Q2"
    assert fact_record["reopen_ref"]["arguments"]["evidence_id"] == "public-evidence-2"


def test_fundamental_search_ranks_by_query_and_source_information_not_security_code() -> None:
    def source(symbol: str, source_id: str, title: str, available_at: str) -> dict[str, object]:
        return {
            "source_record_id": source_id,
            "title": title,
            "dataset": "financial_statement",
            "report_period": "2026Q2",
            "published_at": available_at,
            "available_at": available_at,
            "fetched_at": available_at,
            "revenue_yoy_pct": 12.5,
            "vintage_status": "captured",
            "vintage_verified": True,
        }

    snapshot = {
        "p9_outputs": [
            {"expert": "fundamental_rule_v1", "symbol": "sh.000001", "evidence": [
                source("000001", "old", "旧的通用摘要", "2026-07-01T10:00:00+08:00"),
            ]},
            {"expert": "fundamental_rule_v1", "symbol": "sz.999999", "evidence": [
                source("999999", "new", "收入同比变化", "2026-08-30T10:00:00+08:00"),
            ]},
        ],
    }
    _, handlers, _ = _fundamental_handlers(
        snapshot,
        decision_time=datetime.fromisoformat("2026-09-01T16:00:00+08:00"),
    )

    result = handlers["search_financial_facts_and_announcements"]({
        "question": "收入同比",
        "symbols": [],
        "limit": 1,
    })

    assert result["symbol_selection"][0]["symbol"] == "999999"
    assert result["recall_audit"][0]["selected_source_count"] == 1


def test_compacted_fundamental_tool_result_retains_fact_card_and_exact_reopen_path() -> None:
    snapshot = {
        "p9_outputs": [{
            "expert": "fundamental_rule_v1",
            "symbol": "sh.600000",
            "evidence": [{
                "source_record_id": "captured-public-record",
                "dataset": "financial_statement",
                "title": "二季度收入同比变化",
                "report_period": "2026Q2",
                "published_at": "2026-08-30T18:00:00+08:00",
                "available_at": "2026-08-30T18:00:00+08:00",
                "fetched_at": "2026-09-01T09:00:00+08:00",
                "revenue_yoy_pct": 12.5,
                "vintage_status": "captured",
                "vintage_verified": True,
            }],
        }],
    }
    _, handlers, catalog = _fundamental_handlers(
        snapshot,
        decision_time=datetime.fromisoformat("2026-09-01T16:00:00+08:00"),
    )
    search_result = handlers["search_financial_facts_and_announcements"]({
        "question": "收入同比",
        "symbols": ["600000"],
        "limit": 1,
    })
    projected = _compact_tool_message({
        "name": "search_financial_facts_and_announcements",
        "content": json.dumps(search_result, ensure_ascii=False),
    })["result"]
    context = json.loads(projected["context_messages"][0]["content"])
    card = context["public_evidence"][0]
    reference = projected["evidence_refs"][0]
    reopened = handlers["read_cached_source"](reference["reopen_ref"]["arguments"])

    assert card["report_period"] == "2026Q2"
    assert card["facts"]["revenue_yoy_pct"] == 12.5
    assert card["facts_units"]["revenue_yoy_pct"] == "percent"
    assert card["published_at"] == "2026-08-30T18:00:00+08:00"
    assert card["available_at"] == "2026-08-30T18:00:00+08:00"
    assert card["fetched_at"] == "2026-09-01T09:00:00+08:00"
    assert card["vintage_verified"] is True
    assert reference["evidence_id"] in catalog
    assert projected["recall_audit"][0]["selected_source_count"] == 1
    assert reopened["facts"]["revenue_yoy_pct"] == 12.5
    assert reopened["reopen_ref"]["arguments"]["evidence_id"] == reference["evidence_id"]


def test_compaction_fallback_keeps_multi_symbol_source_identity_and_reopens_each_fact() -> None:
    def source(source_id: str, period: str, value: float, day: str) -> dict[str, object]:
        return {
            "source_record_id": source_id,
            "content_sha256": f"fixture-vintage-{source_id}",
            "dataset": "financial_statement",
            "title": f"收入同比{period}",
            "report_period": period,
            "published_at": f"{day}T18:00:00+08:00",
            "available_at": f"{day}T18:00:00+08:00",
            "fetched_at": f"{day}T18:05:00+08:00",
            "revenue_yoy_pct": value,
            "vintage_status": "captured_fixture",
            "vintage_verified": True,
        }

    snapshot = {
        "p9_outputs": [
            {"expert": "fundamental_rule_v1", "symbol": "sh.600000", "evidence": [
                source("600000-q1", "2026Q1", 10.5, "2026-05-01"),
                source("600000-q2", "2026Q2", 12.5, "2026-08-30"),
            ]},
            {"expert": "fundamental_rule_v1", "symbol": "sz.600001", "evidence": [
                source("600001-q1", "2026Q1", -5.5, "2026-05-02"),
                source("600001-q2", "2026Q2", 2.5, "2026-08-31"),
            ]},
        ],
    }
    _, handlers, _ = _fundamental_handlers(
        snapshot,
        decision_time=datetime.fromisoformat("2026-09-01T16:00:00+08:00"),
    )
    tool_result = handlers["search_financial_facts_and_announcements"]({
        "question": "收入同比",
        "symbols": ["600000", "600001"],
        "limit": 2,
    })
    # Large public titles force the reference-only fallback while remaining
    # disposable prose; the facts and their exact reopen identities must survive.
    title_padding = " fixture-title-padding" * 300
    for reference in tool_result["evidence_refs"]:
        reference["title"] = f"{reference['title']}{title_padding}"
    for message in tool_result["context_messages"]:
        context = json.loads(message["content"])
        for card in context.get("public_evidence", []):
            card["exact_title"] = f"{card['exact_title']}{title_padding}"
        message["content"] = json.dumps(context, ensure_ascii=False)
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "fixture task"},
    ]
    for index in range(1):
        messages.extend([
            {"role": "assistant", "tool_calls": [{"id": f"search-{index}", "type": "function"}]},
            {
                "role": "tool",
                "name": "search_financial_facts_and_announcements",
                "content": json.dumps(tool_result, ensure_ascii=False),
            },
        ])
    messages.extend([
        {"role": "assistant", "tool_calls": [{"id": "latest", "type": "function"}]},
        {"role": "tool", "name": "read_cached_source", "content": json.dumps({"summary": "latest"})},
    ])

    compacted = compact_agent_messages(messages)
    summary = json.loads(compacted[2]["content"])
    records = [
        record
        for step in summary["steps"]
        for record in step["result"].get("structured_facts", [])
        if record.get("facts")
    ]
    by_source = {record["evidence_id"]: record for record in records}
    expected = {
        ("600000", "2026Q1"): 10.5,
        ("600000", "2026Q2"): 12.5,
        ("600001", "2026Q1"): -5.5,
        ("600001", "2026Q2"): 2.5,
    }

    assert len(records) == len(by_source) == 4
    for record in by_source.values():
        assert record["reopen_ref"]["arguments"]["evidence_id"] == record["evidence_id"]
        identity = (record["symbol"], record["report_period"])
        assert record["facts"]["revenue_yoy_pct"] == expected[identity]
        reopened = handlers["read_cached_source"](record["reopen_ref"]["arguments"])
        assert reopened["symbol"] == record["symbol"]
        assert reopened["report_period"] == record["report_period"]
        assert reopened["facts"]["revenue_yoy_pct"] == expected[identity]


def test_runtime_allows_only_one_report_correction_with_validation_detail() -> None:
    source = _synthetic_result()["evidence_catalog"]["public-evidence-1"]
    evidence_id = source["evidence_id"]

    def report(value: float) -> dict[str, object]:
        return {
            "summary": "A cited report.",
            "findings": [{
                "symbol": source["symbol"],
                "conclusion": "Revenue increased.",
                "evidence_ids": [evidence_id],
                "observations": [{
                    "evidence_id": evidence_id,
                    "field": "revenue_yoy_pct",
                    "value": value,
                    "unit": "percent",
                    "report_period": "2026Q2",
                    "period_start": None,
                    "period_end": None,
                }],
                "uncertainties": [],
            }],
            "unresolved_items": [],
        }

    class SequenceClient:
        model = "fixture-only"

        def __init__(self) -> None:
            self.inputs = []
            self.outputs = [report(12.6), report(12.5)]

        def complete(self, messages, tools, **kwargs):
            self.inputs.append(messages)
            value = self.outputs.pop(0)
            return {"choices": [{"message": {"tool_calls": [{
                "id": f"call-{len(self.inputs)}",
                "type": "function",
                "function": {"name": "submit_report", "arguments": json.dumps(value)},
            }]}}]}

    client = SequenceClient()
    accepted = _run_tool_agent(
        role_id="fundamental",
        system_prompt="fixture system prompt",
        task={"as_of": "2026-09-01"},
        tools=[SUBMIT_REPORT],
        handlers={"submit_report": lambda arguments: _accept_report(
            arguments,
            {evidence_id: source},
            "fundamental",
            visible_evidence_ids={evidence_id},
            decision_time=datetime.fromisoformat("2026-09-01T16:00:00+08:00"),
        )},
        client=client,
        budget=CallBudget(max_calls=10),
        max_turns=2,
        visible_evidence_ids=set(),
    )

    first_retry_message = next(
        message for message in client.inputs[1]
        if message.get("role") == "tool" and message.get("name") == "submit_report"
    )
    assert json.loads(first_retry_message["content"])["error_category"] == "observation_value_mismatch"
    assert "value does not match" in json.loads(first_retry_message["content"])["validation_detail"]
    assert accepted["report"]["findings"][0]["observations"][0]["value"] == 12.5
    assert len(client.inputs) == 2


def test_runtime_repairs_malformed_tool_arguments_once() -> None:
    source = _synthetic_result()["evidence_catalog"]["public-evidence-1"]
    evidence_id = source["evidence_id"]
    valid_report = {
        "summary": "A cited report.",
        "findings": [{
            "symbol": source["symbol"],
            "conclusion": "Revenue increased.",
            "evidence_ids": [evidence_id],
            "observations": [{
                "evidence_id": evidence_id,
                "field": "revenue_yoy_pct",
                "value": 12.5,
                "unit": "percent",
                "report_period": "2026Q2",
                "period_start": None,
                "period_end": None,
            }],
            "uncertainties": [],
        }],
        "unresolved_items": [],
    }

    class SequenceClient:
        model = "fixture-only"

        def __init__(self) -> None:
            self.inputs = []
            self.outputs = ["{", json.dumps(valid_report)]

        def complete(self, messages, tools, **kwargs):
            self.inputs.append(messages)
            arguments = self.outputs.pop(0)
            return {"choices": [{"message": {"tool_calls": [{
                "id": f"call-{len(self.inputs)}",
                "type": "function",
                "function": {"name": "submit_report", "arguments": arguments},
            }]}}]}

    client = SequenceClient()
    result = _run_tool_agent(
        role_id="fundamental",
        system_prompt="fixture system prompt",
        task={"as_of": "2026-09-01"},
        tools=[SUBMIT_REPORT],
        handlers={"submit_report": lambda arguments: _accept_report(
            arguments,
            {evidence_id: source},
            "fundamental",
            visible_evidence_ids={evidence_id},
            decision_time=datetime.fromisoformat("2026-09-01T16:00:00+08:00"),
        )},
        client=client,
        budget=CallBudget(max_calls=10),
        max_turns=2,
        visible_evidence_ids=set(),
    )

    retry = next(
        message for message in client.inputs[1]
        if message.get("role") == "tool" and message.get("name") == "submit_report"
    )
    assert json.loads(retry["content"])["error_category"] == "malformed_tool_arguments"
    assert result["report"]["findings"][0]["observations"][0]["value"] == 12.5
    assert len(client.inputs) == 2


def test_version_compare_memory_reload_and_explicit_seed_rollback_are_isolated(tmp_path) -> None:
    store = HarnessVersionStore(tmp_path)
    candidate = seed_candidate()
    candidate["procedural_experience"] += " Synthetic fixture content only."
    version_id, _ = store.adopt(
        candidate,
        previous_version_id="seed",
        selection={"fixture_only": True},
        reason="synthetic fixture",
        run_id="synthetic-test",
        origin="synthetic_fixture_only",
        procedural_memory_source="synthetic fixture",
        version_id="synthetic-test-v1",
    )

    compared = store.compare("seed", version_id)
    loaded, loaded_id, version = store.load_active()
    rollback = store.rollback("seed", reason="test fixture cleanup")
    _, restored_id, restored = store.load_active()

    assert compared["changed_components"] == ["procedural_experience"]
    assert loaded_id == version_id and loaded == candidate
    assert version["procedural_memory"]["loaded_from_persistent_store"] is True
    assert version["procedural_memory"]["source"] == "synthetic fixture"
    assert rollback["to_version_id"] == "seed"
    assert restored_id == "seed"
    assert restored["procedural_memory"]["status"] == "seed_defaults"


def test_actual_gepa_reflection_renderer_stays_inside_outgoing_payload_limit() -> None:
    result = _synthetic_result()
    metrics = score_research_result(result)
    side_info = _reflection_safe_info(result, metrics)
    candidate = seed_candidate()
    template = _build_reflection_prompt_template(
        objective="Improve factual accuracy, citations and cost.",
        background="Only public cached research is allowed.",
    )

    assert "source_record_ids" not in json.dumps(side_info, ensure_ascii=False)
    assert "private-internal-id" not in json.dumps(side_info, ensure_ascii=False)
    for component, text in candidate.items():
        prompt = InstructionProposalSignature.prompt_renderer({
            "current_instruction_doc": text,
            "dataset_with_feedback": [side_info],
            "prompt_template": template,
        })
        payload = {
            "model": "qwen3.7-max-2026-06-08",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "max_tokens": 1_200,
            "enable_thinking": False,
        }
        serialized = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        assert len(serialized) <= 24_000, component
