from __future__ import annotations

import json
from datetime import date

import pytest

from gepa.optimize_anything import _build_reflection_prompt_template
from gepa.strategies.instruction_proposal import InstructionProposalSignature

from adaptive_mas.agents.harness import CallBudget, _validate_report
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
        "symbol": "600000",
        "source_type": "financial_statement",
        "title": "Q2 public financial statement",
        "report_period": "2026Q2",
        "published_at": "2026-08-30T18:00:00+08:00",
        "available_at": "2026-08-30T18:00:00+08:00",
        "facts": {"revenue_yoy_pct": 12.5},
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
                    "report_period": "2026Q2",
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
    assert metrics["evaluator_version"] == "adaptive-mas.harness-evaluator.v2"

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
        "report_period": "2026Q2",
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
