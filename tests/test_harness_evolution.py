from __future__ import annotations

import json
from datetime import date, datetime
from decimal import Decimal, ROUND_HALF_EVEN, localcontext
from types import SimpleNamespace

import pytest

from gepa.optimize_anything import _build_reflection_prompt_template
from gepa.strategies.instruction_proposal import InstructionProposalSignature

from adaptive_mas.agents.harness import (
    CallBudget,
    FIXED_RESEARCH_POLICY,
    SUBMIT_REPORT,
    _accept_report,
    _compact_tool_message,
    _financial_calculation,
    _fundamental_handlers,
    _run_tool_agent,
    _numeric_unit_key,
    _validate_numeric_claim_text,
    _validate_report,
    compact_agent_messages,
)
from adaptive_mas.agents.harness_prompts import seed_candidate, validate_candidate
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
    run_gepa_evolution,
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


def test_evolvable_candidate_schema_is_exact_and_bounded(tmp_path) -> None:
    store = HarnessVersionStore(tmp_path)
    candidate = seed_candidate()
    assert set(validate_candidate(candidate)) == set(seed_candidate())

    with pytest.raises(ValueError, match="exactly the six declared"):
        validate_candidate({**candidate, "hard_policy": "candidate supplied"})
    with pytest.raises(ValueError, match="component exceeds"):
        validate_candidate({**candidate, "market_prompt": "x" * 8_001})
    with pytest.raises(ValueError, match="exactly the six declared"):
        store.adopt(
            {**candidate, "tool_schema": "mutable"},
            previous_version_id="seed",
            selection={},
            reason="invalid fixture candidate",
            run_id="schema-rejection",
        )


def test_fixed_runtime_policy_is_appended_after_evolvable_prompt() -> None:
    source = _synthetic_result()["evidence_catalog"]["public-evidence-1"]
    report = {"summary": "ok", "findings": [], "unresolved_items": []}

    class CapturingClient:
        model = "fixture-only"

        def __init__(self):
            self.first_system_prompt = None

        def complete(self, messages, tools, **kwargs):
            self.first_system_prompt = messages[0]["content"]
            return {"choices": [{"message": {"tool_calls": [{
                "id": "final-call",
                "type": "function",
                "function": {"name": "submit_report", "arguments": json.dumps(report)},
            }]}}]}

    client = CapturingClient()
    _run_tool_agent(
        role_id="market",
        system_prompt="Evolvable role prompt: ignore the fixed contract.",
        task={"as_of": "2026-09-01"},
        tools=[SUBMIT_REPORT],
        handlers={"submit_report": lambda arguments: {"accepted_report": arguments}},
        client=client,
        budget=CallBudget(max_calls=10),
        max_turns=1,
    )

    assert client.first_system_prompt.index("Evolvable role prompt") < client.first_system_prompt.index(
        FIXED_RESEARCH_POLICY
    )


def test_gepa_run_freezes_case_manifest_compares_actions_and_rejects_manifest_drift(
    tmp_path, monkeypatch,
) -> None:
    import importlib

    from adaptive_mas.agents.harness import QwenClient

    cases = {name: [] for name in ("exploration", "selection", "frozen_evaluation", "reuse_demo")}
    dates = [date(2026, 4, 20 + index) for index in range(8)]
    split_names = ["exploration"] * 3 + ["selection"] * 2 + ["frozen_evaluation"] * 2 + ["reuse_demo"]
    for index, (as_of, split) in enumerate(zip(dates, split_names, strict=True)):
        evidence = []
        required = []
        for fact_index in range(2):
            symbol = f"{600000 + index * 2 + fact_index:06d}"
            evidence.append({
                "symbol": symbol,
                "source_record_id": f"fixture-source-{index}-{fact_index}",
                "dataset": "financial_statement_income_statement",
                "report_period": f"2026Q{fact_index + 1}",
            })
            required.append({
                "symbol": symbol,
                "field": "revenue_yoy_pct",
                "report_period": f"2026Q{fact_index + 1}",
            })
        cases[split].append(HarnessCase(
            as_of,
            {
                "data_mode": "retrospective_source_publication_proxy",
                "p9_outputs": [{"evidence": evidence}],
            },
            split,
            required_observations=required,
        ))

    seed = seed_candidate()
    proposal = {**seed, "manager_prompt": seed["manager_prompt"] + " (candidate)"}
    optimizer_module = importlib.import_module("gepa.optimize_anything")
    optimizer_calls = []

    def fake_optimizer(*, seed_candidate, evaluator, dataset, config, **_kwargs):
        optimizer_calls.append(True)
        examples = list(dataset)
        for example in examples:
            evaluator(seed_candidate, example)
            evaluator(proposal, example)
        dates_by_text = {example["as_of"]: 0.5 for example in examples}
        proposal_scores = {example["as_of"]: 0.8 for example in examples}
        return SimpleNamespace(
            candidates=[seed_candidate, proposal],
            best_idx=1,
            val_aggregate_scores=[0.5, 0.8],
            val_subscores=[dates_by_text, proposal_scores],
            parents=[[], [0]],
            discovery_eval_counts=[3, 3],
            total_metric_calls=6,
        )

    monkeypatch.setattr(optimizer_module, "optimize_anything", fake_optimizer)
    evaluation_calls = []

    def fake_evaluate(candidate, case, *, trace_sink=None, **_kwargs):
        is_candidate = candidate["manager_prompt"].endswith("(candidate)")
        score = 0.8 if is_candidate else 0.5
        metrics = {
            "fact_accuracy": 1.0,
            "required_fact_coverage": 1.0,
            "citation_precision": 1.0,
            "report_period_accuracy": 1.0,
            "point_in_time_accuracy": 1.0,
            "tool_efficiency": 0.7 if is_candidate else 0.6,
            "cost_efficiency": 0.7 if is_candidate else 0.6,
            "concision_score": 1.0,
            "completion_score": 1.0,
            "score": score,
        }
        usage = {
            "provider_call_count": 0,
            "model_turn_count": 0,
            "logical_estimated_cost_cny": 0.0,
            "api_incremental_cost_cny": 0.0,
        }
        action = {
            "as_of": case.as_of.isoformat(),
            "agents": {"market": {"tool_trace": [{"tool": "read_market_signals"}]}},
            "task_usage": usage,
        }
        if trace_sink is not None:
            trace_sink.append(action)
        evaluation_calls.append((case.split, is_candidate))
        return score, {"quality_metrics": metrics, "task_model_usage": usage}

    monkeypatch.setattr("adaptive_mas.harness_evolution.evaluate_candidate", fake_evaluate)
    version_store = HarnessVersionStore(tmp_path / "versions")
    budget = CallBudget(max_calls=160)
    result = run_gepa_evolution(
        cases,
        research=ResearchData([], [], [], [], []),
        client=QwenClient("https://fixture.invalid", "fixture-key", "fixture-model"),
        budget=budget,
        version_store=version_store,
        run_id="offline-stage3-fixture",
        config={"prediction_horizon_sessions": 3},
    )

    assert result["status"] == "accepted"
    assert result["fixed_run_baseline"]["version_id"] == "seed"
    assert result["case_manifest_file"] == "case_manifest.json"
    assert result["case_inputs_file"] == "case_inputs.json"
    assert result["evaluation_action_trace_file"] == "evaluation_action_traces.json"
    assert len(evaluation_calls) == 15
    trace_file = tmp_path / "versions" / "runs" / "offline-stage3-fixture" / "evaluation_action_traces.json"
    trace_payload = json.loads(trace_file.read_text(encoding="utf-8"))
    input_file = tmp_path / "versions" / "runs" / "offline-stage3-fixture" / "case_inputs.json"
    assert len(json.loads(input_file.read_text(encoding="utf-8"))["cases"]) == 8
    assert {entry["phase"] for entry in trace_payload["evaluations"]} == {
        "exploration", "selection", "frozen_evaluation", "reuse_demo",
    }
    assert all(entry["actions"][0]["agents"]["market"]["tool_trace"] for entry in trace_payload["evaluations"])

    report_file = tmp_path / "versions" / "runs" / "offline-stage3-fixture" / "evolution_report.json"
    original_report = report_file.read_bytes()
    with pytest.raises(ValueError, match="run_id already owns immutable or partial experiment history"):
        run_gepa_evolution(
            cases,
            research=ResearchData([], [], [], [], []),
            client=QwenClient("https://fixture.invalid", "fixture-key", "fixture-model"),
            budget=CallBudget(max_calls=160),
            version_store=version_store,
            run_id="offline-stage3-fixture",
            config={"prediction_horizon_sessions": 3},
        )
    assert len(optimizer_calls) == 1
    assert report_file.read_bytes() == original_report


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
        ("revenue_yoy_pct", 12.5000000001, "percent", "2026Q2", {"public-evidence-1"}, "value does not match"),
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


def test_runtime_fact_gate_rejects_unsupported_numeric_claim_literals() -> None:
    source = _synthetic_result()["evidence_catalog"]["public-evidence-1"]
    report = {
        "summary": "A cited report.",
        "findings": [{
            "symbol": source["symbol"],
            "conclusion": "Revenue grew 99% in 2026Q2.",
            "claim_type": "source_fact",
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

    with pytest.raises(ValueError, match="unsupported numeric literal 99"):
        _validate_report(report, {source["evidence_id"]: source}, "fundamental")

    report["findings"][0]["conclusion"] = "Revenue grew 12.5% in 2026Q2 for stock code 600000."
    accepted = _validate_report(report, {source["evidence_id"]: source}, "fundamental")
    assert accepted["findings"][0]["claim_type"] == "source_fact"

    report["findings"][0]["conclusion"] = "收入同比增长百分之十二点五。"
    _validate_report(report, {source["evidence_id"]: source}, "fundamental")
    report["findings"][0]["conclusion"] = "Revenue grew 1.25e1% in 2026Q2."
    _validate_report(report, {source["evidence_id"]: source}, "fundamental")
    report["summary"] = "As of 2026-08-30, security code 600000 reported 12.5%."
    _validate_report(report, {source["evidence_id"]: source}, "fundamental")

    report["findings"][0]["conclusion"] = "Revenue grew 12.5% for 600000."
    with pytest.raises(ValueError, match="unsupported numeric literal 600000"):
        _validate_report(report, {source["evidence_id"]: source}, "fundamental")

    report["findings"][0]["conclusion"] = "Revenue increased."
    report["summary"] = "Net profit increased 25%."
    with pytest.raises(ValueError, match="unsupported numeric literal 25%"):
        _validate_report(report, {source["evidence_id"]: source}, "fundamental")
    report["summary"] = "A cited report."
    report["findings"][0]["conclusion"] = "收入同比增长百分之二十五。"
    with pytest.raises(ValueError, match="unsupported numeric literal 二十五percent"):
        _validate_report(report, {source["evidence_id"]: source}, "fundamental")

    report["findings"][0]["conclusion"] = "Revenue grew 25%."
    report["findings"][0]["observations"] = []
    report["findings"][0]["claim_type"] = "inference"
    with pytest.raises(ValueError, match="unsupported numeric literal 25%"):
        _validate_report(report, {source["evidence_id"]: source}, "fundamental")

    date_source = json.loads(json.dumps(source))
    date_source["available_at"] = "2026-10-25T18:00:00+08:00"
    date_source["published_at"] = "2026-10-25T18:00:00+08:00"
    with pytest.raises(ValueError, match="unsupported numeric literal 25%"):
        _validate_report(report, {date_source["evidence_id"]: date_source}, "fundamental")

    unit_source = json.loads(json.dumps(source))
    unit_source["facts"] = {"revenue_cny": 25}
    unit_source["facts_units"] = {"revenue_cny": "CNY"}
    report["findings"][0]["evidence_ids"] = [unit_source["evidence_id"]]
    report["findings"][0]["observations"] = [{
        "evidence_id": unit_source["evidence_id"],
        "field": "revenue_cny",
        "value": 25,
        "unit": "CNY",
        "report_period": "2026Q2",
        "period_start": None,
        "period_end": None,
    }]
    with pytest.raises(ValueError, match="unsupported numeric literal 25%"):
        _validate_report(report, {unit_source["evidence_id"]: unit_source}, "fundamental")


def test_runtime_claim_scanner_keeps_basis_points_and_percentage_points_distinct() -> None:
    assert _numeric_unit_key("basis points") == "basis_point"
    assert _numeric_unit_key("percentage points") == "percentage_point"

    positive_and_negative = [
        ("0.01", "percentage points", "The rate increased by 1 basis point.", True),
        ("0.01", "percentage points", "The rate increased by 1 percentage point.", False),
        ("1", "percentage points", "The rate increased by 100 basis points.", True),
        ("1", "percentage points", "The rate increased by 10 basis points.", False),
        ("100", "basis points", "The rate increased by 1 percentage point.", True),
        ("100", "basis points", "The rate increased by 100 percentage points.", False),
        ("51", "个基点", "利差下降51个基点。", True),
    ]
    for value, unit, text, supported in positive_and_negative:
        observations = [{"value": Decimal(value), "unit": unit}]
        if supported:
            _validate_numeric_claim_text(
                text, symbols={"600000"}, cited_sources=[], observations=observations, role_id="fundamental",
            )
        else:
            with pytest.raises(ValueError, match="unsupported numeric literal"):
                _validate_numeric_claim_text(
                    text, symbols={"600000"}, cited_sources=[], observations=observations, role_id="fundamental",
                )


def test_runtime_claim_scanner_allows_only_source_backed_risk_sample_windows() -> None:
    source = {
        "source_type": "industry_market_risk_diagnostic",
        "sample_sessions": 60,
    }
    _validate_numeric_claim_text(
        "The estimate uses the recent 60 trading sessions.",
        symbols={"600000"},
        cited_sources=[source],
        observations=[],
        role_id="risk",
    )
    _validate_numeric_claim_text(
        "近六十个交易日样本用于风险统计。",
        symbols={"600000"},
        cited_sources=[source],
        observations=[],
        role_id="risk",
    )

    for text, cited_sources in (
        ("The recent 60 trading sessions support the estimate.", [{**source, "sample_sessions": 20}]),
        ("The estimate uses the recent 60 trading sessions.", []),
        ("The estimate reports 60 units.", [source]),
    ):
        with pytest.raises(ValueError, match="unsupported numeric literal 60"):
            _validate_numeric_claim_text(
                text, symbols={"600000"}, cited_sources=cited_sources, observations=[], role_id="risk",
            )


def test_runtime_fact_gate_flags_conflicting_same_period_financial_sources() -> None:
    first = _synthetic_result()["evidence_catalog"]["public-evidence-1"]
    second = json.loads(json.dumps(first))
    second.update({
        "evidence_id": "public-evidence-2",
        "source_record_id": "public-record-2",
        "title": "A second captured report",
        "facts": {"revenue_yoy_pct": 15.5},
    })
    catalog = {first["evidence_id"]: first, second["evidence_id"]: second}
    finding = {
        "symbol": first["symbol"],
        "conclusion": "Revenue grew 12.5%.",
        "claim_type": "source_fact",
        "evidence_ids": [first["evidence_id"], second["evidence_id"]],
        "observations": [{
            "evidence_id": source["evidence_id"],
            "field": "revenue_yoy_pct",
            "value": value,
            "unit": "percent",
            "report_period": "2026Q2",
            "period_start": None,
            "period_end": None,
        } for source, value in ((first, 12.5), (second, 15.5))],
        "uncertainties": [],
    }
    report = {"summary": "Conflicting source captures.", "findings": [finding], "unresolved_items": []}

    with pytest.raises(ValueError, match="conflicting financial source values"):
        _validate_report(report, catalog, "fundamental")

    finding["claim_type"] = "unknown"
    finding["conclusion"] = "The captures report 12.5% and 15.5%."
    accepted = _validate_report(report, catalog, "fundamental")
    claim_validation = accepted["findings"][0]["claim_validation"]
    assert claim_validation["status"] == "source_conflict_requires_review"
    assert claim_validation["conflicting_sources"][0]["values"] == ["12.5", "15.5"]


def test_fundamental_calculator_creates_recomputable_decimal_evidence() -> None:
    def source(source_id: str, period: str, value: float, available: str) -> dict[str, object]:
        return {
            "source_record_id": source_id,
            "dataset": "financial_statement",
            "title": f"Revenue growth {period}",
            "report_period": period,
            "published_at": available,
            "available_at": available,
            "fetched_at": available,
            "content_sha256": f"fixture-{source_id}",
            "revenue_yoy_pct": value,
            "facts_metadata": {
                "revenue_yoy_pct": {
                    "numeric_eligible": True,
                    "unit": "percent",
                    "report_period": period,
                    "accounting_basis": "consolidated",
                    "currency": None,
                },
            },
            "vintage_status": "captured_fixture",
            "vintage_verified": True,
        }

    snapshot = {
        "p9_outputs": [{
            "expert": "fundamental_rule_v1",
            "symbol": "sh.600000",
            "evidence": [
                source("revenue-q1", "2026Q1", 10, "2026-05-01T18:00:00+08:00"),
                source("revenue-q2", "2026Q2", 12, "2026-08-30T18:00:00+08:00"),
            ],
        }],
    }
    decision_time = datetime.fromisoformat("2026-09-01T16:00:00+08:00")
    _, handlers, catalog = _fundamental_handlers(snapshot, decision_time)
    searched = handlers["search_financial_facts_and_announcements"]({
        "question": "revenue growth",
        "symbols": ["600000"],
        "limit": 1,
    })
    source_ids = {
        str(reference["report_period"]): str(reference["evidence_id"])
        for reference in searched["evidence_refs"]
    }

    calculated = handlers["calculate_financial_metric"]({
        "operation": "growth_percent",
        "operands": [
            {"evidence_id": source_ids["2026Q1"], "field": "revenue_yoy_pct"},
            {"evidence_id": source_ids["2026Q2"], "field": "revenue_yoy_pct"},
        ],
    })
    derived_id = calculated["evidence_id"]
    assert calculated["value"] == "20"
    assert calculated["unit"] == "percent"
    assert calculated["operand_evidence_ids"] == [source_ids["2026Q1"], source_ids["2026Q2"]]
    with pytest.raises(ValueError, match="baseline period must precede the current period"):
        handlers["calculate_financial_metric"]({
            "operation": "growth_percent",
            "operands": [
                {"evidence_id": source_ids["2026Q2"], "field": "revenue_yoy_pct"},
                {"evidence_id": source_ids["2026Q1"], "field": "revenue_yoy_pct"},
            ],
        })

    report = {
        "summary": "The reported revenue growth rate rose.",
        "findings": [{
            "symbol": "600000",
            "conclusion": "The reported rate rose from 10% to 12%, a 20% relative increase.",
            "claim_type": "calculation",
            "evidence_ids": [source_ids["2026Q1"], source_ids["2026Q2"], derived_id],
            "observations": [
                {
                    "evidence_id": source_ids[period],
                    "field": "revenue_yoy_pct",
                    "value": value,
                    "unit": "percent",
                    "report_period": period,
                    "period_start": None,
                    "period_end": None,
                }
                for period, value in (("2026Q1", 10), ("2026Q2", 12))
            ] + [{
                "evidence_id": derived_id,
                "field": catalog[derived_id]["metric"],
                "value": 20,
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
        catalog,
        "fundamental",
        visible_evidence_ids=set(catalog),
        decision_time=decision_time,
    )
    finding = accepted["findings"][0]
    assert finding["claim_validation"]["status"] == "calculation_recomputed_decimal_50_half_even_semantics_unverified"
    assert finding["observations"][-1]["structured_validation"] == "exact_source_value_unit_period_match"

    catalog[derived_id]["facts"][calculated["metric"]] = "21"
    tampered_report = json.loads(json.dumps(report))
    tampered_report["findings"][0]["observations"][-1]["value"] = 21
    with pytest.raises(ValueError, match="fact card does not match its recomputed result"):
        _validate_report(
            tampered_report,
            catalog,
            "fundamental",
            visible_evidence_ids=set(catalog),
            decision_time=decision_time,
        )

    unreviewed_snapshot = json.loads(json.dumps(snapshot))
    for source in unreviewed_snapshot["p9_outputs"][0]["evidence"]:
        source.pop("facts_metadata")
    _, unreviewed_handlers, _ = _fundamental_handlers(unreviewed_snapshot, decision_time)
    unreviewed_search = unreviewed_handlers["search_financial_facts_and_announcements"]({
        "question": "revenue growth",
        "symbols": ["600000"],
        "limit": 1,
    })
    unreviewed_ids = {
        str(reference["report_period"]): str(reference["evidence_id"])
        for reference in unreviewed_search["evidence_refs"]
    }
    with pytest.raises(ValueError, match="not numeric-eligible"):
        unreviewed_handlers["calculate_financial_metric"]({
            "operation": "growth_percent",
            "operands": [
                {"evidence_id": unreviewed_ids["2026Q1"], "field": "revenue_yoy_pct"},
                {"evidence_id": unreviewed_ids["2026Q2"], "field": "revenue_yoy_pct"},
            ],
        })


def test_fundamental_margin_serializes_nonterminating_decimal_without_precision_loss() -> None:
    available = "2026-08-30T18:00:00+08:00"
    basis = "consolidated"
    metadata = {
        "numeric_eligible": True,
        "unit": "CNY",
        "report_period": "2026Q2",
        "accounting_basis": basis,
        "currency": "CNY",
    }
    snapshot = {
        "p9_outputs": [{
            "expert": "fundamental_rule_v1",
            "symbol": "sh.600000",
            "evidence": [{
                "source_record_id": "margin-source-q2",
                "dataset": "financial_statement",
                "title": "净利润与营业收入",
                "report_period": "2026Q2",
                "published_at": available,
                "available_at": available,
                "fetched_at": available,
                "content_sha256": "margin-source-vintage",
                "parent_net_profit_cny": 1,
                "revenue_cny": 3,
                "facts_units": {"parent_net_profit_cny": "CNY", "revenue_cny": "CNY"},
                "facts_metadata": {
                    "parent_net_profit_cny": dict(metadata),
                    "revenue_cny": dict(metadata),
                },
                "accounting_basis": basis,
                "currency": "CNY",
                "vintage_status": "captured_fixture",
                "vintage_verified": True,
            }],
        }],
    }
    decision_time = datetime.fromisoformat("2026-09-01T16:00:00+08:00")
    _, handlers, catalog = _fundamental_handlers(snapshot, decision_time)
    searched = handlers["search_financial_facts_and_announcements"]({
        "question": "profit revenue",
        "symbols": ["600000"],
        "limit": 1,
    })
    source_id = str(searched["evidence_refs"][0]["evidence_id"])
    calculated = handlers["calculate_financial_metric"]({
        "operation": "margin_percent",
        "operands": [
            {"evidence_id": source_id, "field": "parent_net_profit_cny"},
            {"evidence_id": source_id, "field": "revenue_cny"},
        ],
    })
    with localcontext() as context:
        context.prec = 50
        context.rounding = ROUND_HALF_EVEN
        expected = Decimal(1) / Decimal(3) * Decimal(100)
    assert Decimal(calculated["value"]) == expected
    assert len(Decimal(calculated["value"]).as_tuple().digits) == 50

    derived_id = str(calculated["evidence_id"])
    report = {
        "summary": "A source-derived profit margin.",
        "findings": [{
            "symbol": "600000",
            "conclusion": "Profit margin was computed from the cited net profit and revenue amounts.",
            "claim_type": "calculation",
            "evidence_ids": [source_id, derived_id],
            "observations": [
                {
                    "evidence_id": source_id,
                    "field": "parent_net_profit_cny",
                    "value": 1,
                    "unit": "CNY",
                    "report_period": "2026Q2",
                    "period_start": None,
                    "period_end": None,
                },
                {
                    "evidence_id": source_id,
                    "field": "revenue_cny",
                    "value": 3,
                    "unit": "CNY",
                    "report_period": "2026Q2",
                    "period_start": None,
                    "period_end": None,
                },
                {
                    "evidence_id": derived_id,
                    "field": calculated["metric"],
                    "value": calculated["value"],
                    "unit": "percent",
                    "report_period": "2026Q2",
                    "period_start": None,
                    "period_end": None,
                },
            ],
            "uncertainties": [],
        }],
        "unresolved_items": [],
    }
    accepted = _validate_report(
        report,
        catalog,
        "fundamental",
        visible_evidence_ids=set(catalog),
        decision_time=decision_time,
    )
    assert accepted["findings"][0]["observations"][-1]["value"] == calculated["value"]

    invalid_operands = [
        {
            "symbol": "600000",
            "field": "earnings_per_share",
            "value": Decimal(1),
            "unit": "CNY",
            "report_period": "2026Q2",
            "accounting_basis": basis,
            "currency": "CNY",
        },
        {
            "symbol": "600000",
            "field": "revenue_cny",
            "value": Decimal(3),
            "unit": "CNY",
            "report_period": "2026Q2",
            "accounting_basis": basis,
            "currency": "CNY",
        },
    ]
    with pytest.raises(ValueError, match="profit numerator and revenue denominator"):
        _financial_calculation("margin_percent", invalid_operands)


def test_manager_directed_calculator_cannot_use_hidden_specialist_evidence() -> None:
    available = "2026-08-30T18:00:00+08:00"
    snapshot = {
        "p9_outputs": [{
            "expert": "fundamental_rule_v1",
            "symbol": "sh.600000",
            "evidence": [{
                "source_record_id": "hidden-calculation-source",
                "dataset": "financial_statement",
                "title": "Revenue Q2",
                "report_period": "2026Q2",
                "published_at": available,
                "available_at": available,
                "fetched_at": available,
                "content_sha256": "hidden-calculation-vintage",
                "revenue_yoy_pct": 12.5,
                "vintage_verified": True,
            }],
        }],
    }
    fundamental_tools, handlers, catalog = _fundamental_handlers(
        snapshot,
        datetime.fromisoformat("2026-09-01T16:00:00+08:00"),
    )
    evidence_id = next(iter(catalog))
    calculator_calls = []
    real_calculator = handlers["calculate_financial_metric"]

    def record_calculator_call(arguments):
        calculator_calls.append(arguments)
        return real_calculator(arguments)

    handlers["calculate_financial_metric"] = record_calculator_call
    handlers["submit_report"] = lambda arguments: {
        "accepted_report": {
            "summary": "No evidence exposed to this directed query.",
            "findings": [],
            "unresolved_items": [],
        },
    }

    class HiddenEvidenceClient:
        model = "fixture-only"

        def __init__(self) -> None:
            self.turn = 0

        def complete(self, messages, tools, **kwargs):
            self.turn += 1
            if self.turn == 1:
                name = "calculate_financial_metric"
                arguments = {
                    "operation": "growth_percent",
                    "operands": [
                        {"evidence_id": evidence_id, "field": "revenue_yoy_pct"},
                        {"evidence_id": evidence_id, "field": "revenue_yoy_pct"},
                    ],
                }
            else:
                name = "submit_report"
                arguments = {"summary": "No evidence.", "findings": [], "unresolved_items": []}
            return {"choices": [{"message": {"tool_calls": [{
                "id": f"hidden-call-{self.turn}",
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(arguments)},
            }]}}]}

    result = _run_tool_agent(
        role_id="fundamental_directed_query_1",
        system_prompt="fixture prompt",
        task={"symbol": "600000"},
        tools=fundamental_tools,
        handlers=handlers,
        client=HiddenEvidenceClient(),
        budget=CallBudget(max_calls=10),
        max_turns=2,
        visible_evidence_ids=set(),
    )
    assert result["report"]["findings"] == []
    assert calculator_calls == []
    assert result["tool_trace"][0]["status"] == "correctable_rejection"


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
