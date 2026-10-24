from datetime import date, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import adaptive_mas.agents.collaborative as collaborative_module
import adaptive_mas.decision as decision_module
from scripts.replay_contextual_rl_feedback import _read_decisions
from scripts.run_raama_decision import _date_seed

from adaptive_mas.agents.collaborative import (
    investment_manager_explanation,
    one_round_challenge,
    research_manager,
)
from adaptive_mas.evaluation.contextual_rl import (
    ACTIONS,
    BASELINE_ACTION,
    apply_action,
    choose_action,
    replay_contextual_feedback,
    update_policy,
)
from adaptive_mas.evaluation.public_expert_overlay import adjust_market_relative_forecast
from adaptive_mas.portfolio.optimizer import OptimizationResult


def test_p11_challenge_is_one_round_cited_and_keeps_original_views():
    shared = {
        "symbol": "600000",
        "evidence": [{
            "source_record_id": "announcement-1",
            "dataset": "cninfo_pdf",
            "available_at": "2026-09-27T16:00:00+08:00",
        }],
    }
    reports = {
        "research_inputs": {"decision_date": "2026-09-27", "horizon_sessions": 3},
        "market": {"findings": []},
        "fundamental": {"findings": [{
            **shared,
            "record_id": "full-text",
            "agent_id": "cninfo_full_text_llm_v1",
            "direction_score": 1,
            "conclusion": "full text supports growth",
            "evidence_text_scope": "full_announcement_pdf_text",
        }]},
        "news": {"findings": [{
            **shared,
            "record_id": "headline",
            "agent_id": "announcement_title_rule_v1",
            "direction_score": -1,
            "conclusion": "headline rule is negative",
            "evidence_text_scope": "headline_only_no_announcement_body",
        }]},
        "market_attention": {"findings": []},
    }

    debate = one_round_challenge(reports)

    assert debate["round_count"] == 1
    assert debate["p11_evidence_challenges"][0]["status"] == "evidence_scope_precedence_resolved"
    assert debate["p11_evidence_challenges"][0]["direction_correctness"] == "not_established"
    assert debate["p11_evidence_challenges"][0]["original_forecasts_changed"] is False
    assert debate["conflicts"][0]["bull_case"][0]["record_id"] == "full-text"


def test_live_qwen_cannot_backfill_history_and_uses_completion_time():
    today = datetime.now(collaborative_module.SHANGHAI).date()
    requested_time = datetime(
        today.year, today.month, today.day, 16, tzinfo=collaborative_module.SHANGHAI
    )
    completed_at = requested_time + timedelta(minutes=17)
    research = SimpleNamespace(snapshot_at=lambda _: object())
    forecast = {
        "horizon_sessions": 3,
        "expected_returns_by_method": {"market_plus_relative_fusion": {}},
        "forecast_errors_by_method": {"market_plus_relative_fusion": 0.01},
    }
    public_snapshot = {
        "p9_outputs": [{
            "record_id": "fundamental-1", "symbol": "600000",
            "expert": "fundamental_rule_v1", "direction_score": 1,
            "rule": "fixture evidence",
            "available_at": requested_time.isoformat(),
            "evidence": [{
                "source_record_id": "source-1", "dataset": "fixture",
                "available_at": requested_time.isoformat(),
            }],
        }],
        "p10_outputs": [], "macro_outputs": [], "attention_outputs": [],
    }
    qwen_result = {
        "status": "completed", "completed_at": completed_at.isoformat(),
        "estimated_token_cost_cny": 0.01, "elapsed_ms": 1.0,
    }
    with (
        patch.object(collaborative_module, "quantitative_agents", return_value={
            "trend": [], "reversal": [], "liquidity_confirmation": [],
        }),
        patch.object(collaborative_module, "call_qwen_research_role", return_value=qwen_result) as call_qwen,
    ):
        with pytest.raises(ValueError, match="historical decision date"):
            collaborative_module.build_agent_reports(
                research, today - timedelta(days=1), forecast, public_snapshot,
                enable_qwen=True,
            )
        call_qwen.assert_not_called()

        reports = collaborative_module.build_agent_reports(
            research, today, forecast, public_snapshot, enable_qwen=True,
        )

    assert call_qwen.call_count == 1
    assert reports["fundamental"]["available_at"] == completed_at.isoformat()
    assert reports["research_inputs"]["input_cutoff"] == today.isoformat()
    assert reports["research_inputs"]["effective_decision_time"] == completed_at.isoformat()


def test_contextual_rl_updates_only_mature_training_feedback():
    trading_dates = tuple(date(2026, 1, 1) + timedelta(days=index) for index in range(10))
    feedback = [
        {
            "data_split": "training", "decision_date": trading_dates[0].isoformat(),
            "label_available_at": trading_dates[3].isoformat(), "horizon_sessions": 3,
            "state": "market_up:aligned:cash_heavy:within_budget",
            "action": BASELINE_ACTION, "net_return_after_costs": 0.02,
            "realized_annualized_risk": 0.4, "risk_budget_annualized": 0.2,
            "starting_nav_cny": 10000, "role_qwen_cost_cny": 20,
            "supplementary_query_cost_cny": 10, "analysis_cost_cny": 30,
        },
        {
            "data_split": "training", "decision_date": trading_dates[5].isoformat(),
            "label_available_at": trading_dates[8].isoformat(), "horizon_sessions": 3,
            "state": "future", "action": BASELINE_ACTION,
            "net_return_after_costs": 0.5, "realized_annualized_risk": 0.1,
            "risk_budget_annualized": 0.2, "starting_nav_cny": 10000,
            "supplementary_query_cost_cny": 0,
        },
        {
            "data_split": "evaluation", "decision_date": trading_dates[0].isoformat(),
            "label_available_at": trading_dates[3].isoformat(), "horizon_sessions": 3,
            "state": "holdout", "action": BASELINE_ACTION,
            "net_return_after_costs": 0.5, "realized_annualized_risk": 0.1,
            "risk_budget_annualized": 0.2, "starting_nav_cny": 10000,
            "supplementary_query_cost_cny": 0,
        },
    ]
    q_values, action_counts = {}, {}

    result = update_policy(
        feedback, trading_dates, trading_dates[7], q_values, action_counts
    )

    assert result["matured_training_feedback_used"] == 1
    assert result["unmatured_training_feedback_ignored"] == 1
    assert result["evaluation_feedback_excluded_from_training"] == 1
    assert q_values["market_up:aligned:cash_heavy:within_budget"][BASELINE_ACTION] == pytest.approx(-0.003)


def test_update_policy_skips_duplicate_feedback_ids_across_replays():
    dates = tuple(date(2026, 2, 1) + timedelta(days=index) for index in range(8))
    feedback = {
        "feedback_id": "stable-feedback-id",
        "data_split": "training",
        "decision_date": dates[5].isoformat(),
        "label_available_at": dates[6].isoformat(),
        "horizon_sessions": 1,
        "state": "market_up:aligned:cash_heavy:within_budget",
        "action": BASELINE_ACTION,
        "net_return_after_costs": 0.01,
        "realized_annualized_risk": 0.10,
        "risk_budget_annualized": 0.20,
        "starting_nav_cny": 100000.0,
        "supplementary_query_cost_cny": 0.0,
    }
    q_values, action_counts = {}, {}
    first = update_policy(
        [feedback, feedback], dates, dates[7], q_values, action_counts,
        processed_feedback_ids=set(),
    )
    assert first["matured_training_feedback_used"] == 1
    assert first["new_feedback_ids_processed"] == 1
    assert first["duplicate_feedback_rows_skipped"] == 1
    assert action_counts[feedback["state"]][BASELINE_ACTION] == 1

    second = update_policy(
        [feedback], dates, dates[7], q_values, action_counts,
        processed_feedback_ids=set(first["processed_feedback_ids"]),
    )
    assert second["matured_training_feedback_used"] == 0
    assert second["new_feedback_ids_processed"] == 0
    assert second["duplicate_feedback_rows_skipped"] == 1
    assert action_counts[feedback["state"]][BASELINE_ACTION] == 1


def test_immature_fixed_p9_forecast_returns_warmup_before_rl_state_construction():
    as_of = date(2026, 3, 2)
    research = SimpleNamespace(trading_dates=[as_of, date(2026, 3, 3)], execution_bars=[])
    portfolio = SimpleNamespace(
        validate=lambda: None, nav_cny=100000.0, cash_cny=100000.0,
        holdings_cny={},
    )
    forecast = {
        "prediction_date": as_of.isoformat(),
        "horizon_sessions": 3,
        "forecast_errors_by_method": {
            "market_plus_relative_fusion": None,
            "loss_gated_fusion": 0.01,
        },
        "expected_returns_by_method": {
            "market_plus_relative_fusion": {"600000": 0.02},
            "loss_gated_fusion": {"600000": 0.03},
        },
        "forecast_error_status": "warming",
        "loss_gate_weights": {"trend": 1.0},
    }
    diagnostic = {
        "prediction_decisions": [forecast],
        "matured_rank_ic_windows_at_end": 0,
        "training_rows_at_end": 0,
        "training_seed": {"matured_rank_ics": {}},
    }
    with (
        patch.object(decision_module, "run_mvp_research", return_value=diagnostic),
        patch.object(decision_module, "_manifest_sha256", return_value=None),
        patch.object(decision_module, "_predictions_by_symbol", return_value={}),
        patch.object(decision_module, "_expert_matured_performance", return_value={}),
        patch.object(decision_module, "build_agent_reports") as build_reports,
        patch.object(decision_module, "contextual_state") as make_state,
    ):
        decision = decision_module.create_daily_decision(
            research,
            {"risk_budget_annualized": 0.20, "reliability_lookback_windows": 10},
            as_of,
            portfolio,
            collaboration_context={},
        )

    assert decision.status == "warmup_no_action"
    assert decision.evidence["collaboration_status"] == "warmup_unavailable_fixed_p9_calibration"
    assert "固定 P9 预测误差尚未成熟" in decision.evidence["reason"]
    build_reports.assert_not_called()
    make_state.assert_not_called()


def test_contextual_rl_needs_comparable_mature_actions_and_blends_calibrated_methods():
    state = "market_down:split:invested:within_budget"
    q_values = {state: {BASELINE_ACTION: 0.1, "trend_tilt_25:no_query": 0.2}}
    action_counts = {state: {BASELINE_ACTION: 20, "trend_tilt_25:no_query": 20}}

    chosen = choose_action(
        state, q_values, action_counts, query_available=False, minimum_samples=20
    )
    forecast = {
        "prediction_date": "2026-09-27",
        "expected_returns_by_method": {
            "market_plus_relative_fusion": {"600000": 0.10},
            "trend": {"600000": 0.20},
        },
        "forecast_errors_by_method": {
            "market_plus_relative_fusion": 0.01,
            "trend": 0.02,
        },
        "target_weights_by_method": {
            "market_plus_relative_fusion": {"600000": 0.5},
        },
    }
    blended = apply_action(forecast, chosen["weights"])

    assert chosen["action"] == "trend_tilt_25:no_query"
    assert chosen["supplementary_query"] is False
    assert blended["expected_returns_by_method"]["rl_controlled_fusion"]["600000"] == pytest.approx(0.125)
    assert blended["forecast_errors_by_method"]["rl_controlled_fusion"] == 0.02
    assert blended["target_weights_by_method"]["rl_controlled_fusion"] == {"600000": 0.5}


def test_cold_start_training_exploration_can_sample_available_actions():
    selected = choose_action(
        "market_up", {}, {}, query_available=True,
        exploration_rate=1.0, seed=7,
    )
    repeated = choose_action(
        "market_up", {}, {}, query_available=True,
        exploration_rate=1.0, seed=7,
    )

    assert selected["explored"] is True
    assert selected["policy_status"] == "training_exploration"
    assert selected["action"] in ACTIONS
    assert selected["action"].endswith(":no_query")
    assert selected["supplementary_query"] is False
    assert selected["query_policy_status"] == "disabled_no_calibrated_numeric_effect"
    assert selected["action"] == repeated["action"]


def test_feedback_cli_reads_single_daily_decision_json(tmp_path):
    decision_path = tmp_path / "daily-decision.json"
    decision_path.write_text(
        '{"as_of_date":"2026-09-22","execution_date":"2026-09-23"}\n',
        encoding="utf-8",
    )

    assert _read_decisions(decision_path) == [{
        "as_of_date": "2026-09-22",
        "execution_date": "2026-09-23",
    }]


def test_training_exploration_seed_is_repeatable_and_changes_by_decision_date():
    first_day = date(2026, 9, 22)

    assert _date_seed(7, first_day) == _date_seed(7, first_day)
    assert _date_seed(7, first_day) != _date_seed(7, first_day + timedelta(days=1))


def test_decision_cost_accounting_sums_role_calls_and_supplementary_cost_once():
    as_of = date(2026, 7, 1)
    methods = (
        "market_plus_relative_fusion", "trend", "reversal", "liquidity_confirmation",
    )
    forecast = {
        "prediction_date": as_of.isoformat(),
        "horizon_sessions": 3,
        "expected_returns_by_method": {
            method: {"600000": 0.01} for method in methods
        },
        "forecast_errors_by_method": {method: 0.01 for method in methods},
        "forecast_error_status": "matured",
        "loss_gate_weights": {method: 0.25 for method in methods},
        "loss_gate_source": "test-fixture",
        "calibration": {},
        "market_annualized_volatility": 0.1,
        "annualized_volatility_by_symbol": {"600000": 0.1},
        "factor_beta_by_symbol": {"600000": 1.0},
        "status": "ready",
    }
    reports = {
        role: {"findings": [], "estimated_token_cost_cny": cost}
        for role, cost in zip(("market", "fundamental", "news", "market_attention"), (1, 2, 3, 4))
    }
    reports["research_inputs"] = {"effective_decision_time": "2026-07-01T16:00:00+08:00"}
    optimization = OptimizationResult(
        target_weights={}, capacity_weights={}, gross_expected_return=0.0,
        forecast_error_allowance=0.0, expected_transaction_cost=0.0,
        annualized_risk=0.0, maximum_one_way_turnover=0.0,
        risk_budget_annualized=0.2, objective_value=0.0,
        holding_objective_value=0.0, incremental_objective_value=0.0,
        risk_penalty=0.0, iterations=1,
    )
    research = SimpleNamespace(
        trading_dates=[as_of, date(2026, 7, 2)], execution_bars=[],
    )
    config = {
        "risk_budget_annualized": 0.2, "risk_lookback_sessions": 60,
        "risk_minimum_observations": 40, "forecast_error_multiplier": 0.5,
        "reliability_lookback_windows": 20, "maximum_symbol_weight": 0.05,
        "maximum_one_way_turnover": 0.5, "commission_per_side_bps": 3.0,
        "transfer_fee_per_side_bps": 0.1, "slippage_per_side_bps": 5.0,
        "stamp_duty_sell_bps": 5.0,
    }
    diagnostic = {
        "prediction_decisions": [forecast], "matured_rank_ic_windows_at_end": 0,
        "training_rows_at_end": 0, "training_seed": {"matured_rank_ics": []},
    }
    with (
        patch.object(decision_module, "run_mvp_research", return_value=diagnostic),
        patch.object(decision_module, "_manifest_sha256", return_value=None),
        patch.object(decision_module, "build_agent_reports", return_value=reports),
        patch.object(decision_module, "add_public_expert_overlay", return_value=(forecast, {})),
        patch.object(decision_module, "one_round_challenge", return_value={
            "conflicts": [], "supplementary_query_cost_cny": 0.0,
        }),
        patch.object(decision_module, "contextual_state", return_value={"state": "market_up"}),
        patch.object(decision_module, "optimize_forecast_target", return_value=(optimization, {})),
    ):
        decision = decision_module.create_daily_decision(
            research, config, as_of, decision_module.PortfolioState(100_000.0, {}),
            collaboration_context={"query_available": True},
        )

    costs = decision.evidence["collaborative_architecture"]["decision_cost_accounting"]
    assert costs["role_qwen_cost_cny"] == 10.0
    assert costs["supplementary_query_cost_cny"] == 0.0
    assert costs["analysis_cost_cny"] == 10.0
    assert costs["estimated_total_decision_cost_cny"] == 10.0


def test_contextual_policy_pools_mature_action_values_across_market_regimes():
    q_values = {
        "market_up": {BASELINE_ACTION: 0.01, "trend_tilt_25:no_query": 0.03},
        "market_down": {BASELINE_ACTION: 0.01, "trend_tilt_25:no_query": 0.03},
    }
    action_counts = {
        "market_up": {BASELINE_ACTION: 10, "trend_tilt_25:no_query": 10},
        "market_down": {BASELINE_ACTION: 10, "trend_tilt_25:no_query": 10},
    }

    chosen = choose_action(
        "market_up", q_values, action_counts,
        query_available=False, minimum_samples=20,
        available_methods={"market_plus_relative_fusion", "trend"},
    )

    assert chosen["action"] == "trend_tilt_25:no_query"
    assert chosen["learned_action_count"] == 20
    assert chosen["policy_status"] == "learned_contextual_bandit"

    state_specific = choose_action(
        "market_up",
        {
            "market_up": {BASELINE_ACTION: 0.02, "trend_tilt_25:no_query": 0.01},
            "market_down": {BASELINE_ACTION: 0.02, "trend_tilt_25:no_query": 0.50},
        },
        {
            "market_up": {BASELINE_ACTION: 20, "trend_tilt_25:no_query": 20},
            "market_down": {BASELINE_ACTION: 20, "trend_tilt_25:no_query": 20},
        },
        query_available=False, minimum_samples=20,
        available_methods={"market_plus_relative_fusion", "trend"},
    )
    assert state_specific["action"] == BASELINE_ACTION
    assert state_specific["estimate_scope"] == "state_specific"


def test_policy_action_becomes_comparable_at_sample_threshold():
    state = "market_up"
    action = "trend_tilt_25:no_query"
    q_values = {state: {BASELINE_ACTION: 0.0, action: 0.1}}
    action_counts = {state: {BASELINE_ACTION: 20, action: 19}}
    available_methods = {"market_plus_relative_fusion", "trend"}

    below_threshold = choose_action(
        state, q_values, action_counts, query_available=False,
        available_methods=available_methods, minimum_samples=20,
    )
    action_counts[state][action] = 20
    at_threshold = choose_action(
        state, q_values, action_counts, query_available=False,
        available_methods=available_methods, minimum_samples=20,
    )

    assert below_threshold["action"] == BASELINE_ACTION
    assert at_threshold["action"] == action
def test_mature_fundamental_news_components_are_rl_selectable_and_gated():
    state = "market_up:aligned:cash_heavy:within_budget"
    action = "fundamental_news_balanced:no_query"
    forecast = {
        "expected_returns_by_method": {"market_plus_relative_fusion": {"600000": 0.10}},
        "forecast_errors_by_method": {"market_plus_relative_fusion": 0.01},
        "target_weights_by_method": {"market_plus_relative_fusion": {"600000": 1.0}},
    }
    signals = [
        {"symbol": "600000", "expert": "fundamental_rule_v1", "horizon_sessions": 3, "direction_score": 1, "record_id": "f-1"},
        {"symbol": "600000", "expert": "cls_news_rule_v1", "horizon_sessions": 3, "direction_score": 1, "record_id": "n-1"},
        {"symbol": "600000", "expert": "market_attention", "horizon_sessions": 3, "direction_score": 1, "record_id": "a-1"},
    ]
    matured_effects = {
        "fundamental_rule_v1:3s": {"status": "candidate_effect_eligible", "candidate_return_adjustment": 0.04},
        "cls_news_rule_v1:3s": {"status": "candidate_effect_eligible", "candidate_return_adjustment": 0.02},
        "market_attention:3s": {"status": "candidate_effect_eligible", "candidate_return_adjustment": 0.50},
    }
    calibrated = adjust_market_relative_forecast(forecast, signals, matured_effects)
    available = {
        name for name, error in calibrated["forecast_errors_by_method"].items()
        if error is not None
    }
    q_values = {state: {BASELINE_ACTION: 0.0, action: 1.0}}
    action_counts = {state: {BASELINE_ACTION: 20, action: 20}}
    chosen = choose_action(
        state, q_values, action_counts, query_available=False,
        available_methods=available, minimum_samples=20,
    )
    blended = apply_action(calibrated, chosen["weights"])

    assert calibrated["expected_returns_by_method"]["market_plus_relative_fusion"]["600000"] == pytest.approx(0.10)
    assert calibrated["expected_returns_by_method"]["fundamental_p11"]["600000"] == pytest.approx(0.14)
    assert calibrated["expected_returns_by_method"]["news_p11"]["600000"] == pytest.approx(0.12)
    assert "attention_p11" not in calibrated["expected_returns_by_method"]
    assert calibrated["public_expert_numeric_component_status"]["market_attention"].startswith("excluded")
    assert calibrated["public_expert_numeric_component_status"]["qwen"].startswith("evidence_only")
    assert chosen["action"] == action
    assert blended["expected_returns_by_method"]["rl_controlled_fusion"]["600000"] == pytest.approx(0.115)

    immature = adjust_market_relative_forecast(forecast, signals, {})
    immature_available = {
        name for name, error in immature["forecast_errors_by_method"].items()
        if error is not None
    }
    cold = choose_action(
        state, q_values, action_counts, query_available=False,
        available_methods=immature_available, minimum_samples=20,
    )
    assert cold["action"] == BASELINE_ACTION
    assert "fundamental_p11" not in immature["expected_returns_by_method"]
    assert "news_p11" not in immature["expected_returns_by_method"]

    all_actions = {
        BASELINE_ACTION: 0.0,
        "fundamental_tilt_25:no_query": 1.0,
        "news_tilt_25:no_query": 2.0,
        "fundamental_news_balanced:no_query": 3.0,
    }
    all_counts = {name: 20 for name in all_actions}
    fundamental_only = adjust_market_relative_forecast(
        forecast, signals, {"fundamental_rule_v1:3s": matured_effects["fundamental_rule_v1:3s"]}
    )
    fundamental_available = {
        name for name, error in fundamental_only["forecast_errors_by_method"].items()
        if error is not None
    }
    fundamental_choice = choose_action(
        state, {state: all_actions}, {state: all_counts}, query_available=False,
        available_methods=fundamental_available, minimum_samples=20,
    )
    assert fundamental_choice["action"] == "fundamental_tilt_25:no_query"

    news_only = adjust_market_relative_forecast(
        forecast, signals, {"cls_news_rule_v1:3s": matured_effects["cls_news_rule_v1:3s"]}
    )
    news_available = {
        name for name, error in news_only["forecast_errors_by_method"].items()
        if error is not None
    }
    news_choice = choose_action(
        state, {state: all_actions}, {state: all_counts}, query_available=False,
        available_methods=news_available, minimum_samples=20,
    )
    assert news_choice["action"] == "news_tilt_25:no_query"


def test_qwen_citations_reach_debate_and_final_manager_view_without_changing_rank():
    source_id = "trend-source-600000"
    reports = {
        "research_inputs": {"decision_date": "2026-09-27", "horizon_sessions": 3},
        "market": {
            "findings": [{
                "record_id": "trend-600000", "agent_id": "trend",
                "symbol": "600000", "direction_score": 1,
                "conclusion": "trend evidence supports the candidate",
                "evidence": [{"source_record_id": source_id, "dataset": "bars"}],
            }],
            "qwen_review": {
                "status": "completed", "model": "qwen-test",
                "conclusion": "the market evidence supports this role-level view",
                "direction_score": 1,
                "supports": [{"source_record_id": source_id, "reason": "positive recent trend"}],
                "opposes": [], "uncertainties": ["short history"],
                "cited_source_record_ids": [source_id],
            },
        },
        "fundamental": {"findings": []},
        "news": {"findings": []},
        "market_attention": {"findings": []},
    }
    debate = one_round_challenge(reports)
    forecast = {
        "expected_returns_by_method": {"rl_controlled_fusion": {"600000": 0.05}},
        "forecast_errors_by_method": {"rl_controlled_fusion": 0.01},
    }

    manager = research_manager(forecast, "rl_controlled_fusion", reports, 0.5)
    final_view = investment_manager_explanation({
        "status": "pending_execution", "as_of_date": "2026-09-27", "prediction_horizon_sessions": 3,
        "annualized_risk": 0.1, "annualized_risk_budget": 0.2,
        "expected_transaction_cost_cny": 100.0,
        "estimated_cash_after_costs_cny": 500000.0,
        "orders": [], "current_holdings_cny": {"600000": 500000.0},
        "target_holdings_cny": {"600000": 500000.0},
        "target_weights": {"600000": 0.5},
    }, manager)

    candidate = manager["candidate_priorities"][0]
    assert debate["qwen_role_reviews"][0]["cited_source_record_ids"] == [source_id]
    assert candidate["qwen_review_evidence"][0]["source_record_id"] == source_id
    assert candidate["qwen_review_evidence"][0]["direction_scope"] == "role_level_only_not_a_per_security_return"
    assert candidate["ranking_score"] == pytest.approx(0.045)
    assert candidate["qwen_evidence_changes_numeric_priority"] is False
    assert final_view["research_context_by_symbol"]["600000"]["qwen_review_evidence"]
def test_hard_risk_review_blocks_optimizer_orders_and_feedback_reports_actual_hold_state():
    decision = decision_module.DailyDecision(
        schema_version="adaptive-mas.decision.v1",
        as_of_date="2026-09-27",
        execution_date="2026-09-28",
        status="pending_execution",
        prediction_status="pending",
        prediction_horizon_sessions=3,
        predictions_by_symbol={"600000": {"expected_return": 0.1}},
        current_nav_cny=1_000_000.0,
        current_cash_cny=500_000.0,
        current_holdings_cny={"600000": 500_000.0},
        target_cash_cny=399_900.0,
        target_holdings_cny={"600000": 600_000.0},
        target_weights={"600000": 0.6},
        estimated_cash_after_costs_cny=399_900.0,
        buy_notional_cny=100_000.0,
        sell_notional_cny=0.0,
        orders=(decision_module.DecisionOrder(
            symbol="600000", side="buy", notional_cny=100_000.0,
            current_value_cny=500_000.0, target_value_cny=600_000.0,
            expected_return=0.1, forecast_error=0.01,
            expected_incremental_value_cny=8_000.0, estimated_cost_cny=100.0,
            estimated_liquidity_capacity_cny=200_000.0,
            decision_date_tradestatus="1", next_session_execution_uncertain=True,
            execution_note="fixture", status="pending_next_session",
            source_prediction_id="pred-test-1", prediction_date="2026-09-27",
            prediction_horizon_sessions=3,
        ),),
        expected_gross_return=0.06,
        forecast_error_allowance=0.01,
        expected_transaction_cost_cny=100.0,
        current_holding_objective_value_cny=0.02,
        optimized_objective_value_cny=0.05,
        risk_adjusted_net_incremental_value_cny=30_000.0,
        risk_penalty_cny=1_000.0,
        annualized_risk=0.24,
        annualized_risk_budget=0.20,
        one_way_turnover=0.05,
        evidence={"risk_model": {
            "market_annualized_volatility": 0.1,
            "annualized_volatility_by_symbol": {"600000": 0.2},
            "factor_beta_by_symbol": {"600000": 1.0},
        }},
        versions={},
    )

    blocked, reviews, outcome = decision_module._hard_risk_review(
        decision,
        {"maximum_symbol_weight": 0.5, "maximum_one_way_turnover": 0.5},
    )
    feedback = investment_manager_explanation(blocked.to_dict())

    assert blocked.status == "risk_blocked"
    assert blocked.orders == ()
    assert blocked.target_holdings_cny == decision.current_holdings_cny
    assert outcome["failed_checks"] == [
        "annualized_risk_within_user_budget", "single_name_weight_within_limit",
    ]
    assert reviews["conservative"]["status"] == "constraint_needs_attention"
    assert outcome["actual_hold_state"]["annualized_risk"] == pytest.approx(0.1)
    assert "所有新增订单已阻断" in feedback["summary"]

    unresolved_input = decision_module.replace(
        decision,
        current_cash_cny=0.0,
        current_holdings_cny={"600000": 1_000_000.0},
        evidence={"risk_model": {
            "market_annualized_volatility": 0.1,
            "annualized_volatility_by_symbol": {"600000": 0.4},
            "factor_beta_by_symbol": {"600000": 1.0},
        }},
    )
    unresolved, _, unresolved_outcome = decision_module._hard_risk_review(
        unresolved_input,
        {"maximum_symbol_weight": 0.5, "maximum_one_way_turnover": 0.5},
    )
    unresolved_feedback = investment_manager_explanation(unresolved.to_dict())
    assert unresolved.status == "risk_blocked"
    assert unresolved_outcome["status"] == "blocked_risk_unresolved"
    assert unresolved_outcome["actual_hold_state"]["risk_within_budget"] is False
    assert unresolved_feedback["status"] == "risk_blocked"


def test_replay_builds_net_account_feedback_and_excludes_holdout_before_update():
    dates = tuple(date(2026, 1, 1) + timedelta(days=index) for index in range(32))
    nav_rows = []
    nav = 100000.0
    for index, day in enumerate(dates):
        daily_return = 0.001 + index * 0.00001
        nav *= 1 + daily_return
        nav_rows.append({
            "trade_date": day.isoformat(), "nav_cny": nav,
            "daily_return": daily_return,
        })
    decisions = [
        {
            "run_id": "train-1", "as_of_date": dates[24].isoformat(),
            "execution_date": dates[25].isoformat(), "current_nav_cny": nav_rows[24]["nav_cny"],
            "annualized_risk_budget": 0.2, "data_split": "training",
            "evidence": {"collaborative_architecture": {
                "rl_controller": {"state": "market_up", "action": "p9_fixed:no_query"},
                "agent_reports": {
                    "market": {"estimated_token_cost_cny": 5.0},
                    "news": {"estimated_token_cost_cny": 2.0},
                },
                "one_round_bull_bear_challenge": {"supplementary_query_cost_cny": 2.0},
            }},
        },
        {
            "run_id": "holdout-1", "decision_date": dates[26].isoformat(),
            "execution_date": dates[27].isoformat(), "current_nav_cny": nav_rows[26]["nav_cny"],
            "annualized_risk_budget": 0.2,
            "evidence": {"collaborative_architecture": {
                "rl_controller": {
                    "state": "market_up", "action": BASELINE_ACTION,
                    "data_split": "training",
                },
            }},
        },
    ]
    q_values, action_counts = {}, {}

    replay = replay_contextual_feedback(
        decisions, nav_rows, dates[27], dates[26], 0.2, q_values, action_counts
    )

    assert replay["training_feedback_rows"] == 1
    assert replay["evaluation_feedback_rows_excluded"] == 1
    assert replay["incompatible_state_rows"] == 0
    assert replay["incompatible_action_rows"] == 0
    assert replay["feedback_rows"][1]["data_split"] == "evaluation"
    assert replay["feedback_rows"][1]["state_compatible"] is True
    assert replay["feedback_rows"][1]["action_compatible"] is True
    assert replay["policy_updates"] == 1
    assert replay["feedback_rows"][0]["net_return_after_costs"] == pytest.approx(nav_rows[25]["daily_return"])
    assert replay["feedback_rows"][0]["role_qwen_cost_cny"] == 7.0
    assert replay["feedback_rows"][0]["supplementary_query_cost_cny"] == 2.0
    assert replay["feedback_rows"][0]["analysis_cost_cny"] == 9.0
    assert q_values["market_up"]["p9_fixed:no_query"] == pytest.approx(
        nav_rows[25]["daily_return"]
        - 0.5 * max(replay["feedback_rows"][0]["realized_annualized_risk"] - 0.2, 0.0) ** 2
        - 9.0 / replay["feedback_rows"][0]["starting_nav_cny"]
    )
    assert "holdout-1" not in {
        row["run_id"] for row in replay["feedback_rows"] if row["data_split"] == "training"
    }


def test_training_replay_requires_recorded_contextual_state_and_action():
    dates = tuple(date(2026, 1, 1) + timedelta(days=index) for index in range(24))
    nav_rows = [
        {"trade_date": day.isoformat(), "nav_cny": 100000 + index, "daily_return": 0.001}
        for index, day in enumerate(dates)
    ]
    decision = {
        "run_id": "missing-context", "decision_date": dates[21].isoformat(),
        "execution_date": dates[22].isoformat(), "current_nav_cny": 100021,
    }

    with pytest.raises(ValueError, match="lacks a compatible market-regime state"):
        replay_contextual_feedback(
            [decision], nav_rows, dates[22], None, 0.2, {}, {}
        )
