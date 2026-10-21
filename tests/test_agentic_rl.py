from datetime import date, timedelta
from math import isclose

from adaptive_mas.backtest.simulator import simulate_portfolio
from adaptive_mas.data.research import MembershipSnapshot, ResearchData
from adaptive_mas.evaluation.agentic_rl import LinearExpectedSarsa, RidgeResidualModel, state_vector
from scripts.infer_agentic_rl_frozen import infer_one
from scripts.run_agentic_rl_historical import (
    _apply_profile,
    _fusion_actions,
    _study_actions,
    _with_research_result,
)


def _research():
    dates = [date(2024, 1, 2) + timedelta(days=index) for index in range(36)]
    adjusted = []
    execution = []
    for symbol, drift, base in (("000001", 0.001, 10.0), ("000002", 0.0005, 20.0)):
        previous = base
        for day in dates:
            opening = previous * 1.0001
            closing = previous * (1 + drift)
            adjusted.append({
                "symbol": symbol, "trade_date": day, "available_at": day,
                "open": opening, "close": closing, "return_1d": drift * 100,
            })
            execution.append({
                "symbol": symbol, "trade_date": day, "available_at": day,
                "open": opening, "close": closing, "amount": 10_000_000.0,
                "tradestatus": "1",
            })
            previous = closing
    index_bars = [
        {"symbol": "sh.000300", "trade_date": day, "available_at": day,
         "open": 1000.0 + index, "close": 1000.0 + index * 1.1}
        for index, day in enumerate(dates)
    ]
    membership = [MembershipSnapshot(dates[0], dates[0], frozenset({"000001", "000002"}), "fixture")]
    return ResearchData(dates, adjusted, execution, index_bars, membership), dates


def test_expected_sarsa_updates_from_matured_three_step_reward():
    learner = LinearExpectedSarsa(alpha=0.2, gamma=0.9)
    features = state_vector({
        "market_return_20s": 0.1,
        "expert_disagreement": 0.5,
        "fundamental_anomaly": 0.2,
        "fundamental_coverage": 1.0,
        "query_available": 0.0,
        "evidence_conflict": 0.0,
        "query_trade_importance": 0.0,
        "account_cash_weight": 0.5,
        "account_holdings_weight": 0.5,
        "risk_to_budget": 0.7,
        "days_since_query": 1.0,
    })
    update = learner.update_n_step(
        "fusion:p9", features, [0.01, 0.0, -0.002], None, (), 0.0,
    )
    assert update["discounted_matured_reward"] > 0
    assert learner.updates == 1
    assert learner.value("fusion:p9", features) > 0


def test_study_and_fusion_are_separate_same_day_stages():
    pre_state = {
        "query_available": 1.0,
        "fundamental_coverage": 1.0,
        "research_performed": 0.0,
    }
    assert _study_actions(pre_state) == ("study:finish", "study:query")
    assert _fusion_actions(_with_research_result(pre_state, None)) == (
        "fusion:p9", "fusion:fundamental",
    )
    event = {"analysis": {
        "direction_score": 1,
        "evidence_sufficiency": "sufficient",
        "conflict_resolved": True,
        "supporting_evidence": ["public_announcement_title"],
        "contrary_evidence": [],
    }}
    post_state = _with_research_result(pre_state, event)
    assert post_state["research_performed"] == 1.0
    assert _fusion_actions(post_state) == (
        "fusion:p9", "fusion:fundamental", "fusion:research", "fusion:joint",
    )


def test_same_day_study_update_bootstraps_separate_fusion_value():
    learner = LinearExpectedSarsa(alpha=0.2, gamma=0.5)
    study_state = state_vector({"query_available": 1.0})
    fusion_state = state_vector({"fusion_stage": 1.0, "research_performed": 1.0})
    learner.weights["fusion:p9"][0] = 0.75
    update = learner.update_one_step(
        "study:query", study_state, 0.0, fusion_state,
        ("fusion:p9",), 0.0, intraday_discount=1.0,
    )
    assert update["discounted_matured_reward"] == 0.0
    assert update["intraday_discount"] == 1.0
    assert update["expected_sarsa_bootstrap"] == 0.75
    assert learner.updates == 1
    assert learner.value("study:query", study_state) > 0


def test_ridge_residual_prediction_uses_only_added_mature_rows():
    model = RidgeResidualModel(("signal",), alpha=0.01, minimum_rows=3)
    model.add([0.0], 0.005, "a")
    model.add([1.0], 0.01, "b")
    model.add([2.0], 0.015, "c")
    model.fit()
    assert model.fitted_rows == 3
    assert 0.015 < model.predict([3.0]) < 0.025


def test_fundamental_adjustments_outside_forecast_basket_are_ignored():
    forecast = {
        "expected_returns_by_method": {
            "market_plus_relative_fusion": {"000001": 0.01},
        },
        "forecast_errors_by_method": {"market_plus_relative_fusion": 0.0},
        "target_weights_by_method": {"market_plus_relative_fusion": {"000001": 0.25}},
    }
    controlled, adjustment = _apply_profile(
        forecast,
        "fundamental",
        facts={},
        fundamental_increments={"000001": 0.002, "600606": 0.3},
        query_event=None,
        research_model=RidgeResidualModel(("feature",)),
    )
    assert adjustment == 0.0
    assert controlled["expected_returns_by_method"]["agentic_fusion"] == {"000001": 0.012}


def test_frozen_agentic_artifacts_restore_for_one_historical_inference():
    policy = LinearExpectedSarsa(alpha=0.015, gamma=0.96)
    policy.weights["study:query"][0] = 1.0
    policy.weights["study:finish"][0] = 0.0
    policy.weights["fusion:research"][0] = 1.0
    model = RidgeResidualModel(tuple("abcde"), alpha=0.01, minimum_rows=1)
    model.add([0.0] * 5, 0.01, "matured-a")
    model.fit()
    artifact = {
        "freeze_date": "2025-06-30",
        "policy": policy.to_dict(),
        "C_text_residual_model": model.to_dict(),
    }
    trace = [{
        "decision_date": "2025-09-15",
        "study_state_features": {"query_available": 1.0},
        "allowed_study_actions": ["study:finish", "study:query"],
        "finish_fusion_state_features": {"fusion_stage": 1.0},
        "finish_allowed_fusion_actions": ["fusion:p9"],
        "query_fusion_state_features": {
            "fusion_stage": 1.0,
            "research_performed": 1.0,
        },
        "query_allowed_fusion_actions": ["fusion:p9", "fusion:research"],
        "query_event_id": "historical-event",
        "query_text_residual_features": [0.0] * 5,
    }]

    result = infer_one(artifact, trace, date(2025, 9, 15))
    assert result["study_action"] == "study:query"
    assert result["fusion_action"] == "fusion:research"
    assert result["research_residual_applied"] is True
    assert isclose(result["research_residual_candidate"], model.predict([0.0] * 5))


def test_research_only_inference_zeros_joint_fundamental_feature():
    policy = LinearExpectedSarsa(alpha=0.015, gamma=0.96)
    policy.weights["study:query"][0] = 1.0
    policy.weights["fusion:research"][0] = 1.0
    model = RidgeResidualModel(tuple("abcde"), alpha=0.01, minimum_rows=1)
    model.add([0.0] * 5, 0.0, "matured-a")
    model.fit()
    model.coef[:] = 0.0
    model.coef[-1] = 0.1
    artifact = {
        "freeze_date": "2025-06-30",
        "policy": policy.to_dict(),
        "C_text_residual_model": model.to_dict(),
    }
    query_features = [0.0, 0.5, 1.0, 0.02, 0.4]
    trace = [{
        "decision_date": "2025-09-15",
        "study_state_features": {"query_available": 1.0},
        "allowed_study_actions": ["study:finish", "study:query"],
        "finish_fusion_state_features": {"fusion_stage": 1.0},
        "finish_allowed_fusion_actions": ["fusion:p9"],
        "query_fusion_state_features": {
            "fusion_stage": 1.0,
            "research_performed": 1.0,
        },
        "query_allowed_fusion_actions": ["fusion:p9", "fusion:research"],
        "query_event_id": "historical-event",
        "query_text_residual_features": query_features,
    }]

    result = infer_one(artifact, trace, date(2025, 9, 15))

    assert result["fusion_action"] == "fusion:research"
    assert result["research_residual_applied"] is True
    assert model.predict(query_features) > 0.0
    assert result["research_residual_candidate"] == 0.0


def test_query_cost_is_debited_once_from_portfolio_nav_and_reported():
    research, dates = _research()
    forecast = {
        "prediction_date": dates[24].isoformat(),
        "horizon_sessions": 3,
        "expected_returns_by_method": {
            "market_plus_relative_fusion": {"000001": 0.02, "000002": 0.01},
        },
        "target_weights_by_method": {
            "market_plus_relative_fusion": {"000001": 0.25, "000002": 0.25},
        },
        "forecast_errors_by_method": {"market_plus_relative_fusion": 0.0},
    }
    config = {
        "optimizer_return_method": "market_plus_relative_fusion",
        "optimizer_candidate_scope": "frozen_method_target",
        "risk_aversion": 0.0,
        "forecast_error_multiplier": 0.0,
        "risk_budget_annualized": 0.20,
        "maximum_symbol_weight": 1.0,
        "maximum_one_way_turnover": 1.0,
        "liquidity_lookback_sessions": 5,
        "maximum_daily_participation": 1.0,
        "risk_lookback_sessions": 20,
        "risk_minimum_observations": 10,
        "commission_per_side_bps": 0.0,
        "transfer_fee_per_side_bps": 0.0,
        "slippage_per_side_bps": 0.0,
        "stamp_duty_sell_bps": 0.0,
    }

    def query_controller(**kwargs):
        return kwargs["forecast"], {"research_cost_cny": 12.34}

    replay = simulate_portfolio(
        research, [], 1_000_000.0,
        {"commission_per_side_bps": 0.0, "transfer_fee_per_side_bps": 0.0,
         "stamp_duty_sell_bps": 0.0, "slippage_per_side_bps": 0.0},
        liquidity_lookback_sessions=5,
        maximum_daily_participation=1.0,
        maximum_one_way_turnover=1.0,
        forecast_decisions=[forecast],
        optimizer_config=config,
        evaluation_start_date=dates[24],
        forecast_controller=query_controller,
    )
    assert replay["daily_nav"][0]["research_cost_cny"] == 12.34
    assert isclose(replay["summary"]["research_cost_cny"], 12.34)
    assert isclose(replay["summary"]["total_cost_including_research_cny"], 12.34)
    assert replay["daily_nav"][0]["cash_cny"] < 1_000_000.0


def test_query_cost_funds_fully_invested_account_with_liquid_holding():
    research, dates = _research()
    forecast = {
        "prediction_date": dates[24].isoformat(),
        "horizon_sessions": 3,
        "expected_returns_by_method": {
            "market_plus_relative_fusion": {"000001": 0.02},
        },
        "target_weights_by_method": {
            "market_plus_relative_fusion": {"000001": 1.0},
        },
        "forecast_errors_by_method": {"market_plus_relative_fusion": 0.0},
    }
    next_forecast = {**forecast, "prediction_date": dates[25].isoformat()}
    costs = {
        "commission_per_side_bps": 0.0,
        "transfer_fee_per_side_bps": 0.0,
        "slippage_per_side_bps": 0.0,
        "stamp_duty_sell_bps": 0.0,
    }
    config = {
        "optimizer_return_method": "market_plus_relative_fusion",
        "optimizer_candidate_scope": "frozen_method_target",
        "risk_aversion": 0.0,
        "forecast_error_multiplier": 0.0,
        "risk_budget_annualized": 0.20,
        "maximum_symbol_weight": 1.0,
        "maximum_one_way_turnover": 1.0,
        "liquidity_lookback_sessions": 5,
        "maximum_daily_participation": 1.0,
        "risk_lookback_sessions": 20,
        "risk_minimum_observations": 10,
        **costs,
    }

    def query_controller(**kwargs):
        fee = 5.0 if kwargs["decision_date"] == dates[25] else 0.0
        return kwargs["forecast"], {"research_cost_cny": fee}

    replay = simulate_portfolio(
        research,
        [],
        1_000.0,
        costs,
        liquidity_lookback_sessions=5,
        maximum_daily_participation=1.0,
        maximum_one_way_turnover=1.0,
        forecast_decisions=[forecast, next_forecast],
        optimizer_config=config,
        evaluation_start_date=dates[24],
        forecast_controller=query_controller,
    )

    fee_day = next(row for row in replay["daily_nav"] if row["research_cost_cny"] == 5.0)
    funding_orders = [
        row for row in replay["orders"]
        if row["status"] == "filled_for_research_fee"
    ]
    assert fee_day["cash_cny"] >= 0.0
    assert len(funding_orders) == 1
    assert isclose(funding_orders[0]["filled_notional_cny"], 5.0)
    assert isclose(replay["summary"]["research_cost_cny"], 5.0)
    assert isclose(replay["summary"]["total_cost_including_research_cny"], 5.0)


def test_research_fee_sale_and_rebalance_share_daily_liquidity_capacity():
    research, dates = _research()
    for row in research.execution_bars:
        if row["symbol"] == "000001":
            row["amount"] = 1_000_000.0 if row["trade_date"] == dates[20] else 1_000.0
    for row in research.adjusted_bars:
        row["close"] = row["open"]
    forecast = {
        "prediction_date": dates[24].isoformat(),
        "horizon_sessions": 3,
        "expected_returns_by_method": {
            "market_plus_relative_fusion": {"000001": 0.02, "000002": 0.01},
        },
        "target_weights_by_method": {
            "market_plus_relative_fusion": {"000001": 0.5, "000002": 0.5},
        },
        "forecast_errors_by_method": {"market_plus_relative_fusion": 0.0},
    }
    next_forecast = {**forecast, "prediction_date": dates[25].isoformat()}
    costs = {
        "commission_per_side_bps": 0.0,
        "transfer_fee_per_side_bps": 0.0,
        "slippage_per_side_bps": 0.0,
        "stamp_duty_sell_bps": 0.0,
    }
    config = {
        "optimizer_return_method": "market_plus_relative_fusion",
        "optimizer_candidate_scope": "frozen_method_target",
        "risk_aversion": 0.0,
        "forecast_error_multiplier": 0.0,
        "risk_budget_annualized": 0.20,
        "maximum_symbol_weight": 0.5,
        "maximum_one_way_turnover": 1.0,
        "liquidity_lookback_sessions": 5,
        "maximum_daily_participation": 0.0065,
        "risk_lookback_sessions": 20,
        "risk_minimum_observations": 10,
        **costs,
    }

    def query_controller(**kwargs):
        fee = 5.0 if kwargs["decision_date"] == dates[25] else 0.0
        return kwargs["forecast"], {"research_cost_cny": fee}

    replay = simulate_portfolio(
        research,
        [],
        1_000.0,
        costs,
        liquidity_lookback_sessions=5,
        maximum_daily_participation=0.0065,
        maximum_one_way_turnover=1.0,
        forecast_decisions=[forecast, next_forecast],
        optimizer_config=config,
        evaluation_start_date=dates[24],
        forecast_controller=query_controller,
    )

    same_symbol_day_orders = [
        row for row in replay["orders"]
        if row["trade_date"] == dates[26].isoformat() and row["symbol"] == "000001"
    ]
    funding_order = next(
        row for row in same_symbol_day_orders
        if row["status"] == "filled_for_research_fee"
    )
    rebalance_buy = next(
        row for row in same_symbol_day_orders if row["side"] == "buy"
    )
    total_filled = sum(float(row["filled_notional_cny"]) for row in same_symbol_day_orders)

    assert isclose(funding_order["filled_notional_cny"], 5.0)
    assert rebalance_buy["status"] == "partial_liquidity_limited"
    assert isclose(rebalance_buy["liquidity_limit_cny"], 1.5, abs_tol=1e-8)
    assert total_filled <= funding_order["liquidity_limit_cny"] + 1e-8
