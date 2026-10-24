from datetime import date, timedelta
import hashlib
import json
from pathlib import Path

import pytest

from adaptive_mas.agents.quantitative import AGENT_IDS
from adaptive_mas.decision import (
    PortfolioState,
    create_daily_decision,
    optimize_forecast_target,
)
from adaptive_mas.backtest.simulator import (
    simulate_fixed_risk_market_exposure,
    simulate_fixed_horizon_portfolio,
    simulate_portfolio,
)
from adaptive_mas.data.research import MembershipSnapshot, ResearchData
from adaptive_mas.models.ridge import ridge_predictions
from adaptive_mas.portfolio.allocation import (
    apply_weight_and_turnover_limits,
    inverse_volatility_weights,
)
from adaptive_mas.regime.market_state import conditional_weights, market_state
from adaptive_mas.evaluation.uncertainty import paired_block_bootstrap
from adaptive_mas.evaluation.mvp_research import (
    STATE_FUSION_FEATURE_SCHEMA,
    _evaluate_matured_window,
    _extend_calibration_moments,
    _fit_return_calibrators,
    _market_horizon_forecast,
    _return_calibration_buckets,
    _state_conditioned_features,
)
from adaptive_mas.evaluation.mvp_research import run_mvp_research
from adaptive_mas.evaluation.post_p8 import (
    expert_prediction_diagnostics,
    paired_block_mean_difference,
)


def _research(status_by_date=None, periods=5):
    dates = [date(2024, 1, 2) + timedelta(days=index) for index in range(periods)]
    adjusted = []
    execution = []
    for symbol, daily_prices in {
        "000001": [(10, 10), (10, 11), (11, 11), (11, 11), (11, 11)],
        "000002": [(20, 20), (20, 20), (20, 21), (21, 21), (21, 21)],
    }.items():
        daily_prices = daily_prices + [daily_prices[-1]] * (periods - len(daily_prices))
        for day, (open_price, close_price) in zip(dates, daily_prices, strict=True):
            adjusted.append(
                {
                    "symbol": symbol,
                    "trade_date": day,
                    "available_at": day,
                    "open": float(open_price),
                    "high": float(max(open_price, close_price)),
                    "low": float(min(open_price, close_price)),
                    "close": float(close_price),
                    "return_1d": 0.0,
                    "source_ref": "synthetic",
                }
            )
            status = (status_by_date or {}).get((symbol, day), "1")
            execution.append(
                {
                    "symbol": symbol,
                    "trade_date": day,
                    "available_at": day,
                    "open": float(open_price),
                    "close": float(close_price),
                    "amount": 1000000.0,
                    "tradestatus": status,
                    "source_ref": "synthetic",
                }
            )
    index_bars = [
        {
            "symbol": "sh.000300",
            "trade_date": day,
            "available_at": day,
            "open": 999.0 + index,
            "close": 1000.0 + index,
        }
        for index, day in enumerate(dates)
    ]
    membership = [
        MembershipSnapshot(dates[0], dates[0], frozenset({"000001", "000002"}), "synthetic")
    ]
    return ResearchData(dates, adjusted, execution, index_bars, membership), dates


def _walk_forward_research(reverse_future=False):
    dates = [date(2024, 1, 2) + timedelta(days=index) for index in range(36)]
    symbols = ("000001", "000002", "600000", "600001")
    adjusted = []
    execution = []
    for symbol_index, symbol in enumerate(symbols):
        previous_close = 10.0 * (symbol_index + 1)
        for day_index, trade_date in enumerate(dates):
            rate = (symbol_index - 1.5) * 0.001 + ((day_index % 4) - 1.5) * 0.0002
            if reverse_future and day_index >= 30:
                rate = -rate
            open_price = previous_close * (1 + rate * 0.25)
            close_price = previous_close * (1 + rate)
            high = max(open_price, close_price) * 1.001
            low = min(open_price, close_price) * 0.999
            amount = (
                1_000_000.0
                * (symbol_index + 1)
                * (1 + day_index**2 * 0.00001 * (symbol_index + 1))
            )
            adjusted.append(
                {
                    "symbol": symbol,
                    "trade_date": trade_date,
                    "available_at": trade_date,
                    "open": open_price,
                    "high": high,
                    "low": low,
                    "close": close_price,
                    "return_1d": rate * 100,
                    "amount": amount,
                    "source_ref": "synthetic",
                }
            )
            execution.append(
                {
                    "symbol": symbol,
                    "trade_date": trade_date,
                    "available_at": trade_date,
                    "open": open_price,
                    "high": high,
                    "low": low,
                    "close": close_price,
                    "amount": amount,
                    "tradestatus": "1",
                    "source_ref": "synthetic",
                }
            )
            previous_close = close_price
    index_bars = [
        {
            "symbol": "sh.000300",
            "trade_date": trade_date,
            "available_at": trade_date,
            "open": 1000.0 * (1 + index * 0.0005) / 1.001,
            "close": 1000.0 * (1 + index * 0.0005),
        }
        for index, trade_date in enumerate(dates)
    ]
    membership = [
        MembershipSnapshot(dates[0], dates[0], frozenset(symbols), "synthetic")
    ]
    return ResearchData(dates, adjusted, execution, index_bars, membership)


def _state_conditioned_research(reverse_future=False):
    research = _walk_forward_research(reverse_future=reverse_future)
    index_closes = [1000.0 + index for index in range(20)] + [
        900.0, 1200.0, 1180.0, 1160.0, 1140.0, 1120.0, 1100.0, 1300.0,
        1280.0, 1260.0, 1240.0, 1220.0, 1200.0, 1400.0, 1380.0, 1360.0,
    ]
    index_bars = [dict(row) for row in research.index_bars]
    for row, close in zip(index_bars, index_closes, strict=True):
        row["close"] = close
        row["open"] = close / 1.001
    return ResearchData(
        list(research.trading_dates),
        list(research.adjusted_bars),
        list(research.execution_bars),
        index_bars,
        list(research.membership_snapshots),
    )


def _research_through(research, cutoff):
    return ResearchData(
        [day for day in research.trading_dates if day <= cutoff],
        [row for row in research.adjusted_bars if row["trade_date"] <= cutoff],
        [row for row in research.execution_bars if row["trade_date"] <= cutoff],
        [row for row in research.index_bars if row["trade_date"] <= cutoff],
        [row for row in research.membership_snapshots if row.available_at <= cutoff],
    )


def test_ridge_standardizes_from_training_fold_and_ranks_signal():
    predictions = ridge_predictions(
        [(0.0, 0.0), (1.0, 1.0), (2.0, 2.0), (3.0, 3.0)],
        [0.0, 0.01, 0.02, 0.03],
        {"000001": (4.0, 4.0), "000002": (1.5, 1.5)},
        alpha=1.0,
    )

    assert predictions["000001"] > predictions["000002"]


def test_inverse_volatility_weights_obey_name_cap_and_turnover_limit():
    scores = {f"{index:06d}": float(100 - index) for index in range(30)}
    returns = {
        symbol: [0.001 * (index + 1) for index in range(40)]
        for symbol in scores
    }
    target = inverse_volatility_weights(
        scores,
        returns,
        fraction=1.0,
        invested_weight=0.95,
        maximum_symbol_weight=0.05,
        minimum_observations=20,
    )
    previous = {symbol: 0.95 / 30 for symbol in list(scores)[-30:]}
    limited = apply_weight_and_turnover_limits(target, previous, 0.05, 0.10)
    one_way_turnover = sum(
        abs(limited.get(symbol, 0.0) - previous.get(symbol, 0.0))
        for symbol in target.keys() | previous.keys()
    ) / 2

    assert abs(sum(target.values()) - 0.95) < 1e-9
    assert max(target.values()) <= 0.05
    assert one_way_turnover <= 0.10 + 1e-9


def test_daily_backtest_uses_next_open_and_records_costs():
    research, dates = _research()
    result = simulate_portfolio(
        research,
        [{"decision_date": dates[0], "target_weights": {"000001": 0.5}}],
        1_000_000.0,
        {
            "commission_per_side_bps": 0.0,
            "transfer_fee_per_side_bps": 0.0,
            "stamp_duty_sell_bps": 0.0,
            "slippage_per_side_bps": 0.0,
        },
        liquidity_lookback_sessions=1,
        maximum_daily_participation=1.0,
    )

    assert result["orders"][0]["trade_date"] == dates[1].isoformat()
    assert result["orders"][0]["status"] == "filled"
    assert result["summary"]["final_nav_cny"] == 1_050_000.0
    assert abs(result["daily_nav"][0]["benchmark_return"] - 0.001) < 1e-12
    assert result["daily_nav"][0]["cash_cny"] + sum(
        row["market_value_cny"]
        for row in result["positions"]
        if row["trade_date"] == dates[1].isoformat()
    ) == result["daily_nav"][0]["nav_cny"]


def test_matched_simulation_starts_from_cash_on_common_evaluation_date():
    research, dates = _research(periods=7)
    result = simulate_portfolio(
        research,
        [
            {"decision_date": dates[0], "target_weights": {"000001": 0.5}},
            {"decision_date": dates[3], "target_weights": {"000001": 0.5}},
        ],
        1_000_000.0,
        {
            "commission_per_side_bps": 0.0,
            "transfer_fee_per_side_bps": 0.0,
            "stamp_duty_sell_bps": 0.0,
            "slippage_per_side_bps": 0.0,
        },
        liquidity_lookback_sessions=1,
        maximum_daily_participation=1.0,
        evaluation_start_date=dates[2],
    )

    assert result["daily_nav"][0]["trade_date"] == dates[3].isoformat()
    assert result["daily_nav"][0]["cash_cny"] == 1_000_000.0
    assert result["orders"][0]["trade_date"] == dates[4].isoformat()


def test_short_simulation_preserves_same_initial_account_cash_and_holdings():
    research, dates = _research(periods=5)
    result = simulate_portfolio(
        research,
        [{"decision_date": dates[0], "target_weights": {"000001": 0.5}}],
        1_000_000.0,
        {
            "commission_per_side_bps": 0.0,
            "transfer_fee_per_side_bps": 0.0,
            "stamp_duty_sell_bps": 0.0,
            "slippage_per_side_bps": 0.0,
        },
        liquidity_lookback_sessions=1,
        maximum_daily_participation=1.0,
        evaluation_start_date=dates[0],
        initial_account={"cash_cny": 500_000.0, "holdings_cny": {"000001": 500_000.0}},
    )

    assert result["daily_nav"][0]["cash_cny"] == 500_000.0
    assert result["daily_nav"][0]["nav_cny"] == 1_050_000.0
    assert result["summary"]["initial_capital_cny"] == 1_000_000.0


def test_suspended_entry_is_reported_as_unfilled():
    _, dates = _research()
    research, _ = _research({("000001", dates[1]): "0"})
    result = simulate_portfolio(
        research,
        [{"decision_date": dates[0], "target_weights": {"000001": 0.5}}],
        1_000_000.0,
        {
            "commission_per_side_bps": 0.0,
            "transfer_fee_per_side_bps": 0.0,
            "stamp_duty_sell_bps": 0.0,
            "slippage_per_side_bps": 0.0,
        },
        liquidity_lookback_sessions=1,
        maximum_daily_participation=1.0,
    )

    assert result["summary"]["blocked_orders"] == 1
    assert result["orders"][0]["status"] == "unfilled_suspended_or_missing"
    assert result["summary"]["final_nav_cny"] == 1_000_000.0


def test_daily_participation_caps_fills_from_prior_traded_amount():
    research, dates = _research()
    result = simulate_portfolio(
        research,
        [{"decision_date": dates[0], "target_weights": {"000001": 0.5}}],
        1_000_000.0,
        {
            "commission_per_side_bps": 0.0,
            "transfer_fee_per_side_bps": 0.0,
            "stamp_duty_sell_bps": 0.0,
            "slippage_per_side_bps": 0.0,
        },
        liquidity_lookback_sessions=1,
        maximum_daily_participation=0.01,
    )

    assert result["orders"][0]["liquidity_limit_cny"] == 10000.0
    assert result["orders"][0]["filled_notional_cny"] == 10000.0
    assert result["orders"][0]["status"] == "partial_liquidity_limited"
    assert result["summary"]["liquidity_limited_orders"] == 1


def test_fixed_horizon_entry_and_predeclared_fifth_session_exit_reconcile_cash():
    research, dates = _research(periods=7)
    result = simulate_fixed_horizon_portfolio(
        research,
        [{"decision_date": dates[0], "target_weights": {"000001": 0.5}}],
        1_000_000.0,
        {
            "commission_per_side_bps": 0.0,
            "transfer_fee_per_side_bps": 0.0,
            "stamp_duty_sell_bps": 0.0,
            "slippage_per_side_bps": 0.0,
        },
        liquidity_lookback_sessions=1,
        maximum_daily_participation=1.0,
    )

    assert [row["order_type"] for row in result["orders"]] == [
        "entry_open",
        "planned_exit_close",
    ]
    assert result["orders"][0]["trade_date"] == dates[1].isoformat()
    assert result["orders"][1]["trade_date"] == dates[5].isoformat()
    assert result["orders"][1]["planned_exit_date"] == dates[5].isoformat()
    assert result["summary"]["final_nav_cny"] == 1_050_000.0
    assert abs(result["daily_nav"][0]["benchmark_return"] - 0.001) < 1e-12
    assert result["summary"]["open_position_count"] == 0
    for day in result["daily_nav"]:
        market_value = sum(
            position["market_value_cny"]
            for position in result["positions"]
            if position["trade_date"] == day["trade_date"]
        )
        assert abs(day["nav_cny"] - day["cash_cny"] - market_value) < 1e-8


def test_suspended_fixed_horizon_exit_carries_position_to_next_tradable_open():
    _, base_dates = _research(periods=7)
    research, dates = _research(
        {("000001", base_dates[5]): "0"}, periods=7
    )
    result = simulate_fixed_horizon_portfolio(
        research,
        [{"decision_date": dates[0], "target_weights": {"000001": 0.5}}],
        1_000_000.0,
        {
            "commission_per_side_bps": 0.0,
            "transfer_fee_per_side_bps": 0.0,
            "stamp_duty_sell_bps": 0.0,
            "slippage_per_side_bps": 0.0,
        },
        liquidity_lookback_sessions=1,
        maximum_daily_participation=1.0,
    )

    blocked_exit = next(row for row in result["orders"] if row["order_type"] == "planned_exit_close")
    carried_exit = next(row for row in result["orders"] if row["order_type"] == "carried_exit_open")
    assert blocked_exit["status"] == "unfilled_suspended_or_missing"
    assert carried_exit["trade_date"] == dates[6].isoformat()
    assert carried_exit["status"] == "filled"
    assert result["summary"]["open_position_count"] == 0
    assert not any(
        row["trade_date"] == dates[6].isoformat()
        for row in result["positions"]
    )


def test_carried_position_counts_toward_next_allocation_before_buying():
    _, base_dates = _research(periods=11)
    research, dates = _research(
        {
            ("000001", base_dates[5]): "0",
            ("000001", base_dates[6]): "0",
        },
        periods=11,
    )
    result = simulate_fixed_horizon_portfolio(
        research,
        [
            {"decision_date": dates[0], "target_weights": {"000001": 0.5}},
            {"decision_date": dates[5], "target_weights": {"000001": 0.5}},
        ],
        1_000_000.0,
        {
            "commission_per_side_bps": 0.0,
            "transfer_fee_per_side_bps": 0.0,
            "stamp_duty_sell_bps": 0.0,
            "slippage_per_side_bps": 0.0,
        },
        liquidity_lookback_sessions=1,
        maximum_daily_participation=1.0,
    )

    entry_orders = [row for row in result["orders"] if row["order_type"] == "entry_open"]
    assert len(entry_orders) == 1
    assert result["summary"]["open_position_count"] == 0
    assert any(
        row["order_type"] == "carried_exit_open"
        and row["trade_date"] == dates[6].isoformat()
        and row["status"] == "unfilled_suspended_or_missing"
        for row in result["orders"]
    )


def test_forecast_settlement_respects_point_in_time_availability():
    research, dates = _research(periods=7)
    pending = {
        "decision_date": dates[0],
        "features_by_symbol": {"000001": (0.1, 0.2, 0.3), "000002": (0.3, 0.2, 0.1)},
        "candidate_symbols_sha256": "41d617fd60314dea21d8a606cc8e67e29dfe223c055e3c976eccb5f060a6fde1",
        "scores_by_method": {
            agent_id: {"000001": 1.0, "000002": 0.5} for agent_id in AGENT_IDS
        },
        "regime": "risk_on",
        "adaptive_weights": {agent_id: 1 / 3 for agent_id in AGENT_IDS},
        "regime_weights": {agent_id: 1 / 3 for agent_id in AGENT_IDS},
        "regime_weight_source": "global_fallback",
        "regime_matured_windows": 0,
    }

    immature = _evaluate_matured_window(research, pending, 0.5, dates[0])
    mature = _evaluate_matured_window(research, pending, 0.5, dates[5])

    assert not immature[0]
    assert {row["status"] for row in immature[3]} == {"not_yet_available"}
    assert len(mature[0]) == len(AGENT_IDS)
    assert {row["status"] for row in mature[3]} == {"settled"}


def _small_research_config():
    return {
        "decision_step_sessions": 5,
        "feature_trend_sessions": 20,
        "feature_reversal_sessions": 5,
        "feature_liquidity_return_sessions": 5,
        "feature_recent_amount_sessions": 5,
        "feature_reference_amount_sessions": 20,
        "evaluation_top_fraction": 0.5,
        "portfolio_top_fraction": 0.5,
        "invested_weight": 0.5,
        "initial_capital_cny": 1_000_000.0,
        "maximum_symbol_weight": 1.0,
        "maximum_one_way_turnover": 1.0,
        "reliability_lookback_windows": 1,
        "reliability_minimum_windows": 1,
        "reliability_equal_weight_share": 0.5,
        "regime_lookback_sessions": 3,
        "regime_minimum_windows": 1,
        "ridge_alpha": 1.0,
        "ridge_minimum_training_rows": 3,
        "risk_lookback_sessions": 5,
        "risk_minimum_observations": 3,
        "commission_per_side_bps": 0.0,
        "transfer_fee_per_side_bps": 0.0,
        "stamp_duty_sell_bps": 0.0,
        "slippage_per_side_bps": 0.0,
    }


def test_future_outcomes_cannot_change_already_written_prediction_or_target():
    config = _small_research_config()

    original = run_mvp_research(_walk_forward_research(), config)
    changed_future = run_mvp_research(
        _walk_forward_research(reverse_future=True), config
    )
    original_decision = next(
        row for row in original["prediction_decisions"]
        if row["prediction_date"] == "2024-01-31"
    )
    changed_decision = next(
        row for row in changed_future["prediction_decisions"]
        if row["prediction_date"] == "2024-01-31"
    )
    original_settlement = next(
        row for row in original["prediction_settlements"]
        if row["prediction_date"] == "2024-01-31" and row["symbol"] == "000001"
    )
    changed_settlement = next(
        row for row in changed_future["prediction_settlements"]
        if row["prediction_date"] == "2024-01-31" and row["symbol"] == "000001"
    )

    assert original_decision == changed_decision
    assert original_decision["candidate_symbols_sha256"] == changed_decision[
        "candidate_symbols_sha256"
    ]
    assert original_settlement["forward_return_5s"] != changed_settlement["forward_return_5s"]
    assert original_settlement["candidate_symbols_sha256"] == original_decision[
        "candidate_symbols_sha256"
    ]


def test_market_horizon_forecast_excludes_future_index_rows():
    research = _walk_forward_research()
    decision_index = 28
    expected = _market_horizon_forecast(research, decision_index, 3, 6)
    changed_future_index = [dict(row) for row in research.index_bars]
    for row in changed_future_index:
        if row["trade_date"] > research.trading_dates[decision_index]:
            row["close"] = float(row["close"]) * 3
    future_changed = ResearchData(
        list(research.trading_dates),
        list(research.adjusted_bars),
        list(research.execution_bars),
        changed_future_index,
        list(research.membership_snapshots),
    )

    assert _market_horizon_forecast(future_changed, decision_index, 3, 6) == expected


def test_common_comparison_start_resets_score_target_path():
    research = _walk_forward_research()
    dates = research.trading_dates
    config = _small_research_config()
    config["decision_step_sessions"] = 1
    config["prediction_horizon_sessions"] = 3
    common_start = dates[30]

    result = run_mvp_research(
        research,
        config,
        comparison_start_date=common_start,
    )

    assert result["protocol"]["comparison_start_date"] == common_start.isoformat()
    assert all(
        row["decision_date"] >= common_start
        for row in result["decisions_by_method"]["trend"]
    )


def test_fixed_risk_market_exposure_uses_only_prestart_risk_and_charges_costs():
    research, dates = _research(periods=7)
    index_bars = [dict(row) for row in research.index_bars]
    for row, return_value in zip(index_bars, (None, 1.0, -2.0, 2.0, -1.5, 3.0, -3.0), strict=True):
        row["return_1d"] = return_value
    research = ResearchData(
        list(research.trading_dates),
        list(research.adjusted_bars),
        list(research.execution_bars),
        index_bars,
        list(research.membership_snapshots),
    )
    costs = {
        "commission_per_side_bps": 3.0,
        "transfer_fee_per_side_bps": 0.1,
        "stamp_duty_sell_bps": 5.0,
        "slippage_per_side_bps": 5.0,
    }

    result = simulate_fixed_risk_market_exposure(
        research,
        dates[4],
        1_000_000.0,
        0.10,
        4,
        costs,
    )

    assert 0 < result["summary"]["target_market_weight"] < 1
    assert result["summary"]["total_cost_cny"] > 0
    assert result["orders"][0]["side"] == "buy"
    assert result["orders"][-1]["side"] == "sell"
    assert result["daily_nav"][-1]["market_value_cny"] == 0.0


def test_daily_decisions_with_three_session_labels_wait_for_maturity():
    config = _small_research_config()
    config["decision_step_sessions"] = 1
    config["prediction_horizon_sessions"] = 3
    result = run_mvp_research(_walk_forward_research(), config)

    decisions = result["decision_log"]
    assert result["protocol"]["prediction_horizon_sessions"] == 3
    assert result["protocol"]["decision_step_sessions"] == 1
    assert [row["matured_rank_ic_windows"] for row in decisions[:4]] == [0, 0, 0, 1]
    assert result["prediction_decisions"][0]["planned_exit_date"] == decisions[3]["decision_date"]
    assert {row["status"] for row in result["prediction_settlements"]} == {"pending", "settled"}
    latest_decision = result["prediction_decisions"][-1]
    assert latest_decision["prediction_date"] == "2024-02-06"
    assert latest_decision["status"] == "pending"
    assert latest_decision["planned_exit_date"] is None
    assert all("forward_return_3s" in row for row in result["prediction_settlements"])
    first_prediction = result["prediction_decisions"][0]
    assert "market_only" in first_prediction["expected_returns_by_method"]
    assert "market_plus_relative_fusion" in first_prediction["expected_returns_by_method"]
    assert len(set(first_prediction["expected_returns_by_method"]["market_only"].values())) == 1
    training_seed = result["training_seed"]
    assert len(training_seed["relative_calibration_features"]) == len(
        training_seed["relative_calibration_returns"]
    )
    assert len(training_seed["relative_calibration_sufficient_statistics"]) == len(AGENT_IDS)


def test_state_conditioned_fusion_uses_matured_labels_and_resumes_identically():
    research = _state_conditioned_research()
    dates = research.trading_dates
    config = _small_research_config()
    config.update({
        "decision_step_sessions": 1,
        "prediction_horizon_sessions": 3,
        "state_fusion_minimum_training_rows": 6,
        "state_fusion_minimum_rows_per_state": 2,
    })
    full = run_mvp_research(research, config)
    flipped_future = run_mvp_research(
        _state_conditioned_research(reverse_future=True), config
    )
    full_by_date = {row["prediction_date"]: row for row in full["prediction_decisions"]}
    flipped_by_date = {row["prediction_date"]: row for row in flipped_future["prediction_decisions"]}

    pre_future_date = dates[29].isoformat()
    assert full_by_date[pre_future_date]["expected_returns_by_method"] == flipped_by_date[
        pre_future_date
    ]["expected_returns_by_method"]
    mature_predictions = [
        row for row in full["prediction_decisions"]
        if row["state_conditioned_fusion"]["status"] == "matured_state_conditioned_ridge"
    ]
    assert mature_predictions
    for row in full["prediction_decisions"]:
        diagnostics = row["state_conditioned_fusion"]
        assert diagnostics["training_label_cutoff"] is None or diagnostics[
            "training_label_cutoff"
        ] <= row["prediction_date"]
        if diagnostics["status"] == "matured_state_conditioned_ridge":
            assert diagnostics["training_rows"] >= diagnostics["minimum_training_rows"]
            assert all(
                diagnostics["training_rows_by_state"][state] >= diagnostics["minimum_rows_per_state"]
                for state in ("risk_on", "risk_off")
            )
            assert set(row["expected_returns_by_method"]["state_conditioned_fusion"]) == set(
                row["scored_symbols"]
            )
        else:
            assert row["expected_returns_by_method"]["state_conditioned_fusion"] == {}

    training_rows = full["training_seed"]["state_fusion_training_rows"]
    settled_rows = [
        row for row in full["prediction_settlements"] if row["status"] == "settled"
    ]
    assert len(training_rows) == len(settled_rows)

    cutoff = dates[30]
    cutoff2 = dates[32]
    truncated1 = _research_through(research, cutoff)
    partial = run_mvp_research(
        truncated1,
        config,
        as_of_date=cutoff,
        ensure_as_of_decision=True,
    )
    future_seed = {
        **partial["training_seed"],
        "state_as_of": dates[31].isoformat(),
    }
    with pytest.raises(ValueError, match="cannot be later than the requested as-of date"):
        run_mvp_research(
            research,
            config,
            initial_state=future_seed,
            as_of_date=cutoff,
        )
    with pytest.raises(ValueError, match="must be earlier than the requested decision range"):
        run_mvp_research(
            truncated1,
            config,
            initial_state=partial["training_seed"],
            as_of_date=cutoff,
            ensure_as_of_decision=True,
        )
    with pytest.raises(ValueError, match="must be earlier than the requested decision range"):
        run_mvp_research(
            truncated1,
            config,
            initial_state=partial["training_seed"],
        )

    partial2 = run_mvp_research(
        _research_through(research, cutoff2),
        config,
        initial_state=partial["training_seed"],
        as_of_date=cutoff2,
        ensure_as_of_decision=True,
        decision_start_date=dates[31],
    )
    resumed = run_mvp_research(
        research,
        config,
        initial_state=partial2["training_seed"],
        as_of_date=dates[-1],
        decision_start_date=dates[33],
    )
    segments = (partial, partial2, resumed)

    combined_predictions = {
        row["prediction_date"]: row
        for segment in segments
        for row in segment["prediction_decisions"]
    }
    assert set(combined_predictions) == set(full_by_date)
    forecast_fields = set(full_by_date[dates[31].isoformat()]) - {
        "entry_date", "status", "planned_exit_date",
    }
    for prediction_date, full_row in full_by_date.items():
        assert {
            field: combined_predictions[prediction_date][field]
            for field in forecast_fields
        } == {field: full_row[field] for field in forecast_fields}

    combined_settlements = {}
    for segment in segments:
        combined_settlements.update({
            (row["prediction_date"], row["symbol"]): row
            for row in segment["prediction_settlements"]
        })
    full_settlements = {
        (row["prediction_date"], row["symbol"]): row
        for row in full["prediction_settlements"]
    }
    assert set(combined_settlements) == set(full_settlements)
    for key, full_row in full_settlements.items():
        assert combined_settlements[key] == full_row

    for method, full_rows in full["decisions_by_method"].items():
        resumed_rows = [
            row for segment in segments for row in segment["decisions_by_method"].get(method, [])
        ]
        assert resumed_rows == full_rows
    assert resumed["training_seed"] == full["training_seed"]


@pytest.mark.parametrize(
    ("features", "state", "message"),
    [
        ((0.0, 0.0), "risk_on", "三个专家分位数"),
        ((0.0, 0.0, 0.0), "unknown", "未知的成熟市场状态"),
        ((0.0, 0.0, 0.6), "risk_off", r"有限的 \[-0.5, 0.5\]"),
    ],
)
def test_state_conditioned_features_reject_invalid_inputs(features, state, message):
    with pytest.raises(ValueError, match=message):
        _state_conditioned_features(features, state)


def test_daily_decision_api_returns_latest_pending_record():
    research = _walk_forward_research()
    config = _small_research_config()
    config["decision_step_sessions"] = 1
    config["prediction_horizon_sessions"] = 3
    as_of = research.trading_dates[-1]

    decision = create_daily_decision(
        research,
        config,
        as_of,
        PortfolioState(1_000_000.0, {}),
    )
    payload = decision.to_dict()

    assert payload["schema_version"] == "adaptive-mas.decision.v1"
    assert payload["as_of_date"] == as_of.isoformat()
    assert payload["execution_date"] is None
    assert payload["prediction_status"] == "pending"
    assert payload["prediction_horizon_sessions"] == 3
    assert payload["current_cash_cny"] == 1_000_000.0
    assert payload["current_nav_cny"] == 1_000_000.0


def test_daily_decision_uses_receiver_costs_for_reused_research():
    research = _walk_forward_research()
    config = _small_research_config()
    config["decision_step_sessions"] = 1
    config["prediction_horizon_sessions"] = 3
    as_of = research.trading_dates[-1]
    source_usage = {
        "provider_call_count": 4,
        "attempt_count": 4,
        "response_cache_reuse_count": 0,
        "model_turn_count": 4,
        "prompt_tokens": 400,
        "completion_tokens": 80,
        "logical_estimated_cost_cny": 0.04,
        "api_incremental_cost_cny": 0.04,
    }
    receiving_usage = {
        **source_usage,
        "provider_call_count": 0,
        "response_cache_reuse_count": 4,
        "api_incremental_cost_cny": 0.0,
        "source_task_api_incremental_cost_cny": 0.04,
        "research_result_reused_from_artifact": True,
    }
    harness_context = {
        "research": {
            "status": "completed",
            "task_usage": source_usage,
            "receiving_task_usage": receiving_usage,
            "agents": {
                "research_manager": {"report": {"priority_symbols": []}},
            },
        },
        "research_cost": {"cost_source": "previously paid source artifact"},
    }

    decision = create_daily_decision(
        research,
        config,
        as_of,
        PortfolioState(1_000_000.0, {}),
        harness_context=harness_context,
    ).to_dict()

    accounting = decision["evidence"]["harness"]["research_cost_accounting"]
    assert accounting["logical_research_cost_cny"] == 0.04
    assert accounting["actual_incremental_api_cost_cny"] == 0.0
    assert accounting["source_task_api_cost_cny"] == 0.04
    assert accounting["research_result_reused"] is True
    decision_accounting = decision["evidence"]["harness"]["decision_cost_accounting"]
    assert decision_accounting["actual_incremental_api_cost_cny"] == 0.0
    assert decision_accounting["estimated_total_decision_cost_cny"] == (
        decision["expected_transaction_cost_cny"] + 0.04
    )


def test_daily_decision_sell_orders_include_capacity_and_execution_uncertainty():
    research = _walk_forward_research()
    config = _small_research_config()
    config["decision_step_sessions"] = 1
    config["prediction_horizon_sessions"] = 3
    decision = create_daily_decision(
        research,
        config,
        research.trading_dates[-1],
        PortfolioState(950_000.0, {"000001": 50_000.0}),
    )
    payload = decision.to_dict()

    assert payload["status"] == "pending_execution"
    assert payload["sell_notional_cny"] > 0
    assert payload["buy_notional_cny"] >= 0
    assert payload["estimated_cash_after_costs_cny"] <= payload["target_cash_cny"]
    assert payload["risk_adjusted_net_incremental_value_cny"] == (
        payload["optimized_objective_value_cny"]
        - payload["current_holding_objective_value_cny"]
    )
    assert set(payload["evidence"]["expert_matured_performance"]) == set(AGENT_IDS)
    assert all(
        order["source_prediction_id"] in {
            prediction["prediction_id"] for prediction in payload["predictions_by_symbol"].values()
        }
        and order["prediction_date"] == payload["as_of_date"]
        and order["prediction_horizon_sessions"] == payload["prediction_horizon_sessions"]
        for order in payload["orders"]
    )
    assert all(
        prediction["available_at"] == f"{payload['as_of_date']}T15:00:00+08:00"
        for prediction in payload["predictions_by_symbol"].values()
    )
    assert all(
        order["status"] == "pending_next_session"
        and order["next_session_execution_uncertain"]
        and order["estimated_liquidity_capacity_cny"] > 0
        and order["decision_date_tradestatus"] in {"0", "1", "unknown"}
        for order in payload["orders"]
    )


def test_daily_decision_schema_lists_account_and_order_contracts():
    schema_path = Path(__file__).parents[1] / "schemas" / "decision_api.v1.schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    daily = schema["$defs"]["dailyDecision"]
    order = schema["$defs"]["decisionOrder"]

    assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    assert "risk_adjusted_net_incremental_value_cny" in daily["required"]
    assert "estimated_cash_after_costs_cny" in daily["required"]
    assert "next_session_execution_uncertain" in order["required"]
    assert "estimated_liquidity_capacity_cny" in order["required"]
    assert "source_prediction_id" in order["required"]
    assert "prediction_id" in daily["properties"]["predictions_by_symbol"]["additionalProperties"]["required"]


def test_matured_calibration_sufficient_statistics_match_batch_fit():
    features = [
        (0.1, 0.3, 0.7),
        (0.2, 0.5, 0.4),
        (0.5, 0.2, 0.1),
        (0.8, 0.6, 0.9),
    ]
    returns = [0.01, -0.02, 0.03, 0.015]
    moments = _extend_calibration_moments(
        [[0.0] * 6 for _ in AGENT_IDS], features, returns
    )

    batch_fit = _fit_return_calibrators(features, returns)
    cached_fit = _fit_return_calibrators(features, returns, moments)

    assert batch_fit == cached_fit


def test_return_calibration_buckets_are_equal_count_within_one_date():
    symbols = [f"{index:06d}" for index in range(8)]
    predictions = {symbol: float(index) for index, symbol in enumerate(symbols)}
    outcomes = {symbol: float(7 - index) / 100 for index, symbol in enumerate(symbols)}

    buckets = _return_calibration_buckets(symbols, predictions, outcomes, bucket_count=5)

    assert [row["bucket"] for row in buckets] == [1, 2, 3, 4, 5]
    assert [row["symbol_count"] for row in buckets] == [2, 2, 1, 2, 1]
    assert [row["mean_predicted_return"] for row in buckets] == sorted(
        row["mean_predicted_return"] for row in buckets
    )


def test_equal_fusion_minus_leave_one_out_interval_has_declared_sign():
    day = "2026-01-05"
    symbols = ["000001", "000002", "000003"]
    candidate_hash = hashlib.sha256("\n".join(symbols).encode("ascii")).hexdigest()
    agent_predictions = {
        "trend": {"000001": 0.0, "000002": 0.15, "000003": -0.15},
        "reversal": {"000001": 0.0, "000002": 0.0, "000003": 0.1},
        "liquidity_confirmation": {"000001": 0.0, "000002": 0.0, "000003": 0.1},
    }
    equal = {
        symbol: sum(agent_predictions[agent][symbol] for agent in AGENT_IDS) / 3
        for symbol in symbols
    }
    result = {
        "prediction_decisions": [
            {
                "prediction_date": day,
                "input_cutoff": day,
                "horizon_sessions": 3,
                "scored_symbols": symbols,
                "candidate_symbols_sha256": candidate_hash,
                "scores_by_method": {
                    agent: {symbol: float(index) for index, symbol in enumerate(symbols)}
                    for agent in AGENT_IDS
                },
                "expected_returns_by_method": {
                    **agent_predictions,
                    "equal_fusion": equal,
                },
                "forecast_errors_by_method": {},
                "target_weights_by_method": {},
            }
        ],
        "prediction_settlements": [
            {
                "prediction_date": day,
                "symbol": symbol,
                "status": "settled",
                "candidate_symbols_sha256": candidate_hash,
                "forward_return_3s": realized,
            }
            for symbol, realized in zip(symbols, (0.01, 0.05, -0.02), strict=True)
        ],
    }

    diagnostics = expert_prediction_diagnostics(result)
    interval = diagnostics["leave_one_expert_out_ablation"][
        "without_trend"
    ]["equal_fusion_minus_ablation_top_decile_gross_return"]

    assert interval["mean_difference"] > 0
    assert interval["mean_difference_95pct_low"] > 0
    assert paired_block_mean_difference([0.5], [0.2])["mean_difference"] > 0


def test_state_fusion_diagnostics_compare_only_dates_with_a_mature_forecast():
    symbols = ["000001", "000002", "000003"]
    candidate_hash = hashlib.sha256("\n".join(symbols).encode("ascii")).hexdigest()
    agent_predictions = {
        agent: {symbol: index / 100 for index, symbol in enumerate(symbols)}
        for agent in AGENT_IDS
    }
    equal = {
        symbol: sum(agent_predictions[agent][symbol] for agent in AGENT_IDS) / len(AGENT_IDS)
        for symbol in symbols
    }
    dates = ("2026-01-05", "2026-01-06")
    predictions = []
    settlements = []
    for index, day in enumerate(dates):
        baseline = {symbol: index * 0.01 + value for symbol, value in equal.items()}
        state_method = (
            {} if index == 0 else {symbol: value * 1.2 for symbol, value in baseline.items()}
        )
        predictions.append({
            "prediction_date": day,
            "input_cutoff": day,
            "horizon_sessions": 3,
            "scored_symbols": symbols,
            "candidate_symbols_sha256": candidate_hash,
            "scores_by_method": {
                **{agent: {symbol: float(rank) for rank, symbol in enumerate(symbols)} for agent in AGENT_IDS},
                "equal_fusion": equal,
            },
            "expected_returns_by_method": {
                **agent_predictions,
                "equal_fusion": equal,
                "loss_gated_fusion": baseline,
                "state_conditioned_fusion": state_method,
            },
            "forecast_errors_by_method": {},
            "target_weights_by_method": {},
            "expert_pairwise_symbol_overlap": {
                "trend:reversal": {
                    "shared_symbol_count": 2,
                    "union_symbol_count": 4,
                    "jaccard": 0.5,
                },
            },
        })
        settlements.extend({
            "prediction_date": day,
            "symbol": symbol,
            "status": "settled",
            "candidate_symbols_sha256": candidate_hash,
            "forward_return_3s": realized,
        } for symbol, realized in zip(symbols, (0.01, 0.05, -0.02), strict=True))

    diagnostics = expert_prediction_diagnostics({
        "prediction_decisions": predictions,
        "prediction_settlements": settlements,
    })

    assert diagnostics["date_level_forecast_metrics"]["state_conditioned_fusion"]["matured_dates"] == 1
    assert diagnostics["expert_pairwise_signal_universe_overlap"]["trend:reversal"]["mean"] == 0.5
    paired = diagnostics["paired_date_level_forecast_comparisons_vs_loss_gated_fusion"][
        "state_conditioned_fusion"
    ]
    assert paired["common_dates"] == 1
    assert paired["by_metric_candidate_minus_baseline"]["mae"]["paired_dates"] == 1


def test_optimizer_replay_continues_actual_holdings_after_blocked_sell():
    base = _walk_forward_research()
    dates = base.trading_dates
    blocked_date = dates[26]
    execution = [dict(row) for row in base.execution_bars]
    for row in execution:
        if row["symbol"] == "000001" and row["trade_date"] == blocked_date:
            row["tradestatus"] = "0"
    research = ResearchData(
        list(base.trading_dates),
        list(base.adjusted_bars),
        execution,
        list(base.index_bars),
        list(base.membership_snapshots),
    )
    config = _small_research_config()
    config["maximum_symbol_weight"] = 0.8
    config["maximum_one_way_turnover"] = 0.5
    no_costs = {
        "commission_per_side_bps": 0.0,
        "transfer_fee_per_side_bps": 0.0,
        "stamp_duty_sell_bps": 0.0,
        "slippage_per_side_bps": 0.0,
    }

    def forecast(day, expected_return):
        return {
            "prediction_date": day.isoformat(),
            "horizon_sessions": 3,
            "expected_returns_by_method": {
                "loss_gated_fusion": {"000001": expected_return}
            },
            "forecast_errors_by_method": {"loss_gated_fusion": 0.0},
        }

    forecasts = [
        forecast(dates[24], 0.2),
        forecast(dates[25], -0.2),
        forecast(dates[26], -0.2),
    ]
    initial_state = PortfolioState(1_000_000.0, {})
    app_target, _ = optimize_forecast_target(
        research, forecasts[0], initial_state, config
    )
    replay = simulate_portfolio(
        research,
        [],
        1_000_000.0,
        no_costs,
        liquidity_lookback_sessions=5,
        maximum_daily_participation=1.0,
        maximum_one_way_turnover=0.5,
        forecast_decisions=forecasts,
        optimizer_config=config,
    )

    assert replay["decisions"][0]["target_weights"] == app_target.target_weights
    blocked_sell = next(
        row for row in replay["orders"]
        if row["trade_date"] == blocked_date.isoformat() and row["side"] == "sell"
    )
    assert blocked_sell["status"] == "unfilled_suspended_or_missing"
    decision_after_block = next(
        row for row in replay["decisions"]
        if row["decision_date"] == blocked_date.isoformat()
    )
    assert decision_after_block["current_holdings_cny"]["000001"] > 0
    assert decision_after_block["current_cash_cny"] < decision_after_block["current_nav_cny"]
    for day in replay["daily_nav"]:
        market_value = sum(
            row["market_value_cny"]
            for row in replay["positions"]
            if row["trade_date"] == day["trade_date"]
        )
        assert abs(day["cash_cny"] + market_value - day["nav_cny"]) < 1e-6


def test_realized_turnover_cap_applies_to_filled_amounts():
    research = _walk_forward_research()
    dates = research.trading_dates
    config = _small_research_config()
    config["maximum_symbol_weight"] = 0.8
    config["maximum_one_way_turnover"] = 0.1
    no_costs = {
        "commission_per_side_bps": 0.0,
        "transfer_fee_per_side_bps": 0.0,
        "stamp_duty_sell_bps": 0.0,
        "slippage_per_side_bps": 0.0,
    }

    replay = simulate_portfolio(
        research,
        [{"decision_date": dates[24], "target_weights": {"000001": 0.8}}],
        1_000_000.0,
        no_costs,
        liquidity_lookback_sessions=5,
        maximum_daily_participation=1.0,
        maximum_one_way_turnover=0.1,
    )

    first_day = replay["daily_nav"][0]
    assert first_day["traded_notional_cny"] <= 200_000.0 + 1e-6
    assert first_day["one_way_turnover"] <= 0.1 + 1e-9
    assert replay["summary"]["turnover_limited_orders"] == 1
    assert replay["orders"][0]["status"] == "partial_turnover_limited"


def test_market_state_and_regime_gate_use_matured_state_records():
    research, dates = _research()
    assert market_state(research, dates[2], lookback_sessions=2) == "risk_on"
    global_rows = [
        {"trend": 0.10, "reversal": 0.01, "liquidity_confirmation": -0.01},
        {"trend": 0.08, "reversal": 0.02, "liquidity_confirmation": 0.00},
    ]
    weights, source, count = conditional_weights(
        {"risk_on": global_rows},
        "risk_on",
        global_rows,
        ("trend", "reversal", "liquidity_confirmation"),
        lookback_windows=2,
        minimum_windows=2,
        equal_weight_share=0.5,
    )

    assert source == "state_conditional"
    assert count == 2
    assert abs(sum(weights.values()) - 1.0) < 1e-9


def test_research_source_does_not_reference_local_scoring_file():
    project_root = Path(__file__).parents[1]
    production_sources = (project_root / "src").rglob("*.py")

    assert all("test.csv" not in path.read_text(encoding="utf-8") for path in production_sources)


def test_paired_block_bootstrap_is_deterministic_for_identical_series():
    returns = [0.001, -0.002, 0.003, 0.0] * 12

    first = paired_block_bootstrap(returns, returns, block_sessions=4, repetitions=100, seed=7)
    second = paired_block_bootstrap(returns, returns, block_sessions=4, repetitions=100, seed=7)

    assert first == second
    assert first["annualized_return_difference_95pct_low"] == 0.0
    assert first["annualized_return_difference_95pct_high"] == 0.0
