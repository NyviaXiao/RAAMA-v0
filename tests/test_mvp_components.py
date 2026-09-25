from datetime import date, timedelta
from pathlib import Path

from adaptive_mas.agents.quantitative import AGENT_IDS
from adaptive_mas.backtest.simulator import (
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
from adaptive_mas.evaluation.mvp_research import _evaluate_matured_window
from adaptive_mas.evaluation.mvp_research import run_mvp_research


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
            "close": 1000.0 * (1 + index * 0.0005),
        }
        for index, trade_date in enumerate(dates)
    ]
    membership = [
        MembershipSnapshot(dates[0], dates[0], frozenset(symbols), "synthetic")
    ]
    return ResearchData(dates, adjusted, execution, index_bars, membership)


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


def test_future_outcomes_cannot_change_already_written_prediction_or_target():
    config = {
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
    assert original_settlement["forward_return_5s"] != changed_settlement["forward_return_5s"]


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
