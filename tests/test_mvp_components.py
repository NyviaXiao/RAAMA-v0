from datetime import date, timedelta
from pathlib import Path

from adaptive_mas.backtest.simulator import simulate_portfolio
from adaptive_mas.data.research import MembershipSnapshot, ResearchData
from adaptive_mas.models.ridge import ridge_predictions
from adaptive_mas.portfolio.allocation import (
    apply_weight_and_turnover_limits,
    inverse_volatility_weights,
)
from adaptive_mas.regime.market_state import conditional_weights, market_state
from adaptive_mas.evaluation.uncertainty import paired_block_bootstrap


def _research(status_by_date=None):
    dates = [date(2024, 1, 2) + timedelta(days=index) for index in range(5)]
    adjusted = []
    execution = []
    for symbol, daily_prices in {
        "000001": [(10, 10), (10, 11), (11, 11), (11, 11), (11, 11)],
        "000002": [(20, 20), (20, 20), (20, 21), (21, 21), (21, 21)],
    }.items():
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
