import pytest

from adaptive_mas.portfolio.optimizer import (
    PortfolioInfeasibleError,
    optimize_portfolio,
)


def _optimize(
    expected_returns,
    current_weights,
    *,
    volatilities=None,
    risk_budget=0.5,
    turnover=0.5,
    maximum_symbol_weight=0.8,
    buy_cost=0.001,
    sell_cost=0.001,
    max_changes=None,
):
    symbols = set(expected_returns) | set(current_weights)
    return optimize_portfolio(
        expected_returns,
        {symbol: 0.0 for symbol in symbols},
        volatilities or {symbol: 0.2 for symbol in symbols},
        current_weights,
        horizon_sessions=10,
        maximum_symbol_weight=maximum_symbol_weight,
        maximum_one_way_turnover=turnover,
        risk_budget_annualized=risk_budget,
        risk_aversion=0.0,
        forecast_error_multiplier=0.0,
        buy_cost_rate=buy_cost,
        sell_cost_rate=sell_cost,
        max_change_weights=max_changes,
        iterations=600,
    )


def test_high_holdings_value_does_not_consume_trade_turnover_budget():
    result = _optimize(
        {"000001": 0.0, "000002": 0.0},
        {"000001": 0.4, "000002": 0.4},
        turnover=0.1,
    )

    assert result.target_weights == {"000001": 0.4, "000002": 0.4}
    assert result.maximum_one_way_turnover == 0.0
    assert result.incremental_objective_value == pytest.approx(0.0)


def test_optimizer_only_buys_when_expected_gain_covers_cost():
    profitable = _optimize(
        {"000001": 0.03},
        {"000001": 0.2},
        buy_cost=0.001,
        sell_cost=0.01,
    )
    expensive = _optimize(
        {"000001": -0.03},
        {"000001": 0.5},
        buy_cost=0.001,
        sell_cost=0.04,
    )

    assert profitable.target_weights["000001"] > 0.2
    assert expensive.target_weights["000001"] == 0.5
    assert expensive.maximum_one_way_turnover == 0.0
    assert profitable.incremental_objective_value == pytest.approx(
        profitable.objective_value - profitable.holding_objective_value
    )


def test_cash_constraint_includes_buy_and_sell_fees():
    result = _optimize(
        {"000001": 0.10, "000002": 0.10},
        {},
        maximum_symbol_weight=0.8,
        turnover=0.5,
        buy_cost=0.02,
        sell_cost=0.03,
    )

    assert result.expected_transaction_cost > 0
    assert sum(result.target_weights.values()) + result.expected_transaction_cost <= 1 + 1e-9


def test_historical_market_risk_is_independent_of_the_risk_budget():
    args = {
        "expected_returns": {"000001": 0.0},
        "forecast_errors": {"000001": 0.0},
        "annualized_volatilities": {"000001": 0.30},
        "current_weights": {"000001": 0.5},
        "horizon_sessions": 10,
        "maximum_symbol_weight": 0.8,
        "maximum_one_way_turnover": 0.5,
        "risk_aversion": 0.0,
        "forecast_error_multiplier": 0.0,
        "buy_cost_rate": 0.001,
        "sell_cost_rate": 0.001,
    }
    relaxed = optimize_portfolio(risk_budget_annualized=0.5, **args)
    tighter = optimize_portfolio(risk_budget_annualized=0.4, **args)

    assert relaxed.annualized_risk == pytest.approx(0.15)
    assert tighter.annualized_risk == pytest.approx(relaxed.annualized_risk)


def test_partial_de_risk_trade_can_meet_risk_budget_exactly():
    result = _optimize(
        {"000001": 0.0},
        {"000001": 0.5},
        volatilities={"000001": 0.4},
        risk_budget=0.16,
        turnover=0.1,
        buy_cost=0.001,
        sell_cost=0.001,
    )

    assert result.target_weights["000001"] == pytest.approx(0.4, abs=1e-6)
    assert result.annualized_risk == pytest.approx(0.16, abs=1e-6)
    assert result.maximum_one_way_turnover == pytest.approx(0.05, abs=1e-6)


def test_infeasible_risk_budget_and_turnover_pair_is_reported():
    with pytest.raises(PortfolioInfeasibleError, match="换手"):
        _optimize(
            {"000001": 0.0},
            {"000001": 0.5},
            volatilities={"000001": 0.8},
            risk_budget=0.3,
            turnover=0.05,
        )


def test_overweight_position_is_trimmed_when_cap_trade_is_feasible():
    result = _optimize(
        {"000001": 0.2},
        {"000001": 0.6},
        maximum_symbol_weight=0.4,
        turnover=0.1,
    )

    assert result.target_weights["000001"] == pytest.approx(0.4, abs=1e-6)
    assert result.maximum_one_way_turnover == pytest.approx(0.1, abs=1e-6)


def test_smallest_risk_repair_is_used_instead_of_costing_a_full_liquidation():
    result = _optimize(
        {"000001": 0.0},
        {"000001": 0.8},
        volatilities={"000001": 0.2},
        risk_budget=0.15,
        turnover=0.1,
        maximum_symbol_weight=0.9,
    )

    assert result.target_weights["000001"] == pytest.approx(0.75, abs=1e-6)
    assert result.annualized_risk == pytest.approx(0.15, abs=1e-6)
    assert result.maximum_one_way_turnover == pytest.approx(0.025, abs=1e-6)
