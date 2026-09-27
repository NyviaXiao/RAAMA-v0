import pytest

from adaptive_mas.evaluation.decision_attribution import (
    attribute_optimizer_decision,
    summarize_next_session_execution,
)


def _inputs(*, expected_return=0.04, forecast_error=0.01, risk_aversion=0.0, volatility=0.2, capacity=0.05):
    forecast = {
        "prediction_date": "2026-01-05",
        "execution_date": "2026-01-06",
        "horizon_sessions": 3,
        "expected_returns_by_method": {
            "loss_gated_fusion": {"000001": expected_return},
        },
        "forecast_errors_by_method": {"loss_gated_fusion": forecast_error},
        "annualized_volatility_by_symbol": {"000001": volatility},
        "factor_beta_by_symbol": {"000001": 0.0},
        "market_annualized_volatility": 0.2,
    }
    decision = {
        "current_nav_cny": 1_000_000.0,
        "current_cash_cny": 1_000_000.0,
        "current_holdings_cny": {},
        "target_weights": {"000001": 0.05},
        "execution_date": "2026-01-06",
        "buy_notional_cny": 50_000.0,
        "sell_notional_cny": 0.0,
        "gross_expected_return": 0.002,
        "forecast_error_allowance": 0.00025,
        "expected_transaction_cost_weight": 0.0000405,
        "risk_penalty_weight": 0.0,
        "incremental_objective_value": 0.0017095,
        "risk_adjusted_net_incremental_value_cny": 1709.5,
        "estimated_cash_after_costs_cny": 949_959.5,
        "one_way_turnover": 0.025,
    }
    config = {
        "forecast_error_multiplier": 0.5,
        "risk_aversion": risk_aversion,
        "risk_budget_annualized": 0.20,
        "maximum_symbol_weight": 0.05,
        "maximum_one_way_turnover": 0.50,
        "commission_per_side_bps": 3.0,
        "transfer_fee_per_side_bps": 0.1,
        "slippage_per_side_bps": 5.0,
        "stamp_duty_sell_bps": 5.0,
    }
    return forecast, decision, config, {"000001": capacity}


def test_hand_calculation_of_gross_error_fee_and_net_incremental_value():
    row = attribute_optimizer_decision(*_inputs())
    candidate = row["best_single_name_screen"]

    assert candidate["incremental_weight"] == pytest.approx(0.05)
    assert candidate["gross_return_contribution"] == pytest.approx(0.002)
    assert candidate["forecast_error_allowance"] == pytest.approx(0.00025)
    assert candidate["transaction_cost"] == pytest.approx(0.0000405)
    assert candidate["risk_penalty_increment"] == pytest.approx(0.0)
    assert candidate["net_incremental_return"] == pytest.approx(0.0017095)
    assert candidate["net_incremental_return"] * row["account_nav_cny"] == pytest.approx(1709.5)
    assert row["optimizer_incremental_value_cny"] == pytest.approx(1709.5)


def test_risk_penalty_uses_the_best_smaller_weight_not_only_the_maximum():
    forecast, decision, config, capacity = _inputs(
        expected_return=0.02,
        forecast_error=0.0,
        risk_aversion=80.0,
        volatility=0.8,
        capacity=0.05,
    )
    row = attribute_optimizer_decision(forecast, decision, config, capacity)
    candidate = row["best_single_name_screen"]

    cost_rate = (3 + 0.1 + 5) / 10_000
    expected_optimum = (0.02 - cost_rate) / (2 * (0.5 * 80 * 3 / 250) * 0.8**2)
    assert expected_optimum < 0.05
    assert candidate["incremental_weight"] == pytest.approx(expected_optimum)
    assert candidate["net_incremental_return"] > 0


def test_cash_capacity_accounts_for_buy_fees_and_error_gate_is_visible():
    forecast, decision, config, capacity = _inputs(
        expected_return=0.003,
        forecast_error=0.008,
        capacity=0.05,
    )
    decision["current_cash_cny"] = 20_000.0
    decision["current_holdings_cny"] = {
        f"{symbol:06d}": 49_000.0 for symbol in range(10, 30)
    }
    forecast["annualized_volatility_by_symbol"].update(
        {symbol: 0.0 for symbol in decision["current_holdings_cny"]}
    )
    forecast["factor_beta_by_symbol"].update(
        {symbol: 0.0 for symbol in decision["current_holdings_cny"]}
    )
    decision["buy_notional_cny"] = 0.0
    decision["target_weights"] = {}
    decision["gross_expected_return"] = 0.0
    decision["forecast_error_allowance"] = 0.0
    decision["expected_transaction_cost_weight"] = 0.0
    decision["incremental_objective_value"] = 0.0
    decision["risk_adjusted_net_incremental_value_cny"] = 0.0
    decision["estimated_cash_after_costs_cny"] = 1_000_000.0
    decision["one_way_turnover"] = 0.0

    row = attribute_optimizer_decision(forecast, decision, config, capacity)
    candidate = row["best_single_name_screen"]
    buy_cost_rate = (3 + 0.1 + 5) / 10_000

    assert candidate["unconstrained_room_weight"] == pytest.approx(0.02 / (1 + buy_cost_rate))
    assert row["first_blocking_reason_diagnostic"] == "forecast_error_allowance"
    assert candidate["net_if_error_removed"] > 0
    assert candidate["net_incremental_return"] <= 0
    assert candidate["delta_if_error_removed"] > 0
    assert candidate["error_removed_probe_full_rule_net"] < 0
    assert candidate["error_removed_probe_net_after_removing_error"] > 0


def test_execution_summary_uses_real_orders_schema_and_preserves_suspension_status():
    rows = [
        {
            "decision_date": "2026-01-05",
            "trade_date": "2026-01-06",
            "requested_notional_cny": "30000",
            "filled_notional_cny": "20000",
            "liquidity_limit_cny": "25000",
            "status": "unfilled_suspended_or_missing",
            "cost_cny": "16.2",
        }
    ]
    result = summarize_next_session_execution(
        "2026-01-05",
        "2026-01-06",
        rows,
        {"nav_cny": "1000000", "cash_cny": "980000"},
        [],
    )

    assert result["requested_notional_cny"] == pytest.approx(30_000)
    assert result["filled_notional_cny"] == pytest.approx(20_000)
    assert result["unfilled_notional_cny"] == pytest.approx(10_000)
    assert result["transaction_cost_cny"] == pytest.approx(16.2)
    assert result["unfilled_order_count"] == 1
    assert result["status_counts"] == {"unfilled_suspended_or_missing": 1}
    assert result["capacity_bound_order_count"] == 0
    assert result["reason_detail_available_in_order_schema"] is False
