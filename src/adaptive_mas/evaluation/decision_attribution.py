"""Explain daily optimizer decisions without changing the frozen policy."""

from __future__ import annotations

import math
from typing import Mapping


def _risk_variance(
    weights: Mapping[str, float],
    volatilities: Mapping[str, float],
    betas: Mapping[str, float],
    market_volatility: float,
) -> float:
    factor_exposure = math.fsum(
        weight * float(betas.get(symbol, 0.0))
        for symbol, weight in weights.items()
    )
    idiosyncratic = math.fsum(
        weight**2
        * max(
            float(volatilities[symbol]) ** 2
            - float(betas.get(symbol, 0.0)) ** 2 * market_volatility**2,
            0.0,
        )
        for symbol, weight in weights.items()
        if symbol in volatilities
    )
    return idiosyncratic + (factor_exposure * market_volatility) ** 2


def _candidate_economics(
    symbol: str,
    delta_weight: float,
    *,
    expected_return: float,
    forecast_error: float,
    current_weights: Mapping[str, float],
    volatilities: Mapping[str, float],
    betas: Mapping[str, float],
    market_volatility: float,
    horizon_sessions: int,
    risk_aversion: float,
    error_multiplier: float,
    buy_cost_rate: float,
) -> dict[str, float]:
    nav = 1.0
    before_variance = _risk_variance(
        current_weights, volatilities, betas, market_volatility
    )
    after_weights = dict(current_weights)
    after_weights[symbol] = after_weights.get(symbol, 0.0) + delta_weight
    after_variance = _risk_variance(
        after_weights, volatilities, betas, market_volatility
    )
    risk_penalty = (
        0.5
        * risk_aversion
        * horizon_sessions
        / 250
        * (after_variance - before_variance)
    )
    gross = delta_weight * expected_return
    error_allowance = delta_weight * error_multiplier * forecast_error
    transaction_cost = delta_weight * buy_cost_rate
    return {
        "incremental_weight": delta_weight,
        "gross_return_contribution": gross,
        "forecast_error_allowance": error_allowance,
        "transaction_cost": transaction_cost,
        "risk_penalty_increment": risk_penalty,
        "net_incremental_return": gross - error_allowance - transaction_cost - risk_penalty,
        "without_forecast_error": gross - transaction_cost - risk_penalty,
        "without_transaction_cost": gross - error_allowance - risk_penalty,
        "without_risk_penalty": gross - error_allowance - transaction_cost,
        "risk_after_trade_annualized": math.sqrt(max(after_variance, 0.0)),
    }


def _best_delta_weight(
    symbol: str,
    maximum_delta: float,
    *,
    expected_return: float,
    forecast_error: float,
    current_weights: Mapping[str, float],
    volatilities: Mapping[str, float],
    betas: Mapping[str, float],
    market_volatility: float,
    horizon_sessions: int,
    risk_aversion: float,
    error_multiplier: float,
    buy_cost_rate: float,
    remove_error: bool = False,
    remove_fee: bool = False,
    remove_risk: bool = False,
) -> float:
    """Maximize the concave one-name quadratic utility on a bounded interval."""
    if maximum_delta <= 0:
        return 0.0
    beta = float(betas.get(symbol, 0.0))
    current_weight = float(current_weights.get(symbol, 0.0))
    factor_exposure = math.fsum(
        weight * float(betas.get(name, 0.0))
        for name, weight in current_weights.items()
    )
    idiosyncratic_variance = max(
        float(volatilities[symbol]) ** 2 - beta**2 * market_volatility**2,
        0.0,
    )
    covariance_with_current = (
        current_weight * idiosyncratic_variance
        + factor_exposure * beta * market_volatility**2
    )
    variance_curvature = idiosyncratic_variance + beta**2 * market_volatility**2
    risk_scale = (
        0.0
        if remove_risk
        else 0.5 * risk_aversion * horizon_sessions / 250
    )
    linear_gain = expected_return
    if not remove_error:
        linear_gain -= error_multiplier * forecast_error
    if not remove_fee:
        linear_gain -= buy_cost_rate
    derivative_at_zero = linear_gain - 2 * risk_scale * covariance_with_current
    curvature = 2 * risk_scale * variance_curvature
    if derivative_at_zero <= 0:
        return 0.0
    if curvature <= 1e-18:
        return maximum_delta
    return min(maximum_delta, derivative_at_zero / curvature)


def summarize_next_session_execution(
    decision_date: str,
    execution_date: str | None,
    orders: list[Mapping[str, object]],
    nav_row: Mapping[str, object] | None,
    position_rows: list[Mapping[str, object]],
) -> dict[str, object]:
    """Summarize the actual simulator CSV contract, including unfilled states."""
    if any(str(order.get("decision_date")) != decision_date for order in orders):
        raise ValueError("成交账中的决策日期与归因日期不一致")
    statuses: dict[str, int] = {}
    requested = 0.0
    filled = 0.0
    costs = 0.0
    capacity_bound = 0
    for order in orders:
        status = str(order["status"])
        statuses[status] = statuses.get(status, 0) + 1
        requested_value = float(order["requested_notional_cny"])
        filled_value = float(order["filled_notional_cny"])
        capacity_value = float(order["liquidity_limit_cny"])
        requested += requested_value
        filled += filled_value
        # The continuous simulator writes `cost_cny`; keep its actual schema.
        costs += float(order["cost_cny"])
        if (
            requested_value > capacity_value + 0.01
            and abs(filled_value - capacity_value) <= 0.01
        ):
            capacity_bound += 1
    return {
        "trade_date": execution_date,
        "order_count": len(orders),
        "requested_notional_cny": requested,
        "filled_notional_cny": filled,
        "unfilled_notional_cny": max(requested - filled, 0.0),
        "transaction_cost_cny": costs,
        "unfilled_order_count": sum(
            count for status, count in statuses.items() if status != "filled"
        ),
        "status_counts": dict(sorted(statuses.items())),
        "capacity_bound_order_count": capacity_bound,
        "nav_cny": (
            float(nav_row["nav_cny"]) if nav_row is not None else None
        ),
        "cash_cny": (
            float(nav_row["cash_cny"]) if nav_row is not None else None
        ),
        "positions": [
            {
                "symbol": str(position["symbol"]),
                "market_value_cny": float(position["market_value_cny"]),
                "portfolio_weight": float(position["portfolio_weight"]),
            }
            for position in position_rows
        ],
        "reason_detail_available_in_order_schema": False,
        "order_status_semantics": "Simulator orders.csv provides status and capacity but no blocking_reason; unfilled_suspended_or_missing must be retained as a distinct status, not recoded as a liquidity rejection.",
    }


def attribute_optimizer_decision(
    forecast: Mapping[str, object],
    optimizer_decision: Mapping[str, object],
    config: Mapping[str, float | int],
    capacity_weights: Mapping[str, float],
) -> dict[str, object]:
    """Return a rebuildable daily decision row and best-name one-stock screen.

    The single-name screen is an attribution diagnostic: it holds the actual
    account fixed and asks what one incremental name would contribute under the
    same error allowance, fees, risk model, hard risk budget, and liquidity
    capacity. It is not a replacement optimizer or a trade recommendation.
    """
    expected_by_method = forecast["expected_returns_by_method"]
    errors_by_method = forecast["forecast_errors_by_method"]
    if not isinstance(expected_by_method, Mapping) or not isinstance(errors_by_method, Mapping):
        raise ValueError("预测记录缺少收益或成熟误差映射")
    returns = expected_by_method.get("loss_gated_fusion")
    error = errors_by_method.get("loss_gated_fusion")
    if not isinstance(returns, Mapping) or error is None:
        raise ValueError("决策归因要求成熟的损失门控预测和样本外误差")
    forecast_error = float(error)
    if not math.isfinite(forecast_error) or forecast_error < 0:
        raise ValueError("成熟预测误差必须有限且非负")

    nav = float(optimizer_decision["current_nav_cny"])
    if not math.isfinite(nav) or nav <= 0:
        raise ValueError("决策归因要求有限正净值")
    holdings = optimizer_decision["current_holdings_cny"]
    if not isinstance(holdings, Mapping):
        raise ValueError("当前持仓账必须是映射")
    current_weights = {str(symbol): float(value) / nav for symbol, value in holdings.items()}
    volatilities = forecast["annualized_volatility_by_symbol"]
    betas = forecast["factor_beta_by_symbol"]
    if not isinstance(volatilities, Mapping) or not isinstance(betas, Mapping):
        raise ValueError("预测记录缺少风险估计")
    market_volatility = float(forecast["market_annualized_volatility"])
    horizon = int(forecast["horizon_sessions"])
    error_multiplier = float(config["forecast_error_multiplier"])
    risk_aversion = float(config["risk_aversion"])
    risk_budget = float(config["risk_budget_annualized"])
    max_symbol = float(config["maximum_symbol_weight"])
    max_turnover = float(config["maximum_one_way_turnover"])
    buy_cost_rate = (
        float(config["commission_per_side_bps"])
        + float(config["transfer_fee_per_side_bps"])
        + float(config["slippage_per_side_bps"])
    ) / 10_000
    cash_weight = max(float(optimizer_decision["current_cash_cny"]) / nav, 0.0)
    turnover_room_weight = 2 * max_turnover
    base_risk = math.sqrt(
        _risk_variance(current_weights, volatilities, betas, market_volatility)
    )
    base_variance = base_risk**2
    existing_gross = math.fsum(
        weight * float(returns.get(symbol, 0.0))
        for symbol, weight in current_weights.items()
    )

    evaluations = []
    for symbol, raw_return in returns.items():
        symbol = str(symbol)
        if symbol not in volatilities:
            continue
        expected_return = float(raw_return)
        current = current_weights.get(symbol, 0.0)
        symbol_room = max(max_symbol - current, 0.0)
        unconstrained_room = min(
            symbol_room,
            turnover_room_weight,
            cash_weight / (1 + buy_cost_rate),
        )
        capacity_room = max(float(capacity_weights.get(symbol, 0.0)), 0.0)
        capacity_limited_room = min(unconstrained_room, capacity_room)

        def risk_limited(room: float) -> float:
            if room <= 0:
                return 0.0
            trial = dict(current_weights)
            trial[symbol] = current + room
            if _risk_variance(trial, volatilities, betas, market_volatility) <= risk_budget**2 + 1e-12:
                return room
            left, right = 0.0, room
            for _ in range(48):
                mid = (left + right) / 2
                trial[symbol] = current + mid
                if _risk_variance(trial, volatilities, betas, market_volatility) <= risk_budget**2:
                    left = mid
                else:
                    right = mid
            return left

        feasible_room = risk_limited(capacity_limited_room)
        economics_at_feasible_limit = _candidate_economics(
            symbol,
            feasible_room,
            expected_return=expected_return,
            forecast_error=forecast_error,
            current_weights=current_weights,
            volatilities=volatilities,
            betas=betas,
            market_volatility=market_volatility,
            horizon_sessions=horizon,
            risk_aversion=risk_aversion,
            error_multiplier=error_multiplier,
            buy_cost_rate=buy_cost_rate,
        )
        feasible_delta = _best_delta_weight(
            symbol,
            feasible_room,
            expected_return=expected_return,
            forecast_error=forecast_error,
            current_weights=current_weights,
            volatilities=volatilities,
            betas=betas,
            market_volatility=market_volatility,
            horizon_sessions=horizon,
            risk_aversion=risk_aversion,
            error_multiplier=error_multiplier,
            buy_cost_rate=buy_cost_rate,
        )
        economics = _candidate_economics(
            symbol,
            feasible_delta,
            expected_return=expected_return,
            forecast_error=forecast_error,
            current_weights=current_weights,
            volatilities=volatilities,
            betas=betas,
            market_volatility=market_volatility,
            horizon_sessions=horizon,
            risk_aversion=risk_aversion,
            error_multiplier=error_multiplier,
            buy_cost_rate=buy_cost_rate,
        )
        no_capacity_room = risk_limited(unconstrained_room)
        no_capacity_delta = _best_delta_weight(
            symbol,
            no_capacity_room,
            expected_return=expected_return,
            forecast_error=forecast_error,
            current_weights=current_weights,
            volatilities=volatilities,
            betas=betas,
            market_volatility=market_volatility,
            horizon_sessions=horizon,
            risk_aversion=risk_aversion,
            error_multiplier=error_multiplier,
            buy_cost_rate=buy_cost_rate,
        )
        economics_without_capacity = _candidate_economics(
            symbol,
            no_capacity_delta,
            expected_return=expected_return,
            forecast_error=forecast_error,
            current_weights=current_weights,
            volatilities=volatilities,
            betas=betas,
            market_volatility=market_volatility,
            horizon_sessions=horizon,
            risk_aversion=risk_aversion,
            error_multiplier=error_multiplier,
            buy_cost_rate=buy_cost_rate,
        )
        no_error_delta = _best_delta_weight(
            symbol,
            feasible_room,
            expected_return=expected_return,
            forecast_error=forecast_error,
            current_weights=current_weights,
            volatilities=volatilities,
            betas=betas,
            market_volatility=market_volatility,
            horizon_sessions=horizon,
            risk_aversion=risk_aversion,
            error_multiplier=error_multiplier,
            buy_cost_rate=buy_cost_rate,
            remove_error=True,
        )
        no_fee_room = risk_limited(
            min(symbol_room, turnover_room_weight, cash_weight)
        )
        no_fee_delta = _best_delta_weight(
            symbol,
            no_fee_room,
            expected_return=expected_return,
            forecast_error=forecast_error,
            current_weights=current_weights,
            volatilities=volatilities,
            betas=betas,
            market_volatility=market_volatility,
            horizon_sessions=horizon,
            risk_aversion=risk_aversion,
            error_multiplier=error_multiplier,
            buy_cost_rate=buy_cost_rate,
            remove_fee=True,
        )
        no_risk_delta = _best_delta_weight(
            symbol,
            feasible_room,
            expected_return=expected_return,
            forecast_error=forecast_error,
            current_weights=current_weights,
            volatilities=volatilities,
            betas=betas,
            market_volatility=market_volatility,
            horizon_sessions=horizon,
            risk_aversion=risk_aversion,
            error_multiplier=error_multiplier,
            buy_cost_rate=buy_cost_rate,
            remove_risk=True,
        )
        no_error_economics = _candidate_economics(
            symbol,
            no_error_delta,
            expected_return=expected_return,
            forecast_error=forecast_error,
            current_weights=current_weights,
            volatilities=volatilities,
            betas=betas,
            market_volatility=market_volatility,
            horizon_sessions=horizon,
            risk_aversion=risk_aversion,
            error_multiplier=error_multiplier,
            buy_cost_rate=buy_cost_rate,
        )
        no_fee_economics = _candidate_economics(
            symbol,
            no_fee_delta,
            expected_return=expected_return,
            forecast_error=forecast_error,
            current_weights=current_weights,
            volatilities=volatilities,
            betas=betas,
            market_volatility=market_volatility,
            horizon_sessions=horizon,
            risk_aversion=risk_aversion,
            error_multiplier=error_multiplier,
            buy_cost_rate=buy_cost_rate,
        )
        no_risk_economics = _candidate_economics(
            symbol,
            no_risk_delta,
            expected_return=expected_return,
            forecast_error=forecast_error,
            current_weights=current_weights,
            volatilities=volatilities,
            betas=betas,
            market_volatility=market_volatility,
            horizon_sessions=horizon,
            risk_aversion=risk_aversion,
            error_multiplier=error_multiplier,
            buy_cost_rate=buy_cost_rate,
        )
        evaluations.append(
            {
                "symbol": symbol,
                "expected_return": expected_return,
                "current_weight": current,
                "available_capacity_weight": capacity_room,
                "unconstrained_room_weight": unconstrained_room,
                "feasible_room_weight": feasible_room,
                "reference_feasible_limit_weight": feasible_room,
                "reference_limit_gross_return": economics_at_feasible_limit["gross_return_contribution"],
                "reference_limit_forecast_error_allowance": economics_at_feasible_limit["forecast_error_allowance"],
                "reference_limit_transaction_cost": economics_at_feasible_limit["transaction_cost"],
                "reference_limit_risk_penalty_increment": economics_at_feasible_limit["risk_penalty_increment"],
                "reference_limit_net_incremental_return": economics_at_feasible_limit["net_incremental_return"],
                **economics,
                "net_without_capacity_limit": economics_without_capacity["net_incremental_return"],
                "net_if_error_removed": no_error_economics["without_forecast_error"],
                "net_if_fees_removed": no_fee_economics["without_transaction_cost"],
                "net_if_risk_penalty_removed": no_risk_economics["without_risk_penalty"],
                "delta_if_error_removed": no_error_delta,
                "error_removed_probe_gross_return": no_error_economics["gross_return_contribution"],
                "error_removed_probe_original_error_allowance": no_error_economics["forecast_error_allowance"],
                "error_removed_probe_transaction_cost": no_error_economics["transaction_cost"],
                "error_removed_probe_risk_penalty_increment": no_error_economics["risk_penalty_increment"],
                "error_removed_probe_full_rule_net": no_error_economics["net_incremental_return"],
                "error_removed_probe_net_after_removing_error": no_error_economics["without_forecast_error"],
                "delta_if_fees_removed": no_fee_delta,
                "delta_if_risk_penalty_removed": no_risk_delta,
            }
        )
    if not evaluations:
        raise ValueError("没有同时具备收益和风险估计的候选股票")
    actual_buys = float(optimizer_decision["buy_notional_cny"])
    if actual_buys > 0:
        best = max(
            evaluations,
            key=lambda row: (
                float(row["net_incremental_return"]),
                float(row["expected_return"]),
                str(row["symbol"]),
            ),
        )
    else:
        # With a full-utility optimum of zero, identify the strongest
        # error-relaxed candidate and report its full-rule economics at that
        # re-optimized probe size. The probe is diagnostic, never a trade.
        best = max(
            evaluations,
            key=lambda row: (
                float(row["net_if_error_removed"]),
                float(row["expected_return"]),
                str(row["symbol"]),
            ),
        )
    causes = []
    if actual_buys <= 0:
        if float(best["expected_return"]) <= 0:
            causes.append("signal_nonpositive")
        if float(best["net_if_error_removed"]) > 0 and float(best["net_incremental_return"]) <= 0:
            causes.append("forecast_error_allowance")
        if float(best["net_if_fees_removed"]) > 0 and float(best["net_incremental_return"]) <= 0:
            causes.append("transaction_cost")
        if float(best["net_if_risk_penalty_removed"]) > 0 and float(best["net_incremental_return"]) <= 0:
            causes.append("optional_risk_penalty")
        if float(best["net_without_capacity_limit"]) > 0 and float(best["net_incremental_return"]) <= 0:
            causes.append("liquidity_capacity")
        if float(best["risk_after_trade_annualized"]) >= risk_budget - 1e-8:
            causes.append("hard_risk_budget")
        if float(best["unconstrained_room_weight"]) <= 1e-12:
            causes.append("cash_symbol_or_turnover_limit")
        if not causes:
            causes.append("combined_economic_terms_or_no_positive_candidate")
    else:
        causes.append("buy_authorized_by_optimizer")

    return {
        "decision_date": str(forecast["prediction_date"]),
        "execution_date": optimizer_decision.get("execution_date"),
        "horizon_sessions": horizon,
        "account_nav_cny": nav,
        "cash_cny": float(optimizer_decision["current_cash_cny"]),
        "current_holdings_cny": dict(holdings),
        "current_holdings_expected_gross_return": existing_gross,
        "optimizer_target_weights": dict(optimizer_decision["target_weights"]),
        "optimizer_buy_notional_cny": actual_buys,
        "optimizer_sell_notional_cny": float(optimizer_decision["sell_notional_cny"]),
        "optimizer_gross_expected_return": float(optimizer_decision["gross_expected_return"]),
        "optimizer_forecast_error_allowance": float(optimizer_decision["forecast_error_allowance"]),
        "optimizer_expected_transaction_cost_weight": float(optimizer_decision["expected_transaction_cost_weight"]),
        "optimizer_risk_penalty_weight": float(optimizer_decision["risk_penalty_weight"]),
        "optimizer_incremental_objective_value": float(optimizer_decision["incremental_objective_value"]),
        "optimizer_incremental_value_cny": float(optimizer_decision["risk_adjusted_net_incremental_value_cny"]),
        "risk_budget_annualized": risk_budget,
        "base_annualized_risk": base_risk,
        "estimated_cash_after_costs_cny": float(optimizer_decision["estimated_cash_after_costs_cny"]),
        "one_way_turnover_weight": float(optimizer_decision["one_way_turnover"]),
        "best_single_name_screen": best,
        "first_blocking_reason_diagnostic": causes[0],
        "blocking_reason_diagnostics": causes,
        "counterfactual_scope": "one additional name at a time from the actual account; uses frozen forecasts, RMSE allowance, fees, factor risk, risk budget, turnover and per-name capacity; does not select parameters or authorize trades",
    }
