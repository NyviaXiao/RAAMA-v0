"""Cost-aware long-only allocation under explicit risk and trading limits."""

from __future__ import annotations

import math
from dataclasses import dataclass


class PortfolioInfeasibleError(ValueError):
    """The requested risk and trading constraints cannot both be met."""


@dataclass(frozen=True)
class OptimizationResult:
    target_weights: dict[str, float]
    capacity_weights: dict[str, float]
    gross_expected_return: float
    forecast_error_allowance: float
    expected_transaction_cost: float
    annualized_risk: float
    maximum_one_way_turnover: float
    risk_budget_annualized: float
    objective_value: float
    holding_objective_value: float
    incremental_objective_value: float
    risk_penalty: float
    iterations: int


def optimize_portfolio(
    expected_returns: dict[str, float],
    forecast_errors: dict[str, float],
    annualized_volatilities: dict[str, float],
    current_weights: dict[str, float],
    *,
    horizon_sessions: int,
    maximum_symbol_weight: float,
    maximum_one_way_turnover: float,
    risk_budget_annualized: float,
    risk_aversion: float,
    forecast_error_multiplier: float,
    buy_cost_rate: float,
    sell_cost_rate: float,
    factor_betas: dict[str, float] | None = None,
    market_annualized_volatility: float = 0.0,
    max_change_weights: dict[str, float] | None = None,
    maximum_target_weights: dict[str, float] | None = None,
    iterations: int = 400,
) -> OptimizationResult:
    """Maximize calibrated return less risk, forecast error and actual trade cost.

    Returns are over the configured prediction horizon. Risk is estimated from
    annualized single-name volatility with a conservative common market factor
    shrinkage. The trading budget is measured against the supplied actual
    holdings, with one-way turnover defined as half of absolute traded weight.
    """
    if horizon_sessions < 1 or iterations < 1:
        raise ValueError("预测期限与优化迭代数必须为正整数")
    if maximum_symbol_weight <= 0 or maximum_symbol_weight > 1:
        raise ValueError("单票权重上限必须处于 (0, 1]")
    if not 0 <= maximum_one_way_turnover <= 1:
        raise ValueError("单向换手预算必须处于 [0, 1]")
    if risk_budget_annualized <= 0 or risk_aversion < 0:
        raise ValueError("风险预算必须为正，风险厌恶系数不能为负")
    if market_annualized_volatility < 0 or not math.isfinite(market_annualized_volatility):
        raise ValueError("市场因子年化波动率必须有限且非负")
    if forecast_error_multiplier < 0 or buy_cost_rate < 0 or sell_cost_rate < 0:
        raise ValueError("误差惩罚和交易成本不能为负")

    symbols = sorted(
        set(expected_returns)
        | set(forecast_errors)
        | set(annualized_volatilities)
        | set(current_weights)
        | set(maximum_target_weights or {})
    )
    if not symbols:
        return OptimizationResult(
            target_weights={},
            capacity_weights={},
            gross_expected_return=0.0,
            forecast_error_allowance=0.0,
            expected_transaction_cost=0.0,
            annualized_risk=0.0,
            maximum_one_way_turnover=0.0,
            risk_budget_annualized=risk_budget_annualized,
            objective_value=0.0,
            holding_objective_value=0.0,
            incremental_objective_value=0.0,
            risk_penalty=0.0,
            iterations=0,
        )
    for symbol in symbols:
        current = float(current_weights.get(symbol, 0.0))
        volatility = float(annualized_volatilities.get(symbol, 0.0))
        if not math.isfinite(current) or current < 0:
            raise ValueError(f"当前持仓权重无效：{symbol} {current}")
        if not math.isfinite(volatility) or volatility < 0:
            raise ValueError(f"年化波动率无效：{symbol} {volatility}")
        for source, values in (
            ("预期收益", expected_returns),
            ("预测误差", forecast_errors),
        ):
            value = float(values.get(symbol, 0.0))
            if not math.isfinite(value) or (source == "预测误差" and value < 0):
                raise ValueError(f"{source}无效：{symbol} {value}")
        maximum_target = float((maximum_target_weights or {}).get(symbol, maximum_symbol_weight))
        if not math.isfinite(maximum_target) or maximum_target < 0:
            raise ValueError(f"单票目标权重上限无效：{symbol} {maximum_target}")

    current = {symbol: float(current_weights.get(symbol, 0.0)) for symbol in symbols}
    if math.fsum(current.values()) > 1 + 1e-8:
        raise PortfolioInfeasibleError("当前持仓权重超过净值，无法满足现金非负约束")

    uncertainty_adjusted = {
        symbol: float(expected_returns.get(symbol, 0.0))
        - forecast_error_multiplier * float(forecast_errors.get(symbol, 0.0))
        for symbol in symbols
    }
    volatility = {
        symbol: float(annualized_volatilities.get(symbol, 0.0)) for symbol in symbols
    }
    betas = {symbol: float((factor_betas or {}).get(symbol, 0.0)) for symbol in symbols}
    idiosyncratic_variance = {
        symbol: max(
            volatility[symbol] ** 2
            - betas[symbol] ** 2 * market_annualized_volatility**2,
            0.0,
        )
        for symbol in symbols
    }
    max_changes = max_change_weights or {}
    lower = {
        symbol: max(current[symbol] - float(max_changes.get(symbol, 1.0)), 0.0)
        for symbol in symbols
    }
    upper = {
        symbol: min(
            maximum_symbol_weight,
            current[symbol] + float(max_changes.get(symbol, 1.0)),
            float((maximum_target_weights or {}).get(symbol, maximum_symbol_weight)),
        )
        for symbol in symbols
    }
    lower = {symbol: min(lower[symbol], upper[symbol]) for symbol in symbols}
    if math.fsum(lower.values()) > 1 + 1e-10:
        raise PortfolioInfeasibleError("最低持仓约束下现金将为负")

    def project_capped_simplex(values: dict[str, float]) -> dict[str, float]:
        projected = {
            symbol: min(max(values[symbol], lower[symbol]), upper[symbol])
            for symbol in symbols
        }
        if cash_used(projected) <= 1.0:
            return projected
        lower_cash_use = cash_used(lower)
        if lower_cash_use > 1.0 + 1e-10:
            raise PortfolioInfeasibleError("最低持仓加交易费用会导致现金为负")
        left = min(values[symbol] - upper[symbol] for symbol in symbols) - 1.0
        right = max(values[symbol] - lower[symbol] for symbol in symbols) + 1.0
        for _ in range(64):
            level = (left + right) / 2
            trial = {
                symbol: min(max(values[symbol] - level, lower[symbol]), upper[symbol])
                for symbol in symbols
            }
            if cash_used(trial) > 1.0:
                left = level
            else:
                right = level
        return {
            symbol: min(max(values[symbol] - right, lower[symbol]), upper[symbol])
            for symbol in symbols
        }

    def transaction_cost_of(weights: dict[str, float]) -> float:
        return math.fsum(
            max(weights[symbol] - current[symbol], 0.0) * buy_cost_rate
            + max(current[symbol] - weights[symbol], 0.0) * sell_cost_rate
            for symbol in symbols
        )

    def cash_used(weights: dict[str, float]) -> float:
        return math.fsum(weights.values()) + transaction_cost_of(weights)

    def risk_of(weights: dict[str, float]) -> float:
        factor_exposure = math.fsum(weights[symbol] * betas[symbol] for symbol in symbols)
        variance = math.fsum(
            weights[symbol] ** 2 * idiosyncratic_variance[symbol]
            for symbol in symbols
        )
        variance += (factor_exposure * market_annualized_volatility) ** 2
        return math.sqrt(max(variance, 0.0))

    def objective_of(weights: dict[str, float]) -> float:
        gross_less_error = math.fsum(
            weights[symbol] * uncertainty_adjusted[symbol] for symbol in symbols
        )
        risk = risk_of(weights)
        risk_cost = (
            0.5 * risk_aversion * (horizon_sessions / 250) * risk * risk
        )
        return gross_less_error - transaction_cost_of(weights) - risk_cost

    current_risk = risk_of(current)
    current_feasible = (
        current_risk <= risk_budget_annualized + 1e-10
        and all(current[symbol] <= maximum_symbol_weight + 1e-10 for symbol in symbols)
    )

    de_risk_start = dict(current)
    if not current_feasible:
        maximum_sale = dict(lower)
        if risk_of(maximum_sale) > risk_budget_annualized + 1e-10 or any(
            maximum_sale[symbol] > maximum_symbol_weight + 1e-10
            for symbol in symbols
        ):
            raise PortfolioInfeasibleError(
                "可成交减仓上限不足以恢复风险或单票约束"
            )
        previous_scale = 0.0
        feasible_scale = None
        for index in range(1, 129):
            scale = index / 128
            trial = {
                symbol: current[symbol]
                + scale * (maximum_sale[symbol] - current[symbol])
                for symbol in symbols
            }
            if risk_of(trial) <= risk_budget_annualized and all(
                trial[symbol] <= maximum_symbol_weight + 1e-10
                for symbol in symbols
            ):
                feasible_scale = scale
                break
            previous_scale = scale
        if feasible_scale is None:
            raise PortfolioInfeasibleError("没有找到满足风险预算的可行减仓")
        left, right = previous_scale, feasible_scale
        for _ in range(60):
            scale = (left + right) / 2
            trial = {
                symbol: current[symbol]
                + scale * (maximum_sale[symbol] - current[symbol])
                for symbol in symbols
            }
            if risk_of(trial) <= risk_budget_annualized and all(
                trial[symbol] <= maximum_symbol_weight + 1e-10
                for symbol in symbols
            ):
                right = scale
            else:
                left = scale
        de_risk_start = {
            symbol: current[symbol]
            + right * (maximum_sale[symbol] - current[symbol])
            for symbol in symbols
        }
        required_turnover = math.fsum(
            abs(de_risk_start[symbol] - current[symbol]) for symbol in symbols
        ) / 2
        if required_turnover > maximum_one_way_turnover + 1e-8:
            raise PortfolioInfeasibleError(
                "恢复风险或单票上限所需的最小换手超过本次预算"
            )

    def constrain(weights: dict[str, float]) -> dict[str, float]:
        constrained = project_capped_simplex(weights)
        traded = math.fsum(abs(constrained[symbol] - current[symbol]) for symbol in symbols) / 2
        if traded > maximum_one_way_turnover:
            scale = maximum_one_way_turnover / traded if traded else 1.0
            constrained = {
                symbol: current[symbol]
                + scale * (constrained[symbol] - current[symbol])
                for symbol in symbols
            }
        risk = risk_of(constrained)
        if risk > risk_budget_annualized:
            if current_feasible:
                left, right = 0.0, 1.0
                for _ in range(60):
                    scale = (left + right) / 2
                    trial = {
                        symbol: current[symbol]
                        + scale * (constrained[symbol] - current[symbol])
                        for symbol in symbols
                    }
                    if risk_of(trial) <= risk_budget_annualized:
                        left = scale
                    else:
                        right = scale
                constrained = {
                    symbol: current[symbol]
                    + left * (constrained[symbol] - current[symbol])
                    for symbol in symbols
                }
            else:
                # Start from a proven feasible risk-repair target and interpolate
                # toward the utility optimum without exceeding turnover limits.
                left, right = 0.0, 1.0
                for _ in range(60):
                    scale = (left + right) / 2
                    trial = {
                        symbol: de_risk_start[symbol]
                        + scale * (constrained[symbol] - de_risk_start[symbol])
                        for symbol in symbols
                    }
                    if risk_of(trial) <= risk_budget_annualized:
                        left = scale
                    else:
                        right = scale
                constrained = {
                    symbol: de_risk_start[symbol]
                    + left * (constrained[symbol] - de_risk_start[symbol])
                    for symbol in symbols
                }
        return constrained

    # Proximal gradient uses the exact diagonal plus single-market-factor
    # covariance gradient, then applies asymmetric buy/sell cost thresholds.
    weights = dict(current if current_feasible else de_risk_start)
    market_variance = market_annualized_volatility**2
    covariance_row_sum_bound = max(
        (
            idiosyncratic_variance[symbol]
            + abs(betas[symbol])
            * market_variance
            * math.fsum(abs(betas[other]) for other in symbols)
            for symbol in symbols
        ),
        default=1e-6,
    )
    curvature = risk_aversion * (horizon_sessions / 250) * max(
        covariance_row_sum_bound, 1e-6
    )
    step = min(2.0, 0.25 / max(curvature, 1e-6))
    for _ in range(iterations):
        factor_exposure = math.fsum(
            weights[symbol] * betas[symbol] for symbol in symbols
        )
        gradient = {}
        for symbol in symbols:
            covariance_times_weight = (
                idiosyncratic_variance[symbol] * weights[symbol]
                + betas[symbol]
                * market_annualized_volatility**2
                * factor_exposure
            )
            gradient[symbol] = uncertainty_adjusted[symbol] - (
                risk_aversion
                * (horizon_sessions / 250)
                * covariance_times_weight
            )
        unconstrained = {
            symbol: weights[symbol] + step * gradient[symbol]
            for symbol in symbols
        }
        proximal = {}
        for symbol in symbols:
            difference = unconstrained[symbol] - current[symbol]
            buy_threshold = step * buy_cost_rate
            sell_threshold = step * sell_cost_rate
            if difference > buy_threshold:
                proximal[symbol] = current[symbol] + difference - buy_threshold
            elif difference < -sell_threshold:
                proximal[symbol] = current[symbol] + difference + sell_threshold
            else:
                proximal[symbol] = current[symbol]
        updated = constrain(proximal)
        if max(abs(updated[symbol] - weights[symbol]) for symbol in symbols) < 1e-9:
            weights = updated
            break
        weights = updated

    turnover = math.fsum(abs(weights[symbol] - current[symbol]) for symbol in symbols) / 2
    annualized_risk = risk_of(weights)
    if turnover > maximum_one_way_turnover + 1e-7:
        raise PortfolioInfeasibleError("目标权重超过本次单向换手预算")
    if annualized_risk > risk_budget_annualized + 1e-7:
        raise PortfolioInfeasibleError("风险预算与现金/交易约束无法同时满足")
    if math.fsum(weights.values()) > 1 + 1e-7:
        raise PortfolioInfeasibleError("组合求解结果违反现金非负约束")
    if any(weight > maximum_symbol_weight + 1e-7 for weight in weights.values()):
        raise PortfolioInfeasibleError("组合求解结果违反单票集中度约束")
    if any(
        abs(weights[symbol] - current[symbol])
        > float(max_changes.get(symbol, 1.0)) + 1e-7
        for symbol in symbols
    ):
        raise PortfolioInfeasibleError("组合求解结果超过单票成交容量约束")

    if current_feasible and objective_of(weights) < objective_of(current) - 1e-10:
        weights = dict(current)
        turnover = 0.0
        annualized_risk = current_risk

    gross = math.fsum(
        weights[symbol] * float(expected_returns.get(symbol, 0.0))
        for symbol in symbols
    )
    error_allowance = math.fsum(
        weights[symbol]
        * forecast_error_multiplier
        * float(forecast_errors.get(symbol, 0.0))
        for symbol in symbols
    )
    transaction_cost = transaction_cost_of(weights)
    objective = objective_of(weights)
    holding_objective = objective_of(current)
    risk_penalty = (
        0.5 * risk_aversion * (horizon_sessions / 250) * annualized_risk**2
    )
    return OptimizationResult(
        target_weights={
            symbol: weight for symbol, weight in weights.items() if weight > 1e-10
        },
        capacity_weights=dict(max_changes),
        gross_expected_return=gross,
        forecast_error_allowance=error_allowance,
        expected_transaction_cost=transaction_cost,
        annualized_risk=annualized_risk,
        maximum_one_way_turnover=turnover,
        risk_budget_annualized=risk_budget_annualized,
        objective_value=objective,
        holding_objective_value=holding_objective,
        incremental_objective_value=objective - holding_objective,
        risk_penalty=risk_penalty,
        iterations=iterations,
    )
