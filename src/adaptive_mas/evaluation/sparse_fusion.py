"""Prequential expert activation from realized single-expert account returns."""

from __future__ import annotations

import math
import statistics
from datetime import date
from typing import Mapping

from adaptive_mas.agents.quantitative import AGENT_IDS
from adaptive_mas.portfolio.allocation import (
    apply_weight_and_turnover_limits,
    top_fraction_weights,
)


def build_sparse_fusion_decisions(
    forecasts: list[Mapping[str, object]],
    realized_expert_daily_returns: Mapping[str, list[Mapping[str, object]]],
    *,
    comparison_start_date: date,
    portfolio_fraction: float,
    invested_weight: float,
    maximum_symbol_weight: float,
    maximum_one_way_turnover: float,
    risk_budget_annualized: float,
    matured_window: int,
    minimum_matured_decisions: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Enable experts with positive trailing *realized net account returns*.

    Input returns come from the project's single-expert continuous simulator,
    so they already include actual next-session fills, capacity limits, holding
    changes, and transaction costs. A forecast date sees only strictly earlier
    simulator mark dates. Experts that do not have enough completed returns or
    whose trailing mean is non-positive receive zero weight.
    """
    if matured_window < 1 or minimum_matured_decisions < 1:
        raise ValueError("稀疏融合的成熟历史窗口与最低样本数必须为正")
    if minimum_matured_decisions > matured_window:
        raise ValueError("稀疏融合最低成熟样本数不能超过滚动窗口")
    if not 0 < portfolio_fraction <= 1 or not 0 < invested_weight <= 1:
        raise ValueError("稀疏融合的候选比例和投入比例必须在 (0,1] 内")
    if set(realized_expert_daily_returns) != set(AGENT_IDS):
        raise ValueError("稀疏融合必须收到全部单专家的同模拟器净值收益账")

    returns_by_expert: dict[str, list[dict[str, object]]] = {}
    for expert in AGENT_IDS:
        rows = sorted(
            [dict(row) for row in realized_expert_daily_returns[expert]],
            key=lambda row: str(row["trade_date"]),
        )
        for row in rows:
            value = float(row["daily_return"])
            if not math.isfinite(value):
                raise ValueError(f"专家净值收益非有限：{expert} {row['trade_date']}")
        returns_by_expert[expert] = rows

    if risk_budget_annualized <= 0:
        raise ValueError("稀疏融合的年化风险预算必须为正")
    decisions = []
    activation_log = []
    previous_targets: dict[str, float] = {}
    for forecast in forecasts:
        as_of = date.fromisoformat(str(forecast["prediction_date"]))
        if as_of < comparison_start_date:
            continue
        windows: dict[str, list[dict[str, object]]] = {}
        means: dict[str, float | None] = {}
        for expert in AGENT_IDS:
            matured = [
                row
                for row in returns_by_expert[expert]
                if str(row["trade_date"]) < as_of.isoformat()
            ][-matured_window:]
            windows[expert] = matured
            means[expert] = (
                statistics.fmean(float(row["daily_return"]) for row in matured)
                if matured else None
            )
        active = [
            expert
            for expert in AGENT_IDS
            if len(windows[expert]) >= minimum_matured_decisions
            and means[expert] is not None
            and float(means[expert]) > 0
        ]
        active_weights = {expert: 1 / len(active) for expert in active}
        pre_risk = 0.0
        risk_scale = 1.0
        risk_budget_before_turnover = 0.0
        market_volatility = float(forecast["market_annualized_volatility"])
        volatilities = forecast["annualized_volatility_by_symbol"]
        betas = forecast["factor_beta_by_symbol"]
        if active:
            scores_by_method = forecast["scores_by_method"]
            common_symbols = set.intersection(
                *(set(scores_by_method[expert]) for expert in active)
            )
            scores = {
                symbol: math.fsum(
                    active_weights[expert] * float(scores_by_method[expert][symbol])
                    for expert in active
                )
                for symbol in common_symbols
            }
            target = top_fraction_weights(scores, portfolio_fraction, invested_weight)
            missing_risk = set(target) - set(volatilities)
            if missing_risk:
                raise ValueError(f"稀疏融合候选缺少已成熟风险估计：{sorted(missing_risk)}")
            idiosyncratic = math.fsum(
                target[symbol] ** 2
                * max(
                    float(volatilities[symbol]) ** 2
                    - float(betas[symbol]) ** 2 * market_volatility**2,
                    0.0,
                )
                for symbol in target
            )
            factor_exposure = math.fsum(
                target[symbol] * float(betas[symbol]) for symbol in target
            )
            pre_risk = math.sqrt(
                idiosyncratic + (factor_exposure * market_volatility) ** 2
            )
            risk_scale = min(1.0, risk_budget_annualized / pre_risk) if pre_risk > 0 else 1.0
            target = {symbol: weight * risk_scale for symbol, weight in target.items()}
            target = apply_weight_and_turnover_limits(
                target,
                previous_targets,
                maximum_symbol_weight,
                maximum_one_way_turnover,
            )
        else:
            target = apply_weight_and_turnover_limits(
                {},
                previous_targets,
                maximum_symbol_weight,
                maximum_one_way_turnover,
            )
        if target:
            missing_risk = set(target) - set(volatilities)
            if missing_risk:
                raise ValueError(f"换手后持仓目标缺少点时风险估计：{sorted(missing_risk)}")
            idiosyncratic = math.fsum(
                target[symbol] ** 2
                * max(
                    float(volatilities[symbol]) ** 2
                    - float(betas[symbol]) ** 2 * market_volatility**2,
                    0.0,
                )
                for symbol in target
            )
            factor_exposure = math.fsum(
                target[symbol] * float(betas[symbol]) for symbol in target
            )
            risk_budget_before_turnover = math.sqrt(
                idiosyncratic + (factor_exposure * market_volatility) ** 2
            )
            if risk_budget_before_turnover > risk_budget_annualized + 1e-10:
                final_scale = risk_budget_annualized / risk_budget_before_turnover
                risk_limited_target = {
                    symbol: weight * final_scale for symbol, weight in target.items()
                }
                actual_turnover = math.fsum(
                    abs(risk_limited_target.get(symbol, 0.0) - previous_targets.get(symbol, 0.0))
                    for symbol in set(risk_limited_target) | set(previous_targets)
                ) / 2
                if actual_turnover > maximum_one_way_turnover + 1e-9:
                    raise ValueError(
                        f"年化风险预算与单向换手上限不可同时满足：{as_of} risk={risk_budget_before_turnover:.6f} turnover={actual_turnover:.6f}"
                    )
                target = risk_limited_target
                risk_scale *= final_scale
        final_risk = 0.0
        if target:
            final_idiosyncratic = math.fsum(
                target[symbol] ** 2
                * max(
                    float(volatilities[symbol]) ** 2
                    - float(betas[symbol]) ** 2 * market_volatility**2,
                    0.0,
                )
                for symbol in target
            )
            final_factor_exposure = math.fsum(
                target[symbol] * float(betas[symbol]) for symbol in target
            )
            final_risk = math.sqrt(
                final_idiosyncratic
                + (final_factor_exposure * market_volatility) ** 2
            )
            if final_risk > risk_budget_annualized + 1e-8:
                raise ValueError(f"风险覆盖后的目标仍超出风险预算：{as_of} {final_risk}")
        previous_targets = target
        decisions.append({"decision_date": as_of, "target_weights": target})
        activation_log.append(
            {
                "decision_date": as_of.isoformat(),
                "input_cutoff": as_of.isoformat(),
                "active_experts": active,
                "expert_weights": active_weights,
                "matured_net_contribution_count": {
                    expert: len(windows[expert]) for expert in AGENT_IDS
                },
                "matured_net_contribution_dates": {
                    expert: [str(row["trade_date"]) for row in windows[expert]]
                    for expert in AGENT_IDS
                },
                "trailing_mean_net_daily_return": means,
                "target_weights": target,
                "target_invested_weight": math.fsum(target.values()),
                "pre_overlay_annualized_risk": pre_risk,
                "post_turnover_pre_final_overlay_risk": risk_budget_before_turnover,
                "final_target_annualized_risk": final_risk,
                "risk_budget_annualized": risk_budget_annualized,
                "hard_risk_overlay_scale": risk_scale,
            }
        )
    return decisions, activation_log
