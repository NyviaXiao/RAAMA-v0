"""Stable single-day decision contract for applications and historical replay."""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from dataclasses import asdict, dataclass, replace
from datetime import date
from pathlib import Path
from typing import Mapping

from adaptive_mas.agents.collaborative import (
    build_agent_reports,
    fixed_p9_comparison,
    investment_manager_explanation,
    one_round_challenge,
    research_manager,
    risk_role_reviews,
    run_supplementary_challenge,
)
from adaptive_mas.data.research import ResearchData
from adaptive_mas.evaluation.public_expert_overlay import add_public_expert_overlay
from adaptive_mas.evaluation.contextual_rl import (
    apply_action,
    choose_action,
    contextual_state,
)
from adaptive_mas.evaluation.mvp_research import _risk_estimates, run_mvp_research
from adaptive_mas.portfolio.optimizer import OptimizationResult, optimize_portfolio


@dataclass(frozen=True)
class PortfolioState:
    cash_cny: float
    holdings_cny: Mapping[str, float]

    @property
    def nav_cny(self) -> float:
        return self.cash_cny + math.fsum(self.holdings_cny.values())

    def validate(self) -> None:
        if not math.isfinite(self.cash_cny) or self.cash_cny < 0:
            raise ValueError("组合现金必须是有限非负金额")
        if any(
            len(symbol) != 6 or not symbol.isascii() or not symbol.isdigit()
            for symbol in self.holdings_cny
        ):
            raise ValueError("持仓证券代码必须是六位数字字符串")
        if any(not math.isfinite(float(value)) or value < 0 for value in self.holdings_cny.values()):
            raise ValueError("持仓市值必须是有限非负金额")
        if self.nav_cny <= 0:
            raise ValueError("组合净值必须为正")


@dataclass(frozen=True)
class DecisionOrder:
    symbol: str
    side: str
    notional_cny: float
    current_value_cny: float
    target_value_cny: float
    expected_return: float
    forecast_error: float
    expected_incremental_value_cny: float
    estimated_cost_cny: float
    estimated_liquidity_capacity_cny: float
    decision_date_tradestatus: str
    next_session_execution_uncertain: bool
    execution_note: str
    status: str
    source_prediction_id: str = ""
    prediction_date: str = ""
    prediction_horizon_sessions: int = 0


@dataclass(frozen=True)
class DailyDecision:
    schema_version: str
    as_of_date: str
    execution_date: str | None
    status: str
    prediction_status: str
    prediction_horizon_sessions: int
    predictions_by_symbol: Mapping[str, Mapping[str, object]]
    current_nav_cny: float
    current_cash_cny: float
    current_holdings_cny: Mapping[str, float]
    target_cash_cny: float
    target_holdings_cny: Mapping[str, float]
    target_weights: Mapping[str, float]
    estimated_cash_after_costs_cny: float
    buy_notional_cny: float
    sell_notional_cny: float
    orders: tuple[DecisionOrder, ...]
    expected_gross_return: float | None
    forecast_error_allowance: float | None
    expected_transaction_cost_cny: float
    current_holding_objective_value_cny: float | None
    optimized_objective_value_cny: float | None
    risk_adjusted_net_incremental_value_cny: float
    risk_penalty_cny: float | None
    annualized_risk: float | None
    annualized_risk_budget: float
    one_way_turnover: float
    evidence: Mapping[str, object]
    versions: Mapping[str, str | None]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _optimizer_expected_returns(
    forecast: Mapping[str, object],
    return_method: str,
    config: Mapping[str, object],
    current_symbols: set[str] | None = None,
) -> dict[str, float]:
    expected_by_method = forecast["expected_returns_by_method"]
    if not isinstance(expected_by_method, Mapping) or return_method not in expected_by_method:
        raise ValueError(f"联合优化缺少指定收益预测方法：{return_method}")
    method_returns = expected_by_method[return_method]
    if not isinstance(method_returns, Mapping):
        raise ValueError(f"{return_method} 预期收益必须为证券映射")
    scope = config.get("optimizer_candidate_scope")
    if scope is None:
        return {str(symbol): float(value) for symbol, value in method_returns.items()}
    if scope == "harness_priority":
        priorities = [str(symbol) for symbol in config["optimizer_priority_symbols"]]
        eligible = set(priorities)
        method_returns_by_symbol = {
            str(symbol): float(value) for symbol, value in method_returns.items()
        }
        if not eligible:
            eligible = set(method_returns_by_symbol)
        eligible_with_existing = eligible | (current_symbols or set())
        return {
            symbol: method_returns_by_symbol.get(symbol, 0.0)
            for symbol in sorted(eligible_with_existing)
        }
    if scope != "frozen_method_target":
        raise ValueError(f"不支持的联合优化候选范围：{scope}")
    targets_by_method = forecast.get("target_weights_by_method")
    if not isinstance(targets_by_method, Mapping):
        raise ValueError(f"{return_method} 候选范围缺少冻结的目标权重")
    method_target = targets_by_method.get(return_method)
    if not isinstance(method_target, Mapping) or not method_target:
        raise ValueError(f"{return_method} 没有冻结候选篮子")
    eligible = {
        str(symbol)
        for symbol, weight in method_target.items()
        if float(weight) > 0
    }
    missing = eligible - {str(symbol) for symbol in method_returns}
    if missing:
        raise ValueError(f"{return_method} 候选篮子缺少预期收益：{sorted(missing)}")
    eligible_with_existing = eligible | (current_symbols or set())
    method_returns_by_symbol = {
        str(symbol): float(value) for symbol, value in method_returns.items()
    }
    # A held stock can leave the point-in-time forecast universe before its
    # position can be sold. It stays sell-only and receives no forecast alpha.
    return {
        symbol: method_returns_by_symbol.get(symbol, 0.0)
        for symbol in sorted(eligible_with_existing)
    }


def _optimizer_candidate_symbols(
    forecast: Mapping[str, object],
    return_method: str,
    config: Mapping[str, object],
) -> set[str] | None:
    scope = config.get("optimizer_candidate_scope")
    if scope is None:
        return None
    if scope == "harness_priority":
        method_returns = forecast["expected_returns_by_method"][return_method]
        return {
            str(symbol)
            for symbol in config["optimizer_priority_symbols"]
            if str(symbol) in method_returns
        }
    if scope != "frozen_method_target":
        raise ValueError(f"不支持的联合优化候选范围：{scope}")
    targets_by_method = forecast.get("target_weights_by_method")
    if not isinstance(targets_by_method, Mapping):
        raise ValueError(f"{return_method} 候选范围缺少冻结的目标权重")
    method_target = targets_by_method.get(return_method)
    if not isinstance(method_target, Mapping) or not method_target:
        raise ValueError(f"{return_method} 没有冻结候选篮子")
    return {
        str(symbol)
        for symbol, weight in method_target.items()
        if float(weight) > 0
    }


def _portfolio_risk_from_model(
    weights: Mapping[str, float], risk_model: Mapping[str, object]
) -> float | None:
    vol_by_symbol = risk_model.get("annualized_volatility_by_symbol")
    beta_by_symbol = risk_model.get("factor_beta_by_symbol")
    market_vol = risk_model.get("market_annualized_volatility")
    if not isinstance(vol_by_symbol, Mapping) or not isinstance(beta_by_symbol, Mapping):
        return None
    try:
        market_vol = float(market_vol)
        factor_exposure = 0.0
        variance = 0.0
        for symbol, raw_weight in weights.items():
            weight = float(raw_weight)
            volatility = float(vol_by_symbol.get(symbol, 0.0))
            beta = float(beta_by_symbol.get(symbol, 0.0))
            idiosyncratic_variance = max(volatility**2 - beta**2 * market_vol**2, 0.0)
            factor_exposure += weight * beta
            variance += weight**2 * idiosyncratic_variance
        variance += (factor_exposure * market_vol) ** 2
    except (TypeError, ValueError):
        return None
    if not math.isfinite(variance) or variance < 0:
        return None
    return math.sqrt(variance)


def _hard_risk_review(
    decision: DailyDecision, config: Mapping[str, object]
) -> tuple[DailyDecision, dict[str, object], dict[str, object]]:
    """Review the optimizer proposal and block orders if any fixed check fails."""
    proposed_payload = decision.to_dict()
    reviews = risk_role_reviews(proposed_payload, dict(config))
    checks = dict(next(iter(reviews.values()))["checks"])
    failed_checks = sorted(key for key, passed in checks.items() if not passed)
    outcome: dict[str, object] = {
        "status": "passed" if not failed_checks else "blocked",
        "orders_blocked": bool(failed_checks),
        "failed_checks": failed_checks,
        "checks": checks,
        "proposal": {
            "target_weights": dict(decision.target_weights),
            "target_holdings_cny": dict(decision.target_holdings_cny),
            "annualized_risk": decision.annualized_risk,
            "risk_budget_annualized": decision.annualized_risk_budget,
            "estimated_cash_after_costs_cny": decision.estimated_cash_after_costs_cny,
            "one_way_turnover": decision.one_way_turnover,
            "order_count": len(decision.orders),
        },
        "execution": "optimizer proposal retained" if not failed_checks else "all proposed orders blocked; hold current positions",
        "hard_constraints_are_narrative_overridable": False,
    }
    if not failed_checks:
        return decision, reviews, outcome

    current_weights = {
        symbol: float(value) / decision.current_nav_cny
        for symbol, value in decision.current_holdings_cny.items()
    }
    evidence = dict(decision.evidence)
    risk_model = evidence.get("risk_model")
    current_risk = (
        _portfolio_risk_from_model(current_weights, risk_model)
        if isinstance(risk_model, Mapping) else None
    )
    actual_hold_state = {
        "target_weights": current_weights,
        "target_holdings_cny": dict(decision.current_holdings_cny),
        "annualized_risk": current_risk,
        "risk_budget_annualized": decision.annualized_risk_budget,
        "cash_cny": decision.current_cash_cny,
        "new_orders": 0,
        "cash_movement": 0.0,
        "note": "Existing positions remain; any pre-existing limit breach is reported and requires separate remediation.",
    }
    actual_hold_state["risk_within_budget"] = (
        current_risk is not None
        and current_risk <= decision.annualized_risk_budget + 1e-10
    )
    outcome["actual_hold_state"] = actual_hold_state
    outcome["status"] = (
        "blocked_risk_unresolved"
        if not actual_hold_state["risk_within_budget"]
        else "blocked_proposal"
    )
    blocked = replace(
        decision,
        status="risk_blocked",
        target_cash_cny=decision.current_cash_cny,
        target_holdings_cny=dict(decision.current_holdings_cny),
        target_weights=current_weights,
        estimated_cash_after_costs_cny=decision.current_cash_cny,
        buy_notional_cny=0.0,
        sell_notional_cny=0.0,
        orders=(),
        expected_gross_return=None,
        forecast_error_allowance=None,
        expected_transaction_cost_cny=0.0,
        current_holding_objective_value_cny=None,
        optimized_objective_value_cny=None,
        risk_adjusted_net_incremental_value_cny=0.0,
        risk_penalty_cny=None,
        annualized_risk=current_risk,
        one_way_turnover=0.0,
        evidence={**evidence, "hard_risk_outcome": outcome},
    )
    return blocked, reviews, outcome


def optimize_forecast_target(
    research: ResearchData,
    forecast: dict[str, object],
    portfolio: PortfolioState,
    config: dict[str, float | int],
) -> tuple[OptimizationResult, dict[str, float]]:
    """Use the same actual-account optimizer in the app API and replay."""
    portfolio.validate()
    as_of = date.fromisoformat(str(forecast["prediction_date"]))
    decision_index = research.trading_dates.index(as_of)
    current_weights = {
        symbol: value / portfolio.nav_cny
        for symbol, value in portfolio.holdings_cny.items()
    }
    current_symbols = set(current_weights)
    return_method = str(config.get("optimizer_return_method", "loss_gated_fusion"))
    candidate_symbols = _optimizer_candidate_symbols(forecast, return_method, config)
    all_expected_returns = _optimizer_expected_returns(
        forecast, return_method, config, current_symbols
    )
    expected_errors = dict(forecast["forecast_errors_by_method"])
    error_value = expected_errors.get(return_method)
    if error_value is None:
        raise ValueError(f"联合优化的 {return_method} 方法仍在暖启动：没有已成熟的样本外预测误差")
    error = float(error_value)
    risk_symbols = set(all_expected_returns) | current_symbols
    risk_vols, factor_betas, market_vol = _risk_estimates(
        research,
        decision_index,
        risk_symbols,
        int(config["risk_lookback_sessions"]),
        int(config["risk_minimum_observations"]),
    )
    missing_current_risk = current_symbols - set(risk_vols)
    if missing_current_risk:
        raise ValueError(
            f"缺少达到最低样本数的实际持仓风险估计：{sorted(missing_current_risk)}"
        )
    expected_returns = {
        symbol: value
        for symbol, value in all_expected_returns.items()
        if symbol in risk_vols
    }
    risk_symbols = set(expected_returns) | current_symbols
    risk_vols = {symbol: risk_vols[symbol] for symbol in risk_symbols}
    factor_betas = {symbol: factor_betas[symbol] for symbol in risk_symbols}
    maximum_target_weights = None
    if candidate_symbols is not None:
        if config.get("optimizer_candidate_scope") == "harness_priority" and not candidate_symbols:
            maximum_target_weights = {
                str(symbol): 0.0
                for symbol in forecast["expected_returns_by_method"][return_method]
            }
        else:
            maximum_target_weights = {
                symbol: float(config["maximum_symbol_weight"])
                for symbol in candidate_symbols
            }
        maximum_target_weights.update({
            symbol: current_weights[symbol]
            for symbol in current_symbols - candidate_symbols
        })

    execution = {
        (str(row["symbol"]), row["trade_date"]): row
        for row in research.execution_bars
    }
    lookback = int(config.get("liquidity_lookback_sessions", 20))
    participation = float(config.get("maximum_daily_participation", 0.01))
    lookback_dates = research.trading_dates[max(0, decision_index - lookback + 1):decision_index + 1]
    capacity_weights = {}
    for symbol in risk_symbols:
        amounts = [
            float(row["amount"])
            for day in lookback_dates
            if (row := execution.get((symbol, day))) is not None
            and row["amount"] is not None
            and float(row["amount"]) > 0
        ]
        capacity_weights[symbol] = (
            statistics.fmean(amounts) * participation / portfolio.nav_cny
            if amounts else 0.0
        )

    buy_cost_rate = (
        float(config["commission_per_side_bps"])
        + float(config["transfer_fee_per_side_bps"])
        + float(config["slippage_per_side_bps"])
    ) / 10_000
    sell_cost_rate = buy_cost_rate + float(config["stamp_duty_sell_bps"]) / 10_000
    errors_by_symbol = {symbol: error for symbol in risk_symbols}
    result = optimize_portfolio(
        expected_returns,
        errors_by_symbol,
        risk_vols,
        current_weights,
        horizon_sessions=int(forecast["horizon_sessions"]),
        maximum_symbol_weight=float(config["maximum_symbol_weight"]),
        maximum_one_way_turnover=float(config["maximum_one_way_turnover"]),
        risk_budget_annualized=float(config.get("risk_budget_annualized", 0.20)),
        risk_aversion=float(config.get("risk_aversion", 8.0)),
        forecast_error_multiplier=float(config.get("forecast_error_multiplier", 0.5)),
        buy_cost_rate=buy_cost_rate,
        sell_cost_rate=sell_cost_rate,
        factor_betas=factor_betas,
        market_annualized_volatility=market_vol,
        max_change_weights=capacity_weights,
        maximum_target_weights=maximum_target_weights,
        iterations=int(config.get("optimizer_iterations", 400)),
    )
    return result, current_weights


def create_daily_decision(
    research: ResearchData,
    config: dict[str, float | int],
    as_of_date: date,
    portfolio: PortfolioState,
    training_state: dict[str, object] | None = None,
    versions: Mapping[str, str | None] | None = None,
    collaboration_context: Mapping[str, object] | None = None,
    harness_context: Mapping[str, object] | None = None,
    research_diagnostic: Mapping[str, object] | None = None,
) -> DailyDecision:
    """Create a pending, point-in-time decision from market data and account state."""
    if harness_context is not None and collaboration_context is not None:
        raise ValueError("the calibrated GEPA harness and historical contextual-RL controller are separate decision paths")
    portfolio.validate()
    diagnostic = (
        dict(research_diagnostic)
        if research_diagnostic is not None
        else run_mvp_research(
            research,
            config,
            initial_state=training_state,
            as_of_date=as_of_date,
            ensure_as_of_decision=True,
        )
    )
    forecast = diagnostic["prediction_decisions"][-1]
    if forecast["prediction_date"] != as_of_date.isoformat():
        raise ValueError(f"没有生成指定日期 {as_of_date} 的决策预测")
    version_info = {
        "config_sha256": hashlib.sha256(
            json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "baostock_manifest_sha256": _manifest_sha256(research),
        "training_seed_sha256": (
            hashlib.sha256(
                json.dumps(training_state, sort_keys=True, separators=(",", ":")).encode("utf-8")
            ).hexdigest()
            if training_state is not None else None
        ),
    }
    if versions:
        version_info.update(versions)
    selected_method = str(config.get("optimizer_return_method", "loss_gated_fusion"))
    optimization_config = config
    collaboration_trace = None
    harness_trace = None
    p9_optimization = None
    collaboration_warmup_reason = None
    if (
        collaboration_context is not None
        and forecast["forecast_errors_by_method"].get("market_plus_relative_fusion") is None
    ):
        selected_method = "market_plus_relative_fusion"
        collaboration_warmup_reason = (
            "固定 P9 预测误差尚未成熟，RL 专家权重暂无校准依据，暂不可决策"
        )
        collaboration_context = None
    if collaboration_context is not None:
        public_snapshot = collaboration_context.get("public_snapshot")
        reports = build_agent_reports(
            research,
            as_of_date,
            forecast,
            public_snapshot,
            config=config,
            enable_qwen=bool(collaboration_context.get("enable_qwen", False)),
        )
        fixed_p9_forecast = forecast
        forecast, overlay_trace = add_public_expert_overlay(forecast, public_snapshot)
        debate = one_round_challenge(reports)
        state = contextual_state(
            forecast,
            portfolio,
            float(config.get("risk_budget_annualized", 0.20)),
            research,
            as_of_date,
            int(config["risk_lookback_sessions"]),
            int(config["risk_minimum_observations"]),
        )
        query_available = (
            bool(collaboration_context.get("query_available", False))
            and bool(debate["conflicts"])
        )
        controller = choose_action(
            state["state"],
            dict(collaboration_context.get("q_values", {})),
            dict(collaboration_context.get("action_counts", {})),
            query_available=query_available,
            available_methods={
                name for name, error in forecast["forecast_errors_by_method"].items()
                if error is not None and name in forecast["expected_returns_by_method"]
            },
            minimum_samples=int(collaboration_context.get("minimum_action_samples", 20)),
            exploration_rate=float(collaboration_context.get("exploration_rate", 0.0)),
            seed=int(collaboration_context.get("seed", 0)),
        )
        if controller["supplementary_query"]:
            debate = run_supplementary_challenge(
                debate,
                reports,
                as_of_date,
                int(forecast["horizon_sessions"]),
            )
        forecast = apply_action(forecast, controller["weights"])
        selected_method = "rl_controlled_fusion"
        optimization_config = {
            **config,
            "optimizer_return_method": selected_method,
        }
        if forecast["forecast_errors_by_method"].get("market_plus_relative_fusion") is not None:
            p9_optimization, _ = optimize_forecast_target(
                research,
                fixed_p9_forecast,
                portfolio,
                {**config, "optimizer_return_method": "market_plus_relative_fusion"},
            )
        collaboration_trace = {
            "schema_version": "raama-collaboration.v1",
            "agent_reports": reports,
            "effective_decision_time": reports["research_inputs"]["effective_decision_time"],
            "one_round_bull_bear_challenge": debate,
            "public_expert_overlay": overlay_trace,
            "p11_memory_summary": (public_snapshot or {}).get("p11_memory_summary"),
            "rl_controller": {
                **controller,
                "context": state,
                "data_split": str(collaboration_context.get("data_split", "training")),
                "exploration_rate": float(collaboration_context.get("exploration_rate", 0.0)),
                "training_data_source": "mature feedback rows emitted by scripts/replay_contextual_rl_feedback.py from executed decision rows and account NAV replay",
                "feedback_used_this_decision": int(collaboration_context.get("matured_feedback_used", 0)),
                "training_evaluation_separation": "evaluation feedback is excluded from policy updates",
                "supplementary_query_policy": (
                    "disabled until cited challenge evidence has a predeclared calibrated numeric effect on candidates or weights"
                ),
            },
            "numeric_signal_source": "mature calibrated returns only",
            "optimizer_return_method": selected_method,
            "forecast_error_allowance_rule": "maximum matured component forecast error in selected calibrated blend",
        }

    if harness_context is not None:
        harness_trace = dict(harness_context)
        harness_research = harness_context["research"]
        if harness_research["status"] == "completed":
            task_usage = harness_research.get(
                "receiving_task_usage", harness_research["task_usage"]
            )
            task_cost_source = harness_context["research_cost"]
            harness_trace["research_cost_accounting"] = {
                "logical_research_cost_cny": float(task_usage["logical_estimated_cost_cny"]),
                "actual_incremental_api_cost_cny": float(task_usage["api_incremental_cost_cny"]),
                "source_task_api_cost_cny": float(task_usage.get("source_task_api_incremental_cost_cny", task_usage["api_incremental_cost_cny"])),
                "cost_source": str(task_cost_source["cost_source"]),
                "research_result_reused": bool(task_usage.get("research_result_reused_from_artifact", False)),
                "logical_cost_counted_once": True,
                "logical_cost_basis": "task token usage at planning rates; exact-response cache reuse leaves comparison cost unchanged",
                "api_cost_basis": "new provider requests in this task only; cache reuse costs zero incremental API spend",
            }
            manager_priorities = harness_research["agents"]["research_manager"]["report"]["priority_symbols"]
            calibrated_universe = forecast["expected_returns_by_method"][selected_method]
            priority_symbols = list(dict.fromkeys(
                str(row["symbol"])
                for row in manager_priorities
                if str(row["symbol"]) in calibrated_universe
            ))
            optimization_config = {
                **config,
                "optimizer_candidate_scope": "harness_priority",
                "optimizer_priority_symbols": priority_symbols,
            }
            harness_trace["candidate_filter"] = {
                "method": "manager-ranked sourced symbols intersected with calibrated forecast universe",
                "priority_symbols": priority_symbols,
                "empty_candidate_set_allows_cash": True,
                "expected_returns_source": f"unchanged {selected_method} calibrated forecast",
            }

    if forecast["forecast_errors_by_method"].get(selected_method) is None:
        execution_index = research.trading_dates.index(as_of_date) + 1
        execution_date = (
            research.trading_dates[execution_index].isoformat()
            if execution_index < len(research.trading_dates) else None
        )
        if harness_trace is not None and "research_cost_accounting" in harness_trace:
            harness_trace["decision_cost_accounting"] = {
                **harness_trace["research_cost_accounting"],
                "new_api_cost_cny": harness_trace["research_cost_accounting"]["actual_incremental_api_cost_cny"],
                "estimated_transaction_cost_cny": 0.0,
                "estimated_total_decision_cost_cny": harness_trace["research_cost_accounting"]["logical_research_cost_cny"],
            }
        return DailyDecision(
            schema_version="adaptive-mas.decision.v1",
            as_of_date=as_of_date.isoformat(),
            execution_date=execution_date,
            status="warmup_no_action",
            prediction_status="pending",
            prediction_horizon_sessions=int(forecast["horizon_sessions"]),
            predictions_by_symbol=_predictions_by_symbol(forecast, selected_method),
            current_nav_cny=portfolio.nav_cny,
            current_cash_cny=portfolio.cash_cny,
            current_holdings_cny=dict(sorted(portfolio.holdings_cny.items())),
            target_cash_cny=portfolio.cash_cny,
            target_holdings_cny=dict(sorted(portfolio.holdings_cny.items())),
            target_weights={
                symbol: value / portfolio.nav_cny
                for symbol, value in portfolio.holdings_cny.items()
            },
            estimated_cash_after_costs_cny=portfolio.cash_cny,
            buy_notional_cny=0.0,
            sell_notional_cny=0.0,
            orders=(),
            expected_gross_return=None,
            forecast_error_allowance=None,
            expected_transaction_cost_cny=0.0,
            current_holding_objective_value_cny=None,
            optimized_objective_value_cny=None,
            risk_adjusted_net_incremental_value_cny=0.0,
            risk_penalty_cny=None,
            annualized_risk=None,
            annualized_risk_budget=float(config.get("risk_budget_annualized", 0.20)),
            one_way_turnover=0.0,
            evidence={
                "input_cutoff": as_of_date.isoformat(),
                "selected_optimizer_return_method": selected_method,
                "forecast_error_status": forecast["forecast_error_status"],
                "reason": collaboration_warmup_reason or "等待成熟预测误差后才允许优化并生成订单",
                "collaboration_status": (
                    "warmup_unavailable_fixed_p9_calibration"
                    if collaboration_warmup_reason else None
                ),
                "matured_rank_ic_windows": diagnostic["matured_rank_ic_windows_at_end"],
                "ridge_training_rows": diagnostic["training_rows_at_end"],
                "expert_matured_performance": _expert_matured_performance(
                    diagnostic["training_seed"]["matured_rank_ics"],
                    forecast["forecast_errors_by_method"],
                    forecast["loss_gate_weights"],
                    int(config["reliability_lookback_windows"]),
                ),
                "harness": harness_trace,
            },
            versions=version_info,
        )
    optimization, _ = optimize_forecast_target(
        research, forecast, portfolio, optimization_config
    )
    target_values = {
        symbol: weight * portfolio.nav_cny
        for symbol, weight in optimization.target_weights.items()
    }
    target_changes = {
        symbol: float(target_values.get(symbol, 0.0))
        - float(portfolio.holdings_cny.get(symbol, 0.0))
        for symbol in set(portfolio.holdings_cny) | set(target_values)
    }
    buy_notional = math.fsum(max(change, 0.0) for change in target_changes.values())
    sell_notional = math.fsum(max(-change, 0.0) for change in target_changes.values())
    expected_returns = forecast["expected_returns_by_method"][selected_method]
    error = float(forecast["forecast_errors_by_method"][selected_method])
    error_allowance_per_name = (
        float(config.get("forecast_error_multiplier", 0.5)) * error
    )
    buy_cost_rate = (
        float(config["commission_per_side_bps"])
        + float(config["transfer_fee_per_side_bps"])
        + float(config["slippage_per_side_bps"])
    ) / 10_000
    sell_cost_rate = buy_cost_rate + float(config["stamp_duty_sell_bps"]) / 10_000
    decision_day_execution = {
        str(row["symbol"]): row
        for row in research.execution_bars
        if row["trade_date"] == as_of_date
    }
    prediction_rows = _predictions_by_symbol(forecast, selected_method)
    proposed_orders = []
    for symbol in sorted(set(portfolio.holdings_cny) | set(target_values)):
        current_value = float(portfolio.holdings_cny.get(symbol, 0.0))
        target_value = float(target_values.get(symbol, 0.0))
        change = target_value - current_value
        if abs(change) <= 0.01:
            continue
        side = "buy" if change > 0 else "sell"
        cost_rate = buy_cost_rate if side == "buy" else sell_cost_rate
        status_row = decision_day_execution.get(symbol)
        decision_status = (
            "unknown"
            if status_row is None
            else str(status_row.get("tradestatus", "unknown"))
        )
        proposed_orders.append(
            DecisionOrder(
                symbol=symbol,
                side=side,
                notional_cny=abs(change),
                current_value_cny=current_value,
                target_value_cny=target_value,
                expected_return=float(expected_returns.get(symbol, 0.0)),
                forecast_error=error,
                expected_incremental_value_cny=(
                    change
                    * (float(expected_returns.get(symbol, 0.0)) - error_allowance_per_name)
                    - abs(change) * cost_rate
                ),
                estimated_cost_cny=abs(change) * cost_rate,
                estimated_liquidity_capacity_cny=(
                    float(optimization.capacity_weights.get(symbol, 0.0))
                    * portfolio.nav_cny
                ),
                decision_date_tradestatus=decision_status,
                next_session_execution_uncertain=True,
                execution_note="次交易日状态与实际成交量须在执行时由行情或模拟账本确认",
                status="pending_next_session",
                source_prediction_id=str(prediction_rows[symbol]["prediction_id"]),
                prediction_date=str(prediction_rows[symbol]["prediction_date"]),
                prediction_horizon_sessions=int(prediction_rows[symbol]["horizon_sessions"]),
            )
        )
    execution_index = research.trading_dates.index(as_of_date) + 1
    execution_date = (
        research.trading_dates[execution_index].isoformat()
        if execution_index < len(research.trading_dates) else None
    )
    decision = DailyDecision(
        schema_version="adaptive-mas.decision.v1",
        as_of_date=as_of_date.isoformat(),
        execution_date=execution_date,
        status="pending_execution" if proposed_orders else "hold",
        prediction_status="pending",
        prediction_horizon_sessions=int(forecast["horizon_sessions"]),
        predictions_by_symbol=prediction_rows,
        current_nav_cny=portfolio.nav_cny,
        current_cash_cny=portfolio.cash_cny,
        current_holdings_cny=dict(sorted(portfolio.holdings_cny.items())),
        target_cash_cny=portfolio.nav_cny * (1 - math.fsum(optimization.target_weights.values())),
        target_holdings_cny=dict(sorted(target_values.items())),
        target_weights=optimization.target_weights,
        estimated_cash_after_costs_cny=portfolio.nav_cny * (
            1
            - math.fsum(optimization.target_weights.values())
            - optimization.expected_transaction_cost
        ),
        buy_notional_cny=buy_notional,
        sell_notional_cny=sell_notional,
        orders=tuple(proposed_orders),
        expected_gross_return=optimization.gross_expected_return,
        forecast_error_allowance=optimization.forecast_error_allowance,
        expected_transaction_cost_cny=optimization.expected_transaction_cost * portfolio.nav_cny,
        current_holding_objective_value_cny=(
            optimization.holding_objective_value * portfolio.nav_cny
        ),
        optimized_objective_value_cny=optimization.objective_value * portfolio.nav_cny,
        risk_adjusted_net_incremental_value_cny=(
            optimization.incremental_objective_value * portfolio.nav_cny
        ),
        risk_penalty_cny=optimization.risk_penalty * portfolio.nav_cny,
        annualized_risk=optimization.annualized_risk,
        annualized_risk_budget=optimization.risk_budget_annualized,
        one_way_turnover=optimization.maximum_one_way_turnover,
        evidence={
            "input_cutoff": as_of_date.isoformat(),
            "selected_optimizer_return_method": selected_method,
            "forecast_errors_by_method": forecast["forecast_errors_by_method"],
            "loss_gate_weights": forecast["loss_gate_weights"],
            "loss_gate_source": forecast["loss_gate_source"],
            "calibration": forecast["calibration"],
            "matured_rank_ic_windows": diagnostic["matured_rank_ic_windows_at_end"],
            "ridge_training_rows": diagnostic["training_rows_at_end"],
            "expert_matured_performance": _expert_matured_performance(
                diagnostic["training_seed"]["matured_rank_ics"],
                forecast["forecast_errors_by_method"],
                forecast["loss_gate_weights"],
                int(config["reliability_lookback_windows"]),
            ),
            "optimizer_objective_value": optimization.objective_value,
            "optimizer_holding_objective_value": optimization.holding_objective_value,
            "risk_adjusted_net_incremental_value_cny": (
                optimization.incremental_objective_value * portfolio.nav_cny
            ),
            "objective_includes_forecast_error_costs_and_portfolio_risk": True,
            "risk_model": {
                "market_annualized_volatility": forecast["market_annualized_volatility"],
                "annualized_volatility_by_symbol": forecast["annualized_volatility_by_symbol"],
                "factor_beta_by_symbol": forecast["factor_beta_by_symbol"],
            },
            "prediction_record_status": forecast["status"],
            "harness": harness_trace,
        },
        versions=version_info,
    )
    decision, hard_risk_reviews, hard_risk_outcome = _hard_risk_review(
        decision, config
    )
    if collaboration_trace is not None:
        decision_payload = decision.to_dict()
        collaboration_trace["fixed_p9_comparison"] = fixed_p9_comparison(
            decision, p9_optimization
        )
        research_manager_view = research_manager(
            forecast,
            selected_method,
            collaboration_trace["agent_reports"],
            float(config.get("forecast_error_multiplier", 0.5)),
        )
        collaboration_trace["research_manager"] = research_manager_view
        collaboration_trace["risk_role_reviews"] = hard_risk_reviews
        collaboration_trace["hard_risk_outcome"] = hard_risk_outcome
        investment_feedback = investment_manager_explanation(
            decision_payload,
            research_manager_view,
        )
        investment_feedback["risk_review_status"] = hard_risk_outcome["status"]
        investment_feedback["risk_failed_checks"] = hard_risk_outcome["failed_checks"]
        investment_feedback["risk_orders_blocked"] = hard_risk_outcome["orders_blocked"]
        investment_feedback["risk_actual_hold_state"] = hard_risk_outcome.get("actual_hold_state")
        investment_feedback["risk_unresolved"] = bool(
            hard_risk_outcome.get("actual_hold_state", {}).get("risk_within_budget") is False
        )
        collaboration_trace["investment_manager"] = investment_feedback
        query_cost = float(
            collaboration_trace["one_round_bull_bear_challenge"].get(
                "supplementary_query_cost_cny", 0.0
            )
        )
        role_qwen_cost = math.fsum(
            float(collaboration_trace["agent_reports"][role].get("estimated_token_cost_cny") or 0.0)
            for role in ("market", "fundamental", "news", "market_attention")
        )
        analysis_cost = role_qwen_cost + query_cost
        collaboration_trace["decision_cost_accounting"] = {
            "estimated_transaction_cost_cny": decision.expected_transaction_cost_cny,
            "role_qwen_cost_cny": role_qwen_cost,
            "supplementary_query_cost_cny": query_cost,
            "analysis_cost_cny": analysis_cost,
            "estimated_total_decision_cost_cny": decision.expected_transaction_cost_cny + analysis_cost,
            "execution_cost_source": "optimizer estimate; realized fills and fees enter account NAV feedback",
            "analysis_cost_source": "sum of role Qwen token estimates and cited-challenge estimate; deducted once from feedback reward",
        }
        decision = replace(
            decision,
            evidence={
                **decision.evidence,
                "selected_optimizer_return_method": selected_method,
                "collaborative_architecture": collaboration_trace,
                "hard_risk_outcome": hard_risk_outcome,
                "risk_role_reviews": hard_risk_reviews,
                "investment_manager_feedback": investment_feedback,
            },
        )
    if harness_trace is not None:
        harness_research = harness_trace["research"]
        manager_report = harness_research["agents"]["research_manager"]["report"]
        investment_feedback = investment_manager_explanation(decision.to_dict())
        investment_feedback["research_priority_symbols"] = [
            row["symbol"] for row in manager_report["priority_symbols"]
        ]
        investment_feedback["risk_research"] = [
            {
                "dimension": item.get("dimension"),
                "status": item.get("status"),
                "symbol": item.get("symbol"),
                "question": item.get("question"),
                "summary": item.get("report", {}).get("summary") if isinstance(item.get("report"), Mapping) else None,
                "source_evidence_ids": list(item.get("source_evidence_ids", [])),
                "source_record_ids": list(item.get("report", {}).get("source_record_ids", [])) if isinstance(item.get("report"), Mapping) else [],
                "findings": [
                    {
                        "conclusion": finding.get("conclusion"),
                        "evidence_ids": list(finding.get("evidence_ids", [])),
                        "observations": list(finding.get("observations", [])),
                    }
                    for finding in item.get("report", {}).get("findings", [])
                    if isinstance(finding, Mapping)
                ] if isinstance(item.get("report"), Mapping) else [],
            }
            for item in manager_report.get("risk_research", [])
            if isinstance(item, Mapping)
        ]
        investment_feedback["risk_review_status"] = hard_risk_outcome["status"]
        investment_feedback["risk_failed_checks"] = hard_risk_outcome["failed_checks"]
        investment_feedback["risk_orders_blocked"] = hard_risk_outcome["orders_blocked"]
        investment_feedback["risk_actual_hold_state"] = hard_risk_outcome.get("actual_hold_state")
        investment_feedback["risk_unresolved"] = bool(
            hard_risk_outcome.get("actual_hold_state", {}).get("risk_within_budget") is False
        )
        harness_trace["risk_role_reviews"] = hard_risk_reviews
        harness_trace["hard_risk_outcome"] = hard_risk_outcome
        harness_trace["investment_manager_feedback"] = investment_feedback
        decision = replace(
            decision,
            evidence={
                **decision.evidence,
                "harness": harness_trace,
                "hard_risk_outcome": hard_risk_outcome,
                "risk_role_reviews": hard_risk_reviews,
                "investment_manager_feedback": investment_feedback,
            },
        )
    if harness_trace is None and collaboration_trace is None:
        decision = replace(
            decision,
            evidence={
                **decision.evidence,
                "hard_risk_outcome": hard_risk_outcome,
                "risk_role_reviews": hard_risk_reviews,
            },
        )
    if harness_trace is not None and "research_cost_accounting" in harness_trace:
        harness_trace["decision_cost_accounting"] = {
            **harness_trace["research_cost_accounting"],
            "new_api_cost_cny": harness_trace["research_cost_accounting"]["actual_incremental_api_cost_cny"],
            "estimated_transaction_cost_cny": decision.expected_transaction_cost_cny,
            "estimated_total_decision_cost_cny": (
                decision.expected_transaction_cost_cny
                + harness_trace["research_cost_accounting"]["logical_research_cost_cny"]
            ),
            "nav_accounting": "deduct this task's logical research cost once when updating realized account NAV",
        }
        decision = replace(
            decision,
            evidence={**decision.evidence, "harness": harness_trace},
        )
    return decision


def _expert_matured_performance(
    matured_rank_ics: list[dict[str, float]],
    forecast_errors: Mapping[str, object],
    weights: Mapping[str, float],
    window: int,
) -> dict[str, dict[str, object]]:
    recent = matured_rank_ics[-window:]
    return {
        agent_id: {
            "loss_gate_weight": float(weights.get(agent_id, 0.0)),
            "matured_rank_ic_windows": sum(agent_id in row for row in recent),
            "mean_matured_rank_ic": (
                statistics.fmean(float(row[agent_id]) for row in recent if agent_id in row)
                if any(agent_id in row for row in recent) else None
            ),
            "rolling_matured_oos_rmse": (
                float(forecast_errors[agent_id])
                if forecast_errors.get(agent_id) is not None else None
            ),
        }
        for agent_id in ("trend", "reversal", "liquidity_confirmation")
    }


def _predictions_by_symbol(forecast: dict[str, object], method: str = "loss_gated_fusion") -> dict[str, dict[str, object]]:
    methods = forecast["expected_returns_by_method"]
    errors = forecast["forecast_errors_by_method"]
    prediction_date = str(forecast["prediction_date"])
    horizon_sessions = int(forecast["horizon_sessions"])
    available_at = f"{prediction_date}T15:00:00+08:00"
    symbols = sorted(methods[method])
    return {
        symbol: {
            "expected_return": float(methods[method][symbol]),
            "prediction_id": _prediction_identifier(
                prediction_date,
                symbol,
                horizon_sessions,
                method,
                float(methods[method][symbol]),
                errors.get(method),
            ),
            "prediction_date": prediction_date,
            "horizon_sessions": horizon_sessions,
            "available_at": available_at,
            "forecast_error": errors.get(method),
            "expert_expected_returns": {
                agent_id: float(methods[agent_id][symbol])
                for agent_id in ("trend", "reversal", "liquidity_confirmation")
                if symbol in methods[agent_id]
            },
            "loss_gate_weights": forecast["loss_gate_weights"],
            "status": forecast["status"],
        }
        for symbol in symbols
    }


def _prediction_identifier(
    prediction_date: str,
    symbol: str,
    horizon_sessions: int,
    method: str,
    expected_return: float,
    forecast_error: object,
) -> str:
    identity = {
        "prediction_date": prediction_date,
        "symbol": symbol,
        "horizon_sessions": horizon_sessions,
        "method": method,
        "expected_return": expected_return,
        "forecast_error": forecast_error,
    }
    canonical = json.dumps(identity, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return f"pred-{hashlib.sha256(canonical.encode('utf-8')).hexdigest()[:24]}"


def _manifest_sha256(research: ResearchData) -> str | None:
    source_refs = [
        str(row.get("source_ref", ""))
        for row in research.index_bars[:1]
    ] + [
        snapshot.source_ref for snapshot in research.membership_snapshots[:1]
    ]
    for source_ref in source_refs:
        source = Path(source_ref)
        if source.suffix.lower() != ".csv":
            continue
        manifest = source.parent.parent / "manifest.jsonl"
        if manifest.is_file():
            return hashlib.sha256(manifest.read_bytes()).hexdigest()
    return None
