"""Daily notional-account simulation using BaoStock execution status."""

import math
import statistics
from collections import defaultdict
from datetime import date
from typing import Callable

from adaptive_mas.data.research import ResearchData


def _cash_after_fill(cash_cny: float, *, trade_date: date, symbol: str) -> tuple[float, float]:
    """Clamp only sub-cent arithmetic noise; fail on a real account deficit."""
    if not math.isfinite(cash_cny):
        raise ValueError(f"成交后现金非有限值：{trade_date} {symbol} cash_cny={cash_cny!r}")
    if cash_cny < -1e-8:
        raise ValueError(f"买入成交超出可用现金：{trade_date} {symbol} cash_cny={cash_cny:.17g}")
    return (0.0, -cash_cny) if cash_cny < 0.0 else (cash_cny, 0.0)


def simulate_fixed_risk_market_exposure(
    research: ResearchData,
    evaluation_start_date: date,
    initial_capital_cny: float,
    risk_budget_annualized: float,
    risk_lookback_sessions: int,
    costs_bps: dict[str, float],
) -> dict[str, object]:
    """Price-only, fixed CSI 300 exposure set from trailing risk, with no return forecast."""
    if evaluation_start_date not in research.trading_dates:
        raise ValueError(f"固定风险市场基线起始日不在交易日历中：{evaluation_start_date}")
    if initial_capital_cny <= 0 or risk_budget_annualized <= 0 or risk_lookback_sessions < 2:
        raise ValueError("固定风险市场基线的资金、风险预算和回看窗口必须为正")
    start_index = research.trading_dates.index(evaluation_start_date)
    first_mark_index = start_index + 1
    if first_mark_index >= len(research.trading_dates):
        raise ValueError("固定风险市场基线起始日之后没有成交日")
    index_by_date = {row["trade_date"]: row for row in research.index_bars}
    history_dates = research.trading_dates[
        max(1, start_index - risk_lookback_sessions + 1) : start_index + 1
    ]
    market_history = []
    for day in history_dates:
        row = index_by_date.get(day)
        prior_index = research.trading_dates.index(day) - 1
        prior = index_by_date.get(research.trading_dates[prior_index])
        if row is None:
            continue
        if row.get("return_1d") is not None:
            market_history.append(float(row["return_1d"]) / 100)
        elif prior is not None and prior.get("close") and row.get("close"):
            market_history.append(float(row["close"]) / float(prior["close"]) - 1)
    if len(market_history) < 2:
        raise ValueError("固定风险市场基线缺少已成熟的沪深300风险历史")
    annualized_volatility = statistics.pstdev(market_history) * math.sqrt(250)
    if annualized_volatility <= 0:
        raise ValueError("固定风险市场基线的历史波动率为零，无法确定风险暴露")
    target_weight = min(1.0, risk_budget_annualized / annualized_volatility)

    buy_rate = math.fsum(
        float(costs_bps[name])
        for name in ("commission_per_side_bps", "transfer_fee_per_side_bps", "slippage_per_side_bps")
    ) / 10_000
    sell_rate = buy_rate + float(costs_bps["stamp_duty_sell_bps"]) / 10_000
    first_date = research.trading_dates[first_mark_index]
    first_bar = index_by_date.get(first_date)
    if first_bar is None or not first_bar.get("open") or not first_bar.get("close"):
        raise ValueError(f"沪深300指数缺少固定风险基线的首次执行价格：{first_date}")
    initial_notional = initial_capital_cny * target_weight
    units = initial_notional / float(first_bar["open"])
    buy_cost = initial_notional * buy_rate
    cash = initial_capital_cny - initial_notional - buy_cost
    previous_nav = initial_capital_cny
    benchmark_nav = initial_capital_cny
    daily_rows = []
    position_rows = []
    orders = [
        {
            "trade_date": first_date.isoformat(),
            "side": "buy",
            "notional_cny": initial_notional,
            "cost_cny": buy_cost,
            "status": "filled_index_price_proxy",
        }
    ]
    total_cost = buy_cost
    total_traded = initial_notional
    peak = initial_capital_cny
    maximum_drawdown = 0.0
    for day_index in range(first_mark_index, len(research.trading_dates)):
        day = research.trading_dates[day_index]
        row = index_by_date.get(day)
        if row is None or not row.get("close"):
            raise ValueError(f"沪深300指数缺少固定风险基线的收盘价：{day}")
        if day_index == first_mark_index:
            benchmark_return = float(row["close"]) / float(row["open"]) - 1
        else:
            prior = index_by_date.get(research.trading_dates[day_index - 1])
            if prior is None or not prior.get("close"):
                raise ValueError(f"沪深300指数缺少基准前收：{day}")
            benchmark_return = float(row["close"]) / float(prior["close"]) - 1
        benchmark_nav *= 1 + benchmark_return
        market_value = units * float(row["close"])
        daily_cost = buy_cost if day_index == first_mark_index else 0.0
        if day_index == len(research.trading_dates) - 1:
            sell_notional = market_value
            sell_cost = sell_notional * sell_rate
            cash += sell_notional - sell_cost
            market_value = 0.0
            total_cost += sell_cost
            total_traded += sell_notional
            daily_cost += sell_cost
            orders.append(
                {
                    "trade_date": day.isoformat(),
                    "side": "sell",
                    "notional_cny": sell_notional,
                    "cost_cny": sell_cost,
                    "status": "filled_index_price_proxy",
                }
            )
        nav = cash + market_value
        daily_return = nav / previous_nav - 1
        peak = max(peak, nav)
        maximum_drawdown = min(maximum_drawdown, nav / peak - 1)
        daily_rows.append(
            {
                "trade_date": day.isoformat(),
                "nav_cny": nav,
                "daily_return": daily_return,
                "benchmark_return": benchmark_return,
                "benchmark_nav_cny": benchmark_nav,
                "cash_cny": cash,
                "market_value_cny": market_value,
                "cash_weight": cash / nav if nav else 1.0,
                "traded_notional_cny": initial_notional if day_index == first_mark_index else (
                    total_traded - initial_notional if day_index == len(research.trading_dates) - 1 else 0.0
                ),
                "cost_cny": daily_cost,
            }
        )
        if market_value > 0:
            position_rows.append(
                {
                    "trade_date": day.isoformat(),
                    "symbol": "sh.000300",
                    "market_value_cny": market_value,
                    "weight": market_value / nav if nav else 0.0,
                    "status": "index_price_proxy",
                }
            )
        previous_nav = nav
    returns = [float(row["daily_return"]) for row in daily_rows]
    benchmark_returns = [float(row["benchmark_return"]) for row in daily_rows]
    final_nav = float(daily_rows[-1]["nav_cny"])
    summary = {
        "baseline_type": "price_only_fixed_initial_risk_exposure",
        "final_nav_cny": final_nav,
        "net_total_return": final_nav / initial_capital_cny - 1,
        "annualized_return": (final_nav / initial_capital_cny) ** (250 / len(returns)) - 1,
        "annualized_volatility": statistics.stdev(returns) * math.sqrt(250) if len(returns) > 1 else 0.0,
        "maximum_drawdown": maximum_drawdown,
        "average_cash_weight": statistics.fmean(float(row["cash_weight"]) for row in daily_rows),
        "total_cost_cny": total_cost,
        "total_traded_notional_cny": total_traded,
        "one_way_turnover": total_traded / (2 * initial_capital_cny),
        "target_risk_budget_annualized": risk_budget_annualized,
        "formation_volatility_annualized": annualized_volatility,
        "target_market_weight": target_weight,
        "index_execution_assumption": "synthetic index-price fills at next-session open and final close; no ETF tracking error or capacity model",
        "cost_assumptions_bps": costs_bps,
    }
    return {
        "summary": summary,
        "daily_nav": daily_rows,
        "orders": orders,
        "positions": position_rows,
        "decisions": [
            {
                "decision_date": evaluation_start_date.isoformat(),
                "execution_date": first_date.isoformat(),
                "target_market_weight": target_weight,
                "formation_volatility_annualized": annualized_volatility,
                "target_risk_budget_annualized": risk_budget_annualized,
            }
        ],
    }


def simulate_portfolio(
    research: ResearchData,
    decisions: list[dict[str, object]],
    initial_capital_cny: float,
    costs_bps: dict[str, float],
    liquidity_lookback_sessions: int,
    maximum_daily_participation: float,
    maximum_one_way_turnover: float = 1.0,
    forecast_decisions: list[dict[str, object]] | None = None,
    optimizer_config: dict[str, float | int] | None = None,
    progress_callback: Callable[[str], None] | None = None,
    evaluation_start_date: date | None = None,
    forecast_controller: Callable[..., tuple[dict[str, object], dict[str, object]]] | None = None,
) -> dict[str, object]:
    from adaptive_mas.decision import PortfolioState, optimize_forecast_target
    from adaptive_mas.portfolio.optimizer import PortfolioInfeasibleError
    from adaptive_mas.evaluation.mvp_research import _risk_estimates

    def choose_optimization(forecast, account):
        selection = optimizer_config.get("optimizer_return_method")
        controller_action = forecast.get("controller_action")
        controller_return_method = forecast.get("controller_return_method")
        if controller_return_method is not None:
            action_config = {
                **optimizer_config,
                "optimizer_return_method": str(controller_return_method),
            }
            optimization, _ = optimize_forecast_target(
                research, forecast, account, action_config
            )
            return optimization, {"selected_return_method": str(controller_return_method)}
        if controller_action is not None:
            if controller_action not in {"trend", "market_plus_relative_fusion"}:
                raise ValueError(f"不支持的控制器动作：{controller_action}")
            action_config = {
                **optimizer_config,
                "optimizer_return_method": str(controller_action),
            }
            optimization, _ = optimize_forecast_target(
                research, forecast, account, action_config
            )
            return optimization, {"selected_return_method": str(controller_action)}
        if selection != "p9_best_trend_or_market_relative":
            optimization, _ = optimize_forecast_target(
                research, forecast, account, optimizer_config
            )
            return optimization, None
        alternatives = {}
        for method in ("trend", "market_plus_relative_fusion"):
            method_config = {**optimizer_config, "optimizer_return_method": method}
            try:
                alternative, _ = optimize_forecast_target(
                    research, forecast, account, method_config
                )
            except PortfolioInfeasibleError as error:
                alternatives[method] = {"objective_value": None, "infeasible_reason": str(error)}
                continue
            alternatives[method] = {"objective_value": alternative.objective_value,
                                    "optimization": alternative}
        feasible = [
            (method, values["optimization"])
            for method, values in alternatives.items()
            if "optimization" in values
        ]
        if not feasible:
            raise PortfolioInfeasibleError(
                f"{forecast['prediction_date']} 趋势和市场相对收益目标均不可行："
                f"{alternatives}"
            )
        selected_method, optimization = max(
            feasible, key=lambda item: item[1].objective_value
        )
        evidence = {
            "selected_return_method": selected_method,
            "candidate_objective_values": {
                method: values["objective_value"] for method, values in alternatives.items()
            },
            "candidate_infeasible_reasons": {
                method: values["infeasible_reason"]
                for method, values in alternatives.items()
                if "infeasible_reason" in values
            },
        }
        return optimization, evidence

    adjusted = {
        (str(row["symbol"]), row["trade_date"]): row for row in research.adjusted_bars
    }
    execution = {
        (str(row["symbol"]), row["trade_date"]): row for row in research.execution_bars
    }
    index_bars = {row["trade_date"]: row for row in research.index_bars}
    targets_by_date: dict[date, dict[str, object]] = {}
    forecasts_by_date = {}
    decision_rows = []
    daily_rows = []
    dynamic = forecast_decisions is not None
    if dynamic:
        if decisions or optimizer_config is None:
            raise ValueError("逐日优化模拟必须提供预测记录和唯一的优化配置")
        forecasts_by_date = {
            date.fromisoformat(str(row["prediction_date"])): row
            for row in forecast_decisions
        }
        if not forecasts_by_date:
            raise ValueError("逐日优化模拟没有可执行预测记录")
    else:
        if forecast_controller is not None:
            raise ValueError("逐日控制器只能用于预测记录驱动的动态模拟")
        for decision in decisions:
            decision_date = decision["decision_date"]
            decision_index = research.trading_dates.index(decision_date)
            if decision_index + 1 >= len(research.trading_dates):
                continue
            execution_date = research.trading_dates[decision_index + 1]
            targets_by_date[execution_date] = decision

    def decision_value_fields(optimization, state, horizon_sessions):
        current_weights = {
            symbol: value / state.nav_cny
            for symbol, value in state.holdings_cny.items()
        }
        changes = {
            symbol: optimization.target_weights.get(symbol, 0.0)
            - current_weights.get(symbol, 0.0)
            for symbol in set(optimization.target_weights) | set(current_weights)
        }
        horizon_sigma = optimization.annualized_risk * math.sqrt(horizon_sessions / 250)
        return {
            "target_cash_cny_before_costs": state.nav_cny * (
                1 - math.fsum(optimization.target_weights.values())
            ),
            "estimated_cash_after_costs_cny": state.nav_cny * (
                1
                - math.fsum(optimization.target_weights.values())
                - optimization.expected_transaction_cost
            ),
            "buy_notional_cny": state.nav_cny
            * math.fsum(max(change, 0.0) for change in changes.values()),
            "sell_notional_cny": state.nav_cny
            * math.fsum(max(-change, 0.0) for change in changes.values()),
            "holding_objective_value": optimization.holding_objective_value,
            "incremental_objective_value": optimization.incremental_objective_value,
            "risk_penalty_weight": optimization.risk_penalty,
            "portfolio_horizon_sigma": horizon_sigma,
            "forecast_error_allowance_over_horizon_sigma": (
                optimization.forecast_error_allowance / horizon_sigma
                if horizon_sigma > 0 else None
            ),
            "risk_adjusted_net_incremental_value_cny": (
                optimization.incremental_objective_value * state.nav_cny
            ),
        }

    def prepare_forecast(forecast, account, decision_date, decision_pause=None):
        if forecast_controller is None:
            return forecast, {}
        controlled, selection = forecast_controller(
            forecast=forecast,
            account=account,
            decision_date=decision_date,
            daily_nav=tuple(daily_rows),
            decisions=tuple(decision_rows),
            decision_pause=decision_pause,
        )
        return controlled, selection

    def untradeable_risk_pause(account, decision_date):
        held_symbols = set(account.holdings_cny)
        if not held_symbols:
            return None
        decision_index = research.trading_dates.index(decision_date)
        lookback = int(optimizer_config["risk_lookback_sessions"])
        minimum = int(optimizer_config["risk_minimum_observations"])
        volatility, _, _ = _risk_estimates(
            research, decision_index, held_symbols, lookback, minimum
        )
        missing_risk = sorted(held_symbols - set(volatility))
        if not missing_risk:
            return None
        history_dates = research.trading_dates[
            max(0, decision_index - lookback + 1):decision_index + 1
        ]
        paused_symbols = []
        for symbol in missing_risk:
            observations = sum(
                (bar := adjusted.get((symbol, day))) is not None
                and bar["return_1d"] is not None
                for day in history_dates
            )
            bar = execution.get((symbol, decision_date))
            execution_status = None if bar is None else str(bar["tradestatus"])
            untradeable = (
                bar is None
                or execution_status != "1"
                or bar["open"] is None
            )
            if observations >= minimum or not untradeable:
                return None
            paused_symbols.append({
                "symbol": symbol,
                "valid_return_observations": observations,
                "minimum_required_observations": minimum,
                "execution_status": execution_status,
            })
        return {
            "reason": "held_symbol_risk_below_minimum_and_untradeable",
            "symbols": paused_symbols,
        }

    if dynamic:
        first_decision_date = min(forecasts_by_date)
        first_decision_index = research.trading_dates.index(first_decision_date)
        if first_decision_index + 1 >= len(research.trading_dates):
            raise ValueError("逐日优化模拟没有可执行的下一交易日")
        first_execution_date = research.trading_dates[first_decision_index + 1]
        initial_state = PortfolioState(initial_capital_cny, {})
        first_forecast, first_controller_selection = prepare_forecast(
            forecasts_by_date[first_decision_date], initial_state, first_decision_date
        )
        initial_optimization, initial_selection = choose_optimization(
            first_forecast, initial_state
        )
        initial_record = {
            "decision_date": first_decision_date.isoformat(),
            "execution_date": first_execution_date.isoformat(),
            "current_cash_cny": initial_state.cash_cny,
            "current_holdings_cny": {},
            "current_nav_cny": initial_state.nav_cny,
            "target_weights": initial_optimization.target_weights,
            "gross_expected_return": initial_optimization.gross_expected_return,
            "forecast_error_allowance": initial_optimization.forecast_error_allowance,
            "expected_transaction_cost_weight": initial_optimization.expected_transaction_cost,
            "annualized_risk": initial_optimization.annualized_risk,
            "one_way_turnover": initial_optimization.maximum_one_way_turnover,
            "objective_value": initial_optimization.objective_value,
            "controller_action": first_forecast.get("controller_action"),
            "decision_status": "selected",
            "action_applied": first_controller_selection.get("action_applied", True),
            **(initial_selection or {}),
            **first_controller_selection,
            **decision_value_fields(
                initial_optimization,
                initial_state,
                int(first_forecast["horizon_sessions"]),
            ),
        }
        decision_rows.append(initial_record)
        targets_by_date[first_execution_date] = {
            "decision_date": first_decision_date,
            "target_weights": initial_optimization.target_weights,
            "research_cost_cny": float(first_controller_selection.get("research_cost_cny", 0.0)),
        }

    if not targets_by_date:
        raise ValueError("组合回测没有可执行的调仓信号")

    commission = costs_bps["commission_per_side_bps"] / 10_000
    transfer = costs_bps["transfer_fee_per_side_bps"] / 10_000
    slippage = costs_bps["slippage_per_side_bps"] / 10_000
    stamp = costs_bps["stamp_duty_sell_bps"] / 10_000
    buy_cost_rate = commission + transfer + slippage
    sell_cost_rate = commission + transfer + slippage + stamp

    cash = initial_capital_cny
    positions: dict[str, float] = {}
    previous_nav = initial_capital_cny
    order_rows = []
    position_rows = []
    blocked_orders = 0
    partial_orders = 0
    liquidity_limited_orders = 0
    turnover_limited_orders = 0
    cash_roundoff_clamp_count = 0
    maximum_cash_roundoff_clamp_cny = 0.0
    turnover_notional = 0.0
    total_cost = 0.0
    total_research_cost = 0.0
    if evaluation_start_date is None:
        first_index = research.trading_dates.index(min(targets_by_date))
    else:
        if evaluation_start_date not in research.trading_dates:
            raise ValueError(f"组合共同评价起始日不在交易日历中：{evaluation_start_date}")
        first_index = research.trading_dates.index(evaluation_start_date) + 1
        if first_index >= len(research.trading_dates):
            raise ValueError("共同评价起始日之后没有可模拟的成交日")

    total_sessions = len(research.trading_dates) - first_index
    for session_count, day_index in enumerate(range(first_index, len(research.trading_dates)), start=1):
        trade_date = research.trading_dates[day_index]
        if progress_callback and (session_count == 1 or session_count % 50 == 0 or session_count == total_sessions):
            progress_callback(
                f"{trade_date.isoformat()} ({session_count}/{total_sessions} sessions)"
            )
        previous_date = research.trading_dates[day_index - 1] if day_index else None
        day_traded_notional = 0.0
        day_research_cost = 0.0
        daily_liquidity_used = defaultdict(float)

        # Mark existing positions from the previous close to today's open.
        if previous_date is not None:
            for symbol, value in list(positions.items()):
                current_bar = adjusted.get((symbol, trade_date))
                previous_bar = adjusted.get((symbol, previous_date))
                if (
                    current_bar is not None
                    and previous_bar is not None
                    and current_bar["open"] is not None
                    and previous_bar["close"] is not None
                    and float(previous_bar["close"]) > 0
                ):
                    positions[symbol] = value * (
                        float(current_bar["open"]) / float(previous_bar["close"])
                    )

        if trade_date in targets_by_date:
            decision = targets_by_date[trade_date]
            day_research_cost = float(decision.get("research_cost_cny", 0.0))
            if not math.isfinite(day_research_cost) or day_research_cost < 0:
                raise ValueError(
                    f"Research fee must be finite and nonnegative on {trade_date}: "
                    f"fee={day_research_cost!r}"
                )
            cash -= day_research_cost
            total_research_cost += day_research_cost
            symbols = sorted(positions.keys() | set(decision["target_weights"]))
            decision_index = research.trading_dates.index(decision["decision_date"])
            liquidity_dates = research.trading_dates[
                max(0, decision_index - liquidity_lookback_sessions + 1) : decision_index + 1
            ]
            liquidity_limits = {}
            for symbol in symbols:
                amounts = [
                    float(bar["amount"])
                    for day in liquidity_dates
                    if (bar := execution.get((symbol, day))) is not None
                    and bar["amount"] is not None
                    and float(bar["amount"]) > 0
                ]
                liquidity_limits[symbol] = (
                    statistics.fmean(amounts) * maximum_daily_participation
                    if amounts
                    else 0.0
                )
            pre_funding_nav = cash + math.fsum(positions.values())
            if not math.isfinite(pre_funding_nav) or pre_funding_nav <= 0:
                raise ValueError(
                    f"Research fee leaves no positive account NAV on {trade_date}: "
                    f"fee={day_research_cost:.6f}, cash={cash:.6f}"
                )
            remaining_turnover_notional = max(
                2 * maximum_one_way_turnover * pre_funding_nav, 0.0
            )

            # Fees are cash expenses. If the account is fully invested, sell
            # enough of the existing liquid holdings to fund the fee before
            # sizing the target rebalance. These sales use the same execution,
            # liquidity, turnover, and transaction-cost rules as other orders.
            if cash < 0:
                for symbol in symbols:
                    if cash >= -1e-8:
                        break
                    current_value = positions.get(symbol, 0.0)
                    if current_value <= 0:
                        continue
                    bar = execution.get((symbol, trade_date))
                    status = None if bar is None else str(bar["tradestatus"])
                    requested = min(
                        current_value,
                        -cash / (1 - sell_cost_rate),
                    )
                    if status != "1" or bar["open"] is None:
                        blocked_orders += 1
                        order_rows.append({
                            "trade_date": trade_date.isoformat(),
                            "decision_date": decision["decision_date"].isoformat(),
                            "symbol": symbol,
                            "side": "sell",
                            "requested_notional_cny": requested,
                            "filled_notional_cny": 0.0,
                            "reference_price": None if bar is None else bar["open"],
                            "liquidity_limit_cny": liquidity_limits[symbol],
                            "status": "unfilled_suspended_or_missing_research_fee_funding",
                            "cost_cny": 0.0,
                        })
                        continue
                    capacity = max(
                        liquidity_limits[symbol] - daily_liquidity_used[symbol],
                        0.0,
                    )
                    filled = min(
                        requested,
                        capacity,
                        remaining_turnover_notional,
                    )
                    cost = filled * sell_cost_rate
                    positions[symbol] = current_value - filled
                    if positions[symbol] <= 1e-10:
                        positions.pop(symbol, None)
                    cash += filled - cost
                    turnover_notional += filled
                    day_traded_notional += filled
                    daily_liquidity_used[symbol] += filled
                    remaining_turnover_notional -= filled
                    total_cost += cost
                    if filled < requested:
                        partial_orders += 1
                        if remaining_turnover_notional <= capacity:
                            turnover_limited_orders += 1
                        else:
                            liquidity_limited_orders += 1
                    order_rows.append({
                        "trade_date": trade_date.isoformat(),
                        "decision_date": decision["decision_date"].isoformat(),
                        "symbol": symbol,
                        "side": "sell",
                        "requested_notional_cny": requested,
                        "filled_notional_cny": filled,
                        "reference_price": bar["open"],
                        "liquidity_limit_cny": capacity,
                        "status": (
                            "filled_for_research_fee"
                            if filled == requested
                            else "partial_turnover_limited_for_research_fee"
                            if remaining_turnover_notional <= capacity
                            else "partial_liquidity_limited_for_research_fee"
                        ),
                        "cost_cny": cost,
                    })
                if cash < -1e-8:
                    raise ValueError(
                        f"Research fee cannot be funded from liquid holdings on {trade_date}: "
                        f"fee={day_research_cost:.6f}, remaining_cash={cash:.6f}"
                    )
                cash, cash_roundoff_cny = _cash_after_fill(
                    cash, trade_date=trade_date, symbol="research_fee_funding"
                )
                if cash_roundoff_cny > 0:
                    cash_roundoff_clamp_count += 1
                    maximum_cash_roundoff_clamp_cny = max(
                        maximum_cash_roundoff_clamp_cny,
                        cash_roundoff_cny,
                    )

            open_nav = cash + math.fsum(positions.values())
            desired = {
                symbol: float(weight) * open_nav
                for symbol, weight in decision["target_weights"].items()
            }
            symbols = sorted(positions.keys() | desired.keys())

            # Sell first so proceeds can fund buys on the same rebalance.
            for symbol in symbols:
                current_value = positions.get(symbol, 0.0)
                target_value = desired.get(symbol, 0.0)
                requested = max(current_value - target_value, 0.0)
                if requested == 0:
                    continue
                bar = execution.get((symbol, trade_date))
                status = None if bar is None else str(bar["tradestatus"])
                if status != "1" or bar["open"] is None:
                    blocked_orders += 1
                    order_rows.append(
                        {
                            "trade_date": trade_date.isoformat(),
                            "decision_date": decision["decision_date"].isoformat(),
                            "symbol": symbol,
                            "side": "sell",
                            "requested_notional_cny": requested,
                            "filled_notional_cny": 0.0,
                            "reference_price": None if bar is None else bar["open"],
                            "liquidity_limit_cny": liquidity_limits[symbol],
                            "status": "unfilled_suspended_or_missing",
                            "cost_cny": 0.0,
                        }
                    )
                    continue
                capacity = max(
                    liquidity_limits[symbol] - daily_liquidity_used[symbol],
                    0.0,
                )
                fill_limit = min(capacity, remaining_turnover_notional)
                filled = min(requested, capacity, remaining_turnover_notional)
                cost = filled * sell_cost_rate
                positions[symbol] = current_value - filled
                if positions[symbol] == 0:
                    del positions[symbol]
                cash += filled - cost
                turnover_notional += filled
                day_traded_notional += filled
                daily_liquidity_used[symbol] += filled
                remaining_turnover_notional -= filled
                total_cost += cost
                if filled < requested:
                    partial_orders += 1
                    if remaining_turnover_notional <= capacity and fill_limit < requested:
                        turnover_limited_orders += 1
                    else:
                        liquidity_limited_orders += 1
                order_rows.append(
                    {
                        "trade_date": trade_date.isoformat(),
                        "decision_date": decision["decision_date"].isoformat(),
                        "symbol": symbol,
                        "side": "sell",
                        "requested_notional_cny": requested,
                        "filled_notional_cny": filled,
                        "reference_price": bar["open"],
                        "liquidity_limit_cny": capacity,
                        "status": (
                            "filled"
                            if filled == requested
                            else "partial_turnover_limited"
                            if remaining_turnover_notional <= capacity
                            else "partial_liquidity_limited"
                        ),
                        "cost_cny": cost,
                    }
                )

            for symbol in symbols:
                current_value = positions.get(symbol, 0.0)
                target_value = desired.get(symbol, 0.0)
                requested = max(target_value - current_value, 0.0)
                if requested == 0:
                    continue
                bar = execution.get((symbol, trade_date))
                status = None if bar is None else str(bar["tradestatus"])
                if status != "1" or bar["open"] is None:
                    blocked_orders += 1
                    order_rows.append(
                        {
                            "trade_date": trade_date.isoformat(),
                            "decision_date": decision["decision_date"].isoformat(),
                            "symbol": symbol,
                            "side": "buy",
                            "requested_notional_cny": requested,
                            "filled_notional_cny": 0.0,
                            "reference_price": None if bar is None else bar["open"],
                            "liquidity_limit_cny": liquidity_limits[symbol],
                            "status": "unfilled_suspended_or_missing",
                            "cost_cny": 0.0,
                        }
                    )
                    continue
                capacity = max(
                    liquidity_limits[symbol] - daily_liquidity_used[symbol],
                    0.0,
                )
                affordable = max(cash, 0.0) / (1 + buy_cost_rate)
                fill_limit = min(affordable, capacity, remaining_turnover_notional)
                filled = min(requested, affordable, capacity, remaining_turnover_notional)
                cost = filled * buy_cost_rate
                if filled < requested:
                    partial_orders += 1
                    if remaining_turnover_notional <= min(affordable, capacity) and fill_limit < requested:
                        turnover_limited_orders += 1
                    elif filled < min(requested, affordable):
                        liquidity_limited_orders += 1
                if filled > 0:
                    positions[symbol] = current_value + filled
                    cash -= filled + cost
                    cash, cash_roundoff_cny = _cash_after_fill(
                        cash, trade_date=trade_date, symbol=symbol
                    )
                    if cash_roundoff_cny > 0:
                        cash_roundoff_clamp_count += 1
                        maximum_cash_roundoff_clamp_cny = max(
                            maximum_cash_roundoff_clamp_cny,
                            cash_roundoff_cny,
                        )
                    turnover_notional += filled
                    day_traded_notional += filled
                    daily_liquidity_used[symbol] += filled
                    remaining_turnover_notional -= filled
                    total_cost += cost
                order_rows.append(
                    {
                            "trade_date": trade_date.isoformat(),
                            "decision_date": decision["decision_date"].isoformat(),
                            "symbol": symbol,
                        "side": "buy",
                        "requested_notional_cny": requested,
                        "filled_notional_cny": filled,
                        "reference_price": bar["open"],
                        "liquidity_limit_cny": capacity,
                        "status": (
                            "filled"
                            if filled == requested
                            else "partial_turnover_limited"
                            if remaining_turnover_notional <= min(affordable, capacity)
                            else "partial_liquidity_limited"
                            if filled < min(requested, affordable)
                            else "partial_cash_limited"
                        ),
                        "cost_cny": cost,
                    }
                )

        # Mark today's open positions to the close using the adjusted total-return price.
        for symbol, value in list(positions.items()):
            bar = adjusted.get((symbol, trade_date))
            if bar is None or bar["open"] is None or bar["close"] is None:
                continue
            open_price = float(bar["open"])
            close_price = float(bar["close"])
            if open_price <= 0:
                raise ValueError(f"回测持仓出现无效开盘价：{symbol} {trade_date}")
            positions[symbol] = value * close_price / open_price

        nav = cash + math.fsum(positions.values())
        if not math.isfinite(nav) or nav <= 0:
            raise ValueError(f"组合净值无效：{trade_date} {nav}")
        daily_return = nav / previous_nav - 1
        position_rows.extend(
            {
                "trade_date": trade_date.isoformat(),
                "symbol": symbol,
                "market_value_cny": value,
                "portfolio_weight": value / nav,
            }
            for symbol, value in sorted(positions.items())
        )
        index_bar = index_bars[trade_date]
        if not daily_rows:
            if index_bar.get("open") is None or float(index_bar["open"]) <= 0:
                raise ValueError(f"基准首日缺少有效指数开盘价：{trade_date}")
            benchmark_return = float(index_bar["close"]) / float(index_bar["open"]) - 1
        else:
            prior_index = index_bars[previous_date]["close"]
            current_index = index_bar["close"]
            benchmark_return = float(current_index) / float(prior_index) - 1
        maximum_position = max(
            positions.items(), key=lambda item: item[1], default=("", 0.0)
        )
        daily_rows.append(
            {
                "trade_date": trade_date.isoformat(),
                "nav_cny": nav,
                "cash_cny": cash,
                "daily_return": daily_return,
                "research_cost_cny": day_research_cost,
                "traded_notional_cny": day_traded_notional,
                "one_way_turnover": day_traded_notional / (2 * open_nav)
                if trade_date in targets_by_date and open_nav > 0 else 0.0,
                "cash_weight": cash / nav,
                "maximum_symbol_weight": max(
                    (value / nav for value in positions.values()), default=0.0
                ),
                "maximum_weight_symbol": maximum_position[0],
                "position_count": len(positions),
                "benchmark_nav_cny": (
                    initial_capital_cny * (1 + benchmark_return)
                    if not daily_rows
                    else float(daily_rows[-1]["benchmark_nav_cny"]) * (1 + benchmark_return)
                ),
                "benchmark_return": benchmark_return,
            }
        )
        previous_nav = nav

        if dynamic and trade_date in forecasts_by_date:
            account = PortfolioState(cash, dict(positions))
            decision_pause = untradeable_risk_pause(account, trade_date)
            forecast, controller_selection = prepare_forecast(
                forecasts_by_date[trade_date], account, trade_date, decision_pause
            )
            if decision_pause is not None:
                next_index = day_index + 1
                next_date = research.trading_dates[next_index] if next_index < len(research.trading_dates) else None
                decision_rows.append({
                    "decision_date": trade_date.isoformat(),
                    "execution_date": None,
                    "skipped_next_execution_date": next_date.isoformat() if next_date else None,
                    "decision_status": "paused",
                    "pause_reason": decision_pause["reason"],
                    "paused_symbols": decision_pause["symbols"],
                    "risk_constraint_status": "unverifiable_for_untradeable_existing_holding",
                    "position_mark_policy": "retain_last_observed_adjusted_value_when_current_mark_is_missing; do_not_impute_return",
                    "current_cash_cny": cash,
                    "current_holdings_cny": dict(sorted(positions.items())),
                    "current_nav_cny": account.nav_cny,
                    "target_weights": None,
                    "controller_action": None,
                    **controller_selection,
                })
                continue
            optimization, selection = choose_optimization(forecast, account)
            next_index = day_index + 1
            next_date = research.trading_dates[next_index] if next_index < len(research.trading_dates) else None
            row = {
                "decision_date": trade_date.isoformat(),
                "execution_date": next_date.isoformat() if next_date else None,
                "current_cash_cny": cash,
                "current_holdings_cny": dict(sorted(positions.items())),
                "current_nav_cny": account.nav_cny,
                "target_weights": optimization.target_weights,
                "gross_expected_return": optimization.gross_expected_return,
                "forecast_error_allowance": optimization.forecast_error_allowance,
                "expected_transaction_cost_weight": optimization.expected_transaction_cost,
                "annualized_risk": optimization.annualized_risk,
                "one_way_turnover": optimization.maximum_one_way_turnover,
                "objective_value": optimization.objective_value,
                "controller_action": forecast.get("controller_action"),
                "decision_status": "selected",
                "action_applied": controller_selection.get("action_applied", True),
                **(selection or {}),
                **controller_selection,
                **decision_value_fields(
                    optimization,
                    account,
                    int(forecast["horizon_sessions"]),
                ),
            }
            decision_rows.append(row)
            if next_date is not None:
                targets_by_date[next_date] = {
                    "decision_date": trade_date,
                    "target_weights": optimization.target_weights,
                    "research_cost_cny": float(controller_selection.get("research_cost_cny", 0.0)),
                }

    navs = [float(row["nav_cny"]) for row in daily_rows]
    returns = [float(row["daily_return"]) for row in daily_rows]
    peak = initial_capital_cny
    max_drawdown = 0.0
    for nav in navs:
        peak = max(peak, nav)
        max_drawdown = min(max_drawdown, nav / peak - 1)
    annualized_volatility = (
        statistics.stdev(returns) * math.sqrt(250) if len(returns) > 1 else 0.0
    )
    average_return = statistics.fmean(returns) if returns else 0.0
    downside = [min(value, 0.0) ** 2 for value in returns]
    downside_deviation = math.sqrt(statistics.fmean(downside)) * math.sqrt(250) if downside else 0.0
    benchmark_nav = float(daily_rows[-1]["benchmark_nav_cny"])
    benchmark_navs = [float(row["benchmark_nav_cny"]) for row in daily_rows]
    benchmark_returns = [float(row["benchmark_return"]) for row in daily_rows]
    benchmark_peak = initial_capital_cny
    benchmark_max_drawdown = 0.0
    for nav in benchmark_navs:
        benchmark_peak = max(benchmark_peak, nav)
        benchmark_max_drawdown = min(benchmark_max_drawdown, nav / benchmark_peak - 1)
    benchmark_volatility = (
        statistics.stdev(benchmark_returns) * math.sqrt(250)
        if len(benchmark_returns) > 1
        else 0.0
    )
    annualization = 250 / len(daily_rows)
    final_nav = navs[-1]
    optimized_decisions = [row for row in decision_rows if "buy_notional_cny" in row]
    paused_decisions = [row for row in decision_rows if row.get("decision_status") == "paused"]
    return {
        "summary": {
            "start_date": daily_rows[0]["trade_date"],
            "end_date": daily_rows[-1]["trade_date"],
            "trading_days": len(daily_rows),
            "initial_capital_cny": initial_capital_cny,
            "final_nav_cny": final_nav,
            "total_return": final_nav / initial_capital_cny - 1,
            "annualized_return": (final_nav / initial_capital_cny) ** annualization - 1,
            "annualized_volatility": annualized_volatility,
            "sharpe_rf_zero": (
                average_return * 250 / annualized_volatility
                if annualized_volatility > 0
                else 0.0
            ),
            "sortino_rf_zero": (
                average_return * 250 / downside_deviation
                if downside_deviation > 0
                else 0.0
            ),
            "maximum_drawdown": max_drawdown,
            "benchmark_final_nav_cny": benchmark_nav,
            "benchmark_total_return": benchmark_nav / initial_capital_cny - 1,
            "benchmark_annualized_return": (
                benchmark_nav / initial_capital_cny
            ) ** annualization
            - 1,
            "benchmark_annualized_volatility": benchmark_volatility,
            "benchmark_sharpe_rf_zero": (
                statistics.fmean(benchmark_returns) * 250 / benchmark_volatility
                if benchmark_volatility > 0
                else 0.0
            ),
            "benchmark_maximum_drawdown": benchmark_max_drawdown,
            "total_traded_notional_cny": turnover_notional,
            "turnover_over_average_nav": turnover_notional
            / statistics.fmean(navs),
            "one_way_turnover_over_average_nav": turnover_notional
            / (2 * statistics.fmean(navs)),
            "total_buy_notional_cny": math.fsum(
                float(row["filled_notional_cny"])
                for row in order_rows if row["side"] == "buy"
            ),
            "total_sell_notional_cny": math.fsum(
                float(row["filled_notional_cny"])
                for row in order_rows if row["side"] == "sell"
            ),
            "total_transaction_cost_cny": total_cost,
            "research_cost_cny": total_research_cost,
            "total_cost_including_research_cny": total_cost + total_research_cost,
            "optimizer_decision_count": len(optimized_decisions),
            "paused_decision_count": len(paused_decisions),
            "mean_optimizer_buy_notional_cny": (
                statistics.fmean(float(row["buy_notional_cny"]) for row in optimized_decisions)
                if optimized_decisions else 0.0
            ),
            "mean_optimizer_sell_notional_cny": (
                statistics.fmean(float(row["sell_notional_cny"]) for row in optimized_decisions)
                if optimized_decisions else 0.0
            ),
            "mean_risk_adjusted_net_incremental_value_cny": (
                statistics.fmean(
                    float(row["risk_adjusted_net_incremental_value_cny"])
                    for row in optimized_decisions
                ) if optimized_decisions else 0.0
            ),
            "positive_risk_adjusted_net_incremental_value_decisions": sum(
                float(row["risk_adjusted_net_incremental_value_cny"]) > 1e-8
                for row in optimized_decisions
            ),
            "mean_cash_weight": statistics.fmean(float(row["cash_weight"]) for row in daily_rows),
            "minimum_cash_weight": min(float(row["cash_weight"]) for row in daily_rows),
            "maximum_observed_symbol_weight": max(
                float(row["maximum_symbol_weight"]) for row in daily_rows
            ),
            "filled_orders": sum(row["status"] == "filled" for row in order_rows),
            "blocked_orders": blocked_orders,
            "partial_orders": partial_orders,
            "liquidity_limited_orders": liquidity_limited_orders,
            "turnover_limited_orders": turnover_limited_orders,
            "liquidity_lookback_sessions": liquidity_lookback_sessions,
            "maximum_daily_participation": maximum_daily_participation,
            "cost_assumptions_bps": costs_bps,
            "maximum_one_way_turnover": maximum_one_way_turnover,
            "sub_1e-8_cash_roundoff_clamp_count": cash_roundoff_clamp_count,
            "maximum_cash_roundoff_clamp_cny": maximum_cash_roundoff_clamp_cny,
        },
        "daily_nav": daily_rows,
        "orders": order_rows,
        "positions": position_rows,
        "decisions": decision_rows,
    }


def simulate_fixed_horizon_portfolio(
    research: ResearchData,
    decisions: list[dict[str, object]],
    initial_capital_cny: float,
    costs_bps: dict[str, float],
    liquidity_lookback_sessions: int,
    maximum_daily_participation: float,
    horizon_sessions: int = 5,
    evaluation_start_date: date | None = None,
) -> dict[str, object]:
    """Trade each frozen signal basket until its predeclared fifth-session close."""
    adjusted = {
        (str(row["symbol"]), row["trade_date"]): row for row in research.adjusted_bars
    }
    execution = {
        (str(row["symbol"]), row["trade_date"]): row for row in research.execution_bars
    }
    index_bars = {row["trade_date"]: row for row in research.index_bars}
    entries: dict[date, list[dict[str, object]]] = defaultdict(list)
    for decision in decisions:
        decision_date = decision["decision_date"]
        decision_index = research.trading_dates.index(decision_date)
        exit_index = decision_index + horizon_sessions
        if exit_index >= len(research.trading_dates):
            continue
        entries[research.trading_dates[decision_index + 1]].append(
            {
                "decision_date": decision_date,
                "planned_exit_date": research.trading_dates[exit_index],
                "target_weights": decision["target_weights"],
            }
        )
    if not entries:
        raise ValueError("固定期限组合没有可执行的决策窗口")

    commission = costs_bps["commission_per_side_bps"] / 10_000
    transfer = costs_bps["transfer_fee_per_side_bps"] / 10_000
    slippage = costs_bps["slippage_per_side_bps"] / 10_000
    stamp = costs_bps["stamp_duty_sell_bps"] / 10_000
    buy_cost_rate = commission + transfer + slippage
    sell_cost_rate = commission + transfer + slippage + stamp

    cash = initial_capital_cny
    positions: list[dict[str, object]] = []
    daily_rows = []
    order_rows = []
    position_rows = []
    capacity_limits: dict[tuple[str, date], float] = {}
    capacity_used: dict[tuple[str, date], float] = defaultdict(float)
    total_traded_notional = 0.0
    total_cost = 0.0
    blocked_orders = 0
    partial_orders = 0
    liquidity_limited_orders = 0
    daily_nav = initial_capital_cny
    benchmark_nav = initial_capital_cny
    if evaluation_start_date is None:
        first_entry_index = research.trading_dates.index(min(entries))
    else:
        if evaluation_start_date not in research.trading_dates:
            raise ValueError(f"组合共同评价起始日不在交易日历中：{evaluation_start_date}")
        first_entry_index = research.trading_dates.index(evaluation_start_date) + 1
        if first_entry_index >= len(research.trading_dates):
            raise ValueError("共同评价起始日之后没有可模拟的成交日")

    for day_index in range(first_entry_index, len(research.trading_dates)):
        trade_date = research.trading_dates[day_index]
        previous_date = research.trading_dates[day_index - 1] if day_index else None

        for position in positions:
            bar = adjusted.get((position["symbol"], trade_date))
            execution_bar = execution.get((position["symbol"], trade_date))
            if (
                execution_bar is not None
                and str(execution_bar["tradestatus"]) == "1"
                and (bar is None or bar["open"] is None or bar["close"] is None)
            ):
                raise ValueError(
                    f"可交易持仓缺少完整调整后估值行情：{position['symbol']} {trade_date}"
                )
            if bar is not None and bar["open"] is not None:
                last_price = float(position["last_adjusted_price"])
                open_price = float(bar["open"])
                if last_price <= 0 or open_price <= 0:
                    raise ValueError(
                        f"固定期限账本出现无效开盘价：{position['symbol']} {trade_date}"
                    )
                position["market_value_cny"] = (
                    float(position["market_value_cny"]) * open_price / last_price
                )
                position["last_adjusted_price"] = open_price
                position["mark_age_sessions"] = 0
            else:
                position["mark_age_sessions"] = int(position["mark_age_sessions"]) + 1

        def remaining_capacity(symbol: str) -> float:
            key = (symbol, trade_date)
            if key not in capacity_limits:
                as_of_index = max(day_index - 1, 0)
                start_index = max(0, as_of_index - liquidity_lookback_sessions + 1)
                amounts = [
                    float(bar["amount"])
                    for day in research.trading_dates[start_index : as_of_index + 1]
                    if (bar := execution.get((symbol, day))) is not None
                    and bar["amount"] is not None
                    and float(bar["amount"]) > 0
                ]
                capacity_limits[key] = (
                    statistics.fmean(amounts) * maximum_daily_participation
                    if amounts
                    else 0.0
                )
            return max(capacity_limits[key] - capacity_used[key], 0.0)

        def record_order(
            position: dict[str, object],
            side: str,
            order_type: str,
            requested: float,
            filled: float,
            reference_price: object,
            status: str,
            cost: float,
        ) -> None:
            nonlocal total_traded_notional, total_cost, blocked_orders
            nonlocal partial_orders, liquidity_limited_orders
            if status == "unfilled_suspended_or_missing":
                blocked_orders += 1
            elif status.startswith("partial"):
                partial_orders += 1
                if status == "partial_liquidity_limited":
                    liquidity_limited_orders += 1
            if filled > 0:
                total_traded_notional += filled
                total_cost += cost
                capacity_used[(str(position["symbol"]), trade_date)] += filled
            order_rows.append(
                {
                    "trade_date": trade_date.isoformat(),
                    "decision_date": position["decision_date"].isoformat(),
                    "planned_exit_date": position["planned_exit_date"].isoformat(),
                    "position_id": position["position_id"],
                    "symbol": position["symbol"],
                    "side": side,
                    "order_type": order_type,
                    "requested_notional_cny": requested,
                    "filled_notional_cny": filled,
                    "reference_price": reference_price,
                    "status": status,
                    "cost_cny": cost,
                }
            )

        # Carry an expired exit order to the first later open that can trade.
        remaining_positions = []
        for position in positions:
            if position["planned_exit_date"] >= trade_date:
                remaining_positions.append(position)
                continue
            bar = execution.get((position["symbol"], trade_date))
            if bar is None or str(bar["tradestatus"]) != "1" or bar["open"] is None:
                record_order(
                    position,
                    "sell",
                    "carried_exit_open",
                    float(position["market_value_cny"]),
                    0.0,
                    None if bar is None else bar["open"],
                    "unfilled_suspended_or_missing",
                    0.0,
                )
                remaining_positions.append(position)
                continue
            if (
                adjusted.get((position["symbol"], trade_date)) is None
                or adjusted[(position["symbol"], trade_date)]["open"] is None
            ):
                raise ValueError(
                    f"可交易退出订单缺少调整后估值开盘价：{position['symbol']} {trade_date}"
                )
            requested = float(position["market_value_cny"])
            filled = min(requested, remaining_capacity(str(position["symbol"])))
            cost = filled * sell_cost_rate
            cash += filled - cost
            position["market_value_cny"] = requested - filled
            record_order(
                position,
                "sell",
                "carried_exit_open",
                requested,
                filled,
                bar["open"],
                "filled" if filled == requested else "partial_liquidity_limited",
                cost,
            )
            if position["market_value_cny"] > 0:
                remaining_positions.append(position)
        positions = remaining_positions

        for decision in entries.get(trade_date, []):
            weights = decision["target_weights"]
            total_weight = math.fsum(float(weight) for weight in weights.values())
            if total_weight <= 0:
                raise ValueError("固定期限组合收到非正投资权重")
            open_nav = cash + math.fsum(
                float(position["market_value_cny"]) for position in positions
            )
            carried_value_by_symbol: dict[str, float] = defaultdict(float)
            for position in positions:
                carried_value_by_symbol[str(position["symbol"])] += float(
                    position["market_value_cny"]
                )
            target_increments = {
                symbol: max(
                    open_nav * float(weight) - carried_value_by_symbol[symbol], 0.0
                )
                for symbol, weight in weights.items()
            }
            requested_total = math.fsum(target_increments.values())
            if requested_total == 0:
                continue
            cash_budget = max(cash, 0.0) / (1 + buy_cost_rate)
            budget = min(requested_total, cash_budget)
            scale = budget / requested_total
            for symbol in sorted(weights):
                requested = target_increments[symbol] * scale
                if requested == 0:
                    continue
                bar = execution.get((symbol, trade_date))
                adjusted_bar = adjusted.get((symbol, trade_date))
                if (
                    bar is None
                    or str(bar["tradestatus"]) != "1"
                    or bar["open"] is None
                ):
                    blocked_orders += 1
                    order_rows.append(
                        {
                            "trade_date": trade_date.isoformat(),
                            "decision_date": decision["decision_date"].isoformat(),
                            "planned_exit_date": decision["planned_exit_date"].isoformat(),
                            "position_id": f"{decision['decision_date']}:{symbol}",
                            "symbol": symbol,
                            "side": "buy",
                            "order_type": "entry_open",
                            "requested_notional_cny": requested,
                            "filled_notional_cny": 0.0,
                            "reference_price": None if bar is None else bar["open"],
                            "status": "unfilled_suspended_or_missing",
                            "cost_cny": 0.0,
                        }
                    )
                    continue
                if (
                    adjusted_bar is None
                    or adjusted_bar["open"] is None
                    or adjusted_bar["close"] is None
                ):
                    raise ValueError(
                        f"可成交订单缺少完整调整后估值行情：{symbol} {trade_date}"
                    )
                capacity = remaining_capacity(symbol)
                affordable = max(cash, 0.0) / (1 + buy_cost_rate)
                filled = min(requested, capacity, affordable)
                cost = filled * buy_cost_rate
                cash -= filled + cost
                status = (
                    "filled"
                    if filled == requested
                    else "partial_liquidity_limited"
                    if filled < min(requested, affordable)
                    else "partial_cash_limited"
                )
                position = {
                    "position_id": f"{decision['decision_date']}:{symbol}",
                    "symbol": symbol,
                    "decision_date": decision["decision_date"],
                    "planned_exit_date": decision["planned_exit_date"],
                    "market_value_cny": filled,
                    "last_adjusted_price": float(adjusted_bar["open"]),
                    "mark_age_sessions": 0,
                    "exit_pending": False,
                }
                if filled > 0:
                    positions.append(position)
                record_order(
                    position,
                    "buy",
                    "entry_open",
                    requested,
                    filled,
                    bar["open"],
                    status,
                    cost,
                )

        for position in positions:
            bar = adjusted.get((position["symbol"], trade_date))
            if bar is None or bar["open"] is None or bar["close"] is None:
                position["exit_pending"] = (
                    position["planned_exit_date"] <= trade_date
                )
                continue
            open_price = float(bar["open"])
            close_price = float(bar["close"])
            if open_price <= 0 or close_price <= 0:
                raise ValueError(
                    f"固定期限账本出现无效估值价格：{position['symbol']} {trade_date}"
                )
            position["market_value_cny"] = (
                float(position["market_value_cny"]) * close_price / open_price
            )
            position["last_adjusted_price"] = close_price

        # The close order was fixed at signal time; no fifth-day signal is read here.
        remaining_positions = []
        for position in positions:
            if position["planned_exit_date"] != trade_date:
                if position["planned_exit_date"] < trade_date:
                    position["exit_pending"] = True
                remaining_positions.append(position)
                continue
            bar = execution.get((position["symbol"], trade_date))
            if bar is None or str(bar["tradestatus"]) != "1" or bar["close"] is None:
                position["exit_pending"] = True
                record_order(
                    position,
                    "sell",
                    "planned_exit_close",
                    float(position["market_value_cny"]),
                    0.0,
                    None if bar is None else bar["close"],
                    "unfilled_suspended_or_missing",
                    0.0,
                )
                remaining_positions.append(position)
                continue
            adjusted_bar = adjusted.get((position["symbol"], trade_date))
            if adjusted_bar is None or adjusted_bar["close"] is None:
                raise ValueError(
                    f"可交易退出订单缺少调整后估值收盘价：{position['symbol']} {trade_date}"
                )
            requested = float(position["market_value_cny"])
            filled = min(requested, remaining_capacity(str(position["symbol"])))
            cost = filled * sell_cost_rate
            cash += filled - cost
            position["market_value_cny"] = requested - filled
            record_order(
                position,
                "sell",
                "planned_exit_close",
                requested,
                filled,
                bar["close"],
                "filled" if filled == requested else "partial_liquidity_limited",
                cost,
            )
            if position["market_value_cny"] > 0:
                position["exit_pending"] = True
                remaining_positions.append(position)
        positions = remaining_positions

        nav = cash + math.fsum(
            float(position["market_value_cny"]) for position in positions
        )
        if not math.isfinite(nav) or nav <= 0:
            raise ValueError(f"固定期限组合净值无效：{trade_date} {nav}")
        account_return = nav / daily_nav - 1
        if not daily_rows:
            index_open = index_bars[trade_date]["open"]
            if index_open is None or float(index_open) <= 0:
                raise ValueError(f"固定期限基准首日缺少有效指数开盘价：{trade_date}")
            benchmark_return = (
                float(index_bars[trade_date]["close"]) / float(index_open) - 1
            )
        else:
            benchmark_return = (
                float(index_bars[trade_date]["close"])
                / float(index_bars[previous_date]["close"])
                - 1
            )
        benchmark_nav *= 1 + benchmark_return
        position_values: dict[str, float] = defaultdict(float)
        for position in positions:
            position_values[str(position["symbol"])] += float(
                position["market_value_cny"]
            )
            position_rows.append(
                {
                    "trade_date": trade_date.isoformat(),
                    **position,
                    "decision_date": position["decision_date"].isoformat(),
                    "planned_exit_date": position["planned_exit_date"].isoformat(),
                    "portfolio_weight": float(position["market_value_cny"]) / nav,
                }
            )
        daily_rows.append(
            {
                "trade_date": trade_date.isoformat(),
                "nav_cny": nav,
                "cash_cny": cash,
                "daily_return": account_return,
                "cash_weight": cash / nav,
                "position_count": len(position_values),
                "pending_exit_notional_cny": math.fsum(
                    float(position["market_value_cny"])
                    for position in positions
                    if position["exit_pending"]
                ),
                "maximum_symbol_weight": max(
                    (value / nav for value in position_values.values()), default=0.0
                ),
                "benchmark_nav_cny": benchmark_nav,
                "benchmark_return": benchmark_return,
            }
        )
        daily_nav = nav

    navs = [float(row["nav_cny"]) for row in daily_rows]
    returns = [float(row["daily_return"]) for row in daily_rows]
    benchmark_returns = [float(row["benchmark_return"]) for row in daily_rows]
    peak = initial_capital_cny
    maximum_drawdown = 0.0
    benchmark_peak = initial_capital_cny
    benchmark_maximum_drawdown = 0.0
    for nav, benchmark_value in zip(
        navs,
        (float(row["benchmark_nav_cny"]) for row in daily_rows),
        strict=True,
    ):
        peak = max(peak, nav)
        benchmark_peak = max(benchmark_peak, benchmark_value)
        maximum_drawdown = min(maximum_drawdown, nav / peak - 1)
        benchmark_maximum_drawdown = min(
            benchmark_maximum_drawdown, benchmark_value / benchmark_peak - 1
        )
    volatility = statistics.stdev(returns) * math.sqrt(250) if len(returns) > 1 else 0.0
    benchmark_volatility = (
        statistics.stdev(benchmark_returns) * math.sqrt(250)
        if len(benchmark_returns) > 1
        else 0.0
    )
    annualization = 250 / len(daily_rows)
    final_nav = navs[-1]
    pending_exit_value = math.fsum(
        float(position["market_value_cny"])
        for position in positions
        if position["exit_pending"]
    )
    average_nav = statistics.fmean(navs)

    return {
        "summary": {
            "start_date": daily_rows[0]["trade_date"],
            "end_date": daily_rows[-1]["trade_date"],
            "trading_days": len(daily_rows),
            "initial_capital_cny": initial_capital_cny,
            "final_nav_cny": final_nav,
            "total_return": final_nav / initial_capital_cny - 1,
            "annualized_return": (final_nav / initial_capital_cny) ** annualization - 1,
            "annualized_volatility": volatility,
            "sharpe_rf_zero": (
                statistics.fmean(returns) * 250 / volatility if volatility > 0 else 0.0
            ),
            "maximum_drawdown": maximum_drawdown,
            "benchmark_final_nav_cny": benchmark_nav,
            "benchmark_total_return": benchmark_nav / initial_capital_cny - 1,
            "benchmark_annualized_return": (
                benchmark_nav / initial_capital_cny
            ) ** annualization
            - 1,
            "benchmark_annualized_volatility": benchmark_volatility,
            "benchmark_sharpe_rf_zero": (
                statistics.fmean(benchmark_returns) * 250 / benchmark_volatility
                if benchmark_volatility > 0
                else 0.0
            ),
            "benchmark_maximum_drawdown": benchmark_maximum_drawdown,
            "total_traded_notional_cny": total_traded_notional,
            "turnover_over_average_nav": total_traded_notional / average_nav,
            "one_way_turnover_over_average_nav": total_traded_notional / (2 * average_nav),
            "total_transaction_cost_cny": total_cost,
            "mean_cash_weight": statistics.fmean(float(row["cash_weight"]) for row in daily_rows),
            "minimum_cash_weight": min(float(row["cash_weight"]) for row in daily_rows),
            "maximum_observed_symbol_weight": max(
                float(row["maximum_symbol_weight"]) for row in daily_rows
            ),
            "filled_orders": sum(row["status"] == "filled" for row in order_rows),
            "blocked_orders": blocked_orders,
            "partial_orders": partial_orders,
            "liquidity_limited_orders": liquidity_limited_orders,
            "open_position_count": len(positions),
            "pending_exit_notional_cny": pending_exit_value,
            "liquidity_lookback_sessions": liquidity_lookback_sessions,
            "maximum_daily_participation": maximum_daily_participation,
            "horizon_sessions": horizon_sessions,
            "cost_assumptions_bps": costs_bps,
        },
        "daily_nav": daily_rows,
        "orders": order_rows,
        "positions": position_rows,
    }
