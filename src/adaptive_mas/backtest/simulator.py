"""Daily notional-account simulation using BaoStock execution status."""

import math
import statistics
from collections import defaultdict
from datetime import date

from adaptive_mas.data.research import ResearchData


def simulate_portfolio(
    research: ResearchData,
    decisions: list[dict[str, object]],
    initial_capital_cny: float,
    costs_bps: dict[str, float],
    liquidity_lookback_sessions: int,
    maximum_daily_participation: float,
) -> dict[str, object]:
    adjusted = {
        (str(row["symbol"]), row["trade_date"]): row for row in research.adjusted_bars
    }
    execution = {
        (str(row["symbol"]), row["trade_date"]): row for row in research.execution_bars
    }
    index_bars = {row["trade_date"]: row for row in research.index_bars}
    targets_by_date: dict[date, dict[str, object]] = {}
    for decision in decisions:
        decision_date = decision["decision_date"]
        decision_index = research.trading_dates.index(decision_date)
        if decision_index + 1 >= len(research.trading_dates):
            continue
        execution_date = research.trading_dates[decision_index + 1]
        targets_by_date[execution_date] = decision

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
    daily_rows = []
    order_rows = []
    position_rows = []
    blocked_orders = 0
    partial_orders = 0
    liquidity_limited_orders = 0
    turnover_notional = 0.0
    total_cost = 0.0
    first_index = research.trading_dates.index(min(targets_by_date))

    for day_index in range(first_index, len(research.trading_dates)):
        trade_date = research.trading_dates[day_index]
        previous_date = research.trading_dates[day_index - 1] if day_index else None

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
            open_nav = cash + math.fsum(positions.values())
            desired = {
                symbol: float(weight) * open_nav
                for symbol, weight in decision["target_weights"].items()
            }
            symbols = sorted(positions.keys() | desired.keys())
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
                capacity = liquidity_limits[symbol]
                filled = min(requested, capacity)
                cost = filled * sell_cost_rate
                positions[symbol] = current_value - filled
                if positions[symbol] == 0:
                    del positions[symbol]
                cash += filled - cost
                turnover_notional += filled
                total_cost += cost
                if filled < requested:
                    partial_orders += 1
                    liquidity_limited_orders += 1
                order_rows.append(
                    {
                        "trade_date": trade_date.isoformat(),
                        "symbol": symbol,
                        "side": "sell",
                        "requested_notional_cny": requested,
                        "filled_notional_cny": filled,
                        "reference_price": bar["open"],
                        "liquidity_limit_cny": capacity,
                        "status": "filled" if filled == requested else "partial_liquidity_limited",
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
                capacity = liquidity_limits[symbol]
                affordable = max(cash, 0.0) / (1 + buy_cost_rate)
                filled = min(requested, affordable, capacity)
                cost = filled * buy_cost_rate
                if filled < requested:
                    partial_orders += 1
                    if filled < min(requested, affordable):
                        liquidity_limited_orders += 1
                if filled > 0:
                    positions[symbol] = current_value + filled
                    cash -= filled + cost
                    turnover_notional += filled
                    total_cost += cost
                order_rows.append(
                    {
                        "trade_date": trade_date.isoformat(),
                        "symbol": symbol,
                        "side": "buy",
                        "requested_notional_cny": requested,
                        "filled_notional_cny": filled,
                        "reference_price": bar["open"],
                        "liquidity_limit_cny": capacity,
                        "status": (
                            "filled"
                            if filled == requested
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
        if previous_date is None:
            benchmark_return = 0.0
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
                "daily_return": daily_return,
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
            "liquidity_lookback_sessions": liquidity_lookback_sessions,
            "maximum_daily_participation": maximum_daily_participation,
            "cost_assumptions_bps": costs_bps,
        },
        "daily_nav": daily_rows,
        "orders": order_rows,
        "positions": position_rows,
    }
