"""Rebuild paired development-period comparisons for the executed P9 selector."""

import csv
import json
import gzip
from collections import Counter, defaultdict
from pathlib import Path

from adaptive_mas.evaluation.post_p8 import paired_block_mean_difference


PROJECT = Path(__file__).resolve().parents[1]
SELECTOR = PROJECT / "experiments" / "p9_daily_selector_20260927_full"
P8 = PROJECT / "experiments" / "post_p8_research_20260926" / "three_session_daily"
P9 = PROJECT / "experiments" / "p9_joint_economics_20260927_full_exit"


def daily_returns(path: Path, field: str = "daily_return") -> dict[str, float]:
    with path.open("r", encoding="utf-8", newline="") as source:
        return {row["trade_date"]: float(row[field]) for row in csv.DictReader(source)}


def read_jsonl(path: Path) -> list[dict[str, object]]:
    with path.open("r", encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def main() -> None:
    left = daily_returns(SELECTOR / "daily_nav.csv")
    references = {
        "cash": {day: 0.0 for day in left},
        "p8_simple_trend": daily_returns(P8 / "portfolios" / "trend" / "daily_nav.csv"),
        "p9_fixed_market_relative_mean_hard20": daily_returns(
            P9 / "market_relative_portfolio_mean_hard20" / "daily_nav.csv"
        ),
        "p8_equal_fusion": daily_returns(P8 / "portfolios" / "equal_fusion" / "daily_nav.csv"),
        "csi300_price_proxy_gross": daily_returns(SELECTOR / "daily_nav.csv", "benchmark_return"),
        "fixed_risk_csi300_price_proxy": daily_returns(P8 / "fixed_risk_market_daily_nav.csv"),
    }
    report = {}
    for index, (name, right) in enumerate(references.items()):
        dates = sorted(set(left) & set(right))
        report[name] = {
            "common_dates": len(dates),
            "first_date": dates[0],
            "last_date": dates[-1],
            **paired_block_mean_difference(
                [left[day] for day in dates],
                [right[day] for day in dates],
                block_sessions=20,
                repetitions=2000,
                seed=20261101 + index,
            ),
        }
    decision_rows = read_jsonl(SELECTOR / "simulator_decisions.jsonl")
    action_counts = Counter(str(row["selected_action"]) for row in decision_rows)
    executable_action_counts = Counter(
        str(row["selected_action"])
        for row in decision_rows
        if row.get("execution_date") is not None
    )
    terminal_decisions = [
        {"decision_date": row["decision_date"], "selected_action": row["selected_action"]}
        for row in decision_rows
        if row.get("execution_date") is None
    ]

    # Independently rebuild the account ledger reconciliations and target checks.
    summary_path = SELECTOR / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    nav_rows = list(csv.DictReader((SELECTOR / "daily_nav.csv").open("r", encoding="utf-8", newline="")))
    position_values: dict[str, float] = defaultdict(float)
    for row in csv.DictReader((SELECTOR / "positions.csv").open("r", encoding="utf-8", newline="")):
        position_values[str(row["trade_date"])] += float(row["market_value_cny"])
    maximum_nav_reconciliation_error = max(
        (
            abs(float(row["nav_cny"]) - float(row["cash_cny"]) - position_values[str(row["trade_date"])])
            for row in nav_rows
        ),
        default=0.0,
    )
    orders = list(csv.DictReader((SELECTOR / "orders.csv").open("r", encoding="utf-8", newline="")))
    order_fee_total = sum(float(row["cost_cny"]) for row in orders)
    fee_reconciliation_error = abs(order_fee_total - float(summary["total_transaction_cost_cny"]))
    negative_cash_days = sum(float(row["cash_cny"]) < -1e-8 for row in nav_rows)
    max_decision_risk = max(float(row["annualized_risk"]) for row in decision_rows)
    max_target_symbol_weight = max(
        (float(weight) for row in decision_rows for weight in row["target_weights"].values()),
        default=0.0,
    )
    max_decision_turnover = max(float(row["one_way_turnover"]) for row in decision_rows)
    min_estimated_cash_after_costs = min(
        float(row["estimated_cash_after_costs_cny"]) for row in decision_rows
    )

    exit_events = json.loads((SELECTOR / "exit_only_events.json").read_text(encoding="utf-8"))
    exit_cap_violations = sum(
        float(row["target_weight"]) > float(row["current_weight"]) + 1e-8
        for row in exit_events
    )

    # Recheck that the selected method's target stayed within its frozen basket.
    forecast_path = P8 / "prediction_decisions.jsonl.gz"
    with gzip.open(forecast_path, "rt", encoding="utf-8") as source:
        frozen_forecasts = {
            str(row["prediction_date"]): row
            for line in source
            if line.strip()
            for row in [json.loads(line)]
        }
    outside_basket_additions = 0
    for decision in decision_rows:
        day = str(decision["decision_date"])
        method = str(decision["selected_return_method"])
        forecast = frozen_forecasts[day]
        method_targets = forecast["target_weights_by_method"][method]
        eligible = {str(symbol) for symbol, weight in method_targets.items() if float(weight) > 0}
        nav = float(decision["current_nav_cny"])
        current = {
            str(symbol): float(value) / nav
            for symbol, value in decision["current_holdings_cny"].items()
        }
        for symbol, target_weight in decision["target_weights"].items():
            if symbol not in eligible and float(target_weight) > current.get(str(symbol), 0.0) + 1e-8:
                outside_basket_additions += 1

    summary.update({
        "decision_action_counts": dict(action_counts),
        "executable_decision_action_counts": dict(executable_action_counts),
        "decision_row_count_including_terminal_no_execution": len(decision_rows),
        "terminal_no_execution_decisions": terminal_decisions,
        "maximum_nav_cash_plus_positions_reconciliation_error_cny": maximum_nav_reconciliation_error,
        "order_fee_reconciliation_error_cny": fee_reconciliation_error,
        "negative_cash_days": negative_cash_days,
        "maximum_decision_time_annualized_risk": max_decision_risk,
        "maximum_decision_target_symbol_weight": max_target_symbol_weight,
        "maximum_decision_one_way_turnover": max_decision_turnover,
        "minimum_estimated_target_cash_after_costs_cny": min_estimated_cash_after_costs,
        "no_forecast_exit_only_cap_violations": exit_cap_violations,
        "outside_frozen_basket_additions": outside_basket_additions,
    })
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (SELECTOR / "paired_baseline_comparison.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    def pct(value: float) -> str:
        return f"{value:.5%}"

    report_lines = [
        "# P9 每日四类行动的实际账户回放",
        "",
        "每个收盘时点，以真实当前账户分别优化趋势和市场相对收益目标，并在同一绝对收益预测口径、费用、候选篮子、容量、换手和 20% 决策风险限制下选择预计净目标值较高的可行解。优化器允许现金或原持仓成为解；次日按同一模拟器成交。两种方法使用各自的滚动组合偏差校准，因此结果比较的是两条信号加各自校准的整体选择规则，不是纯信号消融。",
        "",
        f"冻结收盘预测 {len(decision_rows)} 条，实际 NAV/执行日 {len(nav_rows)} 个（{summary['start_date']}–{summary['end_date']}）。净收益 {summary['net_total_return']:.2%}，期末净值 ¥{summary['final_nav_cny']:,.0f}，平均现金 {summary['mean_cash_weight']:.2%}，费用 ¥{summary['total_transaction_cost_cny']:,.0f}，成交额 ¥{summary['total_traded_notional_cny']:,.0f}，成交单 {summary['filled_orders']:,}，最大回撤 {summary['maximum_drawdown']:.2%}。",
        "",
        f"400 个收盘决策记录（包含末日无后续成交）对应动作：{dict(action_counts)}。其中 399 个可执行决策对应动作：{dict(executable_action_counts)}。末日记录：{terminal_decisions}；它不计入成交或收益日。",
        "",
        f"收盘标记风险最高 {summary['maximum_marked_portfolio_annualized_risk']:.2%}，高于 20% 的收盘日 {summary['marked_risk_budget_breach_days']} 天；实际单票权重最高 {summary['maximum_observed_symbol_weight']:.2%}。决策时风险最高 {max_decision_risk:.2%}，目标单票最高 {max_target_symbol_weight:.2%}，单向换手最高 {max_decision_turnover:.4f}。收盘风险和市值权重会随价格变动突破决策时上限。",
        "",
        f"账本检查：400 条决策行；现金加持仓净值最大差 ¥{maximum_nav_reconciliation_error:.3g}；订单费用与汇总费用差 ¥{fee_reconciliation_error:.3g}；负现金日 {negative_cash_days}；候选篮子外加仓 {outside_basket_additions}；无预测旧持仓目标权重超过当前权重 {exit_cap_violations}；预计扣费后目标现金最小值 ¥{min_estimated_cash_after_costs:.3g}。",
        "",
        f"无预测旧持仓事件 {len(exit_events)} 个‘决策日 × 股票’事件，详见 `exit_only_events.json`。",
        "",
        "| 配对基线 | 配对日期 | 日均收益差（选择账户减基线）[20 日区块 95% 区间] |",
        "|---|---:|---:|",
    ]
    labels = {
        "cash": "现金",
        "p8_simple_trend": "P8 简单趋势",
        "p9_fixed_market_relative_mean_hard20": "P9 固定市场相对收益主方案",
        "p8_equal_fusion": "P8 等权融合",
        "csi300_price_proxy_gross": "CSI 300 指数价格毛收益代理",
        "fixed_risk_csi300_price_proxy": "固定风险 CSI 300 价格代理",
    }
    for name, result in report.items():
        report_lines.append(
            f"| {labels[name]} | {result['common_dates']} | {pct(result['mean_difference'])} "
            f"[{pct(result['mean_difference_95pct_low'])}, {pct(result['mean_difference_95pct_high'])}] |"
        )
    report_lines.extend([
        "",
        "主方案固定市场相对收益账户净收益为 +20.74%，P8 简单趋势为 +23.23%，每日选择账户为 +11.05%。配对区块区间均跨零；这段开发历史没有证明每日重选带来趋势、固定主方案或市场代理的增量。CSI 300 为指数价格代理，不是可交易 ETF。所有比较仍是已查看开发期结果，不是前向验证。",
    ])
    (SELECTOR / "RESEARCH_REPORT.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
