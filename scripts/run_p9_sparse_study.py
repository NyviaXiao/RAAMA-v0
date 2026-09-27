"""Replay the predeclared sparse expert rule through the P8 simulator."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import os
import statistics
import subprocess
import sys
import tomllib
from collections import Counter
from datetime import date, datetime, timezone
from pathlib import Path

from adaptive_mas.backtest.simulator import simulate_portfolio
from adaptive_mas.data import load_research_data
from adaptive_mas.data.research import ResearchData
from adaptive_mas.evaluation.post_p8 import paired_block_mean_difference
from adaptive_mas.evaluation.mvp_research import _risk_estimates
from adaptive_mas.evaluation.sparse_fusion import build_sparse_fusion_decisions


EXPERTS = ("trend", "reversal", "liquidity_confirmation")


def _read_jsonl_gzip(path: Path) -> list[dict[str, object]]:
    with gzip.open(path, "rt", encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_jsonl_gzip(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as zipped:
            with __import__("io").TextIOWrapper(zipped, encoding="utf-8", newline="\n") as output:
                for row in rows:
                    output.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as source:
        return list(csv.DictReader(source))


def _baseline_rows(source: Path, name: str) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    directory = source / "portfolios" / name
    return _read_csv(directory / "daily_nav.csv"), _read_csv(directory / "orders.csv")


def _summarize_nav(
    nav_rows: list[dict[str, object]] | list[dict[str, str]],
    order_rows: list[dict[str, object]] | list[dict[str, str]],
    initial_capital: float,
) -> dict[str, object]:
    if not nav_rows:
        raise ValueError("P9 对照组合没有净值日期")
    nav_values = [float(row["nav_cny"]) for row in nav_rows]
    cash_weights = [float(row["cash_weight"]) for row in nav_rows if row.get("cash_weight") is not None]
    peak = initial_capital
    drawdown = 0.0
    for nav in nav_values:
        peak = max(peak, nav)
        drawdown = min(drawdown, nav / peak - 1)
    return {
        "start_date": str(nav_rows[0]["trade_date"]),
        "end_date": str(nav_rows[-1]["trade_date"]),
        "trading_days": len(nav_rows),
        "final_nav_cny": nav_values[-1],
        "net_total_return": nav_values[-1] / initial_capital - 1,
        "maximum_drawdown": drawdown,
        "mean_cash_weight": statistics.fmean(cash_weights) if cash_weights else None,
        "total_transaction_cost_cny": math.fsum(
            float(row.get("cost_cny") or row.get("transaction_cost_cny") or 0.0)
            for row in order_rows
        ),
        "total_filled_notional_cny": math.fsum(
            float(row.get("filled_notional_cny") or row.get("notional_cny") or 0.0)
            for row in order_rows
        ),
        "filled_order_count": sum(
            str(row.get("status", "")).startswith("filled") for row in order_rows
        ),
        "negative_cash_days": sum(
            row.get("cash_cny") is not None and float(row["cash_cny"]) < -0.01
            for row in nav_rows
        ),
        "unfilled_order_status_counts": {
            status: sum(row.get("status") == status for row in order_rows)
            for status in sorted({
                str(row.get("status"))
                for row in order_rows
                if not str(row.get("status", "")).startswith("filled")
            })
        },
        "daily_returns": [float(row["daily_return"]) for row in nav_rows],
    }


def main() -> None:
    project_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--training-data-dir",
        type=Path,
        default=Path(r"C:\Users\xiao\Desktop\THU-BDC2026\data"),
    )
    parser.add_argument(
        "--baostock-run-dir",
        type=Path,
        default=project_dir / "data" / "raw" / "baostock" / "fetch-20260924T115050Z",
    )
    parser.add_argument(
        "--source-study-dir",
        type=Path,
        default=project_dir / "experiments" / "post_p8_research_20260926" / "three_session_daily",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_dir / "experiments" / "p9_sparse_fusion_20260926",
    )
    parser.add_argument(
        "--smoke-decision-count",
        type=int,
        default=None,
        help="Optional short real-data smoke run; use the first N frozen decision dates only.",
    )
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"稀疏融合输出目录已存在，拒绝覆盖：{output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)
    status_path = output_dir / "RUN_STATUS.json"
    progress_path = output_dir / "progress.log"
    status: dict[str, object] = {
        "status": "running",
        "pid": os.getpid(),
        "command": [sys.executable, *sys.argv],
        "output_dir": str(output_dir),
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "last_progress": "initializing",
    }

    def progress(message: str) -> None:
        status["last_progress_at_utc"] = datetime.now(timezone.utc).isoformat()
        status["last_progress"] = message
        _write_json(status_path, status)
        with progress_path.open("a", encoding="utf-8", newline="\n") as output:
            output.write(f"{status['last_progress_at_utc']} {message}\n")
        print(f"[P9 sparse] {message}", flush=True)

    _write_json(status_path, status)
    source = args.source_study_dir.resolve()
    try:
        progress("loading only train.csv and the frozen BaoStock development cache")
        research = load_research_data(args.training_data_dir, args.baostock_run_dir)
        config = tomllib.loads((source / "config.toml").read_text(encoding="utf-8"))
        study_results_path = source / "study_results.json"
        study_results = json.loads(study_results_path.read_text(encoding="utf-8"))
        common_start = date.fromisoformat(str(study_results["common_start_date"]))
        protocol = study_results["protocol"]
        if str(protocol["comparison_start_date"]) != common_start.isoformat():
            raise ValueError("冻结研究 common_start_date 与 protocol.comparison_start_date 不一致")
        if str(protocol["initial_comparison_date"]) != common_start.isoformat():
            raise ValueError("冻结研究 common_start_date 与 protocol.initial_comparison_date 不一致")
        if common_start not in research.trading_dates:
            raise ValueError("冻结研究共同起始日不在 BaoStock 交易日历中")
        forecasts = _read_jsonl_gzip(source / "prediction_decisions.jsonl.gz")
        expert_returns = {
            expert: _read_csv(source / "portfolios" / expert / "daily_nav.csv")
            for expert in EXPERTS
        }
        trend_daily_nav = _read_csv(source / "portfolios" / "trend" / "daily_nav.csv")
        if not trend_daily_nav:
            raise ValueError("趋势策略净值账没有共同起点")
        first_execution_date = date.fromisoformat(str(trend_daily_nav[0]["trade_date"]))
        start_index = research.trading_dates.index(common_start)
        if start_index + 1 >= len(research.trading_dates):
            raise ValueError("冻结共同起始日之后没有下一交易日")
        if research.trading_dates[start_index + 1] != first_execution_date:
            raise ValueError("趋势净值首日与冻结共同起始日之后的下一交易日不一致")
        costs = {
            key: float(config[key])
            for key in (
                "commission_per_side_bps",
                "transfer_fee_per_side_bps",
                "stamp_duty_sell_bps",
                "slippage_per_side_bps",
            )
        }
        buy_rate = (
            costs["commission_per_side_bps"]
            + costs["transfer_fee_per_side_bps"]
            + costs["slippage_per_side_bps"]
        ) / 10_000
        sell_rate = buy_rate + costs["stamp_duty_sell_bps"] / 10_000
        matured_window = 20
        minimum_matured_decisions = 20
        decisions, activation_log = build_sparse_fusion_decisions(
            forecasts,
            expert_returns,
            comparison_start_date=common_start,
            portfolio_fraction=float(config["portfolio_top_fraction"]),
            invested_weight=float(config["invested_weight"]),
            maximum_symbol_weight=float(config["maximum_symbol_weight"]),
            maximum_one_way_turnover=float(config["maximum_one_way_turnover"]),
            risk_budget_annualized=float(config["risk_budget_annualized"]),
            matured_window=matured_window,
            minimum_matured_decisions=minimum_matured_decisions,
        )
        smoke_mode = args.smoke_decision_count is not None
        if smoke_mode:
            if args.smoke_decision_count < 1:
                raise ValueError("smoke decision count must be positive")
            decisions = decisions[:args.smoke_decision_count]
            activation_log = activation_log[:args.smoke_decision_count]
            if not decisions:
                raise ValueError("smoke run has no decisions at the common start")
            final_decision_index = research.trading_dates.index(decisions[-1]["decision_date"])
            final_execution_index = final_decision_index + 1
            if final_execution_index >= len(research.trading_dates):
                raise ValueError("smoke run has no next-session execution date")
            final_date = research.trading_dates[final_execution_index]
            research = ResearchData(
                trading_dates=[day for day in research.trading_dates if day <= final_date],
                adjusted_bars=[row for row in research.adjusted_bars if row["trade_date"] <= final_date],
                execution_bars=[row for row in research.execution_bars if row["trade_date"] <= final_date],
                index_bars=[row for row in research.index_bars if row["trade_date"] <= final_date],
                membership_snapshots=[
                    snapshot
                    for snapshot in research.membership_snapshots
                    if snapshot.available_at <= final_date
                ],
            )
        rule = {
            "rule_version": "sparse-realized-expert-net-v1",
            "history_source": "single-expert continuous simulator daily NAV net returns; includes actual next-session fills, capacity limits, turnover, holdings and recorded transaction costs",
            "history_window_sessions": matured_window,
            "minimum_completed_returns": minimum_matured_decisions,
            "only_dates_strictly_before_current_decision_are_eligible": True,
            "active_when_trailing_mean_daily_net_return_positive": True,
            "active_experts_share_weight_equally": True,
            "no_positive_expert_means_cash_target": True,
            "portfolio_fraction": float(config["portfolio_top_fraction"]),
            "invested_weight": float(config["invested_weight"]),
            "maximum_symbol_weight": float(config["maximum_symbol_weight"]),
            "maximum_one_way_turnover": float(config["maximum_one_way_turnover"]),
            "risk_budget_annualized": float(config["risk_budget_annualized"]),
            "risk_rule": "at each as-of decision, scale the signal target by at most risk_budget / forecast_factor_risk; no soft quadratic penalty is added; the target account risk is recomputed using only the frozen point-in-time risk estimates",
            "costs_bps": costs,
            "common_start_date": common_start.isoformat(),
            "selection_scope": "development-only prequential diagnostic; not forward validation",
            "smoke_run": smoke_mode,
            "smoke_decision_count": len(decisions) if smoke_mode else None,
        }
        _write_json(output_dir / "frozen_sparse_rule.json", rule)
        _write_jsonl_gzip(output_dir / "activation_log.jsonl.gz", activation_log)
        _write_jsonl_gzip(
            output_dir / "sparse_decisions.jsonl.gz",
            [
                {
                    "decision_date": row["decision_date"].isoformat(),
                    "target_weights": row["target_weights"],
                }
                for row in decisions
            ],
        )

        progress(f"built {len(decisions)} sparse decisions from completed same-simulator expert histories; smoke={smoke_mode}")
        simulated = simulate_portfolio(
            research,
            decisions,
            float(config["initial_capital_cny"]),
            costs,
            int(config["liquidity_lookback_sessions"]),
            float(config["maximum_daily_participation"]),
            float(config["maximum_one_way_turnover"]),
            progress_callback=lambda message: progress(message),
            evaluation_start_date=common_start,
        )
        summary = simulated.pop("summary")
        sparse_nav = simulated["daily_nav"]
        summary["net_total_return"] = summary["total_return"]
        order_statuses = Counter(str(row["status"]) for row in simulated["orders"])
        summary["unfilled_order_status_counts"] = {
            status: count
            for status, count in sorted(order_statuses.items())
            if status != "filled"
        }
        summary["negative_cash_days"] = sum(
            float(row["cash_cny"]) < -0.01 for row in sparse_nav
        )
        positions_by_date: dict[str, list[dict[str, object]]] = {}
        for position in simulated["positions"]:
            positions_by_date.setdefault(str(position["trade_date"]), []).append(position)
        actual_risks = []
        for nav_row in sparse_nav:
            trade_date = date.fromisoformat(str(nav_row["trade_date"]))
            symbols = {
                str(position["symbol"])
                for position in positions_by_date.get(trade_date.isoformat(), [])
            }
            if not symbols:
                risk_value = 0.0
            else:
                decision_index = research.trading_dates.index(trade_date)
                vols, betas, market_vol = _risk_estimates(
                    research,
                    decision_index,
                    symbols,
                    int(config["risk_lookback_sessions"]),
                    int(config["risk_minimum_observations"]),
                )
                missing = symbols - set(vols)
                if missing:
                    raise ValueError(f"稀疏账户日终持仓缺少成熟风险估计：{trade_date} {sorted(missing)}")
                weights = {
                    str(position["symbol"]): float(position["market_value_cny"])
                    / float(nav_row["nav_cny"])
                    for position in positions_by_date[trade_date.isoformat()]
                }
                idiosyncratic = math.fsum(
                    weights[symbol] ** 2
                    * max(float(vols[symbol]) ** 2 - float(betas[symbol]) ** 2 * market_vol**2, 0.0)
                    for symbol in symbols
                )
                factor_exposure = math.fsum(weights[symbol] * float(betas[symbol]) for symbol in symbols)
                risk_value = math.sqrt(idiosyncratic + (factor_exposure * market_vol) ** 2)
            nav_row["annualized_risk_model"] = risk_value
            actual_risks.append(risk_value)
        summary["maximum_marked_portfolio_annualized_risk"] = max(actual_risks, default=0.0)
        summary["marked_risk_budget_breach_days"] = sum(
            value > float(config["risk_budget_annualized"]) + 1e-8 for value in actual_risks
        )
        summary["marked_risk_budget_breach_dates"] = [
            str(row["trade_date"])
            for row in sparse_nav
            if float(row["annualized_risk_model"]) > float(config["risk_budget_annualized"]) + 1e-8
        ]
        summary["maximum_decision_target_annualized_risk"] = max(
            float(row["final_target_annualized_risk"]) for row in activation_log
        ) if activation_log else 0.0
        _write_csv(output_dir / "daily_nav.csv", sparse_nav)
        _write_csv(output_dir / "orders.csv", simulated["orders"])
        _write_csv(output_dir / "positions.csv", simulated["positions"])
        _write_jsonl_gzip(output_dir / "simulator_decisions.jsonl.gz", simulated["decisions"])

        initial_capital = float(config["initial_capital_cny"])
        baselines: dict[str, tuple[list[dict[str, object]] | list[dict[str, str]], list[dict[str, object]] | list[dict[str, str]]]] = {}
        for name in ("trend", "equal_fusion", "joint_optimization"):
            baseline_nav, baseline_orders = _baseline_rows(source, name)
            baselines[name] = (baseline_nav, baseline_orders)
        fixed_nav = _read_csv(source / "fixed_risk_market_daily_nav.csv")
        fixed_orders = _read_csv(source / "fixed_risk_market_orders.csv")
        baselines["fixed_risk_csi300_price_proxy"] = (fixed_nav, fixed_orders)
        matched_start_date = str(sparse_nav[0]["trade_date"])
        matched_end_date = str(sparse_nav[-1]["trade_date"])
        for name, (nav_rows, order_rows) in list(baselines.items()):
            nav_rows = [
                row
                for row in nav_rows
                if matched_start_date <= str(row["trade_date"]) <= matched_end_date
            ]
            order_rows = [
                row
                for row in order_rows
                if matched_start_date <= str(row["trade_date"]) <= matched_end_date
            ]
            baselines[name] = (nav_rows, order_rows)
        cash_rows = [
            {"trade_date": str(row["trade_date"]), "nav_cny": initial_capital, "cash_cny": initial_capital, "cash_weight": 1.0, "daily_return": 0.0}
            for row in sparse_nav
        ]
        baselines["cash"] = (cash_rows, [])
        benchmark_rows = []
        benchmark_nav = initial_capital
        for row in sparse_nav:
            daily_return = float(row["benchmark_return"])
            benchmark_nav *= 1 + daily_return
            benchmark_rows.append(
                {
                    "trade_date": str(row["trade_date"]),
                    "nav_cny": benchmark_nav,
                    "cash_cny": None,
                    "cash_weight": None,
                    "daily_return": daily_return,
                }
            )
        baselines["csi300_price_proxy_gross"] = (benchmark_rows, [])
        baseline_summaries = {
            name: _summarize_nav(nav_rows, order_rows, initial_capital)
            for name, (nav_rows, order_rows) in baselines.items()
        }
        summary["paired_block_mean_difference_vs_baselines"] = {}
        sparse_returns = [float(row["daily_return"]) for row in sparse_nav]
        sparse_dates = [str(row["trade_date"]) for row in sparse_nav]
        for index, (name, baseline) in enumerate(baseline_summaries.items()):
            baseline_daily = baseline["daily_returns"]
            baseline_navs = baselines.get(name, ([], []))[0]
            baseline_dates = [str(row["trade_date"]) for row in baseline_navs]
            left_map = dict(zip(sparse_dates, sparse_returns, strict=True))
            right_map = dict(zip(baseline_dates, baseline_daily, strict=True))
            common = sorted(left_map.keys() & right_map.keys())
            if not common:
                raise ValueError(f"稀疏策略与 {name} 没有共同净值日期")
            summary["paired_block_mean_difference_vs_baselines"][name] = {
                **paired_block_mean_difference(
                    [left_map[day] for day in common],
                    [right_map[day] for day in common],
                    block_sessions=20,
                    repetitions=2000,
                    seed=20260926 + index,
                ),
                "common_dates": len(common),
                "common_start_date": common[0],
                "common_end_date": common[-1],
            }
            baseline_summaries[name]["comparison_start_date"] = common[0]
            baseline_summaries[name]["comparison_end_date"] = common[-1]
            del baseline_summaries[name]["daily_returns"]
        summary["rule_version"] = rule["rule_version"]
        summary["active_expert_days"] = {
            expert: sum(expert in row["active_experts"] for row in activation_log)
            for expert in EXPERTS
        }
        summary["cash_target_days"] = sum(not row["active_experts"] for row in activation_log)
        _write_json(output_dir / "summary.json", summary)
        _write_json(output_dir / "matched_baseline_summary.json", baseline_summaries)

        report = [
            "# P9 Sparse Expert Fusion: Same-Simulator Development Replay",
            "",
            "## Frozen gate",
            "",
            f"The method uses the prior {matured_window} completed daily NAV returns from each single-expert continuous simulator account. Those returns already include fills, liquidity capacity, holdings, turnover, and transaction fees. An expert is active only when its trailing arithmetic mean is positive; active experts share score weight equally. The rule requires {minimum_matured_decisions} completed prior dates, uses only dates strictly before today's after-close decision, and targets cash when no expert qualifies.",
            f"Matched comparison window: {matched_start_date} through {matched_end_date}; every table row is cut to these common simulator dates.",
            "",
            f"Each sparse target is scaled using its as-of factor-plus-idiosyncratic forecast risk so that estimated annualized target risk is at most {float(config['risk_budget_annualized']):.0%}. The rule uses this hard risk ceiling without adding a quadratic soft risk penalty. The same target keeps the {float(config['maximum_symbol_weight']):.0%} per-name cap and {float(config['maximum_one_way_turnover']):.0%} one-way turnover limit; the simulator applies next-session trading status, cash, participation capacity, and the full fee schedule. Separately, end-of-day marked holdings were re-estimated with that date's risk inputs; these retrospective estimates can move above 20% after returns or updated risk inputs.",
            "",
            "## Same-account results",
            "",
            f"Sparse fusion: {summary['net_total_return']:.2%} net return, {summary['maximum_drawdown']:.2%} maximum drawdown, {summary['mean_cash_weight']:.1%} average cash, ¥{summary['total_transaction_cost_cny']:,.0f} fees, ¥{summary['total_traded_notional_cny']:,.0f} traded, {summary['filled_orders']} fills, {summary['blocked_orders']} blocked/unfilled orders. Ex-ante decision targets peaked at {summary['maximum_decision_target_annualized_risk']:.2%} against the 20% ceiling. End-of-day marked holdings, re-estimated using each date's available risk inputs, peaked at {summary['maximum_marked_portfolio_annualized_risk']:.2%}; {summary['marked_risk_budget_breach_days']} such marks exceeded 20% on {', '.join(summary['marked_risk_budget_breach_dates']) or 'none'}.",
            "",
            "| Same-start baseline | Matched window | Net/gross return | Max drawdown | Mean cash | Cash deficit days | Fees | Filled notional | Fills | Unfilled statuses | Paired daily mean difference (95% block interval) |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---|---:|",
        ]
        sparse_summary = summary
        for name, values in baseline_summaries.items():
            paired = summary["paired_block_mean_difference_vs_baselines"][name]
            statuses = json.dumps(values.get("unfilled_order_status_counts", {}), ensure_ascii=False)
            drawdown = "n/a" if values["maximum_drawdown"] is None else f"{values['maximum_drawdown']:.2%}"
            mean_cash = "n/a" if values["mean_cash_weight"] is None else f"{values['mean_cash_weight']:.1%}"
            report.append(
                f"| {name} | {paired['common_start_date']}–{paired['common_end_date']} | {values['net_total_return']:.2%} | {drawdown} | {mean_cash} | {values['negative_cash_days']} | ¥{values['total_transaction_cost_cny']:,.0f} | ¥{values['total_filled_notional_cny']:,.0f} | {values['filled_order_count']} | {statuses} | {paired['mean_difference']:.5%} [{paired['mean_difference_95pct_low']:.5%}, {paired['mean_difference_95pct_high']:.5%}] |"
            )
        report.extend([
            f"| sparse_fusion | {matched_start_date}–{matched_end_date} | {sparse_summary['net_total_return']:.2%} | {sparse_summary['maximum_drawdown']:.2%} | {sparse_summary['mean_cash_weight']:.1%} | {sparse_summary['negative_cash_days']} | ¥{sparse_summary['total_transaction_cost_cny']:,.0f} | ¥{sparse_summary['total_traded_notional_cny']:,.0f} | {sparse_summary['filled_orders']} | {json.dumps(sparse_summary['unfilled_order_status_counts'], ensure_ascii=False)} | — |",
            "",
            "The fixed-risk CSI 300 price proxy is not a tradable ETF. Its source order ledger uses `notional_cny` and status `filled_index_price_proxy`; the historical proxy holds 100% exposure plus fees and therefore records a ¥810 cash deficit on execution dates. It is shown as a price-only reference, not as a feasible long-only account baseline.",
            "",
            "## How to read this result",
            "",
            "This is a prequential development replay on data already examined by the P8 research. It is not future validation. The expert gate learns from realized standalone account returns, not from same-day or future outcomes. Sparse fusion is compared with cash, trend, equal fusion, the P8 full-cash joint optimizer, a fixed-risk CSI 300 price proxy and gross CSI 300 index prices on common dates from the same initial cash account. The index price proxy is not a tradable ETF.",
            "",
            "A positive or negative result does not by itself validate the gate as a production allocation rule: the 20-day gate is one predeclared diagnostic method and was not searched over alternative windows or thresholds.",
            "",
            "## Artifacts",
            "",
            "`activation_log.jsonl.gz` records prior dates, trailing net means, active expert weights and daily targets. `orders.csv`, `positions.csv`, `daily_nav.csv`, and `simulator_decisions.jsonl.gz` are the exact same-simulator account ledgers. `matched_baseline_summary.json` and `summary.json` contain cost, turnover, fill, drawdown, cash and paired block interval results.",
        ])
        (output_dir / "RESEARCH_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
        _write_json(output_dir / "REPRODUCIBILITY.json", {
            "git_head": subprocess.run(["git", "rev-parse", "HEAD"], cwd=project_dir, check=True, capture_output=True, text=True).stdout.strip(),
            "config_sha256": hashlib.sha256((source / "config.toml").read_bytes()).hexdigest(),
            "study_results_sha256": hashlib.sha256(study_results_path.read_bytes()).hexdigest(),
            "source_study_dir": str(source),
            "source_prediction_sha256": hashlib.sha256((source / "prediction_decisions.jsonl.gz").read_bytes()).hexdigest(),
            "source_expert_daily_nav_sha256": {
                expert: hashlib.sha256((source / "portfolios" / expert / "daily_nav.csv").read_bytes()).hexdigest()
                for expert in EXPERTS
            },
            "market_manifest_sha256": hashlib.sha256((args.baostock_run_dir / "manifest.jsonl").read_bytes()).hexdigest(),
        })
        status.update({
            "status": "complete",
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            "last_progress": "sparse same-simulator replay and paired comparisons completed",
            "result_report": str(output_dir / "RESEARCH_REPORT.md"),
        })
        _write_json(status_path, status)
        progress("sparse same-simulator replay and paired comparisons completed")
    except Exception as error:
        status.update({
            "status": "failed",
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            "error": f"{type(error).__name__}: {error}",
        })
        _write_json(status_path, status)
        raise


if __name__ == "__main__":
    main()
