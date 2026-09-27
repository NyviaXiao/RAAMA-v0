"""Replay the predeclared daily cash/hold/trend/market-relative choice."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tomllib
from collections import Counter
from datetime import date, datetime, timezone
from pathlib import Path

from adaptive_mas.backtest.simulator import simulate_portfolio
from adaptive_mas.data import load_research_data
from run_p9_joint_economics import (
    _actual_marked_risk,
    _adjust_forecast,
    _read_jsonl_gzip,
    _write_csv,
    _write_json,
)


def main() -> None:
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-data-dir", type=Path, default=Path(r"C:\Users\xiao\Desktop\THU-BDC2026\data"))
    parser.add_argument("--baostock-run-dir", type=Path, default=project / "data" / "raw" / "baostock" / "fetch-20260924T115050Z")
    parser.add_argument("--source-study-dir", type=Path, default=project / "experiments" / "post_p8_research_20260926" / "three_session_daily")
    parser.add_argument("--completed-replay-dir", type=Path, default=project / "experiments" / "p9_joint_economics_20260927_full_exit")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--smoke-decision-count", type=int)
    args = parser.parse_args()
    completed = args.completed_replay_dir.resolve()
    if json.loads((completed / "RUN_STATUS.json").read_text(encoding="utf-8"))["status"] != "complete":
        raise ValueError("逐日选择回放需要已完成的冻结均值校准")
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"输出目录已存在：{output}")
    output.mkdir(parents=True)
    status_path = output / "RUN_STATUS.json"
    progress_path = output / "progress.log"
    status = {"status": "running", "pid": os.getpid(), "command": [sys.executable, *sys.argv],
              "started_at_utc": datetime.now(timezone.utc).isoformat(), "completed_replay_dir": str(completed)}

    def progress(message: str) -> None:
        stamp = datetime.now(timezone.utc).isoformat()
        status.update({"last_progress": message, "last_progress_at_utc": stamp})
        _write_json(status_path, status)
        with progress_path.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(f"{stamp} {message}\n")
        print(f"[P9 daily selector] {message}", flush=True)

    _write_json(status_path, status)
    try:
        source = args.source_study_dir.resolve()
        config = tomllib.loads((source / "config.toml").read_text(encoding="utf-8"))
        study = json.loads((source / "study_results.json").read_text(encoding="utf-8"))
        common_start = date.fromisoformat(study["common_start_date"])
        calibration = {}
        with (completed / "rolling_calibration_details.jsonl").open(encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                day = row["decision_date"]
                for method in ("trend", "market_plus_relative_fusion"):
                    if row["by_method"][method]["window_end_label_date"] >= day:
                        raise ValueError(f"{day} {method} 校准使用未成熟标签")
                calibration[day] = row["by_method"]
        frozen = {
            row["prediction_date"]: row
            for row in _read_jsonl_gzip(source / "prediction_decisions.jsonl.gz")
            if row["prediction_date"] in calibration
        }
        if set(frozen) != set(calibration):
            raise ValueError("冻结预测日期和校准日期不一致")
        days = sorted(calibration)
        if args.smoke_decision_count is not None:
            if args.smoke_decision_count < 1:
                raise ValueError("smoke 决策数必须为正")
            days = days[:args.smoke_decision_count]
        forecasts = [
            _adjust_forecast(frozen[day], calibration[day], calibrated=True,
                             correction_field="total_mean_bias_estimate")
            for day in days
        ]
        research = load_research_data(args.training_data_dir, args.baostock_run_dir)
        if date.fromisoformat(days[0]) != common_start:
            raise ValueError("首个决策日与冻结共同起点不同")
        if args.smoke_decision_count is not None:
            last_index = research.trading_dates.index(date.fromisoformat(days[-1]))
            if last_index + 1 >= len(research.trading_dates):
                raise ValueError("短跑末日没有次日成交数据")
            end = research.trading_dates[last_index + 1]
            research = type(research)(
                trading_dates=[day for day in research.trading_dates if day <= end],
                adjusted_bars=[row for row in research.adjusted_bars if row["trade_date"] <= end],
                execution_bars=[row for row in research.execution_bars if row["trade_date"] <= end],
                index_bars=[row for row in research.index_bars if row["trade_date"] <= end],
                membership_snapshots=[row for row in research.membership_snapshots if row.available_at <= end],
            )
        costs = {key: float(config[key]) for key in (
            "commission_per_side_bps", "transfer_fee_per_side_bps",
            "stamp_duty_sell_bps", "slippage_per_side_bps")}
        optimizer_config = dict(config)
        optimizer_config.update({
            "optimizer_return_method": "p9_best_trend_or_market_relative",
            "optimizer_candidate_scope": "frozen_method_target",
            "risk_aversion": 0.0,
            "forecast_error_multiplier": 0.0,
            "risk_budget_annualized": 0.20,
        })
        progress(f"replaying {len(forecasts)} frozen decisions with point-in-time two-signal choice")
        result = simulate_portfolio(
            research, [], float(config["initial_capital_cny"]), costs,
            int(config["liquidity_lookback_sessions"]),
            float(config["maximum_daily_participation"]),
            float(config["maximum_one_way_turnover"]),
            forecast_decisions=forecasts, optimizer_config=optimizer_config,
            progress_callback=progress, evaluation_start_date=common_start,
        )
        summary = result.pop("summary")
        max_risk, breach_days, breach_dates = _actual_marked_risk(
            research, result["daily_nav"], result["positions"], optimizer_config
        )
        selection_counts = Counter()
        executable_selection_counts = Counter()
        decisions = result["decisions"]
        exit_events = []
        for decision in decisions:
            day = decision["decision_date"]
            method = decision["selected_return_method"]
            frozen_row = frozen[day]
            eligible = {
                symbol for symbol, weight in frozen_row["target_weights_by_method"][method].items()
                if float(weight) > 0
            }
            current = {
                symbol: float(value) / float(decision["current_nav_cny"])
                for symbol, value in decision["current_holdings_cny"].items()
            }
            target = decision["target_weights"]
            for symbol, weight in target.items():
                if symbol not in eligible and float(weight) > current.get(symbol, 0.0) + 1e-8:
                    raise ValueError(f"{day} {method} 候选外加仓：{symbol}")
            for symbol, weight in current.items():
                if symbol not in frozen_row["expected_returns_by_method"][method]:
                    target_weight = float(target.get(symbol, 0.0))
                    if target_weight > weight + 1e-8:
                        raise ValueError(
                            f"{day} {method} 无预测旧持仓被加仓：{symbol} "
                            f"{weight:.12g}->{target_weight:.12g}"
                        )
                    exit_events.append({"decision_date": day, "symbol": symbol,
                                        "method": method, "current_weight": weight,
                                        "target_weight": target_weight,
                                        "forecast_alpha_used": 0.0})
            if not target:
                action = "cash"
            elif all(abs(float(target.get(symbol, 0.0)) - current.get(symbol, 0.0)) <= 1e-8
                     for symbol in set(target) | set(current)):
                action = "hold"
            else:
                action = method
            decision["selected_action"] = action
            selection_counts[action] += 1
            if decision.get("execution_date") is not None:
                executable_selection_counts[action] += 1
        summary.update({
            "net_total_return": summary["total_return"],
            "maximum_marked_portfolio_annualized_risk": max_risk,
            "marked_risk_budget_breach_days": breach_days,
            "marked_risk_budget_breach_dates": breach_dates,
            "decision_action_counts": dict(selection_counts),
            "executable_decision_action_counts": dict(executable_selection_counts),
            "non_executable_terminal_decisions": [
                {"decision_date": row["decision_date"], "selected_action": row["selected_action"]}
                for row in decisions if row.get("execution_date") is None
            ],
            "no_forecast_held_position_event_count": len(exit_events),
            "calibration_decisions": len(forecasts),
            "execution_days": len(result["daily_nav"]),
        })
        _write_json(output / "summary.json", summary)
        _write_json(output / "exit_only_events.json", exit_events)
        _write_csv(output / "daily_nav.csv", result["daily_nav"])
        _write_csv(output / "orders.csv", result["orders"])
        _write_csv(output / "positions.csv", result["positions"])
        with (output / "simulator_decisions.jsonl").open("w", encoding="utf-8", newline="\n") as stream:
            for row in decisions:
                stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")
        baseline = json.loads((completed / "market_relative_portfolio_mean_hard20" / "summary.json").read_text(encoding="utf-8"))
        report = [
            "# P9 每日四类行动的实际账户回放", "",
            "每次收盘以同一个当前账户分别优化趋势与市场加相对收益候选，比较已校准绝对收益、费用及硬约束下的预计净目标值，选最高可行目标；优化器内部同时允许现金与继续持有。第二天按现有模拟器成交。两种预测各有独立的滚动偏差校准，因此这是两条信号流水线的选择，而非纯粹只换一个排序。", "",
            f"冻结预测 {len(forecasts)} 条，实际成交估值日 {len(result['daily_nav'])} 个；净收益 {summary['net_total_return']:.2%}，平均现金 {summary['mean_cash_weight']:.1%}，费用 ¥{summary['total_transaction_cost_cny']:,.0f}，最大回撤 {summary['maximum_drawdown']:.2%}。固定市场相对收益主账户净收益 {baseline['net_total_return']:.2%}。", "",
            f"400条收盘决策的目标行动次数（含无次日可执行的末日）：{dict(selection_counts)}。其中399条有次日执行，目标行动次数为 {dict(executable_selection_counts)}；末个决策日及动作记录在 summary.json，但不计入执行日收益。收盘重新测算风险高于20%的日数 {breach_days}，最高 {max_risk:.2%}。股票退出预测池而仅可持有/减仓的日期×股票事件 {len(exit_events)}。", "",
            "该结果仍是已查看开发历史，选择规则没有经过未见未来日期验证。相同日期简单趋势和市场价格代理对照见完整 P9 报告。",
        ]
        (output / "RESEARCH_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
        status.update({"status": "complete", "finished_at_utc": datetime.now(timezone.utc).isoformat(),
                       "result_report": str(output / "RESEARCH_REPORT.md")})
        progress("daily selector replay complete")
    except Exception as error:
        status.update({"status": "failed", "finished_at_utc": datetime.now(timezone.utc).isoformat(),
                       "error": f"{type(error).__name__}: {error}"})
        _write_json(status_path, status)
        raise


if __name__ == "__main__":
    main()
