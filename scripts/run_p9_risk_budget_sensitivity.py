"""Replay only the pre-specified 15% and 25% P9 hard-risk comparisons.

This runner reuses completed point-in-time calibration from the five-variant
P9 full run, avoiding recalibration or repetition of its other variants.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import tomllib
from datetime import date, datetime, timezone
from pathlib import Path

from adaptive_mas.backtest.simulator import simulate_portfolio
from adaptive_mas.data import load_research_data
from run_p9_joint_economics import (
    MAIN_METHOD,
    _actual_marked_risk,
    _adjust_forecast,
    _audit_candidate_scope,
    _read_jsonl_gzip,
    _write_csv,
    _write_json,
)


def _held_without_forecast_events(
    decisions: list[dict[str, object]], forecasts_by_date: dict[str, dict[str, object]], method: str
) -> list[dict[str, object]]:
    events = []
    for decision in decisions:
        day = str(decision["decision_date"])
        forecast = forecasts_by_date[day]
        returns = forecast["expected_returns_by_method"][method]
        nav = float(decision["current_nav_cny"])
        for symbol, value in decision["current_holdings_cny"].items():
            market_value = float(value)
            if market_value <= 0 or str(symbol) in returns:
                continue
            current_weight = market_value / nav
            target_weight = float(decision["target_weights"].get(str(symbol), 0.0))
            if target_weight > current_weight + 1e-8:
                raise ValueError(
                    f"{day} {symbol}: no-forecast exit-only position increased "
                    f"from {current_weight:.12g} to {target_weight:.12g}"
                )
            events.append({
                "decision_date": day,
                "execution_date": decision.get("execution_date"),
                "symbol": str(symbol),
                "current_market_value_cny": market_value,
                "current_weight": current_weight,
                "target_weight": target_weight,
                "forecast_alpha_used": 0.0,
                "policy": "no selected-method forecast: zero alpha; retain or reduce only, capped at current weight",
            })
    return events


def main() -> None:
    project_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--completed-replay-dir",
        type=Path,
        default=project_dir / "experiments" / "p9_joint_economics_20260927_full_exit",
    )
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
        default=project_dir / "experiments" / "p9_joint_economics_20260927_risk_sensitivity",
    )
    args = parser.parse_args()

    completed_dir = args.completed_replay_dir.resolve()
    completed_status = json.loads((completed_dir / "RUN_STATUS.json").read_text(encoding="utf-8"))
    if completed_status.get("status") != "complete":
        raise ValueError(
            f"主回放必须先完成，当前状态为 {completed_status.get('status')!r}：{completed_dir}"
        )
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"风险敏感性输出目录已存在，拒绝覆盖：{output_dir}")

    output_dir.mkdir(parents=True, exist_ok=False)
    status_path = output_dir / "RUN_STATUS.json"
    progress_path = output_dir / "progress.log"
    status: dict[str, object] = {
        "status": "running",
        "pid": os.getpid(),
        "command": [sys.executable, *sys.argv],
        "output_dir": str(output_dir),
        "source_full_replay_dir": str(completed_dir),
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "last_progress": "initializing pre-specified risk-budget comparisons",
    }

    def progress(message: str) -> None:
        status["last_progress_at_utc"] = datetime.now(timezone.utc).isoformat()
        status["last_progress"] = message
        _write_json(status_path, status)
        with progress_path.open("a", encoding="utf-8", newline="\n") as output:
            output.write(f"{status['last_progress_at_utc']} {message}\n")
        print(f"[P9 risk sensitivity] {message}", flush=True)

    _write_json(status_path, status)
    try:
        source = args.source_study_dir.resolve()
        config = tomllib.loads((source / "config.toml").read_text(encoding="utf-8"))
        study_results = json.loads((source / "study_results.json").read_text(encoding="utf-8"))
        common_start = date.fromisoformat(str(study_results["common_start_date"]))
        if common_start.isoformat() != str(study_results["protocol"]["comparison_start_date"]):
            raise ValueError("冻结共同起始日与 comparison_start_date 不一致")

        calibration_rows = []
        calibration_path = completed_dir / "rolling_calibration_details.jsonl"
        with calibration_path.open("r", encoding="utf-8") as source_file:
            for line in source_file:
                if not line.strip():
                    continue
                row = json.loads(line)
                day = str(row["decision_date"])
                method_rows = row["by_method"]
                estimate = method_rows[MAIN_METHOD]
                if str(estimate["window_end_label_date"]) >= day:
                    raise ValueError(f"{day} 的校准窗包含未成熟标签")
                calibration_rows.append(row)
        calibration_by_date = {
            str(row["decision_date"]): row["by_method"] for row in calibration_rows
        }
        if not calibration_by_date:
            raise ValueError("完整回放缺少滚动校准明细")

        frozen_forecasts = _read_jsonl_gzip(source / "prediction_decisions.jsonl.gz")
        forecasts_by_date = {
            str(row["prediction_date"]): row
            for row in frozen_forecasts
            if str(row["prediction_date"]) in calibration_by_date
        }
        if set(forecasts_by_date) != set(calibration_by_date):
            raise ValueError("冻结预测与完整回放校准日期没有一一对应")
        replay_forecasts = [forecasts_by_date[day] for day in sorted(calibration_by_date)]
        point_corrected_forecasts = [
            _adjust_forecast(
                row,
                calibration_by_date[str(row["prediction_date"])],
                calibrated=True,
                correction_field="total_mean_bias_estimate",
            )
            for row in replay_forecasts
        ]
        progress(
            f"loaded {len(point_corrected_forecasts)} frozen dates and reused exact completed calibration"
        )

        research = load_research_data(args.training_data_dir, args.baostock_run_dir)
        if common_start not in research.trading_dates:
            raise ValueError("共同起始日不在 BaoStock 交易日历")
        first_decision = date.fromisoformat(str(point_corrected_forecasts[0]["prediction_date"]))
        last_decision = date.fromisoformat(str(point_corrected_forecasts[-1]["prediction_date"]))
        if first_decision != common_start:
            raise ValueError("首个敏感性决策日与冻结共同起始日不一致")
        main_nav_path = completed_dir / "market_relative_portfolio_mean_hard20" / "daily_nav.csv"
        with main_nav_path.open("r", encoding="utf-8", newline="") as source_file:
            main_nav = list(csv.DictReader(source_file))
        if not main_nav:
            raise ValueError("已完成主回放缺少执行净值账")
        match_end = date.fromisoformat(main_nav[-1]["trade_date"])
        if last_decision != match_end:
            raise ValueError("风险对照的末个冻结预测日与主回放结束日不一致")
        research = type(research)(
            trading_dates=[day for day in research.trading_dates if day <= match_end],
            adjusted_bars=[row for row in research.adjusted_bars if row["trade_date"] <= match_end],
            execution_bars=[row for row in research.execution_bars if row["trade_date"] <= match_end],
            index_bars=[row for row in research.index_bars if row["trade_date"] <= match_end],
            membership_snapshots=[
                snapshot for snapshot in research.membership_snapshots
                if snapshot.available_at <= match_end
            ],
        )

        costs = {
            key: float(config[key])
            for key in (
                "commission_per_side_bps",
                "transfer_fee_per_side_bps",
                "stamp_duty_sell_bps",
                "slippage_per_side_bps",
            )
        }
        results: dict[str, object] = {}
        decisions_by_method = {
            str(row["prediction_date"]): row for row in point_corrected_forecasts
        }
        for budget in (0.15, 0.25):
            label = f"risk_budget_{int(budget * 100)}pct"
            scenario_dir = output_dir / label
            scenario_dir.mkdir(parents=True, exist_ok=False)
            optimizer_config = dict(config)
            optimizer_config.update({
                "optimizer_return_method": MAIN_METHOD,
                "optimizer_candidate_scope": "frozen_method_target",
                "risk_aversion": 0.0,
                "forecast_error_multiplier": 0.0,
                "risk_budget_annualized": budget,
            })
            progress(f"replaying {label}; {len(point_corrected_forecasts)} after-close forecasts")
            simulated = simulate_portfolio(
                research,
                [],
                float(config["initial_capital_cny"]),
                costs,
                int(config["liquidity_lookback_sessions"]),
                float(config["maximum_daily_participation"]),
                float(config["maximum_one_way_turnover"]),
                forecast_decisions=point_corrected_forecasts,
                optimizer_config=optimizer_config,
                progress_callback=lambda message, name=label: progress(f"{name}: {message}"),
                evaluation_start_date=common_start,
            )
            summary = simulated.pop("summary")
            candidate_audit = _audit_candidate_scope(
                point_corrected_forecasts, simulated["decisions"], MAIN_METHOD
            )
            maximum_risk, breach_count, breach_dates = _actual_marked_risk(
                research, simulated["daily_nav"], simulated["positions"], optimizer_config
            )
            exit_only_events = _held_without_forecast_events(
                simulated["decisions"], decisions_by_method, MAIN_METHOD
            )
            summary.update({
                "scenario": label,
                "risk_budget_annualized": budget,
                "net_total_return": summary["total_return"],
                "maximum_marked_portfolio_annualized_risk": maximum_risk,
                "marked_risk_budget_breach_days": breach_count,
                "marked_risk_budget_breach_dates": breach_dates,
                "candidate_scope_audit": candidate_audit,
                "no_forecast_held_position_event_count": len(exit_only_events),
                "no_forecast_held_position_events": exit_only_events,
                "calibration_decision_date_count": len(point_corrected_forecasts),
                "execution_day_count": len(simulated["daily_nav"]),
                "calibration_source": str(calibration_path),
                "selection_scope": "pre-specified risk-budget comparison; no choice based on realized return",
            })
            _write_json(scenario_dir / "summary.json", summary)
            _write_json(scenario_dir / "candidate_scope_audit.json", candidate_audit)
            _write_json(scenario_dir / "no_forecast_exit_only_events.json", exit_only_events)
            _write_csv(scenario_dir / "daily_nav.csv", simulated["daily_nav"])
            _write_csv(scenario_dir / "orders.csv", simulated["orders"])
            _write_csv(scenario_dir / "positions.csv", simulated["positions"])
            with (scenario_dir / "simulator_decisions.jsonl").open(
                "w", encoding="utf-8", newline="\n"
            ) as target:
                for row in simulated["decisions"]:
                    target.write(json.dumps(
                        row, ensure_ascii=False, sort_keys=True, allow_nan=False
                    ) + "\n")
            results[label] = summary

        report = [
            "# P9 Pre-specified Risk Budget Sensitivity",
            "",
            f"Reused {len(point_corrected_forecasts)} exact rolling-calibration decisions from `{completed_dir}`. The point-corrected market-relative method, frozen candidates, starting cash/date, cost model, capacity, turnover, name cap, and simulator were held fixed. Only the hard annualized risk budget changed. These 15% and 25% scenarios were pre-specified comparisons; neither is selected based on its realized return.",
            "",
            "The last after-close forecast date is not counted as an executed NAV session unless a following session exists. See each scenario's decision ledger for the realized execution boundary.",
            "",
            "| Risk budget | Net return | Max drawdown | Mean cash | Fees | Filled notional | Fills | Marked risk max | Days above budget |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        main_summary = json.loads(
            (completed_dir / "market_relative_portfolio_mean_hard20" / "summary.json").read_text(
                encoding="utf-8"
            )
        )
        comparison_rows = [
            results["risk_budget_15pct"],
            {**main_summary, "risk_budget_annualized": 0.20},
            results["risk_budget_25pct"],
        ]
        for summary in comparison_rows:
            report.append(
                f"| {summary['risk_budget_annualized']:.0%} | {summary['net_total_return']:.2%} | {summary['maximum_drawdown']:.2%} | {summary['mean_cash_weight']:.1%} | ¥{summary['total_transaction_cost_cny']:,.0f} | ¥{summary['total_traded_notional_cny']:,.0f} | {summary['filled_orders']} | {summary['maximum_marked_portfolio_annualized_risk']:.2%} | {summary['marked_risk_budget_breach_days']} |"
            )
        report.extend([
            "",
            "Actual marked risk can exceed the decision-time target after overnight or intraday price moves. The report distinguishes those realized marked-risk breach days from the optimizer's hard decision-time budget.",
        ])
        (output_dir / "RESEARCH_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
        _write_json(output_dir / "variant_results.json", results)
        status.update({
            "status": "complete",
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            "last_progress": "pre-specified 15% and 25% budget replays completed",
            "result_report": str(output_dir / "RESEARCH_REPORT.md"),
        })
        _write_json(status_path, status)
        progress("pre-specified 15% and 25% budget replays completed")
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
