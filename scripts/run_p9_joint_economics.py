"""Replay P9 market-relative decisions with rolling portfolio-mean bounds."""

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
from collections import defaultdict
from datetime import date, datetime, timezone
from pathlib import Path

from adaptive_mas.backtest.simulator import simulate_portfolio
from adaptive_mas.data import load_research_data
from adaptive_mas.data.research import ResearchData
from adaptive_mas.decision import PortfolioState, optimize_forecast_target
from adaptive_mas.evaluation.conditional_mean import moving_block_mean_bounds
from adaptive_mas.evaluation.mvp_research import _risk_estimates
from adaptive_mas.evaluation.post_p8 import paired_block_mean_difference
from adaptive_mas.portfolio.allocation import top_fraction_weights
from adaptive_mas.portfolio.optimizer import PortfolioInfeasibleError


CALIBRATION_METHODS = ("trend", "loss_gated_fusion", "market_plus_relative_fusion")
MAIN_METHOD = "market_plus_relative_fusion"
NON_OVERLAP_STRIDE = 3
ROLLING_COHORTS = 40
MINIMUM_COHORTS = 20
BOOTSTRAP_BLOCK_COHORTS = 3
BOOTSTRAP_REPETITIONS = 2000
CONFIDENCE = 0.95


def _read_jsonl_gzip(path: Path) -> list[dict[str, object]]:
    with gzip.open(path, "rt", encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


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


def _source_summary(source: Path, name: str) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    directory = source / "portfolios" / name
    return _read_csv(directory / "daily_nav.csv"), _read_csv(directory / "orders.csv")


def _risk_of_weights(
    weights: dict[str, float],
    volatilities: dict[str, object],
    betas: dict[str, object],
    market_volatility: float,
) -> float | None:
    if set(weights) - set(volatilities) or set(weights) - set(betas):
        return None
    exposure = math.fsum(weights[symbol] * float(betas[symbol]) for symbol in weights)
    idiosyncratic = math.fsum(
        weights[symbol] ** 2
        * max(float(volatilities[symbol]) ** 2 - float(betas[symbol]) ** 2 * market_volatility**2, 0.0)
        for symbol in weights
    )
    return math.sqrt(idiosyncratic + (exposure * market_volatility) ** 2)


def _nav_summary(
    nav_rows: list[dict[str, object]] | list[dict[str, str]],
    order_rows: list[dict[str, object]] | list[dict[str, str]],
    initial_capital: float,
) -> dict[str, object]:
    if not nav_rows:
        raise ValueError("联合经济实验组合净值账为空")
    navs = [float(row["nav_cny"]) for row in nav_rows]
    peak = initial_capital
    drawdown = 0.0
    for nav in navs:
        peak = max(peak, nav)
        drawdown = min(drawdown, nav / peak - 1)
    cash = [float(row["cash_weight"]) for row in nav_rows if row.get("cash_weight") is not None]
    statuses: dict[str, int] = {}
    for row in order_rows:
        status = str(row["status"])
        if not status.startswith("filled"):
            statuses[status] = statuses.get(status, 0) + 1
    return {
        "start_date": str(nav_rows[0]["trade_date"]),
        "end_date": str(nav_rows[-1]["trade_date"]),
        "trading_days": len(nav_rows),
        "final_nav_cny": navs[-1],
        "net_total_return": navs[-1] / initial_capital - 1,
        "maximum_drawdown": drawdown,
        "mean_cash_weight": statistics.fmean(cash) if cash else None,
        "negative_cash_days": sum(
            row.get("cash_cny") is not None and float(row["cash_cny"]) < -0.01
            for row in nav_rows
        ),
        "total_transaction_cost_cny": math.fsum(float(row.get("cost_cny") or 0.0) for row in order_rows),
        "total_filled_notional_cny": math.fsum(float(row.get("filled_notional_cny") or row.get("notional_cny") or 0.0) for row in order_rows),
        "filled_order_count": sum(str(row["status"]).startswith("filled") for row in order_rows),
        "unfilled_order_status_counts": dict(sorted(statuses.items())),
        "daily_returns": [float(row["daily_return"]) for row in nav_rows],
    }


def _portfolio_cohorts(
    forecasts: list[dict[str, object]],
    settlements: list[dict[str, object]],
    research,
    *,
    top_fraction: float,
    invested_weight: float,
) -> dict[str, list[dict[str, object]]]:
    settlements_by_date: dict[str, dict[str, dict[str, object]]] = defaultdict(dict)
    for row in settlements:
        if row["status"] == "settled" and row.get("forward_return_3s") is not None:
            settlements_by_date[str(row["prediction_date"])][str(row["symbol"])] = row
    index = {row["trade_date"]: row for row in research.index_bars}
    cohorts: dict[str, list[dict[str, object]]] = {method: [] for method in CALIBRATION_METHODS}
    for source_index, forecast in enumerate(forecasts):
        if source_index % NON_OVERLAP_STRIDE:
            continue
        decision_text = str(forecast["prediction_date"])
        exit_text = forecast.get("planned_exit_date")
        entry_text = forecast.get("entry_date")
        if not exit_text or not entry_text:
            continue
        index_entry = index.get(date.fromisoformat(str(entry_text)))
        index_exit = index.get(date.fromisoformat(str(exit_text)))
        if (
            index_entry is None
            or index_exit is None
            or not index_entry.get("open")
            or not index_exit.get("close")
        ):
            continue
        realized_market = float(index_exit["close"]) / float(index_entry["open"]) - 1
        actual_by_symbol = settlements_by_date.get(decision_text, {})
        expected_by_method = forecast["expected_returns_by_method"]
        scores_by_method = forecast["scores_by_method"]
        market_by_symbol = expected_by_method["market_only"]
        weights_by_method = forecast["target_weights_by_method"]
        for method in CALIBRATION_METHODS:
            weights = weights_by_method.get(method, {})
            if not weights:
                # Before the frozen account start, P8 stored forecasts but no
                # account decisions. Reconstruct the exact static top-decile
                # diagnostic basket from that date's frozen score only.
                weights = top_fraction_weights(
                    scores_by_method.get(method, {}),
                    top_fraction,
                    invested_weight,
                )
            expected = expected_by_method.get(method, {})
            symbols = set(weights)
            if not symbols or symbols - set(expected) or symbols - set(actual_by_symbol) or symbols - set(market_by_symbol):
                continue
            weight_sum = math.fsum(float(weights[symbol]) for symbol in symbols)
            if weight_sum <= 0:
                continue
            predicted_total = math.fsum(float(weights[symbol]) * float(expected[symbol]) for symbol in symbols)
            predicted_market = math.fsum(float(weights[symbol]) * float(market_by_symbol[symbol]) for symbol in symbols)
            predicted_relative = predicted_total - predicted_market
            realized_total = math.fsum(
                float(weights[symbol]) * float(actual_by_symbol[symbol]["forward_return_3s"])
                for symbol in symbols
            )
            realized_market_component = weight_sum * realized_market
            realized_relative_component = realized_total - realized_market_component
            cohorts[method].append(
                {
                    "prediction_date": decision_text,
                    "label_end_date": str(exit_text),
                    "market_residual": realized_market_component - predicted_market,
                    "relative_residual": realized_relative_component - predicted_relative,
                    "total_residual": realized_total - predicted_total,
                    "predicted_market_component": predicted_market,
                    "predicted_relative_component": predicted_relative,
                    "predicted_total": predicted_total,
                    "realized_market_component": realized_market_component,
                    "realized_relative_component": realized_relative_component,
                    "realized_total": realized_total,
                    "constituent_count": len(symbols),
                    "invested_weight": weight_sum,
                }
            )
    return cohorts


def _calibration_for_decision(
    method_cohorts: dict[str, list[dict[str, object]]],
    decision_date: date,
    method: str,
) -> dict[str, object]:
    matured = [
        row
        for row in method_cohorts[method]
        if date.fromisoformat(str(row["label_end_date"])) < decision_date
    ][-ROLLING_COHORTS:]
    if len(matured) < MINIMUM_COHORTS:
        raise ValueError(
            f"{decision_date} 的 {method} 组合均值校准只有 {len(matured)} 个成熟非重叠样本，"
            f"少于冻结下限 {MINIMUM_COHORTS}"
        )
    seed_payload = f"{method}:{decision_date.isoformat()}".encode("ascii")
    seed = int(hashlib.sha256(seed_payload).hexdigest()[:8], 16)
    estimate = moving_block_mean_bounds(
        [
            (float(row["market_residual"]), float(row["relative_residual"]), float(row["total_residual"]))
            for row in matured
        ],
        block_length=BOOTSTRAP_BLOCK_COHORTS,
        repetitions=BOOTSTRAP_REPETITIONS,
        confidence=CONFIDENCE,
        seed=seed,
    )
    return {
        "method": method,
        "decision_date": decision_date.isoformat(),
        "matured_nonoverlapping_cohorts": estimate["observations"],
        "effective_bootstrap_blocks": estimate["effective_contiguous_blocks_per_replicate"],
        "window_start_date": matured[0]["prediction_date"],
        "window_end_label_date": matured[-1]["label_end_date"],
        "market_mean_bias_estimate": estimate["components"][0]["mean"],
        "market_bias_lower_95pct": estimate["components"][0]["one_sided_lower_bound"],
        "relative_mean_bias_estimate": estimate["components"][1]["mean"],
        "relative_bias_lower_95pct": estimate["components"][1]["one_sided_lower_bound"],
        "total_mean_bias_estimate": estimate["components"][2]["mean"],
        "total_bias_lower_95pct": estimate["components"][2]["one_sided_lower_bound"],
        "bootstrap": estimate,
    }


def _adjust_forecast(
    forecast: dict[str, object],
    corrections: dict[str, dict[str, object]],
    *,
    calibrated: bool,
    correction_field: str = "total_bias_lower_95pct",
) -> dict[str, object]:
    row = json.loads(json.dumps(forecast))
    if calibrated:
        for method in CALIBRATION_METHODS:
            correction = float(corrections[method][correction_field])
            values = row["expected_returns_by_method"][method]
            row["expected_returns_by_method"][method] = {
                symbol: float(value) + correction
                for symbol, value in values.items()
            }
            row["forecast_errors_by_method"][method] = 0.0
    return row


def _actual_marked_risk(research, nav_rows, positions, config) -> tuple[float, int, list[str]]:
    positions_by_date: dict[str, list[dict[str, object]]] = defaultdict(list)
    for position in positions:
        positions_by_date[str(position["trade_date"])].append(position)
    risks = []
    for nav_row in nav_rows:
        day_text = str(nav_row["trade_date"])
        symbols = {str(row["symbol"]) for row in positions_by_date[day_text]}
        if not symbols:
            risks.append((day_text, 0.0))
            continue
        decision_index = research.trading_dates.index(date.fromisoformat(day_text))
        vol, betas, market_vol = _risk_estimates(
            research,
            decision_index,
            symbols,
            int(config["risk_lookback_sessions"]),
            int(config["risk_minimum_observations"]),
        )
        if symbols - set(vol) or symbols - set(betas):
            raise ValueError(f"{day_text} 日终持仓缺少风险输入")
        weights = {
            str(row["symbol"]): float(row["market_value_cny"]) / float(nav_row["nav_cny"])
            for row in positions_by_date[day_text]
        }
        risk = _risk_of_weights(weights, vol, betas, market_vol)
        if risk is None:
            raise ValueError(f"{day_text} 日终组合风险无法计算")
        risks.append((day_text, risk))
    budget = float(config["risk_budget_annualized"])
    return (
        max((risk for _, risk in risks), default=0.0),
        sum(risk > budget + 1e-8 for _, risk in risks),
        [day for day, risk in risks if risk > budget + 1e-8],
    )


def _audit_candidate_scope(
    forecasts: list[dict[str, object]],
    decisions: list[dict[str, object]],
    method: str,
) -> dict[str, object]:
    forecast_by_date = {str(row["prediction_date"]): row for row in forecasts}
    checked = 0
    outside_basket_retained = 0
    outside_basket_increases = []
    for decision in decisions:
        day = str(decision["decision_date"])
        forecast = forecast_by_date[day]
        weights_by_method = forecast["target_weights_by_method"]
        method_weights = weights_by_method.get(method)
        if not isinstance(method_weights, dict) or not method_weights:
            raise ValueError(f"{day} 缺少 {method} 的冻结候选篮子")
        eligible = {
            str(symbol)
            for symbol, weight in method_weights.items()
            if float(weight) > 0
        }
        nav = float(decision["current_nav_cny"])
        current = {
            str(symbol): float(value) / nav
            for symbol, value in decision["current_holdings_cny"].items()
        }
        target = {str(symbol): float(value) for symbol, value in decision["target_weights"].items()}
        outside_basket_retained += sum(
            symbol not in eligible and target.get(symbol, 0.0) > 1e-10
            for symbol in current
        )
        for symbol, target_weight in target.items():
            if symbol not in eligible and target_weight > current.get(symbol, 0.0) + 1e-8:
                outside_basket_increases.append({
                    "decision_date": day,
                    "symbol": symbol,
                    "current_weight": current.get(symbol, 0.0),
                    "target_weight": target_weight,
                })
        checked += 1
    if outside_basket_increases:
        raise ValueError(
            f"{method} 优化在冻结候选篮子之外新增了目标持仓：{outside_basket_increases[:5]}"
        )
    return {
        "method": method,
        "decision_count": checked,
        "scope": "nonzero symbols in that date's frozen target_weights_by_method basket; existing holdings outside the basket may be retained or reduced but cannot be increased",
        "outside_basket_existing_positions_retained_count_across_decisions": outside_basket_retained,
        "outside_basket_additions": 0,
    }


def main() -> None:
    project_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-data-dir", type=Path, default=Path(r"C:\Users\xiao\Desktop\THU-BDC2026\data"))
    parser.add_argument("--baostock-run-dir", type=Path, default=project_dir / "data" / "raw" / "baostock" / "fetch-20260924T115050Z")
    parser.add_argument("--source-study-dir", type=Path, default=project_dir / "experiments" / "post_p8_research_20260926" / "three_session_daily")
    parser.add_argument("--output-dir", type=Path, default=project_dir / "experiments" / "p9_joint_economics_20260926")
    parser.add_argument("--smoke-decision-count", type=int, default=None)
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"P9 联合资金决策输出目录已存在，拒绝覆盖：{output_dir}")
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
        print(f"[P9 joint] {message}", flush=True)

    _write_json(status_path, status)
    source = args.source_study_dir.resolve()
    try:
        progress("loading train.csv, frozen P8 predictions, and the BaoStock development cache")
        research = load_research_data(args.training_data_dir, args.baostock_run_dir)
        config_path = source / "config.toml"
        config = tomllib.loads(config_path.read_text(encoding="utf-8"))
        results_path = source / "study_results.json"
        study_results = json.loads(results_path.read_text(encoding="utf-8"))
        common_start = date.fromisoformat(str(study_results["common_start_date"]))
        if common_start.isoformat() != str(study_results["protocol"]["comparison_start_date"]):
            raise ValueError("冻结共同起始日与协议 comparison_start_date 不一致")
        if common_start not in research.trading_dates:
            raise ValueError("冻结共同起始日不在 BaoStock 交易日历")
        forecasts = _read_jsonl_gzip(source / "prediction_decisions.jsonl.gz")
        settlements = _read_jsonl_gzip(source / "prediction_settlements.jsonl.gz")
        progress(f"loaded {len(forecasts)} frozen forecasts; building non-overlapping 3-session portfolio cohorts")
        cohorts = _portfolio_cohorts(
            forecasts,
            settlements,
            research,
            top_fraction=float(config["portfolio_top_fraction"]),
            invested_weight=float(config["invested_weight"]),
        )
        progress("built point-in-time cohort residuals for trend, loss-gated fusion, and market-relative fusion")

        calibration_by_date: dict[str, dict[str, dict[str, object]]] = {}
        calibration_rows = []
        forecasts_by_date = {
            str(row["prediction_date"]): row for row in forecasts
        }
        source_forecasts = [
            row for row in forecasts
            if date.fromisoformat(str(row["prediction_date"])) >= common_start
            and row.get("target_weights_by_method")
        ]
        if args.smoke_decision_count is not None:
            if args.smoke_decision_count < 1:
                raise ValueError("smoke decision count must be positive")
            source_forecasts = source_forecasts[:args.smoke_decision_count]
        for calibration_index, forecast in enumerate(source_forecasts):
            decision_date = date.fromisoformat(str(forecast["prediction_date"]))
            by_method = {
                method: _calibration_for_decision(cohorts, decision_date, method)
                for method in CALIBRATION_METHODS
            }
            calibration_by_date[decision_date.isoformat()] = by_method
            for method, estimate in by_method.items():
                current_cohort = next(
                    (row for row in cohorts[method] if row["prediction_date"] == decision_date.isoformat()),
                    None,
                )
                current = forecast["target_weights_by_method"].get(method, {})
                expected = forecast["expected_returns_by_method"].get(method, {})
                market = forecast["expected_returns_by_method"].get("market_only", {})
                predicted_total = math.fsum(float(weight) * float(expected[symbol]) for symbol, weight in current.items())
                predicted_market = math.fsum(float(weight) * float(market[symbol]) for symbol, weight in current.items())
                calibration_rows.append({
                    "decision_date": decision_date.isoformat(),
                    "method": method,
                    "matured_nonoverlapping_cohorts": estimate["matured_nonoverlapping_cohorts"],
                    "effective_bootstrap_blocks": estimate["effective_bootstrap_blocks"],
                    "window_start_date": estimate["window_start_date"],
                    "window_end_label_date": estimate["window_end_label_date"],
                    "market_forecast_component": predicted_market,
                    "relative_forecast_component": predicted_total - predicted_market,
                    "absolute_forecast_total": predicted_total,
                    "market_mean_bias_estimate": estimate["market_mean_bias_estimate"],
                    "relative_mean_bias_estimate": estimate["relative_mean_bias_estimate"],
                    "portfolio_mean_bias_estimate": estimate["total_mean_bias_estimate"],
                    "market_bias_lower_95pct": estimate["market_bias_lower_95pct"],
                    "relative_bias_lower_95pct": estimate["relative_bias_lower_95pct"],
                    "portfolio_bias_lower_95pct": estimate["total_bias_lower_95pct"],
                    "one_sided_lower_adjusted_portfolio_return": predicted_total + estimate["total_bias_lower_95pct"],
                    "matured_actual_total_for_this_decision": (
                        current_cohort["realized_total"] if current_cohort is not None else None
                    ),
                    "matured_actual_market_component_for_this_decision": (
                        current_cohort["realized_market_component"] if current_cohort is not None else None
                    ),
                    "matured_actual_relative_component_for_this_decision": (
                        current_cohort["realized_relative_component"] if current_cohort is not None else None
                    ),
                    "later_realized_value_used_for_diagnostic_only": True,
                })
            if (calibration_index + 1) % 20 == 0 or calibration_index + 1 == len(source_forecasts):
                progress(f"rolling point-in-time calibration {calibration_index + 1}/{len(source_forecasts)} decision dates complete")
        if not source_forecasts:
            raise ValueError("冻结比较期没有可校准预测")
        if date.fromisoformat(str(source_forecasts[0]["prediction_date"])) != common_start:
            raise ValueError("首个 P9 联合决策日与冻结共同起始日不一致")
        _write_csv(output_dir / "rolling_portfolio_mean_calibration.csv", calibration_rows)
        _write_jsonl = output_dir / "rolling_calibration_details.jsonl"
        with _write_jsonl.open("w", encoding="utf-8", newline="\n") as output:
            for day, methods in calibration_by_date.items():
                output.write(json.dumps({"decision_date": day, "by_method": methods}, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")

        replay_forecasts = [
            forecasts_by_date[day]
            for day in sorted(calibration_by_date)
        ]
        if not replay_forecasts:
            raise ValueError("联合经济实验没有可执行决策")
        if args.smoke_decision_count is not None:
            final_decision_date = date.fromisoformat(str(replay_forecasts[-1]["prediction_date"]))
            final_decision_index = research.trading_dates.index(final_decision_date)
            final_execution_index = final_decision_index + 1
            if final_execution_index >= len(research.trading_dates):
                raise ValueError("短跑末日没有下一交易日")
            final_execution_date = research.trading_dates[final_execution_index]
            research = ResearchData(
                trading_dates=[day for day in research.trading_dates if day <= final_execution_date],
                adjusted_bars=[row for row in research.adjusted_bars if row["trade_date"] <= final_execution_date],
                execution_bars=[row for row in research.execution_bars if row["trade_date"] <= final_execution_date],
                index_bars=[row for row in research.index_bars if row["trade_date"] <= final_execution_date],
                membership_snapshots=[snapshot for snapshot in research.membership_snapshots if snapshot.available_at <= final_execution_date],
            )
        legacy_mr_forecasts = [
            _adjust_forecast(row, calibration_by_date[str(row["prediction_date"])], calibrated=False)
            for row in replay_forecasts
        ]
        calibrated_forecasts = [
            _adjust_forecast(row, calibration_by_date[str(row["prediction_date"])], calibrated=True)
            for row in replay_forecasts
        ]
        mean_adjusted_forecasts = [
            _adjust_forecast(
                row,
                calibration_by_date[str(row["prediction_date"])],
                calibrated=True,
                correction_field="total_mean_bias_estimate",
            )
            for row in replay_forecasts
        ]
        costs = {
            key: float(config[key])
            for key in ("commission_per_side_bps", "transfer_fee_per_side_bps", "stamp_duty_sell_bps", "slippage_per_side_bps")
        }
        variants = (
            ("market_relative_legacy_rmse_soft8", MAIN_METHOD, legacy_mr_forecasts, 8.0),
            ("market_relative_portfolio_mean_soft8", MAIN_METHOD, mean_adjusted_forecasts, 8.0),
            ("market_relative_portfolio_mean_hard20", MAIN_METHOD, mean_adjusted_forecasts, 0.0),
            ("market_relative_portfolio_lcb_hard20", MAIN_METHOD, calibrated_forecasts, 0.0),
            ("trend_portfolio_mean_hard20", "trend", mean_adjusted_forecasts, 0.0),
        )
        variant_results: dict[str, dict[str, object]] = {}
        variant_dirs: dict[str, Path] = {}
        for variant_index, (name, return_method, decisions, aversion) in enumerate(variants):
            variant_dir = output_dir / name
            variant_dir.mkdir(parents=True, exist_ok=False)
            variant_dirs[name] = variant_dir
            optimizer_config = dict(config)
            optimizer_config["optimizer_return_method"] = return_method
            optimizer_config["optimizer_candidate_scope"] = "frozen_method_target"
            optimizer_config["risk_aversion"] = aversion
            if name != "market_relative_legacy_rmse_soft8":
                optimizer_config["forecast_error_multiplier"] = 0.0
            progress(f"same-simulator replay {variant_index + 1}/{len(variants)}: {name}; {len(decisions)} decisions")
            simulated = simulate_portfolio(
                research,
                [],
                float(config["initial_capital_cny"]),
                costs,
                int(config["liquidity_lookback_sessions"]),
                float(config["maximum_daily_participation"]),
                float(config["maximum_one_way_turnover"]),
                forecast_decisions=decisions,
                optimizer_config=optimizer_config,
                progress_callback=lambda message, label=name: progress(f"{label}: {message}"),
                evaluation_start_date=common_start,
            )
            summary = simulated.pop("summary")
            candidate_audit = _audit_candidate_scope(decisions, simulated["decisions"], return_method)
            _write_json(variant_dir / "candidate_scope_audit.json", candidate_audit)
            maximum_risk, breach_count, breach_dates = _actual_marked_risk(
                research, simulated["daily_nav"], simulated["positions"], config
            )
            summary.update({
                "net_total_return": summary["total_return"],
                "maximum_marked_portfolio_annualized_risk": maximum_risk,
                "marked_risk_budget_breach_days": breach_count,
                "marked_risk_budget_breach_dates": breach_dates,
                "decision_target_risk_budget_annualized": float(config["risk_budget_annualized"]),
                "decision_target_max_symbol_weight": float(config["maximum_symbol_weight"]),
                "decision_target_cash_nonnegative": True,
                "decision_target_maximum_one_way_turnover": float(config["maximum_one_way_turnover"]),
                "optimizer_return_method": return_method,
                "optimizer_risk_aversion": aversion,
                "portfolio_error_allowance_used": name == "market_relative_legacy_rmse_soft8",
                "negative_cash_days": sum(float(row["cash_cny"]) < -1e-8 for row in simulated["daily_nav"]),
                "candidate_scope_audit": candidate_audit,
            })
            variant_results[name] = {"summary": summary, **simulated}
            _write_json(variant_dir / "summary.json", summary)
            _write_csv(variant_dir / "daily_nav.csv", simulated["daily_nav"])
            _write_csv(variant_dir / "orders.csv", simulated["orders"])
            _write_csv(variant_dir / "positions.csv", simulated["positions"])
            _write_jsonl = variant_dir / "simulator_decisions.jsonl"
            with _write_jsonl.open("w", encoding="utf-8", newline="\n") as output:
                for row in simulated["decisions"]:
                    output.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")

        main_name = "market_relative_portfolio_mean_hard20"
        main_result = variant_results[main_name]
        main_forecast_by_date = {str(row["prediction_date"]): row for row in mean_adjusted_forecasts}
        main_decision_by_date = {
            str(row["decision_date"]): row for row in main_result["decisions"]
        }
        orders_by_decision: dict[str, list[dict[str, object]]] = defaultdict(list)
        for row in main_result["orders"]:
            orders_by_decision[str(row["decision_date"])].append(row)
        decision_economics = []
        trend_config = dict(config)
        trend_config["optimizer_return_method"] = "trend"
        trend_config["optimizer_candidate_scope"] = "frozen_method_target"
        trend_config["risk_aversion"] = 0.0
        trend_config["forecast_error_multiplier"] = 0.0
        for day, main_decision in main_decision_by_date.items():
            forecast = main_forecast_by_date.get(day)
            if forecast is None:
                continue
            decision_date = date.fromisoformat(day)
            current_cash = float(main_decision["current_cash_cny"])
            current_holdings = {str(k): float(v) for k, v in main_decision["current_holdings_cny"].items()}
            nav = float(main_decision["current_nav_cny"])
            current_weights = {symbol: value / nav for symbol, value in current_holdings.items()}
            estimate_by_method = calibration_by_date[day]
            trend_forecast = _adjust_forecast(
                forecast,
                estimate_by_method,
                calibrated=True,
                correction_field="total_mean_bias_estimate",
            )
            trend_result = None
            trend_error = None
            try:
                trend_result, _ = optimize_forecast_target(
                    research,
                    trend_forecast,
                    PortfolioState(current_cash, current_holdings),
                    trend_config,
                )
            except PortfolioInfeasibleError as error:
                trend_error = str(error)
            expected = forecast["expected_returns_by_method"][MAIN_METHOD]
            market_expected = forecast["expected_returns_by_method"]["market_only"]
            correction = float(estimate_by_method[MAIN_METHOD]["total_mean_bias_estimate"])
            hold_vol, hold_betas, hold_market_vol = _risk_estimates(
                research,
                research.trading_dates.index(decision_date),
                set(current_weights),
                int(config["risk_lookback_sessions"]),
                int(config["risk_minimum_observations"]),
            )
            hold_risk = _risk_of_weights(current_weights, hold_vol, hold_betas, hold_market_vol)
            holding_feasible = (
                hold_risk is not None
                and hold_risk <= float(config["risk_budget_annualized"]) + 1e-8
                and all(weight <= float(config["maximum_symbol_weight"]) + 1e-8 for weight in current_weights.values())
            )
            target = {str(k): float(v) for k, v in main_decision["target_weights"].items()}
            target_market_component = math.fsum(
                weight * float(market_expected.get(symbol, 0.0)) for symbol, weight in target.items()
            )
            target_raw_total = math.fsum(
                weight * float(expected.get(symbol, 0.0)) for symbol, weight in target.items()
            )
            target_relative_component = target_raw_total - target_market_component
            decision_orders = orders_by_decision.get(day, [])
            filled = math.fsum(float(row["filled_notional_cny"]) for row in decision_orders)
            fees = math.fsum(float(row["cost_cny"]) for row in decision_orders)
            target_cash_after_costs = float(main_decision["estimated_cash_after_costs_cny"])
            total_held_weight = math.fsum(current_weights.values())
            total_sale_capacity = (
                math.fsum(
                    min(current_weights[symbol], float(trend_result.capacity_weights.get(symbol, 0.0)))
                    for symbol in current_weights
                )
                if trend_result is not None else 0.0
            )
            full_cash_sale_feasible = (
                not current_holdings
                or (
                    trend_result is not None
                    and total_sale_capacity + 1e-8 >= total_held_weight
                    and total_held_weight <= 2 * float(config["maximum_one_way_turnover"]) + 1e-8
                )
            )
            sell_rate = (
                float(config["commission_per_side_bps"])
                + float(config["transfer_fee_per_side_bps"])
                + float(config["slippage_per_side_bps"])
                + float(config["stamp_duty_sell_bps"])
            ) / 10_000
            cash_value = -nav * total_held_weight * sell_rate if full_cash_sale_feasible else None
            alternatives: dict[str, float | None] = {
                "hold_cash_expected_value_cny": cash_value,
                "hold_existing_expected_value_cny": (
                    float(main_decision["holding_objective_value"]) * nav if holding_feasible else None
                ),
                "trend_candidate_expected_value_cny": (
                    float(trend_result.objective_value) * nav if trend_result is not None else None
                ),
                "fusion_candidate_expected_value_cny": float(main_decision["objective_value"]) * nav,
            }
            scored = [
                (label, value)
                for label, value in alternatives.items()
                if value is not None
            ]
            best_alternative = max(scored, key=lambda item: float(item[1]))[0]
            buy_notional = float(main_decision["buy_notional_cny"])
            if buy_notional > 0.01:
                action = f"buy ¥{buy_notional:,.0f}; leave ¥{target_cash_after_costs:,.0f} estimated cash"
            elif float(main_decision["sell_notional_cny"]) > 0.01:
                action = f"reduce by ¥{float(main_decision['sell_notional_cny']):,.0f}; estimated cash ¥{target_cash_after_costs:,.0f}"
            elif target:
                action = "hold current target; no positive incremental trade"
            else:
                action = "keep the target in cash"
            decision_economics.append({
                "decision_date": day,
                "execution_date": main_decision["execution_date"],
                "current_nav_cny": nav,
                "current_cash_cny": current_cash,
                "current_cash_weight": current_cash / nav,
                "current_holding_count": len(current_holdings),
                "hold_existing_feasible_under_hard_risk_and_name_limits": holding_feasible,
                "hold_existing_annualized_risk": hold_risk,
                "cash_reference_immediate_liquidation_feasible_under_capacity_and_turnover": full_cash_sale_feasible,
                "cash_reference_infeasible_reason": None if full_cash_sale_feasible else "current holdings exceed next-session sale capacity or turnover",
                "hold_cash_expected_value_cny": alternatives["hold_cash_expected_value_cny"],
                "hold_existing_expected_value_cny": alternatives["hold_existing_expected_value_cny"],
                "trend_candidate_expected_value_cny": alternatives["trend_candidate_expected_value_cny"],
                "trend_candidate_infeasible_reason": trend_error,
                "fusion_candidate_expected_value_cny": alternatives["fusion_candidate_expected_value_cny"],
                "highest_scored_alternative": best_alternative,
                "model_target_action": action,
                "target_invested_weight": math.fsum(target.values()),
                "target_cash_after_estimated_trade_cost_cny": target_cash_after_costs,
                "gross_forecast_market_component_return": target_market_component,
                "gross_forecast_relative_component_return": target_relative_component,
                "gross_forecast_absolute_return_before_bias_adjustment": target_raw_total,
                "portfolio_mean_bias_point_estimate": correction,
                "portfolio_mean_bias_one_sided_lower_95pct": float(estimate_by_method[MAIN_METHOD]["total_bias_lower_95pct"]),
                "gross_forecast_absolute_return_after_bias_adjustment": float(main_decision["gross_expected_return"]),
                "forecast_error_allowance_cny": float(main_decision["forecast_error_allowance"]) * nav,
                "estimated_transaction_cost_cny": float(main_decision["expected_transaction_cost_weight"]) * nav,
                "actual_next_session_filled_notional_cny": filled,
                "actual_next_session_transaction_cost_cny": fees,
                "risk_adjusted_incremental_value_vs_hold_cny": float(main_decision["incremental_objective_value"]) * nav,
                "estimated_target_annualized_risk": float(main_decision["annualized_risk"]),
                "target_one_way_turnover": float(main_decision["one_way_turnover"]),
            })

        # Match every strategy on the exact daily NAV dates produced by the new account.
        main_nav = main_result["daily_nav"]
        match_start, match_end = str(main_nav[0]["trade_date"]), str(main_nav[-1]["trade_date"])
        baseline_pairs = {}
        baseline_sources = {}
        for name in ("trend", "equal_fusion", "joint_optimization"):
            rows, orders = _source_summary(source, name)
            baseline_sources[name] = (
                [row for row in rows if match_start <= row["trade_date"] <= match_end],
                [row for row in orders if match_start <= row["trade_date"] <= match_end],
            )
        fixed_rows = _read_csv(source / "fixed_risk_market_daily_nav.csv")
        fixed_orders = _read_csv(source / "fixed_risk_market_orders.csv")
        baseline_sources["fixed_risk_csi300_price_proxy"] = (
            [row for row in fixed_rows if match_start <= row["trade_date"] <= match_end],
            [row for row in fixed_orders if match_start <= row["trade_date"] <= match_end],
        )
        csi_returns = {str(row["trade_date"]): float(row["benchmark_return"]) for row in main_nav}
        csi_nav = float(config["initial_capital_cny"])
        csi_rows = []
        for row in main_nav:
            csi_nav *= 1 + csi_returns[str(row["trade_date"])]
            csi_rows.append({"trade_date": row["trade_date"], "nav_cny": csi_nav, "cash_cny": None, "cash_weight": None, "daily_return": csi_returns[str(row["trade_date"])]})
        baseline_sources["csi300_price_proxy_gross"] = (csi_rows, [])
        initial_capital = float(config["initial_capital_cny"])
        pair_report: dict[str, object] = {}
        for variant_index, (name, result) in enumerate(variant_results.items()):
            summary = result["summary"]
            baseline_rows = {}
            for base_name, (rows, orders) in baseline_sources.items():
                baseline_rows[base_name] = _nav_summary(rows, orders, initial_capital)
            summary["paired_block_mean_difference_vs_baselines"] = {}
            sparse_dates = [str(row["trade_date"]) for row in result["daily_nav"]]
            sparse_returns = [float(row["daily_return"]) for row in result["daily_nav"]]
            left_map = dict(zip(sparse_dates, sparse_returns, strict=True))
            for baseline_index, (base_name, baseline) in enumerate(baseline_rows.items()):
                nav_dates = [str(row["trade_date"]) for row in baseline_sources[base_name][0]]
                right_map = dict(zip(nav_dates, baseline["daily_returns"], strict=True))
                common = sorted(left_map.keys() & right_map.keys())
                summary["paired_block_mean_difference_vs_baselines"][base_name] = {
                    **paired_block_mean_difference(
                        [left_map[day] for day in common],
                        [right_map[day] for day in common],
                        block_sessions=20,
                        repetitions=2000,
                        seed=20261001 + variant_index * 10 + baseline_index,
                    ),
                    "common_dates": len(common),
                    "common_start_date": common[0],
                    "common_end_date": common[-1],
                }
                del baseline["daily_returns"]
            pair_report[name] = {
                "summary": summary,
                "matched_baseline_summary": baseline_rows,
            }
            _write_json(variant_dirs[name] / "summary.json", summary)
            _write_json(variant_dirs[name] / "matched_baseline_summary.json", baseline_rows)

        _write_csv(output_dir / "daily_decision_economics.csv", decision_economics)
        _write_json(output_dir / "variant_results.json", pair_report)
        _write_json(output_dir / "frozen_protocol.json", {
            "method_version": "portfolio_mean_calibration_mbb_v1",
            "return_method": MAIN_METHOD,
            "return_decomposition": "market-index forecast component plus stock-relative-to-index component; residuals remain synchronized and are combined for the conservative total correction",
            "residual_anchor_stride_sessions": NON_OVERLAP_STRIDE,
            "residual_label_maturity_rule": "label_end_date strictly earlier than current after-close decision",
            "rolling_matured_cohort_window": ROLLING_COHORTS,
            "minimum_matured_nonoverlapping_cohorts": MINIMUM_COHORTS,
            "bootstrap": "circular moving blocks over synchronized three-session residual cohorts; block length three cohorts; two thousand replicates; one-sided 95% lower bound",
            "primary_calibrated_return_rule": "add the rolling portfolio-level residual mean point estimate to each candidate mean forecast for the selected method; do not add the per-name realized-return RMSE allowance",
            "conservative_uncertainty_comparator": "add the same-date rolling portfolio-level one-sided lower 95% residual mean bound; this is a predeclared conservative alternative, not asserted to be the uniquely correct mean estimate",
            "soft_risk_comparison": "risk aversion 8 versus hard-budget-only risk aversion 0 under the point-estimate calibration",
            "signal_comparison": "trend versus market-relative forecasts under the same point-estimate calibration, optimizer, and hard 20% risk budget",
            "optimizer_candidate_scope": "for each P9 method, expected-return decision variables are restricted to that date's frozen target_weights_by_method nonzero symbols; existing holdings outside the basket remain in the account state and can be sold subject to the same constraints",
            "hard_constraints": {
                "risk_budget_annualized": float(config["risk_budget_annualized"]),
                "maximum_symbol_weight": float(config["maximum_symbol_weight"]),
                "cash_nonnegative": True,
                "maximum_one_way_turnover": float(config["maximum_one_way_turnover"]),
                "maximum_daily_participation": float(config["maximum_daily_participation"]),
                "transaction_costs_bps": costs,
            },
            "common_start_date": common_start.isoformat(),
            "selection_scope": "development-only prequential replay; no parameter search; not forward validation",
            "smoke_run": args.smoke_decision_count is not None,
            "smoke_decision_count": len(replay_forecasts) if args.smoke_decision_count is not None else None,
        })
        report = [
            "# P9 Joint Economics: Rolling Portfolio Mean Calibration",
            "",
            "## Frozen method and point-in-time boundary",
            "",
            f"The account starts from cash at frozen after-close date {common_start}; results are matched through {match_end}. Calibration uses only non-overlapping three-session portfolio return cohorts. At each decision date, a training residual enters only when its label end date is strictly earlier than that decision. The most recent {ROLLING_COHORTS} cohorts are used, with a minimum of {MINIMUM_COHORTS}; synchronized market, relative, and total portfolio residuals are resampled in contiguous blocks of {BOOTSTRAP_BLOCK_COHORTS} cohorts for {BOOTSTRAP_REPETITIONS} replicates. The primary economic variant adds the rolling total residual mean point estimate to the selected conditional-mean forecast; a separate conservative comparator uses its one-sided 95% lower confidence bound. The interval is an alternative uncertainty treatment, not asserted as the uniquely correct mean estimate. Neither method divides RMSE by the square root of a sample count or assumes independent overlapping labels.",
            "",
            "Market return and stock-relative return remain separate in `rolling_portfolio_mean_calibration.csv`. The total residual is bootstrapped jointly so their covariance is retained. The later realized value columns in that artifact are evaluation-only diagnostics; they do not enter the correction on that row.",
            "",
            "## Same-simulator variants",
            "",
            "| Decision rule | Net return | Max drawdown | Mean cash | Fees | Filled notional | Fills | Negative cash days | Marked risk max / days over 20% | vs trend daily mean difference (95% block interval) |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        labels = {
            "market_relative_legacy_rmse_soft8": "Market-relative signal, legacy per-name RMSE allowance, soft risk 8",
            "market_relative_portfolio_mean_soft8": "Market-relative signal, portfolio mean point correction, soft risk 8",
            "market_relative_portfolio_mean_hard20": "Market-relative signal, portfolio mean point correction, hard 20% risk budget only",
            "market_relative_portfolio_lcb_hard20": "Market-relative signal, portfolio mean lower bound, hard 20% risk budget only",
            "trend_portfolio_mean_hard20": "Trend signal, portfolio mean point correction, hard 20% risk budget only",
        }
        for name, results in pair_report.items():
            summary = results["summary"]
            paired = summary["paired_block_mean_difference_vs_baselines"]["trend"]
            report.append(
                f"| {labels[name]} | {summary['net_total_return']:.2%} | {summary['maximum_drawdown']:.2%} | {summary['mean_cash_weight']:.1%} | ¥{summary['total_transaction_cost_cny']:,.0f} | ¥{summary['total_traded_notional_cny']:,.0f} | {summary['filled_orders']} | {summary['negative_cash_days']} | {summary['maximum_marked_portfolio_annualized_risk']:.2%} / {summary['marked_risk_budget_breach_days']} | {paired['mean_difference']:.5%} [{paired['mean_difference_95pct_low']:.5%}, {paired['mean_difference_95pct_high']:.5%}] |"
            )
        report.extend([
            "",
            "The legacy RMSE variant changes the selected return signal to market-relative forecasts while keeping the original error charge and risk aversion. Comparing it with the point-correction/soft-8 variant isolates the changed forecast-error price. Comparing point-correction soft risk 8 against hard-budget-only risk 0 isolates the optional soft-risk penalty. The lower-bound variant is a predeclared conservative uncertainty alternative. Comparing trend and market-relative point-correction hard-budget variants uses the same optimizer, date-specific nonzero top-decile candidate count, and capital rules, isolating the selected signal. Current holdings outside the active basket remain eligible for reduction. The P8 original loss-gated joint optimizer remains a separate same-window baseline in each paired table.",
            "",
            "Daily decisions record the market and relative forecast components, the portfolio-level correction, target risk, estimated trade costs, buy/sell amount, next-session fills, and a four-way expected-value screen for cash, current holdings, trend, and market-relative fusion. The screen uses a common net-return objective; its cash value includes liquidation fees and is unavailable when same-session turnover or capacity prevents a full sale. This is a decision diagnostic and does not switch the main market-relative account into another policy. A separate full trend account is replayed under the same corrected optimizer rules.",
            "",
            "P8's independent outcome-noise diagnostic for market-relative fusion reported 1.64% mean absolute portfolio error and 2.04% mean horizon portfolio sigma. Those describe realized horizon outcome dispersion; they are reported separately from this block-bootstrap uncertainty of the conditional portfolio mean. The mean correction does not remove market/relative price noise from the explicit 20% holdings-risk boundary.",
            "",
            "The CSI 300 references are price-only proxies. No ETF execution series is present in this replay. Development-period paired intervals are descriptive and do not establish future performance.",
            "",
            "## Rebuild artifacts",
            "",
            "`frozen_protocol.json`, `rolling_portfolio_mean_calibration.csv`, `daily_decision_economics.csv`, per-variant ledgers, `variant_results.json`, and `RUN_STATUS.json` record the method and outputs.",
        ])
        (output_dir / "RESEARCH_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
        _write_json(output_dir / "REPRODUCIBILITY.json", {
            "git_head": subprocess.run(["git", "rev-parse", "HEAD"], cwd=project_dir, check=True, capture_output=True, text=True).stdout.strip(),
            "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
            "study_results_sha256": hashlib.sha256(results_path.read_bytes()).hexdigest(),
            "predictions_sha256": hashlib.sha256((source / "prediction_decisions.jsonl.gz").read_bytes()).hexdigest(),
            "settlements_sha256": hashlib.sha256((source / "prediction_settlements.jsonl.gz").read_bytes()).hexdigest(),
            "baostock_manifest_sha256": hashlib.sha256((args.baostock_run_dir / "manifest.jsonl").read_bytes()).hexdigest(),
            "training_csv_sha256": hashlib.sha256((args.training_data_dir / "train.csv").read_bytes()).hexdigest(),
        })
        status.update({
            "status": "complete",
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            "last_progress": "rolling portfolio calibration and five joint same-simulator variants completed",
            "result_report": str(output_dir / "RESEARCH_REPORT.md"),
        })
        _write_json(status_path, status)
        progress("rolling portfolio calibration and five joint same-simulator variants completed")
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
