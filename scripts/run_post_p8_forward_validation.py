"""Freeze one daily forward-validation snapshot; exclude all already-viewed dates."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import os
import sys
import tomllib
from datetime import date, datetime, timezone
from pathlib import Path

from adaptive_mas.backtest.simulator import simulate_fixed_risk_market_exposure
from adaptive_mas.data import load_oos_research_data
from adaptive_mas.evaluation.mvp_research import (
    run_mvp_research,
    run_portfolio_comparisons,
)
from adaptive_mas.evaluation.post_p8 import expert_prediction_diagnostics
from adaptive_mas.evaluation.uncertainty import paired_block_bootstrap


VIEWED_THROUGH = date(2026, 9, 23)
DIAGNOSTIC_START = date(2026, 3, 16)
METHODS = {
    "trend",
    "reversal",
    "liquidity_confirmation",
    "equal_fusion",
    "adaptive_fusion",
    "loss_gated_fusion",
    "market_plus_relative_fusion",
}
REQUIRED_SEED_FIELDS = {
    "training_features",
    "training_returns",
    "calibration_features",
    "calibration_returns",
    "calibration_sufficient_statistics",
    "relative_calibration_features",
    "relative_calibration_returns",
    "relative_calibration_sufficient_statistics",
    "matured_losses",
    "matured_rank_ics",
    "matured_by_state",
}


def _paired_return_interval(
    left_rows: list[dict[str, object]],
    right_rows: list[dict[str, object]],
    seed: int,
) -> dict[str, object]:
    left_by_date = {str(row["trade_date"]): row for row in left_rows}
    right_by_date = {str(row["trade_date"]): row for row in right_rows}
    common_dates = sorted(left_by_date.keys() & right_by_date.keys())
    if not common_dates:
        raise ValueError("前向组合与固定基线没有共同净值观察日期")
    left_returns = [float(left_by_date[day]["daily_return"]) for day in common_dates]
    right_returns = [float(right_by_date[day]["daily_return"]) for day in common_dates]
    return {
        "paired_observation_dates": len(common_dates),
        "paired_start_date": common_dates[0],
        "paired_end_date": common_dates[-1],
        "paired_block_bootstrap_annualized_return_difference": paired_block_bootstrap(
            left_returns,
            right_returns,
            block_sessions=20,
            repetitions=2000,
            seed=seed,
        ),
    }


def _audit_optimizer_portfolio(
    portfolio: dict[str, object],
    method: str,
    config: dict[str, object],
) -> dict[str, object]:
    records = portfolio["artifacts"][method]
    daily = records["daily_nav"]
    orders = records["orders"]
    positions = records["positions"]
    decisions = records["decisions"]
    positions_by_date: dict[str, float] = {}
    for row in positions:
        day = str(row["trade_date"])
        positions_by_date[day] = positions_by_date.get(day, 0.0) + float(
            row["market_value_cny"]
        )
    ledger_ok = all(
        abs(
            float(row["cash_cny"])
            + positions_by_date.get(str(row["trade_date"]), 0.0)
            - float(row["nav_cny"])
        ) <= 1e-5
        for row in daily
    )
    cash_ok = all(float(row["cash_cny"]) >= -1e-7 for row in daily)
    weight_ok = all(
        float(row["target_cash_cny_before_costs"]) >= -1e-6
        and float(row["estimated_cash_after_costs_cny"]) >= -1e-6
        and max((float(value) for value in row["target_weights"].values()), default=0.0)
        <= float(config["maximum_symbol_weight"]) + 1e-7
        for row in decisions
    )
    risk_ok = all(
        float(row["annualized_risk"]) <= float(config["risk_budget_annualized"]) + 1e-7
        for row in decisions
    )
    turnover_ok = all(
        float(row["one_way_turnover"]) <= float(config["maximum_one_way_turnover"]) + 1e-7
        for row in decisions
    )
    capacity_ok = all(
        float(row["filled_notional_cny"]) <= float(row["requested_notional_cny"]) + 1e-7
        and float(row["filled_notional_cny"]) <= float(row["liquidity_limit_cny"]) + 1e-7
        for row in orders
    )
    costs_reconcile = abs(
        sum(float(row["cost_cny"]) for row in orders)
        - float(portfolio["summary"][method]["total_transaction_cost_cny"])
    ) <= 1e-5
    actual_max_weight = max(
        (float(row["portfolio_weight"]) for row in positions), default=0.0
    )
    checks = {
        "cash_nonnegative": cash_ok,
        "cash_plus_holdings_equals_nav": ledger_ok,
        "target_cash_and_symbol_caps": weight_ok,
        "optimizer_risk_budget": risk_ok,
        "optimizer_one_way_turnover_budget": turnover_ok,
        "fills_respect_requested_and_liquidity_amount": capacity_ok,
        "order_costs_reconcile_to_summary": costs_reconcile,
    }
    return {
        "method": method,
        "all_frozen_account_and_risk_checks_pass": all(checks.values()),
        "checks": checks,
        "decision_records": len(decisions),
        "filled_orders": portfolio["summary"][method]["filled_orders"],
        "blocked_orders": portfolio["summary"][method]["blocked_orders"],
        "partial_orders": portfolio["summary"][method]["partial_orders"],
        "maximum_realized_single_symbol_weight": actual_max_weight,
        "target_single_symbol_cap": float(config["maximum_symbol_weight"]),
        "note": "Realized weights may drift between rebalances; the hard cap check applies to each optimizer target.",
    }


def _forward_decision_assessment(
    portfolio: dict[str, object] | None,
    risk_baseline: dict[str, object] | None,
    config: dict[str, object],
    matured_date_count: int,
) -> dict[str, object]:
    if matured_date_count < 120:
        return {
            "status": "pending",
            "reason": "Fewer than 120 new decision dates have matured; no pass/fail conclusion is available.",
            "matured_new_decision_dates": matured_date_count,
            "required_matured_new_decision_dates": 120,
        }
    if portfolio is None or risk_baseline is None:
        return {
            "status": "insufficient_data",
            "reason": "No common executable portfolio ledger is available.",
            "matured_new_decision_dates": matured_date_count,
        }

    baseline_daily = {
        "cash": [
            {"trade_date": row["trade_date"], "daily_return": 0.0}
            for row in portfolio["artifacts"]["joint_market_relative"]["daily_nav"]
        ],
        "csi300_benchmark": [
            {
                "trade_date": row["trade_date"],
                "daily_return": float(row["benchmark_return"]),
            }
            for row in portfolio["artifacts"]["joint_market_relative"]["daily_nav"]
        ],
        "fixed_risk_market": risk_baseline["daily_nav"],
        "trend": portfolio["artifacts"]["trend"]["daily_nav"],
        "reversal": portfolio["artifacts"]["reversal"]["daily_nav"],
        "liquidity_confirmation": portfolio["artifacts"]["liquidity_confirmation"]["daily_nav"],
        "equal_fusion": portfolio["artifacts"]["equal_fusion"]["daily_nav"],
        "adaptive_fusion": portfolio["artifacts"]["adaptive_fusion"]["daily_nav"],
        "loss_gated_fusion": portfolio["artifacts"]["loss_gated_fusion"]["daily_nav"],
    }
    baseline_intervals = {
        name: _paired_return_interval(
            portfolio["artifacts"]["joint_market_relative"]["daily_nav"],
            rows,
            seed=20260926 + index,
        )
        for index, (name, rows) in enumerate(baseline_daily.items())
    }
    checks = {
        method: _audit_optimizer_portfolio(portfolio, method, config)
        for method in ("joint_optimization", "joint_market_relative")
    }
    primary_intervals_positive = all(
        float(
            record["paired_block_bootstrap_annualized_return_difference"][
                "annualized_return_difference_95pct_low"
            ]
        ) > 0
        for record in baseline_intervals.values()
    )
    primary = checks["joint_market_relative"]
    primary_net_contribution = float(
        portfolio["summary"]["joint_market_relative"]["total_return"]
    )
    passed = (
        primary_intervals_positive
        and primary["all_frozen_account_and_risk_checks_pass"]
        and primary_net_contribution > 0
    )
    return {
        "status": "pass" if passed else "fail",
        "primary_candidate": "joint_market_relative",
        "decision_rule": "Pass only if its 95% 20-session block interval is positive versus every predeclared fixed baseline, the realized net contribution versus cash is positive, and all account/risk checks pass.",
        "matured_new_decision_dates": matured_date_count,
        "paired_intervals_against_fixed_baselines": baseline_intervals,
        "all_primary_intervals_exclude_zero_in_positive_direction": primary_intervals_positive,
        "net_trade_contribution_vs_cash": {
            "net_total_return": primary_net_contribution,
            "net_profit_cny": float(
                portfolio["summary"]["joint_market_relative"]["final_nav_cny"]
                - portfolio["summary"]["joint_market_relative"]["initial_capital_cny"]
            ),
            "fees_cny": portfolio["summary"]["joint_market_relative"]["total_transaction_cost_cny"],
            "filled_orders": portfolio["summary"]["joint_market_relative"]["filled_orders"],
            "blocked_orders": portfolio["summary"]["joint_market_relative"]["blocked_orders"],
            "traded_notional_cny": portfolio["summary"]["joint_market_relative"]["total_traded_notional_cny"],
        },
        "optimizer_account_and_risk_checks": checks,
    }


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_jsonl_gzip(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("wb") as raw_output:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw_output, mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8", newline="\n") as output:
                for row in rows:
                    output.write(
                        json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                    )
                    output.write("\n")


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]), extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    project_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--baostock-run-dir", type=Path, required=True)
    parser.add_argument("--as-of-date", type=date.fromisoformat, required=True)
    parser.add_argument("--development-result", type=Path, required=True)
    parser.add_argument("--training-seed", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=project_dir / "configs" / "p8_daily_three_session.toml")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=project_dir / "experiments" / "post_p8_forward_validation",
    )
    args = parser.parse_args()
    if args.as_of_date <= VIEWED_THROUGH:
        raise ValueError(
            f"前向日期必须严格晚于已查看截止日 {VIEWED_THROUGH}; 已查看区间只用于状态暖启动"
        )
    if args.baostock_run_dir.resolve() == (project_dir / "data" / "raw" / "baostock" / "fetch-20260924T150656Z").resolve():
        raise ValueError("已查看诊断缓存不可作为新前向验证日；需要包含真实新日期的 BaoStock 采集目录")

    config_bytes = args.config.read_bytes()
    config = tomllib.loads(config_bytes.decode("utf-8"))
    config.setdefault("market_return_lookback_windows", 60)
    development = json.loads(args.development_result.read_text(encoding="utf-8"))
    seed_bytes = args.training_seed.read_bytes()
    if _sha256_bytes(seed_bytes) != development["artifacts"]["training_seed_sha256"]:
        raise ValueError("冻结开发结果与训练种子哈希不匹配")
    if _sha256_bytes(config_bytes) != development["config_sha256"]:
        raise ValueError("前向配置与冻结开发配置哈希不匹配")
    with gzip.open(io.BytesIO(seed_bytes), "rt", encoding="utf-8") as source:
        development_seed = json.load(source)
    if set(development_seed) != REQUIRED_SEED_FIELDS:
        raise ValueError("前向验证需要包含绝对/相对收益校准状态的新冻结开发种子")
    if len(development_seed["relative_calibration_features"]) != len(
        development_seed["relative_calibration_returns"]
    ):
        raise ValueError("相对收益校准特征与成熟标签数量不一致")
    if len(development_seed["relative_calibration_sufficient_statistics"]) != 3:
        raise ValueError("相对收益充分统计量与三个专家数量不一致")

    output_dir = args.output_root.resolve() / args.as_of_date.isoformat()
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=False)
    status_path = output_dir / "RUN_STATUS.json"
    progress_path = output_dir / "progress.log"
    command = [sys.executable, *sys.argv]
    started = datetime.now(timezone.utc).isoformat()
    status: dict[str, object] = {
        "status": "running",
        "pid": os.getpid(),
        "command": command,
        "output_dir": str(output_dir),
        "started_at_utc": started,
        "last_progress_at_utc": started,
    }

    def progress(message: str) -> None:
        status["last_progress_at_utc"] = datetime.now(timezone.utc).isoformat()
        status["last_progress"] = message
        _write_json(status_path, status)
        with progress_path.open("a", encoding="utf-8", newline="\n") as output:
            output.write(f"{status['last_progress_at_utc']} {message}\n")
        print(f"[forward-validation] {message}", flush=True)

    _write_json(status_path, status)
    try:
        progress(f"pid={os.getpid()} command={' '.join(command)} output={output_dir}")
        progress("loading explicit BaoStock cache; no local CSV is opened")
        research = load_oos_research_data(args.baostock_run_dir)
        if research.trading_dates[0] > DIAGNOSTIC_START:
            raise ValueError(
                f"BaoStock 运行目录须从已查看诊断起始日 {DIAGNOSTIC_START} 或更早开始，以提供决策/风险暖启动"
            )
        if research.trading_dates[-1] != args.as_of_date:
            raise ValueError(
                f"BaoStock 运行目录末日 {research.trading_dates[-1]} 必须等于 --as-of-date {args.as_of_date}；拒绝读取未来日期"
            )
        if VIEWED_THROUGH not in research.trading_dates:
            raise ValueError(f"BaoStock 运行目录缺少已查看状态切换日 {VIEWED_THROUGH}")
        fresh_dates = [day for day in research.trading_dates if day > VIEWED_THROUGH]
        if not fresh_dates:
            raise ValueError("没有 2026-09-23 之后的新 BaoStock 交易日")
        manifest_path = args.baostock_run_dir / "manifest.jsonl"
        manifest_hash = _sha256_bytes(manifest_path.read_bytes())
        progress(
            f"loaded {len(research.trading_dates)} dates; viewed warm-up ends {VIEWED_THROUGH}; new dates={len(fresh_dates)} through {args.as_of_date}"
        )

        progress("replaying already-viewed dates only to establish the frozen 2026-09-23 online state")
        viewed_state_run = run_mvp_research(
            research,
            config,
            initial_state=development_seed,
            as_of_date=VIEWED_THROUGH,
            ensure_as_of_decision=True,
            decision_start_date=research.trading_dates[0],
            progress_callback=lambda message: progress(f"viewed warm-up {message}"),
        )
        cutover_seed = viewed_state_run["training_seed"]
        cutover_seed_bytes = gzip.compress(
            json.dumps(cutover_seed, sort_keys=True, separators=(",", ":")).encode("utf-8"),
            mtime=0,
        )
        cutover_seed_path = output_dir / "cutover_state_asof_2026-09-23.json.gz"
        cutover_seed_path.write_bytes(cutover_seed_bytes)
        progress(
            f"cutover state frozen: matured Rank-IC windows={viewed_state_run['matured_rank_ic_windows_at_end']} relative calibration rows={len(cutover_seed['relative_calibration_returns'])}"
        )

        progress("running only decisions after 2026-09-23 from the frozen cutover state")
        forward_result = run_mvp_research(
            research,
            config,
            initial_state=cutover_seed,
            as_of_date=args.as_of_date,
            ensure_as_of_decision=True,
            comparison_start_date=fresh_dates[0],
            decision_start_date=fresh_dates[0],
            progress_callback=lambda message: progress(f"new-date replay {message}"),
        )
        forward_predictions = [
            row for row in forward_result["prediction_decisions"]
            if date.fromisoformat(row["prediction_date"]) > VIEWED_THROUGH
        ]
        if not forward_predictions:
            raise ValueError("新日期切片没有生成预测决策")
        settled_new_dates = {
            str(row["prediction_date"])
            for row in forward_result["prediction_settlements"]
            if row["status"] == "settled"
            and date.fromisoformat(str(row["prediction_date"])) > VIEWED_THROUGH
        }
        portfolio = None
        risk_baseline = None
        if len(fresh_dates) >= 2:
            portfolio = run_portfolio_comparisons(
                research,
                forward_result,
                config,
                execution_protocol="continuous_v1",
                progress_callback=lambda message: progress(f"new-date portfolio {message}"),
                common_start_date=fresh_dates[0],
                methods=METHODS,
                run_sensitivities=False,
            )
            risk_baseline = simulate_fixed_risk_market_exposure(
                research,
                fresh_dates[0],
                float(config["initial_capital_cny"]),
                float(config["risk_budget_annualized"]),
                int(config["risk_lookback_sessions"]),
                {
                    key: float(config[key])
                    for key in (
                        "commission_per_side_bps",
                        "transfer_fee_per_side_bps",
                        "stamp_duty_sell_bps",
                        "slippage_per_side_bps",
                    )
                },
            )
        diagnostics = expert_prediction_diagnostics(
            forward_result,
            forecast_error_multiplier=float(config["forecast_error_multiplier"]),
        )
        _write_jsonl_gzip(output_dir / "new_date_prediction_decisions.jsonl.gz", forward_predictions)
        _write_jsonl_gzip(output_dir / "new_date_prediction_settlements.jsonl.gz", forward_result["prediction_settlements"])
        _write_jsonl_gzip(output_dir / "new_date_observations.jsonl.gz", forward_result["observations"])
        _write_jsonl_gzip(output_dir / "new_date_decision_log.jsonl.gz", forward_result["decision_log"])
        new_state_bytes = gzip.compress(
            json.dumps(
                forward_result["training_seed"],
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8"),
            mtime=0,
        )
        (output_dir / "online_state_asof.json.gz").write_bytes(new_state_bytes)
        _write_json(output_dir / "prediction_diagnostics.json", diagnostics)
        if portfolio is not None:
            for method, rows in portfolio["artifacts"].items():
                method_dir = output_dir / "portfolios" / method
                method_dir.mkdir(parents=True, exist_ok=False)
                for kind in ("daily_nav", "orders", "positions"):
                    _write_csv(method_dir / f"{kind}.csv", rows[kind])
                _write_jsonl_gzip(method_dir / "decisions.jsonl.gz", rows["decisions"])
            for method, rows in portfolio["summary"].items():
                daily = portfolio["artifacts"][method]["daily_nav"]
                portfolio["summary"][method]["paired_block_bootstrap_vs_cash"] = paired_block_bootstrap(
                    [float(row["daily_return"]) for row in daily],
                    [0.0] * len(daily),
                    block_sessions=20,
                    repetitions=2000,
                    seed=20260926,
                )
        if risk_baseline is not None:
            _write_csv(output_dir / "fixed_risk_market_daily_nav.csv", risk_baseline["daily_nav"])
            _write_csv(output_dir / "fixed_risk_market_orders.csv", risk_baseline["orders"])
            _write_csv(output_dir / "fixed_risk_market_positions.csv", risk_baseline["positions"])

        evaluation_ready = len(settled_new_dates) >= 120
        decision_assessment = _forward_decision_assessment(
            portfolio,
            risk_baseline,
            config,
            len(settled_new_dates),
        )
        report = {
            "schema_version": "adaptive-mas.post-p8-forward.v1",
            "as_of_date": args.as_of_date.isoformat(),
            "training_source": "New-date market bars and outcomes come only from the explicit BaoStock OOS cache. The frozen development seed was built with train.csv plus BaoStock development data; test.csv was never used.",
            "baostock_run_dir": str(args.baostock_run_dir.resolve()),
            "baostock_manifest_sha256": manifest_hash,
            "config_sha256": _sha256_bytes(config_bytes),
            "development_training_seed_sha256": _sha256_bytes(seed_bytes),
            "cutover_training_seed_sha256": _sha256_bytes(cutover_seed_bytes),
            "viewed_diagnostic_warmup": {
                "start_date": research.trading_dates[0].isoformat(),
                "end_date": VIEWED_THROUGH.isoformat(),
                "decision_windows": viewed_state_run["decision_windows"],
                "matured_label_windows": viewed_state_run["evaluated_label_windows"],
                "counted_as_forward_validation": False,
            },
            "new_forward_validation": {
                "start_date": fresh_dates[0].isoformat(),
                "end_date": args.as_of_date.isoformat(),
                "new_trading_dates": len(fresh_dates),
                "new_prediction_dates": len(forward_predictions),
                "new_matured_decision_dates": len(settled_new_dates),
                "required_new_matured_decision_dates": 120,
                "required_fresh_trading_dates_for_120_matured_decisions_by_horizon": {
                    "one_session": 121,
                    "three_sessions": 123,
                    "five_sessions": 125,
                },
                "meets_120_matured_date_requirement": evaluation_ready,
                "status": "ready_for_predeclared_decision" if evaluation_ready else "awaiting_120_new_matured_decision_dates",
            },
            "frozen_config": config,
            "protocol": forward_result["protocol"],
            "prediction_quality": forward_result["prediction_quality"],
            "decision_value_quality": forward_result.get("decision_value_quality"),
            "prediction_and_portfolio_diagnostics": diagnostics,
            "portfolio_results": portfolio["summary"] if portfolio else None,
            "cost_sensitivity": None,
            "risk_sensitivity": None,
            "fixed_risk_market_baseline": risk_baseline["summary"] if risk_baseline else None,
            "predeclared_decision_assessment": decision_assessment,
            "claim": (
                decision_assessment["status"]
                if evaluation_ready
                else "pending: fewer than 120 new matured decision dates; no pass/fail conclusion is available."
            ),
        }
        _write_json(output_dir / "forward_validation_report.json", report)
        status["status"] = "complete"
        status["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        status["new_trading_dates"] = len(fresh_dates)
        status["new_matured_decision_dates"] = len(settled_new_dates)
        status["meets_120_matured_date_requirement"] = evaluation_ready
        status["report"] = str(output_dir / "forward_validation_report.json")
        progress(
            f"snapshot complete: new dates={len(fresh_dates)} matured new decisions={len(settled_new_dates)} ready={evaluation_ready}"
        )
        _write_json(status_path, status)
    except Exception as error:
        status["status"] = "failed"
        status["error_type"] = type(error).__name__
        status["error"] = str(error)
        status["failed_at_utc"] = datetime.now(timezone.utc).isoformat()
        _write_json(status_path, status)
        raise


if __name__ == "__main__":
    main()
