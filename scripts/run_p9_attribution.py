"""Build the P9 decision-attribution ledger from the completed post-P8 replay."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import statistics
import subprocess
import sys
import tomllib
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from adaptive_mas.data import load_research_data
from adaptive_mas.evaluation.decision_attribution import (
    attribute_optimizer_decision,
    summarize_next_session_execution,
)


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


def _candidate_capacity_weights(research, execution_by_symbol, decision_date: str, symbols: set[str], config, nav: float):
    from datetime import date

    day = date.fromisoformat(decision_date)
    index = research.trading_dates.index(day)
    lookback = int(config["liquidity_lookback_sessions"])
    participation = float(config["maximum_daily_participation"])
    dates = research.trading_dates[max(0, index - lookback + 1):index + 1]
    capacities = {}
    for symbol in symbols:
        amounts = [
            float(amount)
            for date_value in dates
            if (amount := execution_by_symbol.get(symbol, {}).get(date_value)) is not None
            and float(amount) > 0
        ]
        capacities[symbol] = (
            statistics.fmean(amounts) * participation / nav if amounts else 0.0
        )
    return capacities


def main() -> None:
    import math

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
        default=project_dir / "experiments" / "p9_return_first_20260926_v2",
    )
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"P9 输出目录已存在，拒绝覆盖：{output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)
    status_path = output_dir / "RUN_STATUS.json"
    progress_path = output_dir / "progress.log"
    status: dict[str, object] = {
        "status": "running",
        "pid": __import__("os").getpid(),
        "command": [sys.executable, *sys.argv],
        "output_dir": str(output_dir),
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "last_progress": "initializing",
    }

    def progress(message: str) -> None:
        status["last_progress_at_utc"] = datetime.now(timezone.utc).isoformat()
        status["last_progress"] = message
        _write_json(status_path, status)
        with progress_path.open("a", encoding="utf-8", newline="\n") as log:
            log.write(f"{status['last_progress_at_utc']} {message}\n")
        print(f"[P9 attribution] {message}", flush=True)

    _write_json(status_path, status)
    source = args.source_study_dir.resolve()
    try:
        progress("loading only train.csv and the frozen BaoStock research cache")
        research = load_research_data(args.training_data_dir, args.baostock_run_dir)
        execution_by_symbol: dict[str, dict[object, float]] = defaultdict(dict)
        for bar in research.execution_bars:
            amount = bar["amount"]
            if amount is not None:
                execution_by_symbol[str(bar["symbol"])][bar["trade_date"]] = float(amount)
        config_path = source / "config.toml"
        config_bytes = config_path.read_bytes()
        config = tomllib.loads(config_bytes.decode("utf-8"))
        prediction_path = source / "prediction_decisions.jsonl.gz"
        forecasts = _read_jsonl_gzip(prediction_path)
        prediction_by_date = {str(row["prediction_date"]): row for row in forecasts}

        portfolio_dir = source / "portfolios" / "joint_optimization"
        optimizer_rows = _read_jsonl_gzip(portfolio_dir / "decisions.jsonl.gz")
        orders = _read_csv(portfolio_dir / "orders.csv")
        daily_nav = _read_csv(portfolio_dir / "daily_nav.csv")
        positions = _read_csv(portfolio_dir / "positions.csv")
        orders_by_decision: dict[str, list[dict[str, str]]] = defaultdict(list)
        nav_by_date = {row["trade_date"]: row for row in daily_nav}
        positions_by_date: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in orders:
            orders_by_decision[row["decision_date"]].append(row)
        for row in positions:
            positions_by_date[row["trade_date"]].append(row)

        ledger = []
        progress(f"attributing {len(optimizer_rows)} same-account daily optimizer decisions")
        for count, decision in enumerate(optimizer_rows, start=1):
            decision_date = str(decision["decision_date"])
            forecast = prediction_by_date[decision_date]
            symbols = set(forecast["expected_returns_by_method"]["loss_gated_fusion"])
            symbols.update(str(symbol) for symbol in decision["current_holdings_cny"])
            capacity = _candidate_capacity_weights(
                research,
                execution_by_symbol,
                decision_date,
                symbols,
                config,
                float(decision["current_nav_cny"]),
            )
            row = attribute_optimizer_decision(forecast, decision, config, capacity)
            best = row["best_single_name_screen"]
            execution_date = decision.get("execution_date")
            execution_nav = nav_by_date.get(str(execution_date), {}) if execution_date else {}
            row["next_session_execution"] = summarize_next_session_execution(
                decision_date,
                str(execution_date) if execution_date else None,
                orders_by_decision.get(decision_date, []),
                execution_nav if execution_nav else None,
                positions_by_date.get(str(execution_date), []) if execution_date else [],
            )
            row["best_candidate_expected_gross_cny"] = (
                float(best["error_removed_probe_gross_return"]) * float(row["account_nav_cny"])
            )
            row["best_candidate_forecast_error_cny"] = (
                float(best["error_removed_probe_original_error_allowance"]) * float(row["account_nav_cny"])
            )
            row["best_candidate_transaction_cost_cny"] = (
                float(best["error_removed_probe_transaction_cost"]) * float(row["account_nav_cny"])
            )
            row["best_candidate_risk_penalty_cny"] = (
                float(best["error_removed_probe_risk_penalty_increment"]) * float(row["account_nav_cny"])
            )
            row["best_candidate_net_incremental_cny"] = (
                float(best["error_removed_probe_full_rule_net"]) * float(row["account_nav_cny"])
            )
            row["best_candidate_net_without_error_cny"] = (
                float(best["error_removed_probe_net_after_removing_error"]) * float(row["account_nav_cny"])
            )
            row["best_candidate_error_removed_probe_weight"] = float(best["delta_if_error_removed"])
            row["best_candidate_full_rule_optimal_weight"] = float(best["incremental_weight"])
            row["best_candidate_full_rule_optimal_net_cny"] = (
                float(best["net_incremental_return"]) * float(row["account_nav_cny"])
            )
            row["best_candidate_capacity_cny"] = (
                float(best["available_capacity_weight"]) * float(row["account_nav_cny"])
            )
            row["best_candidate_net_if_error_removed_cny"] = (
                float(best["error_removed_probe_net_after_removing_error"]) * float(row["account_nav_cny"])
            )
            row["best_candidate_net_if_fees_removed_cny"] = (
                float(best["net_if_fees_removed"]) * float(row["account_nav_cny"])
            )
            row["best_candidate_net_if_risk_removed_cny"] = (
                float(best["net_if_risk_penalty_removed"]) * float(row["account_nav_cny"])
            )
            row["best_candidate_net_without_capacity_cny"] = (
                float(best["net_without_capacity_limit"]) * float(row["account_nav_cny"])
            )
            ledger.append(row)
            if count % 50 == 0 or count == len(optimizer_rows):
                progress(f"attributed {count}/{len(optimizer_rows)} dates through {decision_date}")

        causes = Counter(str(row["first_blocking_reason_diagnostic"]) for row in ledger)
        total_nav = float(ledger[0]["account_nav_cny"]) if ledger else 0.0
        cause_summary = {
            "decision_dates": len(ledger),
            "zero_buy_decisions": sum(float(row["optimizer_buy_notional_cny"]) <= 0 for row in ledger),
            "positive_actual_buy_decisions": sum(float(row["optimizer_buy_notional_cny"]) > 0 for row in ledger),
            "first_reason_counts": dict(sorted(causes.items())),
            "any_blocker_counts": dict(sorted(Counter(
                cause
                for row in ledger
                for cause in row["blocking_reason_diagnostics"]
            ).items())),
            "mean_best_candidate_gross_cny_at_error_removed_probe": statistics.fmean(
                float(row["best_candidate_expected_gross_cny"]) for row in ledger
            ) if ledger else 0.0,
            "mean_best_candidate_error_cny_at_error_removed_probe": statistics.fmean(
                float(row["best_candidate_forecast_error_cny"]) for row in ledger
            ) if ledger else 0.0,
            "mean_best_candidate_fees_cny_at_error_removed_probe": statistics.fmean(
                float(row["best_candidate_transaction_cost_cny"]) for row in ledger
            ) if ledger else 0.0,
            "mean_best_candidate_risk_penalty_cny_at_error_removed_probe": statistics.fmean(
                float(row["best_candidate_risk_penalty_cny"]) for row in ledger
            ) if ledger else 0.0,
            "mean_best_candidate_full_rule_net_cny_at_error_removed_probe": statistics.fmean(
                float(row["best_candidate_net_incremental_cny"]) for row in ledger
            ) if ledger else 0.0,
            "mean_best_candidate_net_after_removing_error_cny": statistics.fmean(
                float(row["best_candidate_net_without_error_cny"]) for row in ledger
            ) if ledger else 0.0,
            "sum_realized_next_session_fills_cny": math.fsum(
                float(row["next_session_execution"]["filled_notional_cny"]) for row in ledger
            ),
            "sum_realized_next_session_fees_cny": math.fsum(
                float(row["next_session_execution"]["transaction_cost_cny"]) for row in ledger
            ),
            "starting_nav_cny": total_nav,
            "counterfactual_scope": "one-name-at-a-time screening from the recorded portfolio state; every removed economic term is re-optimized over its feasible one-name interval. It diagnoses reasons and does not alter the full portfolio optimizer or select thresholds.",
        }
        _write_csv(output_dir / "decision_attribution.csv", [
            {
                **{
                    key: value
                    for key, value in row.items()
                    if key not in {"best_single_name_screen", "blocking_reason_diagnostics", "current_holdings_cny", "optimizer_target_weights", "next_session_execution"}
                },
                "current_holdings_cny_json": json.dumps(row["current_holdings_cny"], sort_keys=True),
                "optimizer_target_weights_json": json.dumps(row["optimizer_target_weights"], sort_keys=True),
                "best_single_name_screen_json": json.dumps(row["best_single_name_screen"], sort_keys=True),
                "blocking_reason_diagnostics_json": json.dumps(row["blocking_reason_diagnostics"]),
                "next_session_execution_json": json.dumps(row["next_session_execution"], ensure_ascii=False, sort_keys=True),
            }
            for row in ledger
        ])
        _write_json(output_dir / "reason_distribution.json", cause_summary)

        diagnostics_path = source / "prediction_and_portfolio_diagnostics.json"
        diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
        by_uncertainty = diagnostics["portfolio_uncertainty_vs_risk_diagnostic"]["by_allocation_method"]
        main_uncertainty = by_uncertainty.get("joint_optimization")
        if main_uncertainty is None:
            main_uncertainty = by_uncertainty.get("joint_market_relative")
        forward_path = source.parent / "forward_validation_status.json"
        forward_status = json.loads(forward_path.read_text(encoding="utf-8"))
        report_lines = [
            "# P9 Return-First: Daily Decision Attribution",
            "",
            "This add-on reuses the completed, frozen P8/post-P8 development replay. It does not replace that experiment or re-run the optimizer with different thresholds.",
            "",
            "## Scope and data boundary",
            "",
            f"- Decisions: {len(ledger)} same-account three-session daily decisions, {ledger[0]['decision_date'] if ledger else 'n/a'} through {ledger[-1]['decision_date'] if ledger else 'n/a'}.",
            f"- Data: BaoStock cache `{args.baostock_run_dir}` plus `train.csv` only; local `test.csv` was not opened.",
            f"- Per-name cap: {float(config['maximum_symbol_weight']):.1%}; annualized hard risk budget: {float(config['risk_budget_annualized']):.1%}; per-day participation: {float(config['maximum_daily_participation']):.2%}; total buy fee rate: {(float(config['commission_per_side_bps']) + float(config['transfer_fee_per_side_bps']) + float(config['slippage_per_side_bps'])):.1f} bps.",
            "- The exact full-portfolio optimizer result, next-session orders/fills, costs, holdings and NAV are copied from the completed simulator ledgers. The additional counterfactual evaluates one name at a time from that recorded account state and re-optimizes its own position for each term removal.",
            "",
            "## Findings",
            "",
            f"- The optimizer authorized buys on {cause_summary['positive_actual_buy_decisions']} of {cause_summary['decision_dates']} dates. Recorded next-session filled notional was ¥{cause_summary['sum_realized_next_session_fills_cny']:,.0f}, with ¥{cause_summary['sum_realized_next_session_fees_cny']:,.0f} in fees.",
            f"- Primary single-name screen reason counts: `{json.dumps(cause_summary['first_reason_counts'], ensure_ascii=False, sort_keys=True)}`.",
            f"- At the one-name position size that maximizes utility after removing only the forecast-error charge, the strongest candidate averaged gross ¥{cause_summary['mean_best_candidate_gross_cny_at_error_removed_probe']:,.0f}; under the original rule that same probe would still pay ¥{cause_summary['mean_best_candidate_error_cny_at_error_removed_probe']:,.0f} forecast-error allowance, ¥{cause_summary['mean_best_candidate_fees_cny_at_error_removed_probe']:,.0f} fees, and ¥{cause_summary['mean_best_candidate_risk_penalty_cny_at_error_removed_probe']:,.0f} incremental risk penalty, for full-rule net ¥{cause_summary['mean_best_candidate_full_rule_net_cny_at_error_removed_probe']:,.0f}. Removing only the error allowance yields ¥{cause_summary['mean_best_candidate_net_after_removing_error_cny']:,.0f} net. These are re-optimized diagnostic probes, not investable or realized returns.",
            f"- The exact optimizer's account-level objective increment is recorded daily in `decision_attribution.csv`; the full P8 comparison and 20-session paired intervals remain in `{source.parent / 'RESEARCH_REPORT.md'}`.",
            "",
            "## Typical decisions",
            "",
            "| Decision date | Reason | Best name | Forecast | Error-relaxed probe weight | Gross at probe (CNY) | Original error (CNY) | Fees (CNY) | Risk change (CNY) | Full-rule net at probe (CNY) | Net after removing error (CNY) | Full-rule optimal weight | Optimizer buy (CNY) | Next-session fill (CNY) |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        example_indexes = sorted(set([0, len(ledger) // 2, len(ledger) - 1])) if ledger else []
        for index in example_indexes:
            row = ledger[index]
            best = row["best_single_name_screen"]
            report_lines.append(
                f"| {row['decision_date']} | {row['first_blocking_reason_diagnostic']} | {best['symbol']} | {float(best['expected_return']):.3%} | {float(best['delta_if_error_removed']):.3%} | {row['best_candidate_expected_gross_cny']:,.0f} | {row['best_candidate_forecast_error_cny']:,.0f} | {row['best_candidate_transaction_cost_cny']:,.0f} | {row['best_candidate_risk_penalty_cny']:,.0f} | {row['best_candidate_net_incremental_cny']:,.0f} | {row['best_candidate_net_without_error_cny']:,.0f} | {row['best_candidate_full_rule_optimal_weight']:.3%} | {row['optimizer_buy_notional_cny']:,.0f} | {row['next_session_execution']['filled_notional_cny']:,.0f} |"
            )
        report_lines.extend([
            "",
            "## Uncertainty versus risk",
            "",
            "The P8 diagnostic reports forecast allowance beside historical factor-plus-idiosyncratic portfolio horizon sigma. Because the joint optimizer was fully in cash in this window, there is no nonzero joint-optimizer portfolio residual series with which to validate overlap. Existing nonzero strategy portfolios show that both measures can be of similar order; this is a warning against treating both as independently calibrated risk prices, not evidence that either should be reduced to induce trades.",
            "",
            f"Selected nonzero allocation uncertainty diagnostic: `{json.dumps(main_uncertainty, ensure_ascii=False, sort_keys=True) if main_uncertainty else 'not available'}`.",
            "",
            "## Forward and market-data status",
            "",
            f"- Forward status from the frozen protocol: `{json.dumps(forward_status, ensure_ascii=False, sort_keys=True)}`. Already-viewed dates through 2026-09-23 are not forward validation; at least 120 newly matured decisions are still required.",
            "- The frozen BaoStock cache contains CSI 300 index prices and constituent stock bars, but no verified ETF instrument/quote/fill record. The CSI 300 price proxy remains non-tradable and is not relabeled as an ETF.",
            "- This run does not include the sparse matured-net-contribution fusion comparison; that remains a separate P9 implementation item and must use a newly frozen method with the same simulator/account before drawing any conclusion.",
            "",
            "## Rebuild",
            "",
            "Run `scripts/run_p9_attribution.py` with the exact paths recorded in `RUN_STATUS.json`. It refuses to overwrite an existing output directory. Inputs include the frozen three-session P8 replay and the development BaoStock cache; no score file is an input.",
        ])
        (output_dir / "RESEARCH_REPORT.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
        fingerprints = {
            "source_study_dir": str(source),
            "source_prediction_sha256": hashlib.sha256(prediction_path.read_bytes()).hexdigest(),
            "source_optimizer_decisions_sha256": hashlib.sha256((portfolio_dir / "decisions.jsonl.gz").read_bytes()).hexdigest(),
            "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
            "git_head": subprocess.run(["git", "rev-parse", "HEAD"], cwd=project_dir, check=True, capture_output=True, text=True).stdout.strip(),
            "data_manifest_sha256": hashlib.sha256((args.baostock_run_dir / "manifest.jsonl").read_bytes()).hexdigest(),
        }
        _write_json(output_dir / "REPRODUCIBILITY.json", fingerprints)
        status.update({
            "status": "complete",
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            "last_progress": f"completed daily attribution for {len(ledger)} decisions",
            "result_report": str(output_dir / "RESEARCH_REPORT.md"),
        })
        _write_json(status_path, status)
        progress(f"completed daily attribution for {len(ledger)} decisions")
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
