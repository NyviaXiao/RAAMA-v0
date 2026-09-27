"""Run the predeclared 1/3/5-session and review-frequency development study."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import math
import os
import statistics
import subprocess
import sys
import time
import tomllib
from datetime import date, datetime, timezone
from pathlib import Path

from adaptive_mas.backtest.simulator import simulate_fixed_risk_market_exposure
from adaptive_mas.data import load_research_data
from adaptive_mas.evaluation.mvp_research import (
    run_mvp_research,
    run_portfolio_comparisons,
)
from adaptive_mas.evaluation.post_p8 import expert_prediction_diagnostics
from adaptive_mas.evaluation.post_p8_report import render_report
from adaptive_mas.evaluation.uncertainty import paired_block_bootstrap


CONFIGS = {
    "one_session_daily": "p8_daily_one_session.toml",
    "three_session_daily": "p8_daily_three_session.toml",
    "five_session_daily": "p8_daily_five_session.toml",
    "three_session_review_every_five": "p8_five_session_review_three_session.toml",
}
ALPHA_METHODS = {
    "trend",
    "reversal",
    "liquidity_confirmation",
    "equal_fusion",
    "adaptive_fusion",
    "loss_gated_fusion",
    "market_plus_relative_fusion",
}
SOURCE_FILES = (
    "scripts/run_post_p8_study.py",
    "scripts/render_post_p8_report.py",
    "scripts/run_mvp_research.py",
    "scripts/run_mvp_oos_research.py",
    "scripts/run_post_p8_forward_validation.py",
    "src/adaptive_mas/evaluation/mvp_research.py",
    "src/adaptive_mas/evaluation/post_p8.py",
    "src/adaptive_mas/evaluation/post_p8_report.py",
    "src/adaptive_mas/backtest/simulator.py",
    "src/adaptive_mas/decision.py",
    "src/adaptive_mas/portfolio/optimizer.py",
    "src/adaptive_mas/agents/quantitative.py",
    "src/adaptive_mas/targets/five_session.py",
    "src/adaptive_mas/data/research.py",
    "src/adaptive_mas/data/market_data.py",
)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_jsonl_gzip(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as raw_output:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw_output, mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8", newline="\n") as output:
                for row in rows:
                    output.write(
                        json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                    )
                    output.write("\n")


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    columns = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=columns, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def _source_fingerprints(project_dir: Path) -> dict[str, object]:
    hashes = {
        name: _sha256_bytes((project_dir / name).read_bytes())
        for name in SOURCE_FILES
    }
    digest = hashlib.sha256()
    for name, value in sorted(hashes.items()):
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(value.encode("ascii"))
        digest.update(b"\n")
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=project_dir,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--short"],
        cwd=project_dir,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return {
        "git_head": revision,
        "git_worktree_dirty": bool(dirty.strip()),
        "relevant_source_file_sha256": hashes,
        "relevant_source_bundle_sha256": digest.hexdigest(),
    }


def _summarize_returns(
    returns: list[float], initial_capital_cny: float, *, gross: bool
) -> dict[str, float | int]:
    if not returns:
        raise ValueError("组合基线没有可评价日收益")
    nav = initial_capital_cny
    peak = nav
    maximum_drawdown = 0.0
    for daily_return in returns:
        nav *= 1 + daily_return
        peak = max(peak, nav)
        maximum_drawdown = min(maximum_drawdown, nav / peak - 1)
    volatility = statistics.stdev(returns) * (250**0.5) if len(returns) > 1 else 0.0
    return {
        "evaluation_dates": len(returns),
        "final_nav_cny": nav,
        "net_total_return" if not gross else "gross_total_return": nav / initial_capital_cny - 1,
        "annualized_volatility": volatility,
        "maximum_drawdown": maximum_drawdown,
        "mean_daily_return": statistics.fmean(returns),
    }


def _common_daily_rows(records: dict[str, object]) -> list[dict[str, object]]:
    rows = records["daily_nav"]
    if not rows:
        raise ValueError("组合账本没有日净值记录")
    return rows


def _matched_interval(
    left_rows: list[dict[str, object]], right_rows: list[dict[str, object]], seed: int
) -> dict[str, object]:
    left_by_date = {str(row["trade_date"]): row for row in left_rows}
    right_by_date = {str(row["trade_date"]): row for row in right_rows}
    common = sorted(left_by_date.keys() & right_by_date.keys())
    if not common:
        raise ValueError("比较策略之间没有共同的组合观察日期")
    left_returns = [float(left_by_date[day]["daily_return"]) for day in common]
    right_returns = [float(right_by_date[day]["daily_return"]) for day in common]
    return {
        "paired_observation_dates": len(common),
        "paired_start_date": common[0],
        "paired_end_date": common[-1],
        "same_initial_capital": True,
        "account_start_state": "cash on the shared after-close decision date",
        "paired_block_bootstrap": paired_block_bootstrap(
            left_returns, right_returns, block_sessions=20, repetitions=2000, seed=seed
        ),
    }


def _benchmark_rows(
    nav_rows: list[dict[str, object]], initial_capital_cny: float
) -> dict[str, dict[str, object]]:
    zero = [0.0] * len(nav_rows)
    index_returns = [float(row["benchmark_return"]) for row in nav_rows]
    cash_summary = _summarize_returns(zero, initial_capital_cny, gross=False)
    cash_summary.update(
        {
            "average_cash_weight": 1.0,
            "total_cost_cny": 0.0,
            "total_traded_notional_cny": 0.0,
            "filled_orders": 0,
            "blocked_orders": 0,
            "baseline_scope": "cash account from the shared start date",
        }
    )
    index_summary = _summarize_returns(index_returns, initial_capital_cny, gross=True)
    index_summary.update(
        {
            "average_cash_weight": None,
            "total_cost_cny": 0.0,
            "total_traded_notional_cny": 0.0,
            "filled_orders": 0,
            "blocked_orders": 0,
            "baseline_scope": "BaoStock CSI 300 price return; gross index proxy without an ETF fee/track record",
        }
    )
    return {"cash": cash_summary, "csi300_benchmark": index_summary}


def _save_replay_artifacts(
    directory: Path,
    result: dict[str, object],
    portfolio: dict[str, object],
    seed_bytes: bytes,
) -> dict[str, object]:
    training_seed_path = directory / "training_seed.json.gz"
    training_seed_path.write_bytes(seed_bytes)
    _write_jsonl_gzip(directory / "observations.jsonl.gz", result["observations"])
    _write_jsonl_gzip(directory / "decision_log.jsonl.gz", result["decision_log"])
    _write_jsonl_gzip(directory / "prediction_decisions.jsonl.gz", result["prediction_decisions"])
    _write_jsonl_gzip(directory / "prediction_settlements.jsonl.gz", result["prediction_settlements"])
    artifacts = {}
    for method, rows in portfolio["artifacts"].items():
        method_dir = directory / "portfolios" / method
        method_dir.mkdir(parents=True, exist_ok=False)
        for kind in ("daily_nav", "orders", "positions"):
            relative_path = method_dir / f"{kind}.csv"
            _write_csv(relative_path, rows[kind])
            artifacts[f"{method}_{kind}"] = relative_path.relative_to(directory).as_posix()
        decisions_path = method_dir / "decisions.jsonl.gz"
        _write_jsonl_gzip(decisions_path, rows["decisions"])
        artifacts[f"{method}_decisions"] = decisions_path.relative_to(directory).as_posix()
    return {
        "training_seed": training_seed_path.relative_to(directory).as_posix(),
        "training_seed_sha256": _sha256_bytes(seed_bytes),
        "prediction_decisions": "prediction_decisions.jsonl.gz",
        "prediction_settlements": "prediction_settlements.jsonl.gz",
        "observations": "observations.jsonl.gz",
        "decision_log": "decision_log.jsonl.gz",
        "portfolio_ledgers": artifacts,
    }


def _format_percent(value: object) -> str:
    return "n/a" if value is None else f"{float(value):.2%}"


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
        "--output-dir",
        type=Path,
        default=project_dir / "experiments" / "post_p8_research_20260926",
    )
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"研究输出目录已存在，拒绝覆盖：{output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)
    progress_path = output_dir / "progress.log"
    status_path = output_dir / "RUN_STATUS.json"
    command = [sys.executable, *sys.argv]
    started_at = datetime.now(timezone.utc).isoformat()
    status: dict[str, object] = {
        "status": "running",
        "pid": os.getpid(),
        "command": command,
        "output_dir": str(output_dir),
        "started_at_utc": started_at,
        "last_progress_at_utc": started_at,
        "last_progress": "initializing",
    }

    def progress(message: str) -> None:
        status["last_progress_at_utc"] = datetime.now(timezone.utc).isoformat()
        status["last_progress"] = message
        _write_json(status_path, status)
        with progress_path.open("a", encoding="utf-8", newline="\n") as output:
            output.write(f"{status['last_progress_at_utc']} {message}\n")
        print(f"[post-P8] {message}", flush=True)

    _write_json(status_path, status)
    progress(f"pid={os.getpid()} command={' '.join(command)} output={output_dir}")
    try:
        progress("loading only train.csv and the frozen BaoStock development cache")
        research = load_research_data(args.training_data_dir, args.baostock_run_dir)
        if research.trading_dates[0] != date(2024, 1, 2) or research.trading_dates[-1] != date(2026, 3, 6):
            raise ValueError(
                "开发研究日期必须严格为 2024-01-02 至 2026-03-06；拒绝把已查看诊断区间并入开发样本"
            )
        train_path = args.training_data_dir / "train.csv"
        manifest_path = args.baostock_run_dir / "manifest.jsonl"
        manifest_bytes = manifest_path.read_bytes()
        provenance = {
            "market_source": "BaoStock only",
            "local_research_input": "train.csv only",
            "train_csv_sha256": _sha256_bytes(train_path.read_bytes()),
            "baostock_manifest_sha256": _sha256_bytes(manifest_bytes),
            "baostock_run_dir": str(args.baostock_run_dir.resolve()),
            "development_start_date": research.trading_dates[0].isoformat(),
            "development_end_date": research.trading_dates[-1].isoformat(),
            "development_trading_dates": len(research.trading_dates),
            "adjusted_bars": len(research.adjusted_bars),
            "execution_bars": len(research.execution_bars),
            "index_bars": len(research.index_bars),
            "historical_membership_snapshots": len(research.membership_snapshots),
        }
        fingerprints = _source_fingerprints(project_dir)
        _write_json(output_dir / "source_fingerprints.json", fingerprints)
        _write_json(output_dir / "data_provenance.json", provenance)
        progress(
            f"loaded {len(research.trading_dates)} development dates through {research.trading_dates[-1]}; no already-viewed dates are in this run"
        )

        config_payloads = {}
        start_dates = {}
        config_bytes_by_name = {}
        for study_name, config_name in CONFIGS.items():
            config_path = project_dir / "configs" / config_name
            config_bytes = config_path.read_bytes()
            config = tomllib.loads(config_bytes.decode("utf-8"))
            config.setdefault("market_return_lookback_windows", 60)
            config_payloads[study_name] = (config_path, config)
            config_bytes_by_name[study_name] = config_bytes
            progress(f"warm-up readiness pass: {study_name}")
            warmup_result = run_mvp_research(
                research,
                config,
                progress_callback=lambda message, label=study_name: progress(
                    f"{label} readiness {message}"
                ),
            )
            start_dates[study_name] = date.fromisoformat(
                warmup_result["protocol"]["initial_comparison_date"]
            )
            progress(f"{study_name} first eligible decision: {start_dates[study_name]}")
            del warmup_result

        common_start_date = max(start_dates.values())
        progress(
            f"common matched-account start is {common_start_date}; all four portfolios start from cash and are marked from the next session"
        )
        study_outputs: dict[str, dict[str, object]] = {}
        portfolio_paths: dict[str, dict[str, list[dict[str, object]]]] = {}
        for study_index, (study_name, (config_path, config)) in enumerate(config_payloads.items()):
            study_dir = output_dir / study_name
            study_dir.mkdir(parents=True, exist_ok=False)
            config_bytes = config_bytes_by_name[study_name]
            (study_dir / "config.toml").write_bytes(config_bytes)
            progress(f"frozen-date replay starting: {study_name}; common start {common_start_date}")
            result = run_mvp_research(
                research,
                config,
                comparison_start_date=common_start_date,
                progress_callback=lambda message, label=study_name: progress(
                    f"{label} {message}"
                ),
            )
            run_sensitivities = study_name == "three_session_daily"
            portfolio = run_portfolio_comparisons(
                research,
                result,
                config,
                execution_protocol="continuous_v1",
                progress_callback=lambda message, label=study_name: progress(
                    f"{label} {message}"
                ),
                common_start_date=common_start_date,
                methods=ALPHA_METHODS,
                run_sensitivities=run_sensitivities,
            )
            equal_daily = portfolio["artifacts"]["equal_fusion"]["daily_nav"]
            for method, summary in portfolio["summary"].items():
                daily_rows = portfolio["artifacts"][method]["daily_nav"]
                metric_seed = int(_sha256_bytes(method.encode("utf-8"))[:8], 16)
                summary["paired_block_bootstrap_vs_cash"] = paired_block_bootstrap(
                    [float(row["daily_return"]) for row in daily_rows],
                    [0.0] * len(daily_rows),
                    block_sessions=20,
                    repetitions=2000,
                    seed=metric_seed,
                )
                if method != "equal_fusion":
                    summary["paired_block_bootstrap_vs_equal_fusion"] = _matched_interval(
                        daily_rows,
                        equal_daily,
                        metric_seed,
                    )
            diagnostics = expert_prediction_diagnostics(
                result,
                forecast_error_multiplier=float(config["forecast_error_multiplier"]),
            )
            ledger_paths = _save_replay_artifacts(
                study_dir,
                result,
                portfolio,
                gzip.compress(
                    json.dumps(result["training_seed"], sort_keys=True, separators=(",", ":")).encode("utf-8"),
                    mtime=0,
                ),
            )
            baseline_source = portfolio["artifacts"]["joint_optimization"]["daily_nav"]
            baselines = _benchmark_rows(
                baseline_source, float(config["initial_capital_cny"])
            )
            baseline_artifacts = {}
            for base_name, baseline in baselines.items():
                returns = (
                    [0.0] * len(baseline_source)
                    if base_name == "cash"
                    else [float(row["benchmark_return"]) for row in baseline_source]
                )
                baseline_artifacts[base_name] = [
                    {
                        "trade_date": str(row["trade_date"]),
                        "daily_return": value,
                        "nav_cny": float(config["initial_capital_cny"])
                        * math.prod(1 + prior for prior in returns[: index + 1]),
                        "benchmark_return": value,
                    }
                    for index, (row, value) in enumerate(zip(baseline_source, returns, strict=True))
                ]
            risk_baseline = simulate_fixed_risk_market_exposure(
                research,
                common_start_date,
                float(config["initial_capital_cny"]),
                float(config.get("risk_budget_annualized", 0.20)),
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
            risk_baseline["summary"]["paired_block_bootstrap_vs_csi300"] = paired_block_bootstrap(
                [float(row["daily_return"]) for row in risk_baseline["daily_nav"]],
                [float(row["benchmark_return"]) for row in risk_baseline["daily_nav"]],
                block_sessions=20,
                repetitions=2000,
                seed=20260926,
            )
            _write_csv(study_dir / "fixed_risk_market_daily_nav.csv", risk_baseline["daily_nav"])
            _write_csv(study_dir / "fixed_risk_market_orders.csv", risk_baseline["orders"])
            _write_csv(study_dir / "fixed_risk_market_positions.csv", risk_baseline["positions"])
            if run_sensitivities:
                risk_cost_summary = {}
                base_costs = {
                    key: float(config[key])
                    for key in (
                        "commission_per_side_bps",
                        "transfer_fee_per_side_bps",
                        "stamp_duty_sell_bps",
                        "slippage_per_side_bps",
                    )
                }
                for label, multiple in (("0x", 0.0), ("1x", 1.0), ("2x", 2.0)):
                    risk_cost_summary[label] = simulate_fixed_risk_market_exposure(
                        research,
                        common_start_date,
                        float(config["initial_capital_cny"]),
                        float(config.get("risk_budget_annualized", 0.20)),
                        int(config["risk_lookback_sessions"]),
                        {key: value * multiple for key, value in base_costs.items()},
                    )["summary"]
                risk_baseline["cost_sensitivity"] = risk_cost_summary
                risk_baseline["risk_budget_sensitivity"] = {
                    f"{int(budget * 100)}pct": simulate_fixed_risk_market_exposure(
                        research,
                        common_start_date,
                        float(config["initial_capital_cny"]),
                        budget,
                        int(config["risk_lookback_sessions"]),
                        base_costs,
                    )["summary"]
                    for budget in (0.15, 0.25)
                }
            _write_json(study_dir / "prediction_and_portfolio_diagnostics.json", diagnostics)
            report = {
                "study_name": study_name,
                "config_file": config_path.name,
                "config_sha256": _sha256_bytes(config_bytes),
                "effective_config": config,
                "protocol": result["protocol"],
                "common_start_date": common_start_date.isoformat(),
                "account_start_state": "cash for every strategy on the common after-close date",
                "decision_windows": result["decision_windows"],
                "evaluated_label_windows": result["evaluated_label_windows"],
                "prediction_quality": result["prediction_quality"],
                "decision_value_quality": result.get("decision_value_quality"),
                "portfolio_results": portfolio["summary"],
                "cash_and_csi300_baselines": baselines,
                "fixed_risk_market_baseline": risk_baseline["summary"],
                "cost_sensitivity": portfolio["cost_sensitivity"],
                "risk_sensitivity": portfolio["risk_sensitivity"],
                "fixed_risk_market_sensitivity": {
                    "cost": risk_baseline.get("cost_sensitivity"),
                    "risk_budget": risk_baseline.get("risk_budget_sensitivity"),
                },
                "prediction_and_portfolio_diagnostics": diagnostics,
                "prediction_diagnostics_artifact": "prediction_and_portfolio_diagnostics.json",
                "artifacts": ledger_paths,
            }
            _write_json(study_dir / "study_results.json", report)
            study_outputs[study_name] = report
            portfolio_paths[study_name] = {
                method: rows["daily_nav"]
                for method, rows in portfolio["artifacts"].items()
            }
            for base_name, rows in baseline_artifacts.items():
                portfolio_paths[study_name][base_name] = rows
            portfolio_paths[study_name]["fixed_risk_market"] = risk_baseline["daily_nav"]
            progress(
                f"completed {study_index + 1}/4 replays: {study_name}; labels={result['evaluated_label_windows']} portfolios={len(portfolio['summary'])}"
            )

        pair_specs = (
            ("one_session_daily", "three_session_daily", "prediction_horizon_sessions"),
            ("five_session_daily", "three_session_daily", "prediction_horizon_sessions"),
            ("three_session_daily", "three_session_review_every_five", "review_frequency"),
        )
        pair_reports = []
        method_pairs = (
            "cash",
            "csi300_benchmark",
            "fixed_risk_market",
            "trend",
            "reversal",
            "liquidity_confirmation",
            "equal_fusion",
            "market_plus_relative_fusion",
            "joint_optimization",
            "joint_market_relative",
        )
        for pair_index, (left_name, right_name, contrast) in enumerate(pair_specs):
            method_results = {}
            for method in method_pairs:
                if method not in portfolio_paths[left_name] or method not in portfolio_paths[right_name]:
                    continue
                method_results[method] = _matched_interval(
                    portfolio_paths[left_name][method],
                    portfolio_paths[right_name][method],
                    seed=20260926 + pair_index,
                )
            pair_reports.append(
                {
                    "contrast": contrast,
                    "left_study": left_name,
                    "right_study": right_name,
                    "warmup_dates_differ": True,
                    "account_start_and_observation_dates_matched": all(
                        item["account_start_state"] == "cash on the shared after-close decision date"
                        for item in method_results.values()
                    ),
                    "interpretation": "Paired historical diagnostics with equal starting capital and aligned account dates; horizon/review warm-up histories still differ and this is not a randomized causal comparison.",
                    "by_method": method_results,
                }
            )
        _write_json(output_dir / "matched_horizon_frequency_comparisons.json", pair_reports)

        all_candidates_frozen = all(
            item["prediction_and_portfolio_diagnostics"]["candidate_freeze_audit"][
                "all_candidate_hashes_recompute"
            ]
            and item["prediction_and_portfolio_diagnostics"]["candidate_freeze_audit"][
                "all_settled_symbols_were_scored"
            ]
            for item in study_outputs.values()
        )
        frozen_protocol = {
            "protocol_version": "post-p8-research-v1",
            "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
            "development_dates": ["2024-01-02", "2026-03-06"],
            "already_viewed_diagnostic_dates": ["2026-03-16", "2026-09-23"],
            "already_viewed_dates_are_not_unseen_validation": True,
            "future_validation_start_exclusive": "2026-09-23",
            "minimum_new_matured_decision_dates": 120,
            "minimum_fresh_trading_dates_for_120_matured_decisions_by_horizon": {
                "one_session": 121,
                "three_sessions": 123,
                "five_sessions": 125,
            },
            "horizons_sessions": [1, 3, 5],
            "review_steps_sessions": [1, 5],
            "common_start_date": common_start_date.isoformat(),
            "common_initial_capital_cny": float(
                config_payloads["three_session_daily"][1]["initial_capital_cny"]
            ),
            "historical_comparison_scope": "cash; gross CSI 300; price-only fixed-risk CSI 300 proxy; each single expert; equal fusion; P8 adaptive fusion; P8 loss-gated fusion; market plus relative fusion; absolute-return joint optimizer; market-relative joint optimizer",
            "moving_block_sessions": 20,
            "moving_block_repetitions": 2000,
            "bootstrap_seed": 20260926,
            "cost_sensitivity_multipliers": [0.0, 1.0, 2.0],
            "risk_budget_sensitivity_annualized": [0.15, 0.20, 0.25],
            "forecast_error_multiplier_frozen": 0.5,
            "forecast_error_threshold_was_not_lowered": True,
            "candidate_set_freeze_audit_passed": all_candidates_frozen,
            "calibration_and_gating_update": "predeclared prequential update on matured labels only",
            "primary_decision_rule": "After at least 120 new matured decision dates, report paired 20-session block intervals against every fixed baseline. Claim incremental value only when the predeclared cost-after-return contrast interval excludes zero in the positive direction and account/risk rules pass; otherwise report no demonstrated improvement.",
            "no_parameter_search_after_forward_start": True,
        }
        _write_json(output_dir / "frozen_research_protocol.json", frozen_protocol)
        _write_json(
            output_dir / "forward_validation_status.json",
            {
                "status": "awaiting_new_dates",
                "last_already_viewed_date": "2026-09-23",
                "new_trading_dates_in_this_development_run": 0,
                "new_dates_in_current_market_source_not_checked": True,
                "new_matured_decision_dates_in_this_development_run": 0,
                "required_new_matured_decision_dates": 120,
                "required_fresh_trading_dates_for_120_matured_decisions_by_horizon": {
                    "one_session": 121,
                    "three_sessions": 123,
                    "five_sessions": 125,
                },
                "daily_runner": "scripts/run_post_p8_forward_validation.py",
                "eligible_dates_start_strictly_after": "2026-09-23",
                "claim": "No true forward-validation result is available yet.",
            },
        )

        report_lines = [
            "# Post-P8 量化研究报告",
            "",
            f"开发数据：{provenance['development_start_date']} 至 {provenance['development_end_date']}（{provenance['development_trading_dates']} 个交易日）。",
            "已查看的 2026-03-16 至 2026-09-23 区间没有并入本次开发重放，也不会被称为未见测试。",
            f"四种策略共同从 {common_start_date} 的收盘决策后以相同资金、全现金状态开始组合评价；期限与复审频率的滚动训练历史不同，因此差值仅作配对历史诊断，不作因果结论。",
            "",
            "## 逐期限预测评价",
            "",
            "完整日期级 Rank-IC、五分位预测/实现校准、专家相关性、留一消融和组合层误差/风险对照见各目录中的 `study_results.json`、`observations.jsonl.gz` 与 `prediction_and_portfolio_diagnostics.json`。",
            "",
            "## 扣费后连续账户结果",
            "",
            "| 配置 | 策略 | 净收益 | 平均现金 | 最大回撤 | 换手 | 费用 | 成交额 | 成交单 | 阻塞单 |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for study_name, report in study_outputs.items():
            for method, summary in report["portfolio_results"].items():
                report_lines.append(
                    "| {} | {} | {} | {} | {} | {} | ¥{:,.0f} | ¥{:,.0f} | {} | {} |".format(
                        study_name,
                        method,
                        _format_percent(summary.get("net_total_return", summary.get("total_return"))),
                        _format_percent(summary.get("average_cash_weight", summary.get("mean_cash_weight"))),
                        _format_percent(summary.get("maximum_drawdown")),
                        _format_percent(summary.get("average_one_way_turnover", summary.get("one_way_turnover_over_average_nav"))),
                        float(summary.get("total_cost_cny", summary.get("total_transaction_cost_cny", 0.0))),
                        float(summary.get("total_traded_notional_cny", 0.0)),
                        int(summary.get("filled_orders", 0)),
                        int(summary.get("blocked_orders", 0)),
                    )
                )
        report_lines.extend(
            [
                "",
                "Cash 和 CSI 300 是独立基线；CSI 300 使用 BaoStock 指数价格回报，未计可交易 ETF 跟踪误差。固定风险暴露基线使用起始日前历史波动率设定一次指数权重，采用相同费用率估算进出成本；其指数价格成交不代表 ETF 的真实可成交性。",
                "",
                "## 配对期限与复审频率差值",
                "",
                "完整共同日期数、日期边界和 20 交易日区块 Bootstrap 区间见 `matched_horizon_frequency_comparisons.json`。所有区间明确标为已查看开发历史上的配对诊断。",
                "",
                "## 成本、风险和误差折让",
                "",
                "三日每日主配置测试 0×、1×、2× 成本和 15%、20%、25% 风险预算。所有风险敏感性保持 0.5× 成熟预测 RMSE 不变；没有为增加交易而降低误差门槛。开发样本中若联合优化继续全现金，将如实报告现金账户与市场/简单策略差值。",
                "",
                "## 前向验证状态",
                "",
                "截至 2026-09-26，2026-09-23 之后尚无新交易日可用于本次运行，真实前向已成熟决策数为 0。冻结的逐日 BaoStock 运行命令和 120 个新已成熟决策日条件见 `POST_P8_RESEARCH_PROTOCOL.md` 与 `frozen_research_protocol.json`。",
                "",
                "本报告为开发期诊断，不构成真实前向验证结果。",
            ]
        )
        (output_dir / "RESEARCH_REPORT.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
        render_report(output_dir)
        status["status"] = "complete"
        status["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        status["result_report"] = str(output_dir / "RESEARCH_REPORT.md")
        progress("all four horizon/review replays and matched diagnostics completed")
        _write_json(status_path, status)
        print(f"[post-P8] report={output_dir / 'RESEARCH_REPORT.md'}", flush=True)
    except Exception as error:
        status["status"] = "failed"
        status["failed_at_utc"] = datetime.now(timezone.utc).isoformat()
        status["error_type"] = type(error).__name__
        status["error"] = str(error)
        _write_json(status_path, status)
        raise


if __name__ == "__main__":
    main()
