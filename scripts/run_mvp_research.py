"""Run the frozen P3-P8 walk-forward portfolio prototype on local train data."""

import argparse
import csv
import gzip
import hashlib
import io
import json
import math
import tomllib
from pathlib import Path

from adaptive_mas.data import load_research_data
from adaptive_mas.evaluation.mvp_research import run_mvp_research, run_portfolio_comparisons


def main() -> None:
    project_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-data-dir", type=Path, required=True)
    parser.add_argument("--baostock-run-dir", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path, default=project_dir / "experiments" / "mvp_portfolio_research.json"
    )
    args = parser.parse_args()

    config_path = project_dir / "configs" / "mvp_research.toml"
    config_bytes = config_path.read_bytes()
    config = tomllib.loads(config_bytes.decode("utf-8"))
    expected_numeric = {
        "decision_step_sessions",
        "feature_trend_sessions",
        "feature_reversal_sessions",
        "feature_liquidity_return_sessions",
        "feature_recent_amount_sessions",
        "feature_reference_amount_sessions",
        "evaluation_top_fraction",
        "portfolio_top_fraction",
        "invested_weight",
        "initial_capital_cny",
        "maximum_symbol_weight",
        "maximum_one_way_turnover",
        "liquidity_lookback_sessions",
        "maximum_daily_participation",
        "reliability_lookback_windows",
        "reliability_minimum_windows",
        "reliability_equal_weight_share",
        "regime_lookback_sessions",
        "regime_minimum_windows",
        "ridge_alpha",
        "ridge_minimum_training_rows",
        "risk_lookback_sessions",
        "risk_minimum_observations",
        "commission_per_side_bps",
        "transfer_fee_per_side_bps",
        "stamp_duty_sell_bps",
        "slippage_per_side_bps",
    }
    if set(config) != expected_numeric:
        raise ValueError(
            f"MVP 配置字段与运行协议不一致：缺少 {sorted(expected_numeric - set(config))}，"
            f"多余 {sorted(set(config) - expected_numeric)}"
        )
    if any(not math.isfinite(float(value)) for value in config.values()):
        raise ValueError("MVP 配置必须全部为有限数值")
    if config["decision_step_sessions"] != 5:
        raise ValueError("五日目标要求 decision_step_sessions 为 5")
    if not 0 < config["evaluation_top_fraction"] <= 1:
        raise ValueError("evaluation_top_fraction 必须处于 (0, 1]")
    if not 0 < config["portfolio_top_fraction"] <= 1:
        raise ValueError("portfolio_top_fraction 必须处于 (0, 1]")
    if not 0 < config["invested_weight"] <= 1:
        raise ValueError("invested_weight 必须处于 (0, 1]")
    if config["maximum_symbol_weight"] <= 0 or config["maximum_symbol_weight"] > 1:
        raise ValueError("maximum_symbol_weight 必须处于 (0, 1]")
    if config["maximum_symbol_weight"] * math.ceil(
        300 * config["portfolio_top_fraction"]
    ) < config["invested_weight"]:
        raise ValueError("候选股票数量与单股上限无法满足目标投资比例")
    if not 0 <= config["maximum_one_way_turnover"] <= 1:
        raise ValueError("maximum_one_way_turnover 必须处于 [0, 1]")
    if config["liquidity_lookback_sessions"] < 1 or not 0 < config["maximum_daily_participation"] <= 1:
        raise ValueError("流动性窗口必须为正，最大成交参与率必须处于 (0, 1]")
    if not 0 <= config["reliability_equal_weight_share"] <= 1:
        raise ValueError("reliability_equal_weight_share 必须处于 [0, 1]")
    if any(
        config[name] < 1
        for name in (
            "feature_trend_sessions",
            "feature_reversal_sessions",
            "feature_liquidity_return_sessions",
            "feature_recent_amount_sessions",
            "feature_reference_amount_sessions",
            "reliability_lookback_windows",
            "reliability_minimum_windows",
            "regime_lookback_sessions",
            "regime_minimum_windows",
            "ridge_minimum_training_rows",
            "risk_lookback_sessions",
            "risk_minimum_observations",
        )
    ):
        raise ValueError("窗口、样本数和最小风险观测数必须至少为 1")
    if config["reliability_minimum_windows"] > config["reliability_lookback_windows"]:
        raise ValueError("可靠性最小成熟窗口数不能超过回看窗口")
    if config["initial_capital_cny"] <= 0 or config["ridge_alpha"] < 0:
        raise ValueError("初始资金必须为正，Ridge alpha 不能为负")
    if any(
        config[name] < 0
        for name in (
            "commission_per_side_bps",
            "transfer_fee_per_side_bps",
            "stamp_duty_sell_bps",
            "slippage_per_side_bps",
        )
    ):
        raise ValueError("费用和滑点假设不能为负")

    research = load_research_data(args.training_data_dir, args.baostock_run_dir)
    diagnostic = run_mvp_research(research, config)
    training_seed = diagnostic.pop("training_seed")
    portfolio = run_portfolio_comparisons(research, diagnostic, config)
    processed_dir = project_dir / "data" / "processed" / "mvp"
    processed_dir.mkdir(parents=True, exist_ok=True)
    training_seed_path = processed_dir / "training_seed.json.gz"
    with training_seed_path.open("wb") as raw_output:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw_output, mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8", newline="\n") as seed_file:
                seed_file.write(
                    json.dumps(training_seed, sort_keys=True, separators=(",", ":"))
                )
                seed_file.write("\n")
    artifact_paths = {}
    for method, records in portfolio["artifacts"].items():
        for kind in ("daily_nav", "orders", "positions"):
            rows = records[kind]
            path = processed_dir / f"{method}_{kind}.csv"
            columns = list(rows[0]) if rows else []
            with path.open("w", encoding="utf-8", newline="") as output:
                writer = csv.DictWriter(output, fieldnames=columns)
                writer.writeheader()
                writer.writerows(rows)
            artifact_paths[f"{method}_{kind}"] = path.relative_to(project_dir).as_posix()

    decision_log_path = processed_dir / "decision_log.jsonl.gz"
    with decision_log_path.open("wb") as raw_output:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw_output, mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8", newline="\n") as log:
                for row in diagnostic["decision_log"]:
                    log.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
                    log.write("\n")

    manifest_path = args.baostock_run_dir / "manifest.jsonl"
    output = {
        "protocol": diagnostic["protocol"],
        "data": {
            "source": "BaoStock + existing train.csv",
            "start_date": research.trading_dates[0].isoformat(),
            "end_date": research.trading_dates[-1].isoformat(),
            "trading_days": len(research.trading_dates),
            "historical_membership_snapshots": len(research.membership_snapshots),
            "adjusted_bars": len(research.adjusted_bars),
            "execution_bars": len(research.execution_bars),
            "index_bars": len(research.index_bars),
            "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        },
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "training_seed_sha256": hashlib.sha256(training_seed_path.read_bytes()).hexdigest(),
        "decision_windows": diagnostic["decision_windows"],
        "evaluated_label_windows": diagnostic["evaluated_label_windows"],
        "comparison_decision_count": diagnostic["comparison_decision_count"],
        "training_rows_at_end": diagnostic["training_rows_at_end"],
        "matured_rank_ic_windows_at_end": diagnostic["matured_rank_ic_windows_at_end"],
        "observations": diagnostic["observations"],
        "portfolio_results": portfolio["summary"],
        "cost_sensitivity": portfolio["cost_sensitivity"],
        "processed_artifacts": {
            **artifact_paths,
            "decision_log": decision_log_path.relative_to(project_dir).as_posix(),
            "training_seed": training_seed_path.relative_to(project_dir).as_posix(),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(output["portfolio_results"], ensure_ascii=False, indent=2, sort_keys=True))
    print(f"完成窗口：{output['evaluated_label_windows']}；结果：{args.output}")


if __name__ == "__main__":
    main()
