"""Run the frozen agent protocol on BaoStock dates after the development set."""

import argparse
import gzip
import hashlib
import io
import json
import math
import tomllib
from datetime import date
from pathlib import Path

from adaptive_mas.data import load_oos_research_data
from adaptive_mas.evaluation.agent_research import run_agent_research


def main() -> None:
    project_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--baostock-run-dir", type=Path, required=True)
    parser.add_argument(
        "--development-result",
        type=Path,
        default=project_dir / "experiments" / "agent_research.json",
    )
    parser.add_argument(
        "--output", type=Path, default=project_dir / "experiments" / "oos_agent_research.json"
    )
    args = parser.parse_args()

    config_path = project_dir / "configs" / "oos_agent_research.toml"
    with config_path.open("rb") as source:
        config = tomllib.load(source)
    fee_names = (
        "commission_per_side",
        "transfer_fee_per_side",
        "stamp_duty_sell",
        "slippage_per_side",
    )
    transaction_costs = {name: config.pop(name) for name in fee_names}
    if any(not math.isfinite(value) or value < 0 for value in transaction_costs.values()):
        raise ValueError("交易成本配置必须是有限的非负数")
    if any(
        config[name] < 1
        for name in (
            "trend_lookback_sessions",
            "reversal_lookback_sessions",
            "liquidity_return_lookback_sessions",
            "liquidity_recent_amount_sessions",
            "liquidity_reference_amount_sessions",
            "reliability_lookback_windows",
            "reliability_minimum_history_windows",
        )
    ):
        raise ValueError("Agent 与可靠性历史窗口必须至少为一个交易日/评价窗口")
    if config["evaluation_step_sessions"] != 5:
        raise ValueError("为保持五日目标不重叠，evaluation_step_sessions 必须为 5")
    if config["reliability_minimum_history_windows"] > config["reliability_lookback_windows"]:
        raise ValueError("可靠性门控的最低成熟窗口数不能超过回看窗口")
    if not 0 <= config["equal_weight_share"] <= 1:
        raise ValueError("equal_weight_share 必须处于 [0, 1]")
    if not 0 < config["top_decile_fraction"] <= 1:
        raise ValueError("top_decile_fraction 必须处于 (0, 1]")

    research = load_oos_research_data(args.baostock_run_dir)
    development = json.loads(args.development_result.read_text(encoding="utf-8"))
    matured_observations = [
        row
        for row in development["observations"]
        if date.fromisoformat(row["label_available_at"]) < research.trading_dates[0]
    ]
    if len(matured_observations) < config["reliability_minimum_history_windows"]:
        raise ValueError("样本外开始前已成熟的开发期 Rank IC 窗口不足以初始化可靠性门控")
    initial_matured_rank_ics = [
        {
            agent_id: float(row["methods"][agent_id]["rank_ic"])
            for agent_id in ("trend", "reversal", "liquidity_confirmation")
        }
        for row in matured_observations[-config["reliability_lookback_windows"] :]
    ]
    if len(initial_matured_rank_ics) < config["reliability_minimum_history_windows"]:
        raise ValueError("样本外可靠性门控初始化历史少于配置的最低成熟窗口数")

    manifest_path = args.baostock_run_dir / "manifest.jsonl"
    manifest_sha256 = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    config_sha256 = hashlib.sha256(config_path.read_bytes()).hexdigest()
    development_sha256 = hashlib.sha256(args.development_result.read_bytes()).hexdigest()
    result = run_agent_research(
        research,
        transaction_costs=transaction_costs,
        initial_matured_rank_ics=initial_matured_rank_ics,
        **config,
    )
    signal_records = result.pop("signal_records")
    signal_log_path = project_dir / "data" / "processed" / "oos_agent_signals.jsonl.gz"
    signal_log_path.parent.mkdir(parents=True, exist_ok=True)
    with signal_log_path.open("wb") as raw_output:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw_output, mtime=0) as compressed_output:
            with io.TextIOWrapper(
                compressed_output, encoding="utf-8", newline="\n"
            ) as signal_log:
                for record in signal_records:
                    signal_log.write(
                        json.dumps(record, ensure_ascii=False, separators=(",", ":"))
                    )
                    signal_log.write("\n")
    signal_sha256 = hashlib.sha256(signal_log_path.read_bytes()).hexdigest()
    result["data"] = {
        "source": "BaoStock",
        "start_date": research.trading_dates[0].isoformat(),
        "end_date": research.trading_dates[-1].isoformat(),
        "trading_dates": len(research.trading_dates),
        "adjusted_bars": len(research.adjusted_bars),
        "execution_bars": len(research.execution_bars),
        "membership_snapshots": len(research.membership_snapshots),
        "baostock_manifest_sha256": manifest_sha256,
        "config_sha256": config_sha256,
        "development_result_sha256": development_sha256,
        "initial_matured_windows": len(initial_matured_rank_ics),
        "last_initial_label_available_at": matured_observations[-1]["label_available_at"],
    }
    result["signal_log"] = {
        "path": signal_log_path.relative_to(project_dir).as_posix(),
        "records": len(signal_records),
        "sha256": signal_sha256,
    }
    result["agents"] = {
        "trend": "20-session adjusted-close return",
        "reversal": "negative 5-session adjusted-close return",
        "liquidity_confirmation": "5-session return multiplied by log recent/prior mean amount",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
    print(f"结果文件：{args.output}")


if __name__ == "__main__":
    main()
