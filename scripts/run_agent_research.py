"""Run the sequential quantitative-agent diagnostic on the training history."""

import argparse
import gzip
import hashlib
import io
import json
import tomllib
from pathlib import Path

from adaptive_mas.data import load_research_data
from adaptive_mas.evaluation.agent_research import run_agent_research


def main() -> None:
    project_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-data-dir", type=Path, required=True)
    parser.add_argument("--baostock-run-dir", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path, default=project_dir / "experiments" / "agent_research.json"
    )
    args = parser.parse_args()

    config_path = project_dir / "configs" / "agent_research.toml"
    with config_path.open("rb") as source:
        config = tomllib.load(source)
    if any(
        config[name] < 1
        for name in (
            "trend_lookback_sessions",
            "reversal_lookback_sessions",
            "liquidity_return_lookback_sessions",
            "liquidity_recent_amount_sessions",
            "liquidity_reference_amount_sessions",
        )
    ):
        raise ValueError("Agent 的历史窗口必须至少为一个交易日")
    if config["evaluation_step_sessions"] != 5:
        raise ValueError("为保持五日目标不重叠，evaluation_step_sessions 必须为 5")
    if config["reliability_lookback_windows"] < 1:
        raise ValueError("reliability_lookback_windows 必须至少为 1")
    if not 1 <= config["reliability_minimum_history_windows"] <= config["reliability_lookback_windows"]:
        raise ValueError("可靠性门控的最少成熟窗口必须在回看窗口范围内")
    if not 0 <= config["equal_weight_share"] <= 1:
        raise ValueError("equal_weight_share 必须处于 [0, 1]")
    if not 0 < config["top_decile_fraction"] <= 1:
        raise ValueError("top_decile_fraction 必须处于 (0, 1]")
    config_hash = hashlib.sha256(config_path.read_bytes()).hexdigest()
    manifest_path = args.baostock_run_dir / "manifest.jsonl"
    with manifest_path.open("rb") as manifest:
        manifest_hash = hashlib.file_digest(manifest, "sha256").hexdigest()

    research = load_research_data(args.training_data_dir, args.baostock_run_dir)
    result = run_agent_research(research, **config)
    signal_records = result.pop("signal_records")
    signal_log_path = project_dir / "data" / "processed" / "agent_signals.jsonl.gz"
    signal_log_path.parent.mkdir(parents=True, exist_ok=True)
    with signal_log_path.open("wb") as raw_output:
        with gzip.GzipFile(
            filename="", mode="wb", fileobj=raw_output, mtime=0
        ) as compressed_output:
            with io.TextIOWrapper(
                compressed_output, encoding="utf-8", newline="\n"
            ) as signal_log:
                for record in signal_records:
                    signal_log.write(
                        json.dumps(record, ensure_ascii=False, separators=(",", ":"))
                    )
                    signal_log.write("\n")
    with signal_log_path.open("rb") as signal_log:
        signal_log_hash = hashlib.file_digest(signal_log, "sha256").hexdigest()
    result["data"] = {
        "start_date": research.trading_dates[0].isoformat(),
        "end_date": research.trading_dates[-1].isoformat(),
        "trading_dates": len(research.trading_dates),
        "adjusted_bars": len(research.adjusted_bars),
        "membership_snapshots": len(research.membership_snapshots),
        "baostock_manifest_sha256": manifest_hash,
        "config_sha256": config_hash,
    }
    result["signal_log"] = {
        "path": signal_log_path.relative_to(project_dir).as_posix(),
        "records": len(signal_records),
        "sha256": signal_log_hash,
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
