"""Run a non-overlapping cross-sectional diagnostic for simple momentum."""

import argparse
import hashlib
import json
import math
import statistics
import tomllib
from collections import defaultdict
from datetime import date
from pathlib import Path

from adaptive_mas.data import load_research_data
from adaptive_mas.features.momentum import momentum
from adaptive_mas.targets.five_session import five_session_forward_returns


def spearman_rank_correlation(left: list[float], right: list[float]) -> float:
    def average_ranks(values: list[float]) -> list[float]:
        order = sorted(range(len(values)), key=values.__getitem__)
        ranks = [0.0] * len(values)
        start = 0
        while start < len(order):
            end = start + 1
            while end < len(order) and values[order[end]] == values[order[start]]:
                end += 1
            rank = (start + 1 + end) / 2
            for position in order[start:end]:
                ranks[position] = rank
            start = end
        return ranks

    if len(left) != len(right) or len(left) < 2:
        raise ValueError("计算 Rank IC 需要至少两组一一对应的信号和标签")
    left_ranks = average_ranks(left)
    right_ranks = average_ranks(right)
    left_mean = statistics.fmean(left_ranks)
    right_mean = statistics.fmean(right_ranks)
    covariance = math.fsum(
        (a - left_mean) * (b - right_mean)
        for a, b in zip(left_ranks, right_ranks, strict=True)
    )
    left_variance = math.fsum((a - left_mean) ** 2 for a in left_ranks)
    right_variance = math.fsum((b - right_mean) ** 2 for b in right_ranks)
    return covariance / math.sqrt(left_variance * right_variance)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-data-dir", type=Path, required=True)
    parser.add_argument("--baostock-run-dir", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "experiments"
        / "baseline_momentum.json",
    )
    args = parser.parse_args()

    config_path = Path(__file__).resolve().parents[1] / "configs" / "baseline_momentum.toml"
    with config_path.open("rb") as source:
        config = tomllib.load(source)
    lookback_sessions = int(config["lookback_sessions"])
    evaluation_step = int(config["evaluation_step_sessions"])
    if lookback_sessions < 1 or evaluation_step < 1:
        raise ValueError("动量回看窗口与评价间隔必须至少为一个交易日")
    research = load_research_data(args.training_data_dir, args.baostock_run_dir)

    decision_dates = [
        research.trading_dates[index]
        for index in range(
            lookback_sessions,
            len(research.trading_dates) - 5,
            evaluation_step,
        )
    ]
    if not decision_dates:
        raise ValueError("研究交易日不足以形成非重叠的五日评价样本")
    labels = five_session_forward_returns(research, decision_dates)
    labels_by_date: dict[date, dict[str, float]] = defaultdict(dict)
    for row in labels:
        labels_by_date[row["decision_date"]][str(row["symbol"])] = float(
            row["forward_return_5s"]
        )

    observations = []
    for decision_date in decision_dates:
        snapshot = research.snapshot_at(decision_date)
        signals = momentum(snapshot, lookback_sessions)
        signal_by_symbol = {str(row["symbol"]): float(row["momentum"]) for row in signals}
        shared_symbols = sorted(signal_by_symbol.keys() & labels_by_date[decision_date].keys())
        signal_values = [signal_by_symbol[symbol] for symbol in shared_symbols]
        label_values = [labels_by_date[decision_date][symbol] for symbol in shared_symbols]
        observations.append(
            {
                "decision_date": decision_date.isoformat(),
                "rank_ic": spearman_rank_correlation(signal_values, label_values),
                "cross_section": len(shared_symbols),
            }
        )

    by_year: dict[str, list[float]] = defaultdict(list)
    for row in observations:
        by_year[row["decision_date"][:4]].append(row["rank_ic"])
    manifest_path = args.baostock_run_dir / "manifest.jsonl"
    with manifest_path.open("rb") as manifest:
        manifest_hash = hashlib.file_digest(manifest, "sha256").hexdigest()
    result = {
        "protocol": {
            "feature": f"close(t) / close(t-{lookback_sessions}) - 1",
            "decision": "after close; input bars available through decision_date",
            "target": "adjusted_close(t+5) / adjusted_open(t+1) - 1",
            "evaluation_step_sessions": evaluation_step,
            "non_overlapping_targets": True,
            "test_csv_loaded": False,
        },
        "data": {
            "start_date": research.trading_dates[0].isoformat(),
            "end_date": research.trading_dates[-1].isoformat(),
            "trading_dates": len(research.trading_dates),
            "adjusted_bars": len(research.adjusted_bars),
            "membership_snapshots": len(research.membership_snapshots),
            "baostock_manifest_sha256": manifest_hash,
        },
        "summary": {
            "observations": len(observations),
            "mean_rank_ic": statistics.fmean(row["rank_ic"] for row in observations),
            "median_rank_ic": statistics.median(row["rank_ic"] for row in observations),
            "positive_rank_ic_fraction": statistics.fmean(
                row["rank_ic"] > 0 for row in observations
            ),
            "mean_cross_section": statistics.fmean(
                row["cross_section"] for row in observations
            ),
        },
        "by_year": {
            year: {
                "observations": len(values),
                "mean_rank_ic": statistics.fmean(values),
                "positive_fraction": statistics.fmean(value > 0 for value in values),
            }
            for year, values in sorted(by_year.items())
        },
        "observations_by_date": observations,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "summary": result["summary"],
                "by_year": result["by_year"],
                "output": str(args.output),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
