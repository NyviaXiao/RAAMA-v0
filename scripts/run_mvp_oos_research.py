"""Run one frozen P3-P8 diagnostic on the already cached BaoStock OOS period."""

import argparse
import csv
import gzip
import hashlib
import io
import json
import tomllib
from datetime import date
from pathlib import Path

from adaptive_mas.data import load_oos_research_data
from adaptive_mas.evaluation.mvp_research import run_mvp_research, run_portfolio_comparisons


def main() -> None:
    project_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--baostock-run-dir", type=Path, required=True)
    parser.add_argument(
        "--execution-protocol",
        choices=("continuous_v1", "fixed_horizon_v2"),
        default="continuous_v1",
    )
    parser.add_argument(
        "--development-result",
        type=Path,
        default=project_dir / "experiments" / "mvp_portfolio_research.json",
    )
    parser.add_argument(
        "--training-seed",
        type=Path,
        default=project_dir / "data" / "processed" / "mvp" / "training_seed.json.gz",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output is None:
        output_name = (
            "mvp_oos_research.json"
            if args.execution_protocol == "continuous_v1"
            else "mvp_oos_fixed_horizon_research.json"
        )
        args.output = project_dir / "experiments" / output_name

    config_path = project_dir / "configs" / "mvp_research.toml"
    config_bytes = config_path.read_bytes()
    config = tomllib.loads(config_bytes.decode("utf-8"))
    development_bytes = args.development_result.read_bytes()
    development = json.loads(development_bytes)
    config_sha256 = hashlib.sha256(config_bytes).hexdigest()
    if development["config_sha256"] != config_sha256:
        raise ValueError("开发区间结果与当前配置哈希不一致，拒绝运行样本外比较")
    seed_bytes = args.training_seed.read_bytes()
    seed_sha256 = hashlib.sha256(seed_bytes).hexdigest()
    if development["training_seed_sha256"] != seed_sha256:
        raise ValueError("开发区间成熟历史种子哈希不一致，拒绝运行样本外比较")
    with gzip.open(io.BytesIO(seed_bytes), "rt", encoding="utf-8") as seed_file:
        seed = json.load(seed_file)
    expected_seed_fields = {
        "training_features",
        "training_returns",
        "matured_rank_ics",
        "matured_by_state",
    }
    if set(seed) != expected_seed_fields:
        raise ValueError("开发区间成熟历史种子字段与研究协议不一致")
    if len(seed["training_features"]) != len(seed["training_returns"]):
        raise ValueError("开发区间 Ridge 成熟训练特征与标签数量不一致")
    if len(seed["matured_rank_ics"]) < config["reliability_minimum_windows"]:
        raise ValueError("开发区间已成熟 Rank IC 窗口不足以初始化样本外融合")

    research = load_oos_research_data(args.baostock_run_dir)
    latest_development_label = max(
        date.fromisoformat(row["label_available_at"])
        for row in development["observations"]
    )
    if latest_development_label >= research.trading_dates[0]:
        raise ValueError("开发期标签未在样本外开始前成熟")

    diagnostic = run_mvp_research(research, config, seed)
    diagnostic.pop("training_seed")
    portfolio = run_portfolio_comparisons(
        research, diagnostic, config, args.execution_protocol
    )
    processed_name = (
        "mvp" if args.execution_protocol == "continuous_v1" else "mvp_fixed_horizon"
    )
    processed_dir = project_dir / "data" / "processed" / processed_name
    processed_dir.mkdir(parents=True, exist_ok=True)
    artifact_paths = {}
    for method, records in portfolio["artifacts"].items():
        for kind in ("daily_nav", "orders", "positions"):
            rows = records[kind]
            path = processed_dir / f"oos_{method}_{kind}.csv"
            columns = list(rows[0]) if rows else []
            with path.open("w", encoding="utf-8", newline="") as output:
                writer = csv.DictWriter(output, fieldnames=columns)
                writer.writeheader()
                writer.writerows(rows)
            artifact_paths[f"{method}_{kind}"] = path.relative_to(project_dir).as_posix()

    decision_log_path = processed_dir / "oos_decision_log.jsonl.gz"
    with decision_log_path.open("wb") as raw_output:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw_output, mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8", newline="\n") as log:
                for row in diagnostic["decision_log"]:
                    log.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
                    log.write("\n")

    prediction_ledger_paths = {}
    if args.execution_protocol == "fixed_horizon_v2":
        for name, rows in (
            ("prediction_decisions", diagnostic["prediction_decisions"]),
            ("prediction_settlements", diagnostic["prediction_settlements"]),
        ):
            path = processed_dir / f"oos_{name}.jsonl.gz"
            with path.open("wb") as raw_output:
                with gzip.GzipFile(filename="", mode="wb", fileobj=raw_output, mtime=0) as compressed:
                    with io.TextIOWrapper(compressed, encoding="utf-8", newline="\n") as ledger:
                        for row in rows:
                            ledger.write(
                                json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                            )
                            ledger.write("\n")
            prediction_ledger_paths[name] = path.relative_to(project_dir).as_posix()

    manifest_path = args.baostock_run_dir / "manifest.jsonl"
    output = {
        "protocol": diagnostic["protocol"],
        "data": {
            "source": "BaoStock only; no local scoring CSV",
            "start_date": research.trading_dates[0].isoformat(),
            "end_date": research.trading_dates[-1].isoformat(),
            "trading_days": len(research.trading_dates),
            "membership_snapshots": len(research.membership_snapshots),
            "adjusted_bars": len(research.adjusted_bars),
            "execution_bars": len(research.execution_bars),
            "index_bars": len(research.index_bars),
            "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        },
        "config_sha256": config_sha256,
        "development_result_sha256": hashlib.sha256(development_bytes).hexdigest(),
        "training_seed_sha256": seed_sha256,
        "latest_development_label_available_at": latest_development_label.isoformat(),
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
        },
    }
    if args.execution_protocol == "fixed_horizon_v2":
        output["execution_protocol"] = args.execution_protocol
        output["prediction_ledger"] = {
            "decision_records": len(diagnostic["prediction_decisions"]),
            "settlement_records": len(diagnostic["prediction_settlements"]),
            "settled_records": sum(
                row["status"] == "settled"
                for row in diagnostic["prediction_settlements"]
            ),
            "artifacts": prediction_ledger_paths,
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(output["portfolio_results"], ensure_ascii=False, indent=2, sort_keys=True))
    print(f"样本外窗口：{output['evaluated_label_windows']}；结果：{args.output}")


if __name__ == "__main__":
    main()
