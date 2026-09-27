"""Emit one versioned pending decision JSON from BaoStock and actual account state."""

import argparse
import gzip
import hashlib
import json
import tomllib
from datetime import date
from pathlib import Path

from adaptive_mas.data import load_oos_research_data
from adaptive_mas.decision import PortfolioState, create_daily_decision


def main() -> None:
    project_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--baostock-run-dir", type=Path, required=True)
    parser.add_argument("--portfolio-state-json", type=Path, required=True)
    parser.add_argument(
        "--config", type=Path,
        default=project_dir / "configs" / "p8_daily_three_session.toml",
    )
    parser.add_argument("--training-seed", type=Path)
    parser.add_argument("--as-of", type=date.fromisoformat)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    config_bytes = args.config.read_bytes()
    config = tomllib.loads(config_bytes.decode("utf-8"))
    account = json.loads(args.portfolio_state_json.read_text(encoding="utf-8"))
    if set(account) != {"cash_cny", "holdings_cny"}:
        raise ValueError("账户状态 JSON 必须且只能包含 cash_cny 与 holdings_cny")
    if not isinstance(account["holdings_cny"], dict):
        raise ValueError("holdings_cny 必须是证券代码到持仓市值的对象")
    portfolio = PortfolioState(
        cash_cny=float(account["cash_cny"]),
        holdings_cny={str(symbol): float(value) for symbol, value in account["holdings_cny"].items()},
    )
    portfolio.validate()

    seed_path = args.training_seed
    if seed_path is None:
        candidate = (
            project_dir / "data" / "processed"
            / f"{args.config.stem}_continuous_v1" / "training_seed.json.gz"
        )
        seed_path = candidate if candidate.is_file() else None
    training_state = None
    if seed_path is None or not seed_path.is_file():
        raise FileNotFoundError(
            "P8 app 决策要求显式成熟训练种子；请先运行对应开发期研究，"
            f"或通过 --training-seed 指定文件（默认路径：{seed_path}）"
        )
    with gzip.open(seed_path, "rt", encoding="utf-8") as seed_file:
        training_state = json.load(seed_file)

    research = load_oos_research_data(args.baostock_run_dir)
    as_of = args.as_of or research.trading_dates[-1]
    manifest_path = args.baostock_run_dir / "manifest.jsonl"
    versions = {
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "baostock_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "training_seed_sha256": hashlib.sha256(seed_path.read_bytes()).hexdigest(),
    }
    decision = create_daily_decision(
        research,
        config,
        as_of,
        portfolio,
        training_state,
        versions,
    )
    payload = json.dumps(
        decision.to_dict(), ensure_ascii=False, indent=2, sort_keys=True
    ) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    else:
        print(payload, end="")


if __name__ == "__main__":
    main()
