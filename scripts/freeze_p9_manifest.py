"""Record exact P9 code, configuration, candidate, training and data inputs."""

from __future__ import annotations

import hashlib
import json
import csv
from datetime import datetime, timezone
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    project = Path(__file__).resolve().parents[1]
    study = Path("experiments/post_p8_research_20260926/three_session_daily")
    files = sorted(
        path.relative_to(project).as_posix()
        for path in (project / "src" / "adaptive_mas").rglob("*.py")
    )
    files += [
        "scripts/freeze_p9_manifest.py",
        "scripts/run_mvp_research.py",
        "scripts/run_post_p8_study.py",
        "scripts/run_p9_joint_economics.py",
        "scripts/run_p9_daily_selector.py",
        "scripts/run_p9_forward_validation.py",
        "P9_RETURN_FIRST_PLAN.md",
        (study / "config.toml").as_posix(),
        (study / "study_results.json").as_posix(),
        (study / "prediction_decisions.jsonl.gz").as_posix(),
        (study / "prediction_settlements.jsonl.gz").as_posix(),
        (study / "training_seed.json.gz").as_posix(),
        "data/raw/baostock/fetch-20260924T115050Z/manifest.jsonl",
        "data/raw/baostock/fetch-20260924T150656Z/manifest.jsonl",
    ]
    train_csv = Path(r"C:\Users\xiao\Desktop\THU-BDC2026\data\train.csv")
    cutover = project / "experiments" / "p9_forward_cutover_20260927_v5"
    cutover_report = json.loads((cutover / "forward_validation_report.json").read_text(encoding="utf-8"))
    if cutover_report["status"] != "awaiting_new_dates":
        raise ValueError("冻结状态必须是零新交易日的截止日影子账户")
    account_states = {}
    for name in ("p9_daily_selector", "p9_fixed_market_relative", "p8_simple_trend"):
        account = cutover / name
        nav = list(csv.DictReader((account / "daily_nav.csv").open(encoding="utf-8")))[-1]
        positions = {
            row["symbol"]: float(row["market_value_cny"])
            for row in csv.DictReader((account / "positions.csv").open(encoding="utf-8"))
            if row["trade_date"] == "2026-09-23"
        }
        account_states[name] = {
            "nav_cny": float(nav["nav_cny"]),
            "cash_cny": float(nav["cash_cny"]),
            "positions_market_value_cny": positions,
        }
    manifest = {
        "schema_version": "adaptive-mas.p9-frozen-inputs.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "last_already_viewed_date": "2026-09-23",
        "first_genuinely_new_date_must_be_after": "2026-09-23",
        "development_results_only": True,
        "workspace_files_sha256": {relative: sha256(project / relative) for relative in sorted(files)},
        "training_csv_sha256": sha256(train_csv),
        "training_csv_role": "train.csv only; test.csv was not opened",
        "cutover_model_seed_sha256": cutover_report["model_cutover_seed_sha256"],
        "cutover_account_states": account_states,
    }
    target = project / "P9_FROZEN_MANIFEST_20260927.json"
    target.write_text(json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(target)


if __name__ == "__main__":
    main()
