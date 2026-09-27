"""Freeze the final P9/P12 source tree and its permitted training inputs."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _project_files(project: Path) -> list[str]:
    files = {
        path.relative_to(project).as_posix()
        for root in (project / "src", project / "scripts", project / "tests")
        for path in root.rglob("*.py")
    }
    files.update({
        "pyproject.toml",
        "uv.lock",
        "AGENTS.md",
        "ARCHITECTURE.md",
        "README.md",
        "EXECUTION_PLAN.md",
        "P9_FINAL_REPORT_20260927.md",
        "P9_P12_FINAL_REPORT_20260927.md",
        "OPTIMIZATION_ROADMAP.md",
        "P9_RETURN_FIRST_PLAN.md",
        "P9_FORWARD_RUNBOOK.md",
        "P9_FROZEN_MANIFEST_20260927.json",
        "experiments/post_p8_research_20260926/three_session_daily/config.toml",
        "experiments/post_p8_research_20260926/three_session_daily/study_results.json",
        "experiments/post_p8_research_20260926/three_session_daily/prediction_decisions.jsonl.gz",
        "experiments/post_p8_research_20260926/three_session_daily/prediction_settlements.jsonl.gz",
        "experiments/post_p8_research_20260926/three_session_daily/training_seed.json.gz",
        "experiments/p9_joint_economics_20260927_full_exit/rolling_calibration_details.jsonl",
        "experiments/p9_forward_cutover_20260927_v9/forward_validation_report.json",
        "experiments/p9_forward_cutover_20260927_v9/model_cutover_seed_asof_2026-09-23.json.gz",
        "experiments/p9_forward_cutover_20260927_v9/p9_daily_selector/daily_nav.csv",
        "experiments/p9_forward_cutover_20260927_v9/p9_daily_selector/positions.csv",
        "experiments/p9_forward_cutover_20260927_v9/p9_fixed_market_relative/daily_nav.csv",
        "experiments/p9_forward_cutover_20260927_v9/p9_fixed_market_relative/positions.csv",
        "experiments/p9_forward_cutover_20260927_v9/p8_simple_trend/daily_nav.csv",
        "experiments/p9_forward_cutover_20260927_v9/p8_simple_trend/positions.csv",
        "data/raw/baostock/fetch-20260924T115050Z/manifest.jsonl",
        "data/raw/baostock/fetch-20260924T150656Z/manifest.jsonl",
    })
    missing = sorted(relative for relative in files if not (project / relative).is_file())
    if missing:
        raise FileNotFoundError(f"冻结输入不存在：{missing}")
    return sorted(files)


def freeze(
    project: Path,
    training_data_dir: Path,
    output: Path,
) -> Path:
    project = project.resolve()
    training_data_dir = training_data_dir.resolve()
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"拒绝覆盖现有冻结清单：{output}")
    base_path = project / "P9_FROZEN_MANIFEST_20260927.json"
    base = json.loads(base_path.read_text(encoding="utf-8"))
    train_csv = training_data_dir / "train.csv"
    if _sha256(train_csv) != base["training_csv_sha256"]:
        raise ValueError("train.csv 已偏离 P9 基线冻结哈希")
    cutover_report = json.loads(
        (project / "experiments/p9_forward_cutover_20260927_v9/forward_validation_report.json").read_text(encoding="utf-8")
    )
    if cutover_report["status"] != "awaiting_new_dates":
        raise ValueError("P9/P12 冻结必须源于无新交易日的截止切换账户")
    for account_name, account_state in base["cutover_account_states"].items():
        if account_name not in cutover_report["account_full_span_summaries_including_viewed_warmup"]:
            raise ValueError(f"截止报告缺少冻结账户：{account_name}")

    relative_files = _project_files(project)
    manifest = {
        "schema_version": "adaptive-mas.p9-p12-final-freeze.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "last_already_viewed_date": "2026-09-23",
        "first_genuinely_new_date_must_be_after": "2026-09-23",
        "p9_baseline_manifest_sha256": _sha256(base_path),
        "workspace_files_sha256": {
            relative: _sha256(project / relative) for relative in relative_files
        },
        "training_csv_sha256": _sha256(train_csv),
        "training_csv_role": "train.csv only; no test.csv input or content is included",
        "cutover_model_seed_sha256": base["cutover_model_seed_sha256"],
        "cutover_account_states": base["cutover_account_states"],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return output


def main() -> None:
    project = Path(__file__).resolve().parents[1]
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--training-data-dir", type=Path, default=Path(r"C:\Users\xiao\Desktop\THU-BDC2026\data"))
    parser.add_argument("--output", type=Path, default=project / "P9_P12_FROZEN_MANIFEST_20260927.json")
    args = parser.parse_args()
    print(freeze(project, args.training_data_dir, args.output))


if __name__ == "__main__":
    main()
