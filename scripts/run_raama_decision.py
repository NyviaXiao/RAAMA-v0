"""Run a user-account RAAMA decision with auditable multi-agent/RL coordination."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
import tomllib
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from adaptive_mas.agents.public_experts import build_public_expert_snapshot
from adaptive_mas.data import load_oos_research_data
from adaptive_mas.decision import PortfolioState, create_daily_decision
from adaptive_mas.evaluation.contextual_rl import update_policy
from adaptive_mas.evaluation.expert_memory import candidate_performance_effects, update_matured_performance_memory


SHANGHAI = ZoneInfo("Asia/Shanghai")


def _read_jsonl_records(path: Path | None, label: str) -> list[dict[str, object]]:
    if path is None:
        return []
    with path.open("r", encoding="utf-8") as source:
        rows = [json.loads(line) for line in source if line.strip()]
    if any(not isinstance(row, dict) for row in rows):
        raise ValueError(f"{label} JSONL 每行必须是对象")
    return rows


def _read_feedback(path: Path | None) -> list[dict[str, object]]:
    return _read_jsonl_records(path, "RL feedback")


def _date_seed(seed: int, as_of: date) -> int:
    material = f"{seed}:{as_of.isoformat()}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big")

def main() -> None:
    project_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--baostock-run-dir", type=Path, required=True)
    parser.add_argument("--portfolio-state-json", type=Path, required=True)
    parser.add_argument("--as-of", type=date.fromisoformat, required=True)
    parser.add_argument(
        "--evaluation-start-date", type=date.fromisoformat, required=True,
        help="显式评价边界；该日期及之后冻结策略更新与探索。",
    )
    parser.add_argument("--horizon-sessions", type=int, required=True)
    parser.add_argument("--risk-budget-annualized", type=float, required=True)
    parser.add_argument(
        "--config", type=Path,
        default=project_dir / "configs" / "daily_three_session.toml",
    )
    parser.add_argument(
        "--training-seed", type=Path,
        default=project_dir / "data" / "processed"
        / "p8_daily_three_session_continuous_v1" / "training_seed.json.gz",
    )
    parser.add_argument(
        "--public-cache-dir", type=Path,
        default=project_dir / "data" / "raw" / "akshare",
    )
    parser.add_argument("--feedback-jsonl", type=Path, help="成熟反馈 JSONL；由 scripts/replay_contextual_rl_feedback.py 从决策和账户 NAV 回放生成")
    parser.add_argument("--expert-forecasts-jsonl", type=Path)
    parser.add_argument("--expert-outcomes-jsonl", type=Path)
    parser.add_argument("--rl-state-file", type=Path)
    parser.add_argument("--minimum-action-samples", type=int, default=20)
    parser.add_argument(
        "--exploration-rate", type=float, default=0.0,
        help="默认 0 为保守固定 P9；开发期训练必须显式设为非零，评价期自动归零。",
    )
    parser.add_argument("--seed", type=int, default=0, help="基础种子；程序按决策日期派生可复现探索种子。")
    parser.add_argument("--enable-qwen", action="store_true")
    parser.add_argument("--announcement-text-limit", type=int, default=0)
    parser.add_argument("--announcement-document-dir", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if args.horizon_sessions < 1:
        raise ValueError("预测期限必须为正整数")
    if not math.isfinite(args.risk_budget_annualized) or args.risk_budget_annualized <= 0:
        raise ValueError("年化风险预算必须是有限正数")
    if args.minimum_action_samples < 1:
        raise ValueError("RL动作至少成熟样本数必须为正")
    if not 0 <= args.exploration_rate <= 1:
        raise ValueError("RL探索率必须处于 [0, 1]")
    evaluation_period = (
        args.evaluation_start_date is not None
        and args.as_of >= args.evaluation_start_date
    )
    if (args.expert_forecasts_jsonl is None) != (args.expert_outcomes_jsonl is None):
        raise ValueError("P11 绩效记忆需同时指定 expert forecasts 和 outcomes JSONL")

    if args.announcement_text_limit and args.announcement_document_dir is None:
        raise ValueError("读取公告全文需同时指定 --announcement-document-dir")

    config_bytes = args.config.read_bytes()
    config = tomllib.loads(config_bytes.decode("utf-8"))
    config["prediction_horizon_sessions"] = args.horizon_sessions
    config["risk_budget_annualized"] = args.risk_budget_annualized
    account = json.loads(args.portfolio_state_json.read_text(encoding="utf-8"))
    if set(account) != {"cash_cny", "holdings_cny"}:
        raise ValueError("账户状态 JSON 必须且只能包含 cash_cny 与 holdings_cny")
    if not isinstance(account["holdings_cny"], dict):
        raise ValueError("holdings_cny 必须是证券代码到持仓市值的对象")
    portfolio = PortfolioState(
        cash_cny=float(account["cash_cny"]),
        holdings_cny={
            str(symbol): float(value)
            for symbol, value in account["holdings_cny"].items()
        },
    )
    portfolio.validate()

    with gzip.open(args.training_seed, "rt", encoding="utf-8") as seed_file:
        training_state = json.load(seed_file)
    research = load_oos_research_data(args.baostock_run_dir)
    research.trading_dates.index(args.as_of)

    decision_time = datetime.combine(args.as_of, datetime.min.time()).replace(
        hour=16, tzinfo=SHANGHAI
    )
    available_symbols = set(research.membership_at(args.as_of)) | set(portfolio.holdings_cny)
    if args.expert_forecasts_jsonl is not None:
        expert_forecasts = _read_jsonl_records(args.expert_forecasts_jsonl, "P11 forecasts")
        expert_outcomes = _read_jsonl_records(args.expert_outcomes_jsonl, "P11 outcomes")
        matured_expert_memory = update_matured_performance_memory(
            expert_forecasts, expert_outcomes, decision_time
        )
        p11_candidate_effects = candidate_performance_effects(
            matured_expert_memory, research.trading_dates
        )
        p11_memory_summary = {
            "status": "computed_from_maturity_gated_inputs",
            "matured_records": len(matured_expert_memory),
            "candidate_effects": p11_candidate_effects,
            "forecasts_sha256": hashlib.sha256(args.expert_forecasts_jsonl.read_bytes()).hexdigest(),
            "outcomes_sha256": hashlib.sha256(args.expert_outcomes_jsonl.read_bytes()).hexdigest(),
        }
    else:
        p11_candidate_effects = {}
        p11_memory_summary = {
            "status": "no_p11_history_supplied",
            "matured_records": 0,
            "candidate_effects": {},
        }

    public_snapshot = build_public_expert_snapshot(
        args.public_cache_dir,
        decision_time,
        args.horizon_sessions,
        allowed_symbols=available_symbols,
        announcement_text_limit=args.announcement_text_limit,
        announcement_document_dir=args.announcement_document_dir,
        include_qwen=False,
    )
    public_snapshot["p11_candidate_effects"] = p11_candidate_effects
    public_snapshot["p11_memory_summary"] = p11_memory_summary

    q_values = {}
    action_counts = {}
    processed_feedback_ids = set()
    if args.rl_state_file and args.rl_state_file.is_file():
        persisted = json.loads(args.rl_state_file.read_text(encoding="utf-8"))
        if persisted["schema_version"] != "contextual-rl-state.v1":
            raise ValueError("RL状态文件 schema_version 不匹配")
        q_values = persisted["q_values"]
        action_counts = persisted["action_counts"]
        processed_feedback_ids = set(persisted.get("processed_feedback_ids", []))
    feedback = _read_feedback(args.feedback_jsonl)
    if evaluation_period:
        learning_update = {
            "matured_training_feedback_used": 0,
            "unmatured_training_feedback_ignored": 0,
            "evaluation_feedback_excluded_from_training": sum(
                row.get("data_split") == "evaluation" for row in feedback
            ),
            "new_feedback_ids_processed": 0,
            "duplicate_feedback_rows_skipped": 0,
            "processed_feedback_ids": sorted(processed_feedback_ids),
            "policy_updates": "frozen_for_evaluation_period",
        }
    else:
        learning_update = update_policy(
            feedback,
            research.trading_dates,
            args.as_of,
            q_values,
            action_counts,
            processed_feedback_ids=processed_feedback_ids,
        )

    qwen_configured = bool(
        os.environ.get("POPAGENT_LLM_KEY")
        and os.environ.get("POPAGENT_LLM_URL")
    )
    manifest_path = args.baostock_run_dir / "manifest.jsonl"
    versions = {
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "baostock_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "training_seed_sha256": hashlib.sha256(args.training_seed.read_bytes()).hexdigest(),
        "public_data_cutoff": decision_time.isoformat(),
    }
    decision = create_daily_decision(
        research,
        config,
        args.as_of,
        portfolio,
        training_state,
        versions,
        {
            "public_snapshot": public_snapshot,
            "q_values": q_values,
            "action_counts": action_counts,
            "matured_feedback_used": learning_update["matured_training_feedback_used"],
            "minimum_action_samples": args.minimum_action_samples,
            "exploration_rate": 0.0 if evaluation_period else args.exploration_rate,
            "seed": _date_seed(args.seed, args.as_of),
            "data_split": "evaluation" if evaluation_period else "training",
            "enable_qwen": args.enable_qwen,
            "query_available": args.enable_qwen and qwen_configured,
        },
    )
    payload = decision.to_dict()
    trace = payload["evidence"].get("collaborative_architecture")
    if trace is not None:
        trace["rl_controller"]["learning_update"] = learning_update
        trace["rl_controller"]["feedback_contract"] = {
            "required_fields": [
                "data_split", "decision_date", "label_available_at", "horizon_sessions",
                "state", "action", "net_return_after_costs", "realized_annualized_risk",
                "risk_budget_annualized", "starting_nav_cny", "role_qwen_cost_cny",
                "supplementary_query_cost_cny", "analysis_cost_cny",
            ],
            "generated_by": "scripts/replay_contextual_rl_feedback.py",
            "label_horizon_sessions": 1,
            "one_session_account_nav_return_includes_execution_costs": True,
            "only_mature_training_rows_update_policy": True,
            "evaluation_rows_remain_separate": True,
            "supplementary_query_policy": (
                "disabled_until_cited_challenge evidence has a predeclared calibrated numeric effect on candidates or weights"
            ),
        }
        trace["public_snapshot_coverage"] = public_snapshot["source_coverage"]
    serialized = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        output_temp = args.output.with_name(args.output.name + ".tmp")
        output_temp.write_text(serialized, encoding="utf-8")
        output_temp.replace(args.output)
    else:
        print(serialized, end="")
    if args.rl_state_file and not evaluation_period:
        args.rl_state_file.parent.mkdir(parents=True, exist_ok=True)
        state_payload = {
            "schema_version": "contextual-rl-state.v1",
            "q_values": q_values,
            "action_counts": action_counts,
            "processed_feedback_ids": learning_update["processed_feedback_ids"],
            "last_updated_as_of": args.as_of.isoformat(),
            "learning_update": learning_update,
        }
        state_temp = args.rl_state_file.with_name(args.rl_state_file.name + ".tmp")
        state_temp.write_text(
            json.dumps(state_payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        state_temp.replace(args.rl_state_file)


if __name__ == "__main__":
    main()
