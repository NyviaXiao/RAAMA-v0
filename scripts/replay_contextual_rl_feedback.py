"""Build contextual-RL feedback from executed decision logs and account NAV replays."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import date
from pathlib import Path

from adaptive_mas.evaluation.contextual_rl import replay_contextual_feedback


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    with path.open("r", encoding="utf-8") as source:
        rows = [json.loads(line) for line in source if line.strip()]
    if any(not isinstance(row, dict) for row in rows):
        raise ValueError(f"{path} must contain one JSON object per line")
    return rows


def _read_decisions(path: Path) -> list[dict[str, object]]:
    content = path.read_text(encoding="utf-8").strip()
    if not content:
        return []
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        return _read_jsonl(path)
    if isinstance(parsed, dict):
        return [parsed]
    if isinstance(parsed, list) and all(isinstance(row, dict) for row in parsed):
        return parsed
    raise ValueError(f"{path} must contain a decision object or JSONL decision objects")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_new_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as target:
        target.write(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + chr(10))

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replay one-session account outcomes into mature contextual-RL feedback."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--p12-output-dir", type=Path)
    source.add_argument(
        "--decisions-jsonl", "--decision-log", dest="decisions_jsonl", type=Path,
        help="single run_raama_decision.py JSON output or JSONL decisions",
    )
    parser.add_argument("--daily-nav-csv", type=Path)
    parser.add_argument("--evaluation-start-date", type=date.fromisoformat)
    parser.add_argument("--as-of", type=date.fromisoformat)
    parser.add_argument("--risk-budget-annualized", type=float, default=0.20)
    parser.add_argument("--realized-risk-lookback-sessions", type=int, default=20)
    parser.add_argument("--initial-policy-json", type=Path)
    parser.add_argument("--feedback-output", type=Path, required=True)
    parser.add_argument("--report-output", type=Path, required=True)
    parser.add_argument("--policy-output", type=Path)
    args = parser.parse_args()

    if args.p12_output_dir is not None:
        decision_path = args.p12_output_dir / "q_learning_selector_decisions.jsonl"
        nav_path = args.p12_output_dir / "q_learning_selector_daily_nav.csv"
        summary_path = args.p12_output_dir / "evaluation_summary.json"
        protocol_path = args.p12_output_dir / "frozen_protocol.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
        controller = summary["controllers"]["q_learning_selector"]
        evaluation_start = date.fromisoformat(controller["evaluation_start_decision_date"])
        if args.evaluation_start_date and args.evaluation_start_date != evaluation_start:
            raise ValueError("P12 evaluation_start_date must match its frozen evaluation summary")
        args.evaluation_start_date = evaluation_start
        if args.risk_budget_annualized == 0.20:
            args.risk_budget_annualized = float(protocol["risk_budget_annualized"])
    else:
        decision_path = args.decisions_jsonl
        if args.daily_nav_csv is None or args.evaluation_start_date is None or args.as_of is None:
            raise ValueError(
                "decision-log replay requires --daily-nav-csv, --evaluation-start-date and --as-of"
            )
        summary_path = None
        protocol_path = None

    decision_rows = (
        _read_jsonl(decision_path)
        if args.p12_output_dir is not None
        else _read_decisions(decision_path)
    )
    nav_path = args.daily_nav_csv or nav_path
    with nav_path.open("r", encoding="utf-8-sig", newline="") as source_file:
        daily_nav_rows = list(csv.DictReader(source_file))
    if not decision_rows or not daily_nav_rows:
        raise ValueError("decision log and daily NAV replay must both contain rows")

    as_of = args.as_of or max(date.fromisoformat(str(row["trade_date"])) for row in daily_nav_rows)
    policy = (
        json.loads(args.initial_policy_json.read_text(encoding="utf-8"))
        if args.initial_policy_json else {}
    )
    q_values = policy.get("q_values", {})
    action_counts = policy.get("action_counts", {})
    processed_feedback_ids = set(policy.get("processed_feedback_ids", []))
    replay = replay_contextual_feedback(
        decision_rows,
        daily_nav_rows,
        as_of,
        args.evaluation_start_date,
        args.risk_budget_annualized,
        q_values,
        action_counts,
        realized_risk_lookback_sessions=args.realized_risk_lookback_sessions,
        processed_feedback_ids=processed_feedback_ids,
    )
    sources = {
        "decisions_sha256": _sha256(decision_path),
        "daily_nav_sha256": _sha256(nav_path),
    }
    if summary_path:
        sources["evaluation_summary_sha256"] = _sha256(summary_path)
        sources["frozen_protocol_sha256"] = _sha256(protocol_path)
    run_material = {
        "sources": sources,
        "as_of": as_of.isoformat(),
        "evaluation_start_date": args.evaluation_start_date.isoformat(),
        "risk_budget_annualized": args.risk_budget_annualized,
        "realized_risk_lookback_sessions": args.realized_risk_lookback_sessions,
    }
    run_id = hashlib.sha256(
        json.dumps(run_material, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    report = {
        "schema_version": "contextual-rl-replay.v1",
        "run_id": run_id,
        **run_material,
        "decision_rows_seen": len(decision_rows),
        "mature_feedback_rows": len(replay["feedback_rows"]),
        "training_feedback_rows": replay["training_feedback_rows"],
        "compatible_decision_rows": replay["compatible_decision_rows"],
        "incompatible_state_rows": replay["incompatible_state_rows"],
        "incompatible_action_rows": replay["incompatible_action_rows"],
        "evaluation_feedback_rows_excluded": replay["evaluation_feedback_rows_excluded"],
        "policy_updates": replay["policy_updates"],
        "learning_update": replay["learning_update"],
        "limitation": (
            "The current P12 artifact contains only evaluation decisions, omits per-decision contextual state, and uses legacy actions outside contextual_rl.ACTIONS. "
            "It can verify evaluation exclusion but cannot train the contextual policy. "
            "Future training replay decisions must persist a compatible market-regime state and contextual_rl.ACTIONS value at decision time."
            if args.p12_output_dir is not None and replay["policy_updates"] == 0
            else None
        ),
    }
    feedback_path = args.feedback_output
    feedback_path.parent.mkdir(parents=True, exist_ok=True)
    with feedback_path.open("x", encoding="utf-8") as target:
        for row in replay["feedback_rows"]:
            target.write(json.dumps(
                {**row, "replay_run_id": run_id},
                ensure_ascii=False,
                sort_keys=True,
                allow_nan=False,
            ) + chr(10))
    _write_new_json(args.report_output, report)
    if args.policy_output:
        _write_new_json(args.policy_output, {
            "schema_version": "contextual-rl-state.v1",
            "replay_run_id": run_id,
            "source_sha256": sources,
            "q_values": replay["q_values"],
            "action_counts": replay["action_counts"],
            "processed_feedback_ids": replay["processed_feedback_ids"],
        })
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
