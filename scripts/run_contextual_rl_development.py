"""Run one chronological contextual-RL development replay on frozen P9 inputs."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import statistics
import tomllib
from collections import Counter
from datetime import date, datetime, timezone
from pathlib import Path

from adaptive_mas.backtest.simulator import simulate_portfolio
from adaptive_mas.data import load_research_data
from adaptive_mas.decision import PortfolioState
from adaptive_mas.evaluation.contextual_rl import (
    ACTIONS,
    BASELINE_ACTION,
    FUSION_PROFILES,
    MINIMUM_ACTION_SAMPLES,
    apply_action,
    choose_action,
    contextual_state,
    update_policy,
)


TRAIN_FRACTION = 0.70
SEED = 20261012
RETURN_METHOD = "rl_controlled_fusion"
CALIBRATED_METHODS = ("market_plus_relative_fusion", "trend")


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def _read_jsonl_gzip(path: Path) -> list[dict[str, object]]:
    with gzip.open(path, "rt", encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")


def _write_gzip_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    with gzip.open(path, "wt", encoding="utf-8", newline="\n") as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _date_seed(seed: int, day: date) -> int:
    material = f"{seed}:{day.isoformat()}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big")


def _calibrate_forecast(
    forecast: dict[str, object],
    corrections: dict[str, dict[str, object]],
) -> dict[str, object]:
    """Apply only the frozen P9 mean corrections and preserve matured RMSE values."""
    row = json.loads(json.dumps(forecast))
    expected = row["expected_returns_by_method"]
    for method in CALIBRATED_METHODS:
        correction = float(corrections[method]["total_mean_bias_estimate"])
        expected[method] = {
            symbol: float(value) + correction
            for symbol, value in expected[method].items()
        }
    return row


def _portfolio_state(q_values: dict[str, dict[str, float]], counts: dict[str, dict[str, int]]) -> dict[str, object]:
    return {
        "q_values": {state: dict(values) for state, values in q_values.items()},
        "action_counts": {state: dict(values) for state, values in counts.items()},
    }


def _filter_research(research, end: date):
    from adaptive_mas.data.research import ResearchData

    return ResearchData(
        trading_dates=[day for day in research.trading_dates if day <= end],
        adjusted_bars=[row for row in research.adjusted_bars if row["trade_date"] <= end],
        execution_bars=[row for row in research.execution_bars if row["trade_date"] <= end],
        index_bars=[row for row in research.index_bars if row["trade_date"] <= end],
        membership_snapshots=[row for row in research.membership_snapshots if row.available_at <= end],
    )


def _controller(
    *,
    research,
    risk_budget: float,
    risk_lookback: int,
    risk_minimum: int,
    evaluation_start: date,
    q_values: dict[str, dict[str, float]],
    counts: dict[str, dict[str, int]],
    processed_ids: set[str],
    progress,
    training: bool,
):
    matured_updates = 0
    waiting_for_risk_window = 0

    def decide(*, forecast, account, decision_date, daily_nav, decisions, decision_pause=None):
        nonlocal matured_updates, waiting_for_risk_window
        day = decision_date
        split = "training" if day < evaluation_start else "evaluation"
        if training and day < evaluation_start and decisions:
            previous = decisions[-1]
            previous_date = date.fromisoformat(str(previous["decision_date"]))
            if (
                previous.get("data_split") == "training"
                and previous.get("execution_date") == day.isoformat()
                and previous.get("decision_status") != "paused"
                and previous.get("action_applied", True)
            ):
                returns = [float(row["daily_return"]) for row in daily_nav[-20:]]
                if len(returns) < 20:
                    waiting_for_risk_window += 1
                else:
                    realized_risk = statistics.stdev(returns) * math.sqrt(250)
                    feedback_id = hashlib.sha256(
                        f"{previous_date.isoformat()}:{previous['action']}:{day.isoformat()}".encode("utf-8")
                    ).hexdigest()
                    update = update_policy(
                        [{
                            "feedback_id": feedback_id,
                            "data_split": "training",
                            "decision_date": previous_date.isoformat(),
                            "label_available_at": day.isoformat(),
                            "horizon_sessions": 1,
                            "state": str(previous["state"]),
                            "action": str(previous["action"]),
                            "net_return_after_costs": float(daily_nav[-1]["daily_return"]),
                            "realized_annualized_risk": realized_risk,
                            "risk_budget_annualized": risk_budget,
                            "starting_nav_cny": float(previous["current_nav_cny"]),
                            "role_qwen_cost_cny": 0.0,
                            "supplementary_query_cost_cny": 0.0,
                            "analysis_cost_cny": 0.0,
                        }],
                        research.trading_dates,
                        day,
                        q_values,
                        counts,
                        processed_feedback_ids=processed_ids,
                    )
                    processed_ids.update(update["processed_feedback_ids"])
                    matured_updates += int(update["matured_training_feedback_used"])

        if decision_pause is not None:
            audit = {
                "data_split": split,
                "state": None,
                "action": None,
                "action_applied": False,
                "decision_status": "paused",
                "pause_reason": decision_pause["reason"],
                "paused_symbols": decision_pause["symbols"],
                "supplementary_query": False,
                "matured_feedback_updates_before_decision": matured_updates,
                "feedback_skipped_for_risk_warmup": waiting_for_risk_window,
                "q_values_snapshot": _portfolio_state(q_values, counts),
            }
            progress(
                f"{day.isoformat()} split={split} paused="
                f"{decision_pause['reason']} matured={matured_updates}"
            )
            return forecast, audit

        state = contextual_state(
            forecast,
            account,
            risk_budget,
            research,
            day,
            risk_lookback_sessions=risk_lookback,
            risk_minimum_observations=risk_minimum,
        )
        available = set(forecast["expected_returns_by_method"])
        exploration = 1.0 if training and split == "training" else 0.0
        seed = _date_seed(SEED, day)
        choice = choose_action(
            state["state"],
            q_values,
            counts,
            query_available=False,
            available_methods=available,
            minimum_samples=MINIMUM_ACTION_SAMPLES,
            exploration_rate=exploration,
            seed=seed,
        )
        controlled = apply_action(forecast, choice["weights"])
        controlled["controller_action"] = choice["action"]
        controlled["controller_return_method"] = RETURN_METHOD
        audit = {
            "data_split": split,
            "state": state["state"],
            "market_return_20s": state["market_return_20s"],
            "expert_disagreement": state["expert_disagreement"],
            "account_cash_weight": state["account_cash_weight"],
            "current_account_annualized_risk_estimate": state["current_account_annualized_risk_estimate"],
            "risk_state": state["risk_state"],
            "risk_audit_unavailable_symbols": state["risk_audit_unavailable_symbols"],
            "action": choice["action"],
            "action_applied": True,
            "decision_status": "selected",
            "fusion_profile": choice["fusion_profile"],
            "action_weights": choice["weights"],
            "learned_action_count": choice["learned_action_count"],
            "estimate_scope": choice["estimate_scope"],
            "policy_status": choice["policy_status"],
            "explored": choice["explored"],
            "exploration_rate": exploration,
            "exploration_seed": seed,
            "supplementary_query": False,
            "matured_feedback_updates_before_decision": matured_updates,
            "feedback_skipped_for_risk_warmup": waiting_for_risk_window,
            "q_values_snapshot": _portfolio_state(q_values, counts),
        }
        progress(
            f"{day.isoformat()} split={split} action={choice['action']} "
            f"matured={matured_updates}"
        )
        return controlled, audit

    return decide, lambda: {
        "matured_training_feedback_used": matured_updates,
        "training_feedback_skipped_before_20_return_window": waiting_for_risk_window,
    }


def main() -> None:
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-data-dir", type=Path, default=Path(r"C:\Users\xiao\Desktop\THU-BDC2026\data"))
    parser.add_argument("--baostock-run-dir", type=Path, default=project / "data" / "raw" / "baostock" / "fetch-20260924T115050Z")
    parser.add_argument("--source-study-dir", type=Path, default=project / "experiments" / "post_p8_research_20260926" / "three_session_daily")
    parser.add_argument("--calibration-dir", type=Path, default=project / "experiments" / "p9_joint_economics_20260927_full_exit")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output directory: {output}")
    output.mkdir(parents=True)
    status = {
        "status": "incomplete",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "train_fraction": TRAIN_FRACTION,
        "seed": SEED,
    }

    def progress(message: str) -> None:
        status["last_progress"] = message
        status["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
        _write_json(output / "RUN_STATUS.json", status)
        with (output / "progress.log").open("a", encoding="utf-8", newline="\n") as log:
            log.write(f"{status['updated_at_utc']} {message}\n")
        print(f"[contextual RL development] {message}", flush=True)

    source = args.source_study_dir.resolve()
    calibration_dir = args.calibration_dir.resolve()
    config = tomllib.loads((source / "config.toml").read_text(encoding="utf-8"))
    calibration_rows = _read_jsonl(calibration_dir / "rolling_calibration_details.jsonl")
    calibration = {}
    for row in calibration_rows:
        day = str(row["decision_date"])
        methods = row["by_method"]
        if any(str(methods[method]["window_end_label_date"]) >= day for method in CALIBRATED_METHODS):
            raise ValueError(f"{day} P9 mean calibration uses an unmatured label")
        calibration[day] = methods
    source_forecasts = {
        str(row["prediction_date"]): row
        for row in _read_jsonl_gzip(source / "prediction_decisions.jsonl.gz")
    }
    common_days = sorted(set(source_forecasts) & set(calibration))
    if not common_days:
        raise ValueError("No matching frozen forecasts and mature P9 calibration rows")
    research = load_research_data(args.training_data_dir, args.baostock_run_dir)
    days = [
        day for day in common_days
        if date.fromisoformat(day) in research.trading_dates
        and research.trading_dates.index(date.fromisoformat(day)) + 1 < len(research.trading_dates)
    ]
    if len(days) < 40:
        raise ValueError("Need at least 40 point-in-time forecast dates with next-session execution")
    if any(source_forecasts[day].get("input_cutoff") != day for day in days):
        raise ValueError("A forecast input cutoff does not match its decision date")
    for day in days:
        forecast = source_forecasts[day]
        errors = forecast.get("forecast_errors_by_method", {})
        expected = forecast.get("expected_returns_by_method", {})
        for method in ("market_plus_relative_fusion", "trend", "reversal", "liquidity_confirmation"):
            if method not in expected or errors.get(method) is None:
                raise ValueError(f"{day} lacks a mature input for contextual action method {method}")
        for method in CALIBRATED_METHODS:
            if method not in forecast.get("target_weights_by_method", {}):
                raise ValueError(f"{day} lacks its frozen P9 candidate basket for {method}")

    split = int(len(days) * TRAIN_FRACTION)
    train_days = days[:split]
    evaluation_days = days[split:]
    if not train_days or not evaluation_days:
        raise ValueError("Chronological split has no training or frozen-evaluation dates")
    evaluation_start = date.fromisoformat(evaluation_days[0])
    calibrated = {
        day: _calibrate_forecast(source_forecasts[day], calibration[day])
        for day in days
    }
    q_values = {state: {action: 0.0 for action in ACTIONS} for state in ("market_up", "market_down")}
    counts = {state: {action: 0 for action in ACTIONS} for state in ("market_up", "market_down")}
    processed_ids: set[str] = set()
    costs = {
        key: float(config[key])
        for key in (
            "commission_per_side_bps", "transfer_fee_per_side_bps",
            "stamp_duty_sell_bps", "slippage_per_side_bps",
        )
    }
    optimizer_config = dict(config)
    optimizer_config.update({
        "optimizer_return_method": RETURN_METHOD,
        "optimizer_candidate_scope": "frozen_method_target",
        "risk_aversion": 0.0,
        "forecast_error_multiplier": 0.0,
        "risk_budget_annualized": float(config["risk_budget_annualized"]),
    })
    boundary_date = date.fromisoformat(evaluation_days[0])
    train_forecasts = [calibrated[day] for day in train_days]
    # End before the first evaluation date. The final training action's
    # next-session NAV matures on the boundary date and is intentionally
    # excluded from Q updates.
    research_training = _filter_research(research, date.fromisoformat(train_days[-1]))
    train_controller, train_controller_summary = _controller(
        research=research,
        risk_budget=float(config["risk_budget_annualized"]),
        risk_lookback=int(config["risk_lookback_sessions"]),
        risk_minimum=int(config["risk_minimum_observations"]),
        evaluation_start=boundary_date,
        q_values=q_values,
        counts=counts,
        processed_ids=processed_ids,
        progress=progress,
        training=True,
    )
    progress(f"training replay begins: {len(train_days)} training decisions; evaluation starts {boundary_date}")
    training_result = simulate_portfolio(
        research_training,
        [],
        float(config["initial_capital_cny"]),
        costs,
        int(config["liquidity_lookback_sessions"]),
        float(config["maximum_daily_participation"]),
        float(config["maximum_one_way_turnover"]),
        forecast_decisions=train_forecasts,
        optimizer_config=optimizer_config,
        progress_callback=lambda message: progress(f"training simulator {message}"),
        evaluation_start_date=date.fromisoformat(train_days[0]),
        forecast_controller=train_controller,
    )
    training_controller_stats = train_controller_summary()

    frozen_controller, frozen_controller_summary = _controller(
        research=research,
        risk_budget=float(config["risk_budget_annualized"]),
        risk_lookback=int(config["risk_lookback_sessions"]),
        risk_minimum=int(config["risk_minimum_observations"]),
        evaluation_start=boundary_date,
        q_values=q_values,
        counts=counts,
        processed_ids=set(),
        progress=progress,
        training=False,
    )
    evaluation_forecasts = [calibrated[day] for day in evaluation_days]
    progress(f"frozen RL evaluation begins: {len(evaluation_forecasts)} decisions from {boundary_date}")
    rl_evaluation = simulate_portfolio(
        research,
        [],
        float(config["initial_capital_cny"]),
        costs,
        int(config["liquidity_lookback_sessions"]),
        float(config["maximum_daily_participation"]),
        float(config["maximum_one_way_turnover"]),
        forecast_decisions=evaluation_forecasts,
        optimizer_config=optimizer_config,
        progress_callback=lambda message: progress(f"frozen RL simulator {message}"),
        evaluation_start_date=boundary_date,
        forecast_controller=frozen_controller,
    )
    frozen_controller_summary()

    baseline_config = dict(optimizer_config)
    baseline_config["optimizer_return_method"] = "market_plus_relative_fusion"
    progress("same-account P9 baseline evaluation begins")
    p9_evaluation = simulate_portfolio(
        research,
        [],
        float(config["initial_capital_cny"]),
        costs,
        int(config["liquidity_lookback_sessions"]),
        float(config["maximum_daily_participation"]),
        float(config["maximum_one_way_turnover"]),
        forecast_decisions=evaluation_forecasts,
        optimizer_config=baseline_config,
        progress_callback=lambda message: progress(f"P9 baseline simulator {message}"),
        evaluation_start_date=boundary_date,
    )

    train_decisions = [
        row for row in training_result["decisions"]
        if str(row["decision_date"]) < boundary_date.isoformat()
    ]
    training_actions = [
        row for row in train_decisions
        if row.get("action_applied", True)
        and row.get("action") is not None
        and row.get("execution_date") is not None
    ]
    training_pauses = [row for row in train_decisions if row.get("decision_status") == "paused"]
    eval_decisions = [
        row for row in rl_evaluation["decisions"]
        if str(row["data_split"]) == "evaluation"
    ]
    eval_actions = [
        row for row in eval_decisions
        if row.get("action_applied", True)
        and row.get("action") is not None
        and row.get("execution_date") is not None
    ]
    evaluation_pauses = [row for row in eval_decisions if row.get("decision_status") == "paused"]
    p9_pauses = [row for row in p9_evaluation["decisions"] if row.get("decision_status") == "paused"]
    train_actions = Counter(str(row["action"]) for row in training_actions)
    evaluation_actions = Counter(str(row["action"]) for row in eval_actions)
    matured_action_counts = {
        action: sum(int(counts[state][action]) for state in counts)
        for action in ACTIONS
    }
    pooled_q_value_estimates = {
        action: (
            sum(q_values[state][action] * counts[state][action] for state in counts)
            / matured_action_counts[action]
            if matured_action_counts[action]
            else None
        )
        for action in ACTIONS
    }
    evaluation_non_p9_dates = [
        str(row["decision_date"])
        for row in eval_actions
        if row["action"] != BASELINE_ACTION
    ]
    training_pause_audit = [
        {"decision_date": row["decision_date"], "reason": row["pause_reason"], "symbols": row["paused_symbols"]}
        for row in training_pauses
    ]
    rl_pause_audit = [
        {"decision_date": row["decision_date"], "reason": row["pause_reason"], "symbols": row["paused_symbols"]}
        for row in evaluation_pauses
    ]
    p9_pause_audit = [
        {"decision_date": row["decision_date"], "reason": row["pause_reason"], "symbols": row["paused_symbols"]}
        for row in p9_pauses
    ]
    risk_audit_unavailable_counts = Counter(
        symbol
        for row in [*train_decisions, *eval_actions]
        for symbol in row.get("risk_audit_unavailable_symbols", [])
    )
    common_summary = {
        "evaluation_decision_start": boundary_date.isoformat(),
        "evaluation_trade_start": rl_evaluation["summary"]["start_date"],
        "evaluation_trade_end": rl_evaluation["summary"]["end_date"],
        "evaluation_trading_days": rl_evaluation["summary"]["trading_days"],
        "initial_capital_cny": float(config["initial_capital_cny"]),
        "same_starting_account_cash_and_holdings": True,
        "common_market_input_sha256": _sha256(source / "prediction_decisions.jsonl.gz"),
        "costs_bps": costs,
    }
    result = {
        "run_status": "complete",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "forecast_source": str(source),
            "calibration_source": str(calibration_dir),
            "training_decisions": len(train_days),
            "evaluation_decisions": len(evaluation_days),
            "training_decision_start": train_days[0],
            "training_decision_end": train_days[-1],
            "evaluation_decision_start": boundary_date.isoformat(),
            "train_fraction": TRAIN_FRACTION,
            "actions": list(ACTIONS),
            "eligible_actions": [
                action for action in ACTIONS
                if set(FUSION_PROFILES[action.split(":", 1)[0]])
                <= set(calibrated[train_days[0]]["expected_returns_by_method"])
            ],
            "state": "market_up/market_down from the decision date and prior 20-session index close",
            "exploration_rate_during_training": 1.0,
            "exploration_rate_during_evaluation": 0.0,
            "minimum_samples_per_action": MINIMUM_ACTION_SAMPLES,
            "training_reward": "one-session net NAV return - 0.5*excess_realized_annualized_risk^2 - analysis costs / starting NAV",
            "feedback_timing": "one-session feedback updates only when its label date is strictly before evaluation_start; the final training action matures on the boundary and is excluded",
            "training_feedback_used": training_controller_stats["matured_training_feedback_used"],
            "training_feedback_skipped_before_20_return_window": training_controller_stats["training_feedback_skipped_before_20_return_window"],
            "training_selected_action_days_with_next_session": len(training_actions),
            "training_paused_decision_count": len(training_pauses),
            "training_pause_decisions": training_pause_audit,
            "matured_feedback_action_counts": matured_action_counts,
                "paused_position_mark_policy": "retain last observed adjusted value when a current mark is missing; do not impute a return",
            "qwen_calls": 0,
            "numeric_policy_learning_is_implemented": True,
            "analysis_query_actions_implemented": False,
            "qwen_opinions_change_numeric_policy_or_portfolio_weights": False,
                "policy_evaluation_status": (
                    "p9_selected_on_all_frozen_evaluation_dates"
                    if not evaluation_non_p9_dates
                    else "historical_non_p9_actions_observed_not_unseen_validation"
                ),
            "market_data_downloads": 0,
            "test_csv_read": False,
            "historical_evaluation_is_not_unseen": True,
        },
        "training": {
            "summary": training_result["summary"],
            "frozen_policy_path": str(output / "frozen_contextual_rl_state.json"),
            "action_counts": dict(train_actions),
            "selected_action_days_with_next_session": len(training_actions),
            "paused_decision_count": len(training_pauses),
            "pause_decisions": training_pause_audit,
            "matured_feedback_action_counts": matured_action_counts,
            "final_q_values": q_values,
            "final_action_counts_by_state": counts,
        },
        "frozen_evaluation": {
            **common_summary,
            "rl_summary": rl_evaluation["summary"],
            "p9_fixed_summary": p9_evaluation["summary"],
            "rl_action_counts": dict(evaluation_actions),
            "rl_pause_decision_count": len(evaluation_pauses),
            "rl_pause_decisions": rl_pause_audit,
            "p9_pause_decision_count": len(p9_pauses),
            "p9_pause_decisions": p9_pause_audit,
            "rl_non_p9_action_days": len(evaluation_non_p9_dates),
            "rl_non_p9_action_dates": evaluation_non_p9_dates,
            "rl_non_p9_action_counts": {
                action: evaluation_actions[action]
                for action in ACTIONS
                if action != BASELINE_ACTION
            },
                "policy_effectiveness_evaluated": bool(evaluation_non_p9_dates),
                "pooled_q_value_estimates": pooled_q_value_estimates,
                "action_selection_explanation": (
                    "P9 had the highest pooled mature Q estimate and was selected on every frozen-evaluation decision."
                    if not evaluation_non_p9_dates
                    else "The frozen evaluation included non-P9 selected actions; see the action dates and pooled Q estimates."
                ),
            "return_difference_vs_p9": float(rl_evaluation["summary"]["total_return"]) - float(p9_evaluation["summary"]["total_return"]),
            "max_drawdown_difference_vs_p9": float(rl_evaluation["summary"]["maximum_drawdown"]) - float(p9_evaluation["summary"]["maximum_drawdown"]),
        },
        "risk_audit_unavailable_symbol_decision_counts": dict(risk_audit_unavailable_counts),
    }
    policy_artifact = {
        "schema_version": "contextual-rl-state.v1",
        "q_values": q_values,
        "action_counts": counts,
        "processed_feedback_ids": sorted(processed_ids),
        "last_updated_as_of": train_days[-1],
        "freeze_metadata": {
            "training_decision_end": train_days[-1],
            "evaluation_start_date": boundary_date.isoformat(),
            "policy_frozen_before_evaluation": True,
            "training_selected_action_days_with_next_session": len(training_actions),
            "training_paused_decision_count": len(training_pauses),
            "matured_feedback_action_counts": matured_action_counts,
            "historical_evaluation_non_p9_action_days": len(evaluation_non_p9_dates),
                "policy_evaluation_status": (
                    "p9_selected_on_all_frozen_evaluation_dates"
                    if not evaluation_non_p9_dates
                    else "historical_non_p9_actions_observed_not_unseen_validation"
                ),
            "historical_evaluation_is_not_unseen": True,
        },
    }
    _write_json(output / "frozen_contextual_rl_state.json", policy_artifact)
    _write_json(output / "summary.json", result)
    for name, run in (("training", training_result), ("rl_evaluation", rl_evaluation), ("p9_fixed_evaluation", p9_evaluation)):
        directory = output / name
        directory.mkdir(exist_ok=False)
        _write_csv(directory / "daily_nav.csv", run["daily_nav"])
        _write_csv(directory / "orders.csv", run["orders"])
        _write_csv(directory / "positions.csv", run["positions"])
        _write_jsonl(directory / "simulator_decisions.jsonl", run["decisions"])
    _write_gzip_jsonl(output / "matured_forecasts.jsonl.gz", [calibrated[day] for day in days])
    status.update({
        "status": "complete",
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "summary_path": str(output / "summary.json"),
    })
    _write_json(output / "RUN_STATUS.json", status)
    progress("complete")


if __name__ == "__main__":
    main()
