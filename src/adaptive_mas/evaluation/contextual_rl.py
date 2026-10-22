"""Contextual reinforcement controller for the live multi-agent portfolio path.

The legacy P12 two-state/two-action Q table stays reproducible in rl_control.py.
This controller learns a small contextual action set only from mature account
outcomes after transaction costs, risk and any Qwen query cost are observed.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import statistics
from datetime import date

from adaptive_mas.evaluation.mvp_research import _risk_estimates


FUSION_PROFILES = {
    "p9_fixed": {"market_plus_relative_fusion": 1.0},
    "trend_tilt_25": {"market_plus_relative_fusion": 0.75, "trend": 0.25},
    "reversal_tilt_25": {"market_plus_relative_fusion": 0.75, "reversal": 0.25},
    "liquidity_tilt_25": {
        "market_plus_relative_fusion": 0.75,
        "liquidity_confirmation": 0.25,
    },
    "balanced_research": {
        "market_plus_relative_fusion": 0.50,
        "trend": 1 / 6,
        "reversal": 1 / 6,
        "liquidity_confirmation": 1 / 6,
    },
    "fundamental_tilt_25": {
        "market_plus_relative_fusion": 0.75,
        "fundamental_p11": 0.25,
    },
    "news_tilt_25": {
        "market_plus_relative_fusion": 0.75,
        "news_p11": 0.25,
    },
    "fundamental_news_balanced": {
        "market_plus_relative_fusion": 0.50,
        "fundamental_p11": 0.25,
        "news_p11": 0.25,
    },
}
ACTIONS = tuple(
    f"{profile}:no_query"
    for profile in FUSION_PROFILES
)
BASELINE_ACTION = "p9_fixed:no_query"
MINIMUM_ACTION_SAMPLES = 20


def contextual_state(
    forecast,
    portfolio,
    risk_budget_annualized,
    research,
    decision_date,
    risk_lookback_sessions=60,
    risk_minimum_observations=40,
):
    """Build a point-in-time state from calibrated expert returns and the account."""
    index = research.trading_dates.index(decision_date)
    if index < 20:
        raise ValueError(f"{decision_date} has no complete 20-session market-state window")
    index_by_date = {
        row["trade_date"]: float(row["close"])
        for row in research.index_bars
        if row.get("close") is not None
    }
    market_return = (
        index_by_date[decision_date]
        / index_by_date[research.trading_dates[index - 20]]
        - 1
    )
    market_state = "market_up" if market_return >= 0 else "market_down"

    expected = forecast["expected_returns_by_method"]
    common = sorted(set.intersection(*(
        set(expected[agent])
        for agent in ("trend", "reversal", "liquidity_confirmation")
    )))
    if not common:
        raise ValueError("No shared calibrated quantitative expert returns for RL state")
    spreads = [
        statistics.pstdev([
            float(expected[agent][symbol])
            for agent in ("trend", "reversal", "liquidity_confirmation")
        ])
        for symbol in common
    ]
    errors = forecast["forecast_errors_by_method"]
    matured_errors = [
        float(errors[agent])
        for agent in ("trend", "reversal", "liquidity_confirmation")
        if errors.get(agent) is not None
    ]
    spread = statistics.median(spreads)
    error_scale = statistics.median(matured_errors) if matured_errors else 0.0
    disagreement = "split" if spread > error_scale else "aligned"

    nav = portfolio.nav_cny
    holdings = portfolio.holdings_cny
    risk_symbols = set(holdings) | set(
        forecast["expected_returns_by_method"]["market_plus_relative_fusion"]
    )
    volatility, betas, market_volatility = _risk_estimates(
        research,
        index,
        risk_symbols,
        risk_lookback_sessions,
        risk_minimum_observations,
    )
    risk_audit_unavailable = sorted(
        (set(holdings) - set(volatility)) | (set(holdings) - set(betas))
    )
    if risk_audit_unavailable:
        # Account risk is retained as an audit field, not a policy-state key.
        # Preserve the market-regime state while making missing estimates explicit.
        current_risk = None
        risk_state = "unavailable_missing_holding_risk"
    else:
        factor_exposure = math.fsum(
            value / nav * float(betas[symbol])
            for symbol, value in holdings.items()
        )
        idiosyncratic_variance = math.fsum(
            (value / nav) ** 2
            * max(
                float(volatility[symbol]) ** 2
                - float(betas[symbol]) ** 2 * market_volatility**2,
                0.0,
            )
            for symbol, value in holdings.items()
        )
        current_risk = math.sqrt(
            idiosyncratic_variance + (factor_exposure * market_volatility) ** 2
        )
        risk_state = (
            "near_budget"
            if current_risk >= risk_budget_annualized * 0.8
            else "within_budget"
        )
    cash_weight = portfolio.cash_cny / nav
    account_state = "cash_heavy" if cash_weight >= 0.5 else "invested"
    # Market regime is the only policy context. Other measurements remain in
    # the audit record and risk is scored directly in reward; keeping them out
    # of the key lets mature action values pool across accounts and risk states.
    state = market_state
    return {
        "state": state,
        "market_state": market_state,
        "market_return_20s": market_return,
        "expert_disagreement": disagreement,
        "median_calibrated_expert_spread": spread,
        "median_matured_expert_rmse": error_scale,
        "account_cash_weight": cash_weight,
        "account_holdings_count": len(holdings),
        "current_account_annualized_risk_estimate": current_risk,
        "risk_audit_unavailable_symbols": risk_audit_unavailable,
        "risk_budget_annualized": risk_budget_annualized,
        "risk_state": risk_state,
    }


def update_policy(
    training_feedback,
    trading_dates,
    as_of_date,
    q_values,
    action_counts,
    processed_feedback_ids=None,
):
    """Update mature account outcomes once, using feedback_id as an idempotency key."""
    trained = 0
    excluded_evaluation = 0
    ignored_unmatured = 0
    new_feedback_ids = 0
    duplicate_feedback_rows = 0
    processed_ids = {str(value) for value in (processed_feedback_ids or ())}
    seen_feedback_ids = set()
    date_index = {day: index for index, day in enumerate(trading_dates)}
    for row in training_feedback:
        raw_feedback_id = row.get("feedback_id")
        feedback_id = str(raw_feedback_id) if raw_feedback_id is not None and str(raw_feedback_id) else None
        if feedback_id is not None:
            if feedback_id in processed_ids or feedback_id in seen_feedback_ids:
                duplicate_feedback_rows += 1
                continue
            seen_feedback_ids.add(feedback_id)
        if row["data_split"] != "training":
            excluded_evaluation += 1
            if feedback_id is not None:
                processed_ids.add(feedback_id)
                new_feedback_ids += 1
            continue
        decision_date = date.fromisoformat(str(row["decision_date"]))
        available_date = date.fromisoformat(str(row["label_available_at"]))
        if available_date > as_of_date:
            ignored_unmatured += 1
            continue
        if date_index[available_date] - date_index[decision_date] < int(row["horizon_sessions"]):
            ignored_unmatured += 1
            continue
        state = str(row["state"])
        action = str(row["action"])
        if action not in ACTIONS:
            raise ValueError(f"Unknown contextual RL action: {action}")
        net_return = float(row["net_return_after_costs"])
        realized_risk = float(row["realized_annualized_risk"])
        risk_budget = float(row["risk_budget_annualized"])
        starting_nav = float(row["starting_nav_cny"])
        query_cost = float(row.get("supplementary_query_cost_cny", 0.0))
        role_qwen_cost = float(row.get("role_qwen_cost_cny", 0.0))
        analysis_cost = float(
            row.get("analysis_cost_cny", role_qwen_cost + query_cost)
        )
        if (
            starting_nav <= 0 or query_cost < 0 or role_qwen_cost < 0
            or analysis_cost < 0 or realized_risk < 0 or risk_budget <= 0
        ):
            raise ValueError("Invalid account, query-cost or risk value in feedback")
        reward = (
            net_return
            - 0.5 * max(realized_risk - risk_budget, 0.0) ** 2
            - analysis_cost / starting_nav
        )
        q_values.setdefault(state, {})
        action_counts.setdefault(state, {})
        old_count = action_counts[state].get(action, 0)
        old_value = q_values[state].get(action, 0.0)
        q_values[state][action] = old_value + (reward - old_value) / (old_count + 1)
        action_counts[state][action] = old_count + 1
        trained += 1
        if feedback_id is not None:
            processed_ids.add(feedback_id)
            new_feedback_ids += 1
    return {
        "matured_training_feedback_used": trained,
        "unmatured_training_feedback_ignored": ignored_unmatured,
        "evaluation_feedback_excluded_from_training": excluded_evaluation,
        "new_feedback_ids_processed": new_feedback_ids,
        "duplicate_feedback_rows_skipped": duplicate_feedback_rows,
        "processed_feedback_ids": sorted(processed_ids),
        "reward": "net_return_after_costs - 0.5*max(realized_annualized_risk-risk_budget,0)^2 - analysis_cost_cny/starting_nav_cny",
        "label_rule": "label_available_at <= as_of and at least horizon_sessions trading sessions after decision_date",
    }


def choose_action(
    state,
    q_values,
    action_counts,
    *,
    query_available,
    available_methods=None,
    minimum_samples=MINIMUM_ACTION_SAMPLES,
    exploration_rate=0.0,
    seed=0,
):
    """Choose from market-state estimates, falling back to shared mature action values."""
    if minimum_samples < 1 or not 0 <= exploration_rate <= 1:
        raise ValueError("Invalid contextual RL sample threshold or exploration rate")
    if available_methods is None:
        available_methods = {
            "market_plus_relative_fusion", "trend", "reversal", "liquidity_confirmation"
        }
    available_methods = set(available_methods)
    eligible = [
        action
        for action in ACTIONS
        if set(FUSION_PROFILES[action.split(":", 1)[0]]) <= available_methods
    ]
    pooled_counts = {
        action: sum(int(counts.get(action, 0)) for counts in action_counts.values())
        for action in ACTIONS
    }
    pooled_values = {
        action: (
            math.fsum(
                float(values.get(action, 0.0)) * int(action_counts.get(row_state, {}).get(action, 0))
                for row_state, values in q_values.items()
            ) / pooled_counts[action]
            if pooled_counts[action] else float("-inf")
        )
        for action in ACTIONS
    }
    local_counts = action_counts.get(state, {})
    baseline_count = (
        int(local_counts.get(BASELINE_ACTION, 0))
        if int(local_counts.get(BASELINE_ACTION, 0)) >= minimum_samples
        else pooled_counts[BASELINE_ACTION]
    )
    estimate_counts = {
        action: (
            int(local_counts.get(action, 0))
            if int(local_counts.get(action, 0)) >= minimum_samples
            else pooled_counts[action]
        )
        for action in ACTIONS
    }
    estimate_values = {
        action: (
            float(q_values.get(state, {}).get(action, float("-inf")))
            if int(local_counts.get(action, 0)) >= minimum_samples
            else pooled_values[action]
        )
        for action in ACTIONS
    }
    comparable = [
        action
        for action in eligible
        if estimate_counts[action] >= minimum_samples
        and (action == BASELINE_ACTION or baseline_count >= minimum_samples)
    ]
    selected = BASELINE_ACTION
    if comparable:
        selected = max(
            comparable,
            key=lambda action: estimate_values[action],
        )
    randomizer = random.Random(seed)
    explored = bool(eligible) and randomizer.random() < exploration_rate
    if explored:
        selected = randomizer.choice(eligible)
    profile, _ = selected.split(":", 1)
    return {
        "action": selected,
        "fusion_profile": profile,
        "weights": dict(FUSION_PROFILES[profile]),
        "supplementary_query": False,
        "state": state,
        "learned_action_count": estimate_counts[selected],
        "estimate_scope": (
            "state_specific"
            if int(local_counts.get(selected, 0)) >= minimum_samples
            else "shared_across_market_regimes"
        ),
        "minimum_samples": minimum_samples,
        "exploration_seed": seed,
        "policy_status": (
            "training_exploration"
            if explored
            else
            "learned_contextual_bandit"
            if selected != BASELINE_ACTION and comparable
            else "baseline_action_has_mature_samples"
            if baseline_count >= minimum_samples
            else "cold_start_fixed_p9"
        ),
        "explored": explored,
        "query_runtime_available": bool(query_available),
        "query_policy_status": "disabled_no_calibrated_numeric_effect",
    }


def apply_action(forecast, weights):
    """Blend calibrated return estimates; keep a conservative mature error allowance."""
    methods = forecast["expected_returns_by_method"]
    errors = forecast["forecast_errors_by_method"]
    missing = set(weights) - set(methods)
    if missing:
        raise ValueError(f"RL action references missing calibrated experts: {sorted(missing)}")
    symbols = set.intersection(*(set(methods[method]) for method in weights))
    if not symbols:
        raise ValueError("RL action has no common calibrated candidate symbols")
    combined = {
        symbol: math.fsum(
            weight * float(methods[method][symbol])
            for method, weight in weights.items()
        )
        for symbol in sorted(symbols)
    }
    component_errors = [
        float(errors[method])
        for method in weights
        if errors.get(method) is not None
    ]
    if not component_errors:
        raise ValueError("RL action has no matured out-of-sample forecast error")
    updated = dict(forecast)
    updated["expected_returns_by_method"] = {
        **methods,
        "rl_controlled_fusion": combined,
    }
    updated["forecast_errors_by_method"] = {
        **errors,
        "rl_controlled_fusion": max(component_errors),
    }
    if "target_weights_by_method" in forecast:
        updated["target_weights_by_method"] = {
            **forecast["target_weights_by_method"],
            "rl_controlled_fusion": forecast["target_weights_by_method"].get(
                "market_plus_relative_fusion", {}
            ),
        }
    return updated

def replay_contextual_feedback(
    decision_rows,
    daily_nav_rows,
    as_of_date,
    evaluation_start_date,
    risk_budget_annualized,
    q_values,
    action_counts,
    *,
    realized_risk_lookback_sessions=20,
    processed_feedback_ids=None,
):
    """Derive one-session account feedback from executed decisions and replay NAV, then update policy."""
    if realized_risk_lookback_sessions < 2:
        raise ValueError("realized_risk_lookback_sessions must be at least 2")
    nav_rows = sorted(
        daily_nav_rows,
        key=lambda row: str(row["trade_date"]),
    )
    nav_by_date = {}
    for row in nav_rows:
        day = date.fromisoformat(str(row["trade_date"]))
        nav = float(row["nav_cny"])
        daily_return = float(row["daily_return"])
        if not math.isfinite(nav) or nav <= 0 or not math.isfinite(daily_return):
            raise ValueError(f"Invalid daily account result for {day}")
        nav_by_date[day] = {**row, "nav_cny": nav, "daily_return": daily_return}

    feedback_rows = []
    for decision in decision_rows:
        decision_date_value = decision.get("decision_date")
        if decision_date_value is None:
            decision_date_value = decision["as_of_date"]
        decision_date = date.fromisoformat(str(decision_date_value))
        label_date = date.fromisoformat(str(decision["execution_date"]))
        if label_date > as_of_date or label_date not in nav_by_date:
            continue
        history = [row for row in nav_rows if str(row["trade_date"]) <= label_date.isoformat()]
        risk_returns = [float(row["daily_return"]) for row in history[-realized_risk_lookback_sessions:]]
        if len(risk_returns) < realized_risk_lookback_sessions:
            continue
        realized_risk = statistics.stdev(risk_returns) * math.sqrt(250)

        trace = decision.get("evidence", {}).get("collaborative_architecture", {})
        controller = trace.get("rl_controller", {})
        state = decision.get("state", controller.get("state"))
        action = decision.get("action", controller.get("action", decision.get("controller_action")))
        recorded_splits = {
            str(value)
            for value in (decision.get("data_split"), controller.get("data_split"))
            if value is not None
        }
        if not recorded_splits <= {"training", "evaluation"}:
            raise ValueError(f"Unknown data_split in decision {decision.get('run_id', decision_date)}")
        if evaluation_start_date is not None:
            boundary_split = (
                "evaluation" if decision_date >= evaluation_start_date else "training"
            )
            if boundary_split == "training" and "evaluation" in recorded_splits:
                raise ValueError(
                    f"Decision {decision.get('run_id', decision_date)} is marked evaluation before evaluation_start_date"
                )
            data_split = boundary_split
        else:
            if len(recorded_splits) > 1:
                raise ValueError(
                    f"Decision {decision.get('run_id', decision_date)} has conflicting data_split labels"
                )
            data_split = next(iter(recorded_splits), "training")
        state_parts = str(state or "").split(":")
        state_compatible = (
            len(state_parts) == 1
            and state_parts[0] in {"market_up", "market_down"}
        )
        action_compatible = action in ACTIONS
        if data_split == "training" and not state_compatible:
            raise ValueError(
                f"Training decision {decision.get('run_id', decision_date)} lacks a compatible market-regime state"
            )
        if data_split == "training" and not action_compatible:
            raise ValueError(
                f"Training decision {decision.get('run_id', decision_date)} action {action!r} is incompatible with contextual ACTIONS"
            )
        if action is None:
            continue
        starting_nav = float(decision.get("starting_nav_cny", decision.get("current_nav_cny")))
        if not math.isfinite(starting_nav) or starting_nav <= 0:
            raise ValueError("Decision current_nav_cny must be finite and positive")
        debate = trace.get("one_round_bull_bear_challenge", {})
        query_cost = float(
            decision.get("supplementary_query_cost_cny", debate.get("supplementary_query_cost_cny", 0.0))
        )
        role_reports = trace.get("agent_reports", {})
        role_qwen_cost = math.fsum(
            float(report.get("estimated_token_cost_cny") or 0.0)
            for report in role_reports.values()
        )
        cost_accounting = trace.get("decision_cost_accounting", {})
        role_qwen_cost = float(cost_accounting.get("role_qwen_cost_cny", role_qwen_cost))
        query_cost = float(cost_accounting.get("supplementary_query_cost_cny", query_cost))
        analysis_cost = float(
            cost_accounting.get("analysis_cost_cny", role_qwen_cost + query_cost)
        )
        source_hash = hashlib.sha256(
            json.dumps(decision, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
        ).hexdigest()
        run_id = str(decision.get("run_id", source_hash))
        feedback_rows.append({
            "run_id": run_id,
            "feedback_id": hashlib.sha256(
                f"{run_id}:{label_date.isoformat()}".encode("utf-8")
            ).hexdigest(),
            "source_decision_sha256": source_hash,
            "data_split": data_split,
            "decision_date": decision_date.isoformat(),
            "label_available_at": label_date.isoformat(),
            "horizon_sessions": 1,
            "state": str(state or ""),
            "action": str(action),
            "state_compatible": state_compatible,
            "action_compatible": action_compatible,
            "net_return_after_costs": float(nav_by_date[label_date]["daily_return"]),
            "realized_annualized_risk": realized_risk,
            "risk_budget_annualized": float(
                decision.get("risk_budget_annualized", decision.get("annualized_risk_budget", risk_budget_annualized))
            ),
            "starting_nav_cny": starting_nav,
            "role_qwen_cost_cny": role_qwen_cost,
            "supplementary_query_cost_cny": query_cost,
            "analysis_cost_cny": analysis_cost,
            "outcome_nav_cny": nav_by_date[label_date]["nav_cny"],
            "outcome_basis": "executed decision joined to account NAV after simulator or broker fills and fees",
        })

    trading_dates = sorted({
        date.fromisoformat(str(row.get("decision_date", row.get("as_of_date"))))
        for row in decision_rows
    } | {
        date.fromisoformat(str(row["execution_date"]))
        for row in decision_rows
    } | {
        date.fromisoformat(str(row["trade_date"]))
        for row in nav_rows
    })
    learning_update = update_policy(
        feedback_rows,
        trading_dates,
        as_of_date,
        q_values,
        action_counts,
        processed_feedback_ids=processed_feedback_ids,
    )
    return {
        "feedback_rows": feedback_rows,
        "learning_update": learning_update,
        "q_values": q_values,
        "action_counts": action_counts,
        "processed_feedback_ids": learning_update["processed_feedback_ids"],
        "training_feedback_rows": sum(row["data_split"] == "training" for row in feedback_rows),
        "compatible_decision_rows": sum(row["state_compatible"] and row["action_compatible"] for row in feedback_rows),
        "incompatible_state_rows": sum(not row["state_compatible"] for row in feedback_rows),
        "incompatible_action_rows": sum(not row["action_compatible"] for row in feedback_rows),
        "evaluation_feedback_rows_excluded": learning_update["evaluation_feedback_excluded_from_training"],
        "policy_updates": learning_update["matured_training_feedback_used"],
    }
