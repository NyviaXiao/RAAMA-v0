"""Matured public-signal effects applied as an auditable P9 candidate overlay."""

from __future__ import annotations

from datetime import datetime
import statistics
from zoneinfo import ZoneInfo


SHANGHAI = ZoneInfo("Asia/Shanghai")


def signals_for_decision(snapshot: dict[str, object]) -> list[dict[str, object]]:
    """Select this snapshot's new, directional company evidence only."""
    event_start = datetime.fromisoformat(str(snapshot["event_start_time"]))
    rows = (
        list(snapshot["p9_outputs"])
        + list(snapshot["p10_outputs"])
        + list(snapshot["llm_outputs"]["outputs"])
        + list(snapshot.get("carry_forward_expert_signals", []))
    )
    signals = []
    for row in rows:
        if row.get("symbol") is None or int(row["direction_score"]) == 0:
            continue
        if row.get("record_kind") == "state":
            available_at = datetime.fromisoformat(str(row["available_at"]))
            if available_at <= event_start:
                continue
        signals.append(row)
    return signals


def add_public_expert_overlay(
    forecast: dict[str, object],
    snapshot: dict[str, object] | None,
) -> tuple[dict[str, object], dict[str, object]]:
    if snapshot is None:
        return forecast, {
            "decision_date": str(forecast["prediction_date"]),
            "status": "no_expert_snapshot",
            "adjustments": {},
        }
    signals = signals_for_decision(snapshot)
    updated = adjust_market_relative_forecast(
        forecast, signals, snapshot.get("p11_candidate_effects", {})
    )
    return updated, {
        "decision_date": str(forecast["prediction_date"]),
        "snapshot_decision_time": str(snapshot["decision_time"]),
        "candidate_effects": snapshot.get("p11_candidate_effects", {}),
        "signal_count": len(signals),
        "status": updated["public_expert_adjustment_status"],
        "adjustments": updated["public_expert_adjustments"],
        "numeric_components": updated["public_expert_numeric_component_status"],
    }


def adjust_market_relative_forecast(
    forecast: dict[str, object],
    signals: list[dict[str, object]],
    candidate_effects: dict[str, dict[str, object]],
) -> dict[str, object]:
    """Expose only mature fundamental/news effects as independent RL return components."""
    base_method = "market_plus_relative_fusion"
    role_methods = {"fundamental": "fundamental_p11", "news": "news_p11"}
    expert_roles = {
        "fundamental": "fundamental",
        "fundamental_rule_v1": "fundamental",
        "announcement_title_rule_v1": "fundamental",
        "cninfo_full_text_llm_v1": "fundamental",
        "news": "news",
        "cls_news_rule_v1": "news",
        "news_macro_context": "news",
    }
    eligible = {
        str(symbol)
        for symbol, weight in forecast["target_weights_by_method"][base_method].items()
        if float(weight) > 0
    }
    by_symbol_and_expert: dict[tuple[str, str], list[dict[str, object]]] = {}
    for signal in signals:
        expert = str(signal["expert"])
        if expert not in expert_roles:
            continue
        symbol = str(signal["symbol"]).rsplit(".", 1)[-1]
        effect = candidate_effects.get(f"{expert}:{int(signal['horizon_sessions'])}s")
        if symbol not in eligible or effect is None or effect["status"] != "candidate_effect_eligible":
            continue
        by_symbol_and_expert.setdefault((symbol, expert), []).append(signal)

    updated = {
        **forecast,
        "expected_returns_by_method": {
            name: dict(values)
            for name, values in forecast["expected_returns_by_method"].items()
        },
        "forecast_errors_by_method": dict(forecast["forecast_errors_by_method"]),
    }
    base_returns = updated["expected_returns_by_method"][base_method]
    role_adjustments: dict[str, dict[str, list[float]]] = {role: {} for role in role_methods}
    by_symbol: dict[str, list[tuple[str, float, list[str]]]] = {}
    for (symbol, expert), rows in sorted(by_symbol_and_expert.items()):
        effect = candidate_effects[f"{expert}:{int(rows[0]['horizon_sessions'])}s"]
        alpha = float(effect["candidate_return_adjustment"])
        direction = statistics.fmean(int(row["direction_score"]) for row in rows)
        adjustment = direction * alpha
        role = expert_roles[expert]
        role_adjustments[role].setdefault(symbol, []).append(adjustment)
        by_symbol.setdefault(symbol, []).append((expert, adjustment, [str(row["record_id"]) for row in rows]))

    for role, return_method in role_methods.items():
        if not role_adjustments[role]:
            continue
        component_returns = dict(base_returns)
        for symbol, values in role_adjustments[role].items():
            component_returns[symbol] = float(component_returns[symbol]) + statistics.fmean(values)
        updated["expected_returns_by_method"][return_method] = component_returns
        updated["forecast_errors_by_method"][return_method] = updated["forecast_errors_by_method"].get(base_method)

    adjustments: dict[str, dict[str, object]] = {}
    for symbol, components in sorted(by_symbol.items()):
        adjustments[symbol] = {
            "return_adjustment": statistics.fmean(value for _, value, _ in components),
            "components": [
                {"expert": expert, "return_adjustment": value, "record_ids": record_ids}
                for expert, value, record_ids in components
            ],
        }
    updated["public_expert_adjustments"] = adjustments
    updated["public_expert_adjustment_status"] = (
        "applied_matured_candidate_effects" if adjustments else "no_eligible_matured_candidate_effects"
    )
    updated["public_expert_numeric_component_status"] = {
        "fundamental": "matured_p11_component_available" if role_adjustments["fundamental"] else "unavailable_no_eligible_matured_effect",
        "news": "matured_p11_component_available" if role_adjustments["news"] else "unavailable_no_eligible_matured_effect",
        "market_attention": "excluded_attention_has_no_directional_return_component",
        "qwen": "evidence_only_no_model_direction_return_score",
    }
    return updated
