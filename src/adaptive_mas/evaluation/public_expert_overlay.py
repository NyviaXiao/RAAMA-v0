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
    }


def adjust_market_relative_forecast(
    forecast: dict[str, object],
    signals: list[dict[str, object]],
    candidate_effects: dict[str, dict[str, object]],
) -> dict[str, object]:
    """Add only positive-lower-bound, matured effects to frozen P9 candidates."""
    method = "market_plus_relative_fusion"
    eligible = {
        str(symbol)
        for symbol, weight in forecast["target_weights_by_method"][method].items()
        if float(weight) > 0
    }
    by_symbol_and_expert: dict[tuple[str, str], list[dict[str, object]]] = {}
    for signal in signals:
        symbol = str(signal["symbol"]).rsplit(".", 1)[-1]
        expert = str(signal["expert"])
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
    }
    adjustments: dict[str, dict[str, object]] = {}
    by_symbol: dict[str, list[tuple[str, float, list[str]]]] = {}
    for (symbol, expert), rows in sorted(by_symbol_and_expert.items()):
        effect = candidate_effects[f"{expert}:{int(rows[0]['horizon_sessions'])}s"]
        alpha = float(effect["candidate_return_adjustment"])
        direction = statistics.fmean(int(row["direction_score"]) for row in rows)
        by_symbol.setdefault(symbol, []).append((expert, direction * alpha, [str(row["record_id"]) for row in rows]))
    for symbol, components in sorted(by_symbol.items()):
        adjustment = statistics.fmean(value for _, value, _ in components)
        updated["expected_returns_by_method"][method][symbol] = (
            float(updated["expected_returns_by_method"][method][symbol]) + adjustment
        )
        adjustments[symbol] = {
            "return_adjustment": adjustment,
            "components": [
                {"expert": expert, "return_adjustment": value, "record_ids": record_ids}
                for expert, value, record_ids in components
            ],
        }
    updated["public_expert_adjustments"] = adjustments
    updated["public_expert_adjustment_status"] = (
        "applied_matured_candidate_effects" if adjustments else "no_eligible_matured_candidate_effects"
    )
    return updated
