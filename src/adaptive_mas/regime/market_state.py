"""Binary market state and explicitly reported reliability fallback."""

from datetime import date

from adaptive_mas.reliability.gating import reliability_weights


def market_state(research, decision_date: date, lookback_sessions: int) -> str:
    index = research.trading_dates.index(decision_date)
    if index < lookback_sessions:
        return "warming_up"
    closes = {
        row["trade_date"]: row["close"]
        for row in research.index_bars
        if row["available_at"] <= decision_date
    }
    earlier = research.trading_dates[index - lookback_sessions]
    start = closes[earlier]
    end = closes[decision_date]
    return "risk_on" if end >= start else "risk_off"


def conditional_weights(
    matured_by_state: dict[str, list[dict[str, float]]],
    state: str,
    matured_global: list[dict[str, float]],
    agent_ids: tuple[str, ...],
    lookback_windows: int,
    minimum_windows: int,
    equal_weight_share: float,
) -> tuple[dict[str, float], str, int]:
    state_history = matured_by_state.get(state, [])
    if len(state_history) < minimum_windows:
        weights = reliability_weights(
            matured_global,
            agent_ids,
            lookback_windows,
            minimum_windows,
            equal_weight_share,
        )
        return weights, "global_fallback", len(state_history)
    weights = reliability_weights(
        state_history,
        agent_ids,
        lookback_windows,
        minimum_windows,
        equal_weight_share,
    )
    return weights, "state_conditional", len(state_history)
