"""A small Q-learning expert selector evaluated by the shared simulator."""

from __future__ import annotations

from datetime import date
import random


ACTIONS = ("trend", "market_plus_relative_fusion")
STATES = ("market_up", "market_down")


def market_state_by_date(research, decision_dates: list[str]) -> dict[str, str]:
    """Use only the decision close and the preceding 20-session index close."""
    index_by_date = {
        row["trade_date"]: float(row["close"])
        for row in research.index_bars
        if row.get("close") is not None
    }
    states = {}
    for day_text in decision_dates:
        day = date.fromisoformat(day_text)
        index = research.trading_dates.index(day)
        if index < 20:
            raise ValueError(f"{day} 没有完整 20 交易日状态窗口")
        earlier = research.trading_dates[index - 20]
        start_close = index_by_date[earlier]
        current_close = index_by_date[day]
        states[day_text] = "market_up" if current_close >= start_close else "market_down"
    return states


def choose_actions(
    forecasts: list[dict[str, object]],
    states: dict[str, str],
    q_values: dict[str, dict[str, float]],
    epsilon: float,
    seed: int,
) -> tuple[list[dict[str, object]], list[dict[str, str]]]:
    """Create a deterministic-epsilon-greedy replay from a frozen Q table."""
    if not 0 <= epsilon <= 1:
        raise ValueError("epsilon 必须处于 [0, 1]")
    randomizer = random.Random(seed)
    acted = []
    decisions = []
    for forecast in forecasts:
        day = str(forecast["prediction_date"])
        state = states[day]
        if randomizer.random() < epsilon:
            action = randomizer.choice(ACTIONS)
        else:
            action = max(ACTIONS, key=lambda candidate: q_values[state][candidate])
        row = {**forecast, "controller_action": action}
        acted.append(row)
        decisions.append({"decision_date": day, "state": state, "action": action})
    return acted, decisions


def update_q_values(
    q_values: dict[str, dict[str, float]],
    transitions: list[dict[str, object]],
    learning_rate: float = 0.1,
    discount: float = 0.9,
) -> None:
    """Apply one-step Q-learning updates to net daily portfolio rewards."""
    if not 0 < learning_rate <= 1 or not 0 <= discount < 1:
        raise ValueError("Q-learning 学习率或折扣系数无效")
    for row in transitions:
        state = str(row["state"])
        action = str(row["action"])
        next_state = str(row["next_state"])
        reward = float(row["reward"])
        current = q_values[state][action]
        future_value = 0.0 if row.get("terminal", False) else max(q_values[next_state].values())
        target = reward + discount * future_value
        q_values[state][action] = current + learning_rate * (target - current)
