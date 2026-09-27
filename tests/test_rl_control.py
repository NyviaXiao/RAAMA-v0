import sys
from datetime import date, timedelta
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from run_p12_rl_control import _transitions

from adaptive_mas.evaluation.rl_control import ACTIONS, choose_actions


def test_frozen_holdout_action_does_not_update_policy_and_reward_is_next_executed_nav():
    dates = [date(2024, 1, 2) + timedelta(days=index) for index in range(24)]
    decision_dates = [dates[20].isoformat(), dates[21].isoformat()]
    states = {decision_dates[0]: "market_up", decision_dates[1]: "market_down"}
    q_values = {
        "market_up": {"trend": 0.2, "market_plus_relative_fusion": -0.1},
        "market_down": {"trend": -0.2, "market_plus_relative_fusion": 0.3},
    }
    frozen_policy = {state: dict(actions) for state, actions in q_values.items()}
    forecasts = [{"prediction_date": day} for day in decision_dates]

    acted, policy_rows = choose_actions(forecasts, states, q_values, epsilon=0.0, seed=1)

    assert [row["controller_action"] for row in acted] == ["trend", "market_plus_relative_fusion"]
    assert tuple(ACTIONS) == ("trend", "market_plus_relative_fusion")
    assert q_values == frozen_policy

    nav_rows = [
        {"trade_date": dates[21].isoformat(), "daily_return": -0.012},
        {"trade_date": dates[22].isoformat(), "daily_return": 0.007},
    ]
    transitions = _transitions(
        policy_rows,
        nav_rows,
        SimpleNamespace(trading_dates=dates),
        states,
    )

    assert transitions[0]["execution_date"] == dates[21].isoformat()
    assert transitions[0]["reward"] == -0.012
    assert transitions[1]["execution_date"] == dates[22].isoformat()
    assert transitions[1]["reward"] == 0.007
    assert transitions[1]["terminal"] is True
