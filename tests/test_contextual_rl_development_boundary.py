from copy import deepcopy
from datetime import date
from types import SimpleNamespace

import scripts.run_contextual_rl_development as development
from adaptive_mas.evaluation.contextual_rl import ACTIONS, BASELINE_ACTION


def test_evaluation_boundary_nav_does_not_update_frozen_policy(monkeypatch):
    boundary = date(2025, 9, 1)
    q_values = {
        state: {action: 0.01 for action in ACTIONS}
        for state in ("market_up", "market_down")
    }
    counts = {
        state: {action: 24 for action in ACTIONS}
        for state in ("market_up", "market_down")
    }
    before = deepcopy((q_values, counts))

    def unexpected_update(*args, **kwargs):
        raise AssertionError("evaluation-boundary NAV must not update Q")

    monkeypatch.setattr(development, "update_policy", unexpected_update)
    monkeypatch.setattr(
        development,
        "contextual_state",
        lambda *args, **kwargs: {
            "state": "market_up",
            "market_return_20s": 0.01,
            "expert_disagreement": "aligned",
            "account_cash_weight": 0.6,
            "current_account_annualized_risk_estimate": None,
            "risk_state": "unavailable_missing_holding_risk",
            "risk_audit_unavailable_symbols": ["600837"],
        },
    )
    monkeypatch.setattr(
        development,
        "choose_action",
        lambda *args, **kwargs: {
            "action": BASELINE_ACTION,
            "fusion_profile": "p9_fixed",
            "weights": {"market_plus_relative_fusion": 1.0},
            "learned_action_count": 24,
            "estimate_scope": "shared_across_market_regimes",
            "policy_status": "baseline_action_has_mature_samples",
            "explored": False,
        },
    )
    monkeypatch.setattr(development, "apply_action", lambda forecast, weights: dict(forecast))
    controller, _ = development._controller(
        research=SimpleNamespace(trading_dates=[]),
        risk_budget=0.20,
        risk_lookback=60,
        risk_minimum=40,
        evaluation_start=boundary,
        q_values=q_values,
        counts=counts,
        processed_ids=set(),
        progress=lambda message: None,
        training=True,
    )
    prior_decision = {
        "data_split": "training",
        "decision_date": "2025-08-29",
        "execution_date": boundary.isoformat(),
        "action": BASELINE_ACTION,
        "state": "market_up",
        "current_nav_cny": 1_000_000.0,
    }
    controller(
        forecast={
            "prediction_date": boundary.isoformat(),
            "expected_returns_by_method": {"market_plus_relative_fusion": {}},
        },
        account=object(),
        decision_date=boundary,
        daily_nav=[{"daily_return": 0.01} for _ in range(20)],
        decisions=[prior_decision],
    )

    assert (q_values, counts) == before
