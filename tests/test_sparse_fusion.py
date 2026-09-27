import datetime

import pytest

from adaptive_mas.agents.quantitative import AGENT_IDS
from adaptive_mas.evaluation.sparse_fusion import build_sparse_fusion_decisions


def _forecast(prediction_date, volatility=0.15):
    scores = {"000001": 1.0, "000002": 0.5, "000003": 0.0}
    return {
        "prediction_date": prediction_date,
        "scores_by_method": {expert: scores for expert in AGENT_IDS},
        "annualized_volatility_by_symbol": {symbol: volatility for symbol in scores},
        "factor_beta_by_symbol": {symbol: 1.0 for symbol in scores},
        "market_annualized_volatility": 0.20,
    }


def test_sparse_activation_uses_only_prior_realized_net_account_dates():
    forecasts = [
        _forecast("2026-01-01"),
        _forecast("2026-01-03"),
        _forecast("2026-01-04"),
    ]
    realized = {
        "trend": [
            {"trade_date": "2026-01-01", "daily_return": 0.01},
            {"trade_date": "2026-01-02", "daily_return": 0.02},
            {"trade_date": "2026-01-03", "daily_return": -0.04},
        ],
        "reversal": [
            {"trade_date": "2026-01-01", "daily_return": -0.01},
            {"trade_date": "2026-01-02", "daily_return": -0.02},
            {"trade_date": "2026-01-03", "daily_return": 0.04},
        ],
        "liquidity_confirmation": [],
    }
    decisions, activation = build_sparse_fusion_decisions(
        forecasts,
        realized,
        comparison_start_date=datetime.date(2026, 1, 1),
        portfolio_fraction=0.30,
        invested_weight=0.05,
        maximum_symbol_weight=0.05,
        maximum_one_way_turnover=0.50,
        risk_budget_annualized=0.20,
        matured_window=2,
        minimum_matured_decisions=2,
    )

    assert activation[0]["active_experts"] == []
    assert activation[1]["active_experts"] == ["trend"]
    assert activation[1]["matured_net_contribution_dates"]["trend"] == [
        "2026-01-01",
        "2026-01-02",
    ]
    assert all(
        trade_date < activation[1]["decision_date"]
        for dates in activation[1]["matured_net_contribution_dates"].values()
        for trade_date in dates
    )
    assert decisions[1]["target_weights"] == {"000001": 0.05}
    assert activation[2]["active_experts"] == ["reversal"]
    assert activation[2]["matured_net_contribution_dates"]["reversal"] == [
        "2026-01-02",
        "2026-01-03",
    ]


def test_stopped_expert_uses_today_risk_to_reconcile_holding_and_turnover():
    forecasts = [
        _forecast("2026-01-03", volatility=0.15),
        _forecast("2026-01-04", volatility=1.0),
    ]
    realized = {
        "trend": [
            {"trade_date": "2026-01-01", "daily_return": 0.02},
            {"trade_date": "2026-01-02", "daily_return": 0.02},
            {"trade_date": "2026-01-03", "daily_return": -0.03},
        ],
        "reversal": [
            {"trade_date": "2026-01-01", "daily_return": -0.01},
            {"trade_date": "2026-01-02", "daily_return": -0.01},
            {"trade_date": "2026-01-03", "daily_return": -0.01},
        ],
        "liquidity_confirmation": [
            {"trade_date": "2026-01-01", "daily_return": -0.01},
            {"trade_date": "2026-01-02", "daily_return": -0.01},
            {"trade_date": "2026-01-03", "daily_return": -0.01},
        ],
    }

    with pytest.raises(ValueError, match="风险预算与单向换手上限不可同时满足"):
        build_sparse_fusion_decisions(
            forecasts,
            realized,
            comparison_start_date=datetime.date(2026, 1, 3),
            portfolio_fraction=0.30,
            invested_weight=0.05,
            maximum_symbol_weight=0.05,
            maximum_one_way_turnover=0.01,
            risk_budget_annualized=0.0125,
            matured_window=2,
            minimum_matured_decisions=2,
        )
