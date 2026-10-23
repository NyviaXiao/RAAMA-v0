import pytest

from adaptive_mas.decision import _optimizer_candidate_symbols, _optimizer_expected_returns
from adaptive_mas.portfolio.optimizer import optimize_portfolio


def _forecast() -> dict[str, object]:
    return {
        "expected_returns_by_method": {
            "trend": {"000001": 0.01, "000002": 0.02, "000003": 0.03},
            "loss_gated_fusion": {"000001": 0.04, "000002": 0.05, "000003": 0.06},
        },
        "target_weights_by_method": {
            "trend": {"000001": 0.05, "000002": 0.0},
            "loss_gated_fusion": {"000002": 0.05, "000003": 0.05},
        },
    }


def test_default_optimizer_scope_keeps_all_original_method_candidates() -> None:
    actual = _optimizer_expected_returns(_forecast(), "trend", {})
    assert actual == {"000001": 0.01, "000002": 0.02, "000003": 0.03}


def test_frozen_method_scope_uses_only_that_methods_nonzero_target_basket() -> None:
    actual = _optimizer_expected_returns(
        _forecast(),
        "trend",
        {"optimizer_candidate_scope": "frozen_method_target"},
    )
    assert actual == {"000001": 0.01}


def test_frozen_method_scope_keeps_real_return_forecasts_for_existing_holdings() -> None:
    actual = _optimizer_expected_returns(
        _forecast(),
        "trend",
        {"optimizer_candidate_scope": "frozen_method_target"},
        {"000003"},
    )
    assert actual == {"000001": 0.01, "000003": 0.03}


def test_held_stock_leaving_forecast_universe_has_zero_alpha_until_sold() -> None:
    actual = _optimizer_expected_returns(
        _forecast(),
        "trend",
        {"optimizer_candidate_scope": "frozen_method_target"},
        {"600000"},
    )
    assert actual == {"000001": 0.01, "600000": 0.0}


def test_harness_priority_filters_optimizer_candidates_but_keeps_calibrated_returns() -> None:
    actual = _optimizer_expected_returns(
        _forecast(),
        "loss_gated_fusion",
        {
            "optimizer_candidate_scope": "harness_priority",
            "optimizer_priority_symbols": ["000003"],
        },
        {"000002"},
    )

    assert actual == {"000002": 0.05, "000003": 0.06}
    assert _optimizer_candidate_symbols(
        _forecast(),
        "loss_gated_fusion",
        {
            "optimizer_candidate_scope": "harness_priority",
            "optimizer_priority_symbols": ["000003"],
        },
    ) == {"000003"}


def test_empty_harness_priority_set_allows_cash_without_inventing_returns() -> None:
    config = {
        "optimizer_candidate_scope": "harness_priority",
        "optimizer_priority_symbols": [],
    }

    assert _optimizer_expected_returns(_forecast(), "loss_gated_fusion", config) == {
        "000001": 0.04,
        "000002": 0.05,
        "000003": 0.06,
    }
    assert _optimizer_candidate_symbols(_forecast(), "loss_gated_fusion", config) == set()


def test_frozen_method_scope_rejects_missing_candidate_returns_or_targets() -> None:
    forecast = _forecast()
    forecast["expected_returns_by_method"]["trend"].pop("000001")
    with pytest.raises(ValueError, match="候选篮子缺少预期收益"):
        _optimizer_expected_returns(
            forecast,
            "trend",
            {"optimizer_candidate_scope": "frozen_method_target"},
        )
    forecast = _forecast()
    forecast["target_weights_by_method"]["trend"] = {}
    with pytest.raises(ValueError, match="没有冻结候选篮子"):
        _optimizer_expected_returns(
            forecast,
            "trend",
            {"optimizer_candidate_scope": "frozen_method_target"},
        )


def test_outside_basket_holding_can_be_sold_but_cannot_be_increased() -> None:
    result = optimize_portfolio(
        {"000001": 0.02, "000002": -0.05},
        {"000001": 0.0, "000002": 0.0},
        {"000001": 0.20, "000002": 0.20},
        {"000002": 0.02},
        horizon_sessions=3,
        maximum_symbol_weight=0.05,
        maximum_one_way_turnover=0.50,
        risk_budget_annualized=0.20,
        risk_aversion=0.0,
        forecast_error_multiplier=0.0,
        buy_cost_rate=0.0,
        sell_cost_rate=0.0,
        max_change_weights={"000001": 0.50, "000002": 0.50},
        maximum_target_weights={"000001": 0.05, "000002": 0.02},
        iterations=100,
    )
    assert result.target_weights["000001"] > 0
    assert result.target_weights.get("000002", 0.0) <= 0.02 + 1e-10
    assert result.target_weights.get("000002", 0.0) < 0.02
