from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

from adaptive_mas.backtest.simulator import simulate_portfolio
from adaptive_mas.data.research import MembershipSnapshot, ResearchData
from adaptive_mas.evaluation.public_expert_overlay import add_public_expert_overlay


SHANGHAI = ZoneInfo("Asia/Shanghai")


def _forecast():
    expected = {"000001": 0.001, "000002": 0.002}
    weights = {"000001": 0.5, "000002": 0.5}
    return {
        "prediction_date": "2024-02-05",
        "horizon_sessions": 3,
        "expected_returns_by_method": {
            "trend": dict(expected),
            "market_plus_relative_fusion": dict(expected),
        },
        "forecast_errors_by_method": {
            "trend": 0.0,
            "market_plus_relative_fusion": 0.0,
        },
        "target_weights_by_method": {
            "trend": dict(weights),
            "market_plus_relative_fusion": dict(weights),
        },
    }


def _snapshot(effect_status):
    signal = {
        "record_id": "same-day-evidence-1",
        "record_kind": "event",
        "expert": "announcement_title_rule_v1",
        "symbol": "sh.000001",
        "horizon_sessions": 3,
        "direction_score": 1,
    }
    return {
        "decision_time": "2024-02-06T18:00:00+08:00",
        "event_start_time": "2024-02-05T18:00:00+08:00",
        "p9_outputs": [signal],
        "p10_outputs": [],
        "llm_outputs": {"outputs": []},
        "p11_candidate_effects": {
            "announcement_title_rule_v1:3s": {
                "status": effect_status,
                "candidate_return_adjustment": 0.01,
            }
        },
    }


def _research():
    dates = [date(2024, 1, 2) + timedelta(days=index) for index in range(36)]
    adjusted = []
    execution = []
    for symbol, drift in (("000001", 0.001), ("000002", 0.0005)):
        previous = 10.0 if symbol == "000001" else 20.0
        for index, day in enumerate(dates):
            opening = previous * (1 + drift * 0.2)
            closing = previous * (1 + drift)
            adjusted.append({
                "symbol": symbol,
                "trade_date": day,
                "available_at": day,
                "open": opening,
                "high": max(opening, closing) * 1.001,
                "low": min(opening, closing) * 0.999,
                "close": closing,
                "return_1d": drift * 100,
            })
            execution.append({
                "symbol": symbol,
                "trade_date": day,
                "available_at": day,
                "open": opening,
                "high": max(opening, closing) * 1.001,
                "low": min(opening, closing) * 0.999,
                "close": closing,
                "amount": 10_000_000.0,
                "tradestatus": "1",
            })
            previous = closing
    index_bars = [
        {
            "symbol": "sh.000300",
            "trade_date": day,
            "available_at": day,
            "open": 1000.0 + index * 0.1,
            "close": 1000.0 + index * 0.2,
        }
        for index, day in enumerate(dates)
    ]
    membership = [MembershipSnapshot(dates[0], dates[0], frozenset({"000001", "000002"}), "fixture")]
    return ResearchData(dates, adjusted, execution, index_bars, membership)


def _replay(research, forecast):
    config = {
        "optimizer_return_method": "market_plus_relative_fusion",
        "optimizer_candidate_scope": "frozen_method_target",
        "risk_aversion": 0.0,
        "forecast_error_multiplier": 0.0,
        "risk_budget_annualized": 0.20,
        "maximum_symbol_weight": 1.0,
        "maximum_one_way_turnover": 1.0,
        "liquidity_lookback_sessions": 5,
        "maximum_daily_participation": 1.0,
        "risk_lookback_sessions": 20,
        "risk_minimum_observations": 10,
        "commission_per_side_bps": 0.0,
        "transfer_fee_per_side_bps": 0.0,
        "stamp_duty_sell_bps": 0.0,
        "slippage_per_side_bps": 0.0,
    }
    return simulate_portfolio(
        research,
        [],
        1_000_000.0,
        {
            "commission_per_side_bps": 0.0,
            "transfer_fee_per_side_bps": 0.0,
            "stamp_duty_sell_bps": 0.0,
            "slippage_per_side_bps": 0.0,
        },
        liquidity_lookback_sessions=5,
        maximum_daily_participation=1.0,
        maximum_one_way_turnover=1.0,
        forecast_decisions=[forecast],
        optimizer_config=config,
        evaluation_start_date=date.fromisoformat(forecast["prediction_date"]),
    )


def test_unmatured_effect_does_not_change_expected_returns_or_account():
    forecast = _forecast()
    overlaid, audit = add_public_expert_overlay(forecast, _snapshot("pending_or_no_positive_lower_bound"))

    assert overlaid["expected_returns_by_method"] == forecast["expected_returns_by_method"]
    assert overlaid["target_weights_by_method"] == forecast["target_weights_by_method"]
    assert audit["status"] == "no_eligible_matured_candidate_effects"
    baseline_replay = _replay(_research(), forecast)
    pending_replay = _replay(_research(), overlaid)
    assert baseline_replay["decisions"][0]["current_nav_cny"] == pending_replay["decisions"][0]["current_nav_cny"]
    assert baseline_replay["decisions"][0]["target_weights"] == pending_replay["decisions"][0]["target_weights"]


def test_matured_effect_changes_only_same_day_candidate_and_replays_same_account():
    forecast = _forecast()
    overlaid, audit = add_public_expert_overlay(forecast, _snapshot("candidate_effect_eligible"))

    assert overlaid["expected_returns_by_method"]["market_plus_relative_fusion"] == forecast["expected_returns_by_method"]["market_plus_relative_fusion"]
    assert overlaid["expected_returns_by_method"]["fundamental_p11"]["000001"] == 0.011
    assert overlaid["expected_returns_by_method"]["fundamental_p11"]["000002"] == 0.002
    assert overlaid["expected_returns_by_method"]["trend"] == forecast["expected_returns_by_method"]["trend"]
    assert overlaid["target_weights_by_method"] == forecast["target_weights_by_method"]
    assert audit["adjustments"]["000001"]["components"][0]["record_ids"] == ["same-day-evidence-1"]

    research = _research()
    baseline_replay = _replay(research, forecast)
    overlay_replay = _replay(research, overlaid)
    assert baseline_replay["decisions"][0]["current_nav_cny"] == overlay_replay["decisions"][0]["current_nav_cny"]
    assert baseline_replay["decisions"][0]["decision_date"] == overlay_replay["decisions"][0]["decision_date"]


def test_weekend_carry_signal_is_available_to_next_trading_overlay_once():
    snapshot = _snapshot("candidate_effect_eligible")
    snapshot["p9_outputs"] = []
    snapshot["llm_outputs"] = {"outputs": []}
    snapshot["carry_forward_expert_signals"] = [{
        "record_id": "qwen:weekend:carry:2024-02-06",
        "record_kind": "event",
        "expert": "cls_news_rule_v1",
        "symbol": "sh.000001",
        "horizon_sessions": 3,
        "direction_score": 1,
    }]
    snapshot["p11_candidate_effects"] = {
        "cls_news_rule_v1:3s": {
            "status": "candidate_effect_eligible",
            "candidate_return_adjustment": 0.01,
        }
    }

    overlaid, audit = add_public_expert_overlay(_forecast(), snapshot)

    assert overlaid["expected_returns_by_method"]["market_plus_relative_fusion"]["000001"] == 0.001
    assert overlaid["expected_returns_by_method"]["news_p11"]["000001"] == 0.011
    assert audit["adjustments"]["000001"]["components"][0]["record_ids"] == [
        "qwen:weekend:carry:2024-02-06"
    ]


def test_q_learning_action_uses_the_same_simulator_and_is_audited():
    forecast = {**_forecast(), "controller_action": "trend"}

    replay = _replay(_research(), forecast)

    assert replay["decisions"][0]["controller_action"] == "trend"
    assert replay["decisions"][0]["selected_return_method"] == "trend"
    assert replay["decisions"][0]["current_nav_cny"] == 1_000_000.0
