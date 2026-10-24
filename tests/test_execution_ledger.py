import hashlib
from datetime import date

import pytest

from adaptive_mas.execution_ledger import simulate_execution_ledger


DAYS = [date(2026, 1, day) for day in (5, 6, 7, 8)]


def _rules(*, participation=1.0, t_plus_one=True, price_limit=0.1):
    return {
        "ruleset_id": "fixture-cn-equity-2026-01-v1",
        "effective_date": "2026-01-01",
        "source_title": "Fixture market rule pack for tests",
        "source_url": "https://example.invalid/fixture-rules",
        "symbols": {
            "600000": {
                "board": "fixture-main-board",
                "order_unit": 100,
                "minimum_buy_quantity": 100,
                "odd_lot_sell": True,
                "t_plus_one": t_plus_one,
                "tick_size": 0.01,
                "price_limit_pct": price_limit,
                "maximum_daily_participation": participation,
                "liquidity_lookback_sessions": 1,
            }
        },
        "fees": {
            "commission_per_side_bps": 5,
            "minimum_commission_cny": 5,
            "transfer_fee_per_side_bps": 0.1,
            "stamp_duty_sell_bps": 5,
            "slippage_per_side_bps": 5,
        },
    }


def _bar(day, *, open_price=10, close_price=10, preclose=10, amount=1_000_000, status="1", high=None, low=None):
    return {
        "symbol": "600000",
        "trade_date": day,
        "tradestatus": status,
        "open": open_price,
        "high": open_price if high is None else high,
        "low": open_price if low is None else low,
        "close": close_price,
        "preclose": preclose,
        "amount": amount,
        "volume": 100_000,
    }


def _prediction(value=0.1):
    return {
        "prediction_id": "pred-1",
        "prediction_date": DAYS[0].isoformat(),
        "label_maturity_date": DAYS[3].isoformat(),
        "horizon_sessions": 3,
        "expected_return": value,
        "label_status": "pending",
        "realized_return": None,
    }


def test_quantity_mode_enforces_buy_lot_t_plus_one_fees_and_reconciles():
    result = simulate_execution_ledger(
        execution_mode="quantity",
        trading_dates=DAYS[:3],
        market_bars=[_bar(day) for day in DAYS[:3]],
        orders=[
            {"trade_date": DAYS[1], "symbol": "600000", "side": "buy", "quantity": 150, "source_prediction_id": "pred-1"},
            {"trade_date": DAYS[1], "symbol": "600000", "side": "sell", "quantity": 100, "source_prediction_id": "pred-1"},
            {"trade_date": DAYS[2], "symbol": "600000", "side": "sell", "quantity": 100, "source_prediction_id": "pred-1"},
        ],
        rule_pack=_rules(),
        initial_cash_cny=10_000,
        prediction_records=[_prediction()],
    )
    buy, same_day_sell, next_day_sell = result["transactions"]
    assert buy["filled_quantity"] == 100
    assert buy["status"] == "partial_or_limited"
    assert same_day_sell["status"] == "blocked"
    assert same_day_sell["reason"] == "no_sellable_or_lot_eligible_quantity"
    assert next_day_sell["filled_quantity"] == 100
    assert buy["fees_cny"] >= 5
    assert result["summary"]["all_cash_reconciliations_exact"] is True
    assert result["summary"]["all_nav_reconciliations_exact"] is True
    assert result["summary"]["all_quantity_reconciliations_exact"] is True
    assert result["broker_orders_enabled"] is False


def test_research_cost_is_debited_once_into_cash_nav_and_reconciliation():
    result = simulate_execution_ledger(
        execution_mode="nominal_research",
        trading_dates=DAYS[:3],
        market_bars=[_bar(day) for day in DAYS[:3]],
        orders=[],
        rule_pack=_rules(),
        initial_cash_cny=10_000,
        research_cost_cny_by_date={DAYS[1].isoformat(): 12.34},
    )
    assert result["daily_nav"][0]["nav_cny"] == 10_000
    assert result["daily_nav"][1]["nav_cny"] == 9_987.66
    assert result["reconciliation"][1]["research_cost_cny"] == 12.34
    assert result["reconciliation"][1]["cash_expected_close_cny"] == 9_987.66
    assert result["reconciliation"][1]["cash_residual_cny"] == 0
    assert result["summary"]["total_research_cost_cny"] == 12.34
    assert result["summary"]["all_cash_reconciliations_exact"] is True
    assert result["summary"]["all_nav_reconciliations_exact"] is True


def test_research_cost_must_fit_the_calendar_and_available_cash():
    kwargs = {
        "execution_mode": "nominal_research",
        "trading_dates": DAYS[:2],
        "market_bars": [_bar(day) for day in DAYS[:2]],
        "orders": [],
        "rule_pack": _rules(),
        "initial_cash_cny": 10,
    }
    with pytest.raises(ValueError, match="research cost date must be on the execution calendar"):
        simulate_execution_ledger(**kwargs, research_cost_cny_by_date={DAYS[2].isoformat(): 1})
    with pytest.raises(ValueError, match="research cost exceeds available cash"):
        simulate_execution_ledger(**kwargs, research_cost_cny_by_date={DAYS[1].isoformat(): 11})


def test_quantity_mode_blocks_suspension_locked_limit_and_capacity():
    suspended = simulate_execution_ledger(
        execution_mode="quantity", trading_dates=DAYS[:2],
        market_bars=[_bar(DAYS[0]), _bar(DAYS[1], status="0")],
        orders=[{"trade_date": DAYS[1], "symbol": "600000", "side": "buy", "quantity": 100, "source_prediction_id": "pred-1"}],
        rule_pack=_rules(), initial_cash_cny=10_000,
    )
    assert suspended["transactions"][0]["reason"] == "suspended_or_missing_bar"

    locked = simulate_execution_ledger(
        execution_mode="quantity", trading_dates=DAYS[:2],
        market_bars=[_bar(DAYS[0]), _bar(DAYS[1], open_price=11, close_price=11, preclose=10, high=11, low=11)],
        orders=[{"trade_date": DAYS[1], "symbol": "600000", "side": "buy", "quantity": 100, "source_prediction_id": "pred-1"}],
        rule_pack=_rules(), initial_cash_cny=10_000,
    )
    assert locked["transactions"][0]["reason"] == "locked_at_upper_limit"

    capacity = simulate_execution_ledger(
        execution_mode="quantity", trading_dates=DAYS[:2],
        market_bars=[_bar(DAYS[0], amount=1_000), _bar(DAYS[1])],
        orders=[{"trade_date": DAYS[1], "symbol": "600000", "side": "buy", "quantity": 100, "source_prediction_id": "pred-1"}],
        rule_pack=_rules(participation=0.01), initial_cash_cny=10_000,
    )
    assert capacity["transactions"][0]["status"] == "blocked"
    assert capacity["transactions"][0]["reason"] == "capacity_or_risk_limit"


def test_nominal_mode_preserves_notional_sizing_and_prediction_nav_separation():
    orders = [{
        "trade_date": DAYS[1], "symbol": "600000", "side": "buy",
        "notional_cny": 1000, "source_prediction_id": "pred-1",
    }]
    first = simulate_execution_ledger(
        execution_mode="nominal_research", trading_dates=DAYS[:3],
        market_bars=[_bar(day, close_price=10 + index) for index, day in enumerate(DAYS[:3])],
        orders=orders, rule_pack=_rules(), initial_cash_cny=10_000,
        prediction_records=[_prediction(0.1)],
    )
    changed_prediction = simulate_execution_ledger(
        execution_mode="nominal_research", trading_dates=DAYS[:3],
        market_bars=[_bar(day, close_price=10 + index) for index, day in enumerate(DAYS[:3])],
        orders=orders, rule_pack=_rules(), initial_cash_cny=10_000,
        prediction_records=[_prediction(9.0)],
    )
    assert first["transactions"][0]["filled_notional_cny"] > 990
    assert [row["nav_cny"] for row in first["daily_nav"]] == [row["nav_cny"] for row in changed_prediction["daily_nav"]]
    assert first["prediction_ledger"][0]["expected_return"] != changed_prediction["prediction_ledger"][0]["expected_return"]
    assert first["prediction_ledger"][0]["ledger"] == "prediction_only_no_NAV_effect"
    assert first["summary"]["all_cash_reconciliations_exact"] is True


def test_corporate_action_dividend_and_share_distribution_reconcile_exactly():
    source_bytes = b"versioned corporate action source fixture"
    source_hash = hashlib.sha256(source_bytes).hexdigest()
    actions = [
        {
            "type": "cash_dividend", "symbol": "600000", "effective_date": DAYS[1],
            "cash_per_share_cny": 1.0, "payment_date": DAYS[2],
            "source_ref": "corp-action-fixture", "source_sha256": source_hash,
        },
        {
            "type": "stock_distribution", "symbol": "600000", "effective_date": DAYS[1],
            "new_shares_per_share": 0.1, "available_for_sale_date": DAYS[2],
            "source_ref": "corp-action-fixture", "source_sha256": source_hash,
        },
    ]
    bars = [
        _bar(DAYS[0], open_price=100, close_price=100, preclose=100),
        _bar(DAYS[1], open_price=90, close_price=90, preclose=100),
        _bar(DAYS[2], open_price=90, close_price=90, preclose=90),
    ]
    result = simulate_execution_ledger(
        execution_mode="quantity",
        trading_dates=DAYS[:3], market_bars=bars, orders=[], rule_pack=_rules(), initial_cash_cny=0,
        initial_lots={"600000": [{"quantity": 100, "acquired_date": "2025-12-31", "unit_cost_cny": 100}]},
        corporate_actions=actions,
        source_artifact_hashes={"corp-action-fixture": source_hash},
    )
    assert result["daily_nav"][0]["nav_cny"] == 10_000
    assert result["daily_nav"][1]["nav_cny"] == 10_000
    assert result["daily_nav"][2]["nav_cny"] == 10_000
    assert result["daily_nav"][1]["position_quantities"]["600000"] == 110
    assert result["daily_nav"][1]["position_sellable_quantities"]["600000"] == 100
    assert result["daily_nav"][2]["position_sellable_quantities"]["600000"] == 110
    assert result["summary"]["all_cash_reconciliations_exact"] is True
    assert result["summary"]["all_nav_reconciliations_exact"] is True
    assert result["summary"]["all_quantity_reconciliations_exact"] is True


def test_versioned_rules_require_per_symbol_source_and_reject_missing_rules():
    rules = _rules()
    rules["symbols"]["600000"].pop("order_unit")
    with pytest.raises(ValueError, match="market rules for 600000 are missing"):
        simulate_execution_ledger(
            execution_mode="quantity", trading_dates=DAYS[:1], market_bars=[_bar(DAYS[0])],
            orders=[], rule_pack=rules, initial_cash_cny=1000,
        )
