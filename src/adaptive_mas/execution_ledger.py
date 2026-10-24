"""Offline financial execution ledger with explicit market rules and exact reconciliation."""

from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from datetime import date, datetime, time, timedelta, timezone
from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR, ROUND_HALF_UP
from typing import Mapping, Sequence

from adaptive_mas.research_provenance import stable_hash


_CENT = Decimal("0.01")
_ZERO = Decimal("0")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_CHINA_TZ = timezone(timedelta(hours=8))


def _decimal(value: object, label: str, *, minimum: Decimal | None = None) -> Decimal:
    try:
        result = Decimal(str(value))
    except Exception as exc:
        raise ValueError(f"{label} must be a finite number") from exc
    if not result.is_finite() or (minimum is not None and result < minimum):
        raise ValueError(f"{label} must be finite and at least {minimum}")
    return result


def _money(value: Decimal) -> Decimal:
    return value.quantize(_CENT, rounding=ROUND_HALF_UP)


def _day(value: object, label: str) -> date:
    try:
        return value if type(value) is date else date.fromisoformat(str(value))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a trading-date ISO string") from exc


def _out(value: Decimal | None) -> float | None:
    return float(value) if value is not None else None


def _stable_tick(price: Decimal, tick: Decimal, rounding: str) -> Decimal:
    return (price / tick).quantize(Decimal("1"), rounding=rounding) * tick


def _verify_rule_pack(rule_pack: Mapping[str, object]) -> tuple[dict[str, object], str]:
    required = {"ruleset_id", "effective_date", "source_title", "source_url", "symbols", "fees"}
    if not required.issubset(rule_pack):
        raise ValueError(f"market rule pack is missing {sorted(required - set(rule_pack))}")
    if not str(rule_pack["ruleset_id"]).strip() or not str(rule_pack["source_title"]).strip():
        raise ValueError("market rule pack needs a versioned ID and source title")
    if not str(rule_pack["source_url"]).startswith("https://"):
        raise ValueError("market rule pack source_url must use HTTPS")
    effective_date = _day(rule_pack["effective_date"], "rule effective_date")
    symbols = rule_pack["symbols"]
    fees = rule_pack["fees"]
    if not isinstance(symbols, Mapping) or not symbols:
        raise ValueError("market rule pack needs per-symbol execution rules")
    if not isinstance(fees, Mapping):
        raise ValueError("market fee rules must be an object")
    normalized_symbols = {}
    for symbol, raw in symbols.items():
        if not re.fullmatch(r"[0-9]{6}", str(symbol)) or not isinstance(raw, Mapping):
            raise ValueError("per-symbol market rules need six-digit symbols and object values")
        required_symbol = {"board", "order_unit", "minimum_buy_quantity", "odd_lot_sell", "t_plus_one", "tick_size", "price_limit_pct", "maximum_daily_participation", "liquidity_lookback_sessions"}
        if not required_symbol.issubset(raw):
            raise ValueError(f"market rules for {symbol} are missing {sorted(required_symbol - set(raw))}")
        order_unit = int(raw["order_unit"])
        min_buy = int(raw["minimum_buy_quantity"])
        lookback = int(raw["liquidity_lookback_sessions"])
        participation = _decimal(raw["maximum_daily_participation"], f"{symbol} daily participation", minimum=_ZERO)
        tick = _decimal(raw["tick_size"], f"{symbol} tick_size", minimum=Decimal("0.0000001"))
        price_limit = raw["price_limit_pct"]
        if price_limit is not None:
            price_limit = _decimal(price_limit, f"{symbol} price_limit_pct", minimum=_ZERO)
        if order_unit < 1 or min_buy < 1 or lookback < 1 or participation > 1:
            raise ValueError(f"market rules for {symbol} contain invalid order, capacity, or lookback values")
        normalized_symbols[str(symbol)] = {
            **dict(raw),
            "order_unit": order_unit,
            "minimum_buy_quantity": min_buy,
            "liquidity_lookback_sessions": lookback,
            "maximum_daily_participation": participation,
            "tick_size": tick,
            "price_limit_pct": price_limit,
        }
    fee_names = {
        "commission_per_side_bps", "minimum_commission_cny", "transfer_fee_per_side_bps",
        "stamp_duty_sell_bps", "slippage_per_side_bps",
    }
    if not fee_names.issubset(fees):
        raise ValueError(f"market fee rules are missing {sorted(fee_names - set(fees))}")
    normalized_fees = {
        key: _decimal(fees[key], key, minimum=_ZERO) for key in fee_names
    }
    normalized = {
        **dict(rule_pack),
        "effective_date": effective_date.isoformat(),
        "symbols": normalized_symbols,
        "fees": normalized_fees,
    }
    return normalized, stable_hash(normalized)


def _commission(notional: Decimal, fees: Mapping[str, Decimal]) -> Decimal:
    variable = _money(notional * fees["commission_per_side_bps"] / Decimal(10_000))
    return max(variable, _money(fees["minimum_commission_cny"]))


def _order_fees(notional: Decimal, side: str, fees: Mapping[str, Decimal]) -> dict[str, Decimal]:
    commission = _commission(notional, fees)
    transfer = _money(notional * fees["transfer_fee_per_side_bps"] / Decimal(10_000))
    stamp = (
        _money(notional * fees["stamp_duty_sell_bps"] / Decimal(10_000))
        if side == "sell" else _ZERO
    )
    return {"commission": commission, "transfer_fee": transfer, "stamp_duty": stamp}


def _round_quantity(value: Decimal, unit: int, *, rounding: str) -> Decimal:
    return (value / Decimal(unit)).to_integral_value(rounding=rounding) * Decimal(unit)


def _holding_quantity(lots: Sequence[Mapping[str, object]]) -> Decimal:
    return sum((_decimal(row["quantity"], "lot quantity", minimum=_ZERO) for row in lots), _ZERO)


def simulate_execution_ledger(
    *,
    execution_mode: str,
    trading_dates: Sequence[date | str],
    market_bars: Sequence[Mapping[str, object]],
    orders: Sequence[Mapping[str, object]],
    rule_pack: Mapping[str, object],
    initial_cash_cny: float,
    initial_lots: Mapping[str, Sequence[Mapping[str, object]]] | None = None,
    initial_holdings_cny: Mapping[str, float] | None = None,
    corporate_actions: Sequence[Mapping[str, object]] = (),
    source_artifact_hashes: Mapping[str, str] | None = None,
    research_cost_cny_by_date: Mapping[str, object] | None = None,
    prediction_records: Sequence[Mapping[str, object]] = (),
    prediction_linkage_mode: str = "linked",
    risk_limits: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Replay proposed orders without broker connectivity and keep forecast labels out of NAV.

    ``nominal_research`` sizes orders by CNY and allows fractional synthetic shares;
    ``quantity`` consumes share quantities, settlement availability, and per-board lots.
    Both modes apply prices, fees, capacity, limits, corporate actions, and cash checks.
    """
    if execution_mode not in {"nominal_research", "quantity"}:
        raise ValueError("execution_mode must be nominal_research or quantity")
    if prediction_linkage_mode not in {"linked", "independent"}:
        raise ValueError("prediction_linkage_mode must be linked or independent")
    calendar = tuple(_day(item, "trading date") for item in trading_dates)
    if not calendar or tuple(sorted(set(calendar))) != calendar:
        raise ValueError("trading_dates must be strictly increasing and unique")
    if execution_mode == "quantity" and initial_holdings_cny:
        raise ValueError("quantity mode uses initial_lots, not nominal CNY holdings")
    if execution_mode == "nominal_research" and initial_lots and initial_holdings_cny:
        raise ValueError("nominal mode accepts initial_lots or initial_holdings_cny, not both")
    cash = _money(_decimal(initial_cash_cny, "initial_cash_cny", minimum=_ZERO))
    rules, ruleset_sha = _verify_rule_pack(rule_pack)
    fees: Mapping[str, Decimal] = rules["fees"]  # type: ignore[assignment]
    risk = dict(risk_limits or {})
    max_single = _decimal(risk.get("maximum_single_name_weight", 1.0), "maximum_single_name_weight", minimum=_ZERO)
    max_gross = _decimal(risk.get("maximum_gross_exposure", 1.0), "maximum_gross_exposure", minimum=_ZERO)
    if max_single > 1 or max_gross > 1:
        raise ValueError("single-name and gross-exposure risk limits cannot exceed 100% of NAV")
    max_turnover = risk.get("maximum_daily_turnover_cny")
    max_turnover = _decimal(max_turnover, "maximum_daily_turnover_cny", minimum=_ZERO) if max_turnover is not None else None
    maximum_stress = risk.get("maximum_stress_loss_fraction")
    maximum_stress = _decimal(maximum_stress, "maximum_stress_loss_fraction", minimum=_ZERO) if maximum_stress is not None else None
    if maximum_stress is not None and maximum_stress > 1:
        raise ValueError("maximum_stress_loss_fraction cannot exceed 100% of NAV")
    stress_returns = risk.get("stress_returns_by_symbol", {})
    if not isinstance(stress_returns, Mapping):
        raise ValueError("stress_returns_by_symbol must be an object")

    bars: dict[tuple[str, date], Mapping[str, object]] = {}
    for row in market_bars:
        symbol = str(row.get("symbol", ""))
        day = _day(row.get("trade_date"), "market bar trade_date")
        if (symbol, day) in bars:
            raise ValueError(f"duplicate execution bar: {symbol} {day}")
        bars[(symbol, day)] = row
    if not bars:
        raise ValueError("execution ledger needs local market bars")
    order_rows = [dict(row) for row in orders]
    symbols = {str(row.get("symbol", "")) for row in order_rows}
    symbols |= set((initial_lots or {}).keys()) | set((initial_holdings_cny or {}).keys())
    symbols |= {str(row.get("symbol", "")) for row in corporate_actions}
    for symbol in symbols:
        if symbol not in rules["symbols"]:
            raise ValueError(f"market rule pack has no versioned rule for {symbol}")
    calendar_index = {day: index for index, day in enumerate(calendar)}
    research_costs: dict[date, Decimal] = {}
    for raw_day, raw_cost in (research_cost_cny_by_date or {}).items():
        cost_day = _day(raw_day, "research cost date")
        if cost_day not in calendar_index:
            raise ValueError("research cost date must be on the execution calendar")
        research_costs[cost_day] = _money(_decimal(raw_cost, "research cost", minimum=_ZERO))
    for key, bar in bars.items():
        if key[1] not in calendar_index:
            raise ValueError(f"execution bar date is outside the supplied calendar: {key}")
        status = str(bar.get("tradestatus", "unknown"))
        if status not in {"0", "1", "unknown"}:
            raise ValueError(f"invalid tradestatus for {key}")
    normalized_predictions: list[dict[str, object]] = []
    predictions_by_id: dict[str, dict[str, object]] = {}
    for prediction in prediction_records:
        if not isinstance(prediction, Mapping):
            raise ValueError("prediction records must be objects")
        prediction_id = prediction.get("prediction_id")
        symbol = prediction.get("symbol")
        if not isinstance(prediction_id, str) or not prediction_id.strip():
            raise ValueError("prediction_id must be a nonempty string")
        if prediction_id in predictions_by_id:
            raise ValueError("prediction IDs must be unique within the execution ledger")
        if not isinstance(symbol, str) or not symbol.strip():
            raise ValueError("prediction records must identify a symbol")
        prediction_day = _day(prediction.get("prediction_date"), "prediction_date")
        if prediction_day not in calendar_index:
            raise ValueError("prediction_date must be on the execution calendar")
        horizon = prediction.get("horizon_sessions")
        if not isinstance(horizon, int) or isinstance(horizon, bool) or horizon < 1:
            raise ValueError("prediction horizon_sessions must be a positive integer")
        available_raw = prediction.get("available_at")
        try:
            available_at = datetime.fromisoformat(str(available_raw).replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError("prediction available_at must be an ISO timestamp with a timezone") from exc
        if available_at.tzinfo is None or available_at.utcoffset() is None:
            raise ValueError("prediction available_at must include a timezone")
        available_local = available_at.astimezone(_CHINA_TZ)
        if available_local.date() != prediction_day or available_local.time().replace(tzinfo=None) < time(15, 0):
            raise ValueError("linked daily predictions must be available after the prediction-day close")
        prediction_row: dict[str, object] = {
            "prediction_id": prediction_id,
            "symbol": symbol,
            "prediction_date": prediction_day.isoformat(),
            "available_at": available_at.isoformat(),
            "label_maturity_date": (
                _day(prediction["label_maturity_date"], "label_maturity_date").isoformat()
                if prediction.get("label_maturity_date") else None
            ),
            "horizon_sessions": horizon,
            "expected_return": float(_decimal(prediction.get("expected_return"), "prediction expected_return")),
            "label_status": str(prediction.get("label_status", "pending")),
            "realized_return": prediction.get("realized_return"),
            "ledger": "prediction_only_no_NAV_effect",
        }
        predictions_by_id[prediction_id] = prediction_row
        normalized_predictions.append(prediction_row)

    if prediction_linkage_mode == "independent" and normalized_predictions:
        raise ValueError("independent account replay cannot include prediction records")

    orders_by_date: dict[date, list[dict[str, object]]] = defaultdict(list)
    for row in order_rows:
        trade_day = _day(row.get("trade_date"), "order trade_date")
        if trade_day not in calendar_index:
            raise ValueError("order execution date is outside the supplied calendar")
        if row.get("side") not in {"buy", "sell"}:
            raise ValueError("order side must be buy or sell")
        if prediction_linkage_mode == "independent":
            if any(key in row for key in ("source_prediction_id", "prediction_date", "prediction_horizon_sessions")):
                raise ValueError("independent account replay orders must not claim a prediction link")
        else:
            prediction_id = row.get("source_prediction_id")
            if not isinstance(prediction_id, str) or not prediction_id.strip():
                raise ValueError("each linked order must reference an existing prediction ID")
            prediction = predictions_by_id.get(prediction_id)
            if prediction is None:
                raise ValueError(f"order source_prediction_id is unknown: {prediction_id}")
            if prediction["symbol"] != row.get("symbol"):
                raise ValueError("order symbol does not match its source prediction")
            if row.get("prediction_date") != prediction["prediction_date"]:
                raise ValueError("order prediction_date does not match its source prediction")
            order_horizon = row.get("prediction_horizon_sessions")
            if not isinstance(order_horizon, int) or isinstance(order_horizon, bool) or order_horizon != prediction["horizon_sessions"]:
                raise ValueError("order prediction horizon does not match its source prediction")
            prediction_day = date.fromisoformat(str(prediction["prediction_date"]))
            prediction_index = calendar_index[prediction_day]
            trade_index = calendar_index[trade_day]
            if trade_index != prediction_index + 1:
                raise ValueError("linked orders must execute at the next session open after their prediction date")
            available_at = datetime.fromisoformat(str(prediction["available_at"]))
            trade_open = datetime.combine(trade_day, time(9, 30), tzinfo=_CHINA_TZ)
            if available_at.astimezone(_CHINA_TZ) >= trade_open:
                raise ValueError("source prediction was not available before the execution-session open")
        orders_by_date[trade_day].append(row)

    lots: dict[str, list[dict[str, object]]] = defaultdict(list)
    initialization = []
    for symbol, rows in (initial_lots or {}).items():
        for row in rows:
            quantity = _decimal(row.get("quantity"), "initial lot quantity", minimum=Decimal("0.000001"))
            acquired = _day(row.get("acquired_date"), "initial lot acquired_date")
            acquired_index = calendar_index.get(acquired, -1 if acquired < calendar[0] else None)
            if acquired_index is None:
                raise ValueError("initial lot acquired_date must be in or before the trading calendar")
            lots[str(symbol)].append({
                "quantity": quantity,
                "acquired_index": acquired_index,
                "unit_cost_cny": _decimal(row.get("unit_cost_cny", 0), "unit_cost_cny", minimum=_ZERO),
            })
            initialization.append({"symbol": str(symbol), "quantity": _out(quantity), "source": "initial_lot"})

    if initial_holdings_cny:
        first_day = calendar[0]
        for symbol, raw_value in initial_holdings_cny.items():
            value = _money(_decimal(raw_value, "initial holding value", minimum=_ZERO))
            bar = bars.get((str(symbol), first_day))
            if bar is None or bar.get("preclose") is None or _decimal(bar["preclose"], "preclose", minimum=Decimal("0.000001")) <= 0:
                raise ValueError(f"nominal holdings need a first-day preclose for {symbol}")
            price = _decimal(bar["preclose"], "initial preclose", minimum=Decimal("0.000001"))
            quantity = value / price
            lots[str(symbol)].append({
                "quantity": quantity,
                "acquired_index": -1,
                "unit_cost_cny": price,
            })
            initialization.append({
                "symbol": str(symbol),
                "nominal_value_cny": float(value),
                "reference_preclose": float(price),
                "synthetic_quantity": float(quantity),
                "source": "nominal_value_conversion",
            })

    action_rows = []
    hashes = dict(source_artifact_hashes or {})
    actions_by_date: dict[date, list[Mapping[str, object]]] = defaultdict(list)
    for action in corporate_actions:
        effective = _day(action.get("effective_date"), "corporate action effective_date")
        if effective not in calendar_index:
            raise ValueError("corporate action effective_date must be on the supplied calendar")
        source_ref = str(action.get("source_ref", ""))
        source_hash = str(action.get("source_sha256", ""))
        if not source_ref or not _HEX64.fullmatch(source_hash) or hashes.get(source_ref) != source_hash:
            raise ValueError("corporate action must match a verified source_ref and SHA-256")
        if action.get("type") not in {"cash_dividend", "stock_distribution"}:
            raise ValueError("corporate action type must be cash_dividend or stock_distribution")
        actions_by_date[effective].append(action)

    prediction_ledger = normalized_predictions

    def quantity(symbol: str) -> Decimal:
        return _holding_quantity(lots.get(symbol, []))

    previous_close: dict[str, Decimal] = {}
    last_mark: dict[str, Decimal] = {}
    initial_value = cash
    if initial_holdings_cny:
        initial_value += sum((_money(_decimal(value, "initial holding value", minimum=_ZERO)) for value in initial_holdings_cny.values()), _ZERO)
    else:
        first_day = calendar[0]
        for symbol, held_lots in lots.items():
            bar = bars.get((symbol, first_day))
            if bar is None or bar.get("preclose") is None:
                raise ValueError(f"quantity holdings need a first-day preclose for {symbol}")
            preclose = _decimal(bar["preclose"], "preclose", minimum=Decimal("0.000001"))
            previous_close[symbol] = preclose
            last_mark[symbol] = preclose
            initial_value += _money(quantity(symbol) * preclose)

    dividend_receivables: list[dict[str, object]] = []
    transaction_ledger = []
    action_ledger = []
    daily_nav = []
    reconciliation = []
    risk_checks = []
    total_fees = _ZERO
    total_slippage = _ZERO
    total_research_cost = _ZERO
    total_traded = _ZERO
    blocked_order_count = 0

    def market_value(symbol: str, price: Decimal) -> Decimal:
        return _money(quantity(symbol) * price)

    def sellable_quantity(symbol: str, day_index: int, t_plus_one: bool) -> Decimal:
        return sum((
            _decimal(lot["quantity"], "position quantity", minimum=_ZERO)
            for lot in lots.get(symbol, [])
            if not t_plus_one or int(lot["acquired_index"]) < day_index
        ), _ZERO)

    def take_lots(symbol: str, amount: Decimal, day_index: int, t_plus_one: bool) -> None:
        remaining = amount
        for lot in sorted(lots[symbol], key=lambda item: int(item["acquired_index"])):
            if remaining <= 0:
                break
            if t_plus_one and int(lot["acquired_index"]) >= day_index:
                continue
            available = _decimal(lot["quantity"], "position quantity", minimum=_ZERO)
            used = min(available, remaining)
            lot["quantity"] = available - used
            remaining -= used
        if remaining > Decimal("0.000000001"):
            raise RuntimeError("sell execution exceeded sellable share reconciliation")
        lots[symbol] = [lot for lot in lots[symbol] if _decimal(lot["quantity"], "position quantity") > 0]

    def capacity_for(symbol: str, day_index: int, rule: Mapping[str, object]) -> Decimal:
        start = max(0, day_index - int(rule["liquidity_lookback_sessions"]))
        amounts = [
            _decimal(row["amount"], "historical traded amount", minimum=Decimal("0"))
            for prior_day in calendar[start:day_index]
            if (row := bars.get((symbol, prior_day))) is not None and row.get("amount") is not None
            and _decimal(row["amount"], "historical traded amount", minimum=Decimal("0")) > 0
        ]
        if len(amounts) < int(rule["liquidity_lookback_sessions"]):
            return _ZERO
        avg = sum(amounts, _ZERO) / Decimal(len(amounts))
        return _money(avg * rule["maximum_daily_participation"])

    def expected_stress_loss(values: Mapping[str, Decimal]) -> Decimal:
        total = _ZERO
        for symbol, value in values.items():
            shock = _decimal(stress_returns.get(symbol, 0), f"stress return for {symbol}")
            total += value * max(-shock, _ZERO)
        return _money(total)

    for day_index, trade_day in enumerate(calendar):
        open_cash = cash
        open_receivables = sum((_money(row["amount_cny"]) for row in dividend_receivables), _ZERO)
        opening_quantities = {symbol: quantity(symbol) for symbol in set(lots) | symbols}
        day_action_quantity_change: dict[str, Decimal] = defaultdict(lambda: _ZERO)
        day_buy_qty: dict[str, Decimal] = defaultdict(lambda: _ZERO)
        day_sell_qty: dict[str, Decimal] = defaultdict(lambda: _ZERO)
        day_sell_proceeds = _ZERO
        day_buy_gross = _ZERO
        day_fees = _ZERO
        day_dividend_payments = _ZERO
        day_turnover = _ZERO
        day_blocked = 0

        for action in actions_by_date.get(trade_day, []):
            symbol = str(action.get("symbol", ""))
            if symbol not in rules["symbols"]:
                raise ValueError(f"corporate action has no versioned market rule for {symbol}")
            before = quantity(symbol)
            if action["type"] == "cash_dividend":
                amount_per_share = _decimal(action.get("cash_per_share_cny"), "cash dividend per share", minimum=_ZERO)
                amount = _money(before * amount_per_share)
                payment_day = _day(action.get("payment_date"), "dividend payment_date")
                if payment_day not in calendar_index or calendar_index[payment_day] < day_index:
                    raise ValueError("dividend payment date must be on or after its effective date")
                dividend_receivables.append({
                    "symbol": symbol,
                    "effective_date": trade_day.isoformat(),
                    "payment_date": payment_day,
                    "amount_cny": amount,
                    "source_ref": action["source_ref"],
                    "source_sha256": action["source_sha256"],
                })
                action_ledger.append({
                    "effective_date": trade_day.isoformat(),
                    "type": "cash_dividend",
                    "symbol": symbol,
                    "record_quantity": float(before),
                    "cash_per_share_cny": float(amount_per_share),
                    "receivable_cny": float(amount),
                    "payment_date": payment_day.isoformat(),
                    "source_ref": action["source_ref"],
                    "source_sha256": action["source_sha256"],
                })
            else:
                new_shares_per_share = _decimal(action.get("new_shares_per_share"), "stock distribution ratio", minimum=_ZERO)
                added = before * new_shares_per_share
                available_date = _day(action.get("available_for_sale_date", trade_day), "available_for_sale_date")
                if available_date not in calendar_index or calendar_index[available_date] < day_index:
                    raise ValueError("new stock distribution shares cannot be available before the effective date")
                available_index = calendar_index[available_date]
                if added > 0:
                    lots[symbol].append({
                        "quantity": added,
                        "acquired_index": available_index - 1,
                        "unit_cost_cny": _ZERO,
                    })
                    day_action_quantity_change[symbol] += added
                action_ledger.append({
                    "effective_date": trade_day.isoformat(),
                    "type": "stock_distribution",
                    "symbol": symbol,
                    "old_quantity": float(before),
                    "new_shares_per_share": float(new_shares_per_share),
                    "added_quantity": float(added),
                    "available_for_sale_date": available_date.isoformat(),
                    "source_ref": action["source_ref"],
                    "source_sha256": action["source_sha256"],
                })

        still_receivable = []
        for item in dividend_receivables:
            if item["payment_date"] == trade_day:
                amount = _money(item["amount_cny"])
                cash += amount
                day_dividend_payments += amount
            else:
                still_receivable.append(item)
        dividend_receivables = still_receivable

        open_prices: dict[str, Decimal] = {}
        close_prices: dict[str, Decimal] = {}
        for symbol in symbols:
            bar = bars.get((symbol, trade_day))
            if bar is not None and bar.get("open") is not None:
                open_prices[symbol] = _decimal(bar["open"], f"{symbol} open", minimum=Decimal("0.000001"))
            elif symbol in last_mark:
                open_prices[symbol] = last_mark[symbol]
            if bar is not None and bar.get("close") is not None:
                close_prices[symbol] = _decimal(bar["close"], f"{symbol} close", minimum=Decimal("0.000001"))
            elif symbol in last_mark:
                close_prices[symbol] = last_mark[symbol]

        capacity_remaining = {
            symbol: capacity_for(symbol, day_index, rules["symbols"][symbol])
            for symbol in symbols
        }
        for order in orders_by_date.get(trade_day, []):
            symbol = str(order.get("symbol", ""))
            side = str(order["side"])
            rule: Mapping[str, object] = rules["symbols"][symbol]
            bar = bars.get((symbol, trade_day))
            row = {
                "trade_date": trade_day.isoformat(),
                "symbol": symbol,
                "side": side,
                "source_prediction_id": order.get("source_prediction_id"),
                "requested_quantity": None,
                "requested_notional_cny": None,
                "filled_quantity": 0.0,
                "filled_notional_cny": 0.0,
                "reference_price": None,
                "fill_price": None,
                "fees_cny": 0.0,
                "status": "blocked",
                "reason": None,
                "rule_set_id": rules["ruleset_id"],
            }
            if bar is None or str(bar.get("tradestatus", "unknown")) != "1" or bar.get("open") is None:
                row["reason"] = "suspended_or_missing_bar"
                transaction_ledger.append(row)
                day_blocked += 1
                blocked_order_count += 1
                continue
            reference = _decimal(bar["open"], f"{symbol} open", minimum=Decimal("0.000001"))
            tick = rule["tick_size"]
            slip_rate = fees["slippage_per_side_bps"] / Decimal(10_000)
            raw_fill = reference * (Decimal(1) + slip_rate if side == "buy" else Decimal(1) - slip_rate)
            fill_price = _stable_tick(raw_fill, tick, ROUND_CEILING if side == "buy" else ROUND_FLOOR)
            row["reference_price"] = float(reference)
            if fill_price <= 0:
                row["reason"] = "invalid_slippage_price"
                transaction_ledger.append(row)
                day_blocked += 1
                blocked_order_count += 1
                continue
            limit_pct = rule["price_limit_pct"]
            locked_reason = None
            if limit_pct is not None and bar.get("preclose") is not None:
                preclose = _decimal(bar["preclose"], f"{symbol} preclose", minimum=Decimal("0.000001"))
                upper = _stable_tick(preclose * (Decimal(1) + limit_pct), tick, ROUND_HALF_UP)
                lower = _stable_tick(preclose * (Decimal(1) - limit_pct), tick, ROUND_HALF_UP)
                high = _decimal(bar.get("high", reference), f"{symbol} high", minimum=Decimal("0.000001"))
                low = _decimal(bar.get("low", reference), f"{symbol} low", minimum=Decimal("0.000001"))
                locked_up = bool(bar.get("limit_up_locked")) or (reference >= upper and high == upper and low == upper)
                locked_down = bool(bar.get("limit_down_locked")) or (reference <= lower and high == lower and low == lower)
                if side == "buy" and locked_up:
                    locked_reason = "locked_at_upper_limit"
                elif side == "sell" and locked_down:
                    locked_reason = "locked_at_lower_limit"
                elif fill_price > upper or fill_price < lower:
                    locked_reason = "slippage_price_outside_daily_limit"
            if locked_reason:
                row["reason"] = locked_reason
                transaction_ledger.append(row)
                day_blocked += 1
                blocked_order_count += 1
                continue

            if execution_mode == "quantity":
                requested_quantity = (
                    _decimal(order.get("quantity"), "order quantity", minimum=Decimal("0.000001"))
                    if order.get("quantity") is not None
                    else _decimal(order.get("notional_cny"), "order notional", minimum=Decimal("0.01")) / fill_price
                )
                if requested_quantity != requested_quantity.to_integral_value():
                    raise ValueError("quantity mode order.quantity must be a whole share count")
                row["requested_quantity"] = float(requested_quantity)
                desired = requested_quantity
                order_requested_quantity = requested_quantity
                if side == "buy":
                    desired = _round_quantity(desired, int(rule["order_unit"]), rounding=ROUND_FLOOR)
                    if desired < Decimal(int(rule["minimum_buy_quantity"])):
                        row["reason"] = "below_minimum_buy_lot"
                        transaction_ledger.append(row)
                        day_blocked += 1
                        blocked_order_count += 1
                        continue
                else:
                    available = sellable_quantity(symbol, day_index, bool(rule["t_plus_one"]))
                    desired = min(desired, available)
                    if desired == available and bool(rule["odd_lot_sell"]):
                        pass
                    else:
                        desired = _round_quantity(desired, int(rule["order_unit"]), rounding=ROUND_FLOOR)
            else:
                requested_notional = _decimal(order.get("notional_cny"), "order notional", minimum=Decimal("0.01"))
                row["requested_notional_cny"] = float(requested_notional)
                desired = requested_notional / fill_price
                order_requested_quantity = desired
                if side == "sell":
                    available = sellable_quantity(symbol, day_index, False)
                    desired = min(desired, available)
            if desired <= 0:
                row["reason"] = "no_sellable_or_lot_eligible_quantity"
                transaction_ledger.append(row)
                day_blocked += 1
                blocked_order_count += 1
                continue

            requested_fill_quantity = order_requested_quantity
            capacity_qty = capacity_remaining[symbol] / fill_price
            capacity_limited = capacity_qty < desired
            desired = min(desired, capacity_qty)
            if execution_mode == "quantity":
                if side == "buy" or desired < sellable_quantity(symbol, day_index, bool(rule["t_plus_one"])):
                    desired = _round_quantity(desired, int(rule["order_unit"]), rounding=ROUND_FLOOR)
            if max_turnover is not None:
                turnover_room = max_turnover - day_turnover
                desired = min(desired, max(turnover_room, _ZERO) / fill_price)
                if execution_mode == "quantity":
                    desired = _round_quantity(desired, int(rule["order_unit"]), rounding=ROUND_FLOOR)
            if side == "buy":
                current_values = {
                    key: market_value(key, open_prices[key])
                    for key in symbols if key in open_prices
                }
                nav_at_open = cash + sum(current_values.values(), _ZERO) + sum(
                    (_money(item["amount_cny"]) for item in dividend_receivables), _ZERO
                )
                current_gross = sum(current_values.values(), _ZERO)
                if max_gross < 1:
                    gross_room = max(nav_at_open * max_gross - current_gross, _ZERO)
                    desired = min(desired, gross_room / fill_price)
                if max_single < 1:
                    single_room = max(nav_at_open * max_single - current_values.get(symbol, _ZERO), _ZERO)
                    desired = min(desired, single_room / fill_price)
                if maximum_stress is not None:
                    allowable_stress_loss = nav_at_open * maximum_stress
                    if expected_stress_loss(current_values) > allowable_stress_loss:
                        desired = _ZERO
                    else:
                        shock = max(-_decimal(stress_returns.get(symbol, 0), f"stress return for {symbol}"), _ZERO)
                        if shock > 0:
                            current_stress = expected_stress_loss(current_values)
                            desired = min(desired, max(allowable_stress_loss - current_stress, _ZERO) / (fill_price * shock))
                if execution_mode == "quantity":
                    desired = _round_quantity(desired, int(rule["order_unit"]), rounding=ROUND_FLOOR)
                # Use bounded binary search so per-order minimum commissions cannot
                # make affordable notional replay take an unbounded number of steps.
                if execution_mode == "quantity":
                    unit = Decimal(int(rule["order_unit"]))
                    high_units = int((desired / unit).to_integral_value(rounding=ROUND_FLOOR))
                    low_units = 0
                    while low_units < high_units:
                        middle = (low_units + high_units + 1) // 2
                        candidate_qty = unit * middle
                        gross = _money(candidate_qty * fill_price)
                        fee_total = sum(_order_fees(gross, side, fees).values(), _ZERO)
                        if gross + fee_total <= cash:
                            low_units = middle
                        else:
                            high_units = middle - 1
                    desired = unit * low_units
                else:
                    low, high = _ZERO, desired
                    for _ in range(64):
                        middle = (low + high) / 2
                        gross = _money(middle * fill_price)
                        fee_total = sum(_order_fees(gross, side, fees).values(), _ZERO)
                        if gross + fee_total <= cash:
                            low = middle
                        else:
                            high = middle
                    desired = low
                if desired <= 0:
                    row["reason"] = "capacity_or_risk_limit" if capacity_limited else "cash_or_risk_budget"
                    transaction_ledger.append(row)
                    day_blocked += 1
                    blocked_order_count += 1
                    continue
            else:
                if desired > sellable_quantity(symbol, day_index, bool(rule["t_plus_one"])):
                    desired = sellable_quantity(symbol, day_index, bool(rule["t_plus_one"]))
                if execution_mode == "quantity" and desired < sellable_quantity(symbol, day_index, bool(rule["t_plus_one"])):
                    desired = _round_quantity(desired, int(rule["order_unit"]), rounding=ROUND_FLOOR)
            if desired <= 0:
                row["reason"] = "capacity_or_risk_limit"
                transaction_ledger.append(row)
                day_blocked += 1
                blocked_order_count += 1
                continue

            fill_notional = _money(desired * fill_price)
            fee_parts = _order_fees(fill_notional, side, fees)
            fee_total = sum(fee_parts.values(), _ZERO)
            slippage_cost = _money(desired * abs(fill_price - reference))
            if side == "buy":
                cash -= fill_notional + fee_total
                day_buy_gross += fill_notional
                day_buy_qty[symbol] += desired
                lots[symbol].append({
                    "quantity": desired,
                    "acquired_index": day_index,
                    "unit_cost_cny": fill_price,
                })
            else:
                take_lots(symbol, desired, day_index, bool(rule["t_plus_one"]))
                cash += fill_notional - fee_total
                day_sell_proceeds += fill_notional
                day_sell_qty[symbol] += desired
            capacity_remaining[symbol] = max(_ZERO, capacity_remaining[symbol] - fill_notional)
            day_turnover += fill_notional
            day_fees += fee_total
            total_fees += fee_total
            total_slippage += slippage_cost
            total_traded += fill_notional
            row.update({
                "filled_quantity": float(desired),
                "filled_notional_cny": float(fill_notional),
                "fill_price": float(fill_price),
                "fees_cny": float(fee_total),
                "slippage_cost_cny": float(slippage_cost),
                "fee_components_cny": {key: float(value) for key, value in fee_parts.items()},
                "status": "filled" if desired == requested_fill_quantity else "partial_or_limited",
                "reason": "capacity_cash_lot_risk_or_sellability_limited" if desired < requested_fill_quantity else None,
                "sellable_quantity_after_fill": float(sellable_quantity(symbol, day_index, bool(rule["t_plus_one"]))),
            })
            transaction_ledger.append(row)
            risk_checks.append({
                "trade_date": trade_day.isoformat(),
                "symbol": symbol,
                "source_prediction_id": order.get("source_prediction_id"),
                "check": "cash_capacity_lot_settlement_market_limits_and_exposure",
                "status": "pass",
            })

        end_values = {}
        for symbol in symbols | set(lots):
            mark = close_prices.get(symbol) or last_mark.get(symbol) or open_prices.get(symbol)
            if mark is not None:
                last_mark[symbol] = mark
                end_values[symbol] = _money(quantity(symbol) * mark)
                if (symbol, trade_day) in bars and bars[(symbol, trade_day)].get("close") is not None:
                    previous_close[symbol] = mark
        end_receivables = sum((_money(item["amount_cny"]) for item in dividend_receivables), _ZERO)
        day_research_cost = research_costs.get(trade_day, _ZERO)
        if day_research_cost > cash:
            raise ValueError(f"research cost exceeds available cash on {trade_day}")
        cash -= day_research_cost
        total_research_cost += day_research_cost
        nav = _money(cash + end_receivables + sum(end_values.values(), _ZERO))
        if nav <= 0:
            raise ValueError(f"execution ledger NAV is nonpositive on {trade_day}")
        cash_expected = _money(
            open_cash + day_sell_proceeds - day_buy_gross - day_fees
            + day_dividend_payments - day_research_cost
        )
        cash_residual = _money(cash - cash_expected)
        if cash_residual != 0:
            raise RuntimeError(f"cash reconciliation failed on {trade_day}: {cash_residual}")
        quantity_checks = []
        for symbol in sorted(set(opening_quantities) | set(lots)):
            expected_quantity = (
                opening_quantities.get(symbol, _ZERO)
                + day_action_quantity_change.get(symbol, _ZERO)
                + day_buy_qty.get(symbol, _ZERO)
                - day_sell_qty.get(symbol, _ZERO)
            )
            actual_quantity = quantity(symbol)
            residual = actual_quantity - expected_quantity
            if abs(residual) > Decimal("0.000000001"):
                raise RuntimeError(f"share reconciliation failed on {trade_day} for {symbol}: {residual}")
            quantity_checks.append({"symbol": symbol, "expected_quantity": float(expected_quantity), "actual_quantity": float(actual_quantity), "residual": float(residual)})
        nav_residual = _money(nav - (cash + end_receivables + sum(end_values.values(), _ZERO)))
        if nav_residual != 0:
            raise RuntimeError(f"NAV reconciliation failed on {trade_day}: {nav_residual}")
        recon = {
            "trade_date": trade_day.isoformat(),
            "cash_open_cny": float(open_cash),
            "sell_proceeds_cny": float(day_sell_proceeds),
            "buy_notional_cny": float(day_buy_gross),
            "fees_cny": float(day_fees),
            "research_cost_cny": float(day_research_cost),
            "dividend_payments_cny": float(day_dividend_payments),
            "cash_expected_close_cny": float(cash_expected),
            "cash_actual_close_cny": float(cash),
            "cash_residual_cny": float(cash_residual),
            "receivable_close_cny": float(end_receivables),
            "nav_cny": float(nav),
            "nav_residual_cny": float(nav_residual),
            "quantity_checks": quantity_checks,
        }
        reconciliation.append(recon)
        daily_nav.append({
            "trade_date": trade_day.isoformat(),
            "cash_cny": float(cash),
            "dividend_receivable_cny": float(end_receivables),
            "market_value_cny": float(sum(end_values.values(), _ZERO)),
            "nav_cny": float(nav),
            "daily_return": float(nav / (Decimal(str(daily_nav[-1]["nav_cny"])) if daily_nav else initial_value) - 1),
            "risk_checks_failed": day_blocked,
            "position_quantities": {symbol: float(quantity(symbol)) for symbol in sorted(symbols) if quantity(symbol) > 0},
            "position_sellable_quantities": {
                symbol: float(sellable_quantity(symbol, day_index, bool(rules["symbols"][symbol]["t_plus_one"])))
                for symbol in sorted(symbols) if quantity(symbol) > 0
            },
        })

    final_nav = Decimal(str(daily_nav[-1]["nav_cny"]))
    predictions_out = []
    for row in prediction_ledger:
        # The account ledger does not consume a forecast or label value.
        predictions_out.append(row)
    return {
        "schema_version": "adaptive-mas.execution-ledger.v1",
        "execution_mode": execution_mode,
        "prediction_linkage_mode": prediction_linkage_mode,
        "broker_orders_enabled": False,
        "prediction_and_accounting_separation": "prediction_ledger_is_informational_and_has_zero_NAV_effect",
        "ruleset": {
            "ruleset_id": rules["ruleset_id"],
            "source_title": rules["source_title"],
            "source_url": rules["source_url"],
            "effective_date": rules["effective_date"],
            "configuration_sha256": ruleset_sha,
        },
        "initialization": initialization,
        "prediction_ledger": predictions_out,
        "corporate_action_ledger": action_ledger,
        "transactions": transaction_ledger,
        "daily_nav": daily_nav,
        "reconciliation": reconciliation,
        "risk_checks": risk_checks,
        "summary": {
            "initial_nav_cny": float(initial_value),
            "final_nav_cny": float(final_nav),
            "net_total_return": float(final_nav / initial_value - 1) if initial_value else 0.0,
            "total_traded_notional_cny": float(total_traded),
            "total_fees_cny": float(total_fees),
            "total_slippage_cost_cny": float(total_slippage),
            "total_research_cost_cny": float(total_research_cost),
            "blocked_order_count": blocked_order_count,
            "all_cash_reconciliations_exact": all(row["cash_residual_cny"] == 0 for row in reconciliation),
            "all_nav_reconciliations_exact": all(row["nav_residual_cny"] == 0 for row in reconciliation),
            "all_quantity_reconciliations_exact": all(
                check["residual"] == 0 for row in reconciliation for check in row["quantity_checks"]
            ),
            "execution_assumptions": (
                "research notional targets mapped to fractional synthetic shares at open; no exchange lots or T+1"
                if execution_mode == "nominal_research"
                else "whole-share fills constrained by the versioned per-symbol lot and settlement rules"
            ),
        },
    }
