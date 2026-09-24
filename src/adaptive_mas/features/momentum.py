"""Minimal price-only signal used to check the research-data contract."""

from adaptive_mas.data.research import MarketSnapshot


def momentum(snapshot: MarketSnapshot, lookback_sessions: int) -> list[dict[str, object]]:
    if lookback_sessions < 1:
        raise ValueError("动量回看窗口必须至少为一个交易日")
    decision_index = snapshot.trading_dates.index(snapshot.decision_date)
    if decision_index < lookback_sessions:
        return []
    lookback_date = snapshot.trading_dates[decision_index - lookback_sessions]
    bars = {
        (str(row["symbol"]), row["trade_date"]): row
        for row in snapshot.adjusted_bars
    }

    signals = []
    for symbol in sorted(snapshot.members):
        current = bars.get((symbol, snapshot.decision_date))
        prior = bars.get((symbol, lookback_date))
        if current is None or prior is None:
            continue
        current_close = current["close"]
        prior_close = prior["close"]
        if current_close is None or prior_close is None or prior_close <= 0:
            continue
        signals.append(
            {
                "symbol": symbol,
                "decision_date": snapshot.decision_date,
                "momentum": current_close / prior_close - 1,
                "input_cutoff": max(current["trade_date"], prior["trade_date"]),
            }
        )
    return signals
