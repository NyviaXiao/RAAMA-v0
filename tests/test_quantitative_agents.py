from datetime import date, timedelta

from adaptive_mas.agents.quantitative import quantitative_agents
from adaptive_mas.data.research import MarketSnapshot


def test_quantitative_agents_emit_point_in_time_five_session_signals():
    trading_dates = tuple(date(2024, 1, 1) + timedelta(days=index) for index in range(32))
    adjusted_bars = []
    for symbol, daily_return in (("600000", 0.01), ("000001", -0.005), ("000002", 0.002)):
        close = 10.0
        for index, trade_date in enumerate(trading_dates):
            close *= 1 + daily_return
            adjusted_bars.append(
                {
                    "symbol": symbol,
                    "trade_date": trade_date,
                    "available_at": trade_date,
                    "close": close,
                    "amount": 1000.0 * (1 + index / 20),
                    "source_ref": "synthetic-adjusted-bars",
                }
            )
    snapshot = MarketSnapshot(
        decision_date=trading_dates[-1],
        trading_dates=trading_dates,
        members=frozenset({"600000", "000001", "000002"}),
        adjusted_bars=tuple(adjusted_bars),
        execution_bars=(),
        index_bars=(),
    )

    signals = quantitative_agents(snapshot)

    assert set(signals) == {"trend", "reversal", "liquidity_confirmation"}
    assert all(signals.values())
    assert all(
        signal.horizon_sessions == 5
        and signal.decision_date == snapshot.decision_date
        and signal.input_cutoff == snapshot.decision_date
        and signal.evidence.end_date == snapshot.decision_date
        for rows in signals.values()
        for signal in rows
    )
    trend_by_symbol = {signal.symbol: signal.score for signal in signals["trend"]}
    reversal_by_symbol = {signal.symbol: signal.score for signal in signals["reversal"]}
    assert trend_by_symbol["600000"] > 0
    assert reversal_by_symbol["600000"] < 0
