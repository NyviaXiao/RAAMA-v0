import math
from datetime import date, timedelta

from adaptive_mas.data.research import MembershipSnapshot, ResearchData
from adaptive_mas.evaluation.agent_research import run_agent_research


def test_agent_research_replays_matured_outcomes_before_changing_gate_weights():
    trading_dates = tuple(date(2024, 1, 1) + timedelta(days=index) for index in range(45))
    symbols = ("600000", "000001", "000002", "000003")
    adjusted_bars = []
    for symbol_index, symbol in enumerate(symbols):
        close = 20.0 + symbol_index
        for index, trade_date in enumerate(trading_dates):
            daily_return = (
                (symbol_index - 1.5) * 0.0007
                + 0.006 * math.sin(index * 0.43 + symbol_index * 1.1)
            )
            close *= 1 + daily_return
            adjusted_bars.append(
                {
                    "symbol": symbol,
                    "trade_date": trade_date,
                    "available_at": trade_date,
                    "open": close * 0.999,
                    "high": close * 1.002,
                    "low": close * 0.997,
                    "close": close,
                    "amount": 1000 * (1 + index / 50) * (1 + 0.1 * math.sin(index + symbol_index)),
                    "source_ref": "synthetic-adjusted-bars",
                }
            )
    index_bars = [
        {
            "trade_date": trade_date,
            "available_at": trade_date,
            "open": 100 + index * 0.1,
            "close": 100 + index * 0.11,
        }
        for index, trade_date in enumerate(trading_dates)
    ]
    research = ResearchData(
        list(trading_dates),
        adjusted_bars,
        [],
        index_bars,
        [
            MembershipSnapshot(
                event_time=trading_dates[0],
                available_at=trading_dates[0],
                members=frozenset(symbols),
                source_ref="synthetic-membership",
            )
        ],
    )

    result = run_agent_research(
        research,
        evaluation_step_sessions=5,
        reliability_lookback_windows=2,
        reliability_minimum_history_windows=2,
        equal_weight_share=0.5,
        top_decile_fraction=0.5,
        trend_lookback_sessions=20,
        reversal_lookback_sessions=5,
        liquidity_return_lookback_sessions=5,
        liquidity_recent_amount_sessions=5,
        liquidity_reference_amount_sessions=20,
    )

    observations = result["observations"]
    assert result["summary"]["decision_windows"] == 4
    assert result["summary"]["adaptive_windows_after_warmup"] == 2
    assert [row["reliability_history_windows"] for row in observations] == [0, 1, 2, 3]
    assert set(observations[0]["weights_used"].values()) == {1 / 3}
    assert observations[-1]["label_available_at"] == trading_dates[44].isoformat()
    assert len(result["signal_records"]) == 4 * len(symbols) * 3
    assert all(
        record["input_cutoff"] == record["decision_date"]
        and record["horizon_sessions"] == 5
        and record["evidence"]["end_date"] == record["decision_date"]
        for record in result["signal_records"]
    )
