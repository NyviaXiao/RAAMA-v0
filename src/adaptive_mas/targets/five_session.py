"""Ex-post five-session return labels, kept outside feature construction."""

from datetime import date

from adaptive_mas.data.research import ResearchData


def five_session_forward_returns(
    research: ResearchData, decision_dates: list[date]
) -> list[dict[str, object]]:
    date_index = {day: index for index, day in enumerate(research.trading_dates)}
    bars = {
        (str(row["symbol"]), row["trade_date"]): row
        for row in research.adjusted_bars
    }

    labels = []
    for decision_date in decision_dates:
        if decision_date not in date_index:
            raise ValueError(f"目标决策日不在研究交易日历中：{decision_date}")
        index = date_index[decision_date]
        entry_index = index + 1
        exit_index = index + 5
        if exit_index >= len(research.trading_dates):
            continue
        entry_date = research.trading_dates[entry_index]
        exit_date = research.trading_dates[exit_index]
        for symbol in sorted(research.membership_at(decision_date)):
            entry = bars.get((symbol, entry_date))
            exit_bar = bars.get((symbol, exit_date))
            if entry is None or exit_bar is None:
                continue
            entry_open = entry["open"]
            exit_close = exit_bar["close"]
            if entry_open is None or exit_close is None or entry_open <= 0:
                continue
            labels.append(
                {
                    "symbol": symbol,
                    "decision_date": decision_date,
                    "forward_return_5s": exit_close / entry_open - 1,
                    "entry_date": entry_date,
                    "exit_date": exit_date,
                    "available_at": exit_date,
                }
            )
    return labels
