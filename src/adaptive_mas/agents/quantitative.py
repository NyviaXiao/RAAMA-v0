"""Deterministic quantitative experts over a single point-in-time snapshot."""

import math
import statistics
from datetime import date

from adaptive_mas.agents.signals import AgentSignal, EvidenceRef
from adaptive_mas.data.research import MarketSnapshot


AGENT_IDS = ("trend", "reversal", "liquidity_confirmation")


def quantitative_agents(
    snapshot: MarketSnapshot,
    trend_lookback_sessions: int = 20,
    reversal_lookback_sessions: int = 5,
    liquidity_return_sessions: int = 5,
    liquidity_recent_amount_sessions: int = 5,
    liquidity_reference_amount_sessions: int = 20,
) -> dict[str, list[AgentSignal]]:
    """Return independent scores; no agent receives another agent's output."""
    decision_index = snapshot.trading_dates.index(snapshot.decision_date)
    recent_history = (
        liquidity_recent_amount_sessions + liquidity_reference_amount_sessions
    )
    first_required_index = max(
        trend_lookback_sessions,
        reversal_lookback_sessions,
        liquidity_return_sessions,
        recent_history - 1,
    )
    if decision_index < first_required_index:
        return {agent_id: [] for agent_id in AGENT_IDS}

    bars = {
        (str(row["symbol"]), row["trade_date"]): row
        for row in snapshot.adjusted_bars
    }
    outputs = {agent_id: [] for agent_id in AGENT_IDS}

    for symbol in sorted(snapshot.members):
        current = bars.get((symbol, snapshot.decision_date))
        if current is None or current["close"] is None:
            continue

        for agent_id, lookback, multiplier, feature in (
            ("trend", trend_lookback_sessions, 1.0, "adjusted_close_momentum"),
            ("reversal", reversal_lookback_sessions, -1.0, "adjusted_close_reversal"),
        ):
            prior_date = snapshot.trading_dates[decision_index - lookback]
            prior = bars.get((symbol, prior_date))
            if prior is None or prior["close"] is None or prior["close"] <= 0:
                continue
            start_date = prior_date
            score = multiplier * (float(current["close"]) / float(prior["close"]) - 1)
            outputs[agent_id].append(
                AgentSignal(
                    agent_id=agent_id,
                    version="1",
                    symbol=symbol,
                    decision_date=snapshot.decision_date,
                    horizon_sessions=5,
                    score=score,
                    input_cutoff=snapshot.decision_date,
                    evidence=EvidenceRef(
                        feature=feature,
                        start_date=start_date,
                        end_date=snapshot.decision_date,
                        source_refs=tuple(
                            sorted(
                                {
                                    str(prior["source_ref"]),
                                    str(current["source_ref"]),
                                }
                            )
                        ),
                    ),
                )
            )

        return_start_index = decision_index - liquidity_return_sessions
        recent_start_index = decision_index - liquidity_recent_amount_sessions + 1
        reference_start_index = recent_start_index - liquidity_reference_amount_sessions
        recent_dates = snapshot.trading_dates[
            recent_start_index : decision_index + 1
        ]
        reference_dates = snapshot.trading_dates[
            reference_start_index:recent_start_index
        ]
        return_start_date = snapshot.trading_dates[return_start_index]
        return_start = bars.get((symbol, return_start_date))
        amount_rows = [bars.get((symbol, day)) for day in (*reference_dates, *recent_dates)]
        if (
            return_start is None
            or return_start["close"] is None
            or return_start["close"] <= 0
            or any(row is None or row["amount"] is None for row in amount_rows)
        ):
            continue
        reference_amounts = [float(row["amount"]) for row in amount_rows[:len(reference_dates)]]
        recent_amounts = [float(row["amount"]) for row in amount_rows[len(reference_dates):]]
        reference_mean = statistics.fmean(reference_amounts)
        recent_mean = statistics.fmean(recent_amounts)
        if reference_mean <= 0 or recent_mean <= 0:
            continue

        price_return = float(current["close"]) / float(return_start["close"]) - 1
        score = price_return * math.log(recent_mean / reference_mean)
        evidence_rows = [row for row in amount_rows if row is not None]
        outputs["liquidity_confirmation"].append(
            AgentSignal(
                agent_id="liquidity_confirmation",
                version="1",
                symbol=symbol,
                decision_date=snapshot.decision_date,
                horizon_sessions=5,
                score=score,
                input_cutoff=snapshot.decision_date,
                evidence=EvidenceRef(
                    feature="five_day_return_x_log_amount_expansion",
                    start_date=reference_dates[0],
                    end_date=snapshot.decision_date,
                    source_refs=tuple(sorted({str(row["source_ref"]) for row in evidence_rows})),
                ),
            )
        )

    return outputs
