"""Prequential diagnostics for agents, fusion, and a gross top-decile portfolio."""

import math
import statistics
from collections import defaultdict

from adaptive_mas.agents.quantitative import AGENT_IDS, quantitative_agents
from adaptive_mas.data.research import ResearchData
from adaptive_mas.reliability.gating import reliability_weights
from adaptive_mas.targets.five_session import five_session_forward_returns


METHODS = (*AGENT_IDS, "equal_fusion", "adaptive_fusion")


def average_ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        rank = (start + 1 + end) / 2
        for position in order[start:end]:
            ranks[position] = rank
        start = end
    return ranks


def rank_ic(scores: list[float], outcomes: list[float]) -> float:
    if len(scores) != len(outcomes) or len(scores) < 2:
        raise ValueError("Rank IC 需要至少两组一一对应的专家分数和五日结果")
    score_ranks = average_ranks(scores)
    outcome_ranks = average_ranks(outcomes)
    score_mean = statistics.fmean(score_ranks)
    outcome_mean = statistics.fmean(outcome_ranks)
    covariance = math.fsum(
        (left - score_mean) * (right - outcome_mean)
        for left, right in zip(score_ranks, outcome_ranks, strict=True)
    )
    score_variance = math.fsum((value - score_mean) ** 2 for value in score_ranks)
    outcome_variance = math.fsum((value - outcome_mean) ** 2 for value in outcome_ranks)
    return covariance / math.sqrt(score_variance * outcome_variance)


def percentile_scores(scores_by_symbol: dict[str, float]) -> dict[str, float]:
    symbols = sorted(scores_by_symbol)
    ranks = average_ranks([scores_by_symbol[symbol] for symbol in symbols])
    return {
        symbol: (rank - (len(symbols) + 1) / 2) / len(symbols)
        for symbol, rank in zip(symbols, ranks, strict=True)
    }


def _evaluate_pending(
    research: ResearchData,
    pending: dict[str, object],
    top_decile_fraction: float,
) -> dict[str, object]:
    decision_date = pending["decision_date"]
    labels = five_session_forward_returns(research, [decision_date])
    outcomes = {str(row["symbol"]): float(row["forward_return_5s"]) for row in labels}
    scores = pending["scores"]
    shared_symbols = sorted(outcomes.keys() & set(scores[AGENT_IDS[0]]))
    for agent_id in AGENT_IDS[1:]:
        shared_symbols = sorted(set(shared_symbols) & scores[agent_id].keys())

    methods = {
        agent_id: {symbol: scores[agent_id][symbol] for symbol in shared_symbols}
        for agent_id in AGENT_IDS
    }
    percentile_by_agent = {
        agent_id: percentile_scores(values) for agent_id, values in methods.items()
    }
    methods["equal_fusion"] = {
        symbol: statistics.fmean(percentile_by_agent[agent_id][symbol] for agent_id in AGENT_IDS)
        for symbol in shared_symbols
    }
    methods["adaptive_fusion"] = {
        symbol: math.fsum(
            pending["weights"][agent_id] * percentile_by_agent[agent_id][symbol]
            for agent_id in AGENT_IDS
        )
        for symbol in shared_symbols
    }

    ordered_outcomes = [outcomes[symbol] for symbol in shared_symbols]
    per_method = {}
    for method, score_by_symbol in methods.items():
        ordered_scores = [score_by_symbol[symbol] for symbol in shared_symbols]
        ranked = sorted(shared_symbols, key=lambda symbol: (-score_by_symbol[symbol], symbol))
        holding_count = max(1, math.ceil(len(ranked) * top_decile_fraction))
        per_method[method] = {
            "rank_ic": rank_ic(ordered_scores, ordered_outcomes),
            "top_decile_gross_return": statistics.fmean(
                outcomes[symbol] for symbol in ranked[:holding_count]
            ),
        }

    signal_correlations = {}
    for left_index, left_agent in enumerate(AGENT_IDS):
        for right_agent in AGENT_IDS[left_index + 1 :]:
            signal_correlations[f"{left_agent}:{right_agent}"] = rank_ic(
                [methods[left_agent][symbol] for symbol in shared_symbols],
                [methods[right_agent][symbol] for symbol in shared_symbols],
            )

    index = {row["trade_date"]: row for row in research.index_bars}
    date_index = research.trading_dates.index(decision_date)
    entry_date = research.trading_dates[date_index + 1]
    exit_date = research.trading_dates[date_index + 5]
    benchmark_return = float(index[exit_date]["close"]) / float(index[entry_date]["open"]) - 1
    return {
        "decision_date": decision_date.isoformat(),
        "label_available_at": exit_date.isoformat(),
        "cross_section": len(shared_symbols),
        "benchmark_gross_return": benchmark_return,
        "weights_used": pending["weights"],
        "reliability_history_windows": pending["reliability_history_windows"],
        "signal_rank_correlations": signal_correlations,
        "methods": per_method,
    }


def run_agent_research(
    research: ResearchData,
    evaluation_step_sessions: int,
    reliability_lookback_windows: int,
    reliability_minimum_history_windows: int,
    equal_weight_share: float,
    top_decile_fraction: float,
    trend_lookback_sessions: int,
    reversal_lookback_sessions: int,
    liquidity_return_lookback_sessions: int,
    liquidity_recent_amount_sessions: int,
    liquidity_reference_amount_sessions: int,
) -> dict[str, object]:
    first_decision_index = max(
        trend_lookback_sessions,
        reversal_lookback_sessions,
        liquidity_return_lookback_sessions,
        liquidity_recent_amount_sessions + liquidity_reference_amount_sessions - 1,
    )
    decision_dates = [
        research.trading_dates[index]
        for index in range(
            first_decision_index,
            len(research.trading_dates) - 5,
            evaluation_step_sessions,
        )
    ]
    if not decision_dates:
        raise ValueError("BaoStock 研究日期不足以形成五日 Agent 评价窗口")
    if len(decision_dates) <= reliability_minimum_history_windows:
        raise ValueError("成熟的五日评价窗口不足以同时完成门控暖启动和其后验收")

    matured_rank_ics: list[dict[str, float]] = []
    observations = []
    signal_records = []
    pending = None
    for decision_date in decision_dates:
        if pending is not None:
            observation = _evaluate_pending(research, pending, top_decile_fraction)
            matured_rank_ics.append(
                {
                    agent_id: observation["methods"][agent_id]["rank_ic"]
                    for agent_id in AGENT_IDS
                }
            )
            observations.append(observation)

        weights = reliability_weights(
            matured_rank_ics,
            AGENT_IDS,
            reliability_lookback_windows,
            reliability_minimum_history_windows,
            equal_weight_share,
        )
        snapshot = research.snapshot_at(decision_date)
        agent_outputs = quantitative_agents(
            snapshot,
            trend_lookback_sessions,
            reversal_lookback_sessions,
            liquidity_return_lookback_sessions,
            liquidity_recent_amount_sessions,
            liquidity_reference_amount_sessions,
        )
        signal_records.extend(
            {
                "agent_id": signal.agent_id,
                "version": signal.version,
                "symbol": signal.symbol,
                "decision_date": signal.decision_date.isoformat(),
                "horizon_sessions": signal.horizon_sessions,
                "score": signal.score,
                "input_cutoff": signal.input_cutoff.isoformat(),
                "evidence": {
                    "feature": signal.evidence.feature,
                    "start_date": signal.evidence.start_date.isoformat(),
                    "end_date": signal.evidence.end_date.isoformat(),
                    "source_refs": signal.evidence.source_refs,
                },
            }
            for agent_id in AGENT_IDS
            for signal in agent_outputs[agent_id]
        )
        pending = {
            "decision_date": decision_date,
            "scores": {
                agent_id: {signal.symbol: signal.score for signal in agent_outputs[agent_id]}
                for agent_id in AGENT_IDS
            },
            "weights": weights,
            "reliability_history_windows": len(matured_rank_ics),
        }

    if pending is not None:
        observations.append(_evaluate_pending(research, pending, top_decile_fraction))

    def summarize_method(rows: list[dict[str, object]], method: str) -> dict[str, object]:
        method_rows = [row["methods"][method] for row in rows]
        return {
            "observations": len(method_rows),
            "mean_rank_ic": statistics.fmean(row["rank_ic"] for row in method_rows),
            "median_rank_ic": statistics.median(row["rank_ic"] for row in method_rows),
            "positive_rank_ic_fraction": statistics.fmean(
                row["rank_ic"] > 0 for row in method_rows
            ),
            "mean_top_decile_gross_return": statistics.fmean(
                row["top_decile_gross_return"] for row in method_rows
            ),
        }

    by_method = {
        method: summarize_method(observations, method) for method in METHODS
    }
    post_warmup = [
        row
        for row in observations
        if row["reliability_history_windows"] >= reliability_minimum_history_windows
    ]
    pair_names = tuple(observations[0]["signal_rank_correlations"])
    signal_correlations = {
        pair: statistics.fmean(row["signal_rank_correlations"][pair] for row in observations)
        for pair in pair_names
    }
    ic_error_correlations = {
        f"{left}:{right}": rank_ic(
            [row["methods"][left]["rank_ic"] for row in observations],
            [row["methods"][right]["rank_ic"] for row in observations],
        )
        for left_index, left in enumerate(AGENT_IDS)
        for right in AGENT_IDS[left_index + 1 :]
    }
    mean_weights = {
        agent_id: statistics.fmean(row["weights_used"][agent_id] for row in observations)
        for agent_id in AGENT_IDS
    }
    adaptive_mean_weights = {
        agent_id: statistics.fmean(row["weights_used"][agent_id] for row in post_warmup)
        for agent_id in AGENT_IDS
    }
    by_year: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in observations:
        by_year[row["decision_date"][:4]].append(row)

    by_method["csi300_benchmark"] = {
        "observations": len(observations),
        "mean_five_session_gross_return": statistics.fmean(
            row["benchmark_gross_return"] for row in observations
        ),
    }
    return {
        "protocol": {
            "decision": "after close; next-session open entry; fifth-session close exit",
            "evaluation_step_sessions": evaluation_step_sessions,
            "non_overlapping_targets": True,
            "reliability_update": "previous matured per-window Rank IC only",
            "reliability_lookback_windows": reliability_lookback_windows,
            "reliability_minimum_history_windows": reliability_minimum_history_windows,
            "equal_weight_share": equal_weight_share,
            "top_decile_fraction": top_decile_fraction,
            "transaction_costs_included": False,
            "source_rows_used_for_agent_input": "adjusted_bars only",
        },
        "summary": {
            "decision_windows": len(decision_dates),
            "evaluated_windows": len(observations),
            "adaptive_windows_after_warmup": max(
                0, len(observations) - reliability_minimum_history_windows
            ),
            "post_warmup_observations": len(post_warmup),
            "mean_common_cross_section": statistics.fmean(
                row["cross_section"] for row in observations
            ),
            "by_method": by_method,
            "post_warmup_by_method": {
                method: summarize_method(post_warmup, method) for method in METHODS
            },
            "post_warmup_csi300_benchmark": {
                "observations": len(post_warmup),
                "mean_five_session_gross_return": statistics.fmean(
                    row["benchmark_gross_return"] for row in post_warmup
                ),
            },
            "mean_agent_weights": mean_weights,
            "mean_adaptive_weights_after_warmup": adaptive_mean_weights,
            "mean_pairwise_signal_rank_correlation": signal_correlations,
            "pairwise_rank_ic_series_correlation": ic_error_correlations,
            "by_year": {
                year: {
                    "observations": len(rows),
                    "by_method": {
                        method: summarize_method(rows, method) for method in METHODS
                    },
                }
                for year, rows in sorted(by_year.items())
            },
        },
        "signal_records": signal_records,
        "observations": observations,
    }
