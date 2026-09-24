"""Transparent long-only target weights and volatility-aware allocation."""

import math
import statistics


def top_fraction_weights(
    scores: dict[str, float], fraction: float, invested_weight: float
) -> dict[str, float]:
    ranked = sorted(scores, key=lambda symbol: (-scores[symbol], symbol))
    if not ranked:
        return {}
    count = max(1, math.ceil(len(ranked) * fraction))
    selected = ranked[:count]
    weight = invested_weight / len(selected)
    return {symbol: weight for symbol in selected}


def apply_weight_and_turnover_limits(
    target: dict[str, float],
    previous: dict[str, float],
    maximum_symbol_weight: float,
    maximum_one_way_turnover: float,
) -> dict[str, float]:
    if any(weight > maximum_symbol_weight for weight in target.values()):
        raise ValueError("组合初始权重超过单只证券上限")
    if not previous:
        return target
    symbols = target.keys() | previous.keys()
    turnover = math.fsum(
        abs(target.get(symbol, 0.0) - previous.get(symbol, 0.0)) for symbol in symbols
    ) / 2
    if turnover <= maximum_one_way_turnover or turnover == 0:
        return target
    blend = maximum_one_way_turnover / turnover
    return {
        symbol: previous.get(symbol, 0.0)
        + blend * (target.get(symbol, 0.0) - previous.get(symbol, 0.0))
        for symbol in sorted(symbols)
        if previous.get(symbol, 0.0)
        + blend * (target.get(symbol, 0.0) - previous.get(symbol, 0.0))
        > 0
    }


def inverse_volatility_weights(
    scores: dict[str, float],
    daily_returns: dict[str, list[float]],
    fraction: float,
    invested_weight: float,
    maximum_symbol_weight: float,
    minimum_observations: int,
) -> dict[str, float]:
    ranked = sorted(scores, key=lambda symbol: (-scores[symbol], symbol))
    eligible = [
        symbol
        for symbol in ranked
        if len(daily_returns.get(symbol, [])) >= minimum_observations
        and statistics.pstdev(daily_returns[symbol]) > 0
    ]
    if not eligible:
        raise ValueError("风险组合没有具备足够历史波动观测的候选证券")
    selected = eligible[: max(1, math.ceil(len(ranked) * fraction))]
    if len(selected) * maximum_symbol_weight < invested_weight:
        selected = eligible[: math.ceil(invested_weight / maximum_symbol_weight)]
    if len(selected) * maximum_symbol_weight < invested_weight:
        raise ValueError("风险组合候选证券不足以满足投资比例和单票上限")
    inverse_volatility = {
        symbol: 1 / statistics.pstdev(daily_returns[symbol]) for symbol in selected
    }
    total = math.fsum(inverse_volatility.values())
    weights = {
        symbol: invested_weight * inverse_volatility[symbol] / total
        for symbol in selected
    }

    # Water-fill any weights above the cap while preserving the invested total.
    capped: dict[str, float] = {}
    remaining = set(weights)
    remaining_total = invested_weight
    while remaining:
        remaining_inverse_vol = math.fsum(inverse_volatility[symbol] for symbol in remaining)
        over_cap = {
            symbol
            for symbol in remaining
            if remaining_total * inverse_volatility[symbol] / remaining_inverse_vol
            > maximum_symbol_weight
        }
        if not over_cap:
            capped.update(
                {
                    symbol: remaining_total * inverse_volatility[symbol] / remaining_inverse_vol
                    for symbol in remaining
                }
            )
            break
        for symbol in over_cap:
            capped[symbol] = maximum_symbol_weight
            remaining_total -= maximum_symbol_weight
        remaining -= over_cap
    return capped
