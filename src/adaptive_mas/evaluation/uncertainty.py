"""Paired moving-block uncertainty intervals for daily portfolio returns."""

import math

import numpy as np


def paired_block_bootstrap(
    strategy_returns: list[float],
    benchmark_returns: list[float],
    block_sessions: int = 20,
    repetitions: int = 1000,
    seed: int = 20260925,
) -> dict[str, float | int]:
    if len(strategy_returns) != len(benchmark_returns) or not strategy_returns:
        raise ValueError("配对区块 Bootstrap 要求两组等长且非空的日收益")
    if block_sessions < 1 or repetitions < 1:
        raise ValueError("Bootstrap 区块长度和重复次数必须为正")
    strategy = np.asarray(strategy_returns, dtype=float)
    benchmark = np.asarray(benchmark_returns, dtype=float)
    if np.any(strategy <= -1) or np.any(benchmark <= -1):
        raise ValueError("日收益必须大于 -100%")

    sample_count = len(strategy)
    blocks_needed = math.ceil(sample_count / block_sessions)
    offsets = np.arange(block_sessions)
    random = np.random.default_rng(seed)
    starts = random.integers(0, sample_count, size=(repetitions, blocks_needed))
    indices = ((starts[:, :, None] + offsets) % sample_count).reshape(repetitions, -1)
    indices = indices[:, :sample_count]
    strategy_samples = strategy[indices]
    benchmark_samples = benchmark[indices]
    strategy_annualized = np.exp(np.log1p(strategy_samples).mean(axis=1) * 250) - 1
    benchmark_annualized = np.exp(np.log1p(benchmark_samples).mean(axis=1) * 250) - 1
    differences = strategy_annualized - benchmark_annualized
    low, high = np.quantile(differences, [0.025, 0.975])
    return {
        "annualized_return_difference_95pct_low": float(low),
        "annualized_return_difference_95pct_high": float(high),
        "probability_annualized_return_exceeds_benchmark": float(
            np.mean(differences > 0)
        ),
        "block_sessions": block_sessions,
        "repetitions": repetitions,
        "seed": seed,
    }
