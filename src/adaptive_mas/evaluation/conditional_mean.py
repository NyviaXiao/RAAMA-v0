"""Block-bootstrap bounds for point-in-time portfolio forecast bias."""

from __future__ import annotations

import math
import random
import statistics
from collections.abc import Sequence


def moving_block_mean_bounds(
    observations: Sequence[Sequence[float]],
    *,
    block_length: int,
    repetitions: int,
    confidence: float,
    seed: int,
) -> dict[str, object]:
    """Estimate mean component biases with a circular moving-block bootstrap.

    Rows are synchronized portfolio-level observations, such as market,
    relative, and total forecast residuals. Sampling contiguous blocks retains
    serial dependence; it does not assume daily labels or assets are independent.
    """
    if len(observations) < 2 or block_length < 1 or repetitions < 1:
        raise ValueError("区块均值估计需要至少两行及正的区块长度和重复次数")
    if not 0 < confidence < 1:
        raise ValueError("区间置信度必须处于 (0, 1)")
    width = len(observations[0])
    if width < 1 or any(len(row) != width for row in observations):
        raise ValueError("区块均值输入必须是等宽非空行")
    values = [[float(value) for value in row] for row in observations]
    if any(not math.isfinite(value) for row in values for value in row):
        raise ValueError("区块均值输入包含非有限观测")

    count = len(values)
    blocks_per_sample = math.ceil(count / block_length)
    generator = random.Random(seed)
    bootstrapped: list[list[float]] = [[] for _ in range(width)]
    for _ in range(repetitions):
        sample: list[list[float]] = []
        for _ in range(blocks_per_sample):
            start = generator.randrange(count)
            sample.extend(values[(start + offset) % count] for offset in range(block_length))
        sample = sample[:count]
        for column in range(width):
            bootstrapped[column].append(
                math.fsum(row[column] for row in sample) / count
            )

    tail = (1 - confidence) / 2

    def quantile(sample: list[float], probability: float) -> float:
        ordered = sorted(sample)
        position = probability * (len(ordered) - 1)
        lower = math.floor(position)
        upper = math.ceil(position)
        if lower == upper:
            return ordered[lower]
        fraction = position - lower
        return ordered[lower] * (1 - fraction) + ordered[upper] * fraction

    means = [statistics.fmean(row[column] for row in values) for column in range(width)]
    central_intervals = [
        {
            "mean": means[column],
            "central_interval_low": quantile(bootstrapped[column], tail),
            "central_interval_high": quantile(bootstrapped[column], 1 - tail),
            "one_sided_lower_bound": quantile(bootstrapped[column], 1 - confidence),
        }
        for column in range(width)
    ]
    return {
        "observations": count,
        "bootstrap_block_length": block_length,
        "effective_contiguous_blocks_per_replicate": blocks_per_sample,
        "repetitions": repetitions,
        "confidence": confidence,
        "seed": seed,
        "components": central_intervals,
        "method": "circular moving-block bootstrap of synchronized, non-overlapping horizon portfolio residual cohorts",
    }
