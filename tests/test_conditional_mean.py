from __future__ import annotations

import pytest

from adaptive_mas.evaluation.conditional_mean import moving_block_mean_bounds


def test_moving_block_mean_bound_preserves_synchronized_components_and_is_repeatable() -> None:
    observations = [
        (0.001 * (index % 4 - 1), 0.0005 * (index % 3 - 1), 0.001 * (index % 4 - 1) + 0.0005 * (index % 3 - 1))
        for index in range(30)
    ]
    first = moving_block_mean_bounds(
        observations,
        block_length=3,
        repetitions=500,
        confidence=0.95,
        seed=20260926,
    )
    second = moving_block_mean_bounds(
        observations,
        block_length=3,
        repetitions=500,
        confidence=0.95,
        seed=20260926,
    )
    assert first == second
    assert first["observations"] == 30
    assert first["effective_contiguous_blocks_per_replicate"] == 10
    assert first["components"][2]["mean"] == pytest.approx(
        first["components"][0]["mean"] + first["components"][1]["mean"]
    )
    assert first["components"][2]["one_sided_lower_bound"] <= first["components"][2]["mean"]


def test_moving_block_mean_bounds_reject_malformed_or_nonfinite_history() -> None:
    with pytest.raises(ValueError, match="至少两行"):
        moving_block_mean_bounds(
            [(0.01,)], block_length=3, repetitions=10, confidence=0.95, seed=1
        )
    with pytest.raises(ValueError, match="等宽"):
        moving_block_mean_bounds(
            [(0.01,), (0.02, 0.03)], block_length=1, repetitions=10, confidence=0.95, seed=1
        )
    with pytest.raises(ValueError, match="非有限"):
        moving_block_mean_bounds(
            [(0.01,), (float("nan"),)], block_length=1, repetitions=10, confidence=0.95, seed=1
        )
