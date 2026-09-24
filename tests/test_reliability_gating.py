import pytest

from adaptive_mas.reliability.gating import reliability_weights


AGENT_IDS = ("trend", "reversal", "liquidity_confirmation")


def test_reliability_gate_stays_equal_until_history_is_matured():
    weights = reliability_weights(
        [{"trend": 0.2, "reversal": -0.1, "liquidity_confirmation": 0.05}],
        AGENT_IDS,
        lookback_windows=20,
        minimum_history_windows=2,
        equal_weight_share=0.5,
    )

    assert weights == pytest.approx({agent_id: 1 / 3 for agent_id in AGENT_IDS})


def test_reliability_gate_uses_past_rank_ic_and_shrinks_toward_equal():
    matured = [
        {"trend": 0.2, "reversal": -0.1, "liquidity_confirmation": -0.05},
        {"trend": 0.1, "reversal": -0.2, "liquidity_confirmation": -0.1},
    ]

    weights = reliability_weights(
        matured,
        AGENT_IDS,
        lookback_windows=2,
        minimum_history_windows=2,
        equal_weight_share=0.5,
    )

    assert sum(weights.values()) == pytest.approx(1.0)
    assert weights["trend"] == pytest.approx(2 / 3)
    assert weights["reversal"] == pytest.approx(1 / 6)
    assert weights["liquidity_confirmation"] == pytest.approx(1 / 6)


def test_reliability_gate_keeps_equal_weights_when_all_past_agents_are_unskilled():
    matured = [
        {"trend": -0.1, "reversal": -0.2, "liquidity_confirmation": -0.05},
        {"trend": -0.2, "reversal": -0.1, "liquidity_confirmation": -0.1},
    ]

    weights = reliability_weights(
        matured,
        AGENT_IDS,
        lookback_windows=2,
        minimum_history_windows=2,
        equal_weight_share=0.5,
    )

    assert weights == pytest.approx({agent_id: 1 / 3 for agent_id in AGENT_IDS})
