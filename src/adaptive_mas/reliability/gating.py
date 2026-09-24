"""Reliability weights from matured, out-of-sample agent Rank IC records."""

import math
import statistics


def reliability_weights(
    matured_rank_ics: list[dict[str, float]],
    agent_ids: tuple[str, ...],
    lookback_windows: int,
    minimum_history_windows: int,
    equal_weight_share: float,
) -> dict[str, float]:
    equal_weight = 1 / len(agent_ids)
    equal_weights = {agent_id: equal_weight for agent_id in agent_ids}
    if len(matured_rank_ics) < minimum_history_windows:
        return equal_weights

    history = matured_rank_ics[-lookback_windows:]
    skills = {
        agent_id: max(statistics.fmean(row[agent_id] for row in history), 0.0)
        for agent_id in agent_ids
    }
    total_skill = math.fsum(skills.values())
    if total_skill == 0:
        return equal_weights

    return {
        agent_id: equal_weight_share * equal_weights[agent_id]
        + (1 - equal_weight_share) * skills[agent_id] / total_skill
        for agent_id in agent_ids
    }
