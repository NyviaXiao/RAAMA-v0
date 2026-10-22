"""Evidence challenges and maturity-gated expert performance records."""

from __future__ import annotations

import hashlib
import json
import time
from datetime import date, datetime
from zoneinfo import ZoneInfo


SHANGHAI = ZoneInfo("Asia/Shanghai")


def _source_ids(row: dict[str, object]) -> set[str]:
    return {str(ref["source_record_id"]) for ref in row["evidence"]}


def _judge_one_round(left: dict[str, object], right: dict[str, object]) -> dict[str, object]:
    left_sources = _source_ids(left)
    right_sources = _source_ids(right)
    same_underlying_source = bool(left_sources & right_sources)
    evidence_scopes = {
        str(left.get("evidence_text_scope", "")),
        str(right.get("evidence_text_scope", "")),
    }
    has_full_text = "full_announcement_pdf_text" in evidence_scopes
    has_headline_only = "headline_only_no_announcement_body" in evidence_scopes
    if same_underlying_source and has_full_text and has_headline_only:
        winner = left if left.get("evidence_text_scope") == "full_announcement_pdf_text" else right
        return {
            "outcome": "full_text_evidence_preferred_for_same_announcement",
            "winner_record_id": str(winner["record_id"]),
            "reason": "两方指向同一份公告；含 PDF 文本哈希的全文证据覆盖标题-only 证据范围。该裁决只适用于这条公告，不外推其收益方向正确性。",
            "cited_source_record_ids": sorted(left_sources | right_sources),
        }
    return {
        "outcome": "abstain_unresolved",
        "winner_record_id": None,
        "reason": (
            "单轮审阅确认方向相反，但可核查证据没有构成同一公告的标题/全文覆盖关系；"
            "不同时间或数据类型的材料没有预先验证的优先级，因此裁判弃权。"
        ),
        "cited_source_record_ids": sorted(left_sources | right_sources),
    }


def _dt(value: object) -> datetime:
    return datetime.fromisoformat(str(value)).astimezone(SHANGHAI)


def build_evidence_challenges(outputs: list[dict[str, object]]) -> list[dict[str, object]]:
    """Record only evidence-backed sign conflicts; leave original views unchanged."""
    started = time.perf_counter_ns()
    by_symbol: dict[tuple[str, int], list[dict[str, object]]] = {}
    for row in outputs:
        if row.get("symbol") is None or int(row["direction_score"]) == 0:
            continue
        by_symbol.setdefault((str(row["symbol"]), int(row["horizon_sessions"])), []).append(row)

    challenges = []
    for (symbol, horizon), rows in sorted(by_symbol.items()):
        for index, left in enumerate(rows):
            for right in rows[index + 1:]:
                if int(left["direction_score"]) == int(right["direction_score"]):
                    continue
                evidence_refs = [
                    {
                        "expert": row["expert"],
                        "record_id": row["record_id"],
                        "direction_score": row["direction_score"],
                        "rule": row.get("rule", row.get("rationale")),
                        "evidence": row["evidence"],
                    }
                    for row in (left, right)
                ]
                challenge_id = hashlib.sha256(
                    json.dumps(evidence_refs, ensure_ascii=False, sort_keys=True).encode("utf-8")
                ).hexdigest()
                left_argument = {
                    "record_id": str(left["record_id"]),
                    "expert": str(left["expert"]),
                    "direction_score": int(left["direction_score"]),
                    "argument": str(left.get("rationale", left.get("rule", ""))),
                    "evidence_text_scope": left.get("evidence_text_scope"),
                    "source_record_ids": sorted(_source_ids(left)),
                }
                right_argument = {
                    "record_id": str(right["record_id"]),
                    "expert": str(right["expert"]),
                    "direction_score": int(right["direction_score"]),
                    "argument": str(right.get("rationale", right.get("rule", ""))),
                    "evidence_text_scope": right.get("evidence_text_scope"),
                    "source_record_ids": sorted(_source_ids(right)),
                }
                judgment = _judge_one_round(left, right)
                scope_resolved = judgment["outcome"] == "full_text_evidence_preferred_for_same_announcement"
                challenges.append({
                    "challenge_id": challenge_id,
                    "symbol": symbol,
                    "horizon_sessions": horizon,
                    "conflicting_record_ids": [left["record_id"], right["record_id"]],
                    "evidence_refs": evidence_refs,
                    "status": "evidence_scope_precedence_resolved" if scope_resolved else "evidence_conflict_unresolved",
                    "direction_correctness": "not_established",
                    "discussion_rounds": [{
                        "round": 1,
                        "positive_argument": left_argument if int(left["direction_score"]) > 0 else right_argument,
                        "negative_argument": left_argument if int(left["direction_score"]) < 0 else right_argument,
                        "judge": judgment,
                    }],
                    "judge_verdict": judgment,
                    "original_forecasts_changed": False,
                    "decision_effect": "advisory_only_until_matured_comparison",
                })
    elapsed_ms = (time.perf_counter_ns() - started) / 1_000_000
    for row in challenges:
        row["judge_model"] = "deterministic_one_round_evidence_judge_v1"
        row["judge_cost_cny"] = 0.0
        row["judge_elapsed_ms"] = elapsed_ms / len(challenges) if challenges else 0.0
    return challenges


def update_matured_performance_memory(
    forecasts: list[dict[str, object]],
    outcomes: list[dict[str, object]],
    decision_time: datetime,
) -> list[dict[str, object]]:
    """Score forecasts only against matching outcomes mature by decision_time."""
    if decision_time.tzinfo is None or decision_time.utcoffset() is None:
        raise ValueError("decision_time 必须带时区")
    cutoff = decision_time.astimezone(SHANGHAI)
    by_record_id = {str(row["record_id"]): row for row in outcomes}
    memory = []
    for forecast in forecasts:
        outcome = by_record_id.get(str(forecast["record_id"]))
        if outcome is None:
            continue
        label_available = _dt(outcome["label_available_at"])
        if label_available > cutoff:
            continue
        if int(outcome["horizon_sessions"]) != int(forecast["horizon_sessions"]):
            raise ValueError("绩效记忆结果期限与预测期限不一致")
        realized = float(outcome["forward_return"])
        score = int(forecast["direction_score"])
        memory.append({
            "record_id": str(forecast["record_id"]),
            "expert": str(forecast["expert"]),
            "symbol": forecast["symbol"],
            "prediction_date": str(forecast["prediction_date"]),
            "label_available_at": label_available.isoformat(),
            "memory_as_of": cutoff.isoformat(),
            "horizon_sessions": int(forecast["horizon_sessions"]),
            "directional_return": score * realized,
            "forward_return": realized,
            "direction_hit": score != 0 and (realized > 0) == (score > 0),
            "outcome_ref": outcome["outcome_ref"],
        })
    return memory

def candidate_performance_effects(
    memory: list[dict[str, object]],
    trading_dates: tuple[date, ...],
) -> dict[str, dict[str, object]]:
    """Reuse P11's 20-date nonoverlap and positive 95% lower-bound gate."""
    import statistics

    index_by_date = {day.isoformat(): index for index, day in enumerate(trading_dates)}
    by_expert: dict[tuple[str, int], dict[str, list[float]]] = {}
    for row in memory:
        if float(row["directional_return"]) == 0:
            continue
        key = (str(row["expert"]), int(row["horizon_sessions"]))
        by_expert.setdefault(key, {}).setdefault(
            str(row["prediction_date"]), []
        ).append(float(row["directional_return"]))
    effects = {}
    for (expert, horizon), values_by_date in sorted(by_expert.items()):
        observations = []
        previous_index = None
        for day in sorted(values_by_date):
            current_index = index_by_date[day]
            if previous_index is None or current_index - previous_index >= horizon:
                observations.append(statistics.fmean(values_by_date[day]))
                previous_index = current_index
        count = len(observations)
        mean = statistics.fmean(observations) if observations else 0.0
        standard_error = (
            statistics.stdev(observations) / count**0.5 if count > 1 else None
        )
        lower = mean - 1.96 * standard_error if standard_error is not None else None
        eligible = count >= 20 and lower is not None and lower > 0
        effects[f"{expert}:{horizon}s"] = {
            "matured_nonoverlapping_decision_dates": count,
            "mean_directional_active_return": mean if count else None,
            "normal_95pct_lower_bound": lower,
            "minimum_nonoverlapping_decision_dates": 20,
            "candidate_return_adjustment": lower if eligible else 0.0,
            "status": "candidate_effect_eligible" if eligible else "pending_or_no_positive_lower_bound",
        }
    return effects
