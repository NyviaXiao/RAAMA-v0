"""Seed text components that GEPA evolves as a genuine multi-component candidate."""

import json
from typing import Any


MAX_COMPONENT_UTF8_BYTES = 8_000
MAX_CANDIDATE_UTF8_BYTES = 32_000

_LEGACY_TEXT_COMPONENTS = {
    "market_prompt",
    "fundamental_prompt",
    "manager_prompt",
    "context_policy",
    "handoff_schema",
    "procedural_experience",
}
_POLICY_KEYS = {
    "channels",
    "max_searches",
    "candidate_limit",
    "symbol_limit",
    "evidence_limit",
    "rrf_rank_constant",
    "reranker_profile",
    "table_expansion_limit",
    "parent_context_mode",
    "context_char_limit",
    "context_source_limit",
    "challenge_policy",
    "stop_policy",
    "memory_refs",
}
_ALLOWED_CHANNELS = {
    "structured_facts", "bm25", "toc_page", "dense", "vision", "source_relations",
}


def _compact_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def default_research_policy() -> dict[str, object]:
    return {
        "channels": ["structured_facts", "bm25"],
        "max_searches": 8,
        "candidate_limit": 2,
        "symbol_limit": 4,
        "evidence_limit": 4,
        "rrf_rank_constant": 60,
        "reranker_profile": "configured",
        "table_expansion_limit": 12,
        "parent_context_mode": "reviewed_parent_and_child",
        "context_char_limit": 1_200,
        "context_source_limit": 2,
        "challenge_policy": "mandatory_correction_search",
        "stop_policy": "coverage_then_bounded_search",
        "memory_refs": ["procedural_experience"],
    }


def parse_research_policy(value: str | None = None) -> dict[str, object]:
    """Parse the non-executable, bounded strategy package used by the harness."""
    if value is None:
        policy = default_research_policy()
    else:
        if not isinstance(value, str):
            raise ValueError("research policy must be a JSON string")

        def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
            result: dict[str, Any] = {}
            for key, item in pairs:
                if key in result:
                    raise ValueError("research policy cannot contain duplicate keys")
                result[key] = item
            return result

        try:
            policy = json.loads(value, object_pairs_hook=reject_duplicate_keys)
        except (json.JSONDecodeError, TypeError) as exc:
            raise ValueError("research policy must be valid JSON") from exc
    if not isinstance(policy, dict) or set(policy) != _POLICY_KEYS:
        raise ValueError("research policy must contain exactly the declared bounded fields")
    channels = policy["channels"]
    if (
        not isinstance(channels, list)
        or not 1 <= len(channels) <= 3
        or any(not isinstance(channel, str) or channel not in _ALLOWED_CHANNELS for channel in channels)
        or len(set(channels)) != len(channels)
        or not set(channels).intersection({"structured_facts", "bm25"})
    ):
        raise ValueError("research policy channels must be unique, bounded, and include a primary channel")
    bounded_integers = {
        "max_searches": (2, 8),
        "candidate_limit": (1, 2),
        "symbol_limit": (1, 4),
        "evidence_limit": (2, 32),
        "rrf_rank_constant": (20, 100),
        "table_expansion_limit": (0, 12),
        "context_char_limit": (400, 1_200),
        "context_source_limit": (1, 2),
    }
    for key, (minimum, maximum) in bounded_integers.items():
        item = policy[key]
        if isinstance(item, bool) or not isinstance(item, int) or not minimum <= item <= maximum:
            raise ValueError(f"research policy {key} must be an integer within {minimum}..{maximum}")
    if policy["reranker_profile"] not in ("configured", "disabled"):
        raise ValueError("research policy reranker_profile is unsupported")
    if policy["parent_context_mode"] not in ("reviewed_parent_and_child", "source_row_only"):
        raise ValueError("research policy parent_context_mode is unsupported")
    if policy["challenge_policy"] != "mandatory_correction_search":
        raise ValueError("research policy cannot disable the mandatory correction search")
    if policy["stop_policy"] != "coverage_then_bounded_search":
        raise ValueError("research policy cannot change the bounded coverage stop contract")
    if policy["memory_refs"] != ["procedural_experience"]:
        raise ValueError("research policy memory_refs must use the active procedural experience only")
    if policy["parent_context_mode"] == "source_row_only" and policy["table_expansion_limit"] != 0:
        raise ValueError("source_row_only policy must set table_expansion_limit to zero")
    return {**policy, "channels": list(channels), "memory_refs": list(policy["memory_refs"])}


def seed_candidate() -> dict[str, str]:
    return {
        "market_prompt": (
            "You are the market specialist. Call read_market_signals once to inspect the supplied "
            "point-in-time public market cache; it returns bounded trend, reversal, and volume-price "
            "groups. Focus on a few well-supported symbols. Return exact "
            "observations and cite their evidence_id. "
            "These are indicators, not calibrated return forecasts. Do not infer fundamentals, "
            "account holdings, private predictions, or orders. Finish with submit_report."
        ),
        "fundamental_prompt": (
            "You are the fundamental specialist. Use search_financial_facts_and_announcements "
            "to find the most relevant locally cached public financial facts, report periods, "
            "and announcement evidence. Use read_cached_source only when a title or structured "
            "card is insufficient. Report title-only evidence explicitly as title-only. Preserve "
            "publication and availability times, cite exact evidence_id values, and keep the "
            "report to at most eight findings and two exact observations per finding. Do not "
            "invent facts or expected returns. Finish with submit_report."
        ),
        "manager_prompt": (
            "You are the research manager. First state a concise goal with plan_research. You may "
            "make at most two query_specialist actions, each targeting one cited claim, symbol, and "
            "question. A risk action must select event, industry, or model risk; use only locally "
            "available, cited evidence and report unavailable dimensions explicitly. Its findings "
            "are research context and never replace deterministic hard-risk checks. After each result, choose the next action from the new evidence or stop and "
            "finalize; never repeat an action and stop when no new cited evidence is found. "
            "Choose only among market, fundamental, news, market_attention, and risk roles. News uses "
            "on-host fixed text-rule output without sending raw news text to a model; attention is a "
            "source-reported index only and is not discussion-text sentiment. If a channel is unavailable, "
            "report that honestly. Include an evidence-based stopping reason in no_query_reason. Finish "
            "with finalize_research. "
            "Rank at most twelve symbols supported by specialist citations. Rank only "
            "symbols supported by specialist citations; this ranking filters the optimizer's "
            "candidate universe while all expected returns continue to come from frozen local "
            "calibration. Never create a return estimate, account action, or unsourced fact."
        ),
        "context_policy": (
            "Context policy: enforce the as_of cutoff on both published_at and available_at; "
            "prefer evidence relevant to the question; read long documents only on demand; "
            "distinguish reporting period from publication time; retain exact evidence IDs."
        ),
        "handoff_schema": (
            "Handoff policy: pass a concise summary, findings grouped by six-digit symbol, exact "
            "evidence IDs, exact observed fields and values, report periods, and unresolved items. "
            "Do not copy full source text into another role. Full evidence remains locally readable "
            "by evidence ID."
        ),
        "procedural_experience": (
            "Procedural experience: check contrary evidence and the source's content scope; "
            "distinguish title-only from extracted body text; state uncertainty where source "
            "vintages or reporting dates are not verified. This is cross-task process guidance, "
            "not task-specific memory."
        ),
        "research_policy_json": _compact_json(default_research_policy()),
    }


def validate_candidate(candidate: object) -> dict[str, str]:
    expected = set(seed_candidate())
    if not isinstance(candidate, dict) or frozenset(candidate) not in {frozenset(expected), frozenset(_LEGACY_TEXT_COMPONENTS)}:
        raise ValueError("candidate must contain exactly the seven declared strategy components")
    if set(candidate) == _LEGACY_TEXT_COMPONENTS:
        candidate = {**candidate, "research_policy_json": _compact_json(default_research_policy())}
    if any(not isinstance(value, str) or not value.strip() for value in candidate.values()):
        raise ValueError("every evolvable text component must be a nonempty string")
    policy = parse_research_policy(candidate["research_policy_json"])
    candidate = {**candidate, "research_policy_json": _compact_json(policy)}
    encoded_sizes = {key: len(value.encode("utf-8")) for key, value in candidate.items()}
    if any(size > MAX_COMPONENT_UTF8_BYTES for size in encoded_sizes.values()):
        raise ValueError("candidate component exceeds its fixed UTF-8 size bound")
    if sum(encoded_sizes.values()) > MAX_CANDIDATE_UTF8_BYTES:
        raise ValueError("candidate exceeds its fixed total UTF-8 size bound")
    if any("\x00" in value for value in candidate.values()):
        raise ValueError("candidate text cannot contain null bytes")
    return {key: candidate[key] for key in sorted(expected)}
