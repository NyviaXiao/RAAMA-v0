from __future__ import annotations

import json
from collections import Counter
from datetime import date, timedelta

import pytest

from adaptive_mas.agents.harness import CallBudget, run_research_harness
from adaptive_mas.agents.harness_prompts import seed_candidate
from adaptive_mas.data.research import MembershipSnapshot, ResearchData


AS_OF = date(2026, 9, 1)
SYMBOL = "600000"
RAW_NEWS_FIXTURE = "PRIVATE_RAW_NEWS_FIXTURE_NEVER_EGRESS"


def _research_data() -> ResearchData:
    dates = [AS_OF - timedelta(days=offset) for offset in (2, 1, 0)]
    bars = []
    for index, day in enumerate(dates):
        bars.append({
            "symbol": "sh.600000",
            "trade_date": day,
            "available_at": day,
            "close": 10.0 + index,
            "amount": 1_000.0 * (index + 1),
            "source_ref": f"fixture-market-{day.isoformat()}",
        })
    membership = MembershipSnapshot(
        event_time=dates[0],
        available_at=dates[0],
        members=frozenset({"sh.600000"}),
        source_ref="fixture-membership",
    )
    return ResearchData(dates, bars, [], [], [membership])


def _public_snapshot() -> dict[str, object]:
    published_at = "2026-09-01T14:00:00+08:00"
    available_at = "2026-09-01T14:00:00+08:00"
    fetched_at = "2026-09-01T14:05:00+08:00"
    return {
        "data_mode": "captured_public_cache_cutoff",
        "strict_historical_pit_verified": False,
        "p9_outputs": [{
            "expert": "announcement_title_rule_v1",
            "symbol": "sh.600000",
            "evidence": [{
                "source_record_id": "fixture-announcement-1",
                "dataset": "announcement",
                "title": "公司公告：经营情况更新",
                "published_at": published_at,
                "available_at": available_at,
                "fetched_at": fetched_at,
                "content_sha256": "fixture-announcement-sha256",
                "vintage_status": "captured_fixture",
                "vintage_verified": True,
            }],
        }],
        "p10_outputs": [{
            "expert": "cls_forward_rule_v1",
            "symbol": "sh.600000",
            "event_type": "market_news_mention",
            "prediction_date": AS_OF.isoformat(),
            "event_time": published_at,
            "available_at": available_at,
            "direction_score": 1,
            "rule": "cls_text_rule_positive",
            "evidence_text_scope": "captured_cls_news_content",
            "evidence": [{
                "source_record_id": "fixture-news-1",
                "title": "Public-news fixture title",
                "content": RAW_NEWS_FIXTURE,
                "published_at": published_at,
                "available_at": available_at,
                "fetched_at": fetched_at,
                "content_sha256": "fixture-news-sha256",
                "vintage_status": "captured_fixture",
                "vintage_verified": True,
            }],
        }],
        "attention_outputs": [{
            "expert": "market_attention_index_v1",
            "symbol": SYMBOL,
            "prediction_date": AS_OF.isoformat(),
            "event_time": published_at,
            "available_at": available_at,
            "attention_index": 120.0,
            "attention_index_change": 10.0,
            "interpretation_scope": "source_reported_market_attention_only_not_discussion_text_sentiment",
            "evidence": [{
                "source_record_id": "fixture-attention-1",
                "dataset": "attention_index",
                "trade_date": AS_OF.isoformat(),
                "published_at": published_at,
                "available_at": available_at,
                "fetched_at": fetched_at,
                "content_sha256": "fixture-attention-sha256",
                "vintage_status": "captured_fixture",
                "vintage_verified": True,
            }],
        }],
    }


def _tool_call(call_id: str, name: str, arguments: dict[str, object]) -> dict[str, object]:
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments, ensure_ascii=False)},
    }


def _task_from_messages(messages: list[dict[str, object]]) -> dict[str, object]:
    for message in messages:
        if message.get("role") == "user":
            try:
                value = json.loads(str(message.get("content", "")))
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict) and value.get("kind") != "completed_steps_summary":
                return value
    return {}


def _latest_tool_result(messages: list[dict[str, object]], tool_name: str) -> dict[str, object]:
    for message in reversed(messages):
        if message.get("role") == "tool" and message.get("name") == tool_name:
            value = json.loads(str(message["content"]))
            if isinstance(value, dict):
                return value
    raise AssertionError(f"no result for {tool_name}")


class HarnessFlowClient:
    model = "fixture-only"

    def __init__(self, mode: str) -> None:
        self.mode = mode
        self.calls: list[dict[str, object]] = []
        self.role_counts: Counter[str] = Counter()
        self.first_query: dict[str, object] | None = None

    def complete(self, messages, tools, **kwargs):
        role = str(kwargs["role"])
        self.calls.append({"role": role, "messages": messages})
        self.role_counts[role] += 1
        call_number = self.role_counts[role]

        if role == "market" and call_number == 1:
            response_calls = [_tool_call("market-search", "read_market_signals", {
                "symbols": [SYMBOL], "limit_per_signal": 1,
            })]
        elif role == "market":
            source_result = _latest_tool_result(messages, "read_market_signals")
            row = source_result["findings"][0]
            response_calls = [_tool_call("market-submit", "submit_report", {
                "summary": "The market indicator is cited as an indicator only.",
                "findings": [{
                    "symbol": SYMBOL,
                    "conclusion": "The market indicator merits a targeted check.",
                    "evidence_ids": [row["evidence_id"]],
                    "observations": [{
                        "evidence_id": row["evidence_id"],
                        "field": row["metric"],
                        "value": row["value"],
                        "unit": row["unit"],
                        "report_period": None,
                        "period_start": row["period_start"],
                        "period_end": row["period_end"],
                    }],
                    "uncertainties": [],
                }],
                "unresolved_items": [],
            })]
        elif role == "fundamental" and call_number == 1:
            response_calls = [_tool_call("fundamental-search", "search_financial_facts_and_announcements", {
                "question": "经营公告",
                "symbols": [SYMBOL],
                "limit": 1,
            })]
        elif role == "fundamental":
            search = _latest_tool_result(messages, "search_financial_facts_and_announcements")
            evidence_id = search["evidence_refs"][0]["evidence_id"]
            response_calls = [_tool_call("fundamental-submit", "submit_report", {
                "summary": "One cached public announcement title is available.",
                "findings": [{
                    "symbol": SYMBOL,
                    "conclusion": "The announcement title should be checked against other evidence.",
                    "evidence_ids": [evidence_id],
                    "observations": [],
                    "uncertainties": ["Only the cached title is available."],
                }],
                "unresolved_items": [],
            })]
        elif role.startswith("market_directed_query_"):
            task = _task_from_messages(messages)
            finding = next(
                row for row in task["prior_public_handoff"]["findings"]
                if row["role"] == "market"
            )
            response_calls = [_tool_call("directed-market-submit", "submit_report", {
                "summary": "The existing market source did not add a new cited record.",
                "findings": [{
                    "symbol": SYMBOL,
                    "conclusion": finding["conclusion"],
                    "evidence_ids": finding["evidence_ids"],
                    "observations": finding["observations"],
                    "uncertainties": ["No new source was returned by this directed check."],
                }],
                "unresolved_items": [],
            })]
        elif role == "research_manager":
            task = _task_from_messages(messages)
            manager_turn = call_number
            if manager_turn == 1:
                market_finding = task["specialist_handoffs"]["market"]["findings"][0]
                self.first_query = (
                    {
                        "role": "risk",
                        "risk_dimension": self.mode.removeprefix("risk_"),
                        "symbol": SYMBOL,
                        "question": "Check sourced risk evidence for the selected symbol.",
                        "target_evidence_id": market_finding["evidence_ids"][0],
                        "target_claim": market_finding["conclusion"],
                    }
                    if self.mode.startswith("risk_") else {
                        "role": "news",
                        "symbol": SYMBOL,
                        "question": "Check whether a locally captured news event supports this claim.",
                        "target_evidence_id": market_finding["evidence_ids"][0],
                        "target_claim": market_finding["conclusion"],
                    }
                )
                response_calls = [
                    _tool_call("manager-plan", "plan_research", {
                        "goal": "Check one material public claim and stop when no new evidence appears.",
                    }),
                    _tool_call("manager-query-1", "query_specialist", self.first_query),
                ]
            elif manager_turn == 2:
                news_result = _latest_tool_result(messages, "query_specialist")
                if self.mode.startswith("risk_"):
                    risk_ids = news_result.get("newly_cited_evidence_ids", [])
                    market_ids = task["specialist_handoffs"]["market"]["findings"][0]["evidence_ids"]
                    response_calls = [_tool_call("manager-final", "finalize_research", {
                        "summary": "The manager checked the selected synthetic risk research dimension.",
                        "priority_symbols": [{"symbol": SYMBOL, "evidence_ids": list(dict.fromkeys(market_ids + risk_ids))[:4]}],
                        "unresolved_items": [] if risk_ids else ["The selected risk research source was unavailable."],
                        "no_query_reason": "Stop after the selected bounded, sourced risk review.",
                    })]
                    return {"choices": [{"message": {"tool_calls": response_calls}}]}
                if self.mode == "unavailable":
                    market_ids = [
                        evidence_id
                        for finding in task["specialist_handoffs"]["market"]["findings"]
                        for evidence_id in finding["evidence_ids"]
                    ]
                    response_calls = [_tool_call("manager-final", "finalize_research", {
                        "summary": "The requested local channel was unavailable, so the evidence check stopped.",
                        "priority_symbols": [{"symbol": SYMBOL, "evidence_ids": market_ids[:1]}],
                        "unresolved_items": ["No eligible captured public-news source was available."],
                        "no_query_reason": "Stop because the selected local public-news source was unavailable.",
                    })]
                else:
                    news_finding = news_result["report"]["findings"][0]
                    if self.mode == "duplicate":
                        query = dict(self.first_query or {})
                    elif self.mode == "no_gain":
                        query = {
                            "role": "market",
                            "symbol": SYMBOL,
                            "question": "Recheck the market indicator against the news classification.",
                            "target_evidence_id": news_finding["evidence_ids"][0],
                            "target_claim": news_finding["conclusion"],
                        }
                    else:
                        query = {
                            "role": "market_attention",
                            "symbol": SYMBOL,
                            "question": "Check the source-reported attention index after the news evidence.",
                            "target_evidence_id": news_finding["evidence_ids"][0],
                            "target_claim": news_finding["conclusion"],
                        }
                    response_calls = [_tool_call("manager-query-2", "query_specialist", query)]
            else:
                initial_ids = [
                    evidence_id
                    for report in task["specialist_handoffs"].values()
                    for finding in report["findings"]
                    for evidence_id in finding["evidence_ids"]
                ]
                for message in messages:
                    if message.get("role") != "tool" or message.get("name") != "query_specialist":
                        continue
                    result = json.loads(str(message["content"]))
                    for finding in result.get("report", {}).get("findings", []):
                        initial_ids.extend(finding["evidence_ids"])
                response_calls = [_tool_call("manager-final", "finalize_research", {
                    "summary": "Follow-up actions were bounded and grounded in cited public evidence.",
                    "priority_symbols": [{"symbol": SYMBOL, "evidence_ids": list(dict.fromkeys(initial_ids))[:4]}],
                    "unresolved_items": [],
                    "no_query_reason": "Stop after the bounded evidence check; preserve unresolved interpretation.",
                })]
        else:
            raise AssertionError(f"unexpected model role: {role}")

        return {"choices": [{"message": {"tool_calls": response_calls}}]}


@pytest.mark.parametrize(
    ("mode", "expected_roles", "expected_stop_code"),
    [
        ("followup", ["news", "market_attention"], "manager_chose_to_stop"),
        ("duplicate", ["news"], "duplicate_action_stopped"),
        ("no_gain", ["news", "market"], "no_new_cited_evidence_stopped"),
            ("unavailable", ["news"], "no_new_cited_evidence_stopped"),
    ],
)
def test_manager_changes_actions_from_new_evidence_and_stops_on_repeat_or_no_gain(
    mode: str,
    expected_roles: list[str],
    expected_stop_code: str,
) -> None:
    client = HarnessFlowClient(mode)
    public_snapshot = _public_snapshot()
    if mode == "unavailable":
        public_snapshot["p10_outputs"] = []
    result = run_research_harness(
        _research_data(),
        AS_OF,
        public_snapshot,
        seed_candidate(),
        client,
        CallBudget(max_calls=10),
        config={
            "feature_trend_sessions": 1,
            "feature_reversal_sessions": 1,
            "feature_liquidity_return_sessions": 1,
            "feature_recent_amount_sessions": 1,
            "feature_reference_amount_sessions": 1,
            "prediction_horizon_sessions": 3,
        },
    )

    loop = result["research_loop"]
    action_roles = [row["role_id"].removesuffix("_directed_query_1").removesuffix("_directed_query_2") for row in loop["actions"] if row.get("status") != "duplicate_action_stopped"]
    assert action_roles == expected_roles
    assert loop["action_count"] <= loop["action_limit"] == 2
    assert loop["stop_code"] == expected_stop_code
    assert result["optional_experts"]["news"]["raw_source_text_sent_to_model"] is False
    assert result["optional_experts"]["market_attention"]["text_sentiment_available"] is False
    assert client.role_counts["news"] == 0
    assert client.role_counts["market_attention"] == 0
    assert RAW_NEWS_FIXTURE not in json.dumps(client.calls, ensure_ascii=False)


@pytest.mark.parametrize("dimension", ["event", "industry", "model"])
def test_manager_risk_action_is_sourced_and_separate_from_hard_checks(dimension: str) -> None:
    research = _research_data()
    config: dict[str, object] = {
        "feature_trend_sessions": 1,
        "feature_reversal_sessions": 1,
        "feature_liquidity_return_sessions": 1,
        "feature_recent_amount_sessions": 1,
        "feature_reference_amount_sessions": 1,
        "prediction_horizon_sessions": 3,
    }
    if dimension == "industry":
        dates = [AS_OF - timedelta(days=offset) for offset in range(20, -1, -1)]
        research = ResearchData(
            dates,
            [
                {
                    "symbol": f"sh.{symbol}", "trade_date": day, "available_at": day,
                    "close": 10 + index * (0.05 if symbol == SYMBOL else 0.04) + (index % 3) * 0.03,
                    "amount": 1000 + index, "source_ref": f"synthetic-bar-{symbol}-{day.isoformat()}",
                }
                for symbol in (SYMBOL, "600001")
                for index, day in enumerate(dates)
            ],
            [], [],
            [MembershipSnapshot(dates[0], dates[0], frozenset({f"sh.{SYMBOL}", "sh.600001"}), "synthetic-membership")],
        )
        config["risk_industry_sources"] = [{
            "symbol": SYMBOL,
            "industry": "Synthetic Financials",
            "member_symbols": [SYMBOL, "600001"],
            "source_title": "Synthetic fixture industry classification",
            "source_url": "https://fixture.invalid/industry-map",
            "published_at": "2026-08-31T15:00:00+08:00",
            "available_at": "2026-08-31T15:00:00+08:00",
            "fetched_at": "2026-08-31T15:05:00+08:00",
        }]
    client = HarnessFlowClient(f"risk_{dimension}")
    model_context = {
        "status": "available",
        "available_at": "2026-09-01T16:00:00+08:00",
        "period_start": "2026-08-01",
        "period_end": AS_OF.isoformat(),
        "metrics": [{"metric": "matured_forecast_error_loss_gated_fusion", "value": 0.025, "unit": "return_fraction"}],
        "source_refs": ["synthetic-training-seed", "synthetic-calibration-window"],
        "context_sha256": "a" * 64,
    }
    snapshot = _public_snapshot()
    result = run_research_harness(
        research,
        AS_OF,
        snapshot,
        seed_candidate(),
        client,
        CallBudget(max_calls=10),
        config=config,
        model_risk_context=model_context,
    )

    manager = result["agents"]["research_manager"]["report"]
    risk_review = manager["risk_research"][0]
    assert risk_review["dimension"] == dimension
    assert risk_review["status"] == "completed_new_evidence"
    assert risk_review["source_evidence_ids"]
    assert all(evidence_id in result["evidence_catalog"] for evidence_id in risk_review["source_evidence_ids"])
    assert result["optional_experts"]["risk"]["hard_checks_separate"] is True


def test_optional_experts_report_missing_channels_as_unavailable() -> None:
    snapshot = {"p10_outputs": [], "attention_outputs": []}
    from adaptive_mas.agents.harness import _attention_handlers, _news_handlers

    _, news_handlers, _ = _news_handlers(snapshot, datetime_at_cutoff())
    _, attention_handlers, _ = _attention_handlers(snapshot, datetime_at_cutoff())

    news = news_handlers["search_public_news_evidence"]({
        "question": "Check for captured news.", "symbol": SYMBOL, "limit": 1,
    })
    attention = attention_handlers["search_public_attention_index"]({"symbol": SYMBOL, "limit": 1})

    assert news["status"] == "unavailable"
    assert news["raw_text_sent_to_model"] is False
    assert attention["status"] == "unavailable"
    assert attention["text_sentiment_available"] is False


def datetime_at_cutoff():
    from datetime import datetime

    return datetime.fromisoformat("2026-09-01T16:00:00+08:00")
