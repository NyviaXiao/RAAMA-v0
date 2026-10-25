from __future__ import annotations

from datetime import datetime, timezone

from adaptive_mas.industrial_rag.models import QueryPlan
from adaptive_mas.industrial_rag.planning import normalize_period, plan_research, requested_metrics


def _plan(query: str, *, report_periods: tuple[str, ...] = ()) -> QueryPlan:
    return QueryPlan(
        plan_id="planning-case",
        query=query,
        security_code="600036",
        as_of=datetime(2026, 10, 26, tzinfo=timezone.utc),
        principal_id="local-public",
        task_type="comparison",
        report_periods=report_periods,
    )


def test_financial_query_is_split_into_metric_period_dimensions_and_counter_search() -> None:
    plan = _plan("对比2023年和2024年营业收入、归母净利润")
    searches = plan_research(plan)
    supporting = [search for search in searches if search.kind == "supporting_evidence"]
    assert {(search.metric, search.report_period) for search in supporting} == {
        ("revenue", "2023年"),
        ("revenue", "2024年"),
        ("parent_net_profit", "2023年"),
        ("parent_net_profit", "2024年"),
    }
    assert searches[-1].kind == "counter_evidence_candidate"
    assert "更正" in searches[-1].query
    assert "追溯调整" in searches[-1].query


def test_planner_keeps_a_counter_search_inside_the_fixed_eight_search_budget() -> None:
    searches = plan_research(_plan("比较2022年至2025年收入、净利润、经营现金流和资产负债率"))
    assert len(searches) == 8
    assert searches[-1].kind == "counter_evidence_candidate"
    assert sum(search.kind == "supporting_evidence" for search in searches) == 7


def test_metric_alias_overlap_and_period_variants_are_normalized() -> None:
    assert requested_metrics("归属于上市公司股东的净利润") == ("parent_net_profit",)
    assert normalize_period("2024年第一季度") == "2024Q1"
    assert normalize_period("2024年1-3月") == "2024Q1"
    assert normalize_period("2024年12月31日") == "2024-12-31"
