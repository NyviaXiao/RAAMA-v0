from __future__ import annotations

from datetime import datetime, timezone

import pytest

from adaptive_mas.industrial_rag.models import QueryPlan
from adaptive_mas.industrial_rag.planning import normalize_period, plan_research, requested_metrics, requested_periods


def _plan(query: str, *, report_periods: tuple[str, ...] = (), top_k: int = 20) -> QueryPlan:
    return QueryPlan(
        plan_id="planning-case",
        query=query,
        security_code="600036",
        as_of=datetime(2026, 10, 26, tzinfo=timezone.utc),
        principal_id="local-public",
        task_type="comparison",
        report_periods=report_periods,
        top_k=top_k,
    )


def test_financial_query_is_split_into_metric_period_dimensions_and_counter_search() -> None:
    plan = _plan("对比2023年和2024年营业收入、归母净利润")
    searches = plan_research(plan)
    supporting = [search for search in searches if search.kind == "supporting_evidence"]
    assert {(search.metric, search.report_period) for search in supporting} == {
        ("operating_revenue", "2023年"),
        ("operating_revenue", "2024年"),
        ("parent_net_profit", "2023年"),
        ("parent_net_profit", "2024年"),
    }
    assert searches[-1].kind == "counter_evidence_candidate"
    assert "更正" in searches[-1].query
    assert "追溯调整" in searches[-1].query


def test_planner_keeps_a_counter_search_inside_the_fixed_eight_search_budget() -> None:
    query = "比较2022年至2025年收入、净利润、经营现金流和资产负债率"
    searches = plan_research(_plan(query))
    expected_dimensions = {
        (metric, period)
        for metric in requested_metrics(query)
        for period in requested_periods(query, ())
    }
    covered_dimensions = set()
    for search in searches:
        if search.kind != "supporting_evidence":
            continue
        metrics = (search.metric,) if search.metric is not None else requested_metrics(query)
        periods = (
            search.report_periods
            if search.report_periods is not None
            else (search.report_period,) if search.report_period is not None
            else requested_periods(query, ())
        )
        covered_dimensions.update((metric, period) for metric in metrics for period in periods)
    assert len(searches) <= 8
    assert searches[-1].kind == "counter_evidence_candidate"
    assert sum(search.kind == "supporting_evidence" for search in searches) <= 7
    assert covered_dimensions == expected_dimensions


def test_two_period_comparison_stays_visible_when_one_support_search_is_budgeted() -> None:
    query = "2024年度营业收入是多少，和2023年度相比如何？"
    plan = _plan(query, report_periods=("2023", "2024"))
    searches = plan_research(plan, max_searches=2)
    supporting = [search for search in searches if search.kind == "supporting_evidence"]

    assert len(searches) == 2
    assert len(supporting) == 1
    assert supporting[0].metric == "operating_revenue"
    assert supporting[0].report_period is None
    assert supporting[0].report_periods == ("2023", "2024")
    assert "2023" in supporting[0].query
    assert "2024" in supporting[0].query
    assert searches[-1].kind == "counter_evidence_candidate"


def test_planner_fails_closed_when_search_and_candidate_budgets_cannot_cover_dimensions() -> None:
    query = "对比营业收入、净利润和经营现金流的2022、2023、2024年数据"
    plan = _plan(query, report_periods=("2022", "2023", "2024"), top_k=2)

    with pytest.raises(ValueError, match="cannot cover every requested metric-period dimension"):
        plan_research(plan, max_searches=2)


def test_evaluation_planner_switches_remove_rewrite_and_multi_hop_but_keep_counter_search() -> None:
    plan = _plan("对比2023年和2024年营业收入、归母净利润")
    baseline = plan_research(plan)
    no_rewrite = plan_research(plan, query_rewrite_enabled=False)
    no_multi_hop = plan_research(plan, multi_hop_enabled=False)

    assert len([item for item in baseline if item.kind == "supporting_evidence"]) == 4
    assert len([item for item in no_rewrite if item.kind == "supporting_evidence"]) == 4
    assert all(item.query == plan.query for item in no_rewrite if item.kind == "supporting_evidence")
    assert len([item for item in no_multi_hop if item.kind == "supporting_evidence"]) == 1
    assert all(items[-1].kind == "counter_evidence_candidate" for items in (baseline, no_rewrite, no_multi_hop))
    assert "更正" in no_multi_hop[-1].query


def test_metric_alias_overlap_and_period_variants_are_normalized() -> None:
    assert requested_metrics("归属于上市公司股东的净利润") == ("parent_net_profit",)
    assert normalize_period("2024年第一季度") == "2024Q1"
    assert normalize_period("2024年1-3月") == "2024Q1"
    assert normalize_period("2024年12月31日") == "2024-12-31"


def test_recall_terms_do_not_conflate_financial_metric_labels() -> None:
    from adaptive_mas.industrial_rag.planning import (
        METRIC_RECALL_TERMS,
        metric_for_source_label,
        requested_metric_specs,
        requested_metrics,
    )

    assert requested_metrics("净利润") == ("net_profit",)
    assert requested_metrics("利润总额") == ("total_profit",)
    assert requested_metrics("主营业务收入") == ("main_business_revenue",)
    assert requested_metrics("营业总收入") == ("total_operating_revenue",)
    assert requested_metrics("基本每股收益") == ("basic_eps",)
    assert requested_metrics("稀释每股收益") == ("diluted_eps",)
    assert requested_metrics("资产总计") == ("total_assets",)
    assert requested_metric_specs("收入")[0].match_status == "ambiguous_request"
    assert metric_for_source_label("利润总额") == "total_profit"
    assert metric_for_source_label("归属于本行股东的净利润") == "parent_net_profit"
    assert metric_for_source_label("资产总计") == "total_assets"
    assert requested_metrics("归属于本行股东的净利润") == ("parent_net_profit",)
    assert "利润总额" in METRIC_RECALL_TERMS["net_profit"]


@pytest.mark.parametrize(
    "basis",
    [None, "", "unknown", "not_stated_in_target_table", "not disclosed", "未披露", "不适用"],
)
def test_unknown_accounting_basis_labels_never_get_a_calculation_signature(basis: str | None) -> None:
    from adaptive_mas.industrial_rag.planning import accounting_basis_signature, accounting_basis_status

    assert accounting_basis_status(basis) == "unknown"
    assert accounting_basis_signature(basis) is None


def test_known_group_basis_remains_calculation_eligible() -> None:
    from adaptive_mas.industrial_rag.planning import accounting_basis_signature, accounting_basis_status

    assert accounting_basis_status("本集团") == "known"
    assert accounting_basis_signature("本集团") == "consolidated"


def test_unit_basis_and_q1_labels_are_preserved_for_coverage() -> None:
    from adaptive_mas.industrial_rag.planning import (
        accounting_basis_signature,
        requested_accounting_basis,
        requested_unit,
        unit_signature,
    )

    assert requested_periods("2024年1-3月营业收入", ()) == ("2024年1-3月",)
    assert requested_unit("以人民币百万元列示2024年营业收入") == "人民币百万元"
    assert requested_unit("以千元列示2024年营业收入") == "千元"
    assert unit_signature("千元") == "unspecified_currency:1000"
    from adaptive_mas.industrial_rag.planning import metric_for_source_label
    assert metric_for_source_label("营业收入（千元）") == "operating_revenue"
    assert unit_signature("人民币元", "1000000") == unit_signature("人民币百万元")
    assert requested_accounting_basis("合并归母净利润") == "consolidated"
    assert accounting_basis_signature("本集团") == "consolidated"
