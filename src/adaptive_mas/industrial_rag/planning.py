"""Deterministic, bounded planning for financial document research."""

from __future__ import annotations

import re
from dataclasses import dataclass

from adaptive_mas.industrial_rag.models import QueryPlan


MAX_RESEARCH_SEARCHES = 8

# Keep this vocabulary explicit and reviewable. It expands lexical queries; it
# never changes the metric or period recorded on a source citation.
METRIC_ALIASES: dict[str, tuple[str, ...]] = {
    "revenue": ("营业收入", "营业总收入", "主营业务收入", "收入"),
    "parent_net_profit": (
        "归母净利润", "归属于母公司股东的净利润", "归属于上市公司股东的净利润", "归属于母公司所有者的净利润",
    ),
    "net_profit": ("净利润", "利润总额"),
    "operating_cash_flow": (
        "经营活动产生的现金流量净额", "经营活动现金流量净额", "经营现金流", "经营活动现金流",
    ),
    "roe": ("净资产收益率", "加权平均净资产收益率", "ROE"),
    "gross_margin": ("毛利率", "销售毛利率"),
    "debt_ratio": ("资产负债率",),
    "total_assets": ("资产总额", "总资产"),
    "eps": ("基本每股收益", "稀释每股收益", "每股收益", "EPS"),
}

_PERIOD_RE = re.compile(
    r"20\d{2}(?:年(?:第?[一二三四1-4]季度|年度|半年度|上半年|前三季度|年报|半年报|三季报|一季报|1-6月|1-9月)?|Q[1-4])?",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class PlannedSearch:
    step_id: str
    kind: str
    query: str
    metric: str | None = None
    report_period: str | None = None


def requested_metrics(query: str) -> tuple[str, ...]:
    """Return canonical metrics mentioned by the user, with overlapping aliases folded."""
    matches: list[tuple[int, int, str]] = []
    for metric, aliases in METRIC_ALIASES.items():
        for alias in aliases:
            start = query.casefold().find(alias.casefold())
            if start >= 0:
                matches.append((start, start + len(alias), metric))
    matches.sort(key=lambda item: (item[0], -(item[1] - item[0])))
    selected: list[str] = []
    covered: list[tuple[int, int]] = []
    for start, end, metric in matches:
        if any(start >= left and end <= right for left, right in covered):
            continue
        if metric not in selected:
            selected.append(metric)
        covered.append((start, end))
    return tuple(selected)


def requested_periods(query: str, explicit_periods: tuple[str, ...]) -> tuple[str, ...]:
    if explicit_periods:
        return tuple(dict.fromkeys(explicit_periods))
    return tuple(dict.fromkeys(match.group(0) for match in _PERIOD_RE.finditer(query)))


def metric_terms(metric: str) -> tuple[str, ...]:
    return METRIC_ALIASES.get(metric, ())


def _period_query_terms(period: str) -> str:
    year_match = re.search(r"20\d{2}", period)
    return f"{year_match.group(0)} {period}" if year_match else period


def plan_research(plan: QueryPlan, *, max_searches: int = MAX_RESEARCH_SEARCHES) -> tuple[PlannedSearch, ...]:
    """Build at most eight fixed-template searches, reserving one for counter evidence."""
    if isinstance(max_searches, bool) or not 2 <= max_searches <= MAX_RESEARCH_SEARCHES:
        raise ValueError(f"max_searches must be between 2 and {MAX_RESEARCH_SEARCHES}")

    metrics = requested_metrics(plan.query)
    periods = requested_periods(plan.query, plan.report_periods)
    dimensions = [(metric, period) for metric in metrics for period in (periods or (None,))]
    if not dimensions:
        dimensions = [(None, period) for period in (periods or (None,))]
    positive_limit = max_searches - 1
    searches: list[PlannedSearch] = []
    for metric, period in dimensions[:positive_limit]:
        terms = metric_terms(metric) if metric else ()
        query_parts = [plan.query]
        if terms:
            query_parts.append(" ".join(terms))
        if period:
            query_parts.append(_period_query_terms(period))
        searches.append(PlannedSearch(
            step_id=f"search-{len(searches) + 1}",
            kind="supporting_evidence",
            query=" ".join(query_parts)[:2000],
            metric=metric,
            report_period=period,
        ))

    # This result is intentionally labeled as a candidate. A lexical hit does
    # not establish a contradiction until a human or supported claim check reviews it.
    searches.append(PlannedSearch(
        step_id=f"search-{len(searches) + 1}",
        kind="counter_evidence_candidate",
        query=(plan.query + " 更正 重述 追溯调整 会计差错 前期差错 修订 更正版").strip()[:2000],
    ))
    return tuple(searches)


def normalize_period(value: object) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    text = re.sub(r"\s+", "", value).casefold()
    year_match = re.search(r"(?:19|20)\d{2}", text)
    if year_match is None:
        return re.sub(r"[\s年月日度第季度报上半前]", "", text) or None
    year = year_match.group(0)
    quarter_aliases = (
        (r"(?:第一季度|第?一季度|1季度|一季报|q1)", "Q1"),
        (r"(?:第二季度|第?二季度|2季度|二季报|q2)", "Q2"),
        (r"(?:第三季度|第?三季度|3季度|三季报|q3)", "Q3"),
        (r"(?:第四季度|第?四季度|4季度|四季报|q4)", "Q4"),
    )
    for pattern, quarter in quarter_aliases:
        if re.search(pattern, text):
            return year + quarter
    month_ranges = (
        (r"(?:1|01)[-—~至](?:3|03)月", "Q1"),
        (r"(?:4|04)[-—~至](?:6|06)月", "Q2"),
        (r"(?:7|07)[-—~至](?:9|09)月", "Q3"),
        (r"(?:10)[-—~至](?:12)月", "Q4"),
    )
    for pattern, quarter in month_ranges:
        if re.search(pattern, text):
            return year + quarter
    if re.search(r"(?:上半年|半年度|1[-—~至]6月)", text):
        return year + "H1"
    if re.search(r"(?:前三季度|1[-—~至]9月)", text):
        return year + "9M"
    month_day = re.search(r"(\d{1,2})月(?:/|-)?(\d{1,2})日", text)
    if month_day:
        return f"{year}-{int(month_day.group(1)):02d}-{int(month_day.group(2)):02d}"
    if re.match(r"^(?:19|20)\d{2}(?:年度|年报|年|全年)", text):
        return year
    compact = re.sub(r"[\s年月日度第季度报上半前]", "", text)
    return compact or None
