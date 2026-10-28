"""Deterministic, bounded planning for financial document research."""

from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation

from adaptive_mas.industrial_rag.models import QueryPlan


MAX_RESEARCH_SEARCHES = 8

# These labels are accepted as the same requested metric. Broader synonyms live
# in METRIC_RECALL_TERMS and are never sufficient to satisfy query coverage.
METRIC_ALIASES: dict[str, tuple[str, ...]] = {
    "operating_revenue": ("营业收入",),
    "total_operating_revenue": ("营业总收入",),
    "main_business_revenue": ("主营业务收入",),
    "parent_net_profit": (
        "归母净利润", "归属于本行股东的净利润", "归属于母公司股东的净利润",
        "归属于上市公司股东的净利润", "归属于母公司所有者的净利润",
    ),
    "net_profit": ("净利润",),
    "total_profit": ("利润总额",),
    "operating_cash_flow": (
        "经营活动产生的现金流量净额", "经营活动现金流量净额",
    ),
    "roe": ("净资产收益率",),
    "weighted_average_roe": ("加权平均净资产收益率",),
    "gross_margin": ("毛利率", "销售毛利率"),
    "debt_ratio": ("资产负债率",),
    "total_assets": ("资产总额", "资产总计", "总资产"),
    "basic_eps": ("基本每股收益",),
    "diluted_eps": ("稀释每股收益",),
}

# Search-only expansions improve lexical recall. Their hits remain candidates
# unless the source label exactly identifies the requested metric above.
METRIC_RECALL_TERMS: dict[str, tuple[str, ...]] = {
    "operating_revenue": ("营业收入", "营业总收入", "主营业务收入", "收入"),
    "total_operating_revenue": ("营业总收入", "营业收入", "主营业务收入", "收入"),
    "main_business_revenue": ("主营业务收入", "营业收入", "营业总收入", "收入"),
    "parent_net_profit": (
        "归母净利润", "归属于本行股东的净利润", "归属于母公司股东的净利润",
        "归属于上市公司股东的净利润", "净利润",
    ),
    "net_profit": ("净利润", "利润总额", "归母净利润"),
    "total_profit": ("利润总额", "净利润", "归母净利润"),
    "operating_cash_flow": ("经营活动产生的现金流量净额", "经营活动现金流量净额", "经营活动现金流", "经营现金流"),
    "roe": ("净资产收益率", "加权平均净资产收益率", "ROE"),
    "weighted_average_roe": ("加权平均净资产收益率", "净资产收益率", "ROE"),
    "gross_margin": ("毛利率", "销售毛利率"),
    "debt_ratio": ("资产负债率",),
    "total_assets": ("资产总额", "资产总计", "总资产"),
    "basic_eps": ("基本每股收益", "稀释每股收益", "每股收益", "EPS"),
    "diluted_eps": ("稀释每股收益", "基本每股收益", "每股收益", "EPS"),
    "ambiguous_revenue": ("收入", "营业收入", "营业总收入", "主营业务收入"),
    "ambiguous_operating_cash_flow": ("经营现金流", "经营活动现金流", "经营活动现金流量净额"),
    "ambiguous_eps": ("每股收益", "EPS", "基本每股收益", "稀释每股收益"),
}

AMBIGUOUS_METRIC_ALIASES: dict[str, tuple[str, ...]] = {
    "ambiguous_revenue": ("收入",),
    "ambiguous_operating_cash_flow": ("经营现金流", "经营活动现金流"),
    "ambiguous_eps": ("每股收益", "EPS"),
}

_PERIOD_RE = re.compile(
    r"20\d{2}(?:年(?:第?[一二三四1-4]季度|年度|半年度|上半年|前三季度|年报|半年报|三季报|一季报|"
    r"1-3月|4-6月|7-9月|10-12月|1-6月|1-9月|\d{1,2}月(?:\d{1,2}日)?)?|Q[1-4])?",
    re.IGNORECASE,
)

_UNIT_PATTERNS = (
    (r"人民币\s*百万元|百万元", "人民币百万元"),
    (r"人民币\s*亿元|亿元", "人民币亿元"),
    (r"人民币\s*万元|万元", "人民币万元"),
    (r"千元", "千元"),
    (r"人民币\s*元\s*/\s*股|元\s*/\s*股", "人民币元/股"),
    (r"百分比|百分数|%", "%"),
    (r"基点|\bbps?\b", "基点"),
    (r"人民币\s*元|(?<![百千万元亿])元(?!\s*/\s*股)", "人民币元"),
)


@dataclass(frozen=True, slots=True)
class PlannedSearch:
    step_id: str
    kind: str
    query: str
    metric: str | None = None
    requested_metric: str | None = None
    metric_match_status: str = "exact"
    report_period: str | None = None
    report_periods: tuple[str, ...] | None = None


@dataclass(frozen=True, slots=True)
class RequestedMetric:
    metric: str
    label: str
    match_status: str


def _normalized_label(value: str) -> str:
    text = re.sub(
        r"[（(]\s*(?:人民币\s*)?(?:百万元|亿元|万元|千元|元\s*/\s*股|元|%)\s*[）)]\s*$",
        "",
        value,
    )
    return re.sub(r"[\s:：,，。()（）]", "", text).casefold()


def metric_for_source_label(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    label = _normalized_label(value)
    return next(
        (metric for metric, aliases in METRIC_ALIASES.items()
         if label in {_normalized_label(alias) for alias in aliases}),
        None,
    )


def requested_metric_specs(query: str) -> tuple[RequestedMetric, ...]:
    matches: list[tuple[int, int, str, str, str]] = []
    for metric, aliases in METRIC_ALIASES.items():
        for alias in aliases:
            start = query.casefold().find(alias.casefold())
            if start >= 0:
                matches.append((start, start + len(alias), metric, alias, "exact"))
    for metric, aliases in AMBIGUOUS_METRIC_ALIASES.items():
        for alias in aliases:
            start = query.casefold().find(alias.casefold())
            if start >= 0:
                matches.append((start, start + len(alias), metric, alias, "ambiguous_request"))
    matches.sort(key=lambda item: (item[0], -(item[1] - item[0]), item[4] != "exact"))
    selected: list[RequestedMetric] = []
    covered: list[tuple[int, int]] = []
    for start, end, metric, label, match_status in matches:
        if any(start >= left and end <= right for left, right in covered):
            continue
        selected.append(RequestedMetric(metric, label, match_status))
        covered.append((start, end))
    return tuple(selected)


def requested_metrics(query: str) -> tuple[str, ...]:
    """Return exact or explicitly ambiguous requested metric keys."""
    return tuple(dict.fromkeys(item.metric for item in requested_metric_specs(query)))


def requested_periods(query: str, explicit_periods: tuple[str, ...]) -> tuple[str, ...]:
    if explicit_periods:
        return tuple(dict.fromkeys(explicit_periods))
    return tuple(dict.fromkeys(match.group(0) for match in _PERIOD_RE.finditer(query)))


def metric_terms(metric: str) -> tuple[str, ...]:
    return METRIC_RECALL_TERMS.get(metric, METRIC_ALIASES.get(metric, ()))


def requested_unit(query: str) -> str | None:
    for pattern, label in _UNIT_PATTERNS:
        if re.search(pattern, query, flags=re.IGNORECASE):
            return label
    return None


def unit_signature(value: object, multiplier: object = "1") -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    text = re.sub(r"\s+", "", value).casefold()
    units = {
        "人民币百万元": ("CNY", Decimal("1000000")),
        "百万元": ("CNY", Decimal("1000000")),
        "人民币万元": ("CNY", Decimal("10000")),
        "万元": ("CNY", Decimal("10000")),
        "千元": ("unspecified_currency", Decimal("1000")),
        "人民币亿元": ("CNY", Decimal("100000000")),
        "亿元": ("CNY", Decimal("100000000")),
        "人民币元": ("CNY", Decimal("1")),
        "人民币": ("CNY", Decimal("1")),
        "元": ("CNY", Decimal("1")),
        "cny": ("CNY", Decimal("1")),
        "人民币元/股": ("CNY/share", Decimal("1")),
        "元/股": ("CNY/share", Decimal("1")),
        "%": ("percent", Decimal("1")),
        "百分比": ("percent", Decimal("1")),
        "百分数": ("percent", Decimal("1")),
        "基点": ("basis_point", Decimal("1")),
        "bp": ("basis_point", Decimal("1")),
        "bps": ("basis_point", Decimal("1")),
    }
    unit = units.get(text)
    if unit is None:
        return None
    try:
        extra_multiplier = Decimal(str(multiplier))
    except (InvalidOperation, TypeError, ValueError):
        return None
    if not extra_multiplier.is_finite() or extra_multiplier <= 0:
        return None
    base, scale = unit
    effective_scale = (scale * extra_multiplier).normalize()
    return f"{base}:{format(effective_scale, 'f')}"


def requested_accounting_basis(query: str, explicit_basis: str | None = None) -> str | None:
    if explicit_basis:
        return explicit_basis
    text = re.sub(r"\s+", "", query)
    if "合并" in text or "本集团口径" in text or "集团口径" in text:
        return "consolidated"
    if re.search(r"(?:母公司|本公司|单体)(?:财务)?(?:报表|口径)", text):
        return "parent_company"
    return None


_UNKNOWN_ACCOUNTING_BASIS_LABELS = {
    "",
    "unknown",
    "unknown basis",
    "unspecified",
    "not stated",
    "not stated in source",
    "not stated in source table",
    "not stated in target table",
    "not specified",
    "not provided",
    "not disclosed",
    "not reported",
    "not available",
    "not applicable",
    "unreported",
    "n/a",
    "na",
    "none",
    "null",
    "nil",
    "-",
    "—",
    "–",
    "－",
    "未知",
    "不明",
    "未注明",
    "未说明",
    "未披露",
    "未提供",
    "未标注",
    "未提取",
    "未识别",
    "不适用",
    "无口径",
    "口径未知",
    "口径未注明",
    "会计口径未知",
    "会计口径未注明",
}


def accounting_basis_status(value: object) -> str:
    if not isinstance(value, str):
        return "unknown"
    label = re.sub(r"[\s_-]+", " ", value.strip()).strip(" .,:;，。；：").casefold()
    return "unknown" if label in _UNKNOWN_ACCOUNTING_BASIS_LABELS else "known"


def accounting_basis_signature(value: object) -> str | None:
    if accounting_basis_status(value) != "known":
        return None
    text = re.sub(r"\s+", "", value).casefold()
    if any(token in text for token in ("本集团", "合并", "集团")):
        return "consolidated"
    if any(token in text for token in ("本公司", "母公司口径", "单体")):
        return "parent_company"
    return text


def _period_query_terms(period: str) -> str:
    year_match = re.search(r"20\d{2}", period)
    return f"{year_match.group(0)} {period}" if year_match else period


def plan_research(
    plan: QueryPlan,
    *,
    max_searches: int = MAX_RESEARCH_SEARCHES,
    query_rewrite_enabled: bool = True,
    multi_hop_enabled: bool = True,
) -> tuple[PlannedSearch, ...]:
    """Build bounded searches without silently dropping requested metric-period coverage."""
    if isinstance(max_searches, bool) or not 2 <= max_searches <= MAX_RESEARCH_SEARCHES:
        raise ValueError(f"max_searches must be between 2 and {MAX_RESEARCH_SEARCHES}")
    if not isinstance(query_rewrite_enabled, bool) or not isinstance(multi_hop_enabled, bool):
        raise ValueError("query rewrite and multi-hop switches must be booleans")

    metric_specs = requested_metric_specs(plan.query)
    periods = requested_periods(plan.query, plan.report_periods)
    dimensions = [(metric, period) for metric in metric_specs for period in (periods or (None,))]
    if not dimensions:
        dimensions = [(None, period) for period in (periods or (None,))]
    if not multi_hop_enabled:
        dimensions = dimensions[:1]
    positive_limit = max_searches - 1
    if len(dimensions) <= positive_limit:
        dimension_groups = [[dimension] for dimension in dimensions]
    else:
        candidate_capacity = min(plan.top_k, 10)
        metrics_in_order = list(dict.fromkeys(metric.metric if metric is not None else None for metric, _ in dimensions))
        periods_in_order = list(dict.fromkeys(period for _, period in dimensions))
        grouping_candidates = ([[list(dimensions)]] if len(dimensions) <= candidate_capacity else []) + [
            [
                [dimension for dimension in dimensions if (dimension[0].metric if dimension[0] is not None else None) == metric]
                for metric in metrics_in_order
            ],
            [
                [dimension for dimension in dimensions if dimension[1] == period]
                for period in periods_in_order
            ],
        ]
        grouping_candidates = [
            groups for groups in grouping_candidates
            if groups
            and len(groups) <= positive_limit
            and all(0 < len(group) <= candidate_capacity for group in groups)
            and sum(len(group) for group in groups) == len(dimensions)
        ]
        if not grouping_candidates:
            raise ValueError(
                "research search and candidate budgets cannot cover every requested metric-period dimension "
                "while reserving a counter-evidence search"
            )
        dimension_groups = min(
            grouping_candidates,
            key=lambda groups: (len(groups), max(len(group) for group in groups)),
        )

    searches: list[PlannedSearch] = []
    for group in dimension_groups:
        metric_values = list(dict.fromkeys(metric for metric, _ in group))
        period_values = list(dict.fromkeys(period for _, period in group if period is not None))
        metric = metric_values[0] if len(metric_values) == 1 else None
        period = period_values[0] if len(period_values) == 1 else None
        if len(group) == 1:
            metric = group[0][0]
            period = group[0][1]
        metric_key = metric.metric if metric is not None else None
        query_parts = [plan.query]
        if query_rewrite_enabled:
            for value in metric_values:
                terms = metric_terms(value.metric) if value is not None else ()
                if terms:
                    query_parts.append(" ".join(terms))
            query_parts.extend(_period_query_terms(value) for value in period_values)
        requested_metric = metric.label if metric is not None else None
        metric_match_status = (
            metric.match_status
            if metric is not None else
            "unspecified" if len(metric_values) == 1 and metric_values[0] is None else
            "multi_dimension"
        )
        searches.append(PlannedSearch(
            step_id=f"search-{len(searches) + 1}",
            kind="supporting_evidence",
            query=" ".join(query_parts)[:2000],
            metric=metric_key,
            requested_metric=requested_metric,
            metric_match_status=metric_match_status,
            report_period=period,
            report_periods=tuple(period_values) if len(group) > 1 and period_values else None,
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
