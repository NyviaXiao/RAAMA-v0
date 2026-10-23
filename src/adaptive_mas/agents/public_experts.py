"""Point-in-time public-record rules for P9 and forward observations for P10."""

from __future__ import annotations

import hashlib
import time
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo
from urllib.error import URLError

from adaptive_mas.agents.qwen_expert import call_qwen_expert
from adaptive_mas.data.cninfo_full_text import CninfoPdfError, fetch_cninfo_full_text
from adaptive_mas.data.public_data import load_public_records


SHANGHAI = ZoneInfo("Asia/Shanghai")


def _dt(value: object) -> datetime:
    return datetime.fromisoformat(str(value)).astimezone(SHANGHAI)


def _number(value: object) -> float | None:
    if value is None or str(value).strip() == "":
        return None
    return float(value)


def _captured_records(records: list[dict[str, object]], decision_time: datetime) -> list[dict[str, object]]:
    return [row for row in records if _dt(row["fetched_at"]) <= decision_time]


def _effective_available_at(record: dict[str, object]) -> str:
    if record.get("retrospective_backfill_proxy") is True:
        return str(record["available_at"])
    return max(_dt(record["available_at"]), _dt(record["fetched_at"])).isoformat()


def _backfill_audit_fields(record: dict[str, object]) -> dict[str, object]:
    return {
        "fetched_at": record.get("fetched_at"),
        "vintage_status": record.get("vintage_status"),
        "retrospective_backfill_proxy": bool(record.get("retrospective_backfill_proxy", False)),
    }


def _headline_direction(title: str) -> tuple[int, str]:
    negative = ("预减", "预亏", "续亏", "首亏", "亏损", "下滑", "下降", "减持", "处罚", "立案", "违规", "诉讼", "违约")
    positive = ("预增", "扭亏", "大增", "增长", "上升", "回购", "增持", "中标", "获批")
    if any(term in title for term in negative):
        return -1, "headline_rule_negative"
    if any(term in title for term in positive):
        return 1, "headline_rule_positive"
    return 0, "headline_rule_no_direction"


def _news_direction(text: str) -> tuple[int, str]:
    negative = ("亏损", "下滑", "下降", "处罚", "立案", "违规", "诉讼", "违约", "减持", "暴跌", "下跌")
    positive = ("预增", "扭亏", "大增", "增长", "回购", "增持", "中标", "获批", "上涨", "大涨")
    if any(term in text for term in negative):
        return -1, "cls_text_rule_negative"
    if any(term in text for term in positive):
        return 1, "cls_text_rule_positive"
    return 0, "cls_text_rule_no_direction"


def build_public_expert_snapshot(
    cache_dir: str | Path,
    decision_time: datetime,
    evaluation_horizon_sessions: int,
    event_start_time: datetime | None = None,
    llm_max_records: int = 32,
    allowed_symbols: set[str] | None = None,
    announcement_text_limit: int = 3,
    announcement_document_dir: str | Path | None = None,
    announcement_ids: set[str] | None = None,
    previously_observed_record_ids: set[str] | None = None,
    include_qwen: bool = True,
    retrospective_backfill_proxy: bool = False,
) -> dict[str, object]:
    """Produce latest financial state plus only newly published/captured events."""
    if decision_time.tzinfo is None or decision_time.utcoffset() is None:
        raise ValueError("decision_time 必须带时区")
    if evaluation_horizon_sessions < 1:
        raise ValueError("evaluation_horizon_sessions 必须为正整数")
    if llm_max_records < 1:
        raise ValueError("llm_max_records 必须为正数")
    if announcement_text_limit < 0:
        raise ValueError("announcement_text_limit 不能为负数")
    if retrospective_backfill_proxy and announcement_text_limit:
        raise ValueError(
            "retrospective source-date mode permits cached disclosure titles only; body capture time is not proven"
        )
    if allowed_symbols is not None and any(
        not isinstance(symbol, str) or len(symbol) != 6 or not symbol.isascii() or not symbol.isdigit()
        for symbol in allowed_symbols
    ):
        raise ValueError("allowed_symbols 必须是六位数字代码")
    decision_time = decision_time.astimezone(SHANGHAI)
    if event_start_time is None:
        event_start_time = decision_time - timedelta(days=3)
    elif event_start_time.tzinfo is None or event_start_time.utcoffset() is None:
        raise ValueError("event_start_time 必须带时区")
    else:
        event_start_time = event_start_time.astimezone(SHANGHAI)
    if event_start_time > decision_time:
        raise ValueError("event_start_time 不能晚于 decision_time")

    started = time.perf_counter_ns()
    datasets = (
        "financial_indicator",
        "attention_index",
        "financial_statement_balance_sheet",
        "financial_statement_income_statement",
        "financial_statement_cash_flow_statement",
        "announcement",
        "news",
        "macro_cpi",
        "macro_pmi",
    )
    provider_symbols = None
    if allowed_symbols is not None:
        provider_symbols = {
            ("sh." if symbol[0] in "569" else "sz.") + symbol
            for symbol in allowed_symbols
        }
    loaded = {
        dataset: (
            [] if dataset == "news" and allowed_symbols is not None else
            load_public_records(
                cache_dir,
                dataset,
                decision_time,
                symbols=provider_symbols if dataset not in {"news", "macro_cpi", "macro_pmi"} else None,
            )
        )
        for dataset in datasets
    }
    historical_backfill_datasets = {
        "financial_indicator",
        "financial_statement_balance_sheet",
        "financial_statement_income_statement",
        "financial_statement_cash_flow_statement",
        "announcement",
    }
    captured = {}
    for dataset, rows in loaded.items():
        if not retrospective_backfill_proxy or dataset not in historical_backfill_datasets:
            captured[dataset] = _captured_records(rows, decision_time)
            continue
        retrospective_rows = []
        for row in rows:
            # load_public_records has already enforced source publication/availability cutoffs.
            # Its historical financial/announcement backfill path retains the real fetched_at
            # and marks versions fetched after the cutoff as unverified.
            fetched_after_decision = _dt(row["fetched_at"]) > decision_time
            retrospective_rows.append({
                **row,
                "retrospective_backfill_proxy": fetched_after_decision,
                "vintage_verified": False if fetched_after_decision else bool(row.get("vintage_verified", False)),
            })
        captured[dataset] = retrospective_rows
    retrospective_capture_lag_count = sum(
        bool(row.get("retrospective_backfill_proxy"))
        for dataset in historical_backfill_datasets
        for row in captured[dataset]
    )
    new_announcements = [
        row for row in captured["announcement"]
        if _dt(row["published_at"]) > event_start_time
    ]
    new_news = [
        row for row in captured["news"]
        if _dt(_effective_available_at(row)) > event_start_time
    ]

    indicators_by_symbol: dict[str, list[dict[str, object]]] = {}
    for row in captured["financial_indicator"]:
        indicators_by_symbol.setdefault(str(row["symbol"]), []).append(row)
    for rows in indicators_by_symbol.values():
        rows.sort(key=lambda row: (str(row["published_at"]), str(row["REPORT_DATE"])))

    statements_by_period: dict[tuple[str, str], list[tuple[str, dict[str, object]]]] = {}
    income_by_period: dict[tuple[str, str], dict[str, object]] = {}
    for dataset in (
        "financial_statement_balance_sheet",
        "financial_statement_income_statement",
        "financial_statement_cash_flow_statement",
    ):
        for row in captured[dataset]:
            period = str(row["报告日"]).replace("-", "")[:8]
            statements_by_period.setdefault((str(row["symbol"]), period), []).append((dataset, row))
            if dataset == "financial_statement_income_statement":
                income_by_period[(str(row["symbol"]), period)] = row

    p9_outputs: list[dict[str, object]] = []
    for symbol in sorted(indicators_by_symbol):
        latest = indicators_by_symbol[symbol][-1]
        revenue_indicator_yoy = _number(latest.get("TOTALOPERATEREVETZ"))
        profit_indicator_yoy = _number(latest.get("PARENTNETPROFITTZ"))
        period = str(latest["REPORT_DATE"])[:10]
        period_key = period.replace("-", "")[:8]
        prior_period_key = f"{int(period_key[:4]) - 1}{period_key[4:]}"
        latest_income = income_by_period.get((symbol, period_key))
        prior_income = income_by_period.get((symbol, prior_period_key))
        revenue_statement_yoy = None
        profit_statement_yoy = None
        if latest_income is not None and prior_income is not None:
            current_revenue = _number(latest_income.get("营业收入"))
            prior_revenue = _number(prior_income.get("营业收入"))
            current_profit = _number(latest_income.get("归属于母公司股东的净利润"))
            prior_profit = _number(prior_income.get("归属于母公司股东的净利润"))
            if current_revenue is not None and prior_revenue not in (None, 0.0):
                revenue_statement_yoy = (current_revenue / prior_revenue - 1) * 100
            if current_profit is not None and prior_profit not in (None, 0.0):
                profit_statement_yoy = (current_profit / prior_profit - 1) * 100

        statement_growth = [revenue_statement_yoy, profit_statement_yoy]
        indicator_growth = [revenue_indicator_yoy, profit_indicator_yoy]
        statement_complete = all(value is not None for value in statement_growth)
        indicator_complete = all(value is not None for value in indicator_growth)
        if statement_complete and indicator_complete:
            statement_signs = tuple(value >= 0 for value in statement_growth)
            indicator_signs = tuple(value >= 0 for value in indicator_growth)
            if statement_signs == indicator_signs and any(statement_growth):
                direction = 1 if all(statement_signs) else -1 if not any(statement_signs) else 0
                rule = "statement_and_indicator_growth_signs_agree"
            else:
                direction, rule = 0, "statement_and_indicator_growth_signs_disagree"
        elif statement_complete:
            if all(value >= 0 for value in statement_growth) and any(statement_growth):
                direction, rule = 1, "income_statement_growth_nonnegative"
            elif all(value <= 0 for value in statement_growth) and any(value < 0 for value in statement_growth):
                direction, rule = -1, "income_statement_growth_nonpositive"
            else:
                direction, rule = 0, "income_statement_growth_signs_disagree"
        elif indicator_complete:
            if all(value >= 0 for value in indicator_growth) and any(indicator_growth):
                direction, rule = 1, "indicator_growth_nonnegative"
            elif all(value <= 0 for value in indicator_growth) and any(value < 0 for value in indicator_growth):
                direction, rule = -1, "indicator_growth_nonpositive"
            else:
                direction, rule = 0, "indicator_growth_signs_disagree"
        else:
            direction, rule = 0, "insufficient_comparable_growth_fields"

        evidence = [{
            "source_record_id": latest["source_record_id"],
            "dataset": "financial_indicator",
            "raw_response_ref": latest["raw_response_ref"],
            "content_sha256": latest["content_sha256"],
            "published_at": latest["published_at"],
            "available_at": _effective_available_at(latest),
            **_backfill_audit_fields(latest),
            "vintage_verified": latest["vintage_verified"],
            "report_period": period,
            "revenue_yoy_pct": revenue_indicator_yoy,
            "parent_net_profit_yoy_pct": profit_indicator_yoy,
            "roe_pct": _number(latest.get("ROEJQ")),
            "income_statement_revenue_yoy_pct": revenue_statement_yoy,
            "income_statement_parent_profit_yoy_pct": profit_statement_yoy,
        }]
        for statement_period, statement in (
            (period_key, latest_income), (prior_period_key, prior_income)
        ):
            if statement is None:
                continue
            evidence.append({
                "source_record_id": statement["source_record_id"],
                "dataset": "financial_statement_income_statement",
                "raw_response_ref": statement["raw_response_ref"],
                "content_sha256": statement["content_sha256"],
                "published_at": statement["published_at"],
                "available_at": _effective_available_at(statement),
                **_backfill_audit_fields(statement),
                "vintage_verified": statement["vintage_verified"],
                "report_period": statement_period,
                "revenue_cny": _number(statement.get("营业收入")),
                "parent_net_profit_cny": _number(statement.get("归属于母公司股东的净利润")),
            })
        for dataset, statement in sorted(
            statements_by_period.get((symbol, period_key), []),
            key=lambda pair: str(pair[1]["source_record_id"]),
        ):
            evidence.append({
                "source_record_id": statement["source_record_id"],
                "dataset": dataset,
                "raw_response_ref": statement["raw_response_ref"],
                "content_sha256": statement["content_sha256"],
                "published_at": statement["published_at"],
                "available_at": _effective_available_at(statement),
                **_backfill_audit_fields(statement),
                "vintage_verified": statement["vintage_verified"],
                "report_period": str(statement["报告日"]),
            })
        p9_outputs.append({
            "record_id": hashlib.sha256(
            f"fundamental-rule-v1:{symbol}:{latest['source_record_id']}:{latest['content_sha256']}".encode("utf-8")
            ).hexdigest(),
            "record_kind": "state",
            "expert": "fundamental_rule_v1",
            "symbol": symbol,
            "event_type": "quarterly_fundamental_growth",
            "prediction_date": decision_time.date().isoformat(),
            "event_time": latest["published_at"],
            "available_at": _effective_available_at(latest),
            "horizon_sessions": evaluation_horizon_sessions,
            "horizon_kind": "comparison_label_window",
            "direction_score": direction,
            "rule": rule,
            "evidence_text_scope": "structured_financial_fields",
            "evidence": evidence,
            "vendor_vintage_verified": bool(latest["vintage_verified"]),
        })

    for row in new_announcements:
        title = str(row["公告标题"])
        direction, rule = _headline_direction(title)
        p9_outputs.append({
            "record_id": hashlib.sha256(
                f"announcement-title-v1:{row['source_record_id']}:{row['content_sha256']}".encode("utf-8")
            ).hexdigest(),
            "record_kind": "event",
            "expert": "announcement_title_rule_v1",
            "symbol": row["symbol"],
            "event_type": "announcement_headline",
            "prediction_date": decision_time.date().isoformat(),
            "event_time": row["published_at"],
            "available_at": _effective_available_at(row),
            "horizon_sessions": evaluation_horizon_sessions,
            "horizon_kind": "comparison_label_window",
            "direction_score": direction,
            "rule": rule,
            "evidence_text_scope": "headline_only_no_announcement_body",
            "evidence": [{
                "source_record_id": row["source_record_id"],
                "dataset": "announcement",
                "title": title,
                "url": row["公告链接"],
                "raw_response_ref": row["raw_response_ref"],
                "content_sha256": row["content_sha256"],
                "published_at": row["published_at"],
                "available_at": _effective_available_at(row),
                **_backfill_audit_fields(row),
            }],
            "vendor_vintage_verified": bool(row["vintage_verified"]),
        })

    latest_publication_date = decision_time.date() - timedelta(days=30)
    announcement_text_candidates = [
        row for row in captured["announcement"]
        if _dt(_effective_available_at(row)) > event_start_time
        and _dt(row["published_at"]).date() >= latest_publication_date
        and (announcement_ids is None or str(row["公告链接"]).split("announcementId=", 1)[-1].split("&", 1)[0] in announcement_ids)
    ]
    announcement_text_candidates.sort(
        key=lambda row: (str(row["published_at"]), str(row["source_record_id"])),
        reverse=True,
    )
    announcement_text_candidates = announcement_text_candidates[:announcement_text_limit]
    full_text_documents: list[dict[str, object]] = []
    full_text_attempts: list[dict[str, object]] = []
    for row in announcement_text_candidates:
        if announcement_document_dir is None:
            full_text_attempts.append({
                "source_record_id": row["source_record_id"],
                "status": "not_run_missing_document_directory",
            })
            break
        try:
            document = fetch_cninfo_full_text(
                str(row["公告链接"]),
                announcement_document_dir,
                str(row["source_record_id"]),
                str(row["公告标题"]),
                str(row["published_at"]),
            )
        except (URLError, CninfoPdfError) as error:
            full_text_attempts.append({
                "source_record_id": row["source_record_id"],
                "status": "fetch_or_extraction_failed",
                "error": f"{type(error).__name__}: {error}",
            })
            full_text_attempts.extend({
                "source_record_id": later["source_record_id"],
                "status": "not_attempted_after_first_failure",
            } for later in announcement_text_candidates[len(full_text_attempts):])
            break
        if datetime.fromisoformat(str(document["captured_at"])).date() != decision_time.date():
            raise ValueError("巨潮 PDF 抓取跨越了决策日期；请以抓取完成后的新决策日重跑")
        full_text_documents.append(document)
        full_text_attempts.append({
            "source_record_id": document["source_record_id"],
            "status": document["status"],
            "pdf_sha256": document["pdf_sha256"],
            "text_sha256": document["text_sha256"],
        })
        p9_outputs.append({
            "record_id": hashlib.sha256(
                f"cninfo-full-text-llm-v1:{row['source_record_id']}:{document['pdf_sha256']}".encode("utf-8")
            ).hexdigest(),
            "record_kind": "event",
            "expert": "cninfo_full_text_llm_v1",
            "symbol": row["symbol"],
            "event_type": "announcement_full_text",
            "prediction_date": decision_time.date().isoformat(),
            "event_time": row["published_at"],
            "available_at": document["captured_at"],
            "horizon_sessions": evaluation_horizon_sessions,
            "horizon_kind": "comparison_label_window",
            "direction_score": 0,
            "rule": "full_text_sent_for_evidence_grounded_llm_analysis",
            "evidence_text_scope": "full_announcement_pdf_text",
            "evidence": [{
                "source_record_id": row["source_record_id"],
                "dataset": "announcement_full_text_pdf",
                "title": row["公告标题"],
                "pdf_url": document["pdf_url"],
                "raw_pdf_ref": document["pdf_path"],
                "extracted_text_ref": document["text_path"],
                "pdf_sha256": document["pdf_sha256"],
                "text_sha256": document["text_sha256"],
                "page_count": document["page_count"],
                "extracted_character_count": document["extracted_character_count"],
                "text_extraction_status": document["text_extraction_status"],
                "published_at": row["published_at"],
                "available_at": document["captured_at"],
                "fetched_at": document["captured_at"],
                "full_text": document["llm_text_excerpt"],
                "llm_text_truncated": document["llm_text_truncated"],
            }],
            "vendor_vintage_verified": True,
        })

    company_by_symbol = {
        symbol: str(rows[-1].get("SECURITY_NAME_ABBR", "")).strip()
        for symbol, rows in indicators_by_symbol.items()
    }
    if allowed_symbols is not None:
        company_by_symbol = {
            symbol: name for symbol, name in company_by_symbol.items()
            if symbol.rsplit(".", 1)[-1] in allowed_symbols
        }
    if company_by_symbol:
        captured["news"] = load_public_records(
            cache_dir,
            "news",
            decision_time,
            company_names={name for name in company_by_symbol.values() if name},
        )
        new_news = [
            row for row in captured["news"]
            if _dt(_effective_available_at(row)) > event_start_time
        ]
    p10_outputs: list[dict[str, object]] = []
    for row in new_news:
        content = f"{row['新闻标题']} {row['新闻内容']}"
        for symbol, name in sorted(company_by_symbol.items()):
            if not name or name not in content:
                continue
            direction, rule = _news_direction(content)
            p10_outputs.append({
                "record_id": hashlib.sha256(
                    f"cls-rule-v1:{row['source_record_id']}:{symbol}".encode("utf-8")
                ).hexdigest(),
                "record_kind": "event",
                "expert": "cls_forward_rule_v1",
                "symbol": symbol,
                "company_name": name,
                "event_type": "market_news_mention",
                "prediction_date": decision_time.date().isoformat(),
                "event_time": row["published_at"],
                "available_at": _effective_available_at(row),
                "horizon_sessions": evaluation_horizon_sessions,
                "horizon_kind": "comparison_label_window",
                "direction_score": direction,
                "rule": rule,
                "evidence_text_scope": "captured_cls_news_content",
                "link_scope": row["链接范围"],
                "evidence": [{
                    "source_record_id": row["source_record_id"],
                    "title": row["新闻标题"],
                    "content": row["新闻内容"],
                    "url": row["新闻链接"],
                    "raw_response_ref": row["raw_response_ref"],
                    "content_sha256": row["content_sha256"],
                    "published_at": row["published_at"],
                    "available_at": _effective_available_at(row),
                    **_backfill_audit_fields(row),
                }],
            })

    macro_outputs = []
    for dataset in ("macro_cpi", "macro_pmi"):
        for row in captured[dataset]:
            series = dataset.removeprefix("macro_")
            value_field = "全国-同比增长" if series == "cpi" else "制造业-指数"
            macro_outputs.append({
                "record_id": hashlib.sha256(
                    f"nbs-context-v1:{series}:{row['source_record_id']}:{row['content_sha256']}".encode("utf-8")
                ).hexdigest(),
                "record_kind": "state",
                "expert": "nbs_macro_context_v1",
                "symbol": None,
                "series": series,
                "observation_period": str(row["月份"]),
                "observation_value": _number(row[value_field]),
                "event_type": "macro_context",
                "prediction_date": decision_time.date().isoformat(),
                "event_time": row["published_at"],
                "available_at": _effective_available_at(row),
                "horizon_sessions": None,
                "direction_score": 0,
                "rule": "context_only_single_captured_vintage",
                "vendor_vintage_verified": bool(row["vintage_verified"]),
                "evidence": [{
                    "source_record_id": row["source_record_id"],
                    "raw_response_ref": row["raw_response_ref"],
                    "content_sha256": row["content_sha256"],
                    "published_at": row["published_at"],
                    "available_at": _effective_available_at(row),
                    **_backfill_audit_fields(row),
                }],
            })

    attention_by_symbol: dict[str, list[dict[str, object]]] = {}
    for row in captured["attention_index"]:
        code = str(row["symbol"]).rsplit(".", 1)[-1]
        attention_by_symbol.setdefault(code, []).append(row)
    attention_outputs = []
    for symbol, rows in sorted(attention_by_symbol.items()):
        rows.sort(key=lambda row: (str(row["published_at"]), str(row["source_record_id"])))
        latest = rows[-1]
        previous = rows[-2] if len(rows) > 1 else None
        attention_outputs.append({
            "record_id": hashlib.sha256(
                f"attention-index-v1:{latest['source_record_id']}:{latest['content_sha256']}".encode("utf-8")
            ).hexdigest(),
            "record_kind": "state",
            "expert": "market_attention_index_v1",
            "symbol": symbol,
            "event_type": "market_attention_index",
            "prediction_date": decision_time.date().isoformat(),
            "event_time": latest["published_at"],
            "available_at": _effective_available_at(latest),
            "horizon_sessions": evaluation_horizon_sessions,
            "direction_score": 0,
            "attention_index": _number(latest.get("用户关注指数")),
            "previous_attention_index": (
                _number(previous.get("用户关注指数")) if previous else None
            ),
            "attention_index_change": (
                _number(latest.get("用户关注指数")) - _number(previous.get("用户关注指数"))
                if previous and _number(latest.get("用户关注指数")) is not None
                and _number(previous.get("用户关注指数")) is not None else None
            ),
            "interpretation_scope": "source_reported_market_attention_only_not_discussion_text_sentiment",
            "evidence": [{
                "source_record_id": latest["source_record_id"],
                "dataset": "attention_index",
                "trade_date": latest["交易日"],
                "attention_index": _number(latest.get("用户关注指数")),
                "published_at": latest["published_at"],
                "available_at": _effective_available_at(latest),
                **_backfill_audit_fields(latest),
                "raw_response_ref": latest["raw_response_ref"],
                "content_sha256": latest["content_sha256"],
            }],
        })

    rule_finished = time.perf_counter_ns()
    fresh_for_llm = [
        row for row in p9_outputs
        if _dt(row["available_at"]) > event_start_time
        and str(row["record_id"]) not in (previously_observed_record_ids or set())
    ]
    final_decision_time = max(
        [decision_time] + [datetime.fromisoformat(str(document["captured_at"])).astimezone(SHANGHAI) for document in full_text_documents]
    )
    if final_decision_time.date() != decision_time.date():
        raise ValueError("巨潮全文抓取完成时间跨越决策日期")
    llm_result = (
        call_qwen_expert(
            fresh_for_llm, final_decision_time, evaluation_horizon_sessions,
            max_records=llm_max_records,
        )
        if include_qwen
        else {
            "status": "disabled_for_role_isolated_review",
            "outputs": [],
            "usage": None,
            "estimated_token_cost_cny": 0.0,
            "elapsed_ms": 0.0,
            "completed_at": None,
        }
    )
    if llm_result.get("completed_at"):
        final_decision_time = max(
            final_decision_time,
            datetime.fromisoformat(str(llm_result["completed_at"])).astimezone(SHANGHAI),
        )
    if final_decision_time.date() != decision_time.date():
        raise ValueError("Qwen 分析完成时间跨越决策日期")
    elapsed_ms = (time.perf_counter_ns() - started) / 1_000_000
    return {
        "schema_version": "public-expert-snapshot.v1",
        "data_mode": (
            "retrospective_source_publication_proxy"
            if retrospective_backfill_proxy else "captured_public_cache_cutoff"
        ),
        "strict_historical_pit_verified": False,
        "retrospective_capture_lag_record_count": retrospective_capture_lag_count,
        "availability_policy": (
            "source publication and date-only next-day availability; later fetched_at is retained and marks vintage unverified"
            if retrospective_backfill_proxy else
            "source availability and fetched_at must both be no later than decision_time"
        ),
        "requested_decision_time": decision_time.isoformat(),
        "decision_time": final_decision_time.isoformat(),
        "evaluation_horizon_sessions": evaluation_horizon_sessions,
        "event_start_time": event_start_time.isoformat(),
        "allowed_symbols": sorted(allowed_symbols) if allowed_symbols is not None else None,
        "symbol_scope": (
            "decision_date_visible_index_members_plus_existing_account_positions"
            if allowed_symbols is not None else "unrestricted"
        ),
        "source_cache": str(Path(cache_dir)),
        "p9_outputs": p9_outputs,
        "p10_outputs": p10_outputs,
        "macro_outputs": macro_outputs,
        "attention_outputs": attention_outputs,
        "llm_outputs": llm_result,
        "announcement_full_text": {
            "status": (
                "extracted" if full_text_documents else
                "failed" if any(row["status"] == "fetch_or_extraction_failed" for row in full_text_attempts) else
                "not_run_no_eligible_recent_records"
            ),
            "maximum_document_count": announcement_text_limit,
            "attempted_count": len(full_text_attempts),
            "extracted_count": len(full_text_documents),
            "attempts": full_text_attempts,
            "documents": full_text_documents,
        },
        "cost_and_latency": {
            "rule_engine_cost_cny": 0.0,
            "rule_engine_elapsed_ms": (rule_finished - started) / 1_000_000,
            "llm_estimated_token_cost_cny": llm_result["estimated_token_cost_cny"],
            "llm_elapsed_ms": llm_result["elapsed_ms"],
            "llm_status": llm_result["status"],
            "snapshot_elapsed_ms": elapsed_ms,
        },
        "source_coverage": {
            "captured_records": {dataset: len(rows) for dataset, rows in captured.items()},
            "new_announcements": len(new_announcements),
            "new_cls_records": len(new_news),
            "matched_cls_observations": len(p10_outputs),
            "captured_cpi_values": len(captured["macro_cpi"]),
            "captured_pmi_values": len(captured["macro_pmi"]),
            "captured_attention_records": len(captured["attention_index"]),
        },
            "limitations": [
            "Financial facts and vendor metrics are current at capture time but historical revisions are unverified.",
            "Backfilled disclosures are event candidates only inside the recent publication window; capture time is never backdated.",
            "CNINFO evidence is headline-only; no announcement body is claimed or fetched.",
            "CLS is a captured latest-20 feed and is forward-only; its stored link is a section page.",
            "NBS CPI/PMI rows require formal PubDate and a completed local capture; vintage revisions remain unverified.",
            "Market attention is the AKShare user-focus index only; this snapshot has no discussion-post text and makes no text-sentiment claim.",
            "Rule and LLM outputs are directional research signals, not expected-return forecasts or live trading inputs.",
        ],
    }
