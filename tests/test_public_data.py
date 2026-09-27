from datetime import date, datetime, time, timezone

import pandas as pd
import pytest

from adaptive_mas.data.public_data import archive_frame, load_public_records


def _at(value: str) -> datetime:
    return datetime.fromisoformat(value)


def test_sina_compact_announcement_date_is_not_visible_same_day(tmp_path):
    archive_frame(
        tmp_path,
        "financial_statement_income_statement",
        "AKShare/Sina",
        "stock_financial_report_sina",
        {"stock": "sh600000", "symbol": "利润表"},
        pd.DataFrame(
            [{"报告日": "2026-06-30", "公告日期": "20260828", "营业收入": 123.4}]
        ),
        _at("2026-09-27T12:00:00+08:00"),
        [],
        "sh.600000",
    )

    same_day = load_public_records(
        tmp_path, "financial_statement_income_statement", _at("2026-08-28T15:30:00+08:00")
    )
    next_session = load_public_records(
        tmp_path, "financial_statement_income_statement", _at("2026-08-31T09:30:00+08:00")
    )

    assert same_day == []
    assert next_session[0]["报告日"] == "2026-06-30"
    assert next_session[0]["published_at"] == "2026-08-28T00:00:00+08:00"
    assert next_session[0]["available_at"] == "2026-08-29T00:00:00+08:00"
    assert next_session[0]["vintage_verified"] is False


def test_future_cninfo_disclosure_is_filtered_and_leading_zero_is_kept(tmp_path):
    archive_frame(
        tmp_path,
        "announcement",
        "AKShare/CNInfo",
        "stock_zh_a_disclosure_report_cninfo",
        {"symbol": "000001", "market": "沪深京"},
        pd.DataFrame(
            [
                {
                    "代码": "000001",
                    "公告标题": "季报",
                    "公告时间": "2026-09-28",
                    "公告链接": "https://www.cninfo.com.cn/detail/1",
                }
            ]
        ),
        _at("2026-09-27T12:00:00+08:00"),
        [],
        "sz.000001",
    )

    rows = load_public_records(
        tmp_path,
        "announcement",
        _at("2026-09-27T20:00:00+08:00"),
        symbols={"sz.000001"},
    )
    assert rows == []


def test_news_uses_capture_time_and_company_name_filter(tmp_path):
    frame = pd.DataFrame(
        [
            {
                "新闻标题": "浦发银行发布公告",
                "新闻内容": "浦发银行相关快讯内容",
                "发布日期": date(2026, 9, 27),
                "发布时间": time(11, 45),
                "新闻来源": "财联社电报",
                "新闻链接": "https://www.cls.cn/telegraph",
                "链接范围": "栏目页",
            }
        ]
    )
    archive_frame(
        tmp_path,
        "news",
        "AKShare/CLS",
        "stock_info_global_cls",
        {},
        frame,
        _at("2026-09-27T13:00:00+08:00"),
        [],
        collection_date="2026-09-27",
    )
    archive_frame(
        tmp_path,
        "news",
        "AKShare/CLS",
        "stock_info_global_cls",
        {},
        frame,
        _at("2026-09-28T10:00:00+08:00"),
        [],
        collection_date="2026-09-28",
    )

    before_capture = load_public_records(
        tmp_path,
        "news",
        _at("2026-09-27T12:59:00+08:00"),
        company_names={"浦发银行"},
    )
    after_capture = load_public_records(
        tmp_path,
        "news",
        _at("2026-09-27T13:00:00+08:00"),
        company_names={"浦发银行"},
    )

    assert before_capture == []
    assert len(after_capture) == 1
    assert after_capture[0]["published_at"] == "2026-09-27T11:45:00+08:00"
    assert after_capture[0]["新闻链接"] == "https://www.cls.cn/telegraph"
    assert after_capture[0]["链接范围"] == "栏目页"
    assert after_capture[0]["vintage_verified"] is True
    after_repeat_capture = load_public_records(
        tmp_path,
        "news",
        _at("2026-09-29T16:00:00+08:00"),
        company_names={"浦发银行"},
    )
    assert len(after_repeat_capture) == 1
    assert after_repeat_capture[0]["fetched_at"] == _at(
        "2026-09-27T13:00:00+08:00"
    ).astimezone(timezone.utc).isoformat()
    assert load_public_records(
        tmp_path,
        "news",
        _at("2026-09-27T13:00:00+08:00"),
        company_names={"不存在的公司"},
    ) == []


def test_news_rejects_symbol_filter_because_feed_has_no_stock_code(tmp_path):
    with pytest.raises(ValueError, match="company_names"):
        load_public_records(
            tmp_path,
            "news",
            _at("2026-09-27T13:00:00+08:00"),
            symbols={"sh.600000"},
        )


def test_news_repeat_capture_keeps_first_available_at(tmp_path):
    frame = pd.DataFrame(
        [
            {
                "新闻标题": "浦发银行发布公告",
                "新闻内容": "浦发银行相关快讯内容",
                "发布日期": date(2026, 9, 27),
                "发布时间": time(11, 45),
                "新闻来源": "财联社电报",
                "新闻链接": "https://www.cls.cn/telegraph",
                "链接范围": "栏目页",
            }
        ]
    )
    for capture_time, collection_date in [
        ("2026-09-27T13:00:00+08:00", "2026-09-27"),
        ("2026-09-28T10:00:00+08:00", "2026-09-28"),
    ]:
        archive_frame(
            tmp_path,
            "news",
            "AKShare/CLS",
            "stock_info_global_cls",
            {},
            frame,
            _at(capture_time),
            [],
            collection_date=collection_date,
        )

    rows = load_public_records(
        tmp_path,
        "news",
        _at("2026-09-29T16:00:00+08:00"),
        company_names={"浦发银行"},
    )

    assert len(rows) == 1
    assert rows[0]["available_at"] == "2026-09-27T13:00:00+08:00"
    assert rows[0]["fetched_at"] == "2026-09-27T05:00:00+00:00"


def test_macro_value_waits_for_nbs_release_and_its_cache_capture(tmp_path):
    archive_frame(
        tmp_path,
        "macro_release",
        "National Bureau of Statistics of China",
        "NBS data release pages",
        {},
        pd.DataFrame(
            [
                {
                    "series_id": "cpi",
                    "observation_month": "2026-08",
                    "released_at": "2026-09-09T09:30:00+08:00",
                    "announcement_title": "2026年8月份居民消费价格",
                    "announcement_url": "https://www.stats.gov.cn/example",
                }
            ]
        ),
        _at("2026-09-09T09:30:00+08:00"),
        [],
    )
    archive_frame(
        tmp_path,
        "macro_cpi",
        "AKShare/Eastmoney",
        "macro_china_cpi",
        {},
        pd.DataFrame([{"月份": "2026年08月份", "全国-同比增长": 0.8}]),
        _at("2026-09-09T09:31:00+08:00"),
        ["AKShare_series_has_no_official_release_timestamp"],
    )
    archive_frame(
        tmp_path,
        "macro_cpi",
        "AKShare/Eastmoney",
        "macro_china_cpi",
        {},
        pd.DataFrame([{"月份": "2026年08月份", "全国-同比增长": 1.2}]),
        _at("2026-09-10T10:00:00+08:00"),
        ["AKShare_series_has_no_official_release_timestamp"],
    )

    before_capture = load_public_records(
        tmp_path, "macro_cpi", _at("2026-09-09T09:30:30+08:00")
    )
    after_capture = load_public_records(
        tmp_path, "macro_cpi", _at("2026-09-09T09:31:00+08:00")
    )

    assert before_capture == []
    assert len(after_capture) == 1
    assert after_capture[0]["published_at"] == "2026-09-09T09:30:00+08:00"
    assert after_capture[0]["available_at"] == "2026-09-09T09:31:00+08:00"
    assert after_capture[0]["vintage_verified"] is False
    assert after_capture[0]["全国-同比增长"] == "0.8"

    after_second_capture = load_public_records(
        tmp_path, "macro_cpi", _at("2026-09-11T10:00:00+08:00")
    )
    assert len(after_second_capture) == 1
    assert after_second_capture[0]["全国-同比增长"] == "1.2"
    assert after_second_capture[0]["fetched_at"] == _at(
        "2026-09-10T10:00:00+08:00"
    ).astimezone(timezone.utc).isoformat()


def test_cpi_uses_latest_capture_available_at_decision_time(tmp_path):
    archive_frame(
        tmp_path,
        "macro_release",
        "National Bureau of Statistics of China",
        "NBS data release pages",
        {},
        pd.DataFrame(
            [
                {
                    "series_id": "cpi",
                    "observation_month": "2026-08",
                    "released_at": "2026-09-09T09:30:00+08:00",
                    "announcement_title": "2026年8月份居民消费价格",
                    "announcement_url": "https://www.stats.gov.cn/example",
                }
            ]
        ),
        _at("2026-09-09T09:30:00+08:00"),
        [],
    )
    for value, capture_time in [
        (0.8, "2026-09-09T09:31:00+08:00"),
        (1.2, "2026-09-09T10:00:00+08:00"),
    ]:
        archive_frame(
            tmp_path,
            "macro_cpi",
            "AKShare/Eastmoney",
            "macro_china_cpi",
            {},
            pd.DataFrame([{"月份": "2026年08月份", "全国-同比增长": value}]),
            _at(capture_time),
            ["AKShare_series_has_no_official_release_timestamp"],
        )

    rows = load_public_records(
        tmp_path, "macro_cpi", _at("2026-09-09T09:45:00+08:00")
    )

    assert len(rows) == 1
    assert rows[0]["全国-同比增长"] == "0.8"
    assert rows[0]["fetched_at"] == "2026-09-09T01:31:00+00:00"
    assert rows[0]["available_at"] == "2026-09-09T09:31:00+08:00"


def test_same_content_cache_is_reproducibly_loaded_with_hash(tmp_path):
    entry = archive_frame(
        tmp_path,
        "announcement",
        "AKShare/CNInfo",
        "stock_zh_a_disclosure_report_cninfo",
        {"symbol": "600000"},
        pd.DataFrame(
            [
                {
                    "代码": "600000",
                    "公告标题": "公告标题",
                    "公告时间": "2026-09-01",
                    "公告链接": "https://www.cninfo.com.cn/detail/2",
                }
            ]
        ),
        _at("2026-09-27T12:00:00+08:00"),
        [],
        "sh.600000",
    )
    first = load_public_records(
        tmp_path, "announcement", _at("2026-09-02T00:00:00+08:00")
    )
    second = load_public_records(
        tmp_path, "announcement", _at("2026-09-02T00:00:00+08:00")
    )

    assert first == second
    assert first[0]["content_sha256"] == entry["sha256"]
    assert first[0]["raw_response_ref"] == entry["response_file"]
    assert first[0]["vintage_verified"] is False
    assert first[0]["vintage_status"] == "historical_backfill_after_decision_unverified"


def test_historical_financial_dedup_prefers_predecision_then_earliest_backfill(tmp_path):
    parameters = {"stock": "sh600000", "symbol": "利润表"}
    first_capture = _at("2026-08-30T12:00:00+08:00")
    archive_frame(
        tmp_path,
        "financial_statement_income_statement",
        "AKShare/Sina",
        "stock_financial_report_sina",
        parameters,
        pd.DataFrame(
            [{"报告日": "2026-06-30", "公告日期": "20260828", "营业收入": 100}]
        ),
        first_capture,
        [],
        "sh.600000",
    )
    archive_frame(
        tmp_path,
        "financial_statement_income_statement",
        "AKShare/Sina",
        "stock_financial_report_sina",
        parameters,
        pd.DataFrame(
            [{"报告日": "2026-06-30", "公告日期": "20260828", "营业收入": 200}]
        ),
        _at("2026-09-10T12:00:00+08:00"),
        [],
        "sh.600000",
    )

    before_any_capture = load_public_records(
        tmp_path,
        "financial_statement_income_statement",
        _at("2026-08-29T12:00:00+08:00"),
        symbols={"sh.600000"},
    )
    with_predecision_capture = load_public_records(
        tmp_path,
        "financial_statement_income_statement",
        _at("2026-09-01T12:00:00+08:00"),
        symbols={"sh.600000"},
    )
    after_both_captures = load_public_records(
        tmp_path,
        "financial_statement_income_statement",
        _at("2026-09-11T12:00:00+08:00"),
        symbols={"sh.600000"},
    )

    assert len(before_any_capture) == 1
    assert before_any_capture[0]["营业收入"] == "100"
    assert before_any_capture[0]["vintage_verified"] is False
    assert len(with_predecision_capture) == 1
    assert with_predecision_capture[0]["营业收入"] == "100"
    assert with_predecision_capture[0]["fetched_at"] == first_capture.astimezone(
        timezone.utc
    ).isoformat()
    assert len(after_both_captures) == 1
    assert after_both_captures[0]["营业收入"] == "200"
