from datetime import date

import pandas as pd

from adaptive_mas.data import akshare_source
from adaptive_mas.data.akshare_source import AkShareSource


class FakeResponse:
    def __init__(self, body):
        self.content = body.encode("utf-8")

    def raise_for_status(self):
        return None


def test_sina_report_uses_baostock_exchange_code(monkeypatch):
    requested = {}
    monkeypatch.setattr(
        akshare_source.ak,
        "stock_financial_report_sina",
        lambda **kwargs: requested.update(kwargs) or pd.DataFrame(),
    )

    AkShareSource().financial_statement("sh.600000", "利润表")

    assert requested == {"stock": "sh600000", "symbol": "利润表"}


def test_financial_indicator_uses_exchange_suffix(monkeypatch):
    requested = {}
    monkeypatch.setattr(
        akshare_source.ak,
        "stock_financial_analysis_indicator_em",
        lambda **kwargs: requested.update(kwargs) or pd.DataFrame(),
    )

    AkShareSource().financial_indicator("sz.000001")

    assert requested == {"symbol": "000001.SZ", "indicator": "按报告期"}


def test_disclosure_dates_are_explicit_and_invalid_range_fails(monkeypatch):
    requested = {}
    monkeypatch.setattr(
        akshare_source.ak,
        "stock_zh_a_disclosure_report_cninfo",
        lambda **kwargs: requested.update(kwargs) or pd.DataFrame(),
    )
    source = AkShareSource()
    source.disclosures("sh.600000", date(2024, 1, 1), date(2024, 1, 31))

    assert requested == {
        "symbol": "600000",
        "market": "沪深京",
        "start_date": "20240101",
        "end_date": "20240131",
    }


def test_cls_news_saves_source_page_without_claiming_article_link(monkeypatch):
    monkeypatch.setattr(
        akshare_source.ak,
        "stock_info_global_cls",
        lambda: pd.DataFrame(
            [
                {
                    "标题": "浦发银行相关快讯",
                    "内容": "快讯摘要",
                    "发布日期": date(2026, 9, 27),
                    "发布时间": "11:45:00",
                }
            ]
        ),
    )

    row = AkShareSource().recent_market_news().iloc[0]

    assert row["新闻标题"] == "浦发银行相关快讯"
    assert row["新闻来源"] == "财联社电报"
    assert row["新闻链接"] == "https://www.cls.cn/telegraph"
    assert row["链接范围"] == "栏目页"


def test_nbs_release_parser_reads_pubdate_meta_from_official_pages(monkeypatch):
    index = """
    <a href="/sj/zxfb/202609/t20260909_1.html">2026年8月份居民消费价格同比上涨</a>
    <a href="/sj/zxfb/202608/t20260831_2.html">2026年8月中国采购经理指数运行情况</a>
    """
    pages = {
        akshare_source.NBS_RELEASE_INDEX: index,
        "https://www.stats.gov.cn/sj/zxfb/202609/t20260909_1.html":
            '<meta name="PubDate" content="2026/09/09 09:30">',
        "https://www.stats.gov.cn/sj/zxfb/202608/t20260831_2.html":
            '<meta name="PubDate" content="2026/08/31 09:30">',
    }
    monkeypatch.setattr(
        akshare_source.requests,
        "get",
        lambda url, **kwargs: FakeResponse(pages[url]),
    )

    records = AkShareSource().nbs_macro_releases("2026-08", max_pages=1).to_dict("records")

    assert [(row["series_id"], row["observation_month"]) for row in records] == [
        ("cpi", "2026-08"),
        ("pmi", "2026-08"),
    ]
    assert records[0]["released_at"] == "2026-09-09T09:30:00+08:00"
    assert records[1]["released_at"] == "2026-08-31T09:30:00+08:00"


def test_nbs_history_stops_at_explicit_start_month(monkeypatch):
    first_index = """
    <a href="202512/cpi.html">2025年12月份居民消费价格同比上涨</a>
    <a href="202512/pmi.html">2025年12月中国采购经理指数运行情况</a>
    """
    second_index = """
    <a href="202401/cpi.html">2024年1月份居民消费价格同比上涨</a>
    <a href="202312/cpi.html">2023年12月份居民消费价格同比上涨</a>
    """
    index_one = "https://www.stats.gov.cn/sj/zxfb/index_1.html"
    pages = {
        akshare_source.NBS_RELEASE_INDEX: first_index,
        index_one: second_index,
        "https://www.stats.gov.cn/sj/zxfb/202512/cpi.html":
            '<meta name="PubDate" content="2026/01/09 09:30">',
        "https://www.stats.gov.cn/sj/zxfb/202512/pmi.html":
            '<meta name="PubDate" content="2025/12/31 09:30">',
        "https://www.stats.gov.cn/sj/zxfb/202401/cpi.html":
            '<meta name="PubDate" content="2024/02/08 09:30">',
    }
    requested = []

    def get(url, **kwargs):
        requested.append(url)
        return FakeResponse(pages[url])

    monkeypatch.setattr(akshare_source.requests, "get", get)
    frame = AkShareSource().nbs_macro_releases("2024-01", max_pages=2)

    assert requested[:2] == [akshare_source.NBS_RELEASE_INDEX, index_one]
    assert len(requested) == 5
    assert set(frame["observation_month"]) == {"2025-12", "2024-01"}
    listing_by_period = dict(zip(frame["observation_month"], frame["listing_page"]))
    assert listing_by_period["2025-12"] == akshare_source.NBS_RELEASE_INDEX
    assert listing_by_period["2024-01"] == index_one

def test_user_focus_index_uses_exchange_free_code(monkeypatch):
    requested = {}
    monkeypatch.setattr(
        akshare_source.ak,
        "stock_comment_detail_scrd_focus_em",
        lambda **kwargs: requested.update(kwargs) or pd.DataFrame(
            [{"交易日": "2026-09-27", "用户关注指数": 12.5}]
        ),
    )

    result = AkShareSource().user_focus_index("sh.600000")

    assert requested == {"symbol": "600000"}
    assert result.iloc[0]["用户关注指数"] == 12.5